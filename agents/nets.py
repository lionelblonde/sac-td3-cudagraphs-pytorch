import math
from collections import OrderedDict
from typing import Callable, Optional, Union
from contextlib import nullcontext

from beartype import beartype
from einops import pack
import torch
from torch import nn
from torch.nn import functional as ff

from helpers import logger
from helpers.normalizer import RunningMoments


STANDARDIZED_OB_CLAMPS = [-5., 5.]
ARCTANH_EPS = 1e-8
SAC_LOG_STD_BOUNDS = [-5., 2.]


@beartype
def log_module_info(model: nn.Module):

    def _fmt(n) -> str:
        if n // 10 ** 6 > 0:
            out = str(round(n / 10 ** 6, 2)) + " M"
        elif n // 10 ** 3:
            out = str(round(n / 10 ** 3, 2)) + " k"
        else:
            out = str(n)
        return out

    logger.info("logging model specs")
    logger.info(model)
    num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    logger.info(f"total trainable params: {_fmt(num_params)}.")


@beartype
def init(constant_bias: float = 0.) -> Callable[[nn.Module], None]:
    """Perform orthogonal initialization"""

    def _init(m: nn.Module) -> None:

        if (isinstance(m, (nn.Conv2d, nn.Linear, nn.Bilinear))):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, constant_bias)
        elif (isinstance(m, (nn.BatchNorm2d, nn.LayerNorm))):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    return _init


@beartype
def arctanh(x: torch.Tensor) -> torch.Tensor:
    """Implementation of the arctanh function.
    Can be very numerically unstable, hence the clamping.
    """
    out = torch.atanh(x)
    if out.sum().isfinite():
        # note: a sum() is often faster than a any() or all()
        # there might be edge cases but at worst we use the clamped version and get notified
        return out
    logger.info("using a numerically stable (and clamped) arctanh")
    one_plus_x = (1 + x).clamp(
        min=ARCTANH_EPS)
    one_minus_x = (1 - x).clamp(
        min=ARCTANH_EPS)
    return 0.5 * torch.log(one_plus_x / one_minus_x)
    # equivalent to 0.5 * (x.log1p() - (-x).log1p()) but with NaN-proof clamping
    # torch.atanh(x) is numerically unstable here
    # note: with both of the methods above, we get NaN at the first iteration


class NormalToolkit(object):
    """Technically, multivariate normal with diagonal covariance"""

    @staticmethod
    @beartype
    def logp(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        neglogp = (0.5 * ((x - mean) / std).pow(2).sum(dim=-1, keepdim=True) +
                   0.5 * math.log(2 * math.pi) +
                   std.log().sum(dim=-1, keepdim=True))
        return -neglogp

    @staticmethod
    @beartype
    def sample(mean: torch.Tensor, std: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        # re-parametrization trick
        eps = torch.empty(mean.size(), device=mean.device).normal_(generator=generator)
        eps.requires_grad = False
        return mean + (std * eps)

    @staticmethod
    @beartype
    def mode(mean: torch.Tensor) -> torch.Tensor:
        return mean


class TanhNormalToolkit(object):
    """Technically, multivariate normal with diagonal covariance"""

    @staticmethod
    @beartype
    def logp(x: torch.Tensor,
             mean: torch.Tensor,
             std: torch.Tensor,
             *,
             scale: torch.Tensor) -> torch.Tensor:
        # we need to assemble the logp of a sample which comes from a Gaussian sample
        # after being mapped through a tanh. This needs a change of variable.
        # See appendix C of the SAC paper for an explanation of this change of variable.
        x_ = arctanh(x / scale)
        logp1 = NormalToolkit.logp(x_, mean, std)
        logp2 = 2. * (math.log(2.) - x_ - ff.softplus(-2. * x_))
        logp2 = logp2.sum(dim=-1, keepdim=True)
        return logp1 - logp2

    @staticmethod
    @beartype
    def sample(mean: torch.Tensor,
               std: torch.Tensor,
               *,
               generator: torch.Generator,
               scale: torch.Tensor,
               bias: torch.Tensor) -> torch.Tensor:
        sample = NormalToolkit.sample(mean, std, generator)
        sample = torch.tanh(sample)
        return sample * scale + bias

    @staticmethod
    @beartype
    def mode(mean: torch.Tensor,
             *,
             scale: torch.Tensor,
             bias: torch.Tensor,
    ) -> torch.Tensor:
        return torch.tanh(mean) * scale + bias


class Critic(nn.Module):

    @beartype
    def __init__(self,
                 ob_shape: tuple[int, ...],
                 ac_shape: tuple[int, ...],
                 hid_dims: tuple[int, int],
                 rms_obs: Optional[RunningMoments],
                 *,
                 layer_norm: bool,
                 device: Optional[Union[str, torch.device]] = None):
        super().__init__()
        ob_dim = ob_shape[-1]
        ac_dim = ac_shape[-1]
        self.rms_obs = rms_obs
        self.layer_norm = layer_norm

        # assemble the last layers and output heads
        self.fc_stack = nn.Sequential(OrderedDict([
            ("fc_block_1", nn.Sequential(OrderedDict([
                ("fc", nn.Linear(ob_dim + ac_dim, hid_dims[0], device=device)),
                ("ln", (nn.LayerNorm if self.layer_norm else nn.Identity)(hid_dims[0],
                                                                          device=device)),
                ("nl", nn.ReLU()),
            ]))),
            ("fc_block_2", nn.Sequential(OrderedDict([
                ("fc", nn.Linear(hid_dims[0], hid_dims[1], device=device)),
                ("ln", (nn.LayerNorm if self.layer_norm else nn.Identity)(hid_dims[1],
                                                                          device=device)),
                ("nl", nn.ReLU()),
            ]))),
        ]))
        self.head = nn.Linear(hid_dims[1], 1, device=device)

        # perform initialization
        self.fc_stack.apply(init())
        self.head.apply(init())

    @beartype
    def forward(self, ob: torch.Tensor, ac: torch.Tensor) -> torch.Tensor:
        if self.rms_obs is not None:
            ob = self.rms_obs.standardize(ob).clamp(*STANDARDIZED_OB_CLAMPS)
        x, _ = pack([ob, ac], "b *")
        x = self.fc_stack(x)
        return self.head(x)


class Actor(nn.Module):

    @beartype
    def __init__(self,
                 ob_shape: tuple[int, ...],
                 ac_shape: tuple[int, ...],
                 hid_dims: tuple[int, int],
                 rms_obs: Optional[RunningMoments],
                 min_ac: torch.Tensor,
                 max_ac: torch.Tensor,
                 *,
                 exploration_noise: float,
                 layer_norm: bool,
                 device: Optional[Union[str, torch.device]] = None):
        super().__init__()
        ob_dim = ob_shape[-1]
        ac_dim = ac_shape[-1]
        self.rms_obs = rms_obs
        self.layer_norm = layer_norm

        # assemble the last layers and output heads
        self.fc_stack = nn.Sequential(OrderedDict([
            ("fc_block_1", nn.Sequential(OrderedDict([
                ("fc", nn.Linear(ob_dim, hid_dims[0], device=device)),
                ("ln", (nn.LayerNorm if self.layer_norm else nn.Identity)(hid_dims[0],
                                                                          device=device)),
                ("nl", nn.ReLU()),
            ]))),
            ("fc_block_2", nn.Sequential(OrderedDict([
                ("fc", nn.Linear(hid_dims[0], hid_dims[1], device=device)),
                ("ln", (nn.LayerNorm if self.layer_norm else nn.Identity)(hid_dims[1],
                                                                          device=device)),
                ("nl", nn.ReLU()),
            ]))),
        ]))
        self.head = nn.Linear(hid_dims[1], ac_dim, device=device)

        # perform initialization
        self.fc_stack.apply(init())
        self.head.apply(init())

        # register buffers: action rescaling
        self.register_buffer("action_scale",
            (max_ac - min_ac) / 2.0)
        self.register_buffer("action_bias",
            (max_ac + min_ac) / 2.0)
        # register buffers: exploration
        self.register_buffer("exploration_noise",
            torch.as_tensor(exploration_noise, device=device))

    @beartype
    def forward(self, ob: torch.Tensor) -> torch.Tensor:
        if self.rms_obs is not None:
            ob = self.rms_obs.standardize(ob).clamp(*STANDARDIZED_OB_CLAMPS)
        x = self.fc_stack(ob)
        x = self.head(x)
        return torch.tanh(x) * self.action_scale + self.action_bias

    @beartype
    def explore(self, ob: torch.Tensor) -> torch.Tensor:
        ac = self(ob)
        return ac + torch.randn_like(ac).mul(self.action_scale * self.exploration_noise)


class TanhGaussActor(nn.Module):

    @beartype
    def __init__(self,
                 ob_shape: tuple[int, ...],
                 ac_shape: tuple[int, ...],
                 hid_dims: tuple[int, int],
                 rms_obs: Optional[RunningMoments],
                 min_ac: torch.Tensor,
                 max_ac: torch.Tensor,
                 *,
                 generator: torch.Generator,
                 layer_norm: bool,
                 device: Optional[Union[str, torch.device]] = None):
        super().__init__()
        ob_dim = ob_shape[-1]
        ac_dim = ac_shape[-1]
        self.rms_obs = rms_obs
        self.rng = generator
        self.layer_norm = layer_norm

        # assemble the last layers and output heads
        self.fc_stack = nn.Sequential(OrderedDict([
            ("fc_block_1", nn.Sequential(OrderedDict([
                ("fc", nn.Linear(ob_dim, hid_dims[0], device=device)),
                ("ln", (nn.LayerNorm if self.layer_norm else nn.Identity)(hid_dims[0],
                                                                          device=device)),
                ("nl", nn.ReLU()),
            ]))),
            ("fc_block_2", nn.Sequential(OrderedDict([
                ("fc", nn.Linear(hid_dims[0], hid_dims[1], device=device)),
                ("ln", (nn.LayerNorm if self.layer_norm else nn.Identity)(hid_dims[1],
                                                                          device=device)),
                ("nl", nn.ReLU()),
            ]))),
        ]))
        self.head = nn.Linear(hid_dims[1], 2 * ac_dim, device=device)

        # perform initialization
        self.fc_stack.apply(init())
        self.head.apply(init())

        # register buffers: action rescaling
        self.register_buffer("action_scale",
            (max_ac - min_ac) / 2.0)
        self.register_buffer("action_bias",
            (max_ac + min_ac) / 2.0)

    @beartype
    def logp(self, ob: torch.Tensor, ac: torch.Tensor) -> torch.Tensor:
        out = self(ob)
        return TanhNormalToolkit.logp(ac, *out, scale=self.action_scale)  # mean, std

    @beartype
    def sample(self, ob: torch.Tensor, *, stop_grad: bool = True) -> torch.Tensor:
        with torch.no_grad() if stop_grad else nullcontext():
            out = self(ob)
            return TanhNormalToolkit.sample(*out,
                                            generator=self.rng,
                                            scale=self.action_scale,
                                            bias=self.action_bias)

    @beartype
    def mode(self, ob: torch.Tensor, *, stop_grad: bool = True) -> torch.Tensor:
        with torch.no_grad() if stop_grad else nullcontext():
            mean, _ = self(ob)
            return TanhNormalToolkit.mode(mean,
                                          scale=self.action_scale,
                                          bias=self.action_bias)

    @staticmethod
    @beartype
    def bound_log_std(log_std: torch.Tensor) -> torch.Tensor:
        """Stability trick from OpenAI SpinUp / Denis Yarats"""
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = SAC_LOG_STD_BOUNDS
        return log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

    @beartype
    def forward(self, ob: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.rms_obs is not None:
            ob = self.rms_obs.standardize(ob).clamp(*STANDARDIZED_OB_CLAMPS)
        x = self.fc_stack(ob)
        ac_mean, ac_log_std = self.head(x).chunk(2, dim=-1)
        ac_log_std = self.bound_log_std(ac_log_std)
        ac_std = ac_log_std.exp()
        return ac_mean, ac_std
