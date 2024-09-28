import math
from collections import OrderedDict
from typing import Callable, Optional
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
    logger.warn("using a numerically stable (and clamped) arctanh")
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

    @beartype
    @staticmethod
    def logp(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        neglogp = (0.5 * ((x - mean) / std).pow(2).sum(dim=-1, keepdim=True) +
                   0.5 * math.log(2 * math.pi) +
                   std.log().sum(dim=-1, keepdim=True))
        return -neglogp

    @beartype
    @staticmethod
    def sample(mean: torch.Tensor, std: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        # re-parametrization trick
        eps = torch.empty(mean.size()).to(mean.device).normal_(generator=generator)
        eps.requires_grad = False
        return mean + (std * eps)

    @beartype
    @staticmethod
    def mode(mean: torch.Tensor) -> torch.Tensor:
        return mean


class TanhNormalToolkit(object):
    """Technically, multivariate normal with diagonal covariance"""

    @beartype
    @staticmethod
    def logp(x: torch.Tensor,
             mean: torch.Tensor,
             std: torch.Tensor,
             *,
             x_scale: float) -> torch.Tensor:
        # we need to assemble the logp of a sample which comes from a Gaussian sample
        # after being mapped through a tanh. This needs a change of variable.
        # See appendix C of the SAC paper for an explanation of this change of variable.
        x_ = arctanh(x / x_scale)
        logp1 = NormalToolkit.logp(x_, mean, std)
        logp2 = 2. * (math.log(2.) - x_ - ff.softplus(-2. * x_))
        logp2 = logp2.sum(dim=-1, keepdim=True)
        # trick for numerical stability from:
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return logp1 - logp2

    @beartype
    @staticmethod
    def sample(mean: torch.Tensor, std: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        sample = NormalToolkit.sample(mean, std, generator)
        return torch.tanh(sample)

    @beartype
    @staticmethod
    def mode(mean: torch.Tensor) -> torch.Tensor:
        return torch.tanh(mean)


class Critic(nn.Module):

    @beartype
    def __init__(self,
                 ob_shape: tuple[int, ...],
                 ac_shape: tuple[int, ...],
                 hid_dims: tuple[int, int],
                 rms_obs: Optional[RunningMoments],
                 *,
                 layer_norm: bool,
                 device: Optional[torch.device] = None):
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
                ("nl", nn.ReLU(inplace=True)),
            ]))),
            ("fc_block_2", nn.Sequential(OrderedDict([
                ("fc", nn.Linear(hid_dims[0], hid_dims[1], device=device)),
                ("ln", (nn.LayerNorm if self.layer_norm else nn.Identity)(hid_dims[1],
                                                                          device=device)),
                ("nl", nn.ReLU(inplace=True)),
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
                 max_ac: float,
                 *,
                 layer_norm: bool,
                 device: Optional[torch.device] = None):
        super().__init__()
        ob_dim = ob_shape[-1]
        self.ac_dim = ac_shape[-1]  # used in child class
        self.rms_obs = rms_obs
        self.max_ac = max_ac
        self.layer_norm = layer_norm

        # assemble the last layers and output heads
        self.fc_stack = nn.Sequential(OrderedDict([
            ("fc_block_1", nn.Sequential(OrderedDict([
                ("fc", nn.Linear(ob_dim, hid_dims[0], device=device)),
                ("ln", (nn.LayerNorm if self.layer_norm else nn.Identity)(hid_dims[0],
                                                                          device=device)),
                ("nl", nn.ReLU(inplace=True)),
            ]))),
            ("fc_block_2", nn.Sequential(OrderedDict([
                ("fc", nn.Linear(hid_dims[0], hid_dims[1], device=device)),
                ("ln", (nn.LayerNorm if self.layer_norm else nn.Identity)(hid_dims[1],
                                                                          device=device)),
                ("nl", nn.ReLU(inplace=True)),
            ]))),
        ]))
        self.head = nn.Linear(hid_dims[1], self.ac_dim, device=device)

        # perform initialization
        self.fc_stack.apply(init())
        self.head.apply(init())

    @beartype
    def act(self, ob: torch.Tensor) -> torch.Tensor:
        if self.rms_obs is not None:
            ob = self.rms_obs.standardize(ob).clamp(*STANDARDIZED_OB_CLAMPS)
        x = self.fc_stack(ob)
        return float(self.max_ac) * torch.tanh(self.head(x))


class TanhGaussActor(Actor):

    @beartype
    def __init__(self,
                 ob_shape: tuple[int, ...],
                 ac_shape: tuple[int, ...],
                 hid_dims: tuple[int, int],
                 rms_obs: Optional[RunningMoments],
                 max_ac: float,
                 *,
                 generator: torch.Generator,
                 layer_norm: bool,
                 device: Optional[torch.device] = None):
        super().__init__(ob_shape,
                         ac_shape,
                         hid_dims,
                         rms_obs,
                         max_ac,
                         layer_norm=layer_norm,
                         device=device)
        self.rng = generator
        # overwrite head
        self.head = nn.Linear(hid_dims[1], 2 * self.ac_dim, device=device)
        # perform initialization (since head written over)
        self.head.apply(init())
        # no need to init the Parameter type object

    @beartype
    def logp(self, ob: torch.Tensor, ac: torch.Tensor, max_ac: float) -> torch.Tensor:
        out = self.mean_std(ob)
        return TanhNormalToolkit.logp(ac, *out, x_scale=max_ac)  # mean, std

    @beartype
    def sample(self, ob: torch.Tensor, *, stop_grad: bool = True) -> torch.Tensor:
        with torch.no_grad() if stop_grad else nullcontext():
            out = self.mean_std(ob)
            return float(self.max_ac) * TanhNormalToolkit.sample(*out, generator=self.rng)

    @beartype
    def mode(self, ob: torch.Tensor, *, stop_grad: bool = True) -> torch.Tensor:
        with torch.no_grad() if stop_grad else nullcontext():
            mean, _ = self.mean_std(ob)
            return float(self.max_ac) * TanhNormalToolkit.mode(mean)

    @beartype
    @staticmethod
    def bound_log_std(log_std: torch.Tensor) -> torch.Tensor:
        """Stability trick from OpenAI SpinUp / Denis Yarats"""
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = SAC_LOG_STD_BOUNDS
        return log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

    @beartype
    def mean_std(self, ob: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.rms_obs is not None:
            ob = self.rms_obs.standardize(ob).clamp(*STANDARDIZED_OB_CLAMPS)
        x = self.fc_stack(ob)
        ac_mean, ac_log_std = self.head(x).chunk(2, dim=-1)
        ac_log_std = self.bound_log_std(ac_log_std)
        ac_std = ac_log_std.exp()
        return ac_mean, ac_std
