from collections import OrderedDict
from typing import Callable, Optional, Union

from beartype import beartype
from einops import pack
import torch
from torch import nn
from torch.distributions import Normal

from helpers import logger


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
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
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


class Critic(nn.Module):

    @beartype
    def __init__(self,
                 ob_shape: tuple[int, ...],
                 ac_shape: tuple[int, ...],
                 hid_dims: tuple[int, int],
                 *,
                 layer_norm: bool,
                 device: Optional[Union[str, torch.device]] = None):
        super().__init__()
        ob_dim = ob_shape[-1]
        ac_dim = ac_shape[-1]
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
        x, _ = pack([ob, ac], "b *")
        x = self.fc_stack(x)
        return self.head(x)


class Actor(nn.Module):

    @beartype
    def __init__(self,
                 ob_shape: tuple[int, ...],
                 ac_shape: tuple[int, ...],
                 hid_dims: tuple[int, int],
                 min_ac: torch.Tensor,
                 max_ac: torch.Tensor,
                 *,
                 exploration_noise: float,
                 layer_norm: bool,
                 device: Optional[Union[str, torch.device]] = None):
        super().__init__()
        ob_dim = ob_shape[-1]
        ac_dim = ac_shape[-1]
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
                 min_ac: torch.Tensor,
                 max_ac: torch.Tensor,
                 *,
                 generator: torch.Generator,
                 layer_norm: bool,
                 device: Optional[Union[str, torch.device]] = None):
        super().__init__()
        ob_dim = ob_shape[-1]
        ac_dim = ac_shape[-1]
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

    @staticmethod
    @beartype
    def bound_log_std(log_std: torch.Tensor) -> torch.Tensor:
        """Stability trick from OpenAI SpinUp / Denis Yarats"""
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = SAC_LOG_STD_BOUNDS
        return log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

    @beartype
    def forward(self, ob: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.fc_stack(ob)
        mean, log_std = self.head(x).chunk(2, dim=-1)
        log_std = self.bound_log_std(log_std)
        std = log_std.exp()
        return mean, std

    @beartype
    def get_action(self, ob: torch.Tensor) -> dict[str, torch.Tensor]:
        mean, std = self(ob)
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return {"sample": action, "log_prob": log_prob, "mode": mean}
