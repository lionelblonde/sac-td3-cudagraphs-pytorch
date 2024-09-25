from collections import OrderedDict
from typing import Callable, Optional

from beartype import beartype
from einops import pack
import torch
from torch import nn

from helpers import logger
from helpers.normalizer import RunningMoments


STANDARDIZED_OB_CLAMPS = [-5., 5.]


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


class Critic(nn.Module):

    @beartype
    def __init__(self,
                 ob_shape: tuple[int, ...],
                 ac_shape: tuple[int, ...],
                 hid_dims: tuple[int, int],
                 rms_obs: Optional[RunningMoments],
                 *,
                 layer_norm: bool):
        super().__init__()
        ob_dim = ob_shape[-1]
        ac_dim = ac_shape[-1]
        self.rms_obs = rms_obs
        self.layer_norm = layer_norm

        # assemble the last layers and output heads
        self.fc_stack = nn.Sequential(OrderedDict([
            ("fc_block_1", nn.Sequential(OrderedDict([
                ("fc", nn.Linear(ob_dim + ac_dim, hid_dims[0])),
                ("ln", (nn.LayerNorm if self.layer_norm else nn.Identity)(hid_dims[0])),
                ("nl", nn.Mish()),
            ]))),
            ("fc_block_2", nn.Sequential(OrderedDict([
                ("fc", nn.Linear(hid_dims[0], hid_dims[1])),
                ("ln", (nn.LayerNorm if self.layer_norm else nn.Identity)(hid_dims[1])),
                ("nl", nn.Mish()),
            ]))),
        ]))
        self.head = nn.Linear(hid_dims[1], 1)

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
                 layer_norm: bool):
        super().__init__()
        ob_dim = ob_shape[-1]
        ac_dim = ac_shape[-1]
        self.rms_obs = rms_obs
        self.max_ac = max_ac
        self.layer_norm = layer_norm

        # assemble the last layers and output heads
        self.fc_stack = nn.Sequential(OrderedDict([
            ("fc_block_1", nn.Sequential(OrderedDict([
                ("fc", nn.Linear(ob_dim, hid_dims[0])),
                ("ln", (nn.LayerNorm if self.layer_norm else nn.Identity)(hid_dims[0])),
                ("nl", nn.Mish()),
            ]))),
            ("fc_block_2", nn.Sequential(OrderedDict([
                ("fc", nn.Linear(hid_dims[0], hid_dims[1])),
                ("ln", (nn.LayerNorm if self.layer_norm else nn.Identity)(hid_dims[1])),
                ("nl", nn.Mish()),
            ]))),
        ]))
        self.head = nn.Linear(hid_dims[1], ac_dim)

        # perform initialization
        self.fc_stack.apply(init())
        self.head.apply(init())

    @beartype
    def act(self, ob: torch.Tensor) -> torch.Tensor:
        if self.rms_obs is not None:
            ob = self.rms_obs.standardize(ob).clamp(*STANDARDIZED_OB_CLAMPS)
        x = self.fc_stack(ob)
        return float(self.max_ac) * torch.tanh(self.head(x))
