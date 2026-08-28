"""A fully connected policy/value network — the architectural control.

Quantik is played on 4x4. A 3x3 convolution on a board that small already
sees most of the position in one step, so the usual justification for a
convolutional trunk — locality and weight sharing over a large grid — is
much weaker here than it is on 19x19 Go or 8x8 chess.

This network exists to make that argument falsifiable. It throws the
spatial structure away entirely: the `(9, 4, 4)` input is flattened to 144
features and pushed through residual dense blocks. If it matches the
ResNet at comparable parameter count, convolution is not earning its place
on this board. If it does not, the gap is the value of the inductive bias.

Torch-only module: import it behind the `[torch]` extra. As with every
architecture here, legality masking is deliberately outside the model —
see `masked_log_softmax` in `policy_value_net`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .spec import ACTION_COUNT, INPUT_FEATURES

MODEL_FAMILY = "quantik-policy-value-mlp"


@dataclass(frozen=True)
class MLPNetConfig:
    hidden: int
    blocks: int
    value_hidden: int = 64


# Widths are solved against the ResNet's parameter counts rather than
# chosen for roundness: the whole point of this network is that a
# difference in accuracy should be attributable to shape, and a control
# that quietly carries twice the capacity proves nothing. Dense layers
# scale as `2 * blocks * hidden^2`, so the width that matches a given
# ResNet is not a number anyone would guess.
PRESETS: dict[str, MLPNetConfig] = {
    # 305,285 parameters against `resnet-c64-b4`'s 304,711 (+0.2%).
    "small": MLPNetConfig(hidden=178, blocks=4),
    # 1,788,343 parameters against `resnet-c128-b6`'s 1,786,823 (+0.1%).
    "medium": MLPNetConfig(hidden=455, blocks=4),
    "smoke": MLPNetConfig(hidden=32, blocks=1),
}


class _DenseBlock(nn.Module):
    """Pre-activation residual dense block.

    Residual rather than plain stacking for the same reason the ResNet uses
    it: without the skip, gradients through four-plus dense layers with
    batch norm are noticeably worse behaved, and any depth difference would
    then confound the comparison this network exists to make.
    """

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, hidden, bias=False)
        self.bn2 = nn.BatchNorm1d(hidden)

    def forward(self, x: Tensor) -> Tensor:
        out = torch.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        return torch.relu(out + x)


class MLPNet(nn.Module):
    """Flattened trunk with a 64-logit policy head and a tanh value head."""

    def __init__(self, config: MLPNetConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden
        self.stem = nn.Sequential(
            nn.Flatten(),
            nn.Linear(INPUT_FEATURES, h, bias=False),
            nn.BatchNorm1d(h),
            nn.ReLU(),
        )
        self.trunk = nn.Sequential(*[_DenseBlock(h) for _ in range(config.blocks)])
        self.policy_head = nn.Linear(h, ACTION_COUNT)
        self.value_head = nn.Sequential(
            nn.Linear(h, config.value_hidden),
            nn.ReLU(),
            nn.Linear(config.value_hidden, 1),
            nn.Tanh(),
        )

    @property
    def architecture(self) -> str:
        return f"mlp-h{self.config.hidden}-b{self.config.blocks}"

    @property
    def model_family(self) -> str:
        return MODEL_FAMILY

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        trunk = self.trunk(self.stem(x))
        return self.policy_head(trunk), self.value_head(trunk).squeeze(-1)
