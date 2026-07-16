"""AlphaZero-style policy/value network for Quantik.

Torch-only module: import it behind the `[torch]` extra. Legality
masking is deliberately NOT part of the model — `masked_log_softmax`
is the single shared implementation used by both the training loss and
engine adapters, per the model-checkpoint.v1 note that runtimes must
apply legal action masks outside the model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class PolicyValueNetConfig:
    channels: int
    blocks: int
    value_hidden: int = 64


PRESETS: dict[str, PolicyValueNetConfig] = {
    # CI-fast preset for smoke tests and examples (<1 MB).
    "smoke": PolicyValueNetConfig(channels=16, blocks=2),
    # Laptop baseline (single-digit MB).
    "small": PolicyValueNetConfig(channels=64, blocks=4),
    # Sized so float32 safetensors lands in the 50-100 MB contract
    # envelope (~15.4M parameters ~= 61 MB).
    "target": PolicyValueNetConfig(channels=256, blocks=13),
}


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + x)


class PolicyValueNet(nn.Module):
    """Shared trunk with a 64-logit policy head and a tanh value head."""

    def __init__(self, config: PolicyValueNetConfig) -> None:
        super().__init__()
        self.config = config
        c = config.channels
        self.stem = nn.Sequential(
            nn.Conv2d(9, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(),
        )
        self.trunk = nn.Sequential(*[_ResidualBlock(c) for _ in range(config.blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(c, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 16, 64),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(c, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16, config.value_hidden),
            nn.ReLU(),
            nn.Linear(config.value_hidden, 1),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        trunk = self.trunk(self.stem(x))
        return self.policy_head(trunk), self.value_head(trunk).squeeze(-1)


def masked_log_softmax(logits: Tensor, legal_mask: Tensor) -> Tensor:
    """Log-softmax with illegal logits forced to -inf (prob 0)."""
    # Note: an all-False mask row yields a uniform distribution over all 64
    # actions rather than NaN, since finfo.min keeps the arithmetic finite;
    # callers must not pass all-illegal rows expecting an error.
    masked = logits.masked_fill(~legal_mask, torch.finfo(logits.dtype).min)
    return torch.log_softmax(masked, dim=-1)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
