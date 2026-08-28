"""Self-attention over the sixteen cells — the fourth architecture.

The reason for this network is **not** range, and an earlier draft of this
file said it was. On a 4x4 board the ResNet's stem plus its first residual
block already sees every cell, so "attention models long-range structure
and convolution does not" is a statement about 19x19 Go, not about Quantik.
Any architecture here that justifies itself on receptive field is
justifying itself on nothing.

The real axis is **content-dependent interaction**. A convolution applies
the same kernel regardless of what occupies the cells it reads: the weight
between two positions is fixed at training time. Attention makes that
weight a function of what is actually on the board, which is closer to how
Quantik's rule works — whether cell 5 constrains cell 7 depends entirely on
which shape sits on cell 5 and who played it.

`ConstraintPoolNet` makes the same bet with a much stronger prior: it is
told which cells are related, and only learns what to do about it. This
network is told nothing and has to discover the row, column and zone
structure as attention patterns. That is why it is fourth in the lineup
rather than third — it is the weaker form of the hypothesis
`ConstraintPoolNet` tests, and its value is in the comparison. If it
matches `cpool`, the explicit group wiring was not necessary; if it loses,
the prior was doing real work.

LayerNorm throughout, pre-norm blocks, and no batch statistics anywhere —
which also removes the batch-of-one serving failure.

Torch-only module: import it behind the `[torch]` extra. Legality masking
stays outside the model, as everywhere else.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .constraint_pool_net import flatten_cell_shape_logits
from .spec import CELL_COUNT, INPUT_PLANES, SHAPE_COUNT

MODEL_FAMILY = "quantik-policy-value-attention"


@dataclass(frozen=True)
class AttentionNetConfig:
    d_model: int
    blocks: int
    heads: int = 4
    ff_multiplier: int = 2
    value_hidden: int = 64


# Solved against the ResNet's counts; see `tests/test_parameter_matching.py`.
PRESETS: dict[str, AttentionNetConfig] = {
    "smoke": AttentionNetConfig(d_model=16, blocks=2, heads=2),
    # 308,485 parameters against `resnet-c64-b4`'s 304,711 (+1.2%).
    "small": AttentionNetConfig(d_model=96, blocks=4),
    # 1,800,709 parameters against `resnet-c128-b6`'s 1,786,823 (+0.8%).
    "medium": AttentionNetConfig(d_model=192, blocks=6),
}


class _EncoderBlock(nn.Module):
    """Pre-norm transformer block over the sixteen cell tokens.

    Pre-norm rather than post-norm because post-norm transformers need a
    warmup schedule to train stably, and the trainer's cosine schedule is
    shared across every architecture in the lineup. A model that needed its
    own schedule would not be comparable to the others.
    """

    def __init__(self, d_model: int, heads: int, ff_multiplier: int) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, heads, batch_first=True, dropout=0.0
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_multiplier * d_model),
            nn.GELU(),
            nn.Linear(ff_multiplier * d_model, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm_attn(x)
        attended, _ = self.attn(h, h, h, need_weights=False)
        x = x + attended
        return x + self.ffn(self.norm_ffn(x))


class AttentionNet(nn.Module):
    """Transformer encoder over cells, with the shared two heads."""

    def __init__(self, config: AttentionNetConfig) -> None:
        super().__init__()
        if config.d_model % config.heads:
            raise ValueError(
                f"d_model {config.d_model} is not divisible by heads {config.heads}"
            )
        self.config = config
        d = config.d_model

        self.stem = nn.Linear(INPUT_PLANES, d)
        # Learned rather than sinusoidal: there are exactly sixteen
        # positions, they never change, and a fixed encoding designed for
        # unbounded sequence length buys nothing here.
        self.position = nn.Parameter(torch.zeros(1, CELL_COUNT, d))
        nn.init.normal_(self.position, std=0.02)

        self.blocks = nn.ModuleList(
            _EncoderBlock(d, config.heads, config.ff_multiplier)
            for _ in range(config.blocks)
        )
        self.norm_out = nn.LayerNorm(d)

        # Per-cell logits over the four shapes, so the flatten has to
        # transpose — the same trap `ConstraintPoolNet` documents, and the
        # same tested helper.
        self.policy_head = nn.Linear(d, SHAPE_COUNT)
        self.value_head = nn.Sequential(
            nn.Linear(d, config.value_hidden),
            nn.GELU(),
            nn.Linear(config.value_hidden, 1),
            nn.Tanh(),
        )

    @property
    def architecture(self) -> str:
        return f"attn-d{self.config.d_model}-b{self.config.blocks}"

    @property
    def model_family(self) -> str:
        return MODEL_FAMILY

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # (B, P, 4, 4) -> (B, 16, P): one token per cell.
        cells = x.flatten(2).transpose(1, 2)
        h = self.stem(cells) + self.position
        for block in self.blocks:
            h = block(h)
        h = self.norm_out(h)
        policy = flatten_cell_shape_logits(self.policy_head(h))
        value = self.value_head(h.mean(dim=1)).squeeze(-1)
        return policy, value
