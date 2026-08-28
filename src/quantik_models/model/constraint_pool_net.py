"""A network wired to Quantik's twelve constraint groups.

Quantik's rule is not about spatial adjacency. A shape may not be placed
where the *opponent* already has that shape in the same row, column, or
2x2 zone — twelve overlapping groups, every cell in exactly three of them,
and the same twelve groups are also the win conditions.

A 3x3 convolution does not align to any of them. A row is a 1x4 strip, a
zone is a 2x2 block, and a kernel centred on a cell straddles the
boundaries of both, so the ResNet has to approximate a group-wise
predicate out of stacked local ones. This architecture asks whether
writing the groups into the wiring is worth anything: each block pools
cell features into the twelve groups, transforms them there, and scatters
the result back to the member cells. One round of bipartite message
passing over the game's actual constraint structure.

Rows and columns are exchanged by transposition, which is in D4, and the
zone partition survives it — so rows and columns share one set of group
weights and zones have their own. That is the symmetry-consistent choice,
not merely the economical one.

LayerNorm rather than BatchNorm throughout: it removes the batch-of-one
failure that only ever shows up in serving, and opset 17 has it natively.

Torch-only module: import it behind the `[torch]` extra. Legality masking
stays outside the model, as everywhere else — see `masked_log_softmax`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .spec import (
    ACTION_COUNT,
    BOARD_SIZE,
    CELL_COUNT,
    GROUP_COUNT,
    GROUP_KINDS,
    GROUPS_PER_CELL,
    INPUT_PLANES,
    SHAPE_COUNT,
    constraint_groups,
)

MODEL_FAMILY = "quantik-policy-value-constraint-pool"


@dataclass(frozen=True)
class ConstraintPoolNetConfig:
    channels: int
    blocks: int
    # Width of the per-cell feed-forward, as a multiple of `channels`.
    expansion: int = 2
    value_hidden: int = 64


# Solved against the ResNet's parameter counts rather than chosen for
# roundness — see `tests/test_parameter_matching.py`.
PRESETS: dict[str, ConstraintPoolNetConfig] = {
    "smoke": ConstraintPoolNetConfig(channels=16, blocks=2),
    # 307,333 parameters against `resnet-c64-b4`'s 304,711 (+0.9%).
    "small": ConstraintPoolNetConfig(channels=96, blocks=4),
    # 1,780,253 parameters against `resnet-c128-b6`'s 1,786,823 (-0.4%).
    "medium": ConstraintPoolNetConfig(channels=191, blocks=6),
}


def _membership() -> tuple[Tensor, Tensor, Tensor]:
    """Pooling and scattering matrices, plus the group-kind index.

    Both matrices are mean-reducing rather than sum-reducing: every group
    has four cells and every cell three groups, so the two are equivalent
    up to a constant here, but mean keeps activations on the same scale as
    the cell features they are mixed with.
    """
    groups = constraint_groups()
    incidence = torch.zeros(GROUP_COUNT, CELL_COUNT)
    for g, cells in enumerate(groups):
        for cell in cells:
            incidence[g, cell] = 1.0

    # (G, N): each group averages its member cells.
    pool = incidence / incidence.sum(dim=1, keepdim=True)
    # (N, G): each cell averages the groups it belongs to.
    scatter = incidence.t() / incidence.t().sum(dim=1, keepdim=True)

    kinds = sorted(set(GROUP_KINDS))
    kind_index = torch.tensor([kinds.index(k) for k in GROUP_KINDS], dtype=torch.long)
    return pool, scatter, kind_index


def flatten_cell_shape_logits(per_cell: Tensor) -> Tensor:
    """`(B, N, S)` per-cell logits -> `(B, 64)` in contract action order.

    Actions are indexed `action_index = shape * 16 + position`, so the
    shape axis is the *outer* one. A head that emits `(cell, shape)` and
    flattens it directly produces `position * 4 + shape`, which is a
    perfectly plausible-looking tensor of the right dtype and shape with
    every logit on the wrong action — the network would train against a
    permuted target and quietly learn the permutation, or fail to.

    Exposed and tested separately for exactly that reason.
    """
    return per_cell.transpose(1, 2).reshape(-1, ACTION_COUNT)


class _ConstraintBlock(nn.Module):
    """One round of cells -> groups -> cells message passing.

    The group transform is shared across all twelve groups; what
    distinguishes them is a learned embedding indexed by kind, added to the
    pooled summary. Sharing is what makes the block a statement about
    constraint structure rather than twelve independent little networks,
    and it is why the parameter count does not scale with the number of
    groups.
    """

    def __init__(self, channels: int, expansion: int, kind_count: int) -> None:
        super().__init__()
        c = channels
        self.norm_in = nn.LayerNorm(c)
        self.kind_embedding = nn.Embedding(kind_count, c)
        self.group_mlp = nn.Sequential(
            nn.Linear(c, c),
            nn.GELU(),
            nn.Linear(c, c),
        )
        # Mixes a cell's own features with the summary of its three groups.
        self.merge = nn.Linear(2 * c, c)

        self.norm_ffn = nn.LayerNorm(c)
        self.ffn = nn.Sequential(
            nn.Linear(c, expansion * c),
            nn.GELU(),
            nn.Linear(expansion * c, c),
        )

    def forward(
        self, x: Tensor, pool: Tensor, scatter: Tensor, kind_index: Tensor
    ) -> Tensor:
        # x: (B, N, C)
        h = self.norm_in(x)
        # Pool and scatter go through `F.linear` on a transposed view rather
        # than `pool @ h`. A plain matmul of a 2-D constant against a 3-D
        # activation traces to ONNX as a Gemm with the batch folded into a
        # hard-coded Reshape, which then fails on any batch but the one that
        # was traced. `F.linear` contracts the last axis and keeps the batch
        # dimension dynamic.
        groups = F.linear(h.transpose(1, 2), pool).transpose(1, 2)  # (B, G, C)
        groups = groups + self.kind_embedding(kind_index)
        groups = self.group_mlp(groups)
        back = F.linear(groups.transpose(1, 2), scatter).transpose(1, 2)  # (B, N, C)
        x = x + self.merge(torch.cat([h, back], dim=-1))
        return x + self.ffn(self.norm_ffn(x))


class ConstraintPoolNet(nn.Module):
    """Group-pooling trunk with a 64-logit policy head and a tanh value."""

    def __init__(self, config: ConstraintPoolNetConfig) -> None:
        super().__init__()
        self.config = config
        c = config.channels

        pool, scatter, kind_index = _membership()
        # Buffers, not parameters: the constraint structure is the game's,
        # not something to learn. Registering them keeps them on whatever
        # device the model moves to and puts them in the ONNX graph as
        # constants.
        self.register_buffer("pool", pool, persistent=False)
        self.register_buffer("scatter", scatter, persistent=False)
        self.register_buffer("kind_index", kind_index, persistent=False)

        self.stem = nn.Linear(INPUT_PLANES, c)
        self.blocks = nn.ModuleList(
            _ConstraintBlock(c, config.expansion, len(set(GROUP_KINDS)))
            for _ in range(config.blocks)
        )
        self.norm_out = nn.LayerNorm(c)

        # Per-cell logits over the four shapes. The transpose in `forward`
        # is load-bearing: actions are indexed `shape * 16 + position`, so
        # flattening (cell, shape) directly would put every logit on the
        # wrong action.
        self.policy_head = nn.Linear(c, SHAPE_COUNT)
        self.value_head = nn.Sequential(
            nn.Linear(c, config.value_hidden),
            nn.GELU(),
            nn.Linear(config.value_hidden, 1),
            nn.Tanh(),
        )

    @property
    def architecture(self) -> str:
        return f"cpool-c{self.config.channels}-b{self.config.blocks}"

    @property
    def model_family(self) -> str:
        return MODEL_FAMILY

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # (B, P, 4, 4) -> (B, N, P): one token per cell, planes as features.
        cells = x.flatten(2).transpose(1, 2)
        h = self.stem(cells)
        for block in self.blocks:
            h = block(h, self.pool, self.scatter, self.kind_index)
        h = self.norm_out(h)

        policy = flatten_cell_shape_logits(self.policy_head(h))

        value = self.value_head(h.mean(dim=1)).squeeze(-1)
        return policy, value


assert BOARD_SIZE * BOARD_SIZE == CELL_COUNT
assert GROUPS_PER_CELL == 3
