"""Shape constants every Quantik policy/value architecture agrees on.

These are the model side of the tensor contract: `tensor-board.v1` on the
way in, `policy-logits-64+value-tanh` on the way out. Every architecture in
this package consumes and produces exactly these shapes, which is what
makes them substitutable behind one evaluator and comparable to each other.

The action layout is the one shared across the whole project:

    action_index = shape * 16 + position
    position     = row * 4 + col

An architecture that emits per-cell logits therefore has to transpose
before flattening — see `ConstraintPoolNet`.
"""

from __future__ import annotations

BOARD_SIZE = 4
CELL_COUNT = BOARD_SIZE * BOARD_SIZE  # 16
SHAPE_COUNT = 4
ACTION_COUNT = SHAPE_COUNT * CELL_COUNT  # 64

# Eight bitboard planes (two players x four shapes) plus one constant plane
# carrying the side to move, matching `quantik_core.ml_data.qfen_to_tensor`.
INPUT_PLANES = 9
INPUT_FEATURES = INPUT_PLANES * CELL_COUNT  # 144


# The twelve constraint groups: four rows, four columns, four 2x2 zones.
# Quantik's placement rule is defined over exactly these — a shape may not
# go where the opponent already has that shape in the same row, column or
# zone — and the same twelve are the win conditions. Every cell belongs to
# exactly three.
#
# Restated here rather than imported from `env.fastboard` so that the model
# package stays free of the numpy board code; `tests/test_constraint_groups
# .py` asserts these are the same twelve as `fastboard.WIN_MASKS`, so the
# duplication cannot drift.
GROUP_COUNT = 12
GROUPS_PER_CELL = 3


def constraint_groups() -> tuple[tuple[int, ...], ...]:
    """Cell indices of each group, rows then columns then zones."""
    n = BOARD_SIZE
    rows = tuple(tuple(r * n + c for c in range(n)) for r in range(n))
    cols = tuple(tuple(r * n + c for r in range(n)) for c in range(n))
    half = n // 2
    zones = tuple(
        tuple(
            (zr * half + r) * n + (zc * half + c)
            for r in range(half)
            for c in range(half)
        )
        for zr in range(half)
        for zc in range(half)
    )
    return rows + cols + zones


# Rows and columns are exchanged by transposition, which is in D4, and the
# zone partition is preserved by it. A model that treats rows and columns
# as one kind of group is therefore consistent with the game's symmetry
# rather than merely economical.
GROUP_KINDS = ("line",) * (2 * BOARD_SIZE) + ("zone",) * BOARD_SIZE
