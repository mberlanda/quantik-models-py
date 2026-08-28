"""Shape constants every Quantik policy/value architecture agrees on.

These are the model side of the tensor contract: `tensor-board.v1` on the
way in, `policy-logits-64+value-tanh` on the way out. Every architecture in
this package consumes and produces exactly these shapes, which is what
makes them substitutable behind one evaluator and comparable to each other.

The action layout is the one shared across the whole project:

    action_index = shape * 16 + position
    position     = row * 4 + col

An architecture that emits per-cell logits therefore has to transpose
before flattening — see `AttentionNet`.
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
