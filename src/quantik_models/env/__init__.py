"""Fast vectorized Quantik environment for self-play training."""

from .fastboard import (
    ACTION_COUNT,
    BatchBoard,
    apply_actions,
    encode_tensors,
    legal_masks,
    side_to_move,
    terminal_status,
)

__all__ = [
    "ACTION_COUNT",
    "BatchBoard",
    "apply_actions",
    "encode_tensors",
    "legal_masks",
    "side_to_move",
    "terminal_status",
]
