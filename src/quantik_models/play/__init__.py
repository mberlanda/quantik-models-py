"""Storage for games played by humans against trained Quantik networks."""

from .store import connect, distinct_positions, game_count, head_to_head, record_game

__all__ = [
    "connect",
    "distinct_positions",
    "game_count",
    "head_to_head",
    "record_game",
]
