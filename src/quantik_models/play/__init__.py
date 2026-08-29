"""The local play service: what can be seated, and what the games become.

Three pieces, deliberately separable. `registry.scan_models` discovers
trained checkpoints; `opponents.roster` turns those plus the classical
engines into the single lineup a browser dropdown shows; `store` holds
the finished games. The HTTP surface that ties them together is a
separate step, so each of these can be tested without a server running.
"""

from __future__ import annotations

from .store import connect, distinct_positions, game_count, head_to_head, record_game

SERVICE_VERSION = "0.1.0"

# The contracts release this service's manifests and specs are read
# against. Pinned rather than resolved at runtime the way
# `export/checkpoint.py` resolves it for its own writes, because this
# package only ever reads checkpoints someone else stamped; a fixed
# expectation is what lets a mismatch surface as a refusal reason instead
# of a version silently drifting apart from what quantik-core-py accepts.
CONTRACT_VERSION = "1.2.0"

__all__ = [
    "CONTRACT_VERSION",
    "SERVICE_VERSION",
    "connect",
    "distinct_positions",
    "game_count",
    "head_to_head",
    "record_game",
]
