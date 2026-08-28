"""Reachable Quantik positions for tests.

Several fixtures used to build training views from `rng.random((n, 9, 4, 4))`.
That was fine while the train/val split hashed raw bytes, but the split is
now keyed on the canonical position, which needs an encoding that decodes
back to a legal board. Noise also gave the network inputs unlike anything
it sees in training, so the fixtures were weaker than they appeared.

This plays random legal moves instead, which is both valid and
representative.
"""

from __future__ import annotations

import numpy as np

from quantik_models.env import fastboard as fb


def random_positions(n: int, *, seed: int = 0, plies: int = 4) -> np.ndarray:
    """`(n, 8)` uint16 boards reached by `plies` random legal moves."""
    rng = np.random.default_rng(seed)
    boards = fb.empty_boards(n)
    for _ in range(plies):
        legal = fb.legal_masks(boards)
        alive = legal.any(axis=1)
        if not alive.any():
            break
        weights = legal.astype(np.float64)
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1.0)
        actions = np.array(
            [rng.choice(len(w), p=w) if a else 0 for w, a in zip(weights, alive)],
            dtype=np.int64,
        )
        boards = np.where(alive[:, None], fb.apply_actions(boards, actions), boards)
    return boards


def random_tensors(n: int, *, seed: int = 0, plies: int = 4) -> np.ndarray:
    """`(n, 9, 4, 4)` float32 encodings of reachable positions."""
    return fb.encode_tensors(random_positions(n, seed=seed, plies=plies))
