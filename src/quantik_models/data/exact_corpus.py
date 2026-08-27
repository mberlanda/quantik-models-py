"""Storage format for exactly-solved position corpora.

Every policy target produced by the oracle is uniform over a set of
outcome-optimal actions, so a dense `(n, 64) float32` array spends 256 bytes
per row to encode what is really a 64-bit set — and 90%+ of rows are
value-only, where those 256 bytes are all zeros. At corpus scale that is
790 MB of mostly-zero policy for 3M rows.

Storing the optimal set as one `uint64` bitmask instead costs 8 bytes per row
and is exact: the dense target is recovered as `mask / popcount(mask)`. An
empty mask means "value label only", which replaces the separate
`policy_weight` column.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from ..env import fastboard as fb

_ACTION_BITS = np.arange(fb.ACTION_COUNT, dtype=np.uint64)


def pack_actions(action_sets: list[list[int]]) -> npt.NDArray[np.uint64]:
    """Bitmask per row from lists of action indices."""
    out = np.zeros(len(action_sets), dtype=np.uint64)
    for i, actions in enumerate(action_sets):
        value = np.uint64(0)
        for action in actions:
            value |= np.uint64(1) << np.uint64(action)
        out[i] = value
    return out


def pack_dense(policy: npt.NDArray[np.float32]) -> npt.NDArray[np.uint64]:
    """Bitmask per row from a dense `(n, 64)` target (nonzero = in the set)."""
    bits = (policy > 0).astype(np.uint64)
    return (bits << _ACTION_BITS[None, :]).sum(axis=1, dtype=np.uint64)


def unpack(masks: npt.NDArray[np.uint64]) -> npt.NDArray[np.float32]:
    """Dense `(n, 64)` targets: uniform over each mask's set, zero if empty."""
    dense = ((masks[:, None] >> _ACTION_BITS[None, :]) & np.uint64(1)).astype(np.float32)
    total = dense.sum(axis=1, keepdims=True)
    return np.divide(dense, total, out=np.zeros_like(dense), where=total > 0)


def policy_weight(masks: npt.NDArray[np.uint64]) -> npt.NDArray[np.float32]:
    """1.0 where the row carries a policy label, 0.0 where it is value-only."""
    return (masks != 0).astype(np.float32)


@dataclass
class ExactCorpus:
    boards: npt.NDArray[np.uint16]
    optimal_mask: npt.NDArray[np.uint64]
    value_target: npt.NDArray[np.float32]
    plies: npt.NDArray[np.int16]

    def __len__(self) -> int:
        return int(self.boards.shape[0])

    @property
    def policy_rows(self) -> int:
        return int((self.optimal_mask != 0).sum())

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            boards=self.boards,
            optimal_mask=self.optimal_mask,
            value_target=self.value_target,
            plies=self.plies,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ExactCorpus":
        with np.load(path) as data:
            if "optimal_mask" in data.files:
                return cls(
                    boards=data["boards"],
                    optimal_mask=data["optimal_mask"],
                    value_target=data["value_target"],
                    plies=data["plies"],
                )
            # Legacy dense format, kept readable so older runs stay reproducible.
            return cls(
                boards=data["boards"],
                optimal_mask=pack_dense(data["policy_target"]),
                value_target=data["value_target"],
                plies=data["plies"],
            )

    @classmethod
    def concat(cls, parts: list["ExactCorpus"]) -> "ExactCorpus":
        """Merge corpora, keeping one row per canonical position.

        Rows carrying a policy label win over value-only rows for the same
        position; among equals the first part wins, so callers should pass the
        more authoritative corpus first.
        """
        merged = cls(
            boards=np.concatenate([p.boards for p in parts]),
            optimal_mask=np.concatenate([p.optimal_mask for p in parts]),
            value_target=np.concatenate([p.value_target for p in parts]),
            plies=np.concatenate([p.plies for p in parts]),
        )
        keys = fb.canonical_keys(merged.boards)
        has_policy = merged.optimal_mask != 0
        order = np.argsort(~has_policy, kind="stable")
        _, first = np.unique(keys[order], return_index=True)
        keep = np.sort(order[first])
        return cls(
            boards=merged.boards[keep],
            optimal_mask=merged.optimal_mask[keep],
            value_target=merged.value_target[keep],
            plies=merged.plies[keep],
        )
