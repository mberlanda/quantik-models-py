"""Training-data loading over materialized `.npz` views.

Pure NumPy: this module must import without torch so the base install
can inspect datasets. Sharding is content-addressed so a row's split is
stable across file order, file grouping, and machines.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import numpy.typing as npt

from ..env import fastboard as fb

from .materialize import load_npz

Split = Literal["train", "val", "test"]

_TAG_JOIN = "|"


def expand_legal_mask(mask: np.ndarray) -> np.ndarray:
    """Expand `(n,)` uint64 bitmasks into `(n, 64)` bool, LSB = action 0."""
    if mask.dtype != np.uint64:
        raise ValueError(f"legal mask must be uint64, got {mask.dtype}")
    bits = np.arange(64, dtype=np.uint64)
    return ((mask[:, None] >> bits[None, :]) & np.uint64(1)).astype(np.bool_)


def boards_from_tensors(tensors: np.ndarray) -> npt.NDArray[np.uint16]:
    """Recover `(n, 8)` uint16 bitboards from `(n, 9, 4, 4)` encodings.

    Inverts `fastboard.encode_tensors`, which is **mover-relative**:
    channels 0-3 hold the side-to-move's shapes and 4-7 the opponent's, so
    the plane order swaps with parity. Channel 8 carries the side to move
    and is what tells us which way round this row is.

    Note that `quantik_core.ml_data.qfen_to_tensor` orders the same nine
    planes by colour instead. Both call themselves `tensor-board.v1`; this
    inverse matches the one the trainer actually feeds the network.
    """
    tensors = np.asarray(tensors)
    n = len(tensors)
    player = (tensors[:, 8].reshape(n, -1)[:, 0] > 0.5).astype(np.int64)

    planes = tensors[:, :8].reshape(n, 8, 16) > 0.5
    weights = 1 << np.arange(16, dtype=np.int64)
    packed = (planes * weights).sum(axis=2).astype(np.uint16)

    rows = np.arange(n)
    order = np.concatenate(
        [
            (player[:, None] * 4) + np.arange(4),
            ((1 - player)[:, None] * 4) + np.arange(4),
        ],
        axis=1,
    )
    boards = np.zeros((n, 8), dtype=np.uint16)
    boards[rows[:, None], order] = packed
    return boards


def split_assignments(
    tensors: np.ndarray,
    policy_target: np.ndarray,
    source_tags: Sequence[str],
    *,
    train_pct: int = 80,
    val_pct: int = 10,
) -> np.ndarray:
    """Deterministic per-row split labels, keyed on the canonical position.

    The bucket is derived from `canonical_keys`, so every row describing the
    same game position lands on the same side of the split no matter how it
    was reached: a rotation of it, a relabelling of its shapes, a different
    policy target, or a different source corpus.

    This replaces a hash of `tensor_bytes || policy_bytes || source_tag`,
    which leaked in two ways. Symmetric images of one position have
    different tensor bytes, so a rotated copy of a training board could sit
    in validation; and the same position carrying different targets across
    merged corpora split independently. Both inflate validation scores by
    letting the model be tested on what it memorised.

    `policy_target` and `source_tags` are kept in the signature — the row
    count is validated against them — but no longer contribute to the
    bucket. That is the point: membership must depend on the position and
    nothing else.
    """
    if not 0 < train_pct + val_pct < 100:
        raise ValueError("train_pct + val_pct must be in (0, 100)")
    if len(tensors) != len(source_tags) or len(tensors) != len(policy_target):
        raise ValueError("tensors, policy_target and source_tags must align")

    keys = fb.canonical_keys(boards_from_tensors(tensors))
    # Same 24-bit spread as `exact_corpus.split_by_key`, so both entry
    # points agree on which positions are held out.
    spread = (keys * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(40)
    bucket = (spread * np.uint64(100)) >> np.uint64(24)

    labels = np.empty(len(tensors), dtype=object)
    labels[:] = "test"
    labels[bucket < np.uint64(train_pct)] = "train"
    in_val = (bucket >= np.uint64(train_pct)) & (bucket < np.uint64(train_pct + val_pct))
    labels[in_val] = "val"
    return labels
    return labels.astype(np.str_)


@dataclass(frozen=True)
class LoadedTrainingData:
    """One concatenated (and optionally split-filtered) training view."""

    tensors: np.ndarray
    policy_target: np.ndarray
    value_target: np.ndarray
    sample_weight: np.ndarray
    legal_mask: np.ndarray
    source_tags: tuple[str, ...]


def load_training_data(
    paths: Sequence[str | Path],
    split: Split | None = None,
    *,
    train_pct: int = 80,
    val_pct: int = 10,
) -> LoadedTrainingData:
    """Load and concatenate `.npz` views, optionally filtering to a split."""
    if not paths:
        raise ValueError("at least one .npz path is required")
    views = [load_npz(path) for path in paths]
    tensors = np.concatenate([v.tensors for v in views])
    policy = np.concatenate([v.policy_target for v in views])
    value = np.concatenate([v.value_target for v in views])
    weight = np.concatenate([v.sample_weight for v in views])
    mask = expand_legal_mask(np.concatenate([v.legal_action_mask for v in views]))
    tags = tuple(
        _TAG_JOIN.join(row_tags) for v in views for row_tags in v.source_tags
    )
    data = LoadedTrainingData(
        tensors=tensors,
        policy_target=policy,
        value_target=value,
        sample_weight=weight,
        legal_mask=mask,
        source_tags=tags,
    )
    if split is not None:
        labels = split_assignments(
            tensors, policy, tags, train_pct=train_pct, val_pct=val_pct
        )
        data = subset(data, labels == split)
    return data


def subset(data: LoadedTrainingData, keep: np.ndarray) -> LoadedTrainingData:
    """Row-filter a loaded view by a boolean mask."""
    return LoadedTrainingData(
        tensors=data.tensors[keep],
        policy_target=data.policy_target[keep],
        value_target=data.value_target[keep],
        sample_weight=data.sample_weight[keep],
        legal_mask=data.legal_mask[keep],
        source_tags=tuple(t for t, k in zip(data.source_tags, keep) if k),
    )
