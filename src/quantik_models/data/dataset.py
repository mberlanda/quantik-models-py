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

from .materialize import load_npz

Split = Literal["train", "val", "test"]

_TAG_JOIN = "|"


def expand_legal_mask(mask: np.ndarray) -> np.ndarray:
    """Expand `(n,)` uint64 bitmasks into `(n, 64)` bool, LSB = action 0."""
    if mask.dtype != np.uint64:
        raise ValueError(f"legal mask must be uint64, got {mask.dtype}")
    bits = np.arange(64, dtype=np.uint64)
    return ((mask[:, None] >> bits[None, :]) & np.uint64(1)).astype(np.bool_)


def split_assignments(
    tensors: np.ndarray,
    policy_target: np.ndarray,
    source_tags: Sequence[str],
    *,
    train_pct: int = 80,
    val_pct: int = 10,
) -> np.ndarray:
    """Deterministic per-row split labels.

    bucket = (first 8 big-endian bytes of sha1(tensor_bytes || policy_bytes
    || tag)) mod 100:
    [0, train_pct) -> train, [train_pct, train_pct+val_pct) -> val,
    the rest -> test.
    """
    if not 0 < train_pct + val_pct < 100:
        raise ValueError("train_pct + val_pct must be in (0, 100)")
    labels = np.empty(len(source_tags), dtype=object)
    for i, tag in enumerate(source_tags):
        digest = hashlib.sha1(
            tensors[i].tobytes() + policy_target[i].tobytes() + tag.encode("utf-8")
        ).digest()
        bucket = int.from_bytes(digest[:8], "big") % 100
        if bucket < train_pct:
            labels[i] = "train"
        elif bucket < train_pct + val_pct:
            labels[i] = "val"
        else:
            labels[i] = "test"
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
    if split is not None:
        labels = split_assignments(
            tensors, policy, tags, train_pct=train_pct, val_pct=val_pct
        )
        keep = labels == split
        tensors, policy, value = tensors[keep], policy[keep], value[keep]
        weight, mask = weight[keep], mask[keep]
        tags = tuple(t for t, k in zip(tags, keep) if k)
    return LoadedTrainingData(
        tensors=tensors,
        policy_target=policy,
        value_target=value,
        sample_weight=weight,
        legal_mask=mask,
        source_tags=tags,
    )
