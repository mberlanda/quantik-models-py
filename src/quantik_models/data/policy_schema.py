"""Convert between the v1 and v2/v3 policy-label schemas, refusing to lose information.

v1 stores the policy label as a dense ``policy_target`` float32 ``(N, 64)`` plus a
``policy_weight`` float32 ``(N,)``; v2/v3 store a ``uint64`` ``optimal_mask`` ``(N,)``
(``docs/corpus-structure.md``). The two agree exactly *when the dense row is uniform over
its support*: the mask says which actions are optimal, and nothing else. A dense row that
is not uniform carries a weighting the mask cannot hold, so converting it anyway would
silently discard that weighting. This module raises instead, naming the row and the reason.

Bit ``i`` of the mask is action ``i`` and ``action = shape * 16 + position``
(``canonical-invariants.md#I6``); ``tests/test_policy_schema.py`` pins that order.

A row with ``policy_weight == 0`` is value-only: mask ``0``, dense row all zeros. Any other
weight, or a nonzero target under weight 0, is refused rather than dropped.

Uniformity is checked for *exact* equality with ``float32(1 / n)`` by default, because that
is what makes dense -> mask -> dense bit-exact. Every labelled row of the published v1
corpus satisfies it. ``atol`` loosens the check, but then the round trip only reproduces
the row to within ``atol``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

ACTION_COUNT = 64
_BITS = np.arange(ACTION_COUNT, dtype=np.uint64)
_MAX_REPORTED = 5


class NonUniformPolicyError(ValueError):
    """A v1 policy row cannot be represented as an ``optimal_mask`` without loss."""


def _fail(problems: list[tuple[int, str]], row_ids: Sequence[object] | None) -> None:
    def name(i: int) -> str:
        return f"row {i}" if row_ids is None else f"row {i} (id {row_ids[i]!r})"

    shown = "; ".join(f"{name(i)}: {why}" for i, why in problems[:_MAX_REPORTED])
    more = len(problems) - _MAX_REPORTED
    tail = f"; and {more} more" if more > 0 else ""
    raise NonUniformPolicyError(
        f"{len(problems)} policy row(s) cannot be converted to optimal_mask without "
        f"losing information: {shown}{tail}"
    )


def dense_to_mask(
    policy_target: np.ndarray,
    policy_weight: np.ndarray,
    *,
    row_ids: Sequence[object] | None = None,
    atol: float = 0.0,
) -> np.ndarray:
    """v1 dense ``(N, 64)`` target + ``(N,)`` weight -> ``uint64`` ``(N,)`` optimal mask.

    Raises `NonUniformPolicyError` (a `ValueError`) if any row is not exactly a uniform
    distribution over its support with weight 1, or a zero row with weight 0. The message
    names the offending rows (index, plus ``row_ids[i]`` when given) and the reason.
    """
    target = np.asarray(policy_target)
    weight = np.asarray(policy_weight)
    if target.ndim != 2 or target.shape[1] != ACTION_COUNT:
        raise ValueError(f"policy_target must have shape (N, {ACTION_COUNT}), got {target.shape}")
    if weight.shape != (len(target),):
        raise ValueError(f"policy_weight must have shape ({len(target)},), got {weight.shape}")
    if row_ids is not None and len(row_ids) != len(target):
        raise ValueError(f"row_ids has {len(row_ids)} entries for {len(target)} rows")

    target = target.astype(np.float32, copy=False)
    support = target > 0
    n = support.sum(axis=1)
    labelled = weight == 1
    value_only = weight == 0

    reasons: list[tuple[np.ndarray, str]] = []
    reasons.append((~np.isfinite(weight) | ~(labelled | value_only), "policy_weight is not 0 or 1 (its weighting cannot be stored in a mask)"))
    bad_values = ~np.isfinite(target).all(axis=1) | (target < 0).any(axis=1)
    reasons.append((bad_values, "policy_target has NaN, inf or negative entries"))
    reasons.append((value_only & (target != 0).any(axis=1), "policy_weight is 0 but policy_target is nonzero"))
    reasons.append((labelled & (n == 0), "policy_weight is 1 but policy_target has no support"))

    with np.errstate(divide="ignore", invalid="ignore"):
        expected = np.where(support, np.float32(1) / n.astype(np.float32)[:, None], np.float32(0))
    dev = np.abs(target - expected).max(axis=1)
    not_uniform = labelled & (n > 0) & ~bad_values & ~(dev <= atol)
    reasons.append((not_uniform, "policy_target is not uniform over its support"))

    problems: dict[int, str] = {}
    for mask, why in reasons:
        for i in np.flatnonzero(mask).tolist():
            if i in problems:
                continue
            if why.endswith("its support"):
                vals = target[i][support[i]]
                why_i = f"{why} ({n[i]} moves, values {vals.min():.9g}..{vals.max():.9g}, uniform would be {1.0 / n[i]:.9g})"
            else:
                why_i = why
            problems[i] = why_i
    if problems:
        _fail(sorted(problems.items()), row_ids)

    bits = support.astype(np.uint64) << _BITS
    mask_out: np.ndarray = np.bitwise_or.reduce(bits, axis=1) if len(target) else np.zeros(0, np.uint64)
    return np.where(labelled, mask_out, np.uint64(0)).astype(np.uint64)


def mask_to_dense(optimal_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``uint64`` ``(N,)`` optimal mask -> v1 ``(policy_target float32 (N, 64), policy_weight float32 (N,))``.

    A nonzero mask becomes a uniform row (``float32(1 / popcount)`` on each set bit) with
    weight 1; a zero mask becomes an all-zero row with weight 0.
    """
    mask = np.asarray(optimal_mask)
    if mask.ndim != 1:
        raise ValueError(f"optimal_mask must be 1-D, got shape {mask.shape}")
    mask = mask.astype(np.uint64, copy=False)
    bits = ((mask[:, None] >> _BITS) & np.uint64(1)).astype(bool)
    n = bits.sum(axis=1).astype(np.float32)
    with np.errstate(divide="ignore"):
        share = np.where(n > 0, np.float32(1) / np.maximum(n, np.float32(1)), np.float32(0)).astype(np.float32)
    target = np.where(bits, share[:, None], np.float32(0)).astype(np.float32)
    weight = (n > 0).astype(np.float32)
    return target, weight
