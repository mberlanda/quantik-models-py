"""Metric aggregation for the training loops.

Pure NumPy, and deliberately outside `supervised.py`: that module imports
torch, and the base install has no torch. Keeping the aggregation here lets
it be tested — and reused — without an accelerator stack.
"""

from __future__ import annotations


def merge_weighted(chunks: list[dict[str, tuple[float, float]]]) -> dict[str, float]:
    """Weighted mean per metric.

    A plain mean over chunks is wrong here: the corpus stores every
    policy-labelled row before every value-only row, so a sorted validation
    index puts nearly all policy rows in one chunk and none in the rest.
    Averaging those chunks equally divided the policy metrics by the chunk
    count — 89% top-1 was being reported as 11%.
    """
    out: dict[str, float] = {}
    for key in chunks[0]:
        total = sum(value * weight for value, weight in (c[key] for c in chunks))
        denominator = sum(weight for _, weight in (c[key] for c in chunks))
        out[key] = total / denominator if denominator else 0.0
    return out
