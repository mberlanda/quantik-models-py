"""Scoring agents against exact truth, with per-position outcomes.

Reporting only a mean accuracy throws away the fact that every agent faces the
*same* positions. Keeping the per-position outcome lets the comparison be
**paired**, which is far more powerful: two agents that agree on 95% of
positions can be separated by a handful of disagreements that an unpaired
confidence interval would call a tie.

Terminology: a position's *outcome accuracy* is measured only over positions
the mover provably wins. In a lost position every move loses, so there is
nothing to get right and including them would just dilute the measurement
toward the fraction of positions that are lost.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..env import fastboard as fb
from .parallel import _init_worker
from .registry import build_agent

_WORKER: dict[str, Any] = {}


@dataclass
class ProbeResult:
    agent: str
    plies: np.ndarray
    won: np.ndarray
    correct: np.ndarray
    seconds: float = 0.0
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def outcome_accuracy(self) -> float:
        return float(self.correct[self.won].mean()) if self.won.any() else 0.0

    def by_ply(self) -> dict[int, tuple[int, int]]:
        """`{ply: (correct, total)}` over won positions only."""
        out: dict[int, tuple[int, int]] = {}
        for ply in sorted(set(self.plies[self.won].tolist())):
            mask = self.won & (self.plies == ply)
            out[int(ply)] = (int(self.correct[mask].sum()), int(mask.sum()))
        return out


def _score_one(job):
    spec, qfens, optimal, seed_base = job
    key = repr(sorted(spec.items(), key=lambda kv: kv[0]))
    if key not in _WORKER:
        _WORKER[key] = build_agent(spec)
    agent = _WORKER[key]
    out = np.zeros(len(qfens), dtype=bool)
    for i, (qfen, best) in enumerate(zip(qfens, optimal)):
        board = fb.from_qfen(qfen)[0]
        out[i] = agent.select(board, seed_base + i) in best
    return out


def score(spec: dict, probe: list[dict], workers: int | None = None, seed: int = 0) -> ProbeResult:
    """Run one agent over the probe, in parallel, keeping per-position results."""
    import time

    agent_name = build_agent(spec).name
    qfens = [r["qfen"] for r in probe]
    optimal = [set(r["outcome_optimal"]) for r in probe]
    boards = np.concatenate([fb.from_qfen(q) for q in qfens])
    plies = fb.popcount(fb.occupancy(boards))
    won = np.array([bool(r["won"]) for r in probe])

    workers = workers or min(os.cpu_count() or 4, 12)
    stride = max(1, math.ceil(len(probe) / workers))
    jobs = [
        (spec, qfens[i : i + stride], optimal[i : i + stride], seed + i)
        for i in range(0, len(probe), stride)
    ]
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        parts = list(pool.map(_score_one, jobs))
    return ProbeResult(
        agent=agent_name,
        plies=plies,
        won=won,
        correct=np.concatenate(parts),
        seconds=time.perf_counter() - started,
    )


def _binomial_two_sided(successes: int, trials: int) -> float:
    """Exact two-sided p-value for a fair-coin null."""
    if trials == 0:
        return 1.0
    probabilities = [math.comb(trials, k) * 0.5**trials for k in range(trials + 1)]
    observed = probabilities[successes]
    # Standard exact two-sided rule: sum every outcome no more likely than
    # the one observed. The 1e-9 slack keeps floating-point ties symmetric.
    return min(1.0, sum(p for p in probabilities if p <= observed * (1 + 1e-9)))


def mcnemar(a: ProbeResult, b: ProbeResult, plies: tuple[int, ...] | None = None) -> dict:
    """Paired comparison of two agents over the same won positions.

    Only positions where the two disagree carry information, so the test is a
    fair-coin question about the split of those disagreements. This is the
    right instrument here: the agents are evaluated on identical positions, and
    an unpaired interval would waste that.
    """
    mask = a.won.copy()
    if plies is not None:
        mask &= np.isin(a.plies, plies)
    a_only = int((mask & a.correct & ~b.correct).sum())
    b_only = int((mask & ~a.correct & b.correct).sum())
    both = int((mask & a.correct & b.correct).sum())
    neither = int((mask & ~a.correct & ~b.correct).sum())
    total = int(mask.sum())
    return {
        "positions": total,
        "accuracy_a": both + a_only and (both + a_only) / total or 0.0,
        "accuracy_b": both + b_only and (both + b_only) / total or 0.0,
        "a_right_b_wrong": a_only,
        "b_right_a_wrong": b_only,
        "both_right": both,
        "both_wrong": neither,
        "p_value": _binomial_two_sided(a_only, a_only + b_only),
    }


def paired_difference_ci(
    a: ProbeResult, b: ProbeResult, plies: tuple[int, ...] | None = None,
    resamples: int = 20_000, seed: int = 12345,
) -> tuple[float, float, float]:
    """Bootstrap 95% interval on (a - b) accuracy, resampling positions jointly."""
    mask = a.won.copy()
    if plies is not None:
        mask &= np.isin(a.plies, plies)
    diff = a.correct[mask].astype(np.int8) - b.correct[mask].astype(np.int8)
    if diff.size == 0:
        return (0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, diff.size, size=(resamples, diff.size))
    means = diff[draws].mean(axis=1)
    return (float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))
