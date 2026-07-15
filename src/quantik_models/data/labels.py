"""Label weighting and provenance rules for Quantik training rows."""

from __future__ import annotations

from typing import Mapping

DEFAULT_VALUE_SOURCE_WEIGHTS: Mapping[str, float] = {
    "exact": 1.0,
    "tablebase": 1.0,
    "opening-book": 1.0,
    "opening_book": 1.0,
    "bounded": 0.9,
    "strong-search": 0.85,
    "strong_search": 0.85,
    "search": 0.7,
    "mcts": 0.7,
    "minimax": 0.7,
    "beam": 0.6,
    "selfplay": 0.65,
    "h2h": 0.45,
    "game-result": 0.45,
    "game_result": 0.45,
    "heuristic": 0.25,
    "synthetic": 0.2,
}

VALUE_PRECEDENCE = (
    "exact/tablebase/opening-book",
    "bounded solver",
    "strong-search",
    "search/mcts/minimax/beam",
    "selfplay outcome",
    "h2h/game-result calibration",
    "heuristic/synthetic",
)


def source_weight(value_source: str, overrides: Mapping[str, float] | None = None) -> float:
    weights = overrides or DEFAULT_VALUE_SOURCE_WEIGHTS
    return float(weights.get(value_source.lower(), 0.5))


def sample_weight(value_source: str, confidence: float, overrides: Mapping[str, float] | None = None) -> float:
    return float(min(1.0, max(0.0, confidence * source_weight(value_source, overrides))))
