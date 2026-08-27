"""The paired probe comparison — the instrument behind every accuracy claim."""

from __future__ import annotations

import numpy as np
import pytest

from quantik_models.arena.probe import (
    ProbeResult,
    _binomial_two_sided,
    mcnemar,
    paired_difference_ci,
)


def _result(name, correct, won=None, plies=None):
    correct = np.array(correct, dtype=bool)
    n = correct.size
    return ProbeResult(
        agent=name,
        plies=np.array(plies if plies is not None else [8] * n),
        won=np.array(won if won is not None else [True] * n),
        correct=correct,
    )


@pytest.mark.parametrize(
    "successes,trials,expected",
    [(0, 0, 1.0), (5, 10, 1.0), (10, 10, 2 * 0.5**10), (0, 10, 2 * 0.5**10)],
)
def test_binomial_two_sided_matches_hand_calculation(successes, trials, expected):
    assert _binomial_two_sided(successes, trials) == pytest.approx(expected)


def test_binomial_two_sided_is_symmetric():
    for k in range(21):
        assert _binomial_two_sided(k, 20) == pytest.approx(_binomial_two_sided(20 - k, 20))


def test_mcnemar_ignores_agreements():
    """Only disagreements carry information; padding with agreements must not
    move the p-value."""
    a = _result("a", [1, 1, 1, 0, 0, 0])
    b = _result("b", [1, 1, 0, 1, 0, 0])
    small = mcnemar(a, b)
    pad = 40
    a2 = _result("a", [1, 1, 1, 0, 0, 0] + [1] * pad)
    b2 = _result("b", [1, 1, 0, 1, 0, 0] + [1] * pad)
    assert mcnemar(a2, b2)["p_value"] == pytest.approx(small["p_value"])
    assert small["a_right_b_wrong"] == 1 and small["b_right_a_wrong"] == 1


def test_mcnemar_counts_partition_the_positions():
    rng = np.random.default_rng(0)
    a = _result("a", rng.random(200) < 0.9)
    b = _result("b", rng.random(200) < 0.8)
    m = mcnemar(a, b)
    assert (m["both_right"] + m["both_wrong"] + m["a_right_b_wrong"] + m["b_right_a_wrong"]
            == m["positions"] == 200)


def test_mcnemar_only_scores_won_positions():
    """A lost position has no move to get right, so it must not be counted."""
    a = _result("a", [1, 1, 0, 0], won=[True, True, False, False])
    b = _result("b", [0, 0, 1, 1], won=[True, True, False, False])
    m = mcnemar(a, b)
    assert m["positions"] == 2
    assert m["a_right_b_wrong"] == 2 and m["b_right_a_wrong"] == 0


def test_mcnemar_can_restrict_to_a_ply_range():
    a = _result("a", [1, 1, 0, 0], plies=[4, 4, 10, 10])
    b = _result("b", [0, 0, 1, 1], plies=[4, 4, 10, 10])
    assert mcnemar(a, b, plies=(4,))["a_right_b_wrong"] == 2
    assert mcnemar(a, b, plies=(10,))["a_right_b_wrong"] == 0


def test_identical_agents_are_never_significant():
    rng = np.random.default_rng(1)
    correct = rng.random(500) < 0.95
    a, b = _result("a", correct), _result("b", correct.copy())
    m = mcnemar(a, b)
    assert m["a_right_b_wrong"] == m["b_right_a_wrong"] == 0
    assert m["p_value"] == 1.0


def test_a_clearly_better_agent_is_significant():
    a = _result("a", [1] * 100)
    b = _result("b", [1] * 85 + [0] * 15)
    assert mcnemar(a, b)["p_value"] < 0.001


def test_bootstrap_interval_brackets_the_observed_difference():
    rng = np.random.default_rng(2)
    a = _result("a", rng.random(400) < 0.97)
    b = _result("b", rng.random(400) < 0.90)
    diff, low, high = paired_difference_ci(a, b, resamples=4000)
    assert low <= diff <= high
    assert diff > 0
    assert high - low < 0.15


def test_bootstrap_interval_of_a_tie_straddles_zero():
    rng = np.random.default_rng(3)
    correct = rng.random(400) < 0.95
    a, b = _result("a", correct), _result("b", correct.copy())
    diff, low, high = paired_difference_ci(a, b, resamples=4000)
    assert low <= 0.0 <= high
