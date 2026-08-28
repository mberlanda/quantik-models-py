"""Occupancy analysis, and the pairing it depends on.

The comparison is between two models answering the *same* positions, so
only their disagreements carry information. An unpaired test throws that
away; McNemar keeps it, and these tests pin down both the occupancy
measure and the test itself, since a wrong p-value here would be believed.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantik_models.env import fastboard as fb
from quantik_models.eval import shift

from boards import random_positions


def test_empty_board_touches_no_group() -> None:
    assert shift.group_occupancy(fb.empty_boards(3)).tolist() == [0, 0, 0]


def test_one_piece_touches_exactly_three_groups() -> None:
    """Every cell is in exactly one row, one column and one zone."""
    board = fb.apply_actions(fb.empty_boards(1), np.array([0], dtype=np.int64))
    assert shift.group_occupancy(board).tolist() == [3]


def test_occupancy_is_bounded_by_twelve() -> None:
    boards = random_positions(64, seed=4, plies=8)
    occupancy = shift.group_occupancy(boards)
    assert (occupancy >= 0).all() and (occupancy <= 12).all()


def test_occupancy_never_decreases_as_pieces_are_added() -> None:
    board = fb.empty_boards(1)
    previous = 0
    for _ in range(6):
        legal = np.flatnonzero(fb.legal_masks(board)[0])
        if not legal.size:
            break
        board = fb.apply_actions(board, np.array([legal[0]], dtype=np.int64))
        current = int(shift.group_occupancy(board)[0])
        assert current >= previous
        previous = current


def test_mcnemar_is_symmetric_and_one_at_parity() -> None:
    assert shift.mcnemar_exact(0, 0) == 1.0
    assert shift.mcnemar_exact(10, 10) == 1.0
    assert shift.mcnemar_exact(3, 9) == pytest.approx(shift.mcnemar_exact(9, 3))


def test_mcnemar_detects_a_lopsided_split() -> None:
    assert shift.mcnemar_exact(10, 33) < 0.01
    assert shift.mcnemar_exact(40, 44) > 0.5


def test_mcnemar_matches_a_known_binomial() -> None:
    """1 of 10 discordant pairs: two-sided binomial p = 2 * (1 + 10) / 1024."""
    assert shift.mcnemar_exact(1, 9) == pytest.approx(22 / 1024)


def _probe(tmp_path, n=200):
    import json

    boards = random_positions(n, seed=8, plies=6)
    legal = fb.legal_masks(boards)
    rows = []
    for i, board in enumerate(boards):
        actions = np.flatnonzero(legal[i]).tolist()
        rows.append(
            {
                "qfen": fb.to_qfen(board),
                "won": True,
                "outcome_optimal": actions[: max(1, len(actions) // 2)],
            }
        )
    path = tmp_path / "probe.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return shift.load_probe(path)


def test_split_is_within_ply(tmp_path) -> None:
    """Occupancy correlates with ply, so an unconditional split re-measures ply."""
    probe = _probe(tmp_path)
    rng = np.random.default_rng(0)
    a = rng.random(len(probe)) < 0.6
    b = rng.random(len(probe)) < 0.6
    rows = shift.occupancy_split(probe, a, b, plies=(6,), min_positions=1)
    assert rows and {row["ply"] for row in rows} == {6}
    assert len({row["bucket"] for row in rows}) <= 2


def test_identical_models_show_no_difference(tmp_path) -> None:
    probe = _probe(tmp_path)
    rng = np.random.default_rng(1)
    same = rng.random(len(probe)) < 0.7
    for row in shift.occupancy_split(probe, same, same, plies=(6,), min_positions=1):
        assert row["difference"] == 0.0
        assert row["a_only"] == row["b_only"] == 0
        assert row["p_value"] == 1.0


def test_a_strictly_better_model_is_reported_as_better(tmp_path) -> None:
    probe = _probe(tmp_path)
    weak = np.zeros(len(probe), dtype=bool)
    strong = np.ones(len(probe), dtype=bool)
    rows = shift.occupancy_split(probe, strong, weak, plies=(6,), min_positions=1)
    assert rows
    for row in rows:
        assert row["difference"] == 1.0
        assert row["b_only"] == 0
        assert row["p_value"] < 0.01
