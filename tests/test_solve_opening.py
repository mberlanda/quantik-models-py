"""Backward induction must reproduce the exact solver.

`scripts/solve_opening.py` derives values and optimal-move sets for a whole
ply from the ply below it. That inference is the load-bearing step of the
opening solve — if it is wrong, every opening label is wrong — so it is
checked here against `quantik_core`'s independent solver at deep plies, where
Python can solve fast enough to be the reference.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantik_core import State
from quantik_core.game_utils import has_winning_line
from quantik_core.minimax import MinimaxConfig, MinimaxEngine
from quantik_core.move import generate_legal_moves_list

from quantik_models.env import fastboard as fb
from scripts.solve_opening import induct


def _live_positions(count: int, plies: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    boards = fb.empty_boards(count * 8)
    for _ in range(plies):
        done, _ = fb.terminal_status(boards)
        boards = boards[~done]
        legal = fb.legal_masks(boards)
        boards = fb.apply_actions(boards, (rng.random(legal.shape) * legal).argmax(axis=1))
    done, _ = fb.terminal_status(boards)
    boards = boards[~done]
    keys = fb.canonical_keys(boards)
    _, first = np.unique(keys, return_index=True)
    return boards[first][:count]


def _python_won(board: np.ndarray) -> bool:
    tup = tuple(int(v) for v in board)
    if has_winning_line(tup) or not generate_legal_moves_list(tup):
        return False
    return MinimaxEngine(MinimaxConfig(max_depth=16, time_limit_s=None)).solve(State(tup)).score > 0


def _child_level(boards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solved `(sorted canonical keys, won)` for every live child of `boards`."""
    legal = fb.legal_masks(boards)
    rows, actions = np.nonzero(legal)
    children = fb.apply_actions(boards[rows], actions)
    done, _ = fb.terminal_status(children)
    live = children[~done]
    keys = fb.canonical_keys(live)
    _, first = np.unique(keys, return_index=True)
    unique = live[first]
    won = np.array([_python_won(b) for b in unique])
    keys = fb.canonical_keys(unique)
    order = np.argsort(keys)
    return keys[order], won[order]


@pytest.mark.parametrize("plies", [11, 12])
def test_induction_matches_the_exact_solver(plies):
    boards = _live_positions(12, plies=plies, seed=900 + plies)
    child_keys, child_won = _child_level(boards)
    won, mask = induct(boards, child_keys, child_won)

    expected = np.array([_python_won(b) for b in boards])
    assert np.array_equal(won, expected), "induced values disagree with the solver"

    # Every move in the induced optimal set must actually preserve the outcome.
    for i, board in enumerate(boards):
        actions = [a for a in range(64) if (int(mask[i]) >> a) & 1]
        assert actions, "an optimal set must never be empty"
        legal = fb.legal_masks(board[None, :])[0]
        assert all(legal[a] for a in actions), "an illegal move was marked optimal"
        for action in actions:
            child = fb.apply_actions(board[None, :], np.array([action], dtype=np.int64))[0]
            done, _ = fb.terminal_status(child[None, :])
            child_wins = False if done[0] else _python_won(child)
            # The move is good for the mover exactly when the child loses.
            assert (not child_wins) == bool(won[i])


def test_induction_marks_an_immediate_win_optimal():
    """Where a move ends the game on the spot, it must be in the optimal set."""
    boards = _live_positions(40, plies=11, seed=77)
    child_keys, child_won = _child_level(boards)
    won, mask = induct(boards, child_keys, child_won)
    checked = 0
    for i, board in enumerate(boards):
        legal = np.flatnonzero(fb.legal_masks(board[None, :])[0])
        for action in legal:
            child = fb.apply_actions(board[None, :], np.array([action], dtype=np.int64))
            if not fb.terminal_status(child)[0][0]:
                continue
            checked += 1
            assert bool(won[i])
            assert (int(mask[i]) >> int(action)) & 1, "an immediate win was left out"
    assert checked, "no immediate wins were sampled"
