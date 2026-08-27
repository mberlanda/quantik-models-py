"""Correctness of the batched MCTS, judged against the exact solver.

`quantik_core.minimax.MinimaxEngine.solve` is a true depth-16 solver, so on
late-game positions (where it returns in well under a second) it is the
ground truth for what the search should find.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from quantik_core import State
from quantik_core.game_utils import has_winning_line as core_has_winning_line
from quantik_core.minimax import MinimaxConfig, MinimaxEngine
from quantik_core.move import apply_move, generate_legal_moves_list

from quantik_models.env import fastboard as fb
from quantik_models.selfplay import BatchedMCTS, MCTSParams, UniformEvaluator


def _live_positions(count: int, plies: int, seed: int) -> np.ndarray:
    """Non-terminal positions reached by `plies` random moves."""
    rng = random.Random(seed)
    out = []
    while len(out) < count:
        bb: tuple[int, ...] = (0,) * 8
        ok = True
        for _ in range(plies):
            moves = generate_legal_moves_list(bb)
            if not moves or core_has_winning_line(bb):
                ok = False
                break
            bb = apply_move(bb, rng.choice(moves))
        if ok and not core_has_winning_line(bb) and generate_legal_moves_list(bb):
            out.append(bb)
    return np.array(out, dtype=np.uint16)


def _child_score(board: np.ndarray, action: int) -> float:
    """Exact score of the position after `action`, from that side's view."""
    child = fb.apply_actions(board[None, :], np.array([action], dtype=np.int64))[0]
    tup = tuple(int(v) for v in child)
    if core_has_winning_line(tup) or not generate_legal_moves_list(tup):
        return -10000.0
    return MinimaxEngine(MinimaxConfig(max_depth=16, time_limit_s=None)).solve(State(tup)).score


def _solve(board: np.ndarray) -> tuple[float, set[int]]:
    """Exact `(score, outcome-optimal action indices)` for one board.

    "Outcome-optimal" means the move keeps the best achievable *result* —
    not necessarily the fastest mate. A search whose values are just
    win/loss cannot rank mate distance, so that is the honest bar.
    """
    tup = tuple(int(v) for v in board)
    best = MinimaxEngine(MinimaxConfig(max_depth=16, time_limit_s=None)).solve(State(tup)).score
    optimal = set()
    for move in generate_legal_moves_list(tup):
        action = move.shape * 16 + move.position
        value = -_child_score(board, action)
        if (value > 0) == (best > 0):
            optimal.add(action)
    return best, optimal


def test_search_only_visits_legal_actions():
    rng = np.random.default_rng(1)
    boards = _live_positions(16, plies=5, seed=3)
    visits, _ = BatchedMCTS(UniformEvaluator(), MCTSParams(simulations=48), rng).search(boards)
    legal = fb.legal_masks(boards)
    assert np.all(visits[~legal] == 0.0)
    assert np.all(visits.sum(axis=1) == 48)


def test_visits_are_conserved_without_noise():
    rng = np.random.default_rng(2)
    boards = _live_positions(8, plies=9, seed=11)
    for sims in (16, 64, 200):
        visits, _ = BatchedMCTS(
            UniformEvaluator(), MCTSParams(simulations=sims), rng
        ).search(boards, add_noise=False)
        assert np.all(visits.sum(axis=1) == sims)


def test_finds_the_immediate_win():
    """Wherever a legal move completes a line, search must take one."""
    boards = _live_positions(40, plies=9, seed=5)
    legal = fb.legal_masks(boards)
    winning = []
    keep = []
    for i in range(boards.shape[0]):
        actions = np.flatnonzero(legal[i])
        wins = {
            int(a)
            for a in actions
            if fb.has_winning_line(fb.apply_actions(boards[i : i + 1], np.array([a])))[0]
        }
        if wins:
            winning.append(wins)
            keep.append(i)
    assert keep, "expected some ply-9 positions to have an immediate win"
    subset = boards[np.array(keep)]
    rng = np.random.default_rng(4)
    visits, values = BatchedMCTS(
        UniformEvaluator(), MCTSParams(simulations=192), rng
    ).search(subset, add_noise=False)
    for k, wins in enumerate(winning):
        assert int(visits[k].argmax()) in wins, f"missed the win at {fb.to_qfen(subset[k])}"
        assert values[k] > 0.0


@pytest.mark.parametrize("plies", [9, 10, 11])
def test_never_throws_away_a_won_position(plies):
    """From a provably won root, 512 simulations must keep the win."""
    boards = _live_positions(24, plies=plies, seed=17 + plies)
    rng = np.random.default_rng(9)
    visits, _ = BatchedMCTS(
        UniformEvaluator(), MCTSParams(simulations=512), rng
    ).search(boards, add_noise=False)
    checked = 0
    for i in range(boards.shape[0]):
        score, optimal = _solve(boards[i])
        if score <= 0:
            continue  # lost root: every move loses, nothing to preserve
        checked += 1
        assert int(visits[i].argmax()) in optimal, (
            f"threw away a won position at {fb.to_qfen(boards[i])}"
        )
    assert checked, f"no won roots sampled at ply {plies}"


def test_root_value_sign_is_mover_relative():
    """A position whose every reply loses must score near -1 for the mover."""
    boards = _live_positions(60, plies=11, seed=23)
    rng = np.random.default_rng(6)
    _, values = BatchedMCTS(
        UniformEvaluator(), MCTSParams(simulations=256), rng
    ).search(boards, add_noise=False)
    exact = np.array([_solve(boards[i])[0] for i in range(boards.shape[0])])
    lost = exact <= -9000
    won = exact >= 9000
    if lost.any():
        assert values[lost].mean() < 0.0
    if won.any():
        assert values[won].mean() > 0.0
    assert values[won].mean() > values[lost].mean() if lost.any() and won.any() else True


@pytest.mark.parametrize("leaf_batch", [4, 16, 64])
def test_leaf_batching_conserves_visits_and_legality(leaf_batch):
    """Virtual loss must leave the edge statistics exactly as clean as
    one-leaf-at-a-time search: `simulations` visits, none illegal."""
    boards = _live_positions(12, plies=6, seed=31)
    rng = np.random.default_rng(2)
    visits, _ = BatchedMCTS(
        UniformEvaluator(),
        MCTSParams(simulations=256, leaf_batch=leaf_batch),
        rng,
    ).search(boards, add_noise=False)
    assert np.all(visits.sum(axis=1) == 256)
    assert np.all(visits[~fb.legal_masks(boards)] == 0.0)


def test_leaf_batching_does_not_degrade_play():
    """A batched search must still keep every provably won root."""
    boards = _live_positions(30, plies=10, seed=41)
    rng = np.random.default_rng(3)
    visits, _ = BatchedMCTS(
        UniformEvaluator(),
        MCTSParams(simulations=512, leaf_batch=32),
        rng,
    ).search(boards, add_noise=False)
    checked = 0
    for i in range(boards.shape[0]):
        score, optimal = _solve(boards[i])
        if score <= 0:
            continue
        checked += 1
        assert int(visits[i].argmax()) in optimal
    assert checked
