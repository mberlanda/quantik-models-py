"""Autoplay is a position generator, so the tests are about reach.

Its output is not labels — the solver provides those. What it has to do is
visit positions the corpus does not have, deduplicate them the way the
solver's cost demands, and never produce an illegal game.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from quantik_models.arena import autoplay
from quantik_models.env import fastboard as fb


class _FirstLegal:
    """A deterministic agent with no torch dependency."""

    def __init__(self, name: str, offset: int = 0) -> None:
        self.name = name
        self.offset = offset

    def select(self, board, seed):
        legal = np.flatnonzero(fb.legal_masks(board[None, :])[0])
        return int(legal[self.offset % len(legal)])


def test_a_recorded_game_is_a_legal_sequence() -> None:
    game = autoplay.play_recorded(
        _FirstLegal("a"), _FirstLegal("b", 1), fb.empty_boards(1)[0], seed=0
    )
    assert game.winner in (0, 1)
    assert len(game.boards) == len(game.actions) == game.plies

    # Replaying the recorded actions has to reproduce the recorded boards,
    # or the trajectory does not describe the game that was played.
    current = fb.empty_boards(1)[0]
    for board, action in zip(game.boards, game.actions):
        assert np.array_equal(board, current)
        assert fb.legal_masks(current[None, :])[0][action]
        current = fb.apply_actions(
            current[None, :], np.array([action], dtype=np.int64)
        )[0]


def test_the_game_ends_terminal() -> None:
    game = autoplay.play_recorded(
        _FirstLegal("a"), _FirstLegal("b", 2), fb.empty_boards(1)[0], seed=0
    )
    current = fb.empty_boards(1)[0]
    for action in game.actions:
        current = fb.apply_actions(
            current[None, :], np.array([action], dtype=np.int64)
        )[0]
    done, _ = fb.terminal_status(current[None, :])
    assert bool(done[0])


def test_positions_deduplicate_on_the_canonical_key() -> None:
    """Two games reaching symmetric positions are one position to solve."""
    board = fb.empty_boards(1)[0]
    game = autoplay.play_recorded(_FirstLegal("a"), _FirstLegal("b", 1), board, seed=0)

    rng = np.random.default_rng(0)
    stacked = np.stack(game.boards)
    spatial, shape = fb.random_symmetries(len(stacked), rng)
    mirrored = autoplay.Game(
        mover="a",
        responder="b",
        boards=list(fb.transform_boards(stacked, spatial, shape)),
        actions=game.actions,
        winner=game.winner,
    )
    both = autoplay.positions_from([game, mirrored])
    assert len(both) == len(autoplay.positions_from([game]))


def test_max_ply_filters_to_the_shallow_end() -> None:
    game = autoplay.play_recorded(
        _FirstLegal("a"), _FirstLegal("b", 1), fb.empty_boards(1)[0], seed=0
    )
    shallow = autoplay.positions_from([game], max_ply=4)
    assert len(shallow) <= len(autoplay.positions_from([game]))
    assert (fb.popcount(fb.occupancy(shallow)) <= 4).all()


def test_novel_positions_drops_what_the_corpus_covers() -> None:
    game = autoplay.play_recorded(
        _FirstLegal("a"), _FirstLegal("b", 1), fb.empty_boards(1)[0], seed=0
    )
    visited = autoplay.positions_from([game])
    assert len(autoplay.novel_positions(visited, visited)) == 0

    # A symmetric image of a corpus position is not novel either.
    rng = np.random.default_rng(1)
    spatial, shape = fb.random_symmetries(len(visited), rng)
    assert len(
        autoplay.novel_positions(visited, fb.transform_boards(visited, spatial, shape))
    ) == 0


def test_leaderboard_counts_both_seats() -> None:
    games = [
        autoplay.Game(mover="a", responder="b", winner=0),
        autoplay.Game(mover="b", responder="a", winner=0),
    ]
    board = {row["agent"]: row for row in autoplay.leaderboard(games)}
    assert board["a"]["games"] == board["b"]["games"] == 2
    assert board["a"]["wins"] == board["b"]["wins"] == 1


def test_qfens_round_trip(tmp_path) -> None:
    """The solver reads these; a lossy write would mislabel everything."""
    game = autoplay.play_recorded(
        _FirstLegal("a"), _FirstLegal("b", 1), fb.empty_boards(1)[0], seed=0
    )
    boards = autoplay.positions_from([game])
    path = autoplay.write_qfens(boards, tmp_path / "to-solve.qfen")
    lines = path.read_text().split()
    assert len(lines) == len(boards)
    restored = np.concatenate([fb.from_qfen(line) for line in lines])
    assert np.array_equal(restored, boards)


def test_run_plays_every_ordered_pairing() -> None:
    """Ordered: moving first is a real advantage and must not be an agent's."""
    specs = [{"kind": "random", "name": "r1"}, {"kind": "random", "name": "r2"}]
    games = autoplay.run(specs, games_per_pairing=3, seed=0)
    assert len(games) == 2 * 1 * 3
    seats = {(g.mover, g.responder) for g in games}
    assert seats == {("r1", "r2"), ("r2", "r1")}
