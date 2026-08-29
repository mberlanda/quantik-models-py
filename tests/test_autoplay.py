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


def test_pairings_against_an_oracle_keeps_only_that_agents_games() -> None:
    """Measuring a field against one oracle is not a round robin.

    With four networks and one oracle, 12 of the 20 ordered pairings are
    network-versus-network and already measured; running them again to
    extract the oracle's eight spends most of the budget on nothing.
    """
    names = ["cpool", "attn", "resnet", "mlp", "minimax-d2"]
    schedule = autoplay.pairings(names, against="minimax-d2")
    assert len(schedule) == 8
    assert all("minimax-d2" in pair for pair in schedule)
    # Both seats, for every opponent: the oracle moves first four times and
    # second four times, or the first-move advantage lands on it.
    assert sum(1 for mover, _ in schedule if mover == "minimax-d2") == 4


def test_pairings_without_an_oracle_is_the_full_round_robin() -> None:
    assert len(autoplay.pairings(["a", "b", "c"])) == 6


def test_pairings_refuses_an_unknown_opponent() -> None:
    """A typo would otherwise play zero games and report a clean empty run."""
    with pytest.raises(ValueError, match="no agent named"):
        autoplay.pairings(["a", "b"], against="minmax")


def test_run_against_restricts_the_schedule() -> None:
    specs = [
        {"kind": "random", "name": "r1"},
        {"kind": "random", "name": "r2"},
        {"kind": "random", "name": "r3"},
    ]
    games = autoplay.run(specs, games_per_pairing=2, seed=0, against="r3")
    assert len(games) == 4 * 2
    assert all("r3" in (g.mover, g.responder) for g in games)
