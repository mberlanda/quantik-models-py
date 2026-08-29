"""Generated puzzles, and the properties that make them puzzles.

A puzzle here is a solved position plus the claim that a particular move is
the answer. Both halves come from the exact corpus, so the risk is not that
the solver is wrong — it is that the *selection* is. A position with two
winning moves offered as "find the only move", or a "double threat" whose
threat the opponent can simply block, is a wrong answer stated with
authority, and a player has no way to tell.

So these tests check the classification rather than the search: given a
position whose properties are known, does each theme accept it for the
right reason and reject it for the right one.

Torch-free — this is `fastboard` and the corpus arrays, no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from quantik_models.data.exact_corpus import ExactCorpus
from quantik_models.env import fastboard as fb
from quantik_models.play import puzzles

FIXTURE = Path(__file__).parent / "fixtures" / "puzzle-corpus.json"


@pytest.fixture
def corpus():
    """Seventy real solver rows, committed.

    Not the corpus in `runs/`: that is 11 MB and gitignored, so a test
    depending on it passes on this machine and errors everywhere else.
    These are genuine `exact_oracle` values and optimal masks, extracted
    once and small enough to keep.
    """
    payload = json.loads(FIXTURE.read_text())["positions"]
    return ExactCorpus(
        boards=np.stack([np.asarray(fb.from_qfen(r["qfen"])).reshape(-1) for r in payload]),
        optimal_mask=np.array([int(r["optimal_mask"]) for r in payload], dtype=np.uint64),
        value_target=np.array([r["value"] for r in payload], dtype=np.float32),
        plies=np.array([r["ply"] for r in payload], dtype=np.int16),
    )


def board_of(qfen: str) -> np.ndarray:
    return np.asarray(fb.from_qfen(qfen)).reshape(-1)


def test_an_immediate_win_is_a_move_that_ends_the_game():
    """Both of Quantik's terminal conditions are losses for the side to
    move, so the last mover always wins and "ends the game" is exactly
    "wins" — no separate line check is needed, and adding one would miss
    the win by suffocation."""
    # Three shapes in the top row, the fourth cell open.
    wins = puzzles.immediate_wins(board_of("AbC./..../..../...."))
    assert wins.size > 0
    for action in wins:
        after = fb.apply_actions(
            board_of("AbC./..../..../....")[None, :], np.array([action], dtype=np.int64)
        )
        done, _ = fb.terminal_status(after)
        assert bool(done[0])
    # Every one of them has to be legal in the first place.
    legal = fb.legal_masks(board_of("AbC./..../..../....")[None, :])[0]
    assert all(legal[a] for a in wins)


def test_a_quiet_position_has_no_immediate_win():
    assert puzzles.immediate_wins(board_of("...a/..../..../CdB.")).size == 0


def test_a_double_threat_is_one_the_opponent_cannot_answer():
    """The theme the name promises: after the key move, *every* reply
    loses, and the win does not always land on the same square — otherwise
    it is a single threat that happens to be unstoppable for some other
    reason, and calling it a double threat would be a lie."""
    board = board_of("..../AD.D/..da/B.d.")
    threat = puzzles.describe_double_threat(board, 2 * 16 + 13)  # C@13
    assert threat is not None
    assert threat["replies"] > 1
    assert len(threat["winning_squares"]) >= 2


def test_a_position_already_winning_on_the_spot_is_not_a_double_threat():
    """A move that just wins is a mate in one. Classifying it as a fork
    would put a trivial position under a theme that promises a hard one."""
    board = board_of("AbC./..../..../....")
    wins = puzzles.immediate_wins(board)
    assert puzzles.describe_double_threat(board, int(wins[0])) is None


def test_every_generated_puzzle_has_a_legal_solution_in_a_playable_position(corpus):
    pack = puzzles.generate(corpus, per_theme=3, seed=7)
    assert pack["puzzles"], "generated nothing"
    for puzzle in pack["puzzles"]:
        board = board_of(puzzle["qfen"])
        done, _ = fb.terminal_status(board[None, :])
        assert not bool(done[0]), f"{puzzle['qfen']} is already over"
        legal = fb.legal_masks(board[None, :])[0]
        # `already-lost` is the one theme with no answer, and that is the
        # point of it: every legal move loses, so naming one as the
        # solution would invent a distinction the game does not have.
        if puzzle["theme"] == "already-lost":
            assert puzzle["solutions"] == []
        else:
            assert puzzle["solutions"], f"{puzzle['qfen']} has no solution"
        for action in puzzle["solutions"]:
            assert legal[action], f"{puzzle['qfen']}: solution {action} is illegal"
        assert puzzle["side_to_move"] == int(
            fb.popcount(fb.occupancy(board[None, :]))[0] % 2
        )


def test_only_move_puzzles_really_have_only_one(corpus):
    pack = puzzles.generate(corpus, per_theme=4, seed=7)
    for puzzle in pack["puzzles"]:
        if puzzle["theme"] not in ("only-move", "double-threat", "endgame"):
            continue
        assert len(puzzle["solutions"]) == 1, f"{puzzle['qfen']} has several answers"


def test_no_two_puzzles_are_the_same_position_up_to_symmetry(corpus):
    """Quantik has 192 symmetries. Two puzzles that are reflections of each
    other are one puzzle shown twice, and a pack that repeats itself is
    what makes a generated set feel worse than a hand-picked one."""
    pack = puzzles.generate(corpus, per_theme=6, seed=7)
    keys = [
        int(fb.canonical_keys(board_of(p["qfen"])[None, :])[0]) for p in pack["puzzles"]
    ]
    assert len(keys) == len(set(keys))


def test_the_pack_records_what_produced_it(corpus):
    """A pack whose provenance is not written down cannot be regenerated,
    and a puzzle nobody can regenerate cannot be corrected."""
    pack = puzzles.generate(corpus, per_theme=2, seed=7)
    assert pack["schema"] == puzzles.SCHEMA
    assert pack["seed"] == 7
    assert pack["counts"]
