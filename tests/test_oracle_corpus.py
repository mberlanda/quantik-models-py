"""Cross-check the Rust exact oracle against Python's solver.

The oracle labels are the ground truth every later claim rests on, and they
come from a different language and a different codebase than the rest of this
repo. `quantik_core.minimax.MinimaxEngine.solve` is an independent exact
solver, so agreeing with it on sampled rows is a real cross-implementation
check rather than a self-consistency check.

Skipped when the corpus has not been generated (see
`scripts/build_oracle_corpus.py`).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from quantik_core import State
from quantik_core.game_utils import has_winning_line
from quantik_core.minimax import MinimaxConfig, MinimaxEngine
from quantik_core.move import generate_legal_moves_list

from quantik_models.env import fastboard as fb

CORPUS = Path("runs/oracle/corpus")
DEEP_PLIES = ("ply11.jsonl", "ply12.jsonl", "ply10.jsonl")


def _rows(limit: int) -> list[dict]:
    """Oracle rows from the deepest available files — cheap for Python to
    re-solve, so the cross-check stays fast."""
    out: list[dict] = []
    for name in DEEP_PLIES:
        path = CORPUS / name
        if not path.exists() or path.stat().st_size == 0:
            continue
        with path.open() as handle:
            for line in handle:
                out.append(json.loads(line))
                if len(out) >= limit:
                    return out
    return out


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    found = _rows(40)
    if not found:
        pytest.skip("oracle corpus not generated")
    return found


def _python_action_values(board: np.ndarray) -> dict[int, float]:
    """Exact value of every legal move, from the mover's perspective."""
    tup = tuple(int(v) for v in board)
    values: dict[int, float] = {}
    for move in generate_legal_moves_list(tup):
        child = fb.apply_actions(
            board[None, :], np.array([move.shape * 16 + move.position], dtype=np.int64)
        )[0]
        child_tuple = tuple(int(v) for v in child)
        if has_winning_line(child_tuple) or not generate_legal_moves_list(child_tuple):
            value = 10_000.0  # the opponent is left dead: the mover wins
        else:
            value = -MinimaxEngine(
                MinimaxConfig(max_depth=16, time_limit_s=None)
            ).solve(State(child_tuple)).score
        values[move.shape * 16 + move.position] = value
    return values


def test_oracle_agrees_with_python_on_who_wins(rows):
    for record in rows:
        board = fb.from_qfen(record["qfen"])[0]
        values = _python_action_values(board)
        assert bool(record["won"]) == (max(values.values()) > 0), record["qfen"]


def test_oracle_outcome_optimal_matches_python(rows):
    for record in rows:
        board = fb.from_qfen(record["qfen"])[0]
        values = _python_action_values(board)
        won = max(values.values()) > 0
        expected = {a for a, v in values.items() if (v > 0) == won}
        assert set(record["outcome_optimal"]) == expected, record["qfen"]


def test_oracle_covers_exactly_the_legal_actions(rows):
    for record in rows:
        board = fb.from_qfen(record["qfen"])[0]
        legal = set(np.flatnonzero(fb.legal_masks(board[None, :])[0]).tolist())
        assert {int(a) for a in record["action_values"]} == legal, record["qfen"]


def test_corpus_child_value_labels_are_consistent():
    """Every child row's value must be the negation of its move's value."""
    npz = CORPUS / "exact-deep.npz"
    if not npz.exists():
        pytest.skip("corpus npz not built")
    rows_ = _rows(12)
    if not rows_:
        pytest.skip("oracle corpus not generated")
    with np.load(npz) as data:
        keys = fb.canonical_keys(data["boards"])
        lookup = dict(zip(keys.tolist(), data["value_target"].tolist()))
    checked = 0
    for record in rows_:
        parent = fb.from_qfen(record["qfen"])
        for action, value in record["action_values"].items():
            child = fb.apply_actions(parent, np.array([int(action)], dtype=np.int64))
            key = int(fb.canonical_keys(child)[0])
            if key not in lookup:
                continue
            done, _ = fb.terminal_status(child)
            expected = -1.0 if bool(done[0]) else (1.0 if -float(value) > 0 else -1.0)
            assert lookup[key] == expected
            checked += 1
    assert checked, "no child rows were found in the corpus"


def test_corpus_builder_excludes_held_out_positions_reached_as_children():
    """A held-out position can arrive as somebody's child even though it was
    never sampled; the exclusion must catch that, not just the parents."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.build_oracle_corpus import rows_from_oracle

    paths = [p for p in DEEP_PLIES if (CORPUS / p).exists() and (CORPUS / p).stat().st_size]
    if not paths:
        pytest.skip("oracle corpus not generated")
    path = CORPUS / paths[0]
    everything = rows_from_oracle([path])
    keys = everything.canonical_keys()
    # Hold out a slice that includes value-only child rows.
    held = set(keys[everything.optimal_mask == 0][:50].tolist())
    if not held:
        pytest.skip("no value-only rows in this slice")
    filtered = rows_from_oracle([path], exclude=held)
    assert not (set(filtered.canonical_keys().tolist()) & held)
    assert len(filtered) == len(everything) - len(held)
