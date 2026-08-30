"""The exporter's one real trap: two canonical-key representations that do
not match without conversion.

`game_positions.canonical_key` is a decimal string (`play/record.py`);
`ExactCorpus` gives a `uint64` array. Comparing them without converting
finds no overlap, the filter drops nothing, and the queue silently contains
positions the corpus already has — the failure looks exactly like success.
`test_the_filter_drops_positions_the_corpus_already_has` is the failing
test the brief asked for first: a filter that is a no-op would pass a test
that only checks "some rows came out", so it asserts the drop count and the
written content both.
"""

from __future__ import annotations

import gzip
import json

import numpy as np

from quantik_models.data.exact_corpus import ExactCorpus
from quantik_models.env import fastboard as fb
from quantik_models.play import export, store

from boards import random_positions


def _game_and_positions(game_id: str, boards: np.ndarray) -> tuple[dict, dict, list[dict]]:
    """A minimal valid game whose `game_positions` rows are `boards`, ply 0..n-1."""
    game = {
        "schema": "game-result.v1",
        "contract_version": "1.2.0",
        "game_id": game_id,
        "started_at": "2026-08-30T00:00:00Z",
        "p0_engine_kind": "human",
        "p0_engine_version": "n/a",
        "p1_engine_kind": "policy_value_net",
        "p1_engine_version": "resnet-v3",
        "initial_position_key": "0",
        "winner": 0,
        "plies": len(boards),
        "terminal_reason": "win_condition",
        "move_action_indices": list(range(len(boards))),
        "run_id": None,
    }
    meta = {"game_id": game_id, "recorded_at": "2026-08-30T00:00:01Z"}
    positions = [
        {
            "ply": ply,
            "qfen": fb.to_qfen(board),
            "canonical_key": str(int(fb.canonical_keys(board[None, :])[0])),
        }
        for ply, board in enumerate(boards)
    ]
    return game, meta, positions


def _corpus_of(boards: np.ndarray) -> ExactCorpus:
    n = len(boards)
    return ExactCorpus(
        boards=boards,
        optimal_mask=np.zeros(n, dtype=np.uint64),
        value_target=np.zeros(n, dtype=np.float32),
        plies=np.zeros(n, dtype=np.int16),
    )


def test_the_filter_drops_positions_the_corpus_already_has(tmp_path) -> None:
    boards = random_positions(2, seed=1, plies=3)  # two independent boards
    novel, known = boards[0:1], boards[1:2]

    conn = store.connect(tmp_path / "games.db")
    game, meta, positions = _game_and_positions("g1", np.concatenate([novel, known]))
    store.record_game(conn, game, meta, positions)
    conn.close()

    corpus_path = tmp_path / "corpus.npz"
    _corpus_of(known).save(corpus_path)

    summary = export.export_queue(
        tmp_path / "games.db", tmp_path / "out", corpus=corpus_path, max_ply=6
    )

    assert summary["positions_found"] == 2
    assert summary["positions_dropped_known"] == 1
    assert summary["positions_written"] == 1

    written = gzip.decompress((tmp_path / "out" / "to-solve.qfen.gz").read_bytes()).decode()
    lines = [line for line in written.splitlines() if line.strip()]
    assert lines == [fb.to_qfen(novel[0])]


def test_with_no_corpus_every_distinct_position_is_exported(tmp_path) -> None:
    boards = random_positions(2, seed=2, plies=3)
    conn = store.connect(tmp_path / "games.db")
    game, meta, positions = _game_and_positions("g1", boards)
    store.record_game(conn, game, meta, positions)
    conn.close()

    summary = export.export_queue(tmp_path / "games.db", tmp_path / "out", corpus=None)

    assert summary["filtered_against"] is None
    assert summary["positions_found"] == 2
    assert summary["positions_dropped_known"] == 0
    assert summary["positions_written"] == 2


def test_max_ply_excludes_deeper_positions(tmp_path) -> None:
    boards = random_positions(3, seed=3, plies=3)
    conn = store.connect(tmp_path / "games.db")
    game, meta, positions = _game_and_positions("g1", boards)
    store.record_game(conn, game, meta, positions)
    conn.close()

    summary = export.export_queue(tmp_path / "games.db", tmp_path / "out", max_ply=1)

    assert summary["positions_found"] == 2  # ply 0 and 1 only


def test_summary_json_is_written_beside_the_queue(tmp_path) -> None:
    boards = random_positions(1, seed=4, plies=2)
    conn = store.connect(tmp_path / "games.db")
    game, meta, positions = _game_and_positions("g1", boards)
    store.record_game(conn, game, meta, positions)
    conn.close()

    export.export_queue(tmp_path / "games.db", tmp_path / "out")

    on_disk = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert on_disk["positions_written"] == 1
    assert on_disk["source_db"] == str(tmp_path / "games.db")


def test_main_exits_nonzero_on_a_missing_database(tmp_path, capsys) -> None:
    code = export.main(["--db", str(tmp_path / "nope.db"), "--out", str(tmp_path / "out")])
    assert code == 1
    assert "no database" in capsys.readouterr().err


def test_main_exits_zero_on_an_empty_queue(tmp_path) -> None:
    """Every position already known is a success — a cron job must not page on it."""
    boards = random_positions(1, seed=5, plies=2)
    conn = store.connect(tmp_path / "games.db")
    game, meta, positions = _game_and_positions("g1", boards)
    store.record_game(conn, game, meta, positions)
    conn.close()

    corpus_path = tmp_path / "corpus.npz"
    _corpus_of(boards).save(corpus_path)

    code = export.main(
        [
            "--db", str(tmp_path / "games.db"),
            "--out", str(tmp_path / "out"),
            "--corpus", str(corpus_path),
        ]
    )
    assert code == 0
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["positions_written"] == 0


def test_the_exporter_never_writes_to_the_database(tmp_path) -> None:
    """Read-only against the store: the one irreplaceable artifact in this
    project must survive an export untouched."""
    db_path = tmp_path / "games.db"
    conn = store.connect(db_path)
    game, meta, positions = _game_and_positions("g1", random_positions(1, seed=6, plies=2))
    store.record_game(conn, game, meta, positions)
    conn.close()

    before = db_path.read_bytes()
    export.export_queue(db_path, tmp_path / "out")
    after = db_path.read_bytes()

    assert before == after
