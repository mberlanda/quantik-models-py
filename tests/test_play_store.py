"""The play store's job is to refuse bad data and stay idempotent.

Every constraint here guards a real failure mode: `winner` outside {0, 1}
would silently record a pasted-in terminal position as a loss for player 0,
`terminal_reason` outside the contract's two values would let the browser's
own vocabulary (`"line"`) leak past the request layer, and a duplicate
`game_id` from a retried POST would otherwise double-count a game in every
head-to-head number downstream.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from quantik_models.play import store


def make_game(**overrides) -> dict:
    game = {
        "schema": "game-result.v1",
        "contract_version": "1.2.0",
        "game_id": "g1",
        "started_at": "2026-08-29T00:00:00Z",
        "p0_engine_kind": "human",
        "p0_engine_version": "n/a",
        "p1_engine_kind": "policy_value_net",
        "p1_engine_version": "resnet-v3",
        "initial_position_key": "0" * 16,
        "winner": 0,
        "plies": 2,
        "terminal_reason": "win_condition",
        "move_action_indices": [0, 1],
        "run_id": None,
    }
    game.update(overrides)
    return game


def make_meta(**overrides) -> dict:
    meta = {
        "game_id": "g1",
        "recorded_at": "2026-08-29T00:00:01Z",
        "human_seat": 0,
        "player_name": "mauro",
        "opponent_seat": 1,
        "opponent_id": "net-a",
        "model_id": "resnet-v3",
        "simulations": 400,
        "agent_kind": "mcts",
        "weights_hash": "deadbeef",
        "manifest_model_id": "resnet-v3-manifest",
        "architecture": "resnet",
        "service_version": "0.1.0",
        "client_user_agent": "pytest",
        "initial_qfen": "initial-qfen",
        "final_qfen": "final-qfen",
        "client_winner": 0,
        "client_terminal_reason": "line",
    }
    meta.update(overrides)
    return meta


def make_positions(game_id: str = "g1", plies: int = 2, keys: list[str] | None = None) -> list[dict]:
    keys = keys or [f"key-{game_id}-{p}" for p in range(plies)]
    return [
        {"ply": p, "qfen": f"qfen-{game_id}-{p}", "canonical_key": keys[p]}
        for p in range(plies)
    ]


def test_a_game_round_trips_with_its_meta_and_positions(tmp_path: Path) -> None:
    conn = store.connect(tmp_path / "play.db")
    positions = make_positions()

    assert store.record_game(conn, make_game(), make_meta(), positions) is True

    game_row = conn.execute("SELECT * FROM games WHERE game_id = ?", ("g1",)).fetchone()
    assert game_row["schema"] == "game-result.v1"
    assert game_row["winner"] == 0
    assert game_row["plies"] == 2

    meta_row = conn.execute("SELECT * FROM game_meta WHERE game_id = ?", ("g1",)).fetchone()
    assert meta_row["player_name"] == "mauro"
    assert meta_row["opponent_id"] == "net-a"

    stored_positions = conn.execute(
        "SELECT ply, qfen, canonical_key FROM game_positions WHERE game_id = ? ORDER BY ply",
        ("g1",),
    ).fetchall()
    assert [dict(row) for row in stored_positions] == [
        {"ply": p["ply"], "qfen": p["qfen"], "canonical_key": p["canonical_key"]}
        for p in positions
    ]


def test_recording_the_same_game_twice_is_a_no_op(tmp_path: Path) -> None:
    conn = store.connect(tmp_path / "play.db")
    game, meta, positions = make_game(), make_meta(), make_positions()

    assert store.record_game(conn, game, meta, positions) is True
    assert store.record_game(conn, game, meta, positions) is False
    assert store.game_count(conn) == 1


def test_winner_outside_zero_or_one_is_refused() -> None:
    conn = store.connect(Path(":memory:"))
    with pytest.raises(Exception, match="winner"):
        store.record_game(conn, make_game(winner=2), make_meta(), make_positions())


def test_zero_plies_is_refused() -> None:
    conn = store.connect(Path(":memory:"))
    with pytest.raises(Exception, match="plies"):
        store.record_game(conn, make_game(plies=0), make_meta(), make_positions(plies=0))


def test_the_browsers_terminal_reason_vocabulary_is_refused() -> None:
    conn = store.connect(Path(":memory:"))
    with pytest.raises(Exception, match="terminal_reason"):
        store.record_game(conn, make_game(terminal_reason="line"), make_meta(), make_positions())


def test_a_schema_other_than_game_result_v1_is_refused() -> None:
    conn = store.connect(Path(":memory:"))
    with pytest.raises(Exception, match="schema"):
        store.record_game(
            conn, make_game(schema="game-result.v2"), make_meta(), make_positions()
        )


def test_deleting_a_game_cascades_to_meta_and_positions() -> None:
    conn = store.connect(Path(":memory:"))
    store.record_game(conn, make_game(), make_meta(), make_positions())

    conn.execute("DELETE FROM games WHERE game_id = ?", ("g1",))
    conn.commit()

    assert store.game_count(conn) == 0
    assert conn.execute("SELECT COUNT(*) FROM game_meta").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM game_positions").fetchone()[0] == 0


def test_head_to_head_splits_by_seat_instead_of_pooling() -> None:
    conn = store.connect(Path(":memory:"))
    store.record_game(
        conn,
        make_game(game_id="g1", winner=0),
        make_meta(game_id="g1", human_seat=0, opponent_id="net-a"),
        make_positions(game_id="g1"),
    )
    store.record_game(
        conn,
        make_game(game_id="g2", winner=0),
        make_meta(game_id="g2", human_seat=1, opponent_id="net-a"),
        make_positions(game_id="g2"),
    )

    rows = {(row["opponent_id"], row["human_seat"]): row for row in store.head_to_head(conn)}

    assert rows[("net-a", 0)]["games"] == 1
    assert rows[("net-a", 0)]["win_rate"] == pytest.approx(1.0)
    assert rows[("net-a", 1)]["games"] == 1
    assert rows[("net-a", 1)]["win_rate"] == pytest.approx(0.0)
    assert len(rows) == 2


def test_distinct_positions_dedupes_shared_openings_and_respects_max_ply() -> None:
    conn = store.connect(Path(":memory:"))
    shared_opening = ["open-0", "open-1"]
    store.record_game(
        conn,
        make_game(game_id="g1"),
        make_meta(game_id="g1"),
        make_positions(game_id="g1", plies=3, keys=[*shared_opening, "g1-only"]),
    )
    store.record_game(
        conn,
        make_game(game_id="g2"),
        make_meta(game_id="g2"),
        make_positions(game_id="g2", plies=3, keys=[*shared_opening, "g2-only"]),
    )

    at_ply_1 = store.distinct_positions(conn, max_ply=1)
    assert {key for _, key in at_ply_1} == {"open-0", "open-1"}

    at_ply_2 = store.distinct_positions(conn, max_ply=2)
    assert {key for _, key in at_ply_2} == {"open-0", "open-1", "g1-only", "g2-only"}


def test_a_failed_insert_leaves_no_partial_rows_in_any_table() -> None:
    conn = store.connect(Path(":memory:"))
    with pytest.raises(Exception):
        store.record_game(
            conn,
            make_game(game_id="broken"),
            make_meta(game_id="broken", human_seat=9),
            make_positions(game_id="broken"),
        )

    assert store.game_count(conn) == 0
    assert conn.execute("SELECT COUNT(*) FROM game_meta").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM game_positions").fetchone()[0] == 0
