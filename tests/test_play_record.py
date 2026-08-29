"""Replaying a submitted game, and refusing to take its word for the result.

The failure these guard against is a browser writing a wrong outcome into
the same table the arena writes to, where nothing downstream could tell the
two apart. So the outcome in `games` is always derived here, and the
client's claim is kept beside it — a disagreement becomes the one routine
check that the JavaScript rules and the Python rules still agree.

No torch anywhere in this file: replay is `fastboard` and nothing else.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantik_models.env import fastboard as fb
from quantik_models.play import record
from quantik_models.play.service import ServiceError

EMPTY = "..../..../..../...."


def play_out(qfen=EMPTY, seed=7):
    """A real finished game: greedy-random legal moves until terminal."""
    rng = np.random.default_rng(seed)
    boards = fb.from_qfen(qfen)
    actions = []
    while True:
        done, _ = fb.terminal_status(boards)
        if bool(done[0]):
            return actions
        legal = np.flatnonzero(fb.legal_masks(boards)[0])
        action = int(rng.choice(legal))
        actions.append(action)
        boards = fb.apply_actions(boards, np.array([action], dtype=np.int64))


def payload_for(actions, **overrides):
    body = {
        "schema": record.SCHEMA,
        "game_id": "g-1",
        "started_at": "2026-08-29T20:00:00+00:00",
        "initial_qfen": EMPTY,
        "move_action_indices": actions,
        "p0_engine_kind": "human",
        "p0_engine_version": "mauro",
        "p1_engine_kind": "net-mcts",
        "p1_engine_version": "cpool@128",
    }
    body.update(overrides)
    return body


def test_a_real_game_replays_to_a_terminal_position():
    actions = play_out()
    result = record.replay(EMPTY, actions)
    assert result.plies == len(actions)
    assert result.winner in (0, 1)
    assert result.terminal_reason in ("win_condition", "no_legal_moves")
    # Every ply is kept, the starting position included.
    assert [p["ply"] for p in result.positions] == list(range(len(actions) + 1))


def test_the_winner_is_the_player_who_moved_last():
    """Both Quantik terminal conditions are losses for the side to move, so
    the last mover won. Derived, never taken from the client."""
    for seed in range(6):
        actions = play_out(seed=seed)
        result = record.replay(EMPTY, actions)
        assert result.winner == (len(actions) - 1) % 2


def test_an_illegal_move_is_a_422_naming_the_position():
    actions = play_out()
    boards = fb.from_qfen(EMPTY)
    illegal = next(
        i for i in range(fb.ACTION_COUNT) if not bool(fb.legal_masks(boards)[0][i])
    ) if not fb.legal_masks(boards)[0].all() else None
    # An empty board has no illegal action, so play one move first.
    boards = fb.apply_actions(boards, np.array([actions[0]], dtype=np.int64))
    illegal = next(i for i in range(fb.ACTION_COUNT) if not bool(fb.legal_masks(boards)[0][i]))
    with pytest.raises(ServiceError) as caught:
        record.replay(EMPTY, [actions[0], illegal])
    assert caught.value.status == 422
    assert "illegal" in caught.value.message


def test_moves_after_the_game_ended_are_a_422():
    actions = play_out()
    with pytest.raises(ServiceError) as caught:
        record.replay(EMPTY, actions + [0])
    assert caught.value.status == 422
    assert "ended at move" in caught.value.message


def test_an_unfinished_game_is_a_422():
    actions = play_out()
    with pytest.raises(ServiceError) as caught:
        record.replay(EMPTY, actions[:-1])
    assert caught.value.status == 422
    assert "not terminal" in caught.value.message


def test_an_empty_move_list_is_a_422():
    with pytest.raises(ServiceError) as caught:
        record.replay(EMPTY, [])
    assert caught.value.status == 422


def test_an_unparseable_initial_position_is_a_422():
    with pytest.raises(ServiceError) as caught:
        record.replay("not/a/board", [0])
    assert caught.value.status == 422


def test_an_out_of_range_move_is_a_422():
    with pytest.raises(ServiceError) as caught:
        record.replay(EMPTY, [64])
    assert caught.value.status == 422


# --- the client's claims ------------------------------------------------


def test_a_matching_client_reading_reports_no_discrepancy():
    """The normal case, and it is worth asserting: an empty list is
    positive evidence the two rule implementations agree on this game."""
    actions = play_out()
    result = record.replay(EMPTY, actions)
    payload = payload_for(
        actions, winner=result.winner, plies=result.plies, terminal_reason="line"
    )
    if result.terminal_reason == "no_legal_moves":
        payload["terminal_reason"] = "no_legal_moves"
    assert record.discrepancies(result, payload) == []


def test_the_visualizers_line_maps_to_the_contracts_win_condition():
    """`game.js` emits "line"; `game-result-v1.md` specifies
    "win_condition", and `games.terminal_reason` carries a CHECK — so an
    unmapped value would fail the insert, not the request."""
    actions = play_out(seed=3)
    result = record.replay(EMPTY, actions)
    assert result.terminal_reason == "win_condition"
    assert record.discrepancies(result, payload_for(actions, terminal_reason="line")) == []


def test_a_wrong_client_winner_is_reported_not_raised():
    """A disagreement about the label on a legal game is a parity signal,
    so it is recorded rather than refused. Only moves that cannot be
    trusted stop the game being stored at all."""
    actions = play_out()
    result = record.replay(EMPTY, actions)
    payload = payload_for(actions, winner=1 - result.winner, plies=99)
    reported = record.discrepancies(result, payload)
    assert len(reported) == 2
    assert any("winner" in line for line in reported)
    assert any("plies" in line for line in reported)


def test_the_stored_row_carries_the_replay_and_the_claim_side_by_side():
    actions = play_out()
    result = record.replay(EMPTY, actions)
    payload = payload_for(actions, winner=1 - result.winner, terminal_reason="line")
    game, meta, positions = record.rows_for(
        payload, result, contract_version="1.2.0", service_version="0.1.0"
    )
    assert game["winner"] == result.winner
    assert meta["client_winner"] == 1 - result.winner
    assert game["terminal_reason"] == result.terminal_reason
    assert meta["final_qfen"] == result.final_qfen
    assert positions == result.positions


def test_the_rows_insert_into_the_real_store(tmp_path):
    """The columns line up with the schema, CHECK constraints included —
    which a hand-built dict is exactly the kind of thing to get wrong."""
    from quantik_models.play import store

    actions = play_out()
    result = record.replay(EMPTY, actions)
    game, meta, positions = record.rows_for(
        payload_for(actions), result, contract_version="1.2.0", service_version="0.1.0"
    )
    conn = store.connect(tmp_path / "games.db")
    assert store.record_game(conn, game, meta, positions) is True
    assert store.game_count(conn) == 1
    # The second POST of the same game is a no-op, not a duplicate.
    assert store.record_game(conn, game, meta, positions) is False
    assert store.game_count(conn) == 1


# --- payload validation -------------------------------------------------


def test_a_foreign_schema_is_a_400():
    with pytest.raises(ServiceError) as caught:
        record.validate_payload(payload_for([0], schema="game-result.v2"))
    assert caught.value.status == 400


@pytest.mark.parametrize("field", record.REQUIRED_FIELDS)
def test_every_required_field_is_required(field):
    payload = payload_for([0])
    payload[field] = None
    with pytest.raises(ServiceError) as caught:
        record.validate_payload(payload)
    assert caught.value.status == 400
    assert field in caught.value.message


def test_a_non_list_move_field_is_a_400():
    with pytest.raises(ServiceError) as caught:
        record.validate_payload(payload_for("0,1,2"))
    assert caught.value.status == 400
