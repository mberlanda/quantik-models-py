"""The HTTP surface, exercised against a real socket.

These start an actual `ThreadingHTTPServer` on an ephemeral port and talk
to it with `urllib`, rather than calling the handler methods directly. The
things most likely to be wrong here are exactly the things a direct call
skips: status lines, `Content-Length` on a keep-alive connection, the JSON
body a browser will actually parse, and whether a path with `..` in it
escapes the served directory.

No torch: every opponent used here is classical, and the freshness guard
runs before any agent is built.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from quantik_models.env import fastboard as fb
from quantik_models.play import server as srv
from quantik_models.play.service import PlayService

EMPTY = "..../..../..../...."


@pytest.fixture
def live(tmp_path):
    """A running server on a port the OS picked, torn down after the test."""
    static = tmp_path / "app"
    static.mkdir()
    (static / "index.html").write_text("<h1>quantik</h1>")
    (static / "styles.css").write_text("body{}")
    secret = tmp_path / "secret.txt"
    secret.write_text("not yours")

    service = PlayService(tmp_path / "models")
    (tmp_path / "models").mkdir(exist_ok=True)
    http = srv.make_server(
        service, host="127.0.0.1", port=0, db_path=tmp_path / "games.db", static_dir=static
    )
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{http.server_address[1]}"
    try:
        yield base
    finally:
        http.shutdown()
        http.server_close()
        thread.join(timeout=5)


def get(base, path):
    with urllib.request.urlopen(base + path, timeout=30) as response:
        return response.status, response.read(), response.headers


def post(base, path, payload):
    request = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def move_request(qfen=EMPTY, **overrides):
    boards = fb.from_qfen(qfen)
    body = {
        "schema": "quantik.engine-request.v1",
        "qfen": qfen,
        "side_to_move": int(fb.side_to_move(boards)[0]),
        "legal_action_indices": [int(i) for i in np.flatnonzero(fb.legal_masks(boards)[0])],
    }
    body.update(overrides)
    return body


def play_out(seed=5):
    rng = np.random.default_rng(seed)
    boards = fb.from_qfen(EMPTY)
    actions = []
    while True:
        done, _ = fb.terminal_status(boards)
        if bool(done[0]):
            return actions
        legal = np.flatnonzero(fb.legal_masks(boards)[0])
        actions.append(int(rng.choice(legal)))
        boards = fb.apply_actions(boards, np.array([actions[-1]], dtype=np.int64))


# --- listings -----------------------------------------------------------


def test_the_opponent_list_is_served_as_json(live):
    status, body, headers = get(live, "/api/opponents")
    assert status == 200
    assert headers["Content-Type"].startswith("application/json")
    ids = {o["opponent_id"] for o in json.loads(body)["opponents"]}
    assert {"random", "minimax-d2", "uniform-mcts128"} <= ids


def test_the_model_list_is_served_even_when_empty(live):
    status, body, _ = get(live, "/api/models")
    assert status == 200
    assert json.loads(body) == {"models": []}


def test_an_unknown_api_route_is_a_404_in_json_not_html(live):
    with pytest.raises(urllib.error.HTTPError) as caught:
        get(live, "/api/nonsense")
    assert caught.value.code == 404
    assert json.loads(caught.value.read())["status"] == 404


# --- moves --------------------------------------------------------------


def test_a_move_comes_back_legal(live):
    status, body = post(live, "/api/move/random", move_request("A.../..../..../...."))
    assert status == 200
    assert body["schema"] == "quantik.engine-response.v1"
    assert body["engine_version"] == "random"
    assert body["action_index"] in move_request("A.../..../..../....")["legal_action_indices"]


def test_a_service_error_keeps_its_status_over_http(live):
    """A 422 that arrives as a 500 tells the client nothing it can act on."""
    # One piece placed, so core says the mover is 1; the request claims 0.
    status, body = post(
        live, "/api/move/random", move_request("A.../..../..../....", side_to_move=0)
    )
    assert status == 422
    assert "quantik-core calculated" in body["error"]


def test_an_unknown_opponent_is_a_404(live):
    status, body = post(live, "/api/move/stockfish", move_request())
    assert status == 404


def test_an_empty_body_is_a_400_rather_than_a_traceback(live):
    request = urllib.request.Request(live + "/api/move/random", data=b"", method="POST")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            status = response.status
    except urllib.error.HTTPError as error:
        status = error.code
    assert status == 400


def test_malformed_json_is_a_400(live):
    request = urllib.request.Request(
        live + "/api/move/random", data=b"{not json", method="POST"
    )
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(request, timeout=30)
    assert caught.value.code == 400


# --- recording ----------------------------------------------------------


def record_payload(actions, **overrides):
    body = {
        "schema": "game-result.v1",
        "game_id": "g-http-1",
        "started_at": "2026-08-29T21:00:00+00:00",
        "initial_qfen": EMPTY,
        "move_action_indices": actions,
        "p0_engine_kind": "human",
        "p0_engine_version": "mauro",
        "p1_engine_kind": "random",
        "p1_engine_version": "random",
        "human_seat": 0,
        "player_name": "mauro",
        "opponent_id": "random",
    }
    body.update(overrides)
    return body


def test_a_finished_game_is_recorded_and_the_second_post_is_not(live):
    """A page reload after the result screen must not double-count."""
    actions = play_out()
    status, body = post(live, "/api/games", record_payload(actions))
    assert status == 201
    assert body["recorded"] is True
    assert body["plies"] == len(actions)

    status, body = post(live, "/api/games", record_payload(actions))
    assert status == 200
    assert body["recorded"] is False

    _, summary, _ = get(live, "/api/games")
    assert json.loads(summary)["games"] == 1


def test_the_outcome_stored_is_the_replayed_one_not_the_claimed_one(live):
    actions = play_out(seed=11)
    truth = (len(actions) - 1) % 2
    status, body = post(
        live,
        "/api/games",
        record_payload(actions, game_id="g-lie", winner=1 - truth, plies=99),
    )
    assert status == 201
    assert body["winner"] == truth
    assert body["plies"] == len(actions)
    assert len(body["discrepancies"]) == 2


def test_agreement_is_reported_as_an_empty_list_not_a_missing_field(live):
    """A caller has to be able to tell "they agreed" from "this server does
    not report disagreements"."""
    actions = play_out(seed=12)
    _, body = post(
        live,
        "/api/games",
        record_payload(actions, game_id="g-agree", winner=(len(actions) - 1) % 2),
    )
    assert body["discrepancies"] == []


def test_a_game_that_does_not_replay_is_refused(live):
    status, body = post(live, "/api/games", record_payload([0, 1, 2], game_id="g-bad"))
    assert status == 422
    _, summary, _ = get(live, "/api/games")
    assert json.loads(summary)["games"] == 0


# --- static -------------------------------------------------------------


def test_the_app_is_served_from_the_root(live):
    status, body, headers = get(live, "/")
    assert status == 200
    assert b"quantik" in body
    assert headers["Content-Type"].startswith("text/html")


def test_a_stylesheet_gets_its_own_content_type(live):
    """Served as `text/plain`, a stylesheet is silently ignored by the
    browser and the page renders unstyled with no error anywhere."""
    _, _, headers = get(live, "/styles.css")
    assert headers["Content-Type"].startswith("text/css")


def test_a_traversal_out_of_the_static_directory_is_refused(live):
    """`secret.txt` sits one level above the served directory. A normalising
    server hands it over and says 200."""
    with pytest.raises(urllib.error.HTTPError) as caught:
        get(live, "/../secret.txt")
    assert caught.value.code in (403, 404)
    assert b"not yours" not in caught.value.read()


def test_a_missing_file_is_a_404(live):
    with pytest.raises(urllib.error.HTTPError) as caught:
        get(live, "/nope.js")
    assert caught.value.code == 404


# --- the LAN address ----------------------------------------------------


def test_the_printed_address_is_not_loopback():
    """`gethostname` resolves to 127.0.0.1 on macOS — the answer that looks
    right and does not work from another device. This must not do that
    while a route to the internet exists."""
    address = srv.lan_address(8000)
    assert address.startswith("http://")
    assert address.endswith(":8000")


def test_the_api_index_lists_every_route_the_server_answers(live):
    """A person with the URL should not have to read the source, or ask, to
    find out where the model list lives."""
    status, body, _ = get(live, "/api")
    assert status == 200
    listed = {(r["method"], r["path"].split("?")[0]) for r in json.loads(body)["routes"]}
    assert ("GET", "/api/opponents") in listed
    assert ("POST", "/api/analyse/{opponent_id}") in listed
    assert ("POST", "/api/move/{opponent_id}") in listed
    assert ("GET", "/api/games") in listed


def test_analysis_is_served_and_refuses_an_unknown_opponent(live):
    request = move_request("A.../..../..../....", side_to_move=1)
    status, body = post(live, "/api/analyse/random", request)
    assert status == 200
    assert body["side_to_move"] == 1
    assert body["value_perspective"] == "side_to_move"
    # A classical opponent has no value head, and says so rather than
    # reporting a level bar for a position it has no opinion about.
    assert body["value"] is None and body["win_probability"] is None

    status, body = post(live, "/api/analyse/nobody@128", request)
    assert status == 404
