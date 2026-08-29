"""The move handler's refusals, which are most of what it does.

Almost every test here asserts a *status code*, not merely that something
raised. The status is the contract with the browser — a 400 tells the
client it sent nonsense, a 422 tells it the position disagrees with what it
claimed, a 409 tells it the model on disk moved — and a handler that
refuses correctly with the wrong number is still broken.

`build_agent` is never imported at module scope; it reaches `model.registry`
and therefore torch, and the torch-free install is a tested configuration.
The classical opponents, the whole validation surface and the stale-weights
guard are all exercised without a network, which is the point of ordering
the freshness check ahead of agent construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantik_models.env import fastboard as fb
from quantik_models.play import service as svc

from test_play_registry import write_checkpoint

EMPTY = "..../..../..../...."


def request_for(position: str = EMPTY, **overrides):
    """A well-formed request for `qfen`, with the legality core computes.

    Overrides are applied last, so a test can replace any single field with
    something malformed and keep the rest well-formed — which is what makes
    the 400 assertions attribute the refusal to the field under test.
    """
    boards = fb.from_qfen(position)
    body = {
        "schema": svc.REQUEST_SCHEMA,
        "qfen": position,
        "side_to_move": int(fb.side_to_move(boards)[0]),
        "legal_action_indices": [int(i) for i in np.flatnonzero(fb.legal_masks(boards)[0])],
    }
    body.update(overrides)
    return body


@pytest.fixture
def service(tmp_path):
    return svc.PlayService(tmp_path)


def test_the_roster_is_offered_even_with_no_models_staged(service):
    """Serving zero models is a valid state, not an error."""
    assert service.list_models() == []
    ids = {o["opponent_id"] for o in service.list_opponents()}
    assert {"random", "minimax-d2", "beam-w32", "uniform-mcts128"} <= ids


def test_a_staged_model_appears_in_both_listings(tmp_path):
    write_checkpoint(tmp_path, "cpool")
    service = svc.PlayService(tmp_path)
    assert [m["model_id"] for m in service.list_models()] == ["cpool"]
    ids = {o["opponent_id"] for o in service.list_opponents()}
    assert {"cpool@0", "cpool@128"} <= ids


def test_a_refused_model_is_listed_with_its_reason_but_offers_no_opponent(tmp_path):
    model_dir = write_checkpoint(tmp_path, "cpool")
    (model_dir / "weights.safetensors").write_bytes(b"tampered")
    service = svc.PlayService(tmp_path)
    (listed,) = service.list_models()
    assert listed["status"] == "refused"
    assert "does not match" in listed["reason"]
    assert not [o for o in service.list_opponents() if o["model_id"] == "cpool"]


# --- the eight refusals -------------------------------------------------


def test_a_foreign_schema_is_a_400(service):
    body = request_for(EMPTY, schema="quantik.engine-request.v2")
    with pytest.raises(svc.ServiceError) as caught:
        service.choose_move("random", body)
    assert caught.value.status == 400
    assert "schema" in caught.value.message


@pytest.mark.parametrize(
    "overrides",
    [
        {"side_to_move": 2},
        {"side_to_move": -1},
        {"side_to_move": "0"},
        {"qfen": ""},
        {"qfen": 4},
        {"legal_action_indices": 5},
        {"legal_action_indices": [64]},
        {"legal_action_indices": [-1]},
        {"legal_action_indices": ["3"]},
        {"config": []},
    ],
)
def test_a_malformed_field_is_a_400(service, overrides):
    with pytest.raises(svc.ServiceError) as caught:
        service.choose_move("random", request_for(**overrides))
    assert caught.value.status == 400


def test_a_non_object_body_is_a_400(service):
    with pytest.raises(svc.ServiceError) as caught:
        service.choose_move("random", ["not", "a", "dict"])
    assert caught.value.status == 400


def test_an_unknown_opponent_is_a_404(service):
    with pytest.raises(svc.ServiceError) as caught:
        service.choose_move("stockfish", request_for(EMPTY))
    assert caught.value.status == 404
    assert "stockfish" in caught.value.message


def test_an_unparseable_qfen_is_a_400(service):
    body = request_for(EMPTY)
    body["qfen"] = "not/a/board"
    with pytest.raises(svc.ServiceError) as caught:
        service.choose_move("random", body)
    assert caught.value.status == 400


def test_a_decided_position_is_a_422(service):
    """Four shapes on one line: the game is over, so there is no move."""
    qfen = "ABCD/..../..../...."
    body = request_for(qfen)
    with pytest.raises(svc.ServiceError) as caught:
        service.choose_move("random", body)
    assert caught.value.status == 422
    assert "decided" in caught.value.message


def test_a_side_to_move_disagreement_is_a_422_naming_both(service):
    qfen = "A.../..../..../...."  # one piece placed, so core says 1
    body = request_for(qfen, side_to_move=0)
    with pytest.raises(svc.ServiceError) as caught:
        service.choose_move("random", body)
    assert caught.value.status == 422
    assert "quantik-core calculated 1" in caught.value.message


def test_a_legality_mismatch_is_a_422_naming_what_differs(service):
    """The client's legality is a claim, and this is the check that tests it.

    One legal index removed and one illegal index added: the message has to
    name both sides of the disagreement, because a bare "mismatch" would
    leave a JS/Python rules divergence undiagnosable.
    """
    qfen = "A.../..../..../...."
    body = request_for(qfen)
    legal = body["legal_action_indices"]
    illegal = next(i for i in range(fb.ACTION_COUNT) if i not in legal)
    dropped = legal.pop(3)
    legal.append(illegal)
    with pytest.raises(svc.ServiceError) as caught:
        service.choose_move("random", body)
    assert caught.value.status == 422
    assert f"claimed but illegal [{illegal}]" in caught.value.message
    assert f"legal but omitted [{dropped}]" in caught.value.message


def test_a_duplicated_index_is_not_a_mismatch(service):
    """The comparison dedupes, matching the Rust gateway's sort-and-dedup."""
    body = request_for(EMPTY)
    body["legal_action_indices"] = body["legal_action_indices"] * 2
    assert service.choose_move("random", body)["action_index"] >= 0


# --- the happy path, without a network ----------------------------------


@pytest.mark.parametrize("opponent_id", ["random", "minimax-d2", "beam-w32"])
def test_a_classical_opponent_returns_a_legal_move(service, opponent_id):
    qfen = "A.../..../..../...."
    body = request_for(qfen)
    response = service.choose_move(opponent_id, body)
    assert response["schema"] == svc.RESPONSE_SCHEMA
    assert response["action_index"] in body["legal_action_indices"]
    assert response["engine_version"] == opponent_id
    assert response["elapsed_ms"] >= 0
    # No network, so no assessment to report — not a null, absent.
    assert "policy" not in response and "value" not in response


def test_a_seed_makes_the_random_opponent_reproducible(service):
    body = request_for(EMPTY, config={"seed": 12345})
    first = service.choose_move("random", body)["action_index"]
    assert service.choose_move("random", body)["action_index"] == first


def test_without_a_seed_the_random_opponent_varies(service):
    """A hardcoded default seed would replay one game forever."""
    body = request_for(EMPTY)
    seen = {service.choose_move("random", body)["action_index"] for _ in range(40)}
    assert len(seen) > 1


# --- the stale-weights guard --------------------------------------------


def test_weights_that_change_under_a_running_service_are_a_409(tmp_path):
    """The failure this prevents is silent: `load_evaluator` caches on the
    checkpoint path, so retraining into a served directory keeps the old
    network alive while every recorded game is labelled with the new one.

    The check runs before the agent is built, which is why this test needs
    no torch: a tampered checkpoint is refused without anything ever trying
    to load it.
    """
    write_checkpoint(tmp_path, "cpool")
    service = svc.PlayService(tmp_path)
    assert "cpool@0" in {o["opponent_id"] for o in service.list_opponents()}

    (tmp_path / "cpool" / "weights.safetensors").write_bytes(b"retrained-in-place")
    with pytest.raises(svc.ServiceError) as caught:
        service.choose_move("cpool@0", request_for(EMPTY))
    assert caught.value.status == 409
    assert "does not match the manifest" in caught.value.message


def test_missing_weights_under_a_running_service_are_a_409(tmp_path):
    write_checkpoint(tmp_path, "cpool")
    service = svc.PlayService(tmp_path)
    (tmp_path / "cpool" / "weights.safetensors").unlink()
    with pytest.raises(svc.ServiceError) as caught:
        service.choose_move("cpool@0", request_for(EMPTY))
    assert caught.value.status == 409
    assert "unreadable" in caught.value.message


def test_refresh_picks_up_a_newly_staged_model(tmp_path):
    service = svc.PlayService(tmp_path)
    assert service.list_models() == []
    write_checkpoint(tmp_path, "cpool")
    service.refresh()
    assert [m["model_id"] for m in service.list_models()] == ["cpool"]


# --- the network path ---------------------------------------------------


def test_a_neural_opponent_reports_the_networks_own_read(tmp_path):
    """Needs a real checkpoint and therefore torch; skipped without either."""
    pytest.importorskip("torch")
    from quantik_models.export.checkpoint import export_checkpoint
    from quantik_models.model.registry import build_from_spec

    model = build_from_spec({"arch": "resnet", "config": {"channels": 16, "blocks": 2}})
    models_dir = tmp_path / "models"
    export_checkpoint(
        model,
        out_dir=models_dir / "tiny",
        model_id="tiny",
        training_report={"note": "untrained; this test is about plumbing, not play"},
        with_onnx=False,
    )

    service = svc.PlayService(models_dir)
    ready = [m for m in service.list_models() if m["status"] == "ready"]
    if not ready:
        pytest.skip(f"staged checkpoint was refused: {service.list_models()}")

    body = request_for("A.../..../..../....")
    response = service.choose_move("tiny@0", body)
    assert response["action_index"] in body["legal_action_indices"]
    assert response["engine_version"] == "tiny@0"
    assert len(response["policy"]) == fb.ACTION_COUNT
    assert -1.0 <= response["value"] <= 1.0
