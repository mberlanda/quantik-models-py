"""`POST /api/analyse/{opponent_id}` — the network's read of a position.

Separate from `/api/move` on purpose: asking "who is winning here" must
not require playing a move, and the browser wants the answer for the
*human's* position too, which no move request ever produces.

Two things this has to get right or it misinforms rather than informs.
The value head is **mover-relative** — it is the side to move's own
prospects, not player 0's — so an endpoint that reports it without saying
whose it is gets read backwards on every odd ply. And it is the network's
estimate, not the exact oracle's: the positions here carry known
game-theoretic values, and the tests assert the *shape* and the framing of
the answer, never that the network agrees with the solver.

Torch-free: the position plumbing and the refusals are tested without a
checkpoint, and the one test needing a network skips without torch.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantik_models.env import fastboard as fb
from quantik_models.play import service as svc

# Verified against `quantik-core-rust`'s exact_oracle and the solved
# corpus, so the framing can be checked against a known answer.
LOST_FOR_MOVER = ".c.D/..../bAC./...."  # ply 5, side 1 to move, every move loses
ONE_WINNING_MOVE = ".C.D/..../.d.a/.d.B"  # ply 6, side 0 to move, exactly C@7 wins


def request_for(qfen: str, side_to_move: int) -> dict:
    board = np.asarray(fb.from_qfen(qfen)).reshape(1, -1)
    return {
        "schema": "quantik.engine-request.v1",
        "qfen": qfen,
        "side_to_move": side_to_move,
        "legal_action_indices": np.flatnonzero(fb.legal_masks(board)[0]).tolist(),
    }


def test_an_unknown_opponent_is_a_404(tmp_path):
    service = svc.PlayService(tmp_path)
    with pytest.raises(svc.ServiceError) as caught:
        service.analyse("nobody@128", request_for(LOST_FOR_MOVER, 1))
    assert caught.value.status == 404


def test_an_opponent_with_no_network_says_so_rather_than_inventing_a_value(tmp_path):
    """`minimax-d2` is a real opponent with no value head. Returning 0.0,
    or 0.5 win probability, would draw an evaluation bar at dead level for
    a position that is actually lost — a confident-looking wrong answer.
    The absence has to be visible."""
    service = svc.PlayService(tmp_path)
    result = service.analyse("minimax-d2", request_for(LOST_FOR_MOVER, 1))
    assert result["value"] is None
    assert result["win_probability"] is None
    assert result["policy"] is None
    assert result["top_moves"] == []
    assert result["side_to_move"] == 1


def test_the_analysis_never_hides_whose_side_the_value_is_on(tmp_path):
    service = svc.PlayService(tmp_path)
    for qfen, side in ((LOST_FOR_MOVER, 1), (ONE_WINNING_MOVE, 0)):
        result = service.analyse("random", request_for(qfen, side))
        assert result["side_to_move"] == side
        assert result["value_perspective"] == "side_to_move"


def test_a_terminal_position_is_refused_not_evaluated(tmp_path):
    """A finished game has no side to move whose prospects mean anything.
    `/api/move` already refuses these; analysis has to agree, or the bar
    shows a number for a position nobody is playing."""
    service = svc.PlayService(tmp_path)
    finished = "A..C/bbd./CD.A/.adB"
    board = np.asarray(fb.from_qfen(finished)).reshape(1, -1)
    done, _ = fb.terminal_status(board)
    assert bool(done[0])
    with pytest.raises(svc.ServiceError) as caught:
        service.analyse("random", {
            "schema": "quantik.engine-request.v1",
            "qfen": finished,
            "side_to_move": 1,
            "legal_action_indices": [],
        })
    assert caught.value.status == 422


def _staged_model(tmp_path):
    """A real `model-checkpoint.v1` on disk. Untrained on purpose — these
    tests are about the shape and framing of the answer, not its quality."""
    from quantik_models.export.checkpoint import export_checkpoint
    from quantik_models.model.registry import build_from_spec

    models_dir = tmp_path / "models"
    export_checkpoint(
        build_from_spec({"arch": "resnet", "config": {"channels": 16, "blocks": 2}}),
        out_dir=models_dir / "tiny",
        model_id="tiny",
        training_report={"note": "untrained; plumbing only"},
        with_onnx=False,
    )
    return models_dir


def test_a_network_returns_a_bounded_value_and_ranked_legal_moves(tmp_path):
    """The contract the browser draws: a win probability in [0, 1], a
    policy over all 64 slots, and `top_moves` ranked, legal, and decoded
    into the shape/position the board actually shows."""
    pytest.importorskip("torch")
    service = svc.PlayService(_staged_model(tmp_path))
    if not [m for m in service.list_models() if m["status"] == "ready"]:
        pytest.skip(f"staged checkpoint was refused: {service.list_models()}")
    opponent = "tiny@0"
    request = request_for(ONE_WINNING_MOVE, 0)
    result = service.analyse(opponent, request)

    assert -1.0 <= result["value"] <= 1.0
    assert 0.0 <= result["win_probability"] <= 1.0
    assert result["win_probability"] == pytest.approx((result["value"] + 1.0) / 2.0)
    assert len(result["policy"]) == fb.ACTION_COUNT

    legal = set(request["legal_action_indices"])
    priors = [m["prior"] for m in result["top_moves"]]
    assert priors == sorted(priors, reverse=True)
    for move in result["top_moves"]:
        assert move["action_index"] in legal
        assert move["shape"] == "ABCD"[move["action_index"] // 16]
        assert move["position"] == move["action_index"] % 16

    # Analysis must not consume the game: asking twice is the same answer.
    assert service.analyse(opponent, request)["value"] == result["value"]


def test_an_illegal_legality_set_is_refused_here_too(tmp_path):
    """The same check `/api/move` makes. A client whose legality has
    drifted would otherwise get an analysis of a position neither side is
    actually playing."""
    service = svc.PlayService(tmp_path)
    request = request_for(ONE_WINNING_MOVE, 0)
    request["legal_action_indices"] = request["legal_action_indices"][:-1]
    with pytest.raises(svc.ServiceError) as caught:
        service.analyse("random", request)
    assert caught.value.status == 422
