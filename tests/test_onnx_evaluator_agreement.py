"""`OnnxEvaluator` has to agree with `NetEvaluator`, or it is not a serving
runtime — it is a second, silently different model wearing the first one's
manifest.

This is the test workstream 13 (`WORKSTREAMS.md`) names as the prerequisite
for the public deployment image: "what must be built first is the
torch-vs-ONNX agreement test on a real checkpoint, not the evaluator". The
checkpoint here is a freshly built `cpool` (smoke preset, random weights) —
"real" in the sense that matters: an actual `export_checkpoint` output with
an actual traced ONNX graph, not a hand-written fixture. Random weights are
enough because the property under test is numerical agreement between two
runtimes, not anything about what the network has learned.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

from quantik_models.arena.registry import load_evaluator, load_onnx_evaluator  # noqa: E402
from quantik_models.env import fastboard as fb  # noqa: E402
from quantik_models.export.checkpoint import export_checkpoint  # noqa: E402
from quantik_models.model import registry as model_registry  # noqa: E402

from boards import random_positions  # noqa: E402


def _checkpoint(tmp_path):
    torch.manual_seed(0)
    model = model_registry.build("cpool", preset="smoke")
    manifest_path = export_checkpoint(
        model, out_dir=tmp_path, model_id="agreement-smoke", training_report={}
    )
    return manifest_path.parent


def test_onnx_and_torch_evaluators_agree_on_a_real_checkpoint(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    torch_eval = load_evaluator(str(checkpoint), device="cpu")
    onnx_eval = load_onnx_evaluator(str(checkpoint))

    # Several batch sizes, none of them one — the same reason
    # `train.preflight._check_onnx` varies its batch: a graph can advertise
    # a symbolic batch dimension and still carry an internal op frozen at
    # whatever batch size it was traced with.
    for n, plies in ((1, 2), (5, 4), (64, 6)):
        boards = random_positions(n, seed=n, plies=plies)
        legal = fb.legal_masks(boards)

        torch_priors, torch_values = torch_eval(boards, legal)
        onnx_priors, onnx_values = onnx_eval(boards, legal)

        np.testing.assert_allclose(
            onnx_priors, torch_priors, atol=1e-5, rtol=1e-4,
            err_msg=f"policy priors diverge at batch {n}",
        )
        np.testing.assert_allclose(
            onnx_values, torch_values, atol=1e-5, rtol=1e-4,
            err_msg=f"values diverge at batch {n}",
        )
        # The graph is the only artefact an onnx-runtime image ships — this
        # asserts the two runtimes agree on where mass is *forbidden*, not
        # just close in aggregate: an evaluator that leaked mass onto an
        # illegal action would still pass a loose norm check.
        assert np.all(onnx_priors[~legal] == 0.0)


def test_build_agent_wires_the_onnx_runtime_through(tmp_path):
    """The path the play service actually calls: `build_agent` with
    `runtime: "onnx"` in the spec, not `load_onnx_evaluator` directly."""
    from quantik_models.arena.registry import build_agent

    checkpoint = _checkpoint(tmp_path)
    agent = build_agent(
        {"kind": "net-policy", "checkpoint": str(checkpoint), "runtime": "onnx", "name": "t"}
    )
    boards = fb.empty_boards(1)
    legal = fb.legal_masks(boards)[0]
    action = agent.select(boards[0], seed=0)
    assert legal[action]


def test_an_unknown_runtime_in_the_spec_is_rejected(tmp_path):
    from quantik_models.arena.registry import build_agent

    checkpoint = _checkpoint(tmp_path)
    with pytest.raises(ValueError):
        build_agent(
            {
                "kind": "net-policy",
                "checkpoint": str(checkpoint),
                "runtime": "tensorflow",
                "name": "t",
            }
        )


def test_onnx_evaluator_rejects_no_positions_the_same_way_torch_does(tmp_path):
    """The empty-batch edge case both evaluators special-case, so a caller
    batching zero live games does not crash either runtime."""
    checkpoint = _checkpoint(tmp_path)
    torch_eval = load_evaluator(str(checkpoint), device="cpu")
    onnx_eval = load_onnx_evaluator(str(checkpoint))

    empty_boards = np.zeros((0, 8), dtype=np.uint16)
    empty_legal = np.zeros((0, fb.ACTION_COUNT), dtype=np.bool_)

    torch_priors, torch_values = torch_eval(empty_boards, empty_legal)
    onnx_priors, onnx_values = onnx_eval(empty_boards, empty_legal)
    assert torch_priors.shape == onnx_priors.shape == (0, fb.ACTION_COUNT)
    assert torch_values.shape == onnx_values.shape == (0,)
