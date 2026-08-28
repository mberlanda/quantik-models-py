"""The registry contract every architecture has to satisfy.

These tests are deliberately architecture-agnostic: they run over whatever
is registered, so a new architecture is covered the moment it is added
rather than when someone remembers to write a test for it.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from quantik_models.export.checkpoint import export_checkpoint  # noqa: E402
from quantik_models.model import registry  # noqa: E402
from quantik_models.model.spec import ACTION_COUNT, BOARD_SIZE, INPUT_PLANES  # noqa: E402

ARCHITECTURES = registry.architectures()


def _smoke_preset(name: str) -> str:
    presets = registry.presets(name)
    return "smoke" if "smoke" in presets else presets[0]


@pytest.mark.parametrize("name", ARCHITECTURES)
def test_forward_shapes_match_the_tensor_contract(name: str) -> None:
    model = registry.build(name, preset=_smoke_preset(name)).eval()
    batch = 3
    policy, value = model(torch.zeros(batch, INPUT_PLANES, BOARD_SIZE, BOARD_SIZE))
    assert policy.shape == (batch, ACTION_COUNT)
    assert value.shape == (batch,)


@pytest.mark.parametrize("name", ARCHITECTURES)
def test_value_head_is_bounded(name: str) -> None:
    """The output contract is `value-tanh`, so a runtime may rely on [-1, 1]."""
    model = registry.build(name, preset=_smoke_preset(name)).eval()
    torch.manual_seed(0)
    _, value = model(torch.randn(16, INPUT_PLANES, BOARD_SIZE, BOARD_SIZE))
    assert torch.all(value >= -1.0) and torch.all(value <= 1.0)


@pytest.mark.parametrize("name", ARCHITECTURES)
def test_identity_strings_are_populated(name: str) -> None:
    """`architecture` and `model_family` land in the published manifest."""
    model = registry.build(name, preset=_smoke_preset(name))
    assert model.architecture.strip()
    assert model.model_family.startswith("quantik-policy-value")


@pytest.mark.parametrize("name", ARCHITECTURES)
def test_batch_size_one_works(name: str) -> None:
    """Serving evaluates single positions; batch norm must be in eval mode.

    A model left in train mode raises on a batch of one, which would only
    ever show up in production rather than in a batched training loop.
    """
    model = registry.build(name, preset=_smoke_preset(name)).eval()
    policy, value = model(torch.zeros(1, INPUT_PLANES, BOARD_SIZE, BOARD_SIZE))
    assert policy.shape == (1, ACTION_COUNT)
    assert value.shape == (1,)


def test_unknown_architecture_names_the_known_ones() -> None:
    with pytest.raises(ValueError, match="unknown architecture"):
        registry.build("does-not-exist", preset="smoke")


def test_unknown_preset_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown preset"):
        registry.build("resnet", preset="does-not-exist")


def test_overrides_apply_on_top_of_a_preset() -> None:
    model = registry.build("resnet", preset="small", channels=32, blocks=1)
    assert model.architecture == "resnet-c32-b1"


@pytest.mark.parametrize("name", ARCHITECTURES)
def test_onnx_export_matches_torch(name: str, tmp_path) -> None:
    """The ONNX graph is only useful if it computes the same function.

    This is the guard for serving the checkpoint from a non-Python runtime:
    without it, an architecture could export a graph that silently disagrees
    with the model that was trained.
    """
    pytest.importorskip("onnxscript")
    ort = pytest.importorskip("onnxruntime")
    import numpy as np

    torch.manual_seed(0)
    model = registry.build(name, preset=_smoke_preset(name)).eval()
    manifest_path = export_checkpoint(
        model, out_dir=tmp_path, model_id=f"{name}-test", training_report={}
    )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["onnx_export"] == "model.onnx"
    # Weights must live inside the graph, or `onnx_hash` describes only part
    # of what a runtime actually loads.
    assert not (tmp_path / "model.onnx.data").exists()

    sample = torch.randn(4, INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)
    with torch.no_grad():
        want_policy, want_value = model(sample)
    session = ort.InferenceSession(
        str(tmp_path / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    got_policy, got_value = session.run(None, {"board": sample.numpy()})

    np.testing.assert_allclose(got_policy, want_policy.numpy(), atol=1e-5)
    np.testing.assert_allclose(got_value, want_value.numpy(), atol=1e-5)
