# tests/test_export_checkpoint.py
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from quantik_models.export.checkpoint import export_checkpoint  # noqa: E402
from quantik_models.model.policy_value_net import (  # noqa: E402
    PRESETS,
    PolicyValueNet,
)


def _export(tmp_path: Path) -> Path:
    model = PolicyValueNet(PRESETS["smoke"])
    return export_checkpoint(
        model,
        out_dir=tmp_path,
        model_id="quantik-pv-test",
        training_report={"final": {"policy_loss": 1.0}},
    )


def test_export_writes_weights_manifest_report(tmp_path: Path) -> None:
    manifest_path = _export(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    weights = tmp_path / "weights.safetensors"
    assert weights.is_file()
    assert manifest["schema"] == "model-checkpoint.v1"
    assert manifest["weights_format"] == "safetensors"
    assert manifest["size_bytes"] == weights.stat().st_size
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    assert manifest["weights_hash"] == f"sha256:{digest}"
    assert manifest["legal_action_mask_required"] is True
    assert manifest["architecture"] == "resnet-c16-b2"
    assert (tmp_path / "training-report.json").is_file()


def test_weights_round_trip(tmp_path: Path) -> None:
    from safetensors.torch import load_file

    model = PolicyValueNet(PRESETS["smoke"])

    # Drive the model in train() mode with a few random batches so BatchNorm
    # running stats diverge from their defaults (running_mean=0,
    # running_var=1). Otherwise a regression that drops BN buffers from the
    # export would go undetected, since a freshly-constructed model already
    # matches the untouched defaults.
    model.train()
    for _ in range(3):
        model(torch.rand(8, 9, 4, 4))

    export_checkpoint(
        model,
        out_dir=tmp_path,
        model_id="quantik-pv-test",
        training_report={},
    )
    restored = PolicyValueNet(PRESETS["smoke"])
    restored.load_state_dict(load_file(tmp_path / "weights.safetensors"))

    # Explicit buffer check: the restored model must pick up the diverged
    # BatchNorm running stats, not just matching parameters.
    assert not torch.allclose(
        model.stem[1].running_mean, torch.zeros_like(model.stem[1].running_mean)
    )
    assert torch.allclose(model.stem[1].running_mean, restored.stem[1].running_mean)

    x = torch.rand(2, 9, 4, 4)
    model.eval()
    restored.eval()
    with torch.no_grad():
        a, av = model(x)
        b, bv = restored(x)
    assert torch.allclose(a, b) and torch.allclose(av, bv)


def test_manifest_validates_through_core_py(tmp_path: Path) -> None:
    artifact_data = pytest.importorskip("quantik_core.artifact_data")
    manifest_path = _export(tmp_path)
    parsed = artifact_data.load_model_checkpoint_manifest(manifest_path)
    assert parsed.weights_format == "safetensors"
    assert parsed.parameter_count is not None and parsed.parameter_count > 0
