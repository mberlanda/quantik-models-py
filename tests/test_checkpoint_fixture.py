# tests/test_checkpoint_fixture.py
"""The committed smoke checkpoint must stay loadable.

Without this the fixture is never exercised and would rot exactly the way
the exported manifests did: stamped with a contracts release that the
current quantik-core-py no longer accepts, with nothing to notice.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "checkpoints" / "smoke-best"


def test_fixture_weights_match_the_manifest_hash() -> None:
    manifest = json.loads((FIXTURE / "manifest.json").read_text())
    weights = FIXTURE / "weights.safetensors"
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    assert manifest["weights_hash"] == f"sha256:{digest}"
    assert manifest["size_bytes"] == weights.stat().st_size


def test_fixture_manifest_validates_through_core_py() -> None:
    artifact_data = pytest.importorskip("quantik_core.artifact_data")
    parsed = artifact_data.load_model_checkpoint_manifest(FIXTURE / "manifest.json")
    assert parsed.model_id == "smoke-best"
    assert parsed.weights_format == "safetensors"
    assert parsed.parameter_count == 13991


def test_fixture_weights_load_into_the_smoke_preset() -> None:
    pytest.importorskip("torch")
    load_file = pytest.importorskip("safetensors.torch").load_file
    from quantik_models.model.policy_value_net import PRESETS, PolicyValueNet

    model = PolicyValueNet(PRESETS["smoke"])
    # strict: every tensor in the file must correspond to a parameter or
    # buffer, so a checkpoint exported from a changed architecture fails here
    # rather than silently loading a subset.
    model.load_state_dict(load_file(FIXTURE / "weights.safetensors"))
