"""The play service's model and opponent registries.

The failure these guard against is not "wrong answer" but "silently
missing": a model that fails to load should still appear, greyed out with a
reason, and every opponent spec has to actually build and move — a spec
that merely looks like what `build_agent` expects is not the same as one
that is.

`build_agent` is imported inside the one test that needs it rather than at
module scope. It reaches `model.registry`, which imports torch — and the
torch-free install is a tested configuration here, so a module-scope import
would fail collection for this whole file instead of skipping the single
test that needs a network. Both registries under test are themselves
torch-free and stay covered in that job.
"""

from __future__ import annotations

import json

import pytest

from quantik_models.env import fastboard as fb
from quantik_models.export.digest import file_digest
from quantik_models.play import opponents as op
from quantik_models.play import registry as reg


def write_checkpoint(root, name, *, weights_bytes=b"not-real-weights", **manifest_overrides):
    """A hand-built `model-checkpoint.v1` directory: no torch, no real weights.

    `weights_hash` is computed from `weights_bytes` unless the caller wants
    a mismatch, in which case it passes its own wrong hash through
    `manifest_overrides`.
    """
    model_dir = root / name
    model_dir.mkdir(parents=True)
    weights_path = model_dir / "weights.safetensors"
    weights_path.write_bytes(weights_bytes)
    manifest = {
        "schema": "model-checkpoint.v1",
        "model_id": f"{name}-artifact",
        "architecture": "resnet-c16-b2",
        "architecture_spec": {"arch": "resnet", "config": {"channels": 16, "blocks": 2}},
        "parameter_count": 13991,
        "weights_hash": file_digest(weights_path),
    }
    manifest.update(manifest_overrides)
    (model_dir / "manifest.json").write_text(json.dumps(manifest))
    return model_dir


def test_a_valid_checkpoint_is_ready(tmp_path):
    write_checkpoint(tmp_path, "cpool-a")
    models = reg.scan_models(tmp_path)
    assert len(models) == 1
    model = models[0]
    assert model.status == "ready"
    assert model.reason is None
    assert model.architecture == "resnet-c16-b2"
    assert model.parameter_count == 13991


def test_a_hash_mismatch_is_refused_and_the_reason_names_the_hash(tmp_path):
    """A checkpoint that would fail `arena.registry.load_evaluator`'s load
    must be caught here first, with the mismatched hash visible in the
    reason rather than just "refused"."""
    model_dir = write_checkpoint(tmp_path, "cpool-a")
    (model_dir / "weights.safetensors").write_bytes(b"tampered-bytes")
    models = reg.scan_models(tmp_path)
    assert len(models) == 1
    model = models[0]
    assert model.status == "refused"
    assert model.reason is not None
    assert "hash" in model.reason
    wrong_hash = file_digest(model_dir / "weights.safetensors")
    assert wrong_hash in model.reason


def test_a_missing_manifest_is_refused_not_skipped(tmp_path):
    """A directory that quietly vanishes from the list is worse than one
    that appears greyed out with a reason."""
    model_dir = tmp_path / "no-manifest"
    model_dir.mkdir()
    (model_dir / "weights.safetensors").write_bytes(b"anything")
    models = reg.scan_models(tmp_path)
    assert len(models) == 1
    assert models[0].model_id == "no-manifest"
    assert models[0].status == "refused"
    assert models[0].reason is not None and "manifest" in models[0].reason


def test_a_wrong_schema_is_refused(tmp_path):
    write_checkpoint(tmp_path, "old-format", schema="checkpoint.v0")
    models = reg.scan_models(tmp_path)
    assert models[0].status == "refused"
    assert models[0].reason is not None and "schema" in models[0].reason


def test_model_id_is_the_directory_name_even_when_the_manifest_disagrees(tmp_path):
    """The directory name is what a person chose and what a game record
    will store; the manifest's id is a separate artifact identity that must
    not be lost, just not used as the primary key."""
    write_checkpoint(tmp_path, "my-favourite-cpool")
    models = reg.scan_models(tmp_path)
    model = models[0]
    assert model.model_id == "my-favourite-cpool"
    assert model.manifest_model_id == "my-favourite-cpool-artifact"
    assert model.model_id != model.manifest_model_id


def test_no_architecture_spec_and_non_resnet_architecture_is_refused(tmp_path):
    """Mirrors the exact condition `arena.registry._model_from_manifest`
    raises on, so a model that cannot be rebuilt is refused at scan time
    rather than 500ing on its first move."""
    write_checkpoint(
        tmp_path, "legacy-mlp", architecture="mlp-h64-b2", architecture_spec=None
    )
    models = reg.scan_models(tmp_path)
    model = models[0]
    assert model.status == "refused"
    assert model.reason is not None
    assert "architecture_spec" in model.reason


def test_every_classical_spec_builds_and_returns_a_legal_move():
    """Proves the specs are right, not merely plausible: each one must
    actually construct through `build_agent` and choose a legal action."""
    pytest.importorskip("torch")
    from quantik_models.arena.registry import build_agent

    boards = fb.empty_boards(1)
    legal = fb.legal_masks(boards)[0]
    for opponent in op.CLASSICAL:
        agent = build_agent(opponent.spec)
        action = agent.select(boards[0], seed=0)
        assert legal[action], f"{opponent.opponent_id} chose an illegal action"


def test_a_ready_model_yields_exactly_two_opponents(tmp_path):
    write_checkpoint(tmp_path, "cpool-a")
    models = reg.scan_models(tmp_path)
    generated = op.neural_opponents(models)
    assert len(generated) == 2
    by_id = {o.opponent_id: o for o in generated}
    assert set(by_id) == {"cpool-a@0", "cpool-a@128"}
    assert by_id["cpool-a@0"].simulations == 0
    assert by_id["cpool-a@0"].kind == "net-policy"
    assert by_id["cpool-a@128"].simulations == 128
    assert by_id["cpool-a@128"].kind == "net-mcts"
    for opponent in generated:
        assert opponent.spec["checkpoint"] == str(tmp_path / "cpool-a")
        assert opponent.model_id == "cpool-a"


def test_a_refused_model_yields_no_opponents(tmp_path):
    model_dir = write_checkpoint(tmp_path, "cpool-a")
    (model_dir / "weights.safetensors").write_bytes(b"tampered")
    models = reg.scan_models(tmp_path)
    assert models[0].status == "refused"
    assert op.neural_opponents(models) == []


def test_roster_combines_classical_and_neural(tmp_path):
    write_checkpoint(tmp_path, "cpool-a")
    models = reg.scan_models(tmp_path)
    full = op.roster(models)
    assert len(full) == len(op.CLASSICAL) + 2
    ids = {o.opponent_id for o in full}
    assert "random" in ids
    assert "cpool-a@0" in ids and "cpool-a@128" in ids
