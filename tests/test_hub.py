"""Loading the published models from the Hugging Face Hub.

Nothing here touches the network: `_snapshot_download` is the one seam that
does, and every test replaces it. What is worth guarding is not the download
— that is `huggingface_hub`'s job — but the three things around it that were
wrong at some point and failed quietly:

* a Hub repo names its weights `model.safetensors` and a local checkpoint
  names them `weights.safetensors`, and the loader has to read both;
* the short names callers pass have to keep resolving to the repo ids that
  are actually published, because a typo here is a 404 for the user and
  nothing at all for us;
* the digest check has to fire on a mismatch, since the failure it stands in
  for — a partial download — otherwise surfaces as a corrupt state dict.
"""

from __future__ import annotations

import json

import pytest

from quantik_models import hub
from quantik_models.arena import registry

MANIFEST = {
    "schema": "model-checkpoint.v1",
    "architecture": "cpool-c191-b6",
    "architecture_spec": {"arch": "cpool", "config": {"blocks": 6, "channels": 191}},
    "contract_version": "1.2.0",
    "weights_format": "safetensors",
}


def write_hub_layout(root, *, weights=b"weights", digest=None):
    """A directory shaped like `snapshot_download` output, not like `runs/`."""
    d = root / "snapshot"
    d.mkdir(parents=True)
    (d / "model.safetensors").write_bytes(weights)
    manifest = dict(MANIFEST)
    if digest is not None:
        manifest["weights_hash"] = digest
    (d / "manifest.json").write_text(json.dumps(manifest))
    return d


# --- the published table -------------------------------------------------


def test_every_published_short_name_is_a_registered_architecture() -> None:
    from quantik_models.model import registry as model_registry

    assert set(hub.PUBLISHED) == set(model_registry.architectures())


def test_repo_ids_are_namespaced_and_carry_the_architecture() -> None:
    for name, model in hub.PUBLISHED.items():
        assert model.repo == f"{hub.NAMESPACE}/quantik-{model.architecture}"
        assert model.architecture.startswith(f"{name}-")
        assert model.url == f"https://huggingface.co/{model.repo}"


def test_a_full_repo_id_passes_through_unresolved() -> None:
    # A fork or a privately retrained model must be loadable without a code
    # change here; PUBLISHED is a convenience table, not a whitelist.
    assert hub.repo_id("someone/quantik-cpool-c191-b6") == (
        "someone/quantik-cpool-c191-b6"
    )


def test_an_unknown_bare_name_names_the_alternatives() -> None:
    with pytest.raises(KeyError) as excinfo:
        hub.repo_id("cpoool")
    message = str(excinfo.value)
    assert "cpool" in message and "<owner>/<repo>" in message


# --- the weights filename ------------------------------------------------


def test_weights_resolve_in_the_hub_layout(tmp_path) -> None:
    # The regression this file exists for: every published model card tells a
    # reader to load `snapshot_download(repo_id)`, which is this layout.
    d = write_hub_layout(tmp_path)
    assert registry.weights_path(d) == d / "model.safetensors"


def test_weights_resolve_in_the_local_checkpoint_layout(tmp_path) -> None:
    d = tmp_path / "best"
    d.mkdir()
    (d / "weights.safetensors").write_bytes(b"w")
    assert registry.weights_path(d) == d / "weights.safetensors"


def test_missing_weights_name_both_candidates(tmp_path) -> None:
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(FileNotFoundError) as excinfo:
        registry.weights_path(d)
    message = str(excinfo.value)
    assert "weights.safetensors" in message and "model.safetensors" in message


# --- digest verification -------------------------------------------------


def test_verify_accepts_a_matching_digest(tmp_path) -> None:
    from quantik_models.export.digest import file_digest

    d = write_hub_layout(tmp_path)
    manifest = json.loads((d / "manifest.json").read_text())
    manifest["weights_hash"] = file_digest(d / "model.safetensors")
    (d / "manifest.json").write_text(json.dumps(manifest))
    hub.verify(d)  # does not raise


def test_verify_rejects_a_truncated_download(tmp_path) -> None:
    d = write_hub_layout(tmp_path, digest="sha256:0000")
    with pytest.raises(ValueError) as excinfo:
        hub.verify(d)
    assert "sha256:0000" in str(excinfo.value)


def test_verify_is_a_no_op_when_the_manifest_carries_no_digest(tmp_path) -> None:
    # Older manifests predate `weights_hash`. Refusing them would make the
    # loader stricter than the checkpoints it has to read.
    hub.verify(write_hub_layout(tmp_path))


# --- resolve / load ------------------------------------------------------


def test_resolve_requests_the_repo_and_reports_what_it_got(tmp_path, monkeypatch):
    d = write_hub_layout(tmp_path)
    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return str(d)

    monkeypatch.setattr(hub, "_snapshot_download", fake)
    resolved = hub.resolve("cpool", revision="abc123")

    assert resolved.path == d
    assert resolved.repo == "brpoplpush/quantik-cpool-c191-b6"
    assert resolved.revision == "abc123"
    assert seen["repo_id"] == "brpoplpush/quantik-cpool-c191-b6"
    assert seen["revision"] == "abc123"
    # Neither runtime should pay to download the other's artifact, but both
    # must be requestable — the pattern list is what keeps that true.
    assert "model.safetensors" in seen["allow_patterns"]
    assert "model.onnx" in seen["allow_patterns"]


def test_resolve_defaults_to_main(tmp_path, monkeypatch) -> None:
    seen = {}
    monkeypatch.setattr(
        hub,
        "_snapshot_download",
        lambda **kw: (seen.update(kw), str(write_hub_layout(tmp_path)))[1],
    )
    assert hub.resolve("mlp").revision == "main"
    assert seen["revision"] == "main"


def test_load_evaluator_rejects_an_unknown_runtime(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(hub, "_snapshot_download", lambda **kw: str(tmp_path))
    with pytest.raises(ValueError, match="unknown runtime"):
        hub.load_evaluator("cpool", runtime="tensorflow")


def test_load_evaluator_verifies_before_it_loads(tmp_path, monkeypatch) -> None:
    # Ordering matters: a bad digest must be reported as a bad digest, not as
    # whatever torch says when it is handed half a tensor file.
    d = write_hub_layout(tmp_path, digest="sha256:0000")
    monkeypatch.setattr(hub, "_snapshot_download", lambda **kw: str(d))
    with pytest.raises(ValueError, match="digest"):
        hub.load_evaluator("cpool")


def test_the_missing_extra_is_named_in_the_import_error(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "huggingface_hub":
            raise ImportError("no module named huggingface_hub")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match=r"quantik-models\[hub\]"):
        hub._snapshot_download(repo_id="x")


def test_hub_imports_without_torch() -> None:
    # `hub` sits at package top level, so a module-scope torch import here
    # would break the torch-free install the e2e workflow tests.
    import subprocess
    import sys

    code = (
        "import sys;"
        "sys.modules['torch'] = None;"
        "import quantik_models.hub as h;"
        "assert h.repo_id('cpool').endswith('quantik-cpool-c191-b6')"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
