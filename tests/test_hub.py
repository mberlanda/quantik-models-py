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
    # The model registry builds torch modules, so this one check cannot run on
    # the torch-free install — the rest of this file deliberately can.
    pytest.importorskip("torch")
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


def test_load_evaluator_rejects_an_unknown_runtime_before_downloading(
    tmp_path, monkeypatch
) -> None:
    # Refused before the download, not after: a typo should not cost 7 MB and
    # a Hub round trip to report.
    called = []
    monkeypatch.setattr(
        hub, "_snapshot_download", lambda **kw: called.append(kw) or str(tmp_path)
    )
    with pytest.raises(ValueError, match="unknown runtime"):
        hub.load_evaluator("cpool", runtime="tensorflow")
    assert called == []


def test_verify_checks_the_artifact_the_runtime_will_load(tmp_path) -> None:
    """`weights_hash` covers model.safetensors, `onnx_hash` covers model.onnx.

    Checking the one the caller is not about to load is a check that cannot
    fail usefully — which is what the ONNX path had before: no check at all.
    """
    from quantik_models.export.digest import file_digest

    d = write_hub_layout(tmp_path)
    (d / "model.onnx").write_bytes(b"graph")

    manifest = json.loads((d / "manifest.json").read_text())
    manifest["weights_hash"] = file_digest(d / "model.safetensors")
    manifest["onnx_hash"] = "sha256:wrong"
    (d / "manifest.json").write_text(json.dumps(manifest))

    hub.verify(d, runtime="torch")  # the safetensors digest is right
    with pytest.raises(ValueError, match="model.onnx"):
        hub.verify(d, runtime="onnx")


def test_load_evaluator_verifies_before_it_loads(tmp_path, monkeypatch) -> None:
    # Ordering matters: a bad digest must be reported as a bad digest, not as
    # whatever torch says when it is handed half a tensor file.
    d = write_hub_layout(tmp_path, digest="sha256:0000")
    monkeypatch.setattr(hub, "_snapshot_download", lambda **kw: str(d))
    loaded = []
    monkeypatch.setattr(
        "quantik_models.arena.registry.load_evaluator",
        lambda *a, **kw: loaded.append(a),
    )
    with pytest.raises(hub.HubError, match="digest"):
        hub.load_evaluator("cpool")
    assert loaded == []


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


# --- what a failed fetch tells the caller --------------------------------
#
# Every one of these was a traceback through `huggingface_hub` before. The
# exceptions were already precise; what they were not is actionable to
# someone who called `load_evaluator("cpool")` and has never heard of a
# snapshot. The assertions are on the *remedy* appearing in the message,
# because that is the part a reader acts on.


def _hf_errors():
    pytest.importorskip("huggingface_hub")
    from huggingface_hub import errors

    return errors


class _Response:
    """Enough of a response for the Hub's HTTP errors to construct.

    They want a real `httpx.Response` (a `requests.Response` in older
    versions) purely to read a couple of headers. Building one for a test
    about *message text* would tie these assertions to whichever HTTP client
    huggingface_hub happens to vendor this month.
    """

    status_code = 404
    headers: dict[str, str] = {}
    text = ""
    request = None
    url = "https://huggingface.co/"


def _http(cls, message):
    try:
        return cls(message, response=_Response())
    except TypeError:  # pragma: no cover - older huggingface_hub
        return cls(message)


def _explained(exc, *, name="cpool", revision="main", cache_dir=None):
    return str(
        hub._explain(
            exc,
            name=name,
            repo=hub.repo_id(name),
            revision=revision,
            cache_dir=cache_dir,
        )
    )


def test_being_offline_with_a_cold_cache_names_the_cache_and_the_fix() -> None:
    errors = _hf_errors()
    message = _explained(errors.LocalEntryNotFoundError("no internet"), cache_dir="/c")
    # The worst case: no network, nothing cached. Nothing can rescue the call,
    # so the message has to carry the whole recovery — where the files would
    # have gone, and the command that puts them there while online.
    assert "/c" in message
    assert "quantik-models-fetch cpool" in message


def test_a_gated_repo_says_accept_the_terms_not_repository_not_found() -> None:
    errors = _hf_errors()
    # GatedRepoError subclasses RepositoryNotFoundError, so a naive isinstance
    # ladder reports a licence gate as a missing repo and sends the reader
    # looking for a typo.
    message = _explained(_http(errors.GatedRepoError, "gated"))
    assert "Accept its terms" in message
    assert "hf auth login" in message


def test_a_missing_repo_lists_the_names_that_do_exist() -> None:
    errors = _hf_errors()
    message = _explained(_http(errors.RepositoryNotFoundError, "404"), name="brpoplpush/typo")
    assert "https://huggingface.co/brpoplpush/typo" in message
    for known in hub.PUBLISHED:
        assert known in message


def test_a_bad_revision_is_not_reported_as_a_bad_repo() -> None:
    errors = _hf_errors()
    message = _explained(_http(errors.RevisionNotFoundError, "404"), revision="v9")
    assert "'v9'" in message
    assert "commit sha" in message


def test_a_rate_limit_says_it_clears_on_its_own() -> None:
    errors = _hf_errors()
    message = _explained(_http(errors.HfHubHTTPError, "429 Too Many Requests"))
    assert "rate limit" in message


def test_an_unrecognised_failure_still_blames_the_hub_not_quantik() -> None:
    message = _explained(OSError("connection reset by peer"))
    assert "connection reset by peer" in message
    assert "from the Hub failed" in message


def test_resolve_reraises_as_hub_error_keeping_the_original_cause(monkeypatch) -> None:
    errors = _hf_errors()
    original = errors.LocalEntryNotFoundError("offline")

    def boom(**_kwargs):
        raise original

    monkeypatch.setattr(hub, "_snapshot_download", boom)
    with pytest.raises(hub.HubError) as excinfo:
        hub.resolve("cpool")
    # One catchable type for callers, without throwing away what actually
    # happened — a `hf` user can still branch on `__cause__`.
    assert excinfo.value.__cause__ is original


def test_the_missing_extra_survives_the_translation(monkeypatch) -> None:
    # ImportError is the one failure that must *not* become a HubError: it is
    # a packaging problem, and `pip install 'quantik-models[hub]'` is not a
    # sentence about the Hub being unreachable.
    def missing(**_kwargs):
        raise ImportError("pip install 'quantik-models[hub]'")

    monkeypatch.setattr(hub, "_snapshot_download", missing)
    with pytest.raises(ImportError, match=r"quantik-models\[hub\]"):
        hub.resolve("cpool")


# --- knowing which weights you got ---------------------------------------


def test_resolve_reports_the_commit_main_pointed_at(tmp_path, monkeypatch) -> None:
    sha = "a" * 40
    snapshot = tmp_path / "snapshots" / sha
    snapshot.mkdir(parents=True)
    monkeypatch.setattr(hub, "_snapshot_download", lambda **kw: str(snapshot))
    resolved = hub.resolve("cpool")
    # `revision="main"` is unreportable on its own — main moves. The commit
    # read back off the cache path is what a caller pins to reproduce a run.
    assert resolved.revision == "main"
    assert resolved.commit == sha


def test_a_path_without_a_sha_reports_no_commit_rather_than_a_guess(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(hub, "_snapshot_download", lambda **kw: str(tmp_path))
    assert hub.resolve("cpool").commit is None


# --- prefetch ------------------------------------------------------------


def test_prefetch_defaults_to_the_whole_family(tmp_path, monkeypatch) -> None:
    asked = []
    monkeypatch.setattr(
        hub,
        "_snapshot_download",
        lambda **kw: asked.append(kw["repo_id"]) or str(tmp_path),
    )
    hub.prefetch()
    assert asked == [m.repo for m in hub.PUBLISHED.values()]


def test_the_fetch_cli_prints_the_commit_and_exits_zero(tmp_path, monkeypatch, capsys):
    sha = "b" * 40
    snapshot = tmp_path / "snapshots" / sha
    snapshot.mkdir(parents=True)
    monkeypatch.setattr(hub, "_snapshot_download", lambda **kw: str(snapshot))
    assert hub.main(["cpool"]) == 0
    assert sha in capsys.readouterr().out


def test_the_fetch_cli_reports_a_failure_as_a_message_not_a_traceback(
    monkeypatch, capsys
) -> None:
    def boom(*_args, **_kwargs):
        raise hub.HubError("cannot reach the Hugging Face Hub")

    monkeypatch.setattr(hub, "resolve", boom)
    assert hub.main(["--all"]) == 1
    out = capsys.readouterr().out
    assert out.startswith("error: ")
    assert "Traceback" not in out


def test_the_fetch_cli_refuses_to_guess_what_to_download() -> None:
    with pytest.raises(SystemExit):
        hub.main([])


# --- the artifact a runtime will actually open ---------------------------


def test_a_missing_onnx_graph_is_named_before_onnxruntime_sees_it(tmp_path) -> None:
    d = write_hub_layout(tmp_path)  # safetensors only
    hub.artifact_path(d, runtime="torch")
    with pytest.raises(FileNotFoundError, match="model.onnx"):
        hub.artifact_path(d, runtime="onnx")


def test_the_torch_artifact_names_match_the_registry_loader() -> None:
    # Two lists of the same filenames, in two modules that cannot import each
    # other at module scope (registry pulls in torch; hub must not). This is
    # the test that keeps them from drifting.
    assert hub._ARTIFACTS["torch"] == registry._WEIGHT_FILENAMES


def test_a_directory_with_no_manifest_says_so(tmp_path) -> None:
    d = tmp_path / "half"
    d.mkdir()
    (d / "model.safetensors").write_bytes(b"w")
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        hub.verify(d)


def test_load_skips_the_digest_but_never_the_existence_check(
    tmp_path, monkeypatch
) -> None:
    d = write_hub_layout(tmp_path)  # no model.onnx
    monkeypatch.setattr(hub, "_snapshot_download", lambda **kw: str(d))
    with pytest.raises(FileNotFoundError, match="model.onnx"):
        hub.load_evaluator("cpool", runtime="onnx", check_digest=False)


# --- self-healing a truncated download -----------------------------------


def test_a_bad_digest_is_refetched_once_before_it_is_raised(
    tmp_path, monkeypatch
) -> None:
    from quantik_models.export.digest import file_digest

    good = write_hub_layout(tmp_path / "good", weights=b"whole weights")
    manifest = json.loads((good / "manifest.json").read_text())
    manifest["weights_hash"] = file_digest(good / "model.safetensors")
    (good / "manifest.json").write_text(json.dumps(manifest))
    truncated = write_hub_layout(tmp_path / "bad", weights=b"whole")
    (truncated / "manifest.json").write_text(json.dumps(manifest))

    calls = []

    def download(**kw):
        calls.append(kw["force_download"])
        return str(good if kw["force_download"] else truncated)

    monkeypatch.setattr(hub, "_snapshot_download", download)
    monkeypatch.setattr(
        hub, "load_evaluator", hub.load_evaluator
    )  # keep the real one explicit
    loaded = {}
    monkeypatch.setattr(
        "quantik_models.arena.registry.load_evaluator",
        lambda path, device="cpu", batch_size=4096: loaded.setdefault("path", path),
    )
    hub.load_evaluator("cpool")
    # A half-written cache entry fails identically forever otherwise, and the
    # remedy — delete a directory under HF_HOME — is not discoverable from a
    # safetensors parse error.
    assert calls == [False, True]
    assert loaded["path"] == good


def test_a_digest_that_survives_a_refetch_is_a_hard_error(tmp_path, monkeypatch):
    d = write_hub_layout(tmp_path, digest="sha256:0000")
    monkeypatch.setattr(hub, "_snapshot_download", lambda **kw: str(d))
    with pytest.raises(hub.HubError, match="forced re-download"):
        hub.load_evaluator("cpool")
