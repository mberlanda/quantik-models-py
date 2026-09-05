"""Load the published Quantik policy/value networks from the Hugging Face Hub.

Installed from PyPI, this package has no `runs/` directory: every checkpoint
this project trained lives on the Hub, not in the wheel. Weights are ~7 MB
each and carry a different licence from the code (CC-BY-NC-4.0 against the
package's MIT), so bundling them would be wrong on both counts. This module
is the supported way to get them.

    from quantik_models import hub

    evaluator = hub.load_evaluator("cpool")        # torch
    evaluator = hub.load_evaluator("cpool", runtime="onnx")   # no torch

Three decisions worth knowing, because each has a plausible alternative:

* **The four repo ids are a table here, not a Hub query.** Listing the
  `brpoplpush` namespace would return whatever exists today — including a
  half-staged repo, or a fifth architecture whose numbers are not in this
  package's documentation. A published short name is an API surface, so it
  resolves the same way offline, rate-limited, or a year from now.
* **`revision` defaults to `main` and is recorded, not pinned.** Pinning the
  commit shas into the library would make every weight update require a
  package release, and the two version streams are genuinely independent.
  The cost is that `main` can move under a caller, so `resolve()` returns the
  revision it actually got and `docs/models.md` says to pin it for anything
  whose result is being reported.
* **The downloaded weights are digest-checked against the manifest by
  default.** `manifest.json` carries `weights_hash`, verification is one
  sha256 over 7 MB, and the failure it catches — a truncated or partial
  download presenting as a corrupt state dict — is otherwise diagnosed as a
  model bug.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "PUBLISHED",
    "NAMESPACE",
    "PublishedModel",
    "repo_id",
    "resolve",
    "load_evaluator",
    "verify",
]

# The Hub account the family is published under. Mirrors
# `export.huggingface.DEFAULT_NAMESPACE`; kept as its own constant because
# that module is the *staging* side and this one is the consuming side —
# a reader here should not have to import the exporter to learn where the
# weights are.
NAMESPACE = "brpoplpush"


@dataclass(frozen=True)
class PublishedModel:
    """One published architecture: its short name and where it lives."""

    name: str
    """The short name callers pass — also the architecture in the registry."""

    architecture: str
    """The full `arch-cCHANNELS-bBLOCKS` id, as stamped in the manifest."""

    repo: str
    """The Hub repo id, `<namespace>/<repo name>`."""

    summary: str
    """One line on what this architecture is for. Full write-ups in `docs/`."""

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}"


# Ordered strongest-first by the arena, which is the ranking this project
# trusts — see docs/models.md. Held-out accuracy gives a different order and
# has failed to predict play strength; do not re-sort by it.
PUBLISHED: dict[str, PublishedModel] = {
    m.name: m
    for m in (
        PublishedModel(
            "cpool",
            "cpool-c191-b6",
            f"{NAMESPACE}/quantik-cpool-c191-b6",
            "Constraint-pooling network. Strongest of the four; the default.",
        ),
        PublishedModel(
            "attn",
            "attn-d192-b6",
            f"{NAMESPACE}/quantik-attn-d192-b6",
            "Transformer encoder over the 16 cells, told nothing about groups.",
        ),
        PublishedModel(
            "resnet",
            "resnet-c128-b6",
            f"{NAMESPACE}/quantik-resnet-c128-b6",
            "Convolutional residual trunk. The incumbent baseline.",
        ),
        PublishedModel(
            "mlp",
            "mlp-h455-b4",
            f"{NAMESPACE}/quantik-mlp-h455-b4",
            "Dense control that discards spatial structure. The falsifier.",
        ),
    )
}


def repo_id(name: str) -> str:
    """Hub repo id for a short name, or the argument if it already is one.

    Passing a full `owner/repo` through unchanged is deliberate: a fork or a
    privately retrained model should not need a code change here to be
    loadable, and `PUBLISHED` is a convenience table rather than a whitelist.
    """
    model = PUBLISHED.get(name)
    if model is not None:
        return model.repo
    if "/" in name:
        return name
    raise KeyError(
        f"unknown model {name!r}: expected one of {', '.join(PUBLISHED)}, "
        "or a full '<owner>/<repo>' Hub id"
    )


def _snapshot_download(**kwargs: Any) -> str:
    """Import `huggingface_hub` late, and explain the extra when it is absent.

    The dependency is in the `hub` extra rather than the base install because
    nothing in training, evaluation or the arena needs it — only this module
    does, and only when it is called. The bare ImportError names a package
    but not the way to get it.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(
            "loading published weights needs huggingface_hub: "
            "pip install 'quantik-models[hub]'"
        ) from exc
    return snapshot_download(**kwargs)


@dataclass(frozen=True)
class Resolved:
    """A downloaded checkpoint directory, and exactly which one it is."""

    path: Path
    repo: str
    revision: str
    """The revision as requested. `main` here means "whatever main was"."""


def resolve(
    name: str,
    *,
    revision: str = "main",
    cache_dir: str | Path | None = None,
) -> Resolved:
    """Download a published checkpoint and return the local directory.

    Only the files a checkpoint needs are fetched — `snapshot_download`'s
    `allow_patterns` keeps the ~7 MB ONNX graph off the wire for a torch
    caller and the ~7 MB safetensors off it for an ONNX one, so neither
    runtime pays for the other.
    """
    target = repo_id(name)
    path = _snapshot_download(
        repo_id=target,
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        allow_patterns=[
            "manifest.json",
            "config.json",
            "model.safetensors",
            "model.onnx",
            "training-report.json",
        ],
    )
    return Resolved(path=Path(path), repo=target, revision=revision)


def verify(checkpoint: str | Path) -> None:
    """Check the weights against the digest the manifest carries.

    Raises `ValueError` naming both digests. This is the same check
    `export.huggingface.verify_staged` runs before a directory is published
    and the play service runs before it serves a move; running it after a
    download closes the loop at the other end.
    """
    import json

    from .arena.registry import weights_path
    from .export.digest import file_digest

    path = Path(checkpoint)
    manifest = json.loads((path / "manifest.json").read_text())
    expected = manifest.get("weights_hash")
    if expected is None:
        return
    actual = file_digest(weights_path(path))
    if actual != expected:
        raise ValueError(
            f"weights in {path} digest {actual}, but manifest.json claims "
            f"{expected} — the download is incomplete or the files are mixed "
            "from two revisions"
        )


def load_evaluator(
    name: str = "cpool",
    *,
    device: str = "cpu",
    runtime: str = "torch",
    revision: str = "main",
    batch_size: int = 4096,
    check_digest: bool = True,
    cache_dir: str | Path | None = None,
):
    """Download a published model and return an evaluator ready to play.

    `runtime="onnx"` needs neither torch nor a GPU — it runs the `model.onnx`
    graph every published repo ships, through the `serve` extra's onnxruntime
    (80 MB against torch's 529 MB). The two runtimes are checked against each
    other in `tests/test_onnx_evaluator_agreement.py`; they are two ways of
    running one set of weights, not two models.

    Legality masking is applied inside the evaluator, the same code path
    training uses, so the returned policy can never favour an illegal move.
    """
    from .arena.registry import load_evaluator as _load_local
    from .arena.registry import load_onnx_evaluator as _load_onnx

    resolved = resolve(name, revision=revision, cache_dir=cache_dir)
    if check_digest and runtime == "torch":
        verify(resolved.path)
    if runtime == "onnx":
        return _load_onnx(resolved.path, batch_size=batch_size)
    if runtime == "torch":
        return _load_local(resolved.path, device=device, batch_size=batch_size)
    raise ValueError(f"unknown runtime {runtime!r}: expected 'torch' or 'onnx'")
