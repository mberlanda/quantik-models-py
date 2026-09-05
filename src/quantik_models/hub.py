"""Load the published Quantik policy/value networks from the Hugging Face Hub.

Installed from PyPI, this package has no `runs/` directory: every checkpoint
this project trained lives on the Hub, not in the wheel. Weights are ~7 MB
each and carry a different licence from the code (CC-BY-NC-4.0 against the
package's MIT), so bundling them would be wrong on both counts. This module
is the supported way to get them.

    from quantik_models import hub

    evaluator = hub.load_evaluator("cpool")        # torch
    evaluator = hub.load_evaluator("cpool", runtime="onnx")   # no torch

The first call reaches the network; every later one is served from the
Hugging Face cache. To take a machine offline, fetch first:

    quantik-models-fetch --all        # or: python -m quantik_models.hub --all

Four decisions worth knowing, because each has a plausible alternative:

* **The four repo ids are a table here, not a Hub query.** Listing the
  `brpoplpush` namespace would return whatever exists today — including a
  half-staged repo, or a fifth architecture whose numbers are not in this
  package's documentation. A published short name is an API surface, so it
  resolves the same way offline, rate-limited, or a year from now.
* **`revision` defaults to `main` and is recorded, not pinned.** Pinning the
  commit shas into the library would make every weight update require a
  package release, and the two version streams are genuinely independent.
  The cost is that `main` can move under a caller, so `resolve()` reports the
  commit it actually got and `docs/models.md` says to pin it for anything
  whose result is being reported.
* **The downloaded weights are digest-checked against the manifest by
  default.** `manifest.json` carries `weights_hash`, verification is one
  sha256 over 7 MB, and the failure it catches — a truncated or partial
  download presenting as a corrupt state dict — is otherwise diagnosed as a
  model bug. A mismatch is re-fetched once before it is raised, because a
  cached half-file is otherwise permanently broken for that user.
* **Every fetch failure is re-raised as `HubError` naming the next step.**
  The underlying exceptions are precise but assume the reader knows the Hub:
  `LocalEntryNotFoundError` is what "you are offline and never downloaded
  this" looks like, and that is not a sentence a caller of this library
  should have to translate. The original is kept as `__cause__`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "PUBLISHED",
    "NAMESPACE",
    "HubError",
    "PublishedModel",
    "Resolved",
    "repo_id",
    "resolve",
    "prefetch",
    "load_evaluator",
    "verify",
]

# The Hub account the family is published under. Mirrors
# `export.huggingface.DEFAULT_NAMESPACE`; kept as its own constant because
# that module is the *staging* side and this one is the consuming side —
# a reader here should not have to import the exporter to learn where the
# weights are.
NAMESPACE = "brpoplpush"

# Only what a checkpoint needs. `snapshot_download` fetches the whole repo
# otherwise, and these repos carry both runtimes' artifacts.
_ALLOW_PATTERNS = [
    "manifest.json",
    "config.json",
    "model.safetensors",
    "model.onnx",
    "training-report.json",
]

# What each runtime loads, in the order a directory may name it. `torch`
# accepts both because `export.checkpoint` writes the first name and
# `export.huggingface.stage` renames it to the second.
_ARTIFACTS: dict[str, tuple[str, ...]] = {
    "torch": ("weights.safetensors", "model.safetensors"),
    "onnx": ("model.onnx",),
}


class HubError(RuntimeError):
    """A published checkpoint could not be fetched, with the reason.

    One type rather than a hierarchy: every case has the same recovery shape
    — read the message, do the one thing it names — and callers that want to
    discriminate can still reach the Hub's own exception through `__cause__`.
    """


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


def _cache_location(cache_dir: str | Path | None) -> str:
    """Where a fetch would have put the files, for an error message.

    Named rather than described: "the Hugging Face cache" is not something a
    reader can `ls`, and the location moves with `HF_HOME`.
    """
    if cache_dir is not None:
        return str(cache_dir)
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        return str(HF_HUB_CACHE)
    except Exception:  # pragma: no cover - only if the constant is renamed
        return "~/.cache/huggingface/hub"


def _explain(
    exc: Exception, *, name: str, repo: str, revision: str, cache_dir: str | Path | None
) -> HubError:
    """Turn a Hub exception into the one sentence that unblocks the caller.

    Subclass order matters here: `GatedRepoError` is a `RepositoryNotFound`
    and `LocalEntryNotFoundError` is a `FileNotFoundError`, so the specific
    cases have to be tested before the general ones.
    """
    from huggingface_hub import errors as hf

    url = f"https://huggingface.co/{repo}"

    if isinstance(exc, hf.GatedRepoError):
        return HubError(
            f"{repo} is gated. Accept its terms at {url}, then authenticate "
            "with `hf auth login`."
        )
    if isinstance(exc, hf.RevisionNotFoundError):
        return HubError(
            f"{repo} has no revision {revision!r}. A revision is a branch, a "
            f"tag, or a commit sha — the repo's are listed at {url}/commits/main."
        )
    if isinstance(exc, (hf.RepositoryNotFoundError, hf.DisabledRepoError)):
        return HubError(
            f"no model repository at {url}. The published models are "
            f"{', '.join(PUBLISHED)}; a private or renamed repo also needs "
            "`hf auth login`."
        )
    if isinstance(exc, hf.HFValidationError):
        return HubError(f"{repo!r} is not a valid Hub id: {exc}")
    if isinstance(exc, (hf.LocalEntryNotFoundError, hf.OfflineModeIsEnabled)):
        # The Hub is unreachable *and* the cache is empty. Everything else is
        # recoverable by retrying; this one needs the caller to be online at
        # some point, so say when and with what.
        return HubError(
            f"cannot reach the Hugging Face Hub, and no copy of {repo} is "
            f"cached in {_cache_location(cache_dir)}. Fetch it once from a "
            f"connected machine with `quantik-models-fetch {name}`, or point "
            "HF_HOME at a cache that already has it."
        )
    if isinstance(exc, hf.HfHubHTTPError):
        return HubError(
            f"the Hub refused the request for {repo}: {exc}. If this is a "
            "rate limit it clears on its own; `hf auth login` raises the "
            "limit for authenticated callers."
        )
    if isinstance(exc, OSError):
        # Connection reset, DNS failure, a full disk mid-write. The cache is
        # the fallback the Hub itself already tried, so there is nothing left
        # to try here — but the message should not read as a bug in Quantik.
        return HubError(f"fetching {repo} from the Hub failed: {exc}")
    return HubError(f"fetching {repo} from the Hub failed: {exc!r}")


@dataclass(frozen=True)
class Resolved:
    """A downloaded checkpoint directory, and exactly which one it is."""

    path: Path
    repo: str
    revision: str
    """The revision as requested. `main` here means "whatever main was"."""

    commit: str | None = None
    """The commit `revision` actually resolved to, when it can be read back.

    This is what makes a `main` download reportable: pass it as `revision`
    later and the same bytes come back. `None` for a `local_dir` layout that
    does not carry the sha in its path.
    """


def _commit_of(path: Path) -> str | None:
    """Read the resolved sha out of the cache path, which ends `snapshots/<sha>`.

    Derived from the path rather than a second API call: the call would need
    the network the download may not have had, and would answer about *now*
    rather than about the files just returned.
    """
    parent = path.parent.name
    name = path.name
    if parent == "snapshots" and len(name) == 40 and all(c in "0123456789abcdef" for c in name):
        return name
    return None


def resolve(
    name: str,
    *,
    revision: str = "main",
    cache_dir: str | Path | None = None,
    force_download: bool = False,
) -> Resolved:
    """Download a published checkpoint and return the local directory.

    Only the files a checkpoint needs are fetched — `snapshot_download`'s
    `allow_patterns` keeps the ~7 MB ONNX graph off the wire for a torch
    caller and the ~7 MB safetensors off it for an ONNX one, so neither
    runtime pays for the other.

    Already-cached files are not re-fetched, and if the Hub is unreachable
    but the cache holds the revision, the cached copy is used. Raises
    `HubError` when neither is possible.
    """
    target = repo_id(name)
    try:
        path = _snapshot_download(
            repo_id=target,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            force_download=force_download,
            allow_patterns=_ALLOW_PATTERNS,
        )
    except ImportError:
        # The missing extra, already explained by `_snapshot_download`.
        raise
    except Exception as exc:
        raise _explain(
            exc, name=name, repo=target, revision=revision, cache_dir=cache_dir
        ) from exc
    local = Path(path)
    return Resolved(
        path=local, repo=target, revision=revision, commit=_commit_of(local)
    )


def prefetch(
    names: list[str] | None = None,
    *,
    revision: str = "main",
    cache_dir: str | Path | None = None,
) -> list[Resolved]:
    """Download checkpoints without loading them — the offline preparation step.

    Loading needs torch or onnxruntime; filling a cache needs neither, so an
    air-gapped or container build can populate `HF_HOME` from a machine that
    has nothing else installed.
    """
    return [
        resolve(name, revision=revision, cache_dir=cache_dir)
        for name in (names if names else list(PUBLISHED))
    ]


def artifact_path(checkpoint: str | Path, *, runtime: str = "torch") -> Path:
    """The file `runtime` will load out of a checkpoint directory.

    Raises `FileNotFoundError` naming the directory and the candidates. The
    alternative is what the ONNX path used to do: hand a missing filename to
    onnxruntime and let it report a protobuf parse failure.
    """
    _check_runtime(runtime)
    path = Path(checkpoint)
    candidates = _ARTIFACTS[runtime]
    for filename in candidates:
        candidate = path / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"no {runtime} artifact in {path}: expected one of "
        f"{', '.join(candidates)}. A checkpoint staged for the Hub carries "
        "both runtimes; a directory with neither is not a checkpoint."
    )


def _check_runtime(runtime: str) -> None:
    if runtime not in _ARTIFACTS:
        raise ValueError(f"unknown runtime {runtime!r}: expected 'torch' or 'onnx'")


def verify(checkpoint: str | Path, *, runtime: str = "torch") -> None:
    """Check the artifact a runtime will load against the manifest's digest.

    `runtime` selects *which* artifact, and that is the whole point: a
    checkpoint carries `weights_hash` for `model.safetensors` and
    `onnx_hash` for `model.onnx`, and checking the one the caller is not
    about to load is a check that cannot fail usefully.

    Raises `ValueError` naming both digests. This is the same comparison
    `export.huggingface.verify_staged` runs before a directory is published
    and the play service runs before it serves a move; running it after a
    download closes the loop at the other end.
    """
    import json

    from .export.digest import file_digest

    path = Path(checkpoint)
    target = artifact_path(path, runtime=runtime)

    manifest_file = path / "manifest.json"
    if not manifest_file.exists():
        raise FileNotFoundError(
            f"no manifest.json in {path}: a checkpoint directory without one "
            "cannot be checked, and cannot be loaded either"
        )
    manifest = json.loads(manifest_file.read_text())
    expected = manifest.get("onnx_hash" if runtime == "onnx" else "weights_hash")

    # Older manifests predate these fields. Refusing them would make the
    # loader stricter than the checkpoints it has to be able to read.
    if expected is None:
        return
    actual = file_digest(target)
    if actual != expected:
        raise ValueError(
            f"{target.name} in {path} digests {actual}, but manifest.json "
            f"claims {expected} — the download is incomplete or the files "
            "are mixed from two revisions"
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

    A digest mismatch is retried once with a forced re-download before it is
    raised: the overwhelmingly likely cause is a truncated cache entry, and
    that entry would otherwise fail identically on every future call with no
    hint that deleting it is the fix.
    """
    from .arena.registry import load_evaluator as _load_local
    from .arena.registry import load_onnx_evaluator as _load_onnx

    _check_runtime(runtime)

    resolved = resolve(name, revision=revision, cache_dir=cache_dir)
    if check_digest:
        try:
            verify(resolved.path, runtime=runtime)
        except ValueError:
            resolved = resolve(
                name, revision=revision, cache_dir=cache_dir, force_download=True
            )
            try:
                verify(resolved.path, runtime=runtime)
            except ValueError as exc:
                raise HubError(
                    f"{resolved.repo} still fails its manifest digest after a "
                    f"forced re-download into {_cache_location(cache_dir)}: "
                    f"{exc}. This is a bad file on the Hub or a cache that "
                    "cannot be written; neither is fixed by retrying."
                ) from exc
    else:
        # Even unverified, the file has to be there — and this is where a
        # torch-only fork of a repo stops being a mystery.
        artifact_path(resolved.path, runtime=runtime)

    if runtime == "onnx":
        return _load_onnx(resolved.path, batch_size=batch_size)
    return _load_local(resolved.path, device=device, batch_size=batch_size)


def main(argv: list[str] | None = None) -> int:
    """`quantik-models-fetch` — fill the cache so later calls work offline."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="quantik-models-fetch",
        description=(
            "Download published Quantik checkpoints into the Hugging Face "
            "cache. Needs no torch and loads nothing; run it before going "
            "offline, or in a container build."
        ),
    )
    parser.add_argument(
        "models",
        nargs="*",
        metavar="MODEL",
        help=f"short names ({', '.join(PUBLISHED)}) or full <owner>/<repo> ids",
    )
    parser.add_argument("--all", action="store_true", help="every published model")
    parser.add_argument("--revision", default="main", help="branch, tag or commit sha")
    parser.add_argument("--cache-dir", default=None, help="override HF_HOME's cache")
    args = parser.parse_args(argv)

    if not args.models and not args.all:
        parser.error("name at least one model, or pass --all")

    names = list(PUBLISHED) if args.all else args.models
    try:
        resolved = prefetch(names, revision=args.revision, cache_dir=args.cache_dir)
    except (HubError, KeyError, ImportError) as exc:
        # The message is the product here; a traceback through
        # huggingface_hub is not something the reader can act on.
        print(f"error: {exc}", flush=True)
        return 1

    for item in resolved:
        print(f"{item.repo}@{item.commit or item.revision} -> {item.path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    raise SystemExit(main())
