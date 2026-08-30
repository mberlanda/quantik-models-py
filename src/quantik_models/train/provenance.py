"""What a run needs recorded to be reproducible after the fact.

`config.json` already records every hyperparameter and the seed, and
`supervised.py` is careful to record the *resolved* learning rate rather than
`None` — because "a config that says None does not reproduce the run it
describes". This module applies the same rule to everything else a rerun needs
and that `config.json` does not carry.

Four gaps, each of which has cost this project something:

* **The code.** Nothing recorded which commit trained a checkpoint, so "what
  did the trainer do at the time" was unanswerable for every run on disk.
* **The corpus identity.** `config.json` records a *filename*.
  `exact-sampled.npz` and `exact-sampled-v2.npz` are different files with
  confusable names, and mixing them up is exactly what produced the wrong
  conclusion corrected in `docs/corpus-v3.md`. A filename is not an identity;
  a hash is.
* **The hardware.** `device` is recorded as the string `"auto"` — the request,
  not what it resolved to. A run on MPS and a run on CPU are not the same run.
* **The dependency versions.** `torch`, `numpy` and `quantik-core` all change
  numerics. `quantik-core` additionally produces the labels.

Everything here is best-effort and never raises: provenance capture must not be
able to fail a training run. A field that could not be determined is recorded as
`null` with the reason beside it, which is more useful than an absent key.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

# src/quantik_models/train/provenance.py -> repository root
_REPO_ROOT = Path(__file__).resolve().parents[3]

_PACKAGES = ("torch", "numpy", "quantik-core", "quantik-models", "onnxruntime")


def _git(*args: str, root: Path) -> str | None:
    try:
        done = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() or None if done.returncode == 0 else None


def _commit_url(remote: str | None, commit: str | None) -> str | None:
    """A browsable permalink, so the recorded commit is reachable by a reader.

    Handles both `git@host:owner/repo.git` and `https://host/owner/repo.git`.
    Returns None rather than guessing for anything else — a wrong link is worse
    than no link, because the reader cannot tell it from a deleted commit.
    """
    if not remote or not commit:
        return None
    url = remote.removesuffix(".git")
    if url.startswith("git@"):
        host, _, path = url.partition(":")
        url = f"https://{host.removeprefix('git@')}/{path}"
    if not url.startswith("https://"):
        return None
    return f"{url}/commit/{commit}"


def code_provenance(root: Path = _REPO_ROOT) -> dict[str, Any]:
    """The commit that ran, and where to find it.

    `dirty` is not a footnote. A dirty tree means the recorded commit does
    **not** describe the code that ran, and no permalink can fix that.
    """
    commit = _git("rev-parse", "HEAD", root=root)
    if commit is None:
        return {"commit": None, "reason": f"not a git checkout: {root}"}
    status = _git("status", "--porcelain", root=root)
    remote = _git("config", "--get", "remote.origin.url", root=root)
    return {
        "commit": commit,
        "dirty": bool(status),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD", root=root),
        "remote": remote,
        "commit_url": _commit_url(remote, commit),
        "repository_root": str(root),
    }


def file_digest(path: Path) -> dict[str, Any]:
    """sha256 and size for an input file. Corpora are ~11 MB; this is free."""
    resolved = Path(path)
    if not resolved.is_file():
        return {"path": str(path), "sha256": None, "reason": "file not found"}
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return {
        "path": str(path),
        "resolved_path": str(resolved.resolve()),
        "sha256": f"sha256:{digest.hexdigest()}",
        "size_bytes": resolved.stat().st_size,
    }


def hardware_provenance(device: str | None = None) -> dict[str, Any]:
    """The machine, and the device actually used — not the one requested."""
    record: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "device": device,
    }
    try:  # torch is optional everywhere outside model/ and train/
        import torch
    except ImportError:
        record["accelerator"] = None
        record["reason"] = "torch not installed"
        return record
    if device and device.startswith("cuda") and torch.cuda.is_available():
        record["accelerator"] = torch.cuda.get_device_name(0)
    elif device == "mps" and torch.backends.mps.is_available():
        # torch exposes no product name for MPS; the machine string is the
        # only identifier available, and it is already recorded above.
        record["accelerator"] = f"mps ({platform.machine()})"
    else:
        record["accelerator"] = None
    return record


def version_provenance(packages: tuple[str, ...] = _PACKAGES) -> dict[str, Any]:
    versions: dict[str, Any] = {"python": sys.version.split()[0]}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def capture(*, corpus: Path | str | None = None, device: str | None = None) -> dict[str, Any]:
    """The full record. Never raises — see the module docstring."""
    record: dict[str, Any] = {
        "schema": "training-provenance.v1",
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }
    for key, produce in (
        ("code", lambda: code_provenance()),
        ("hardware", lambda: hardware_provenance(device)),
        ("versions", lambda: version_provenance()),
    ):
        try:
            record[key] = produce()
        except Exception as error:  # noqa: BLE001 - provenance must never fail a run
            record[key] = {"reason": f"{type(error).__name__}: {error}"}
    if corpus is not None:
        try:
            record["corpus"] = file_digest(Path(corpus))
        except Exception as error:  # noqa: BLE001
            record["corpus"] = {"path": str(corpus), "sha256": None, "reason": str(error)}
    return record
