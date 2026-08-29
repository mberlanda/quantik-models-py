"""Discover playable checkpoints under a models directory.

`scan_models` is the boundary between "whatever is on disk" and the local
play service: one subdirectory per model, holding a `manifest.json` and a
`weights.safetensors`, exactly what `scripts/stage_hub_repos.sh` writes and
what `runs/train/<name>/best` already is. The intended way to populate a
models directory is a directory of symlinks pointing at chosen `best`
checkpoints, so every check below runs against the resolved target — a
`Path` that reaches its manifest through a symlinked parent is not treated
any differently than one that does not.

Every rejection is returned as a `refused` `PlayModel` with a specific
`reason`, never dropped from the list. A model missing from a dropdown
looks like it was never trained; a model greyed out with "weights hash
does not match manifest" tells the person what to go fix.

`model_id` is the directory name, not `manifest["model_id"]`. The
directory is what a person chose when they staged the model and what a
game record will store, so it has to be stable and human-chosen; the
manifest's id is an artifact identity such as `v3-cpool-best` that can
differ from run to run of the same architecture. Both are kept —
`manifest_model_id` — so nothing is lost.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..export.digest import file_digest

_MANIFEST_NAME = "manifest.json"
_WEIGHTS_NAME = "weights.safetensors"
_SCHEMA = "model-checkpoint.v1"


@dataclass(frozen=True)
class PlayModel:
    model_id: str
    path: Path
    status: str
    reason: str | None
    architecture: str | None
    manifest_model_id: str | None
    parameter_count: int | None
    weights_hash: str | None


def _refused(
    model_id: str,
    path: Path,
    reason: str,
    manifest: dict[str, Any] | None = None,
    weights_hash: str | None = None,
) -> PlayModel:
    manifest = manifest or {}
    return PlayModel(
        model_id=model_id,
        path=path,
        status="refused",
        reason=reason,
        architecture=manifest.get("architecture"),
        manifest_model_id=manifest.get("model_id"),
        parameter_count=manifest.get("parameter_count"),
        weights_hash=weights_hash,
    )


def _scan_one(model_dir: Path) -> PlayModel:
    model_id = model_dir.name
    manifest_path = model_dir / _MANIFEST_NAME
    if not manifest_path.is_file():
        return _refused(model_id, model_dir, "no manifest.json")

    try:
        manifest: dict[str, Any] = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        return _refused(model_id, model_dir, f"manifest.json does not parse: {exc}")

    schema = manifest.get("schema")
    if schema != _SCHEMA:
        return _refused(
            model_id, model_dir, f"manifest schema is {schema!r}, expected {_SCHEMA!r}", manifest
        )

    weights_path = model_dir / _WEIGHTS_NAME
    if not weights_path.is_file():
        return _refused(model_id, model_dir, "weights.safetensors missing", manifest)

    actual_hash = file_digest(weights_path)
    expected_hash = manifest.get("weights_hash")
    if actual_hash != expected_hash:
        return _refused(
            model_id,
            model_dir,
            f"weights hash {actual_hash} does not match manifest {expected_hash!r}",
            manifest,
            weights_hash=actual_hash,
        )

    # Mirrors `arena.registry._model_from_manifest` exactly: that is the
    # function that would otherwise raise, mid-game, the first time this
    # model is asked to move. Refusing at scan time turns that into a
    # startup message instead of a 500 on someone's first click.
    architecture = manifest.get("architecture")
    if manifest.get("architecture_spec") is None and not (architecture or "").startswith(
        "resnet-c"
    ):
        return _refused(
            model_id,
            model_dir,
            f"no architecture_spec and architecture {architecture!r} cannot be "
            "rebuilt from the legacy naming convention",
            manifest,
            weights_hash=actual_hash,
        )

    return PlayModel(
        model_id=model_id,
        path=model_dir,
        status="ready",
        reason=None,
        architecture=architecture,
        manifest_model_id=manifest.get("model_id"),
        parameter_count=manifest.get("parameter_count"),
        weights_hash=actual_hash,
    )


def scan_models(models_dir: Path) -> list[PlayModel]:
    """One `PlayModel` per subdirectory of `models_dir`, sorted by id.

    `Path.iterdir` followed by `Path.is_dir` already resolves through a
    symlinked entry the same way it resolves a real directory, so a
    directory of symlinks to `runs/train/<name>/best` needs no special
    handling here.
    """
    if not models_dir.is_dir():
        return []
    return sorted(
        (_scan_one(child) for child in models_dir.iterdir() if child.is_dir()),
        key=lambda model: model.model_id,
    )


def main(argv: list[str] | None = None) -> int:
    from .opponents import roster

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    models = scan_models(args.models_dir)
    print(f"models under {args.models_dir}:")
    for model in models:
        if model.status == "ready":
            print(f"  [ready]    {model.model_id}  ({model.architecture}, {model.parameter_count} params)")
        else:
            print(f"  [refused]  {model.model_id}  {model.reason}")

    print()
    print("opponents:")
    for opponent in roster(models):
        print(f"  {opponent.opponent_id:24s} {opponent.label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
