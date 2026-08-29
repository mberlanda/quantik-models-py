"""Stage a checkpoint as a Hugging Face model repository.

A `runs/train/*/best` directory is already most of a model repo — safetensors
weights, an ONNX graph, and a manifest carrying the hashes and the
architecture spec. What it is missing is the three things the Hub treats as
structural rather than decorative:

* **`README.md` with YAML front matter.** On the Hub the front matter is not
  documentation, it is metadata: `license` gates the download button,
  `library_name` and `pipeline_tag` decide which widget and which snippet
  the page offers, and `model-index` is what puts a number on the model's
  card and in search. A card without front matter publishes fine and is
  close to unfindable.
* **`config.json`.** Nothing on the Hub can reconstruct this architecture
  from weights alone — there is no `AutoModel` for it — so the file exists
  to make the spec readable without loading anything, and to name the code
  that consumes it.
* **`.gitattributes`.** Weights must be LFS-tracked *before* the first
  commit that contains them. A `.safetensors` committed as a plain blob is
  in the repo's history for good, and the fix is a rewrite, not a follow-up
  commit.

Everything here writes files. Nothing here uploads: pushing is a separate,
authenticated, hard-to-undo step that belongs in the user's hands, and a
function that both prepares and publishes makes the dry run impossible.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .digest import file_digest

__all__ = [
    "CARD_FILES",
    "LFS_PATTERNS",
    "file_digest",
    "gitattributes",
    "hf_config",
    "model_card",
    "stage",
    "card_metrics",
    "verify_staged",
]

# Tracked as LFS from the first commit. `*.onnx` is not in the Hub's default
# .gitattributes, which is the trap: safetensors is handled for you and the
# ONNX graph beside it silently is not.
LFS_PATTERNS = ("*.safetensors", "*.onnx", "*.npz")

CARD_FILES = ("README.md", "config.json", ".gitattributes")


def gitattributes(patterns: tuple[str, ...] = LFS_PATTERNS) -> str:
    return "\n".join(f"{p} filter=lfs diff=lfs merge=lfs -text" for p in patterns) + "\n"


def hf_config(manifest: dict[str, Any]) -> dict[str, Any]:
    """`config.json`, derived from the checkpoint manifest.

    Deliberately not a copy of the manifest: the manifest is this project's
    record and stays in the repo unchanged, while this is the Hub-facing
    view. `auto_map` is absent on purpose — there is no trust_remote_code
    path here, and advertising one that does not exist is worse than
    advertising none.
    """
    spec = manifest.get("architecture_spec")
    if not spec:
        raise ValueError(
            "checkpoint manifest has no `architecture_spec`; re-export it "
            "with a current quantik-models before staging it for the Hub"
        )
    return {
        "model_type": "quantik-policy-value",
        "architecture": manifest["architecture"],
        "arch": spec["arch"],
        "config": spec["config"],
        "parameter_count": manifest["parameter_count"],
        "contract_version": manifest["contract_version"],
        "input_contracts": manifest["input_contracts"],
        "output_contract": manifest["output_contract"],
        "legal_action_mask_required": manifest["legal_action_mask_required"],
        "onnx_opset": manifest.get("onnx_opset"),
    }


def _front_matter(
    manifest: dict[str, Any],
    license_id: str,
    metrics: list[dict[str, Any]],
    base_model: str | None,
) -> str:
    lines = [
        "---",
        f"license: {license_id}",
        "library_name: quantik-models",
        "pipeline_tag: reinforcement-learning",
        "tags:",
        "  - quantik",
        "  - board-games",
        "  - policy-value-network",
        f"  - {manifest['architecture_spec']['arch']}",
        "  - onnx",
    ]
    if base_model:
        lines += ["base_model:", f"  - {base_model}"]
    if metrics:
        lines += [
            "model-index:",
            f"  - name: {manifest['architecture']}",
            "    results:",
            "      - task:",
            "          type: reinforcement-learning",
            "          name: Quantik optimal-move prediction",
            "        dataset:",
            "          type: quantik-exact-solutions",
            "          name: exact-sampled",
            "        metrics:",
        ]
        for metric in metrics:
            lines += [
                f"          - type: {metric['type']}",
                f"            name: {metric['name']}",
                f"            value: {metric['value']}",
            ]
    lines.append("---")
    return "\n".join(lines)


def model_card(
    manifest: dict[str, Any],
    *,
    repo_id: str | None = None,
    license_id: str = "apache-2.0",
    metrics: list[dict[str, Any]] | None = None,
    base_model: str | None = None,
    body: str = "",
) -> str:
    """`README.md`: Hub metadata, then whatever prose the caller supplies.

    The generated part is only the part that has to be exact — the hashes,
    the shapes, the contract, the masking requirement. The argument for a
    model, and what it is not good at, is written by a person.
    """
    metrics = metrics or []
    # The snippets are meant to be copied and run, so the placeholder has to
    # look like a placeholder rather than like a repo that exists.
    repo_id = repo_id or f"<your-org>/{manifest['architecture']}"
    header = _front_matter(manifest, license_id, metrics, base_model)
    facts = [
        "",
        f"# {manifest['architecture']}",
        "",
        f"A Quantik policy/value network, {manifest['parameter_count']:,} parameters.",
        "",
        "## The contract",
        "",
        "```",
        "input   (B, 9, 4, 4) float32      tensor-board.v1, mover-relative",
        "output  (B, 64) policy logits     action_index = shape * 16 + position",
        "        (B,)    value in [-1, 1]  +1 = good for the side to move",
        "```",
        "",
        "**Legality masking happens outside this model.** It emits logits over "
        "all 64 actions, including illegal ones, and applying the legal-move "
        "mask before the softmax is the caller's job. The rules are exact in "
        "`quantik-core`, so the network is never asked to approximate them — "
        "an unmasked argmax from this model will play illegal moves.",
        "",
        "## Files",
        "",
        f"- `model.safetensors` — `{manifest['weights_hash']}`",
    ]
    if manifest.get("onnx_export"):
        facts.append(
            f"- `model.onnx` — opset {manifest.get('onnx_opset')}, "
            f"`{manifest['onnx_hash']}`, dynamic batch dimension"
        )
    facts += [
        "- `manifest.json` — the `model-checkpoint.v1` record this repo was staged from",
        "",
        f"Contract version `{manifest['contract_version']}`. Exported "
        f"{manifest['created_at']}.",
        "",
        "## Using it",
        "",
        "There is no `AutoModel` for this architecture. Load the weights "
        "through `quantik-models`, which reads `manifest.json` and rebuilds "
        "the network from `architecture_spec`:",
        "",
        "```python",
        "from huggingface_hub import snapshot_download",
        "from quantik_models.arena.registry import load_evaluator",
        "",
        f'evaluator = load_evaluator(snapshot_download("{repo_id}"), "cpu")',
        "```",
        "",
    ]
    if manifest.get("onnx_export"):
        facts += [
            "Or run the ONNX graph, which needs neither this package nor torch:",
            "",
            "```python",
            "import numpy as np, onnxruntime as ort",
            "from huggingface_hub import hf_hub_download",
            "",
            f'session = ort.InferenceSession(hf_hub_download("{repo_id}", "model.onnx"))',
            'policy, value = session.run(None, {"board": tensors.astype(np.float32)})',
            "# then mask: policy[~legal] = -inf before any softmax or argmax",
            "```",
            "",
        ]
    return header + "\n".join(facts) + ("\n" + body.strip() + "\n" if body.strip() else "")


def verify_staged(out_dir: Path) -> dict[str, str]:
    """Recompute the hashes the card publishes; raise if any disagrees.

    Worth doing before a push rather than after: the card states a digest
    for every weight file, and a card whose digest does not match the file
    beside it is worse than a card with no digest — it invites a check that
    will fail for a reader who cannot tell a bad upload from a bad claim.
    """
    manifest = json.loads((out_dir / "manifest.json").read_text())
    checked = {}
    for filename, key in (("model.safetensors", "weights_hash"), ("model.onnx", "onnx_hash")):
        path = out_dir / filename
        if not path.exists():
            continue
        actual = file_digest(path)
        expected = manifest.get(key)
        if expected and actual != expected:
            raise ValueError(f"{filename}: manifest says {expected}, file is {actual}")
        checked[filename] = actual
    if not checked:
        raise ValueError(f"{out_dir} contains no weight file to verify")
    return checked


def card_metrics(
    shift: list[dict], leaderboard: list[dict], architecture: str, agent: str
) -> list[dict[str, Any]]:
    """The `model-index` entries, read out of the evaluation artifacts.

    Hand-typing three numbers onto a card is how a card ends up describing a
    checkpoint that was retrained after it was written. These come from the
    same files the docs are generated from.

    Held-out accuracy is reported split, not pooled: the corpus holds no
    positions at plies 4-5, so the shallow figure is the only one measuring
    generalisation, and a pooled number is dominated by the deep positions
    where every model is near perfect.
    """
    record = next((r for r in shift if r["architecture"] == architecture), None)
    if record is None:
        raise ValueError(
            f"{architecture!r} is not in the shift evaluation; it has "
            f"{[r['architecture'] for r in shift]}"
        )
    by_ply = record["by_ply"]

    def accuracy(plies: range) -> float:
        rows = [by_ply[str(p)] for p in plies if str(p) in by_ply]
        won = sum(r["won_positions"] for r in rows)
        return sum(r["correct"] for r in rows) / won if won else 0.0

    metrics: list[dict[str, Any]] = [
        {
            "type": "accuracy",
            "name": "Held-out optimal-move accuracy, plies 4-6",
            "value": round(accuracy(range(4, 7)), 4),
        },
        {
            "type": "accuracy",
            "name": "Held-out optimal-move accuracy, plies 7-12",
            "value": round(accuracy(range(7, 13)), 4),
        },
    ]
    row = next((r for r in leaderboard if r["agent"] == agent), None)
    if row is not None:
        metrics.append(
            {
                "type": "win_rate",
                "name": f"Arena win rate vs the field ({row['games']} games)",
                "value": round(row["win_rate"], 4),
            }
        )
    return metrics


def stage(
    checkpoint_dir: Path,
    out_dir: Path,
    *,
    repo_id: str | None = None,
    license_id: str = "apache-2.0",
    metrics: list[dict[str, Any]] | None = None,
    base_model: str | None = None,
    body: str = "",
) -> Path:
    """Write a Hub-ready directory. Copies, never moves, and never uploads."""
    manifest = json.loads((checkpoint_dir / "manifest.json").read_text())
    out_dir.mkdir(parents=True, exist_ok=True)

    # `weights.safetensors` -> `model.safetensors`: the Hub's viewers and
    # half its tooling look for the latter by name.
    shutil.copyfile(checkpoint_dir / "weights.safetensors", out_dir / "model.safetensors")
    for optional in ("model.onnx", "training-report.json"):
        source = checkpoint_dir / optional
        if source.exists():
            shutil.copyfile(source, out_dir / optional)
    shutil.copyfile(checkpoint_dir / "manifest.json", out_dir / "manifest.json")

    (out_dir / "config.json").write_text(json.dumps(hf_config(manifest), indent=2) + "\n")
    (out_dir / ".gitattributes").write_text(gitattributes())
    (out_dir / "README.md").write_text(
        model_card(
            manifest,
            repo_id=repo_id,
            license_id=license_id,
            metrics=metrics,
            base_model=base_model,
            body=body,
        )
    )
    verify_staged(out_dir)
    return out_dir


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path, help="a runs/train/*/best directory")
    parser.add_argument("out", type=Path, help="directory to stage into")
    parser.add_argument("--repo-id", default=None, help="e.g. mberlanda/quantik-cpool-c191-b6")
    parser.add_argument("--license", default="apache-2.0")
    parser.add_argument("--base-model", default=None)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="JSON list of {type, name, value} for the model-index block",
    )
    parser.add_argument(
        "--shift",
        type=Path,
        default=None,
        help="a shift.json to read the model-index accuracies from",
    )
    parser.add_argument(
        "--arena",
        type=Path,
        default=None,
        help="an arena games.json to read the model-index win rate from",
    )
    parser.add_argument(
        "--agent",
        default=None,
        help="the agent name this checkpoint played under in --arena",
    )
    parser.add_argument(
        "--body",
        type=Path,
        default=None,
        help="markdown appended after the generated facts",
    )
    args = parser.parse_args(argv)

    metrics = json.loads(args.metrics.read_text()) if args.metrics else None
    if metrics is None and args.shift:
        manifest = json.loads((args.checkpoint / "manifest.json").read_text())
        leaderboard = (
            json.loads(args.arena.read_text())["leaderboard"] if args.arena else []
        )
        metrics = card_metrics(
            json.loads(args.shift.read_text()),
            leaderboard,
            manifest["architecture"],
            args.agent or manifest["architecture"].split("-")[0],
        )

    out = stage(
        args.checkpoint,
        args.out,
        repo_id=args.repo_id,
        license_id=args.license,
        metrics=metrics,
        base_model=args.base_model,
        body=args.body.read_text() if args.body else "",
    )
    for path in sorted(out.iterdir()):
        print(f"  {path.name}  {path.stat().st_size:,} bytes")
    print("\nhashes verified against the manifest")
    print(f"\nstaged {out}")
    print("nothing has been uploaded; see docs/publishing-to-hugging-face.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
