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

from .. import __version__ as _PACKAGE_VERSION
from . import cards
from .digest import file_digest

__all__ = [
    "CARD_FILES",
    "DEFAULT_LICENSE",
    "DEFAULT_NAMESPACE",
    "LFS_PATTERNS",
    "repo_id_for",
    "repo_name_for",
    "file_digest",
    "gitattributes",
    "hf_config",
    "model_card",
    "stage",
    "card_metrics",
    "run_config",
    "verify_staged",
]

# Tracked as LFS from the first commit. `*.onnx` is not in the Hub's default
# .gitattributes, which is the trap: safetensors is handled for you and the
# ONNX graph beside it silently is not.
LFS_PATTERNS = ("*.safetensors", "*.onnx", "*.npz")

CARD_FILES = ("README.md", "config.json", ".gitattributes")

# The Hub account these models are published under. Overridable per call and
# by QUANTIK_HF_NAMESPACE, but a default matters: a repo id assembled by hand
# on each invocation is how one model ends up under a different account than
# the rest of its family, and a Hub repo cannot be renamed without breaking
# every link and download that already points at it.
DEFAULT_NAMESPACE = "brpoplpush"

# Every model in this family carries the project prefix. On the Hub a repo
# name sits alone in search results with no directory around it, so
# `cpool-c191-b6` says nothing about what it is; `quantik-cpool-c191-b6`
# does, and it groups the family alphabetically for free.
REPO_PREFIX = "quantik"

# The Hub's `license:` field takes an identifier from a fixed table and it is
# lowercase — `mit`, not `MIT`. Every licensed repository in this workspace is
# MIT, and a card asserting a licence the source tree does not carry is worse
# than no card at all.
# CC BY-NC 4.0 for the *weights*. Deliberately not an OSI licence and the
# card says so: every OSI-approved licence permits royalty-free commercial
# use, which is the one thing this is meant to reserve. Free for research,
# teaching and non-commercial use with attribution; commercial use is by
# separate agreement. The *code* that produced these weights stays MIT — a
# split licence between code and model is normal and worth being explicit
# about, because a reader who sees MIT on the repository will otherwise
# assume it covers the download.
DEFAULT_LICENSE = "cc-by-nc-4.0"


def repo_name_for(architecture: str, prefix: str = REPO_PREFIX) -> str:
    """`cpool-c191-b6` -> `quantik-cpool-c191-b6`.

    Derived rather than chosen per model: the architecture string is already
    the thing that distinguishes these checkpoints, and a hand-written name
    is one more place for `c191` and `c192` to diverge.
    """
    if not architecture:
        raise ValueError("architecture is empty; cannot derive a repo name")
    name = architecture if architecture.startswith(f"{prefix}-") else f"{prefix}-{architecture}"
    # `huggingface_hub`'s own REPO_ID_REGEX allows word characters, `-` and
    # `.`, with the repo name 1-96 characters, and rejects `--`, `..` and a
    # trailing `.git`. That validator is the only authoritative constraint
    # found — the Hub docs state no naming rules — so this mirrors it rather
    # than inventing a stricter one. Everything the registry produces is
    # already lowercase alphanumeric-and-hyphen.
    if not all(c.isalnum() or c in "-_." for c in name):
        raise ValueError(f"{name!r} is not a usable Hub repo name")
    return name


def repo_id_for(
    architecture: str, namespace: str | None = None, prefix: str = REPO_PREFIX
) -> str:
    """`<namespace>/<repo name>`, e.g. `brpoplpush/quantik-cpool-c191-b6`."""
    import os

    namespace = namespace or os.environ.get("QUANTIK_HF_NAMESPACE") or DEFAULT_NAMESPACE
    if "/" in namespace:
        raise ValueError(f"namespace {namespace!r} must not contain a slash")
    return f"{namespace}/{repo_name_for(architecture, prefix)}"


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


def _results_table(metrics: list[dict[str, Any]]) -> list[str]:
    if not metrics:
        return []
    lines = ["| metric | value |", "|---|---|"]
    for metric in metrics:
        value = metric["value"]
        shown = f"{value:.1%}" if metric["type"] == "win_rate" else f"{value:.4f}"
        lines.append(f"| {metric['name']} | **{shown}** |")
    return lines


def _short_name(manifest: dict[str, Any]) -> str:
    """The registry name a caller passes to `hub.load_evaluator`.

    Prefers `architecture_spec.arch`, which records it outright. Falls back
    to the leading segment of `architecture` (`cpool-c191-b6` -> `cpool`) for
    checkpoints exported before that field existed.
    """
    spec = manifest.get("architecture_spec") or {}
    arch = spec.get("arch")
    return arch if arch else str(manifest["architecture"]).split("-", 1)[0]


def _install_lines(provenance: dict[str, Any] | None) -> list[str]:
    """How to install the code that loads these weights.

    The release is the line a reader should normally use, and it is first.
    The pinned source install stays underneath it because the two answer
    different questions: "how do I run this model" and "what exact code
    produced the numbers on this card". An unpinned `git+https://...` answers
    neither — it tracks `main`, so a reader following this card a month later
    gets code the card does not describe.
    """
    lines = [f"pip install 'quantik-models[torch,hub]>={_PACKAGE_VERSION}'"]
    commit = ((provenance or {}).get("code") or {}).get("commit")
    if commit:
        base = "git+https://github.com/mberlanda/quantik-models-py"
        lines += [
            "",
            "# Or the exact code that trained these weights:",
            f"# pip install 'quantik-models[torch,hub] @ {base}@{commit}'",
        ]
    return lines


def _provenance_section(provenance: dict[str, Any] | None) -> list[str]:
    """What is needed to reproduce the run, not just to describe it.

    The hyperparameter table above says what was asked for. This says what
    actually ran: which commit, on which machine, against which corpus *by
    hash* rather than by filename. A filename is not an identity — this project
    reached a wrong published conclusion by confusing two corpora whose names
    differed by one character.
    """
    if not provenance:
        return []
    code = provenance.get("code") or {}
    hardware = provenance.get("hardware") or {}
    versions = provenance.get("versions") or {}
    corpus = provenance.get("corpus") or {}
    rows = []
    commit = code.get("commit")
    if commit:
        url = code.get("commit_url")
        shown = f"[`{commit[:12]}`]({url})" if url else f"`{commit[:12]}`"
        if code.get("dirty"):
            # Not a footnote: the commit does not describe the code that ran.
            shown += " — **uncommitted changes present; this commit does not describe the code that ran**"
        rows.append(f"| training code | {shown} |")
    if corpus.get("sha256"):
        rows.append(f"| corpus | `{Path(corpus['path']).name}`, `{corpus['sha256'][:19]}…` |")
    accelerator = hardware.get("accelerator") or hardware.get("device")
    if accelerator:
        rows.append(f"| trained on | {accelerator}, {hardware.get('platform', '?')} |")
    pinned = ", ".join(
        f"{name} {value}" for name, value in versions.items() if value and name != "onnxruntime"
    )
    if pinned:
        rows.append(f"| versions | {pinned} |")
    if not rows:
        return []
    return ["## Reproducing this checkpoint", "", "| | |", "|---|---|", *rows, "",
            "The seed is in the table above. Everything else a rerun needs is here or in "
            "`training-report.json`, which carries the complete resolved config and the full "
            "provenance record.", ""]


def _training_section(config: dict[str, Any] | None) -> list[str]:
    """Hyperparameters, read from the run's own `config.json`.

    Only the fields that change an outcome. A card listing every field of a
    dataclass is not more reproducible, it is less readable, and the full
    record travels in `training-report.json` anyway.
    """
    if not config:
        return []
    lines = [
        "## How it was trained",
        "",
        "| | |",
        "|---|---|",
        f"| corpus | `{Path(str(config.get('corpus', '?'))).name}` |",
        f"| architecture preset | `{config.get('preset')}` |",
        f"| epochs | {config.get('epochs')} |",
        f"| batch size | {config.get('batch_size')} |",
        f"| learning rate | {config.get('lr')} (cosine to {config.get('min_lr')}) |",
        f"| weight decay | {config.get('weight_decay')} |",
        f"| seed | {config.get('seed')} |",
        f"| symmetry augmentation | {'yes' if config.get('augment') else 'no'} |",
        f"| ply-balanced sampling | {'yes' if config.get('balance_plies') else 'no'} |",
        "",
        "Labels are **exact**, not bootstrapped: every training target comes "
        "from a solved position, so the network is fitting ground truth "
        "rather than its own earlier opinions.",
        "",
        "The learning rate is a property of the architecture rather than a "
        "project-wide default. A single shared rate is not equal treatment "
        "between architectures — it privileges whichever one it was chosen "
        "for — and correcting that in this project reversed several "
        "conclusions rather than merely shifting decimals.",
        "",
    ]
    if config.get("balance_plies"):
        lines[-1:] = [
            "Ply-balanced sampling gives every game stage equal attention "
            "instead of attention proportional to how many positions it "
            "happens to contribute. The corpus is dominated by late "
            "positions; the match is decided early.",
            "",
        ]
    return lines


def _limitations_section(shift: list[dict] | None, architecture: str) -> list[str]:
    """What the model is worst at, taken from the held-out evaluation.

    A card that reports one pooled accuracy hides the only region where the
    model is genuinely uncertain. This reports the weakest ply explicitly.
    """
    lines = [
        "## Limitations",
        "",
        "**Accuracy is not uniform across the game.** Deep positions are "
        "nearly forced and every model in this family is close to perfect "
        "there; the shallow openings are where they differ and where they "
        "are weakest.",
        "",
    ]
    record = None
    if shift:
        record = next((r for r in shift if r["architecture"] == architecture), None)
    if record:
        by_ply = record["by_ply"]
        plies = sorted(int(p) for p in by_ply)
        worst = min(plies, key=lambda p: by_ply[str(p)]["accuracy"])
        best = max(plies, key=lambda p: by_ply[str(p)]["accuracy"])
        lines += [
            "| ply | accuracy on provably won positions |",
            "|---|---|",
        ]
        lines += [f"| {p} | {by_ply[str(p)]['accuracy']:.4f} |" for p in plies]
        lines += [
            "",
            f"Weakest at ply {worst} ({by_ply[str(worst)]['accuracy']:.1%}), "
            f"strongest at ply {best} ({by_ply[str(best)]['accuracy']:.1%}).",
            "",
        ]
    lines += [
        "**The evaluation is against solved positions and other engines, not "
        "against people.** Nothing here says how it plays against a human.",
        "",
        "**One training seed.** Every number on this card comes from a single "
        "run of this architecture.",
        "",
    ]
    return lines


def _usage_section(
    repo_id: str,
    manifest: dict[str, Any],
    links: dict[str, str],
    provenance: dict[str, Any] | None = None,
) -> list[str]:
    """Two paths: the Python package, and the ONNX graph with neither.

    Both snippets are meant to be pasted and run, so every name they use is
    either imported in the snippet or defined in it. A snippet with an
    undefined variable is a snippet that has never been run.
    """
    lines = [
        "## Usage",
        "",
        "There is no `AutoModel` for this architecture — the Hub cannot "
        "reconstruct it from weights alone. Two supported paths.",
        "",
        "### With `quantik-models`",
        "",
        "Reads `manifest.json` and rebuilds the network from "
        "`architecture_spec`, and gives you the legality masking for free.",
        "",
        "```bash",
        *_install_lines(provenance),
        "```",
        "",
        "```python",
        "from quantik_models import hub",
        "from quantik_models.env import fastboard as fb",
        "",
        f'evaluator = hub.load_evaluator("{_short_name(manifest)}")',
        "",
        "boards = fb.empty_boards(1)                    # (1, 8) uint16",
        "policy, value = evaluator.evaluate(boards)     # masking applied",
        "```",
        "",
        "`hub.load_evaluator` downloads this repository, checks the weights "
        "against the digest in `manifest.json`, rebuilds the network from "
        "`architecture_spec` and applies legality masking. To load a "
        "directory you already have, pass it to "
        "`quantik_models.arena.registry.load_evaluator` instead.",
        "",
    ]
    if manifest.get("onnx_export"):
        lines += [
            "### With ONNX Runtime, and neither torch nor this package",
            "",
            "```bash",
            "pip install onnxruntime numpy huggingface_hub",
            "```",
            "",
            "```python",
            "import numpy as np, onnxruntime as ort",
            "from huggingface_hub import hf_hub_download",
            "",
            f'path = hf_hub_download("{repo_id}", "model.onnx")',
            "session = ort.InferenceSession(path)",
            "",
            "# (B, 9, 4, 4) float32, mover-relative — see the contract above.",
            "tensors = np.zeros((1, 9, 4, 4), dtype=np.float32)",
            'policy, value = session.run(None, {"board": tensors})',
            "",
            "# The mask is yours to apply. `legal` is a (B, 64) bool array;",
            "# quantik_models.env.fastboard.legal_masks computes it, and so",
            "# does quantik-core in Rust.",
            "# policy = np.where(legal, policy, -np.inf)",
            "```",
            "",
        ]
    lines += [
        "### The rules engine",
        "",
        "Legality, symmetry and the exact solver live in `quantik-core`, "
        "which is published for both languages and is what generated the "
        "training labels.",
        "",
        "```bash",
        "pip install quantik-core     # Python, >=3.12",
        "cargo add quantik-core       # Rust, 2021 edition",
        "```",
        "",
    ]
    return lines


def _licence_section(license_id: str) -> list[str]:
    """Weights and code are licensed differently. Say so.

    A reader who sees MIT on the GitHub repository will otherwise assume it
    covers the download, and a licence mismatch is exactly the kind of thing
    nobody checks until it matters.
    """
    lines = ["## Licence", ""]
    if license_id == "cc-by-nc-4.0":
        lines += [
            "**The weights in this repository are CC BY-NC 4.0.** Free to "
            "use, share and adapt for research, teaching and any other "
            "non-commercial purpose, with attribution. **Commercial use "
            "requires a separate agreement** — open an issue on the source "
            "repository or contact the author.",
            "",
            "This is deliberately not an OSI-approved open-source licence. "
            "Every OSI licence permits royalty-free commercial use, which is "
            "the one thing this reserves.",
            "",
            "**The code is separate and more permissive.** `quantik-models` "
            "and `quantik-core` are MIT, so the training pipeline, the rules "
            "engine and the evaluation harness carry no such restriction — "
            "only these weights do.",
            "",
        ]
    else:
        lines += [
            f"The weights in this repository are `{license_id}`. The code "
            "that produced them (`quantik-models`, `quantik-core`) is MIT.",
            "",
        ]
    return lines


def model_card(
    manifest: dict[str, Any],
    *,
    repo_id: str | None = None,
    license_id: str = DEFAULT_LICENSE,
    metrics: list[dict[str, Any]] | None = None,
    base_model: str | None = None,
    config: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
    shift: list[dict] | None = None,
    siblings: list[str] | None = None,
    links: dict[str, str] | None = None,
    body: str = "",
) -> str:
    """`README.md`: Hub metadata, then the card.

    Everything generated here is derived from a file — the manifest, the
    run's config, the held-out evaluation. Prose that is an argument rather
    than a fact belongs in `body`, written by a person.
    """
    metrics = metrics or []
    links = links or {}
    # Derived, not a placeholder. A card that ships `<your-org>` teaches the
    # reader to edit the snippet before running it, and most will not.
    repo_id = repo_id or repo_id_for(manifest["architecture"])
    architecture = manifest["architecture"]
    header = _front_matter(manifest, license_id, metrics, base_model)

    lines = [
        "",
        f"# {architecture}",
        "",
        "A policy/value network for **Quantik**, "
        f"{manifest['parameter_count']:,} parameters.",
        "",
        "Quantik is a two-player game on a 4x4 board with four piece shapes. "
        "A player may not place a shape in a row, column or 2x2 zone where "
        "that shape already appears, whoever played it — so a move can be "
        "blocked by your own piece. The first player to complete a line or "
        "zone holding all four distinct shapes wins. There are no draws.",
        "",
        "This model predicts, for a given position, which move an exact "
        "solver would play (policy) and who is winning (value).",
        "",
        cards.PROJECT,
    ]

    arch_key = architecture.split("-", 1)[0]
    summary = cards.summary_for(arch_key)
    diagram = cards.diagram_for(arch_key)
    if summary or diagram:
        lines += ["## Architecture", ""]
        if summary:
            lines += [summary, ""]
        if diagram:
            lines += [diagram, ""]
        spec = manifest.get("architecture_spec", {}).get("config", {})
        if spec:
            lines += ["| | |", "|---|---|"]
            lines += [f"| {k} | {v} |" for k, v in sorted(spec.items())]
            lines += [
                f"| parameters | {manifest['parameter_count']:,} |",
                "",
                "Every architecture in this family is matched to within 1.2% "
                "on parameter count, so a comparison between them is about "
                "the design and not about capacity.",
                "",
            ]

    if metrics:
        lines += ["## Results", ""] + _results_table(metrics) + [""]
        lines += [
            "Held-out accuracy is measured on exactly solved positions "
            "sharing no canonical key with the training corpus, up to the "
            "192 board symmetries — so it measures generalisation, not "
            "recall. It is reported split rather than pooled because the "
            "corpus contains nothing at the shallowest plies, and a pooled "
            "figure is dominated by deep positions where every model is "
            "near perfect.",
            "",
        ]

    lines += [
        "## Input and output contract",
        "",
        "```",
        "input   (B, 9, 4, 4) float32      tensor-board.v1, mover-relative",
        "output  (B, 64) policy logits     action_index = shape * 16 + position",
        "        (B,)    value in [-1, 1]  +1 = good for the side to move",
        "```",
        "",
        "Planes 0-3 are the side to move, 4-7 the opponent, 8 a ply "
        "indicator. `position = row * 4 + col`.",
        "",
        "### Legality masking happens outside this model",
        "",
        "It emits logits over all 64 actions, including illegal ones. "
        "Applying the legal-move mask before the softmax is the caller's "
        "job. **An unmasked `argmax` from this model will play illegal "
        "moves** — silently, because an illegal move looks like a bad move "
        "rather than like a bug.",
        "",
        "This is by design. Quantik's rules are exact and cheap to compute "
        "in `quantik-core`, so the network is never asked to approximate "
        "them and never spends capacity on legality.",
        "",
    ]

    lines += _usage_section(repo_id, manifest, links, provenance)
    lines += _training_section(config)
    lines += _provenance_section(provenance)
    lines += _limitations_section(shift, architecture)

    lines += ["## Files", ""]
    lines.append(f"- `model.safetensors` — `{manifest['weights_hash']}`")
    if manifest.get("onnx_export"):
        lines.append(
            f"- `model.onnx` — opset {manifest.get('onnx_opset')}, "
            f"`{manifest['onnx_hash']}`, dynamic batch dimension"
        )
    lines += [
        "- `config.json` — the architecture spec, readable without loading anything",
        "- `manifest.json` — the `model-checkpoint.v1` record this repo was staged from",
        "- `training-report.json` — the epoch that produced these weights, and its metrics",
        "",
        f"Contract version `{manifest['contract_version']}`. "
        f"Exported {manifest['created_at'][:10]}.",
        "",
    ]

    if siblings:
        others = [s for s in siblings if s != repo_id]
        if others:
            lines += [
                "## Other models in this family",
                "",
                "Same contract, same corpus, same training protocol — "
                "interchangeable at the interface, so they can be compared "
                "directly.",
                "",
            ]
            lines += [f"- [`{s}`](https://huggingface.co/{s})" for s in others]
            lines.append("")

    if links:
        lines += ["## Source", ""]
        lines += [f"- {label}: {url}" for label, url in links.items()]
        lines.append("")

    lines += _licence_section(license_id)

    return header + "\n".join(lines) + ("\n" + body.strip() + "\n" if body.strip() else "")


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


def run_config(checkpoint_dir: Path) -> dict[str, Any] | None:
    """The training run's `config.json`, which sits one level up.

    A checkpoint is `runs/train/<name>/best`; the config that produced it is
    `runs/train/<name>/config.json`. Returning None rather than raising is
    deliberate — a checkpoint copied out of its run directory is still worth
    publishing, just with a thinner card.
    """
    path = checkpoint_dir.parent / "config.json"
    return json.loads(path.read_text()) if path.exists() else None


def run_provenance(checkpoint_dir: Path) -> dict[str, Any] | None:
    """The run's provenance record — commit, machine, versions, corpus hash.

    Two places, in order: `training-report.json` beside the checkpoint, which
    travels with it, then `provenance.json` one level up in the run directory,
    which does not. Preferring the travelling copy means a checkpoint moved out
    of its run directory still carries its own provenance, and a checkpoint
    trained before this record existed simply has none — returning None thins
    the card rather than failing the publish.
    """
    report = checkpoint_dir / "training-report.json"
    if report.exists():
        recorded = json.loads(report.read_text()).get("provenance")
        if recorded:
            return recorded
    path = checkpoint_dir.parent / "provenance.json"
    return json.loads(path.read_text()) if path.exists() else None


def stage(
    checkpoint_dir: Path,
    out_dir: Path,
    *,
    repo_id: str | None = None,
    namespace: str | None = None,
    license_id: str = DEFAULT_LICENSE,
    metrics: list[dict[str, Any]] | None = None,
    base_model: str | None = None,
    shift: list[dict] | None = None,
    siblings: list[str] | None = None,
    links: dict[str, str] | None = None,
    body: str = "",
) -> Path:
    """Write a Hub-ready directory. Copies, never moves, and never uploads."""
    manifest = json.loads((checkpoint_dir / "manifest.json").read_text())
    repo_id = repo_id or repo_id_for(manifest["architecture"], namespace)
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
            config=run_config(checkpoint_dir),
            provenance=run_provenance(checkpoint_dir),
            shift=shift,
            siblings=siblings,
            links=links,
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
    parser.add_argument(
        "--repo-id",
        default=None,
        help="full id; derived from the manifest and --namespace when omitted",
    )
    parser.add_argument(
        "--namespace",
        default=None,
        help=f"Hub account (default: $QUANTIK_HF_NAMESPACE, else {DEFAULT_NAMESPACE})",
    )
    parser.add_argument("--license", default=DEFAULT_LICENSE)
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
        "--sibling",
        action="append",
        default=None,
        help="another repo id in this family; repeatable",
    )
    parser.add_argument(
        "--link",
        action="append",
        default=None,
        metavar="LABEL=URL",
        help="a source link for the card; repeatable",
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

    links = {}
    for entry in args.link or []:
        label, _, url = entry.partition("=")
        if not url:
            raise SystemExit(f"--link expects LABEL=URL, got {entry!r}")
        links[label] = url

    out = stage(
        args.checkpoint,
        args.out,
        repo_id=args.repo_id,
        namespace=args.namespace,
        license_id=args.license,
        metrics=metrics,
        shift=json.loads(args.shift.read_text()) if args.shift else None,
        siblings=args.sibling,
        links=links,
        base_model=args.base_model,
        body=args.body.read_text() if args.body else "",
    )
    for path in sorted(out.iterdir()):
        print(f"  {path.name}  {path.stat().st_size:,} bytes")
    print("\nhashes verified against the manifest")
    manifest = json.loads((args.checkpoint / "manifest.json").read_text())
    print(f"\nstaged {out}")
    print(f"  repo id: {args.repo_id or repo_id_for(manifest['architecture'], args.namespace)}")
    print("nothing has been uploaded; see docs/publishing-to-hugging-face.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
