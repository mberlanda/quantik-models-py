"""Stage the development artefacts that `runs/` holds and git does not.

`runs/` is gitignored and lives on one machine. It holds roughly 1.3 GB of
things that are expensive to recreate — solver output measured in days, board
enumerations, and every trained checkpoint. Losing that directory means
recomputing all of it, and nothing about the published model repositories helps:
a model repo carries weights and a card, not the corpus the weights were fitted
to or the probe they were scored against.

This module stages those artefacts into a Hugging Face **dataset** repository
(`quantik-dev-data` by default), one directory per artefact group, each with a
`README.md` that says what it is, what it cost, how to reproduce it, and how to
extend it. It **never uploads** — staging and publishing are separate steps
because publishing is not reversible in the way a local directory is.

Two rules the catalogue enforces:

* **Every group is content-addressed.** `MANIFEST.json` records a sha256 per
  file. A corpus is identified by its hash, never by its filename — this project
  reached a wrong published conclusion by confusing `exact-sampled.npz` with
  `exact-sampled-v2.npz`.
* **The probe is a group of its own and is labelled as held out.** Every
  evaluation number compares against it. A probe that quietly becomes training
  data turns generalisation into recall while every report still looks fine.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .digest import file_digest

DEFAULT_REPO = "quantik-dev-data"
# Weights, arrays and solver output all need LFS. A file committed as a plain
# blob cannot be fixed by a later commit — the same trap `stage_hub_repos.sh`
# documents for model repos.
LFS_PATTERNS = ("*.npz", "*.npy", "*.jsonl", "*.safetensors", "*.onnx", "*.gz")


@dataclass(frozen=True)
class Artefact:
    """One directory in the dataset repo."""

    name: str
    sources: tuple[str, ...]
    summary: str
    produced_by: str
    cost: str
    reproduce: str
    expand: str
    caveats: tuple[str, ...] = field(default_factory=tuple)


CATALOGUE: tuple[Artefact, ...] = (
    Artefact(
        name="corpora",
        sources=("runs/oracle/corpus/*.npz",),
        summary=(
            "The training corpora. One row per canonical position, labelled by the "
            "exact oracle. `exact-sampled.npz` is what every published checkpoint "
            "trained on."
        ),
        produced_by="`quantik-core` exact oracle, then `data.merge_corpus`",
        cost="days of solver time; the labelling, not the merging, is the expense",
        reproduce=(
            "Solve an enumerated level (`enumerations/`) or collect positions with "
            "`arena.autoplay` and label them, then fold them in with "
            "`python -m quantik_models.data.merge_corpus`. See `docs/corpora.md`."
        ),
        expand=(
            "Extend to **new plies**, do not densify covered ones. Plies 0-2 hold no "
            "positions in any corpus. The v2-to-v3 step added 323,568 rows to already-"
            "covered plies and bought a measured zero."
        ),
        caveats=(
            "Two schemas. v1 stores `policy_target float32 (N,64)` + `policy_weight`; "
            "v2/v3 store a `uint64 optimal_mask`. They do not concatenate — convert "
            "first; the mapping is exact both ways.",
            "Only ~8% of rows carry a policy label. The rest are value-only.",
            "Identify a corpus by its hash. Two of these differ by one character in "
            "the filename and are different corpora.",
        ),
    ),
    Artefact(
        name="enumerations",
        sources=("runs/canonical/level*.npy", "runs/canonical/counts.json"),
        summary=(
            "Every canonical live position at plies 1-8, deduplicated up to the 192 "
            "board symmetries. Unlabelled — this is the search, not the solve."
        ),
        produced_by="canonical enumeration in `quantik-core`",
        cost="hours of search; `level08.npy` alone is 273 MB",
        reproduce="Re-run the canonical enumeration per level. Deterministic, no seed.",
        expand=(
            "Levels beyond 8 grow fast and were not enumerated. The useful direction "
            "is not deeper — it is **labelling** levels 1-6, which is what full "
            "opening coverage needs and which no corpus has."
        ),
        caveats=(
            "These are positions, not labels. Feeding them to a trainer requires "
            "solving them first.",
        ),
    ),
    Artefact(
        name="probe",
        sources=("runs/oracle/probe-large.jsonl", "runs/oracle/probe-qfens.txt"),
        summary=(
            "**HELD OUT.** 7,800 exactly-solved positions at plies 4-12 sharing no "
            "canonical key with any corpus. Every shift-evaluation number in the "
            "project is measured on this."
        ),
        produced_by="`scripts/oracle_probe.py`, then the exact oracle",
        cost="small — it is 7,800 positions",
        reproduce="Sample positions outside the corpus keys and solve them.",
        expand=(
            "Growing it is cheap and worthwhile, but a **new** probe must be "
            "re-checked against every corpus for key overlap before it is used."
        ),
        caveats=(
            "Never merge this into a corpus. `merge_corpus` excludes probe keys from "
            "the merged result — not just from incoming rows — because solving a "
            "position also labels its children. Sixteen probe positions reached the "
            "first corpus exactly that way.",
            "It is the common ground that per-corpus validation splits are not. "
            "Cross-corpus comparisons must use this.",
        ),
    ),
    Artefact(
        name="opening-book",
        sources=("runs/oracle/opening/*", "runs/oracle/opening5/*"),
        summary="Exhaustively solved opening positions — the exact book.",
        produced_by="the exact oracle over enumerated shallow levels",
        cost="substantial solver time; this is the expensive shallow work",
        reproduce="Solve `enumerations/` levels 1-5 exhaustively.",
        expand=(
            "Plies 0-6 total 1,019,275 canonical positions — fewer than the 3,087,356 "
            "rows the corpus already has. Complete opening coverage is a *smaller* "
            "job than what has already been done."
        ),
        caveats=(
            "For opening play, prefer the book over any network. The region is solved; "
            "the network is least informed exactly there.",
        ),
    ),
    Artefact(
        name="checkpoints",
        sources=("runs/train/*/best", "runs/train/*/config.json", "runs/train/*/metrics.jsonl", "runs/train/*/provenance.json"),
        summary=(
            "Trained checkpoints with their resolved config, per-epoch metrics and "
            "provenance. Includes the runs that are **not** published as model "
            "repositories — the patience family and the v3-corpus runs."
        ),
        produced_by="`quantik_models.train.supervised`",
        cost="hours per run; the four-architecture lineup is the bulk of it",
        reproduce=(
            "`provenance.json` records the commit, the machine, the dependency "
            "versions and the corpus hash. See `docs/reproducibility.md` — and note "
            "that bit-identical weights are not promised across a change of "
            "accelerator or torch version."
        ),
        expand=(
            "`--init-from` warm-starts from one of these and `--freeze` holds layers, "
            "so a new run does not restart from scratch. See "
            "`docs/retrain-and-finetune.md`."
        ),
        caveats=(
            "The four published checkpoints predate `provenance.json` and have no "
            "recorded commit. It is not recoverable after the fact.",
            "Every checkpoint here is training seed 20260828. The run-to-run spread "
            "has never been measured, so no margin between two of these has an error "
            "bar under it.",
        ),
    ),
    Artefact(
        name="evaluations",
        sources=("runs/eval/*", "runs/arena/*"),
        summary="Arena games and shift-evaluation output — the numbers behind every claim.",
        produced_by="`scripts/evaluate_lineup.sh`, `scripts/evaluate_opening_arena.sh`, `eval.shift`",
        cost="hours of CPU per arena",
        reproduce="Re-run the evaluation scripts with the seed recorded in each directory.",
        expand=(
            "Arena seeds 20260829 and 20260909 are spent. Use a third, and never reuse "
            "a training seed — that would make a seed-linked bias invisible rather "
            "than absent."
        ),
        caveats=(
            "Pairwise side-balanced rates are comparable **within** a run. Comparing "
            "across runs with different seeds is not sound.",
            "The seat dwarfs the model: mover 68-88%, responder 15-39%.",
            "No arena here starts before ply 3, except any produced by "
            "`evaluate_opening_arena.sh`.",
        ),
    ),
)


def gitattributes(patterns: tuple[str, ...] = LFS_PATTERNS) -> str:
    return "".join(f"{pattern} filter=lfs diff=lfs merge=lfs -text\n" for pattern in patterns)


def resolve(artefact: Artefact, root: Path) -> list[Path]:
    """Files an artefact actually matches, sorted and deduplicated."""
    found: set[Path] = set()
    for pattern in artefact.sources:
        for match in sorted(root.glob(pattern)):
            if match.is_file():
                found.add(match)
            elif match.is_dir():
                found.update(item for item in sorted(match.rglob("*")) if item.is_file())
    return sorted(found)


def artefact_card(artefact: Artefact, files: list[Path], root: Path) -> str:
    lines = [
        f"# {artefact.name}",
        "",
        artefact.summary,
        "",
        "| | |",
        "|---|---|",
        f"| produced by | {artefact.produced_by} |",
        f"| cost to recreate | {artefact.cost} |",
        f"| files | {len(files)} |",
        f"| size | {sum(f.stat().st_size for f in files) / 1e6:.1f} MB |",
        "",
        "## Reproducing it",
        "",
        artefact.reproduce,
        "",
        "## Extending it",
        "",
        artefact.expand,
        "",
    ]
    if artefact.caveats:
        lines += ["## Read this before using it", ""]
        lines += [f"- {caveat}" for caveat in artefact.caveats]
        lines += [""]
    lines += [
        "## Contents",
        "",
        "Hashes are in `MANIFEST.json`. **Identify a file by its hash, not its name.**",
        "",
        "| file | MB |",
        "|---|---|",
    ]
    for path in files:
        lines.append(f"| `{path.relative_to(root)}` | {path.stat().st_size / 1e6:.1f} |")
    return "\n".join(lines) + "\n"


def stage_artefact(artefact: Artefact, root: Path, out_dir: Path) -> dict[str, Any]:
    files = resolve(artefact, root)
    target = out_dir / artefact.name
    target.mkdir(parents=True, exist_ok=True)
    entries = []
    for path in files:
        relative = path.relative_to(root)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        entries.append(
            {
                "path": str(relative),
                "sha256": file_digest(path),
                "size_bytes": path.stat().st_size,
            }
        )
    (target / "README.md").write_text(artefact_card(artefact, files, root), encoding="utf-8")
    return {
        "name": artefact.name,
        "summary": artefact.summary,
        "file_count": len(entries),
        "size_bytes": sum(entry["size_bytes"] for entry in entries),
        "files": entries,
    }


def stage(
    root: Path,
    out_dir: Path,
    *,
    only: tuple[str, ...] | None = None,
    catalogue: tuple[Artefact, ...] = CATALOGUE,
) -> Path:
    """Write a Hub-ready dataset directory. Copies, never moves, never uploads."""
    selected = [a for a in catalogue if not only or a.name in only]
    unknown = set(only or ()) - {a.name for a in catalogue}
    if unknown:
        raise ValueError(f"unknown artefact(s): {', '.join(sorted(unknown))}")
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = [stage_artefact(artefact, root, out_dir) for artefact in selected]
    (out_dir / ".gitattributes").write_text(gitattributes(), encoding="utf-8")
    (out_dir / "MANIFEST.json").write_text(
        json.dumps({"schema": "dev-data-manifest.v1", "artefacts": groups}, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "README.md").write_text(dataset_card(groups), encoding="utf-8")
    return out_dir


def dataset_card(groups: list[dict[str, Any]]) -> str:
    total = sum(group["size_bytes"] for group in groups) / 1e6
    lines = [
        "---",
        "license: cc-by-nc-4.0",
        "tags:",
        "- quantik",
        "- board-games",
        "- reinforcement-learning",
        "---",
        "",
        "# Quantik development data",
        "",
        "The artefacts that `runs/` holds and git does not: solved corpora, board "
        "enumerations, the held-out probe, the exact opening book, every trained "
        "checkpoint, and the arena output behind every published number.",
        "",
        "**Why this exists.** `runs/` is gitignored and lives on one machine. Losing "
        "it means recomputing days of solver time and hours of training. The published "
        "model repositories do not help — a model repo carries weights and a card, not "
        "the corpus the weights were fitted to or the probe they were scored against.",
        "",
        f"{len(groups)} artefact groups, {total:.0f} MB total. Each directory has its own "
        "`README.md` with what it is, what it cost, how to reproduce it and how to "
        "extend it. `MANIFEST.json` carries a sha256 per file.",
        "",
        "| group | files | MB | |",
        "|---|---|---|---|",
    ]
    for group in groups:
        lines.append(
            f"| [`{group['name']}`]({group['name']}/) | {group['file_count']} | "
            f"{group['size_bytes'] / 1e6:.0f} | {group['summary'].split('.')[0]}. |"
        )
    lines += [
        "",
        "## Restoring into a checkout",
        "",
        "Paths are kept **relative to the repository root**, so a group restores "
        "straight back to where the tooling expects it:",
        "",
        "```bash",
        "huggingface-cli download --repo-type dataset <namespace>/quantik-dev-data \\",
        "  --local-dir /tmp/devdata",
        "cp -r /tmp/devdata/corpora/runs/ /path/to/quantik-models-py/",
        "```",
        "",
        "Then verify what you got against `MANIFEST.json` rather than trusting the "
        "filenames — see rule 1.",
        "",
        "## Two rules",
        "",
        "1. **Identify a file by its hash, not its name.** `exact-sampled.npz` and "
        "`exact-sampled-v2.npz` are different corpora whose names differ by one "
        "character, and confusing them produced a wrong published conclusion.",
        "2. **The probe is held out.** Never merge `probe/` into a corpus. "
        "`merge_corpus` excludes probe keys from the *merged result*, not just from "
        "incoming rows, because solving a position also labels its children.",
        "",
        "Weights and data are CC BY-NC 4.0; the code that produced them is MIT. "
        "Code: <https://github.com/mberlanda/quantik-models-py>. "
        "Models: <https://huggingface.co/brpoplpush>.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path, help="directory to stage into")
    parser.add_argument("--root", type=Path, default=Path("."), help="repository root")
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        help=f"stage one group; repeatable. One of: {', '.join(a.name for a in CATALOGUE)}",
    )
    args = parser.parse_args(argv)
    out = stage(args.root, args.out, only=tuple(args.only) if args.only else None)
    manifest = json.loads((out / "MANIFEST.json").read_text())
    for group in manifest["artefacts"]:
        print(f"{group['name']:>14}  {group['file_count']:>5} files  {group['size_bytes'] / 1e6:>8.1f} MB")
    print(f"\nstaged {out}")
    print("\nNothing was uploaded. To publish:")
    print(f"  huggingface-cli upload-large-folder --repo-type dataset <namespace>/{DEFAULT_REPO} {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
