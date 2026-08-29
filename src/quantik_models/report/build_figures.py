"""Build every figure the benchmark report embeds, from `runs/`.

Deliberately a thin layer over `figures`: this module knows *which* runs the
published report is about, and nothing else. The run names are the argument
that matters — `lineup-cpool` and `lineup-attn` were trained at the ResNet's
2e-3 and are drawn as superseded, `swept-*` replaced them.

    python -m quantik_models.report.build_figures --runs runs --out docs/figures

Missing inputs are reported and skipped rather than fatal: `runs/` is
gitignored, so on a fresh clone every input is missing and the right
behaviour is to say so, not to crash.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import figures as fg

# The lineup, in the order the report tables use. The value is the run
# directory under `runs/train`; the key is what the legend says.
LINEUP: dict[str, str] = {
    "resnet": "lineup-resnet",
    "mlp": "lineup-mlp",
    "cpool": "swept-cpool",
    "attn": "swept-attn",
}
# Runs kept in the training figure only to show what the inherited rate cost.
SUPERSEDED: dict[str, str] = {
    "cpool": "lineup-cpool",
    "attn": "lineup-attn",
}


def build(runs: Path, out: Path, eval_dir: str, oracle_dir: Path | None = None) -> list[Path]:
    """Write every figure that has its inputs; return the paths written."""
    written: list[Path] = []
    skipped: list[str] = []

    training: list[fg.TrainingRun] = []
    superseded_names: set[str] = set()
    for arch, run_name in list(LINEUP.items()) + list(SUPERSEDED.items()):
        run_dir = runs / "train" / run_name
        if not (run_dir / "metrics.jsonl").exists():
            skipped.append(str(run_dir))
            continue
        label = f"{arch} ({run_name})"
        training.append(fg.load_training_run(run_dir, name=label))
        if run_name in SUPERSEDED.values():
            superseded_names.add(label)
    if training:
        written.append(
            fg.training_curves(training, out / "training-curves.svg", superseded=superseded_names)
        )

    sweep_dir = runs / "lrsweep"
    if any(sweep_dir.glob("sweep-*")):
        written.append(fg.lr_sweep(fg.load_sweep(sweep_dir), out / "lr-sweep.svg"))
    else:
        skipped.append(str(sweep_dir))

    shift_path = runs / "eval" / eval_dir / "shift.json"
    if shift_path.exists():
        shift = fg.load_shift(shift_path)
        written.append(fg.accuracy_by_ply(shift, out / "accuracy-by-ply.svg"))
        written.append(fg.value_mae_by_ply(shift, out / "value-mae-by-ply.svg"))
    else:
        skipped.append(str(shift_path))

    for prefix, plies, title in (
        ("policy", (3, 6, 9), "Policy arena: ranking against start depth"),
        ("mcts", (3, 6), "128-simulation MCTS arena, with the uniform control"),
    ):
        boards: dict[int, list[dict]] = {}
        for ply in plies:
            games = runs / "eval" / eval_dir / f"{prefix}-p{ply}" / "games.json"
            if games.exists():
                boards[ply] = fg.load_leaderboard(games)
            else:
                skipped.append(str(games))
        if boards:
            written.append(fg.arena_by_depth(boards, out / f"arena-{prefix}.svg", title=title))

    if oracle_dir is not None:
        summary_path = oracle_dir / "packed" / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            per_run = {}
            for run in summary["runs"]:
                board = fg.load_leaderboard(oracle_dir / run["name"] / "games.json")
                per_run[run["name"]] = {r["agent"]: r["win_rate"] for r in board}
            oracle = summary.get("oracle") or max(
                summary["pooled"], key=lambda r: r["games"]
            )["agent"]
            depths = {r.get("start_plies") for r in summary["runs"]}
            depth = f"start ply {depths.pop()}" if len(depths) == 1 else "mixed start depths"
            games = sum(r["games"] for r in summary["runs"])
            written.append(
                fg.oracle_benchmark(
                    summary["pooled"],
                    per_run,
                    out / "oracle-benchmark.svg",
                    oracle=oracle,
                    subtitle=f"{depth}, {games:,} games, {len(summary['runs'])} seeds",
                )
            )
        else:
            skipped.append(str(summary_path))

    for missing in skipped:
        print(f"skipped (missing): {missing}")
    for figure in written:
        print(f"wrote {figure}")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("docs/figures"))
    parser.add_argument(
        "--oracle-dir",
        type=Path,
        default=None,
        help="a runs/eval/oracle-* directory holding per-seed runs and packed/",
    )
    parser.add_argument(
        "--eval-dir",
        default="swept-2026-08-30",
        help="subdirectory of runs/eval holding the evaluation to plot",
    )
    args = parser.parse_args(argv)
    written = build(args.runs, args.out, args.eval_dir, args.oracle_dir)
    if not written:
        print("nothing written — is runs/ populated?")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
