#!/usr/bin/env python
"""The headline experiment: network versus every incumbent strategy.

Runs both measurements that matter and writes one report:

1. **Arena** — side-balanced paired games against `random`, `minimax`, `mcts`
   and `beam`, with measured ms/move alongside every result. Measured, not
   nominal: `quantik_core`'s engines overshoot their budgets by design
   (minimax checks its clock between deepening iterations, beam between beam
   levels), so a nominal "100 ms" minimax actually spends ~196 ms.
2. **Oracle probe** — outcome accuracy per ply against exact truth on the
   held-out 640-position probe, which says *where* an agent is wrong rather
   than only that it lost.

A claim of "beats the incumbents" needs both: winning the arena while being
slower per move, or while being less accurate, would not be the same result.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantik_models.arena.match import sample_start_positions
from quantik_models.arena.parallel import play_match_parallel
from quantik_models.arena.registry import build_agent


def leaderboard(results: list[dict]) -> list[dict]:
    tally: dict[str, list[int]] = {}
    for row in results:
        for name, wins in ((row["agent_a"], row["wins_a"]), (row["agent_b"], row["wins_b"])):
            entry = tally.setdefault(name, [0, 0])
            entry[0] += wins
            entry[1] += row["games"]
    table = [
        {"agent": name, "wins": w, "games": g, "win_rate": w / g if g else 0.0}
        for name, (w, g) in tally.items()
    ]
    return sorted(table, key=lambda r: -r["win_rate"])


def ms_per_move(results: list[dict]) -> dict[str, float]:
    """Mean measured think time per move, pooled over every game an agent played."""
    totals: dict[str, list[float]] = {}
    for row in results:
        for name, ms in (
            (row["agent_a"], row["ms_per_move_a"]),
            (row["agent_b"], row["ms_per_move_b"]),
        ):
            totals.setdefault(name, []).append(ms)
    return {name: sum(v) / len(v) for name, v in totals.items()}


def render(report: dict) -> str:
    timings = ms_per_move(report["arena"])
    probe = {row["agent"]: row for row in report.get("probe", [])}
    plies = sorted({p for row in probe.values() for p in row["accuracy_by_ply"]}, key=int)

    lines = [
        f"# {report['title']}",
        "",
        f"Generated `{report['generated_at']}`. "
        f"{report['games_per_pairing']} side-balanced games per pairing from "
        f"{report['positions']} symmetry-distinct openings at plies {report['start_plies']}; "
        f"probe of {report.get('probe_positions', 0)} exactly-solved held-out positions.",
        "",
        "## Leaderboard",
        "",
        "| agent | win rate | wins | games | ms/move | outcome accuracy |",
        "|---|---|---|---|---|---|",
    ]
    for row in leaderboard(report["arena"]):
        accuracy = probe.get(row["agent"], {}).get("outcome_accuracy")
        accuracy_cell = f"{accuracy:.1%}" if accuracy is not None else "—"
        lines.append(
            f"| `{row['agent']}` | **{row['win_rate']:.1%}** | {row['wins']} | {row['games']} "
            f"| {timings.get(row['agent'], 0.0):.0f} | {accuracy_cell} |"
        )

    lines += [
        "",
        "## Head-to-head",
        "",
        "| A | B | A wins | B wins | A win rate | 95% CI | mean plies | ms/move A | ms/move B |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in report["arena"]:
        lines.append(
            f"| `{row['agent_a']}` | `{row['agent_b']}` | {row['wins_a']} | {row['wins_b']} "
            f"| {row['score_a']:.1%} | {row['ci_low']:.1%}-{row['ci_high']:.1%} "
            f"| {row['mean_plies']:.1f} | {row['ms_per_move_a']:.0f} | {row['ms_per_move_b']:.0f} |"
        )

    if probe:
        lines += [
            "",
            "## Outcome accuracy against exact truth, by ply",
            "",
            "Share of provably won positions where the agent picks a move that keeps the win.",
            "",
            "| agent | " + " | ".join(f"ply {p}" for p in plies) + " | overall |",
            "|---" * (len(plies) + 2) + "|",
        ]
        for name, row in sorted(probe.items(), key=lambda kv: -kv[1]["outcome_accuracy"]):
            cells = []
            for p in plies:
                bucket = row["accuracy_by_ply"].get(p)
                cells.append(f"{bucket['accuracy']:.1%}" if bucket else "—")
            lines.append(
                f"| `{name}` | " + " | ".join(cells) + f" | **{row['outcome_accuracy']:.1%}** |"
            )
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="report path prefix")
    parser.add_argument("--title", default="Quantik: network vs the incumbents")
    parser.add_argument("--probe", type=Path, default=Path("runs/oracle/probe.jsonl"))
    parser.add_argument("--positions", type=int, default=48)
    parser.add_argument(
        "--start-plies",
        default="3-5",
        help="opening depth: a single ply (\"4\") or an inclusive range (\"3-5\")",
    )
    parser.add_argument("--position-seed", type=int, default=20260827)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--skip-probe", action="store_true")
    args = parser.parse_args(argv)

    specs = json.loads(args.agents.read_text())
    names = [build_agent(s).name for s in specs]
    if "-" in str(args.start_plies):
        low, high = (int(v) for v in str(args.start_plies).split("-"))
        start_plies: int | list[int] = list(range(low, high + 1))
    else:
        start_plies = int(args.start_plies)
    positions = sample_start_positions(
        args.positions, plies=start_plies, seed=args.position_seed
    )
    seeds = tuple(range(args.seeds))

    report = {
        "title": args.title,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "agents": specs,
        "positions": int(positions.shape[0]),
        "start_plies": args.start_plies,
        "games_per_pairing": args.positions * len(seeds) * 2,
        "arena": [],
        "probe": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json_path, md_path = args.out.with_suffix(".json"), args.out.with_suffix(".md")

    def save() -> None:
        json_path.write_text(json.dumps(report, indent=2))
        md_path.write_text(render(report))

    if not args.skip_probe and args.probe.exists():
        from scripts.oracle_probe import evaluate, load_probe, value_error

        probe = load_probe(args.probe)
        report["probe_positions"] = len(probe)
        for spec, name in zip(specs, names):
            row = evaluate(build_agent(spec), probe)
            extra = value_error(spec, probe)
            if extra:
                row.update(extra)
            report["probe"].append(row)
            print(f"probe {name}: {row['outcome_accuracy']:.1%}", file=sys.stderr, flush=True)
            save()

    pairs = list(combinations(range(len(specs)), 2))
    for k, (i, j) in enumerate(pairs, start=1):
        started = time.perf_counter()

        def progress(done, total, k=k, i=i, j=j):
            print(
                f"\r[{k}/{len(pairs)}] {names[i]} vs {names[j]}: {done}/{total}",
                end="",
                file=sys.stderr,
                flush=True,
            )

        result = play_match_parallel(
            specs[i], specs[j], positions, seeds, workers=args.workers, progress=progress
        )
        row = result.to_dict()
        row["seconds"] = time.perf_counter() - started
        report["arena"].append(row)
        print(f"\r{' ' * 78}\r{result.summary()}", file=sys.stderr, flush=True)
        save()

    print(f"wrote {json_path} and {md_path}", file=sys.stderr)
    print(render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
