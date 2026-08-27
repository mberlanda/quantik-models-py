#!/usr/bin/env python
"""Run a round-robin arena and write a JSON + Markdown report.

Results are appended to the JSON report as each pairing finishes, so a long
run can be inspected — or resumed — while it is still going.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

from quantik_models.arena.match import sample_start_positions
from quantik_models.arena.parallel import play_match_parallel
from quantik_models.arena.registry import build_agent, fixed_time_baselines


def leaderboard(results: list[dict]) -> list[dict]:
    """Win rate of each agent across every game it played."""
    tally: dict[str, list[int]] = {}
    for row in results:
        tally.setdefault(row["agent_a"], [0, 0])
        tally.setdefault(row["agent_b"], [0, 0])
        tally[row["agent_a"]][0] += row["wins_a"]
        tally[row["agent_a"]][1] += row["games"]
        tally[row["agent_b"]][0] += row["wins_b"]
        tally[row["agent_b"]][1] += row["games"]
    table = [
        {"agent": name, "wins": w, "games": g, "win_rate": w / g if g else 0.0}
        for name, (w, g) in tally.items()
    ]
    return sorted(table, key=lambda r: -r["win_rate"])


def render_markdown(report: dict) -> str:
    lines = [
        f"# Arena — {report['label']}",
        "",
        f"- generated: `{report['generated_at']}`",
        f"- start positions: {report['positions']} unique, sampled at ply "
        f"{report['start_plies']} (seed {report['position_seed']})",
        f"- seeds per position: {len(report['seeds'])}",
        f"- games per pairing: {report['games_per_pairing']} (side-balanced)",
        "",
        "## Leaderboard",
        "",
        "| agent | win rate | wins | games |",
        "|---|---|---|---|",
    ]
    for row in leaderboard(report["results"]):
        lines.append(
            f"| `{row['agent']}` | {row['win_rate']:.1%} | {row['wins']} | {row['games']} |"
        )
    lines += [
        "",
        "## Pairings",
        "",
        "| A | B | A wins | B wins | A win rate | 95% CI | mean plies | ms/move A | ms/move B |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in report["results"]:
        lines.append(
            f"| `{row['agent_a']}` | `{row['agent_b']}` | {row['wins_a']} | {row['wins_b']} "
            f"| {row['score_a']:.1%} | {row['ci_low']:.1%}-{row['ci_high']:.1%} "
            f"| {row['mean_plies']:.1f} | {row['ms_per_move_a']:.1f} | {row['ms_per_move_b']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="report path prefix (no suffix)")
    parser.add_argument("--label", default="baseline round-robin")
    parser.add_argument("--agents", type=Path, help="JSON file with a list of agent specs")
    parser.add_argument("--time-limit", type=float, default=0.1, help="per-move budget for baselines")
    parser.add_argument("--positions", type=int, default=32)
    parser.add_argument("--start-plies", type=int, default=4)
    parser.add_argument("--position-seed", type=int, default=20260827)
    parser.add_argument("--seeds", type=int, default=2, help="seeds per position")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)

    specs = (
        json.loads(args.agents.read_text())
        if args.agents
        else fixed_time_baselines(args.time_limit)
    )
    names = [build_agent(s).name for s in specs]
    positions = sample_start_positions(
        args.positions, plies=args.start_plies, seed=args.position_seed
    )
    seeds = tuple(range(args.seeds))
    per_pairing = args.positions * len(seeds) * 2

    report = {
        "label": args.label,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "agents": specs,
        "agent_names": names,
        "positions": int(positions.shape[0]),
        "start_plies": args.start_plies,
        "position_seed": args.position_seed,
        "seeds": list(seeds),
        "games_per_pairing": per_pairing,
        "results": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.out.with_suffix(".json")
    md_path = args.out.with_suffix(".md")

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
        report["results"].append(row)
        print(f"\r{' ' * 70}\r{result.summary()}  [{row['seconds']:.1f}s]", file=sys.stderr)
        json_path.write_text(json.dumps(report, indent=2))
        md_path.write_text(render_markdown(report))

    print(f"\nwrote {json_path} and {md_path}", file=sys.stderr)
    print(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
