"""Pool several arena runs into one compressed, reusable result set.

An oracle benchmark is several runs, not one: the same field against the
same opponent under different seeds and start depths, because a single seed
cannot tell a result from a seed-linked bias. That leaves a directory per
run, each with its own leaderboard and its own set of visited positions,
and two things have to come out of it:

* **a pooled reading**, with each seed still visible separately — pooling
  the counts is what gives the interval its width, and keeping the seeds
  apart is the only way to see that one of them disagrees;
* **one deduplicated position file**, gzipped, that the exact solver can
  consume and the corpus merge can absorb. Runs overlap heavily near the
  root, so concatenating them would send the same opening to the solver
  four times.

Deduplication is on the canonical key — up to the 192 symmetries — for the
same reason the corpus merge is: two boards related by a rotation or a
shape relabeling are one position, and solving both wastes the solver and
double-counts the coverage.
"""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..env import fastboard as fb
from .match import wilson_ci


@dataclass(frozen=True)
class RunSummary:
    """One arena directory's identity and leaderboard."""

    name: str
    seed: int
    games: int
    leaderboard: list[dict]

    @property
    def by_agent(self) -> dict[str, dict]:
        return {row["agent"]: row for row in self.leaderboard}


def read_run(run_dir: Path) -> RunSummary:
    payload = json.loads((run_dir / "games.json").read_text())
    return RunSummary(
        name=run_dir.name,
        seed=int(payload["seed"]),
        games=int(payload["games"]),
        leaderboard=payload["leaderboard"],
    )


def pooled(runs: list[RunSummary]) -> list[dict]:
    """Win/game totals summed across runs, with a 95% Wilson interval.

    Summing counts rather than averaging rates is deliberate: the runs have
    the same size here, but averaging rates would silently weight a short
    run equally with a long one the first time they differ.
    """
    tally: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for run in runs:
        for row in run.leaderboard:
            tally[row["agent"]][0] += int(row["wins"])
            tally[row["agent"]][1] += int(row["games"])
    out = []
    for agent, (wins, games) in tally.items():
        low, high = wilson_ci(wins, games)
        out.append(
            {
                "agent": agent,
                "wins": wins,
                "games": games,
                "win_rate": wins / games if games else 0.0,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return sorted(out, key=lambda row: -row["win_rate"])


def seed_spread(runs: list[RunSummary]) -> dict[str, float]:
    """Widest gap between any two runs' win rates, per agent.

    The number to read against the pooled interval: a spread much wider
    than the interval means the seeds disagree and the pooled figure is
    hiding it.
    """
    spread: dict[str, float] = {}
    for agent in {row["agent"] for run in runs for row in run.leaderboard}:
        rates = [run.by_agent[agent]["win_rate"] for run in runs if agent in run.by_agent]
        spread[agent] = max(rates) - min(rates) if len(rates) > 1 else 0.0
    return spread


def merge_qfens(run_dirs: list[Path]) -> list[str]:
    """Every run's `to-solve.qfen`, deduplicated up to symmetry."""
    lines: list[str] = []
    for run_dir in run_dirs:
        path = run_dir / "to-solve.qfen"
        if path.exists():
            lines.extend(line for line in path.read_text().splitlines() if line.strip())
    if not lines:
        return []
    boards = np.array([fb.from_qfen(line) for line in lines], dtype=np.uint16)
    keys = fb.canonical_keys(boards)
    _, first = np.unique(keys, return_index=True)
    return [lines[i] for i in sorted(first.tolist())]


def write_gzip(text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(text)
    return path


def summarise(runs: list[RunSummary], spread: dict[str, float]) -> str:
    """The markdown table, so no number in the write-up is hand-typed."""
    lines = ["| agent | win rate | 95% CI | games | widest seed gap |", "|---|---|---|---|---|"]
    for row in pooled(runs):
        lines.append(
            f"| `{row['agent']}` | {row['win_rate']:.1%} | "
            f"{row['ci_low']:.1%}–{row['ci_high']:.1%} | {row['games']:,} | "
            f"{spread.get(row['agent'], 0.0):.1%} |"
        )
    lines.append("")
    lines.append("Per run:")
    lines.append("")
    header = sorted({row["agent"] for run in runs for row in run.leaderboard})
    lines.append("| run | seed | " + " | ".join(f"`{a}`" for a in header) + " |")
    lines.append("|---" * (len(header) + 2) + "|")
    for run in runs:
        cells = [
            f"{run.by_agent[a]['win_rate']:.1%}" if a in run.by_agent else "—"
            for a in header
        ]
        lines.append(f"| {run.name} | {run.seed} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def pack(run_dirs: list[Path], out: Path) -> dict:
    """Write the pooled summary and the deduplicated position file."""
    runs = [read_run(d) for d in run_dirs if (d / "games.json").exists()]
    if not runs:
        raise ValueError(f"no arena runs with a games.json among {run_dirs}")
    spread = seed_spread(runs)
    qfens = merge_qfens(run_dirs)

    out.mkdir(parents=True, exist_ok=True)
    write_gzip("\n".join(qfens) + "\n", out / "to-solve.qfen.gz")
    for run_dir in run_dirs:
        games = run_dir / "games.json"
        if games.exists():
            write_gzip(games.read_text(), out / f"games-{run_dir.name}.json.gz")

    summary = {
        "runs": [{"name": r.name, "seed": r.seed, "games": r.games} for r in runs],
        "pooled": pooled(runs),
        "seed_spread": spread,
        "positions_to_solve": len(qfens),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "summary.md").write_text(summarise(runs, spread) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args(argv)
    summary = pack(args.runs, args.out)
    print((args.out / "summary.md").read_text())
    print(f"{summary['positions_to_solve']:,} positions to solve -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
