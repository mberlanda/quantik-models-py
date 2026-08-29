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
from typing import Any

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


def head_to_head(run_dirs: list[Path], oracle: str) -> list[dict]:
    """Each agent's record against `oracle`, split by seat.

    The leaderboard aggregates the seats away, and the seat is not a detail
    here: from a ply-3 start the mover wins most games regardless of who is
    moving, so a single pooled win rate against a fixed opponent mixes the
    agent's strength with the first-move advantage. Splitting them is what
    makes "does this network beat minimax" answerable.
    """
    seats: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: {"as_mover": [0, 0], "as_responder": [0, 0]}
    )
    for run_dir in run_dirs:
        path = run_dir / "games.json"
        if not path.exists():
            continue
        for game in json.loads(path.read_text())["results"]:
            mover, responder = game["mover"], game["responder"]
            if oracle not in (mover, responder):
                continue
            agent = responder if mover == oracle else mover
            seat = "as_responder" if mover == oracle else "as_mover"
            seats[agent][seat][1] += 1
            if game["winner"] == agent:
                seats[agent][seat][0] += 1

    out = []
    for agent, record in seats.items():
        wins = record["as_mover"][0] + record["as_responder"][0]
        games = record["as_mover"][1] + record["as_responder"][1]
        low, high = wilson_ci(wins, games)
        out.append(
            {
                "agent": agent,
                "wins": wins,
                "games": games,
                "win_rate": wins / games if games else 0.0,
                "ci_low": low,
                "ci_high": high,
                "as_mover": record["as_mover"][0] / record["as_mover"][1]
                if record["as_mover"][1]
                else 0.0,
                "as_responder": record["as_responder"][0] / record["as_responder"][1]
                if record["as_responder"][1]
                else 0.0,
                # Above 0.5 the agent is beating the oracle on that seat's
                # own terms; the gap between the two is the seat advantage.
                "beats_oracle": low > 0.5,
                "loses_to_oracle": high < 0.5,
            }
        )
    return sorted(out, key=lambda row: -row["win_rate"])


def merge_qfens(run_dirs: list[Path]) -> list[str]:
    """Every run's `to-solve.qfen`, deduplicated up to symmetry."""
    lines: list[str] = []
    for run_dir in run_dirs:
        path = run_dir / "to-solve.qfen"
        if path.exists():
            lines.extend(line for line in path.read_text().splitlines() if line.strip())
    if not lines:
        return []
    # `from_qfen` returns a (1, 8) batch of one board, so concatenate rather
    # than stack — np.array over the list would give (n, 1, 8).
    boards = np.concatenate([fb.from_qfen(line) for line in lines]).astype(np.uint16)
    keys = fb.canonical_keys(boards)
    _, first = np.unique(keys, return_index=True)
    return [lines[i] for i in sorted(first.tolist())]


def write_gzip(text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(text)
    return path


def summarise(
    runs: list[RunSummary],
    spread: dict[str, float],
    h2h: list[dict] | None = None,
    oracle: str | None = None,
) -> str:
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

    if h2h and oracle:
        lines += [
            "",
            f"Head to head against `{oracle}`, by seat:",
            "",
            "| agent | win rate | 95% CI | as mover | as responder | verdict |",
            "|---|---|---|---|---|---|",
        ]
        for row in h2h:
            if row["beats_oracle"]:
                verdict = "beats it"
            elif row["loses_to_oracle"]:
                verdict = "loses to it"
            else:
                verdict = "indistinguishable"
            lines.append(
                f"| `{row['agent']}` | {row['win_rate']:.1%} | "
                f"{row['ci_low']:.1%}–{row['ci_high']:.1%} | "
                f"{row['as_mover']:.1%} | {row['as_responder']:.1%} | {verdict} |"
            )
    return "\n".join(lines)


def pack(run_dirs: list[Path], out: Path, oracle: str | None = None) -> dict:
    """Write the pooled summary and the deduplicated position file."""
    runs = [read_run(d) for d in run_dirs if (d / "games.json").exists()]
    if not runs:
        raise ValueError(f"no arena runs with a games.json among {run_dirs}")
    spread = seed_spread(runs)
    pooled_rows = pooled(runs)
    # The opponent everything else played: it appears in every pairing and
    # each other agent appears only in its own, so it has the most games.
    oracle = oracle or max(pooled_rows, key=lambda row: row["games"])["agent"]
    qfens = merge_qfens(run_dirs)

    out.mkdir(parents=True, exist_ok=True)
    write_gzip("\n".join(qfens) + "\n", out / "to-solve.qfen.gz")
    for run_dir in run_dirs:
        games = run_dir / "games.json"
        if games.exists():
            write_gzip(games.read_text(), out / f"games-{run_dir.name}.json.gz")

    h2h = head_to_head(run_dirs, oracle)
    summary: dict[str, Any] = {
        "runs": [{"name": r.name, "seed": r.seed, "games": r.games} for r in runs],
        "oracle": oracle,
        "pooled": pooled_rows,
        "head_to_head": h2h,
        "seed_spread": spread,
        "positions_to_solve": len(qfens),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "summary.md").write_text(summarise(runs, spread, h2h, oracle) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument(
        "--oracle",
        default=None,
        help="the fixed opponent; inferred from the game counts when omitted",
    )
    args = parser.parse_args(argv)
    summary = pack(args.runs, args.out, args.oracle)
    print((args.out / "summary.md").read_text())
    print(f"{summary['positions_to_solve']:,} positions to solve -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
