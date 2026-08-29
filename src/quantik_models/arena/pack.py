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
    start_plies: int | None = None

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
        start_plies=_start_plies(payload, run_dir),
    )


def _start_plies(payload: dict, run_dir: Path) -> int | None:
    """The depth the run started from, or None.

    Falls back to a `-p<N>` suffix on the directory name for runs written
    before `autoplay` recorded the field. The fallback is brittle by nature,
    so it is only a fallback: a run with neither is reported as unknown
    rather than silently grouped with the ply-3 runs.
    """
    if "start_plies" in payload:
        return int(payload["start_plies"])
    _, _, tail = run_dir.name.rpartition("-p")
    return int(tail) if tail.isdigit() else None


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
    """Widest gap between two runs' win rates *at the same start depth*.

    The number to read against the pooled interval: a spread much wider
    than the interval means the seeds disagree and the pooled figure is
    hiding it.

    Grouping by start depth is not a refinement, it is the whole point. Two
    runs at plies 3 and 6 are not replicates — the ranking genuinely moves
    with depth — and comparing them here reports a real depth effect as
    seed noise. A first version of this function did exactly that.
    """
    by_depth: dict[int | None, list[RunSummary]] = defaultdict(list)
    for run in runs:
        by_depth[run.start_plies].append(run)
    spread: dict[str, float] = {}
    for agent in {row["agent"] for run in runs for row in run.leaderboard}:
        widest = 0.0
        for group in by_depth.values():
            rates = [r.by_agent[agent]["win_rate"] for r in group if agent in r.by_agent]
            if len(rates) > 1:
                widest = max(widest, max(rates) - min(rates))
        spread[agent] = widest
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


def pairwise(run_dirs: list[Path], a: str, b: str) -> dict:
    """`a`'s record against `b` alone, pooled across seats and split by them.

    `head_to_head` answers "how does everyone do against this one oracle",
    which is the right question when a fixed reference opponent exists. It
    is the wrong one for comparing two candidates to each other: the
    leaderboard mixes in every other agent on the card, so two networks
    that never beat each other can still finish points apart because they
    met a third one different numbers of times.

    The seat split is not decoration. From a ply-3 start the mover wins
    68-88% of these games whoever is moving, so a pooled rate compares two
    networks inside an effect an order of magnitude larger than the
    difference between them. A pairing is only balanced if both seat counts
    are equal, and `balanced` says whether it was.
    """
    mover_wins = mover_games = responder_wins = responder_games = 0
    for run_dir in run_dirs:
        path = run_dir / "games.json"
        if not path.exists():
            continue
        for game in json.loads(path.read_text())["results"]:
            pair = {game["mover"], game["responder"]}
            if pair != {a, b}:
                continue
            if game["mover"] == a:
                mover_games += 1
                mover_wins += game["winner"] == a
            else:
                responder_games += 1
                responder_wins += game["winner"] == a

    wins = mover_wins + responder_wins
    games = mover_games + responder_games
    low, high = wilson_ci(wins, games) if games else (0.0, 1.0)
    return {
        "agent": a,
        "opponent": b,
        "wins": wins,
        "games": games,
        "win_rate": wins / games if games else 0.0,
        "ci_low": low,
        "ci_high": high,
        "as_mover": mover_wins / mover_games if mover_games else 0.0,
        "as_responder": responder_wins / responder_games if responder_games else 0.0,
        "balanced": mover_games == responder_games,
        # The only claim the interval supports on its own: 50% is outside
        # it. With twenty such intervals in a lineup, roughly one is
        # expected to exclude 50% by chance, so a single true here is
        # weaker evidence than it looks.
        "separated": games > 0 and not (low <= 0.5 <= high),
    }


def merge_qfens(run_dirs: list[Path], corpus: Path | None = None) -> list[str]:
    """Every run's `to-solve.qfen`, deduplicated up to symmetry.

    `corpus` re-filters the result **at pack time**, which is not the same as
    the filtering `autoplay` already did. The arena filters against whatever
    corpus it was pointed at while the games were being played, and that
    corpus can be superseded before anyone spends the queue on a solver.
    That is exactly what happened here: the first oracle runs filtered
    against `exact-sampled.npz` while `exact-sampled-v2.npz` already existed,
    and 35% of the resulting 26,157-position queue was already labelled —
    about twelve hours of solver time. Filtering later is strictly better,
    because later is when the queue is actually spent.
    """
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
    kept = sorted(first.tolist())

    if corpus is not None:
        from ..data.exact_corpus import ExactCorpus

        known = fb.canonical_keys(ExactCorpus.load(corpus).boards)
        keep = ~np.isin(keys[kept], known)
        kept = [i for i, k in zip(kept, keep.tolist()) if k]
    return [lines[i] for i in kept]


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
    lines.append(
        "Widest seed gap is within one start depth: runs at different depths "
        "are not replicates of each other."
    )
    lines.append("")
    lines.append("Per run:")
    lines.append("")
    header = sorted({row["agent"] for run in runs for row in run.leaderboard})
    lines.append("| run | seed | start ply | " + " | ".join(f"`{a}`" for a in header) + " |")
    lines.append("|---" * (len(header) + 3) + "|")
    for run in runs:
        cells = [
            f"{run.by_agent[a]['win_rate']:.1%}" if a in run.by_agent else "—"
            for a in header
        ]
        depth = "—" if run.start_plies is None else str(run.start_plies)
        lines.append(f"| {run.name} | {run.seed} | {depth} | " + " | ".join(cells) + " |")

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


def pack(
    run_dirs: list[Path],
    out: Path,
    oracle: str | None = None,
    corpus: Path | None = None,
) -> dict:
    """Write the pooled summary and the deduplicated position file."""
    runs = [read_run(d) for d in run_dirs if (d / "games.json").exists()]
    if not runs:
        raise ValueError(f"no arena runs with a games.json among {run_dirs}")
    spread = seed_spread(runs)
    pooled_rows = pooled(runs)
    # The opponent everything else played: it appears in every pairing and
    # each other agent appears only in its own, so it has the most games.
    oracle = oracle or max(pooled_rows, key=lambda row: row["games"])["agent"]
    qfens = merge_qfens(run_dirs, corpus)

    out.mkdir(parents=True, exist_ok=True)
    write_gzip("\n".join(qfens) + "\n", out / "to-solve.qfen.gz")
    for run_dir in run_dirs:
        games = run_dir / "games.json"
        if games.exists():
            write_gzip(games.read_text(), out / f"games-{run_dir.name}.json.gz")

    h2h = head_to_head(run_dirs, oracle)
    summary: dict[str, Any] = {
        "runs": [
            {"name": r.name, "seed": r.seed, "games": r.games, "start_plies": r.start_plies}
            for r in runs
        ],
        "oracle": oracle,
        "pooled": pooled_rows,
        "head_to_head": h2h,
        "seed_spread": spread,
        "positions_to_solve": len(qfens),
        "filtered_against": str(corpus) if corpus else None,
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
        "--corpus",
        type=Path,
        default=None,
        help="re-filter the solver queue against this corpus, at pack time",
    )
    parser.add_argument(
        "--oracle",
        default=None,
        help="the fixed opponent; inferred from the game counts when omitted",
    )
    args = parser.parse_args(argv)
    summary = pack(args.runs, args.out, args.oracle, args.corpus)
    print((args.out / "summary.md").read_text())
    print(f"{summary['positions_to_solve']:,} positions to solve -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
