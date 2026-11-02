"""Figures for docs/patience-arena-second-seed.md, from the two runs' games.json.

Usage:
    python scripts/plot_patience_second_seed.py [RUN_A RUN_B] [--out DIR]

RUN_A and RUN_B are run directories that each hold `<arena>/games.json` for
`policy-p3`, `policy-p6`, `policy-p9`, `mcts-p3` and `mcts-p6`. They default
to the two committed-by-name runs under `runs/eval/` (which is gitignored,
so pass the directories explicitly on a machine that keeps them elsewhere).
Figures go to `docs/figures/patience-second-seed-*.svg`.

Every number is recomputed from the per-game records with
`quantik_models.arena.autoplay` and matches the tables in the document.
Output is byte-stable: fixed hash salt, no embedded date.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from quantik_models.arena.autoplay import Game, distinct_games
from quantik_models.report.figures import (
    CONTROL_COLOUR,
    GRID,
    INK,
    colour_for,
    _new_axes,
    _save,
)

DEFAULT_RUNS = (
    Path("runs/eval/patience-2026-08-30"),
    Path("runs/eval/patience-seed-20261015"),
)
DEFAULT_OUT = Path("docs/figures")
ARENAS = ("policy-p3", "policy-p6", "policy-p9", "mcts-p3", "mcts-p6")
NETWORKS = ("resnet", "mlp", "cpool", "attn")
Z = 1.96
SEED_STYLE = (("o", 0.45), ("s", 1.0))  # marker, bar alpha: first run, second run


def load_games(path: Path) -> tuple[int, list[Game]]:
    """The run's seed and its games, rebuilt from `results`."""
    data = json.loads(path.read_text())
    games = [
        Game(
            mover=r["mover"],
            responder=r["responder"],
            actions=list(r["actions"]),
            winner=0 if r["winner"] == r["mover"] else 1,
        )
        for r in data["results"]
    ]
    return int(data["seed"]), games


def seat_rate(games: list[Game], agent: str, seat: int) -> dict[str, tuple[int, int]]:
    """`agent`'s wins/games in one seat (0 mover, 1 responder), raw and distinct."""
    mine = [g for g in games if (g.mover if seat == 0 else g.responder) == agent]
    # Distinct games are counted per ordered pairing, as the arena counts them.
    seen: dict[tuple[str, str, tuple[int, ...]], bool] = {}
    for g in mine:
        seen[(g.mover, g.responder, tuple(g.actions))] = g.winner == seat
    return {
        "raw": (sum(g.winner == seat for g in mine), len(mine)),
        "distinct": (sum(seen.values()), len(seen)),
    }


def margin(games: list[Game], seat: int, basis: str) -> tuple[float, float]:
    """cpool minus attn win rate in a seat, in points, with a Wald 95% half-width."""
    ps = []
    for suffix in ("cpool", "attn"):
        agent = next(
            n for g in games for n in (g.mover, g.responder) if n.startswith(suffix)
        )
        w, n = seat_rate(games, agent, seat)[basis]
        ps.append((w / n, n))
    (pc, nc), (pa, na) = ps
    half = Z * math.sqrt(pc * (1 - pc) / nc + pa * (1 - pa) / na)
    return 100 * (pc - pa), 100 * half


def audit(games: list[Game]) -> float:
    """The worst pairing's distinct games per game played."""
    return min(v["distinct"] / v["games"] for v in distinct_games(games).values())


def _legend_handles(labels: tuple[str, ...]):
    from matplotlib.lines import Line2D

    return [
        Line2D([], [], color=INK, marker=m, linestyle="", markersize=6, label=lab)
        for (m, _), lab in zip(SEED_STYLE, labels)
    ]


def fig_margin(runs, out: Path) -> Path:
    """cpool minus attn per seat at MCTS-128, one point per seed per rate basis."""
    labels = tuple(f"seed {seed}" for seed, _ in runs)
    fig, ax = _new_axes(7.0, 4.0)
    groups = [(p, s) for p in ("mcts-p3", "mcts-p6") for s in (0, 1)]
    offsets = {(0, "raw"): -0.27, (0, "distinct"): -0.09, (1, "raw"): 0.09, (1, "distinct"): 0.27}
    for gi, (arena, seat) in enumerate(groups):
        for ri, (_, by_arena) in enumerate(runs):
            marker, _ = SEED_STYLE[ri]
            for basis in ("raw", "distinct"):
                m, h = margin(by_arena[arena], seat, basis)
                ax.errorbar(
                    gi + offsets[(ri, basis)],
                    m,
                    yerr=h,
                    fmt=marker,
                    color=INK,
                    markerfacecolor=INK if basis == "raw" else "white",
                    markersize=6,
                    elinewidth=1.1,
                    capsize=3,
                )
    ax.axhline(0, color=INK, linewidth=1.0, linestyle=":")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(
        [f"ply {a[-1]}, {'mover' if s == 0 else 'responder'}" for a, s in groups]
    )
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylabel("cpool minus attn win rate (points)", color=INK)
    ax.set_title(
        "cpool minus attn, MCTS-128, per seat", color=INK, loc="left", fontsize=11
    )
    from matplotlib.lines import Line2D

    handles = _legend_handles(labels) + [
        Line2D([], [], color=INK, marker="o", linestyle="", markerfacecolor=INK, label="raw games"),
        Line2D([], [], color=INK, marker="o", linestyle="", markerfacecolor="white", label="distinct games"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8, ncol=4, loc="upper left",
              bbox_to_anchor=(0.0, -0.1))
    return _save(fig, out)


def fig_seats(runs, out: Path) -> Path:
    """Mover and responder win rate per network, MCTS-128, plies 3 and 6."""
    import matplotlib.pyplot as plt

    labels = tuple(f"seed {seed}" for seed, _ in runs)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6), sharey=True)
    for row, arena in enumerate(("mcts-p3", "mcts-p6")):
        for col, seat in enumerate((0, 1)):
            ax = axes[row][col]
            ax.set_facecolor("#FFFFFF")
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            for spine in ("left", "bottom"):
                ax.spines[spine].set_color(GRID)
            ax.tick_params(colors=INK, labelsize=9)
            ax.grid(True, axis="y", color=GRID, linewidth=0.6, alpha=0.7)
            ax.set_axisbelow(True)
            for ai, net in enumerate(NETWORKS):
                for ri, (_, by_arena) in enumerate(runs):
                    games = by_arena[arena]
                    agent = f"{net}-mcts128"
                    w, n = seat_rate(games, agent, seat)["raw"]
                    x = ai + (ri - 0.5) * 0.38
                    ax.bar(x, 100 * w / n, width=0.36, color=colour_for(net),
                           alpha=SEED_STYLE[ri][1])
                    ax.text(x, 100 * w / n + 1, f"{100 * w / n:.1f}", ha="center",
                            va="bottom", fontsize=7, color=INK)
            ax.set_xticks(range(len(NETWORKS)))
            ax.set_xticklabels(NETWORKS)
            ax.set_ylim(0, 100)
            ax.set_title(
                f"ply {arena[-1]}, {'mover' if seat == 0 else 'responder'}",
                color=INK, loc="left", fontsize=10,
            )
            if col == 0:
                ax.set_ylabel("win rate (%)", color=INK)
    from matplotlib.patches import Patch

    fig.legend(
        handles=[Patch(facecolor=INK, alpha=a, label=lab)
                 for (_, a), lab in zip(SEED_STYLE, labels)],
        frameon=False, fontsize=8, ncol=2, loc="lower center",
    )
    fig.suptitle("MCTS-128 win rate by seat", color=INK, x=0.07, ha="left",
                 fontsize=11)
    fig.patch.set_facecolor("#FFFFFF")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    return _save(fig, out)


def fig_audit(runs, out: Path) -> Path:
    """Worst-pairing distinct games per game played, per arena, against 0.5."""
    labels = tuple(f"seed {seed}" for seed, _ in runs)
    fig, ax = _new_axes(7.0, 4.0)
    for ai, arena in enumerate(ARENAS):
        for ri, (_, by_arena) in enumerate(runs):
            v = audit(by_arena[arena])
            x = ai + (ri - 0.5) * 0.38
            ax.bar(x, v, width=0.36, color=CONTROL_COLOUR if v >= 0.5 else "#C1483B",
                   alpha=SEED_STYLE[ri][1])
            ax.text(x, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=7,
                    color=INK)
    ax.axhline(0.5, color=INK, linewidth=1.0, linestyle=":")
    ax.text(len(ARENAS) - 0.45, 0.51, "0.5 threshold", ha="right", va="bottom",
            fontsize=8, color=INK)
    ax.set_xticks(range(len(ARENAS)))
    ax.set_xticklabels(ARENAS)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("distinct games per game played", color=INK)
    ax.set_title("Distinct-games audit, worst pairing per arena", color=INK,
                 loc="left", fontsize=11)
    from matplotlib.patches import Patch

    ax.legend(
        handles=[Patch(facecolor=CONTROL_COLOUR, alpha=a, label=lab)
                 for (_, a), lab in zip(SEED_STYLE, labels)]
        + [Patch(facecolor="#C1483B", label="below 0.5")],
        frameon=False, fontsize=8, ncol=3, loc="upper left", bbox_to_anchor=(0.0, -0.1),
    )
    return _save(fig, out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="*", type=Path,
                    help="two run directories: original seed, then new seed")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    run_dirs = tuple(args.runs) if args.runs else DEFAULT_RUNS
    if len(run_dirs) != 2:
        ap.error("give exactly two run directories (or none for the defaults)")

    runs = []
    for d in run_dirs:
        seed = 0
        by_arena = {}
        for arena in ARENAS:
            seed, by_arena[arena] = load_games(d / arena / "games.json")
        runs.append((seed, by_arena))

    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "quantik-models"
    for path in (
        fig_margin(runs, args.out / "patience-second-seed-margin.svg"),
        fig_seats(runs, args.out / "patience-second-seed-seats.svg"),
        fig_audit(runs, args.out / "patience-second-seed-audit.svg"),
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
