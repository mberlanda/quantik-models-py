"""Figures for docs/opening-arena-ply0.md, drawn from the arena games.json files.

Usage:
    python scripts/plot_opening_arena.py RUN_A RUN_B [--out docs/figures]

RUN_A and RUN_B are the two run directories (Config A, Config B); each holds
`policy-p0/games.json`, `mcts-p0/games.json` and so on. The figures plot only
numbers recomputed from the `results` lists with `quantik_models.arena.autoplay`.
The output is deterministic: fixed svg.hashsalt, no embedded date.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from quantik_models.arena.autoplay import Game, distinct_games

GROUND, INK, GRID = "#FFFFFF", "#1A1A1E", "#D8D8DE"
MOVER, RESPONDER, FAIL = "#1F6FB2", "#C98A1E", "#C1483B"
SPEC_ORDER = ["cpool", "attn", "resnet", "mlp", "patience-cpool", "patience-cpool-v2"]
THRESHOLD = 0.5  # autoplay.py warns when the worst-pairing ratio is below this


def load_games(path: Path) -> list[Game]:
    data = json.loads(path.read_text())
    return [
        Game(
            mover=r["mover"],
            responder=r["responder"],
            actions=list(r["actions"]),
            winner=0 if r["winner"] == r["mover"] else 1,
        )
        for r in data["results"]
    ]


def worst_ratio(games: list[Game]) -> tuple[str, int, int]:
    cells = distinct_games(games)
    (mv, rs), c = min(cells.items(), key=lambda kv: kv[1]["distinct"] / kv[1]["games"])
    return f"{mv} vs {rs}", c["distinct"], c["games"]


def seat_rates(games: list[Game]) -> dict[str, dict[str, tuple[int, int]]]:
    """{agent: {"mover": (wins, games), "responder": (wins, games)}}, order not kept."""
    out: dict[str, dict[str, list[int]]] = {}
    for g in games:
        for name, seat in ((g.mover, "mover"), (g.responder, "responder")):
            out.setdefault(name, {"mover": [0, 0], "responder": [0, 0]})[seat][1] += 1
        winner_seat, name = ("mover", g.mover) if g.winner == 0 else ("responder", g.responder)
        out[name][winner_seat][0] += 1
    return {n: {s: (v[0], v[1]) for s, v in seats.items()} for n, seats in out.items()}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["svg.hashsalt"] = "quantik-models"
    import matplotlib.pyplot as plt

    return plt


def _style(ax) -> None:
    ax.set_facecolor(GROUND)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=8)
    ax.set_axisbelow(True)


def _save(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(GROUND)
    fig.savefig(out, format="svg", bbox_inches="tight", facecolor=GROUND, metadata={"Date": None})
    _plt().close(fig)


def audit_figure(cells: list[tuple[str, list[Game]]], out: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    _style(ax)
    ax.grid(True, axis="x", color=GRID, linewidth=0.6, alpha=0.7)
    labels, ratios = [], []
    for i, (label, games) in enumerate(cells):
        name, d, n = worst_ratio(games)
        ratio = d / n
        fail = ratio < THRESHOLD
        labels.append(label)
        ratios.append(ratio)
        ax.barh(i, ratio, color=FAIL if fail else MOVER, hatch="///" if fail else None,
                edgecolor="white" if fail else MOVER, linewidth=0, height=0.62)
        ax.text(ratio + 0.01, i, f"{d}/{n} = {ratio * 100:.1f}%" + ("  FAIL" if fail else ""),
                va="center", fontsize=8, color=FAIL if fail else INK,
                fontweight="bold" if fail else "normal",
                bbox={"facecolor": GROUND, "edgecolor": "none", "pad": 1.5} if fail else None)
    ax.axvline(THRESHOLD, color=INK, linestyle="--", linewidth=1.0)
    ax.text(THRESHOLD, -0.75, "0.5 threshold", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.25)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("distinct games / games, worst ordered pairing", color=INK, fontsize=9)
    ax.set_title("Distinct-games audit, worst pairing per run", color=INK, loc="left", fontsize=11, pad=16)
    _save(fig, out)


def _seat_panel(ax, title: str, games: list[Game]) -> None:
    _style(ax)
    ax.grid(True, axis="x", color=GRID, linewidth=0.6, alpha=0.7)
    rates = seat_rates(games)
    names = list(rates)
    # Spec order, as in the document's tables; the control goes last.
    names.sort(key=lambda n: SPEC_ORDER.index(n.split("-mcts")[0]) if n.split("-mcts")[0] in SPEC_ORDER else len(SPEC_ORDER))
    h = 0.38
    for i, n in enumerate(names):
        for off, seat, col in ((-h / 2, "mover", MOVER), (h / 2, "responder", RESPONDER)):
            w, g = rates[n][seat]
            ax.barh(i + off, w / g, height=h, color=col)
            ax.text(w / g + 0.01, i + off, f"{w / g * 100:.1f}", va="center", fontsize=6.5, color=INK)
    ax.set_yticks(range(len(names)), [n + (" (control)" if n.startswith("uniform") else "") for n in names])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25", "50", "75", "100"])
    ax.axvline(0.5, color=INK, linestyle=":", linewidth=0.8)
    ax.set_xlabel("win rate in seat (%)", color=INK, fontsize=8)
    ax.set_title(title, color=INK, loc="left", fontsize=9)


def _legend(fig) -> None:
    from matplotlib.patches import Patch

    fig.legend(handles=[Patch(color=MOVER, label="mover"), Patch(color=RESPONDER, label="responder")],
               loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))


def seat_figure(panels: list[tuple[str, list[Game]]], out: Path, rows: int, cols: int,
                size: tuple[float, float], suptitle: str) -> None:
    plt = _plt()
    fig, axes = plt.subplots(rows, cols, figsize=size, sharex=True)
    axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for ax, (title, games) in zip(axes_flat, panels):
        _seat_panel(ax, title, games)
    fig.suptitle(suptitle, color=INK, x=0.0, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    _legend(fig)
    _save(fig, out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_a", type=Path, help="Config A run directory (TEMP_PLIES=4)")
    p.add_argument("run_b", type=Path, help="Config B run directory (TEMP_PLIES=6)")
    p.add_argument("--out", type=Path, default=Path("docs/figures"))
    a = p.parse_args(argv)

    g = {
        ("A", "policy", 0): load_games(a.run_a / "policy-p0" / "games.json"),
        ("A", "mcts", 0): load_games(a.run_a / "mcts-p0" / "games.json"),
        ("A", "policy", 1): load_games(a.run_a / "policy-p1" / "games.json"),
        ("A", "mcts", 1): load_games(a.run_a / "mcts-p1" / "games.json"),
        ("B", "policy", 0): load_games(a.run_b / "policy-p0" / "games.json"),
        ("B", "mcts", 0): load_games(a.run_b / "mcts-p0" / "games.json"),
    }
    names = {"policy": "policy", "mcts": "MCTS"}
    cells = [(f"{c}, ply {ply}, {names[f]}", games) for (c, f, ply), games in g.items()]
    audit_figure(cells, a.out / "opening-arena-audit.svg")
    seat_figure(
        [
            ("Config A, ply 0, policy", g["A", "policy", 0]),
            ("Config A, ply 1, policy", g["A", "policy", 1]),
            ("Config A, ply 1, MCTS 128 sims", g["A", "mcts", 1]),
            ("Config B, ply 0, MCTS 128 sims", g["B", "mcts", 0]),
        ],
        a.out / "opening-arena-seats.svg", 2, 2, (9.6, 7.6),
        "Win rate by seat, per agent",
    )
    seat_figure(
        [("Config A (TEMP_PLIES=4), ply 0, policy", g["A", "policy", 0]),
         ("Config B (TEMP_PLIES=6), ply 0, policy", g["B", "policy", 0])],
        a.out / "opening-arena-policy-ab.svg", 1, 2, (9.6, 3.6),
        "Win rate by seat, policy agents, start ply 0",
    )
    # Echo the plotted audit numbers so they can be checked against the doc.
    for label, games in cells:
        name, d, n = worst_ratio(games)
        dc = [c["distinct"] for c in distinct_games(games).values()]
        print(f"{label}: {name} {d}/{n} min/median/max {min(dc)}/{statistics.median(dc)}/{max(dc)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
