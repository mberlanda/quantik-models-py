"""Figures for the benchmark report, generated from the run directories.

Every figure here is a picture of a claim that a table makes hard to see:
the attention network sitting flat for sixteen epochs, the learning-rate
optimum being an inverted U rather than a trend, the per-ply accuracy
crossover, the arena ranking moving with start depth. None of them
introduce a number — each one plots a file that already exists under
`runs/`, which is gitignored, so the committed SVGs are the only form in
which a fresh clone can see them.

Output is SVG with `svg.fonttype = "none"`: the text stays text, so the
files diff line by line in review instead of arriving as opaque blobs of
glyph outlines.

The palette is fixed per architecture and shared across every figure, so a
colour means the same model wherever it appears. Figures paint an explicit
white ground rather than a transparent one — they are read on GitHub, which
serves the same file to a light and a dark reader, and a transparent ground
puts dark axis text on a dark page.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ..arena.match import wilson_ci

# One colour per architecture, used by every figure in the report.
COLOURS: dict[str, str] = {
    "resnet": "#C1483B",
    "mlp": "#7A7A82",
    "cpool": "#1F6FB2",
    "attn": "#C98A1E",
}
# The control agent the MCTS arena carries; grey and dashed everywhere.
CONTROL_COLOUR = "#B0B0B8"

GROUND = "#FFFFFF"
INK = "#1A1A1E"
GRID = "#D8D8DE"


def colour_for(agent: str) -> str:
    """The palette entry for an agent name, matched by architecture prefix.

    Arena agents carry suffixes (`cpool-mcts128`), so the lookup is by
    prefix rather than by equality.
    """
    for arch, colour in COLOURS.items():
        if agent.startswith(arch):
            return colour
    return CONTROL_COLOUR


# --------------------------------------------------------------------------
# Loaders. Each takes an explicit path so the tests can build a synthetic run
# directory: `runs/` is gitignored and absent on a CI runner.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingRun:
    """One training run's per-epoch metrics."""

    name: str
    epochs: list[int]
    val_top1: list[float]
    lr: float

    @property
    def best_top1(self) -> float:
        return max(self.val_top1)


def load_training_run(run_dir: Path, name: str | None = None) -> TrainingRun:
    """Read `metrics.jsonl` from one training run directory."""
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{run_dir}/metrics.jsonl is empty")
    return TrainingRun(
        name=name or run_dir.name,
        epochs=[int(r["epoch"]) for r in rows],
        val_top1=[float(r["val_top1"]) for r in rows],
        # The scheduler decays within the run, so the *configured* rate is the
        # first row's — the last row's would report the floor of the decay.
        lr=float(rows[0]["lr"]),
    )


def load_sweep(sweep_dir: Path) -> dict[str, dict[float, float]]:
    """`{architecture: {lr: final val_top1}}` from a `sweep-{arch}-{lr}` tree."""
    out: dict[str, dict[float, float]] = {}
    for run_dir in sorted(sweep_dir.glob("sweep-*")):
        arch, lr = run_dir.name.removeprefix("sweep-").split("-", 1)
        run = load_training_run(run_dir)
        out.setdefault(arch, {})[float(lr)] = run.val_top1[-1]
    return out


def load_shift(path: Path) -> list[dict]:
    """The per-model records written by `quantik_models.eval.shift`."""
    return json.loads(path.read_text())


def load_leaderboard(games_json: Path) -> list[dict]:
    """`[{agent, wins, games, win_rate}]` from one arena run."""
    return json.loads(games_json.read_text())["leaderboard"]


def short_arch(architecture: str) -> str:
    """`cpool-c191-b6` -> `cpool`; the key the palette is indexed by."""
    return architecture.split("-", 1)[0]


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def _new_axes(width: float, height: float):
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.fonttype"] = "none"
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor(GROUND)
    ax.set_facecolor(GROUND)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    return fig, ax


def _save(fig, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="svg", bbox_inches="tight", facecolor=GROUND)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return out


def training_curves(runs: Iterable[TrainingRun], out: Path, superseded: set[str] = frozenset()) -> Path:
    """Validation top-1 against epoch, one line per run.

    Runs named in `superseded` are drawn dashed: they were trained at a rate
    inherited from the ResNet and their numbers no longer stand. Drawing them
    beside the corrected runs is the point of the figure — the gap between a
    dashed and a solid line of the same colour is what the learning rate cost.
    """
    fig, ax = _new_axes(7.0, 4.2)
    for run in runs:
        arch = short_arch(run.name)
        dashed = run.name in superseded
        ax.plot(
            run.epochs,
            run.val_top1,
            color=colour_for(arch),
            linewidth=1.6,
            linestyle="--" if dashed else "-",
            alpha=0.75 if dashed else 1.0,
            label=f"{run.name} @ {run.lr:.0e}" + (" (superseded)" if dashed else ""),
        )
    ax.set_xlabel("epoch", color=INK)
    ax.set_ylabel("validation top-1", color=INK)
    ax.set_title("Training, per epoch", color=INK, loc="left", fontsize=11)
    # Below the axes rather than inside them: the attention run's superseded
    # curve is flat along the bottom of the frame, exactly where an in-axes
    # legend would sit, and that curve is the whole point of the figure.
    ax.legend(
        frameon=False,
        fontsize=8,
        ncol=2,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.13),
        borderaxespad=0.0,
    )
    return _save(fig, out)


def lr_sweep(sweep: dict[str, dict[float, float]], out: Path) -> Path:
    """Three-epoch validation top-1 against learning rate, one line per architecture.

    The x axis is logarithmic because the grid is: the question is whether
    each architecture's curve is monotone over the range (the incumbent rate
    is fine) or an inverted U (it is not).
    """
    fig, ax = _new_axes(6.4, 4.2)
    for arch in sorted(sweep, key=lambda a: -max(sweep[a].values())):
        points = sorted(sweep[arch].items())
        ax.plot(
            [lr for lr, _ in points],
            [top1 for _, top1 in points],
            marker="o",
            markersize=4,
            color=colour_for(arch),
            linewidth=1.6,
            label=arch,
        )
    ax.set_xscale("log")
    # Label only the rates actually swept. Matplotlib's default log minor
    # ticks put 3e-4 and 4e-4 close enough together to overprint, and neither
    # is a point in the grid.
    rates = sorted({lr for row in sweep.values() for lr in row})
    ax.set_xticks(rates)
    ax.set_xticklabels([f"{lr:g}" for lr in rates])
    ax.set_xticks([], minor=True)
    ax.set_xlabel("learning rate", color=INK)
    ax.set_ylabel("validation top-1 after 3 epochs", color=INK)
    ax.set_title("The rate each architecture wants", color=INK, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    return _save(fig, out)


def accuracy_by_ply(shift: list[dict], out: Path, untrained_plies: tuple[int, ...] = (4, 5)) -> Path:
    """Shift-probe accuracy against ply, one line per model.

    `untrained_plies` are shaded: the corpus holds no positions there, so
    that band is the only part of the figure measuring generalisation rather
    than recall.
    """
    fig, ax = _new_axes(6.8, 4.2)
    if untrained_plies:
        ax.axvspan(
            min(untrained_plies) - 0.4,
            max(untrained_plies) + 0.4,
            color=GRID,
            alpha=0.55,
            linewidth=0,
        )
        # Axes coordinates for y: a data-space constant would drift out of
        # frame the moment the accuracy range moves.
        ax.text(
            min(untrained_plies) - 0.3,
            0.04,
            "no training positions",
            transform=ax.get_xaxis_transform(),
            fontsize=8,
            color=INK,
            alpha=0.8,
        )
    for record in shift:
        arch = short_arch(record["architecture"])
        plies = sorted(int(p) for p in record["by_ply"])
        ax.plot(
            plies,
            [record["by_ply"][str(p)]["accuracy"] for p in plies],
            marker="o",
            markersize=3,
            color=colour_for(arch),
            linewidth=1.6,
            label=record["architecture"],
        )
    ax.set_xlabel("ply", color=INK)
    ax.set_ylabel("accuracy on provably won positions", color=INK)
    ax.set_title("Held-out accuracy by ply", color=INK, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    return _save(fig, out)


def value_mae_by_ply(shift: list[dict], out: Path) -> Path:
    """Value-head MAE against ply. The metric where the constraint prior wins."""
    fig, ax = _new_axes(6.8, 4.2)
    for record in shift:
        arch = short_arch(record["architecture"])
        plies = sorted(int(p) for p in record["by_ply"])
        ax.plot(
            plies,
            [record["by_ply"][str(p)]["value_mae"] for p in plies],
            marker="o",
            markersize=3,
            color=colour_for(arch),
            linewidth=1.6,
            label=record["architecture"],
        )
    ax.set_xlabel("ply", color=INK)
    ax.set_ylabel("value MAE (lower is better)", color=INK)
    ax.set_title("Value error by ply", color=INK, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, out)


def arena_by_depth(boards: dict[int, list[dict]], out: Path, title: str) -> Path:
    """Win rate against start ply, with 95% Wilson intervals.

    `boards` maps start ply to that arena's leaderboard. The intervals are
    the point: the ranking moves with depth, and a reader has to be able to
    see which of those moves are larger than the noise.
    """
    fig, ax = _new_axes(6.4, 4.2)
    plies = sorted(boards)
    agents = sorted({row["agent"] for board in boards.values() for row in board})
    for agent in agents:
        xs, ys, lo, hi = [], [], [], []
        for ply in plies:
            row = next((r for r in boards[ply] if r["agent"] == agent), None)
            if row is None:
                continue
            low, high = wilson_ci(row["wins"], row["games"])
            xs.append(ply)
            ys.append(row["win_rate"])
            lo.append(row["win_rate"] - low)
            hi.append(high - row["win_rate"])
        control = colour_for(agent) == CONTROL_COLOUR
        ax.errorbar(
            xs,
            ys,
            yerr=[lo, hi],
            marker="o",
            markersize=4,
            capsize=3,
            linewidth=1.6,
            linestyle="--" if control else "-",
            color=colour_for(agent),
            label=agent,
        )
    ax.axhline(0.5, color=INK, linewidth=0.8, alpha=0.4)
    ax.set_xticks(plies)
    ax.set_xlabel("start ply", color=INK)
    ax.set_ylabel("win rate against the field", color=INK)
    ax.set_title(title, color=INK, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, out)
