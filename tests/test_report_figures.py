"""The figure module, exercised against a synthetic run directory.

`runs/` is gitignored, so a CI runner has none of the real data. Everything
here builds the files it reads, which also pins the on-disk formats the
figures depend on: if `metrics.jsonl` or an arena `games.json` changes
shape, these fail rather than the nightly report.
"""

from __future__ import annotations

import json

import pytest

from quantik_models.report import figures as fg

pytest.importorskip("matplotlib")


def write_run(root, name, top1, lr=2e-3):
    d = root / name
    d.mkdir(parents=True)
    (d / "metrics.jsonl").write_text(
        "\n".join(
            json.dumps({"epoch": i, "lr": lr * (0.9**i), "val_top1": v})
            for i, v in enumerate(top1)
        )
    )
    return d


def test_colour_for_matches_by_prefix_so_arena_suffixes_resolve():
    assert fg.colour_for("cpool") == fg.COLOURS["cpool"]
    assert fg.colour_for("cpool-mcts128") == fg.COLOURS["cpool"]
    assert fg.colour_for("uniform-mcts128") == fg.CONTROL_COLOUR


def test_load_training_run_reports_the_configured_rate_not_the_decayed_one(tmp_path):
    d = write_run(tmp_path, "swept-cpool", [0.90, 0.94, 0.96], lr=6e-4)
    run = fg.load_training_run(d)
    assert run.epochs == [0, 1, 2]
    assert run.best_top1 == pytest.approx(0.96)
    # The scheduler decays inside the run; the first row is the rate the run
    # was configured with, which is what the legend has to say.
    assert run.lr == pytest.approx(6e-4)


def test_load_training_run_rejects_an_empty_metrics_file(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    (d / "metrics.jsonl").write_text("")
    with pytest.raises(ValueError, match="empty"):
        fg.load_training_run(d)


def test_load_sweep_splits_architecture_from_rate_on_the_first_dash(tmp_path):
    # `6e-4` contains a dash, so a naive rsplit or a split on every dash
    # would read the architecture as `cpool-6e` or the rate as `6e`.
    write_run(tmp_path, "sweep-cpool-6e-4", [0.9, 0.95, 0.97])
    write_run(tmp_path, "sweep-cpool-2e-3", [0.9, 0.93, 0.94])
    write_run(tmp_path, "sweep-attn-6e-4", [0.5, 0.7, 0.79])
    sweep = fg.load_sweep(tmp_path)
    assert set(sweep) == {"cpool", "attn"}
    assert sweep["cpool"] == {6e-4: pytest.approx(0.97), 2e-3: pytest.approx(0.94)}


def test_short_arch_strips_the_size_suffix():
    assert fg.short_arch("cpool-c191-b6") == "cpool"
    assert fg.short_arch("mlp-h455-b4") == "mlp"


def test_training_curves_writes_an_svg_with_text_left_as_text(tmp_path):
    runs = [
        fg.load_training_run(write_run(tmp_path, "swept-cpool", [0.90, 0.94, 0.96])),
        fg.load_training_run(write_run(tmp_path, "lineup-cpool", [0.88, 0.90, 0.91])),
    ]
    out = fg.training_curves(runs, tmp_path / "out" / "curves.svg", superseded={"lineup-cpool"})
    body = out.read_text()
    assert body.startswith("<?xml")
    # svg.fonttype="none" keeps the labels as <text>, which is what makes the
    # committed figures reviewable in a diff.
    assert "<text" in body
    assert "swept-cpool" in body


def test_arena_by_depth_draws_an_interval_per_agent_per_depth(tmp_path):
    boards = {
        3: [{"agent": "cpool", "wins": 1029, "games": 1800, "win_rate": 0.5717}],
        6: [{"agent": "cpool", "wins": 924, "games": 1800, "win_rate": 0.5133}],
    }
    out = fg.arena_by_depth(boards, tmp_path / "arena.svg", title="policy arena")
    assert out.exists() and "cpool" in out.read_text()


def test_arena_by_depth_tolerates_an_agent_missing_from_one_depth(tmp_path):
    # The MCTS arena runs at plies 3 and 6 only; a figure combining depths
    # must not fail because one board is short an agent.
    boards = {
        3: [
            {"agent": "cpool", "wins": 100, "games": 200, "win_rate": 0.5},
            {"agent": "attn", "wins": 90, "games": 200, "win_rate": 0.45},
        ],
        6: [{"agent": "cpool", "wins": 110, "games": 200, "win_rate": 0.55}],
    }
    out = fg.arena_by_depth(boards, tmp_path / "arena.svg", title="policy arena")
    assert out.exists()


def test_accuracy_and_value_figures_read_the_shift_record_shape(tmp_path):
    shift = [
        {
            "architecture": "cpool-c191-b6",
            "by_ply": {
                "4": {"accuracy": 0.878, "value_mae": 0.19},
                "5": {"accuracy": 0.932, "value_mae": 0.12},
                "6": {"accuracy": 0.975, "value_mae": 0.09},
            },
        }
    ]
    (tmp_path / "shift.json").write_text(json.dumps(shift))
    loaded = fg.load_shift(tmp_path / "shift.json")
    a = fg.accuracy_by_ply(loaded, tmp_path / "acc.svg")
    v = fg.value_mae_by_ply(loaded, tmp_path / "mae.svg")
    assert a.exists() and v.exists()
    assert "no training positions" in a.read_text()


def test_load_leaderboard_reads_an_arena_games_json(tmp_path):
    (tmp_path / "games.json").write_text(
        json.dumps({"seed": 1, "games": 4, "leaderboard": [{"agent": "cpool", "wins": 3, "games": 4, "win_rate": 0.75}], "results": []})
    )
    board = fg.load_leaderboard(tmp_path / "games.json")
    assert board[0]["agent"] == "cpool"
