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


def test_build_skips_missing_inputs_rather_than_crashing(tmp_path, capsys):
    """A fresh clone has no `runs/`, so every input is missing.

    The right behaviour is to name what it could not find and write what it
    can, not to raise — the figures are a report step, and a report step
    that aborts on the first absent file reports nothing.
    """
    from quantik_models.report import build_figures

    runs = tmp_path / "runs"
    write_run(runs / "train", "lineup-resnet", [0.90, 0.95])
    written = build_figures.build(runs, tmp_path / "out", eval_dir="nope")
    out = capsys.readouterr().out
    assert [p.name for p in written] == ["training-curves.svg"]
    assert "skipped (missing)" in out
    assert "lrsweep" in out and "shift.json" in out


def test_build_writes_nothing_and_reports_it_when_runs_is_empty(tmp_path, capsys):
    from quantik_models.report import build_figures

    assert build_figures.main(["--runs", str(tmp_path / "nothing"), "--out", str(tmp_path / "out")]) == 1
    assert "is runs/ populated" in capsys.readouterr().out


def test_the_figures_read_a_real_trainer_run(tmp_path):
    """End-to-end against `metrics.jsonl` as the trainer actually writes it.

    Every other test here builds the file itself, which pins the format this
    module *expects* rather than the one it *gets*. This is the only test
    that would notice the trainer renaming a field.
    """
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")

    import numpy as np

    from quantik_models.data.exact_corpus import ExactCorpus
    from quantik_models.env import fastboard as fb
    from quantik_models.train.supervised import SupervisedConfig, train

    from boards import random_positions

    boards = random_positions(128, seed=5, plies=7)
    legal = fb.legal_masks(boards)
    mask = np.zeros(len(boards), dtype=np.uint64)
    for row in range(len(boards)):
        for action in np.flatnonzero(legal[row])[:2]:
            mask[row] |= np.uint64(1) << np.uint64(int(action))
    corpus = tmp_path / "corpus.npz"
    ExactCorpus(
        boards=boards,
        optimal_mask=mask,
        value_target=np.zeros(len(boards), dtype=np.float32),
        plies=np.full(len(boards), 7, dtype=np.int16),
    ).save(corpus)

    train(
        SupervisedConfig(
            name="lineup-resnet",
            corpus=str(corpus),
            arch="resnet",
            preset="smoke",
            epochs=2,
            batch_size=32,
            device="cpu",
            val_fraction=0.2,
            balance_plies=False,
        ),
        tmp_path / "runs" / "train",
    )

    run = fg.load_training_run(tmp_path / "runs" / "train" / "lineup-resnet")
    assert run.epochs == [0, 1]
    assert 0.0 <= run.best_top1 <= 1.0
    assert run.lr > 0.0

    from quantik_models.report import build_figures

    written = build_figures.build(tmp_path / "runs", tmp_path / "figures", eval_dir="none")
    assert [p.name for p in written] == ["training-curves.svg"]


def test_oracle_benchmark_draws_a_tick_per_run_and_omits_the_oracle(tmp_path):
    pooled = [
        {"agent": "minimax-d2", "win_rate": 0.58, "ci_low": 0.57, "ci_high": 0.59},
        {"agent": "cpool", "win_rate": 0.46, "ci_low": 0.44, "ci_high": 0.48},
        {"agent": "mlp", "win_rate": 0.38, "ci_low": 0.36, "ci_high": 0.40},
    ]
    per_run = {
        "s1-p3": {"cpool": 0.45, "mlp": 0.37},
        "s2-p3": {"cpool": 0.47, "mlp": 0.39},
    }
    out = fg.oracle_benchmark(pooled, per_run, tmp_path / "oracle.svg", oracle="minimax-d2")
    body = out.read_text()
    # The oracle is the axis, not a bar on it: its own win rate is just
    # one minus the field's and would double the figure's height for nothing.
    assert "cpool" in body and "mlp" in body
    assert ">minimax-d2<" not in body


def test_build_draws_the_oracle_figure_from_a_packed_summary(tmp_path, capsys):
    """The oracle is picked as the agent with the most games.

    It played every pairing and each network played only its own, so the game
    count identifies it without the caller having to name it — and a wrong
    guess would silently draw the oracle as one more bar in the field.
    """
    import json

    from quantik_models.report import build_figures

    oracle = tmp_path / "oracle"
    for name, cpool in (("s1-p3", 0.45), ("s2-p3", 0.47)):
        d = oracle / name
        d.mkdir(parents=True)
        (d / "games.json").write_text(
            json.dumps(
                {
                    "seed": 1,
                    "games": 100,
                    "leaderboard": [
                        {"agent": "minimax-d2", "wins": 55, "games": 200, "win_rate": 0.55},
                        {"agent": "cpool", "wins": 45, "games": 100, "win_rate": cpool},
                    ],
                    "results": [],
                }
            )
        )
    packed = oracle / "packed"
    packed.mkdir()
    (packed / "summary.json").write_text(
        json.dumps(
            {
                "runs": [{"name": "s1-p3", "seed": 1, "games": 100}, {"name": "s2-p3", "seed": 2, "games": 100}],
                "pooled": [
                    {"agent": "minimax-d2", "wins": 110, "games": 400, "win_rate": 0.55, "ci_low": 0.50, "ci_high": 0.60},
                    {"agent": "cpool", "wins": 92, "games": 200, "win_rate": 0.46, "ci_low": 0.39, "ci_high": 0.53},
                ],
                "seed_spread": {"cpool": 0.02},
                "positions_to_solve": 0,
            }
        )
    )
    written = build_figures.build(tmp_path / "runs", tmp_path / "out", "none", oracle)
    assert [p.name for p in written] == ["oracle-benchmark.svg"]
    body = (tmp_path / "out" / "oracle-benchmark.svg").read_text()
    assert "cpool" in body


def test_build_skips_the_oracle_figure_when_nothing_is_packed(tmp_path, capsys):
    from quantik_models.report import build_figures

    build_figures.build(tmp_path / "runs", tmp_path / "out", "none", tmp_path / "oracle")
    assert "packed/summary.json" in capsys.readouterr().out


def test_regenerating_an_unchanged_figure_produces_identical_bytes(tmp_path):
    """Committed figures must only change when the data or the code does.

    matplotlib seeds its element ids from a per-process random value and
    embeds a creation timestamp, so without both fixed, every regeneration
    rewrites every clip-path id and the date line — and a reviewer cannot
    find a real change in the diff.
    """
    runs = [fg.load_training_run(write_run(tmp_path, "swept-cpool", [0.90, 0.94, 0.96]))]
    first = fg.training_curves(runs, tmp_path / "a.svg").read_bytes()
    second = fg.training_curves(runs, tmp_path / "b.svg").read_bytes()
    assert first == second
