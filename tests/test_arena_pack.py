"""Pooling several arena runs into one reusable result set.

The failure this guards against is quiet: pooling that averages rates, or
that concatenates position files without deduplicating, produces a plausible
number and a solver queue several times larger than it needs to be.
"""

from __future__ import annotations

import gzip
import json

import numpy as np
import pytest

from quantik_models.arena import pack
from quantik_models.env import fastboard as fb


def write_arena(root, name, seed, leaderboard, qfens=()):
    d = root / name
    d.mkdir(parents=True)
    (d / "games.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "games": sum(r["games"] for r in leaderboard) // 2,
                "leaderboard": leaderboard,
                "results": [],
            }
        )
    )
    if qfens:
        (d / "to-solve.qfen").write_text("\n".join(qfens) + "\n")
    return d


def board_qfens(count, seed, plies):
    from quantik_models.arena.match import sample_start_positions

    return [fb.to_qfen(b) for b in sample_start_positions(count, plies, seed)]


def test_pooled_sums_counts_rather_than_averaging_rates(tmp_path):
    """Averaging rates weights a short run equally with a long one."""
    a = pack.read_run(
        write_arena(tmp_path, "a", 1, [{"agent": "x", "wins": 90, "games": 100}])
    )
    b = pack.read_run(
        write_arena(tmp_path, "b", 2, [{"agent": "x", "wins": 100, "games": 900}])
    )
    row = pack.pooled([a, b])[0]
    assert row["wins"] == 190 and row["games"] == 1000
    assert row["win_rate"] == pytest.approx(0.19)
    # The mean of the two rates would be 0.505 — nearly three times as high.
    assert row["ci_low"] < 0.19 < row["ci_high"]


def test_seed_spread_reports_the_widest_disagreement(tmp_path):
    runs = [
        pack.read_run(
            write_arena(tmp_path, n, s, [{"agent": "x", "wins": w, "games": 100, "win_rate": w / 100}])
        )
        for n, s, w in (("a", 1, 40), ("b", 2, 55), ("c", 3, 48))
    ]
    assert pack.seed_spread(runs)["x"] == pytest.approx(0.15)


def test_seed_spread_is_zero_for_a_single_run(tmp_path):
    run = pack.read_run(
        write_arena(tmp_path, "only", 1, [{"agent": "x", "wins": 4, "games": 10, "win_rate": 0.4}])
    )
    assert pack.seed_spread([run])["x"] == 0.0


def test_merge_qfens_deduplicates_up_to_symmetry(tmp_path):
    """Runs overlap near the root; a rotation is not a new position."""
    qfens = board_qfens(6, seed=11, plies=3)
    a = write_arena(tmp_path, "a", 1, [{"agent": "x", "wins": 1, "games": 2}], qfens[:4])
    b = write_arena(tmp_path, "b", 2, [{"agent": "x", "wins": 1, "games": 2}], qfens[2:])
    merged = pack.merge_qfens([a, b])
    assert len(merged) == 6
    boards = np.concatenate([fb.from_qfen(line) for line in merged])
    assert len(set(fb.canonical_keys(boards).tolist())) == 6


def test_merge_qfens_tolerates_a_run_with_no_positions(tmp_path):
    a = write_arena(tmp_path, "a", 1, [{"agent": "x", "wins": 1, "games": 2}])
    assert pack.merge_qfens([a]) == []


def test_pack_writes_a_gzipped_solver_queue_and_a_summary(tmp_path):
    qfens = board_qfens(5, seed=3, plies=4)
    dirs = [
        write_arena(tmp_path / "runs", "s1-p3", 1, [
            {"agent": "minimax-d2", "wins": 60, "games": 100, "win_rate": 0.60},
            {"agent": "cpool", "wins": 40, "games": 100, "win_rate": 0.40},
        ], qfens[:3]),
        write_arena(tmp_path / "runs", "s2-p3", 2, [
            {"agent": "minimax-d2", "wins": 55, "games": 100, "win_rate": 0.55},
            {"agent": "cpool", "wins": 45, "games": 100, "win_rate": 0.45},
        ], qfens[2:]),
    ]
    out = tmp_path / "packed"
    summary = pack.pack(dirs, out)

    assert summary["positions_to_solve"] == 5
    with gzip.open(out / "to-solve.qfen.gz", "rt") as handle:
        assert len([line for line in handle if line.strip()]) == 5
    # Every run's games.json is archived, not just the pooled numbers: the
    # per-game records are what a retrain reads.
    assert (out / "games-s1-p3.json.gz").exists()
    assert (out / "games-s2-p3.json.gz").exists()

    top = summary["pooled"][0]
    assert top["agent"] == "minimax-d2" and top["games"] == 200
    assert summary["seed_spread"]["cpool"] == pytest.approx(0.05)

    md = (out / "summary.md").read_text()
    assert "`minimax-d2`" in md and "57.5%" in md
    # Each seed stays visible beside the pooled figure.
    assert "s1-p3" in md and "s2-p3" in md


def test_pack_refuses_a_directory_with_no_runs(tmp_path):
    with pytest.raises(ValueError, match="no arena runs"):
        pack.pack([tmp_path / "nothing"], tmp_path / "out")


def games_json(root, name, seed, results, leaderboard):
    d = root / name
    d.mkdir(parents=True)
    (d / "games.json").write_text(
        json.dumps({"seed": seed, "games": len(results), "leaderboard": leaderboard, "results": results})
    )
    return d


def test_head_to_head_splits_the_seats(tmp_path):
    """The seat is not a detail: from a shallow start the mover wins most
    games regardless of who is moving, so a pooled rate against a fixed
    opponent mixes strength with the first-move advantage."""
    results = (
        [{"mover": "cpool", "responder": "orc", "winner": "cpool", "plies": 5, "actions": []}] * 8
        + [{"mover": "cpool", "responder": "orc", "winner": "orc", "plies": 5, "actions": []}] * 2
        + [{"mover": "orc", "responder": "cpool", "winner": "orc", "plies": 5, "actions": []}] * 8
        + [{"mover": "orc", "responder": "cpool", "winner": "cpool", "plies": 5, "actions": []}] * 2
    )
    d = games_json(tmp_path, "s1", 1, results, [{"agent": "orc", "wins": 10, "games": 20, "win_rate": 0.5}])
    row = pack.head_to_head([d], "orc")[0]
    assert row["agent"] == "cpool"
    assert row["as_mover"] == pytest.approx(0.8)
    assert row["as_responder"] == pytest.approx(0.2)
    assert row["win_rate"] == pytest.approx(0.5)
    # 10/20 either way: the seats cancel and the pooled figure says nothing.
    assert not row["beats_oracle"] and not row["loses_to_oracle"]


def test_head_to_head_calls_a_verdict_only_when_the_interval_excludes_even(tmp_path):
    results = [{"mover": "mlp", "responder": "orc", "winner": "orc", "plies": 5, "actions": []}] * 400
    d = games_json(tmp_path, "s1", 1, results, [{"agent": "orc", "wins": 400, "games": 400, "win_rate": 1.0}])
    row = pack.head_to_head([d], "orc")[0]
    assert row["loses_to_oracle"] is True and row["beats_oracle"] is False


def test_head_to_head_ignores_pairings_the_oracle_was_not_in(tmp_path):
    results = [
        {"mover": "cpool", "responder": "attn", "winner": "cpool", "plies": 5, "actions": []},
        {"mover": "cpool", "responder": "orc", "winner": "cpool", "plies": 5, "actions": []},
    ]
    d = games_json(tmp_path, "s1", 1, results, [{"agent": "orc", "wins": 0, "games": 1, "win_rate": 0.0}])
    rows = pack.head_to_head([d], "orc")
    assert [r["agent"] for r in rows] == ["cpool"]
    assert rows[0]["games"] == 1


def test_pack_infers_the_oracle_from_the_game_counts(tmp_path):
    """It played every pairing; each network played only its own."""
    results = (
        [{"mover": "cpool", "responder": "orc", "winner": "orc", "plies": 5, "actions": []}] * 4
        + [{"mover": "mlp", "responder": "orc", "winner": "orc", "plies": 5, "actions": []}] * 4
    )
    d = games_json(
        tmp_path / "runs",
        "s1-p3",
        1,
        results,
        [
            {"agent": "orc", "wins": 8, "games": 8, "win_rate": 1.0},
            {"agent": "cpool", "wins": 0, "games": 4, "win_rate": 0.0},
            {"agent": "mlp", "wins": 0, "games": 4, "win_rate": 0.0},
        ],
    )
    summary = pack.pack([d], tmp_path / "packed")
    assert summary["oracle"] == "orc"
    assert {r["agent"] for r in summary["head_to_head"]} == {"cpool", "mlp"}
    assert "Head to head against `orc`" in (tmp_path / "packed" / "summary.md").read_text()
