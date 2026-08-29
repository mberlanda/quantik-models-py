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


def test_seed_spread_does_not_compare_across_start_depths(tmp_path):
    """Two runs at plies 3 and 6 are not replicates of each other.

    The ranking genuinely moves with start depth, so comparing them here
    reports a real depth effect as seed noise. A first version of this
    function did exactly that, and turned a 12.5-point depth difference into
    a "widest seed gap".
    """
    runs = []
    for name, seed, ply, rate in (
        ("s1-p3", 1, 3, 0.50),
        ("s2-p3", 2, 3, 0.52),
        ("s1-p6", 1, 6, 0.625),
    ):
        d = tmp_path / name
        d.mkdir()
        (d / "games.json").write_text(
            json.dumps(
                {
                    "seed": seed,
                    "start_plies": ply,
                    "games": 100,
                    "leaderboard": [{"agent": "orc", "wins": int(rate * 100), "games": 100, "win_rate": rate}],
                    "results": [],
                }
            )
        )
        runs.append(pack.read_run(d))
    # Within ply 3 the two seeds differ by 2 points; the ply-6 run differs by
    # 12.5 and must not be counted.
    assert pack.seed_spread(runs)["orc"] == pytest.approx(0.02)


def test_read_run_falls_back_to_the_directory_name_for_older_runs(tmp_path):
    """Runs written before autoplay recorded the field still group correctly."""
    d = tmp_path / "s20260902-p6"
    d.mkdir()
    (d / "games.json").write_text(
        json.dumps({"seed": 20260902, "games": 10, "leaderboard": [], "results": []})
    )
    assert pack.read_run(d).start_plies == 6


def test_read_run_reports_an_unknown_depth_rather_than_guessing(tmp_path):
    """A run with neither field nor suffix must not be grouped with ply 3."""
    d = tmp_path / "some-arena-run"
    d.mkdir()
    (d / "games.json").write_text(
        json.dumps({"seed": 1, "games": 10, "leaderboard": [], "results": []})
    )
    assert pack.read_run(d).start_plies is None


def test_merge_qfens_refilters_against_a_corpus_at_pack_time(tmp_path):
    """The arena's own filtering can be stale by the time the queue is spent.

    `autoplay` filters against whatever corpus it was pointed at while the
    games were played. The first oracle runs filtered against v1 while v2
    already existed, and 35% of the resulting queue was already labelled —
    about twelve hours of solver time. Filtering here happens later, which
    is when it matters.
    """
    from quantik_models.data.exact_corpus import ExactCorpus
    from quantik_models.arena.match import sample_start_positions

    boards = sample_start_positions(6, 4, seed=21)
    qfens = [fb.to_qfen(b) for b in boards]
    d = write_arena(tmp_path, "s1", 1, [{"agent": "x", "wins": 1, "games": 2}], qfens)

    corpus_path = tmp_path / "corpus.npz"
    covered = boards[:4]
    ExactCorpus(
        boards=covered,
        optimal_mask=np.zeros(len(covered), dtype=np.uint64),
        value_target=np.zeros(len(covered), dtype=np.float32),
        plies=np.full(len(covered), 4, dtype=np.int16),
    ).save(corpus_path)

    assert len(pack.merge_qfens([d])) == 6
    assert len(pack.merge_qfens([d], corpus_path)) == 2


def test_pack_records_what_it_filtered_against(tmp_path):
    """A queue whose provenance is unrecorded is a queue nobody can trust."""
    d = write_arena(
        tmp_path / "runs", "s1-p3", 1, [{"agent": "x", "wins": 1, "games": 2, "win_rate": 0.5}]
    )
    assert pack.pack([d], tmp_path / "out")["filtered_against"] is None


def test_pairwise_ignores_every_game_the_two_agents_were_not_both_in(tmp_path):
    """The reason this exists beside `head_to_head`: a leaderboard mixes in
    the whole card, so two agents that never beat each other can still
    finish points apart on games against a third."""
    results = (
        [{"mover": "a", "responder": "b", "winner": "a", "plies": 5, "actions": []}] * 3
        + [{"mover": "b", "responder": "a", "winner": "a", "plies": 5, "actions": []}] * 1
        + [{"mover": "a", "responder": "c", "winner": "a", "plies": 5, "actions": []}] * 50
        + [{"mover": "c", "responder": "b", "winner": "c", "plies": 5, "actions": []}] * 50
    )
    d = games_json(tmp_path, "s1", 1, results, [{"agent": "a", "wins": 54, "games": 104, "win_rate": 0.52}])
    row = pack.pairwise([d], "a", "b")
    assert row["games"] == 4
    assert row["wins"] == 4
    assert row["as_mover"] == 1.0
    assert row["as_responder"] == 1.0


def test_pairwise_splits_the_seats_and_flags_an_unbalanced_pairing(tmp_path):
    """From a shallow start the mover wins most games whoever is moving, so
    a pairing with unequal seat counts reports a number that is partly the
    first-move advantage. It has to say so."""
    results = (
        [{"mover": "a", "responder": "b", "winner": "a", "plies": 5, "actions": []}] * 8
        + [{"mover": "b", "responder": "a", "winner": "b", "plies": 5, "actions": []}] * 2
    )
    d = games_json(tmp_path, "s1", 1, results, [{"agent": "a", "wins": 8, "games": 10, "win_rate": 0.8}])
    row = pack.pairwise([d], "a", "b")
    assert row["as_mover"] == 1.0
    assert row["as_responder"] == 0.0
    assert row["win_rate"] == 0.8
    assert row["balanced"] is False


def test_pairwise_separates_only_when_the_interval_excludes_even(tmp_path):
    even = [
        {"mover": "a", "responder": "b", "winner": "a", "plies": 5, "actions": []},
        {"mover": "b", "responder": "a", "winner": "b", "plies": 5, "actions": []},
    ] * 100
    d = games_json(tmp_path, "even", 1, even, [{"agent": "a", "wins": 100, "games": 200, "win_rate": 0.5}])
    assert pack.pairwise([d], "a", "b")["separated"] is False

    lopsided = [{"mover": "a", "responder": "b", "winner": "a", "plies": 5, "actions": []}] * 200
    d2 = games_json(tmp_path, "lopsided", 2, lopsided, [{"agent": "a", "wins": 200, "games": 200, "win_rate": 1.0}])
    assert pack.pairwise([d2], "a", "b")["separated"] is True


def test_pairwise_on_a_pairing_that_never_happened_is_empty_not_an_error(tmp_path):
    results = [{"mover": "a", "responder": "b", "winner": "a", "plies": 5, "actions": []}]
    d = games_json(tmp_path, "s1", 1, results, [{"agent": "a", "wins": 1, "games": 1, "win_rate": 1.0}])
    row = pack.pairwise([d], "a", "z")
    assert row["games"] == 0
    assert row["separated"] is False
