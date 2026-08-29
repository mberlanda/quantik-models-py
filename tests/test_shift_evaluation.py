"""The shift evaluation must be able to report a bad result.

Its whole value is that it can contradict the training-time validation
number, so the checks that make it trustworthy are the ones that would fire
if it were quietly measuring something easier: a probe that overlaps the
training corpus, or a policy whose argmax is not actually legal.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from quantik_models.env import fastboard as fb  # noqa: E402
from quantik_models.eval import shift  # noqa: E402
from quantik_models.export.checkpoint import export_checkpoint  # noqa: E402
from quantik_models.model import registry  # noqa: E402

from boards import random_positions  # noqa: E402


def _probe_file(tmp_path, boards):
    """A probe in the on-disk format, with `outcome_optimal` legal by construction."""
    legal = fb.legal_masks(boards)
    rows = []
    for i, board in enumerate(boards):
        actions = np.flatnonzero(legal[i]).tolist()
        rows.append(
            {
                "qfen": fb.to_qfen(board),
                "won": bool(i % 2),
                "outcome_optimal": actions[: max(1, len(actions) // 2)],
                "score": 9999,
            }
        )
    path = tmp_path / "probe.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


def test_probe_round_trips_through_qfen(tmp_path) -> None:
    boards = random_positions(32, seed=3, plies=5)
    probe = shift.load_probe(_probe_file(tmp_path, boards))
    assert len(probe) == 32
    # QFEN carries the position; the plies must survive the round trip or
    # every per-ply number is attributed to the wrong bucket.
    assert probe.plies.tolist() == fb.popcount(fb.occupancy(boards)).tolist()


def test_held_out_check_passes_on_disjoint_sets(tmp_path) -> None:
    probe = shift.load_probe(_probe_file(tmp_path, random_positions(32, seed=1, plies=5)))
    other = random_positions(64, seed=99, plies=9)
    assert shift.assert_held_out(probe, other) == 0


def test_held_out_check_fires_when_the_probe_leaks(tmp_path) -> None:
    """The premise of the whole evaluation, so it has to be checkable.

    A probe overlapping the corpus measures recall and reports it as
    generalisation — and nothing downstream would look wrong.
    """
    boards = random_positions(32, seed=1, plies=5)
    probe = shift.load_probe(_probe_file(tmp_path, boards))
    with pytest.raises(AssertionError, match="not a shift evaluation"):
        shift.assert_held_out(probe, boards)


def test_a_symmetric_image_counts_as_a_leak(tmp_path) -> None:
    """Overlap is on the canonical key, not on raw board bytes."""
    boards = random_positions(32, seed=1, plies=5)
    probe = shift.load_probe(_probe_file(tmp_path, boards))
    rng = np.random.default_rng(0)
    spatial, shape = fb.random_symmetries(len(boards), rng)
    rotated = fb.transform_boards(boards, spatial, shape)
    with pytest.raises(AssertionError, match="not a shift evaluation"):
        shift.assert_held_out(probe, rotated)


@pytest.mark.parametrize("name", registry.architectures())
def test_evaluate_reports_every_ply_and_stays_legal(name: str, tmp_path) -> None:
    boards = random_positions(48, seed=5, plies=6)
    probe = shift.load_probe(_probe_file(tmp_path, boards))

    preset = "smoke" if "smoke" in registry.presets(name) else registry.presets(name)[0]
    out = tmp_path / "ckpt"
    export_checkpoint(
        registry.build(name, preset=preset).eval(),
        out_dir=out,
        model_id=f"{name}-test",
        training_report={},
    )

    # An untrained network is fine here: this asserts the accounting, not
    # the accuracy. The illegal-argmax assertion inside `evaluate` is the
    # part that has to hold for any weights at all.
    report = shift.evaluate(out, probe)
    assert set(report.by_ply) == set(probe.plies.tolist())
    for row in report.by_ply.values():
        assert 0 <= row.correct <= row.won_positions
        assert 0.0 <= row.value_sign <= 1.0

    overall = report.overall()
    assert overall.total == len(probe)
    assert overall.won_positions == int(probe.won.sum())


def test_overall_aggregates_only_the_requested_plies(tmp_path) -> None:
    boards = np.concatenate(
        [random_positions(16, seed=2, plies=4), random_positions(16, seed=2, plies=8)]
    )
    probe = shift.load_probe(_probe_file(tmp_path, boards))
    out = tmp_path / "ckpt"
    export_checkpoint(
        registry.build("resnet", preset="smoke").eval(),
        out_dir=out,
        model_id="t",
        training_report={},
    )
    report = shift.evaluate(out, probe)
    shallow = report.overall((4,))
    assert shallow.total == int((probe.plies == 4).sum())
    assert shallow.total < report.overall().total


def test_reports_are_labelled_by_run_when_the_architecture_repeats():
    """Comparing a model against a retrained version of itself is normal
    here, and two identically-named columns are unreadable — worse, the
    table looks perfectly fine."""
    from quantik_models.eval.shift import Report, Row

    def report(checkpoint):
        r = Report(checkpoint=checkpoint, architecture="cpool-c191-b6", parameter_count=1)
        for ply in (4, 5, 6, 7):
            r.by_ply[ply] = Row(ply=ply, won_positions=1, correct=1, value_abs_error=0.0, value_sign_correct=1, total=1)
        return r

    a = report("runs/train/swept-cpool/best")
    b = report("runs/train/v3-cpool/best")
    assert a.run_name == "swept-cpool" and b.run_name == "v3-cpool"

    md = shift.render([a, b])
    assert "cpool-c191-b6 (swept-cpool)" in md
    assert "cpool-c191-b6 (v3-cpool)" in md


def test_a_single_report_is_not_cluttered_with_its_run_name():
    from quantik_models.eval.shift import Report, Row

    r = Report(checkpoint="runs/train/swept-cpool/best", architecture="cpool-c191-b6", parameter_count=1)
    for ply in (4, 5):
        r.by_ply[ply] = Row(ply=ply, won_positions=1, correct=1, value_abs_error=0.0, value_sign_correct=1, total=1)
    md = shift.render([r])
    assert "`cpool-c191-b6`" in md
    assert "swept-cpool" not in md
