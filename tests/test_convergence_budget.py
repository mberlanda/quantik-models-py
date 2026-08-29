"""Stopping on convergence rather than on a number chosen for one architecture.

A shared epoch budget is not equal treatment, for the same reason a shared
learning rate was not: sixteen epochs was chosen when the ResNet was the
only architecture in the project, and the attention encoder was still
climbing when it ran out — so its published 0.9879 is a floor, not a result.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from quantik_models.train.convergence import epochs_since_best


@pytest.mark.parametrize(
    "history, expected",
    [
        ([], 0),
        ([1.0], 0),
        ([1.0, 0.9, 0.8], 0),
        ([1.0, 0.5, 0.6, 0.7], 2),
        ([0.5, 0.6, 0.7, 0.8], 3),
    ],
)
def test_epochs_since_best_counts_from_the_lowest(history, expected) -> None:
    assert epochs_since_best(history) == expected


def test_a_tie_does_not_buy_more_epochs() -> None:
    """`best` is only rewritten on a strict decrease.

    An epoch that merely equals the best did not produce the weights on
    disk, so counting it as an improvement would keep a converged run alive
    on a checkpoint it never wrote.
    """
    assert epochs_since_best([1.0, 0.5, 0.5]) == 1
    assert epochs_since_best([1.0, 0.5, 0.5, 0.5]) == 2


def test_patience_defaults_to_off_so_published_runs_reproduce() -> None:
    pytest.importorskip("torch")
    from quantik_models.train.supervised import SupervisedConfig, build_parser

    assert SupervisedConfig().patience is None
    assert build_parser().parse_args([]).patience is None


def test_patience_parses_as_an_int() -> None:
    pytest.importorskip("torch")
    from quantik_models.train.supervised import build_parser

    # The `--lr`-as-string bug in miniature: an optional field with no
    # runtime value to infer a type from.
    args = build_parser().parse_args(["--patience", "5", "--epochs", "60"])
    assert args.patience == 5 and isinstance(args.patience, int)


def _tiny_corpus(path, rows=128, plies=7):
    from quantik_models.data.exact_corpus import ExactCorpus
    from quantik_models.env import fastboard as fb

    from boards import random_positions

    boards = random_positions(rows, seed=9, plies=plies)
    legal = fb.legal_masks(boards)
    mask = np.zeros(len(boards), dtype=np.uint64)
    for row in range(len(boards)):
        for action in np.flatnonzero(legal[row])[:2]:
            mask[row] |= np.uint64(1) << np.uint64(int(action))
    ExactCorpus(
        boards=boards,
        optimal_mask=mask,
        value_target=np.zeros(len(boards), dtype=np.float32),
        plies=np.full(len(boards), plies, dtype=np.int16),
    ).save(path)
    return path


def test_patience_zero_stops_after_the_first_epoch(tmp_path) -> None:
    """The cheapest end-to-end proof that the break is wired to the loop.

    With `patience=0` the first epoch is already "no improvement for 0
    epochs", so the run stops at one — and the report has to say one, not
    the cap. A report claiming the cap describes a different run.
    """
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from quantik_models.train.supervised import SupervisedConfig, train

    corpus = _tiny_corpus(tmp_path / "corpus.npz")
    train(
        SupervisedConfig(
            name="stop-now",
            corpus=str(corpus),
            arch="mlp",
            preset="smoke",
            epochs=9,
            patience=0,
            batch_size=32,
            device="cpu",
            val_fraction=0.2,
            balance_plies=False,
        ),
        tmp_path / "out",
    )
    run_dir = tmp_path / "out" / "stop-now"
    written = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert len(written) == 1

    report = json.loads((run_dir / "final" / "training-report.json").read_text())
    assert report["epochs"] == 1
    assert report["epoch_cap"] == 9
    assert report["stopped_early"] is True


def test_without_patience_the_run_uses_its_whole_budget(tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from quantik_models.train.supervised import SupervisedConfig, train

    corpus = _tiny_corpus(tmp_path / "corpus.npz")
    train(
        SupervisedConfig(
            name="full",
            corpus=str(corpus),
            arch="mlp",
            preset="smoke",
            epochs=3,
            batch_size=32,
            device="cpu",
            val_fraction=0.2,
            balance_plies=False,
        ),
        tmp_path / "out",
    )
    run_dir = tmp_path / "out" / "full"
    written = (run_dir / "metrics.jsonl").read_text().splitlines()
    assert len(written) == 3
    report = json.loads((run_dir / "final" / "training-report.json").read_text())
    assert report["epochs"] == 3 and report["stopped_early"] is False
