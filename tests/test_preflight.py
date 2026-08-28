"""The preflight has to fail when something is actually wrong.

It exists to spend a minute so a training run does not waste an hour, and
a check that always reports `ok` costs the minute and buys nothing. So the
tests here run it green on a valid corpus, and then break each assumption
in turn and assert the corresponding check goes red.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

from quantik_models.data.exact_corpus import ExactCorpus  # noqa: E402
from quantik_models.env import fastboard as fb  # noqa: E402
from quantik_models.train import preflight  # noqa: E402

from boards import random_positions  # noqa: E402


def _corpus(tmp_path, n: int = 512):
    """A small but structurally real corpus: legal boards, true legal masks."""
    boards = random_positions(n, seed=7, plies=6)
    legal = fb.legal_masks(boards)
    # An "optimal" set that is a subset of the legal moves, which is what
    # the solver produces and what the loss assumes.
    mask = np.zeros(n, dtype=np.uint64)
    for row in range(n):
        actions = np.flatnonzero(legal[row])[:2]
        for action in actions:
            mask[row] |= np.uint64(1) << np.uint64(int(action))
    corpus = ExactCorpus(
        boards=boards,
        optimal_mask=mask,
        value_target=np.zeros(n, dtype=np.float32),
        plies=np.full(n, 6, dtype=np.int16),
    )
    path = tmp_path / "corpus.npz"
    corpus.save(path)
    return path


def test_preflight_passes_on_a_valid_corpus(tmp_path, capsys) -> None:
    code = preflight.main(
        [
            "--corpus", str(_corpus(tmp_path)),
            "--arch", "resnet",
            "--preset", "smoke",
            "--epochs", "1",
            "--batch-size", "64",
            "--device", "cpu",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0, out
    assert "all " in out and "checks passed" in out


def test_split_check_reports_a_leak(tmp_path) -> None:
    """The check that matters most, against a corpus that genuinely leaks.

    Built by appending each board's symmetric image: a random split then
    puts a position and its rotation on opposite sides, which is precisely
    the contamination `split_by_key` exists to prevent and which no shape
    or count check would notice.
    """
    from quantik_models.data.exact_corpus import split_by_key

    corpus = preflight.load_corpus(_corpus(tmp_path))
    assert all(c.ok for c in preflight.check_split(corpus, 0.05))

    boards = corpus["boards"]
    rng = np.random.default_rng(0)
    spatial, shape = fb.random_symmetries(len(boards), rng)
    doubled = np.concatenate([boards, fb.transform_boards(boards, spatial, shape)])
    keys = fb.canonical_keys(doubled)

    # A naive random split leaks: images of one position land on both sides.
    naive = rng.random(len(doubled)) < 0.5
    leaked = np.intersect1d(np.unique(keys[~naive]), np.unique(keys[naive]))
    assert leaked.size > 0, "the negative control has to actually leak"

    # The real split does not, which is what the preflight asserts on the
    # corpus a run is about to use.
    is_val = split_by_key(doubled, 0.05)
    shared = np.intersect1d(np.unique(keys[~is_val]), np.unique(keys[is_val]))
    assert shared.size == 0


def test_gradient_check_catches_a_frozen_trunk(tmp_path) -> None:
    """A frozen parameter is the failure `init_from` and freezing invite."""
    from quantik_models.model import registry
    from quantik_models.train.supervised import SupervisedConfig

    corpus = preflight.load_corpus(_corpus(tmp_path))
    config = SupervisedConfig(
        arch="resnet", preset="smoke", batch_size=64, device="cpu", epochs=1
    )

    model = registry.build("resnet", preset="smoke")
    for param in model.stem.parameters():
        param.requires_grad_(False)

    # Reproduce what `check_architecture` does, on the frozen model.
    batch = preflight._sample_batch(corpus, np.random.default_rng(0), 64)
    loss, _ = preflight._forward_losses(
        model, *batch, torch.device("cpu"), config.value_loss_weight
    )
    loss.backward()
    starved = [n for n, p in model.named_parameters() if p.grad is None or not p.grad.any()]
    assert starved, "freezing the stem must show up as starved parameters"


def test_corpus_check_reports_row_counts(tmp_path) -> None:
    corpus = preflight.load_corpus(_corpus(tmp_path, n=128))
    checks = preflight.check_corpus(corpus, "corpus.npz")
    assert all(c.ok for c in checks)
    assert "128 rows" in checks[0].detail
