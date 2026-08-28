"""Fine-tuning through the trainer, end to end on a tiny corpus."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from quantik_models.data.exact_corpus import ExactCorpus  # noqa: E402
from quantik_models.env import fastboard as fb  # noqa: E402
from quantik_models.train.supervised import SupervisedConfig, train  # noqa: E402

from boards import random_positions  # noqa: E402


def _corpus(tmp_path, n=256):
    boards = random_positions(n, seed=11, plies=7)
    legal = fb.legal_masks(boards)
    mask = np.zeros(n, dtype=np.uint64)
    for row in range(n):
        for action in np.flatnonzero(legal[row])[:3]:
            mask[row] |= np.uint64(1) << np.uint64(int(action))
    path = tmp_path / "corpus.npz"
    ExactCorpus(
        boards=boards,
        optimal_mask=mask,
        value_target=np.zeros(n, dtype=np.float32),
        plies=np.full(n, 7, dtype=np.int16),
    ).save(path)
    return path


def _config(tmp_path, **kwargs):
    return SupervisedConfig(
        corpus=str(_corpus(tmp_path)),
        arch="resnet",
        preset="smoke",
        epochs=1,
        batch_size=32,
        device="cpu",
        val_fraction=0.2,
        balance_plies=False,
        **kwargs,
    )


def test_freezing_leaves_the_frozen_weights_untouched(tmp_path) -> None:
    from safetensors.torch import load_file

    base = tmp_path / "base"
    train(_config(tmp_path, name="base"), base)
    before = load_file(str(base / "base" / "best" / "weights.safetensors"))

    tuned = tmp_path / "tuned"
    train(
        _config(
            tmp_path,
            name="tuned",
            init_from=str(base / "base" / "best"),
            freeze="stem,trunk",
        ),
        tuned,
    )
    after = load_file(str(tuned / "tuned" / "best" / "weights.safetensors"))

    frozen = [k for k in before if k.startswith(("stem.", "trunk."))]
    assert frozen
    for key in frozen:
        # Includes the batch-norm running buffers, which `requires_grad`
        # alone would not have protected.
        torch.testing.assert_close(after[key], before[key], msg=f"{key} moved")

    heads = [k for k in before if k.startswith(("policy_head.", "value_head."))]
    assert any(not torch.equal(after[k], before[k]) for k in heads), (
        "the unfrozen heads did not move; the fine-tune did nothing"
    )


def test_freezing_without_init_from_is_refused(tmp_path) -> None:
    """Freezing random weights trains a model around noise it cannot fix."""
    with pytest.raises(ValueError, match="init-from"):
        train(_config(tmp_path, name="bad", freeze="stem"), tmp_path / "out")


def test_warm_start_without_freezing_still_moves_everything(tmp_path) -> None:
    from safetensors.torch import load_file

    base = tmp_path / "base"
    train(_config(tmp_path, name="base"), base)
    before = load_file(str(base / "base" / "best" / "weights.safetensors"))

    out = tmp_path / "retrain"
    train(
        _config(tmp_path, name="retrain", init_from=str(base / "base" / "best")),
        out,
    )
    after = load_file(str(out / "retrain" / "best" / "weights.safetensors"))
    assert any(not torch.equal(after[k], before[k]) for k in before if k.startswith("stem."))
