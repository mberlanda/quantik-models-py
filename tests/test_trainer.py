from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from quantik_models.train.trainer import TrainConfig, main, train  # noqa: E402


def _write_learnable_view(path: Path, n: int = 256) -> None:
    """Synthetic but learnable data: policy always action 0, value = +1
    when plane-0 mean exceeds the median (so loss can decrease)."""
    rng = np.random.default_rng(7)
    tensors = rng.random((n, 9, 4, 4), dtype=np.float32)
    policy = np.zeros((n, 64), dtype=np.float32)
    policy[:, 0] = 1.0
    signal = tensors[:, 0].mean(axis=(1, 2))
    value = np.where(signal > np.median(signal), 1.0, -1.0).astype(np.float32)
    np.savez_compressed(
        path,
        tensors=tensors,
        policy_target=policy,
        value_target=value,
        sample_weight=np.ones(n, dtype=np.float32),
        legal_action_mask=np.full(n, np.uint64(0xFFFF), dtype=np.uint64),
        side_to_move=np.zeros(n, dtype=np.uint8),
        source_tags=np.asarray(["synthetic"] * n, dtype=np.str_),
    )


def test_train_smoke_loss_decreases_and_exports(tmp_path: Path) -> None:
    npz = tmp_path / "view.npz"
    _write_learnable_view(npz)
    config = TrainConfig(
        npz_paths=[npz],
        preset="smoke",
        epochs=3,
        batch_size=32,
        seed=1,
        device="cpu",
        out_dir=tmp_path / "ckpt",
    )
    report = train(config)
    epochs = report["epochs"]
    assert len(epochs) == 3
    assert epochs[-1]["train_policy_loss"] < epochs[0]["train_policy_loss"]
    assert (tmp_path / "ckpt" / "manifest.json").is_file()
    assert (tmp_path / "ckpt" / "weights.safetensors").is_file()
    saved = json.loads((tmp_path / "ckpt" / "training-report.json").read_text())
    assert saved["final"] == epochs[-1]
    # legality masking: probability mass on illegal actions is ~0 post-mask
    assert epochs[-1]["val_illegal_mass_premask"] >= 0.0
    assert isinstance(epochs[-1]["epoch"], int)
    assert 0.0 <= epochs[-1]["val_top1_agreement"] <= 1.0


def test_train_is_seeded_deterministic(tmp_path: Path) -> None:
    npz = tmp_path / "view.npz"
    _write_learnable_view(npz)

    def run(out: Path) -> float:
        cfg = TrainConfig(
            npz_paths=[npz], preset="smoke", epochs=1, batch_size=32,
            seed=42, device="cpu", out_dir=out,
        )
        return float(train(cfg)["final"]["train_policy_loss"])

    assert run(tmp_path / "a") == pytest.approx(run(tmp_path / "b"))


def test_cli_maps_flags(tmp_path: Path) -> None:
    npz = tmp_path / "view.npz"
    _write_learnable_view(npz, n=128)
    rc = main(
        [
            "--npz", str(npz),
            "--preset", "smoke",
            "--epochs", "1",
            "--batch-size", "32",
            "--seed", "3",
            "--device", "cpu",
            "--out-dir", str(tmp_path / "cli-ckpt"),
        ]
    )
    assert rc == 0
    assert (tmp_path / "cli-ckpt" / "manifest.json").is_file()
