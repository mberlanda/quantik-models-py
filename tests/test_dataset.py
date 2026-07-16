from __future__ import annotations

from pathlib import Path

import numpy as np

from quantik_models.data.dataset import (
    LoadedTrainingData,
    expand_legal_mask,
    load_training_data,
    split_assignments,
)


def _write_view(path: Path, n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    tensors = rng.random((n, 9, 4, 4), dtype=np.float32)
    policy = rng.random((n, 64), dtype=np.float32)
    policy /= policy.sum(axis=1, keepdims=True)
    np.savez_compressed(
        path,
        tensors=tensors,
        policy_target=policy,
        value_target=rng.uniform(-1, 1, n).astype(np.float32),
        sample_weight=np.ones(n, dtype=np.float32),
        legal_action_mask=rng.integers(1, 2**63, n, dtype=np.uint64),
        side_to_move=rng.integers(0, 2, n, dtype=np.uint8),
        source_tags=np.asarray([f"tag{i}" for i in range(n)], dtype=np.str_),
    )


def test_expand_legal_mask_bits() -> None:
    mask = np.asarray([0b101, 1 << 63], dtype=np.uint64)
    expanded = expand_legal_mask(mask)
    assert expanded.shape == (2, 64)
    assert expanded.dtype == np.bool_
    assert expanded[0, 0] and not expanded[0, 1] and expanded[0, 2]
    assert expanded[1, 63] and not expanded[1, 0]
    assert expanded[0].sum() == 2 and expanded[1].sum() == 1


def test_split_assignments_deterministic_and_order_independent(tmp_path: Path) -> None:
    _write_view(tmp_path / "a.npz", 200, seed=1)
    data = load_training_data([tmp_path / "a.npz"])
    a1 = split_assignments(data.tensors, data.policy_target, data.source_tags)
    a2 = split_assignments(data.tensors, data.policy_target, data.source_tags)
    assert (a1 == a2).all()
    perm = np.random.default_rng(0).permutation(len(a1))
    a3 = split_assignments(
        data.tensors[perm],
        data.policy_target[perm],
        tuple(data.source_tags[i] for i in perm),
    )
    assert (a3 == a1[perm]).all()  # assignment travels with the row
    # all three splits are non-empty at n=200 and fractions are sane
    frac_train = float((a1 == "train").mean())
    assert 0.7 < frac_train < 0.9
    assert (a1 == "val").any() and (a1 == "test").any()


def test_load_training_data_concat_and_split(tmp_path: Path) -> None:
    _write_view(tmp_path / "a.npz", 60, seed=2)
    _write_view(tmp_path / "b.npz", 40, seed=3)
    full = load_training_data([tmp_path / "a.npz", tmp_path / "b.npz"])
    assert isinstance(full, LoadedTrainingData)
    assert full.tensors.shape == (100, 9, 4, 4)
    assert full.legal_mask.shape == (100, 64)
    assert full.legal_mask.dtype == np.bool_
    parts = [
        load_training_data([tmp_path / "a.npz", tmp_path / "b.npz"], split=s)
        for s in ("train", "val", "test")
    ]
    assert sum(p.tensors.shape[0] for p in parts) == 100
    # rows in val/test do not depend on how files are listed
    swapped = load_training_data([tmp_path / "b.npz", tmp_path / "a.npz"], split="val")
    assert {t.tobytes() for t in swapped.tensors} == {t.tobytes() for t in parts[1].tensors}
