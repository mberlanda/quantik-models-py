"""v1 dense policy <-> v2/v3 `optimal_mask`, and its refusal to lose information."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from quantik_models.data.policy_schema import (
    NonUniformPolicyError,
    dense_to_mask,
    mask_to_dense,
)

REPO = Path(__file__).resolve().parents[1]
V1 = "runs/oracle/corpus/exact-sampled.npz"


def _v1_corpus() -> Path | None:
    """`runs/` is gitignored: look here, then in the main checkout this may be a worktree of."""
    cands = [os.environ.get("QUANTIK_V1_CORPUS"), REPO / V1]
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()
        cands.append((REPO / common).resolve().parent / V1)
    except (OSError, subprocess.CalledProcessError):
        pass
    for c in cands:
        if c and Path(c).is_file():
            return Path(c)
    return None


def _action(shape: int, position: int) -> int:
    return shape * 16 + position


def _row(*actions: int) -> np.ndarray:
    r = np.zeros(64, dtype=np.float32)
    r[list(actions)] = np.float32(1) / np.float32(len(actions))
    return r


def test_real_v1_slice_round_trips_bit_exact() -> None:
    path = _v1_corpus()
    if path is None:
        pytest.skip(f"{V1} not available (gitignored; set QUANTIK_V1_CORPUS)")
    z = np.load(path)
    sl = slice(None, None, 10)  # every 10th row: all plies, labelled and value-only
    target, weight = z["policy_target"][sl], z["policy_weight"][sl]
    assert int((weight > 0).sum()) > 10_000 and int((weight == 0).sum()) > 10_000

    mask = dense_to_mask(target, weight)
    assert mask.dtype == np.uint64 and mask.shape == (len(target),)
    back_target, back_weight = mask_to_dense(mask)

    assert back_target.dtype == np.float32 and back_weight.dtype == np.float32
    assert back_target.tobytes() == target.tobytes()
    assert back_weight.tobytes() == weight.tobytes()


def test_bit_order_is_action_index() -> None:
    # bit i is action i, action = shape * 16 + position (invariant I6)
    for shape, position in [(0, 0), (0, 15), (1, 0), (2, 5), (3, 15)]:
        a = _action(shape, position)
        mask = dense_to_mask(_row(a)[None], np.ones(1, np.float32))
        assert int(mask[0]) == 1 << a
        target, weight = mask_to_dense(np.array([1 << a], dtype=np.uint64))
        assert np.flatnonzero(target[0]).tolist() == [a] and weight[0] == 1
    # action 0 is the least significant bit, action 63 the most
    assert int(dense_to_mask(_row(0)[None], np.ones(1, np.float32))[0]) == 1
    assert int(dense_to_mask(_row(63)[None], np.ones(1, np.float32))[0]) == 1 << 63
    assert int(dense_to_mask(_row(0, 63)[None], np.ones(1, np.float32))[0]) == (1 << 63) | 1


def test_value_only_rows_map_to_zero_mask() -> None:
    mask = dense_to_mask(np.zeros((2, 64), np.float32), np.zeros(2, np.float32))
    assert mask.tolist() == [0, 0]
    target, weight = mask_to_dense(mask)
    assert not target.any() and not weight.any()


def test_empty_input() -> None:
    assert dense_to_mask(np.zeros((0, 64), np.float32), np.zeros(0, np.float32)).shape == (0,)
    target, weight = mask_to_dense(np.zeros(0, np.uint64))
    assert target.shape == (0, 64) and weight.shape == (0,)


def test_non_uniform_row_raises_and_names_the_row() -> None:
    target = np.stack([_row(3, 7), _row(1, 2), _row(9)])
    target[1, 1], target[1, 2] = 0.75, 0.25  # a real weighting the mask cannot hold
    weight = np.ones(3, np.float32)
    with pytest.raises(NonUniformPolicyError) as e:
        dense_to_mask(target, weight)
    msg = str(e.value)
    assert isinstance(e.value, ValueError)
    assert "row 1" in msg and "row 0" not in msg and "row 2" not in msg
    assert "not uniform over its support" in msg
    assert "2 moves" in msg and "0.25" in msg and "0.75" in msg


def test_row_ids_are_reported() -> None:
    target = _row(0, 1)[None].copy()
    target[0, 0] = 0.6
    with pytest.raises(NonUniformPolicyError, match=r"row 0 \(id 'qfen-abc'\)"):
        dense_to_mask(target, np.ones(1, np.float32), row_ids=["qfen-abc"])


def test_reports_every_kind_of_loss_and_counts_the_rest() -> None:
    good = _row(4)
    rows = np.stack([good, good * 0.5, good, good, good, good, good, good])
    weight = np.ones(8, np.float32)
    weight[2] = 0.5  # fractional weight
    weight[3] = 0  # weight 0 with a nonzero target
    rows[4] = 0  # weight 1 with no support
    rows[5, 4] = np.nan
    rows[6, 4] = -1
    with pytest.raises(NonUniformPolicyError) as e:
        dense_to_mask(rows, weight)
    msg = str(e.value)
    assert "6 policy row(s)" in msg and "and 1 more" in msg
    for needle in ("row 1", "row 2", "row 3", "row 4", "row 5"):
        assert needle in msg
    assert "not 0 or 1" in msg and "weight is 0 but" in msg and "no support" in msg and "NaN" in msg


def test_atol_admits_an_ulp_of_noise_but_default_refuses() -> None:
    target = _row(1, 2)[None].copy()
    target[0, 1] = np.nextafter(target[0, 1], np.float32(1))
    w = np.ones(1, np.float32)
    with pytest.raises(NonUniformPolicyError):
        dense_to_mask(target, w)
    assert int(dense_to_mask(target, w, atol=1e-6)[0]) == 0b110
    # the admitted row is canonicalised: the ulp of deviation does not survive the round trip
    back, _ = mask_to_dense(dense_to_mask(target, w, atol=1e-6))
    assert back[0].tobytes() == _row(1, 2).tobytes() and back[0].tobytes() != target[0].tobytes()


def test_shape_validation() -> None:
    with pytest.raises(ValueError, match="shape"):
        dense_to_mask(np.zeros((2, 63), np.float32), np.zeros(2, np.float32))
    with pytest.raises(ValueError, match="policy_weight"):
        dense_to_mask(np.zeros((2, 64), np.float32), np.zeros(3, np.float32))
