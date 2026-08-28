"""The split has one job: never test a model on what it memorised.

On a game with a 192-element symmetry group and several overlapping
corpora, that is a stronger requirement than "assign rows deterministically".
These tests pin the three ways it can quietly fail.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantik_models.data.dataset import boards_from_tensors, split_assignments
from quantik_models.env import fastboard as fb


def _corpus(n: int = 4000, seed: int = 7, plies: int = 5):
    """Reachable boards with their tensor encodings.

    Built by playing random legal moves rather than by sampling bit
    patterns, so the positions are real games and the canonical keys are
    distributed the way the trainer will actually see them.
    """
    rng = np.random.default_rng(seed)
    boards = fb.empty_boards(n)
    for _ in range(plies):
        legal = fb.legal_masks(boards)
        alive = legal.any(axis=1)
        if not alive.any():
            break
        weights = legal.astype(np.float64)
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1.0)
        actions = np.array(
            [
                rng.choice(len(w), p=w) if a else 0
                for w, a in zip(weights, alive)
            ],
            dtype=np.int64,
        )
        played = fb.apply_actions(boards, actions)
        boards = np.where(alive[:, None], played, boards)
    return boards, fb.encode_tensors(boards)


def test_boards_round_trip_through_the_tensor_encoding() -> None:
    """The split keys on the board recovered from planes 0-7."""
    boards, tensors = _corpus(512)
    assert (boards_from_tensors(tensors) == boards).all()


def test_no_canonical_position_spans_two_splits() -> None:
    boards, tensors = _corpus()
    labels = split_assignments(
        tensors, np.zeros((len(boards), 64), np.float32), ["x"] * len(boards)
    )
    keys = fb.canonical_keys(boards)
    groups = {s: np.unique(keys[labels == s]) for s in ("train", "val", "test")}
    assert len(np.intersect1d(groups["train"], groups["val"])) == 0
    assert len(np.intersect1d(groups["train"], groups["test"])) == 0
    assert len(np.intersect1d(groups["val"], groups["test"])) == 0


def test_symmetric_images_land_in_the_same_split() -> None:
    """The failure the previous hash had.

    Hashing tensor bytes gave a rotated or shape-relabelled copy of a
    training board a different bucket, so validation could be a
    transformation of something already learned.
    """
    boards, tensors = _corpus(2000)
    rng = np.random.default_rng(11)
    symmetries = fb.random_symmetries(len(boards), rng)
    transformed = fb.transform_boards(boards, *symmetries)

    policy = np.zeros((len(boards), 64), np.float32)
    tags = ["x"] * len(boards)
    original = split_assignments(tensors, policy, tags)
    rotated = split_assignments(fb.encode_tensors(transformed), policy, tags)

    assert (original == rotated).all()


def test_bucket_ignores_target_and_source() -> None:
    """The failure that appears once corpora are merged.

    The same position carrying a different policy target, or arriving from
    a different corpus, must not be able to change sides.
    """
    boards, tensors = _corpus(2000)
    tags_a = ["observations"] * len(boards)
    tags_b = ["selfplay"] * len(boards)
    zeros = np.zeros((len(boards), 64), np.float32)
    ones = np.ones((len(boards), 64), np.float32)

    assert (
        split_assignments(tensors, zeros, tags_a)
        == split_assignments(tensors, ones, tags_b)
    ).all()


def test_proportions_are_approximately_requested() -> None:
    boards, tensors = _corpus(20000)
    labels = split_assignments(
        tensors,
        np.zeros((len(boards), 64), np.float32),
        ["x"] * len(boards),
        train_pct=80,
        val_pct=10,
    )
    share = {s: (labels == s).mean() for s in ("train", "val", "test")}
    assert share["train"] == pytest.approx(0.80, abs=0.02)
    assert share["val"] == pytest.approx(0.10, abs=0.02)
    assert share["test"] == pytest.approx(0.10, abs=0.02)


def test_misaligned_inputs_are_rejected() -> None:
    boards, tensors = _corpus(64)
    with pytest.raises(ValueError, match="must align"):
        split_assignments(tensors, np.zeros((10, 64), np.float32), ["x"] * 64)
