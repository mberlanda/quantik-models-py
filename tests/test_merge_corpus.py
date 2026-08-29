"""Merging must not break the two things every downstream number assumes.

The probe stays held out, and each canonical position appears once. Both
failures are silent: a leaked probe turns generalisation into recall with
every report still looking fine, and a duplicated position quietly
reweights itself in the loss.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from quantik_models.data.exact_corpus import ExactCorpus
from quantik_models.data.merge_corpus import merge, probe_keys
from quantik_models.env import fastboard as fb

from boards import random_positions


def _corpus(boards, *, policy: bool) -> ExactCorpus:
    n = len(boards)
    legal = fb.legal_masks(boards)
    mask = np.zeros(n, dtype=np.uint64)
    if policy:
        for row in range(n):
            for action in np.flatnonzero(legal[row])[:2]:
                mask[row] |= np.uint64(1) << np.uint64(int(action))
    return ExactCorpus(
        boards=boards,
        optimal_mask=mask,
        value_target=np.zeros(n, dtype=np.float32),
        plies=fb.popcount(fb.occupancy(boards)).astype(np.int16),
    )


def test_merging_disjoint_corpora_keeps_everything() -> None:
    a = _corpus(random_positions(40, seed=1, plies=5), policy=True)
    b = _corpus(random_positions(40, seed=2, plies=9), policy=True)
    merged = merge([a, b])
    keys = set(fb.canonical_keys(merged.boards).tolist())
    assert keys == set(fb.canonical_keys(a.boards).tolist()) | set(
        fb.canonical_keys(b.boards).tolist()
    )


def test_a_position_appears_once() -> None:
    boards = random_positions(40, seed=3, plies=6)
    merged = merge([_corpus(boards, policy=True), _corpus(boards, policy=False)])
    keys = fb.canonical_keys(merged.boards)
    assert len(keys) == len(np.unique(keys))


def test_symmetric_images_collapse_to_one_row() -> None:
    """Dedup is on the canonical key, not on raw board bytes."""
    boards = random_positions(40, seed=4, plies=6)
    rng = np.random.default_rng(0)
    spatial, shape = fb.random_symmetries(len(boards), rng)
    mirrored = fb.transform_boards(boards, spatial, shape)
    merged = merge([_corpus(boards, policy=True), _corpus(mirrored, policy=True)])
    assert len(merged) == len(np.unique(fb.canonical_keys(boards)))


def test_policy_rows_win_the_tie_break() -> None:
    """A policy label is strictly more information than a value-only row."""
    boards = random_positions(30, seed=5, plies=6)
    value_only = _corpus(boards, policy=False)
    labelled = _corpus(boards, policy=True)

    for order in ([value_only, labelled], [labelled, value_only]):
        merged = merge(order)
        assert merged.policy_rows == len(merged), (
            "a value-only duplicate displaced a policy-labelled row"
        )


def test_excluded_keys_are_dropped(tmp_path) -> None:
    held_out = random_positions(20, seed=6, plies=5)
    corpus = _corpus(
        np.concatenate([held_out, random_positions(30, seed=7, plies=8)]), policy=True
    )
    exclude = {int(k) for k in fb.canonical_keys(held_out)}
    merged = merge([corpus], exclude)
    assert not (set(fb.canonical_keys(merged.boards).tolist()) & exclude)


def test_a_symmetric_image_of_a_probe_position_is_still_excluded() -> None:
    """The leak that would otherwise survive: a rotated probe board."""
    probe = random_positions(20, seed=8, plies=5)
    rng = np.random.default_rng(1)
    spatial, shape = fb.random_symmetries(len(probe), rng)
    rotated = fb.transform_boards(probe, spatial, shape)

    exclude = {int(k) for k in fb.canonical_keys(probe)}
    merged = merge([_corpus(rotated, policy=True)], exclude)
    assert len(merged) == 0


def test_probe_keys_reads_the_on_disk_format(tmp_path) -> None:
    boards = random_positions(12, seed=9, plies=5)
    path = tmp_path / "probe.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"qfen": fb.to_qfen(b), "won": True, "outcome_optimal": [0]})
            for b in boards
        )
    )
    assert probe_keys([path]) == {int(k) for k in fb.canonical_keys(boards)}


def test_plies_are_recomputed_not_carried() -> None:
    """A stale ply column would misattribute every per-ply number."""
    boards = random_positions(20, seed=10, plies=7)
    corpus = _corpus(boards, policy=True)
    corpus.plies[:] = 99  # deliberately wrong
    merged = merge([corpus])
    assert (merged.plies == fb.popcount(fb.occupancy(merged.boards))).all()


def test_merging_nothing_is_an_error() -> None:
    with pytest.raises(ValueError, match="nothing to merge"):
        merge([])
