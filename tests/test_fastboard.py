"""Cross-check the vectorized engine against quantik_core's reference rules.

Every primitive in `quantik_models.env.fastboard` re-expresses a rule that
`quantik_core` already owns. These tests replay random games and assert the
two agree position-by-position, so the fast path can never silently drift
from the reference implementation.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from quantik_core.game_utils import has_winning_line as core_has_winning_line
from quantik_core.ml_data import qfen_to_tensor
from quantik_core.move import apply_move, generate_legal_moves_list
from quantik_core.qfen import bb_to_qfen

from quantik_models.env import fastboard as fb


def _random_positions(count: int, seed: int) -> list[tuple[int, ...]]:
    """Positions sampled along random playouts from the empty board."""
    rng = random.Random(seed)
    seen: list[tuple[int, ...]] = []
    while len(seen) < count:
        bb: tuple[int, ...] = (0,) * 8
        while True:
            seen.append(bb)
            if core_has_winning_line(bb):
                break
            moves = generate_legal_moves_list(bb)
            if not moves:
                break
            bb = apply_move(bb, rng.choice(moves))
    return seen[:count]


@pytest.fixture(scope="module")
def positions() -> list[tuple[int, ...]]:
    return _random_positions(3000, seed=20260827)


@pytest.fixture(scope="module")
def batch(positions: list[tuple[int, ...]]) -> np.ndarray:
    return np.array(positions, dtype=np.uint16)


def test_legal_masks_match_core(positions, batch):
    masks = fb.legal_masks(batch)
    for i, bb in enumerate(positions):
        if core_has_winning_line(bb):
            # A won position has no continuation in play; core still lists
            # placements, so only compare live positions.
            continue
        expected = np.zeros(64, dtype=bool)
        for move in generate_legal_moves_list(bb):
            expected[move.shape * 16 + move.position] = True
        assert np.array_equal(masks[i], expected), f"mismatch at {bb_to_qfen(bb)}"


def test_win_detection_matches_core(positions, batch):
    got = fb.has_winning_line(batch)
    expected = np.array([core_has_winning_line(bb) for bb in positions])
    assert np.array_equal(got, expected)


def test_side_to_move_matches_core(positions, batch):
    got = fb.side_to_move(batch)
    for i, bb in enumerate(positions):
        if core_has_winning_line(bb) or not generate_legal_moves_list(bb):
            continue
        assert got[i] == generate_legal_moves_list(bb)[0].player


def test_apply_actions_matches_core(positions, batch):
    live = [
        (i, bb)
        for i, bb in enumerate(positions)
        if not core_has_winning_line(bb) and generate_legal_moves_list(bb)
    ]
    idx = np.array([i for i, _ in live])
    rng = np.random.default_rng(7)
    masks = fb.legal_masks(batch[idx])
    actions = np.array(
        [rng.choice(np.flatnonzero(masks[k])) for k in range(len(idx))], dtype=np.int64
    )
    got = fb.apply_actions(batch[idx], actions)
    for k, (_, bb) in enumerate(live):
        action = int(actions[k])
        move = next(
            m
            for m in generate_legal_moves_list(bb)
            if m.shape * 16 + m.position == action
        )
        assert tuple(int(v) for v in got[k]) == tuple(apply_move(bb, move))


def test_qfen_roundtrip_matches_core(positions, batch):
    for i, bb in enumerate(positions):
        qfen = bb_to_qfen(bb)
        assert fb.to_qfen(batch[i]) == qfen
        assert np.array_equal(fb.from_qfen(qfen)[0], batch[i])


def test_core_tensor_encoding_matches_core(positions, batch):
    got = fb.to_core_tensor(batch)
    for i, bb in enumerate(positions):
        expected = qfen_to_tensor(bb_to_qfen(bb), int(fb.side_to_move(batch[i : i + 1])[0]))
        assert np.array_equal(got[i], expected)


def test_mover_relative_tensor_is_a_channel_permutation(batch):
    mover = fb.encode_tensors(batch)
    color = fb.to_core_tensor(batch)
    side = fb.side_to_move(batch)
    p0 = side == 0
    assert np.array_equal(mover[p0], color[p0])
    p1 = side == 1
    swapped = color[p1][:, [4, 5, 6, 7, 0, 1, 2, 3, 8]]
    assert np.array_equal(mover[p1], swapped)


def test_terminal_status_is_always_a_loss_for_the_mover(positions, batch):
    done, value = fb.terminal_status(batch)
    for i, bb in enumerate(positions):
        expected = core_has_winning_line(bb) or not generate_legal_moves_list(bb)
        assert bool(done[i]) == expected
    assert np.all(value[done] == -1.0)
    assert np.all(value[~done] == 0.0)


def test_popcount_table():
    sample = np.array([0, 1, 0xFFFF, 0x8000, 0x0F0F], dtype=np.uint16)
    assert list(fb.popcount(sample)) == [0, 1, 16, 1, 8]
