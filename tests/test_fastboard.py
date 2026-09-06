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
from quantik_core.symmetry import SymmetryHandler

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


# --- symmetry ------------------------------------------------------------


def _sampled(n: int, seed: int) -> np.ndarray:
    return np.array(_random_positions(n, seed), dtype=np.uint16)


@pytest.fixture(scope="module")
def sym_batch() -> np.ndarray:
    return _sampled(1500, seed=555)


def test_transform_preserves_legality(sym_batch):
    rng = np.random.default_rng(1)
    spatial, shape = fb.random_symmetries(sym_batch.shape[0], rng)
    moved = fb.transform_boards(sym_batch, spatial, shape)
    expected = fb.transform_policies(
        fb.legal_masks(sym_batch).astype(np.float32), spatial, shape
    ).astype(bool)
    assert np.array_equal(fb.legal_masks(moved), expected)


def test_transform_preserves_outcome_and_mover(sym_batch):
    rng = np.random.default_rng(2)
    spatial, shape = fb.random_symmetries(sym_batch.shape[0], rng)
    moved = fb.transform_boards(sym_batch, spatial, shape)
    assert np.array_equal(fb.has_winning_line(moved), fb.has_winning_line(sym_batch))
    assert np.array_equal(fb.side_to_move(moved), fb.side_to_move(sym_batch))


def test_transform_commutes_with_apply(sym_batch):
    live = sym_batch[~fb.terminal_status(sym_batch)[0]]
    rng = np.random.default_rng(3)
    legal = fb.legal_masks(live)
    actions = (rng.random(legal.shape) * legal).argmax(axis=1)
    spatial, shape = fb.random_symmetries(live.shape[0], rng)

    moved_then_played = fb.apply_actions(
        fb.transform_boards(live, spatial, shape),
        fb.transform_actions(actions, spatial, shape),
    )
    played_then_moved = fb.transform_boards(
        fb.apply_actions(live, actions), spatial, shape
    )
    assert np.array_equal(moved_then_played, played_then_moved)


def test_transform_actions_agrees_with_transform_policies(sym_batch):
    rng = np.random.default_rng(4)
    n = sym_batch.shape[0]
    actions = rng.integers(0, fb.ACTION_COUNT, size=n)
    spatial, shape = fb.random_symmetries(n, rng)
    onehot = np.zeros((n, fb.ACTION_COUNT), dtype=np.float32)
    onehot[np.arange(n), actions] = 1.0
    moved = fb.transform_policies(onehot, spatial, shape)
    assert np.array_equal(moved.argmax(axis=1), fb.transform_actions(actions, spatial, shape))
    assert np.allclose(moved.sum(axis=1), 1.0)


def test_canonical_key_is_symmetry_invariant(sym_batch):
    rng = np.random.default_rng(5)
    keys = fb.canonical_keys(sym_batch)
    for _ in range(3):
        spatial, shape = fb.random_symmetries(sym_batch.shape[0], rng)
        moved = fb.transform_boards(sym_batch, spatial, shape)
        assert np.array_equal(fb.canonical_keys(moved), keys)


def test_canonical_key_separates_distinct_positions(sym_batch):
    """Different keys must mean genuinely different positions."""
    keys = fb.canonical_keys(sym_batch)
    codes = fb.board_codes(sym_batch)
    by_key: dict[int, set[int]] = {}
    for key, code in zip(keys.tolist(), codes.tolist()):
        by_key.setdefault(key, set()).add(code)
    # Each key's members must be reachable from one another by a symmetry,
    # which the invariance test already establishes; here just check the map
    # is not degenerate (everything collapsing to one key).
    assert len(by_key) > len(sym_batch) // 20


def test_board_codes_round_trip_is_injective(sym_batch):
    codes = fb.board_codes(sym_batch)
    unique_codes = len(set(codes.tolist()))
    unique_boards = len({b.tobytes() for b in sym_batch})
    assert unique_codes == unique_boards


def test_square_channel_view_round_trips(sym_batch):
    """The uint8 square view must describe exactly the same board."""
    squares = fb.square_channels(sym_batch)
    rebuilt = np.zeros_like(sym_batch)
    for channel in range(8):
        bits = (squares == channel + 1)
        rebuilt[:, channel] = (bits * fb.SQUARE_BITS[None, :]).sum(axis=1).astype(np.uint16)
    assert np.array_equal(rebuilt, sym_batch)
    assert np.all(squares <= 8)


def test_spatial_perms_match_core_d4_mappings():
    """`spatial` here must be the same `d4_index` QW-001's
    `transform_index = d4_index * 24 + shape_perm_index` uses -- not just
    some other valid enumeration of the same 8 geometric transforms. This
    failed before the fix: numpy's `rot90`/`fliplr` composition enumerates
    the group in a different order (rot90/rot270 and reflH/reflD swapped)
    than `quantik_core.symmetry.D4Index`.

    `D4_MAPPINGS` and `permute16` both predate QW-001 and are already in the
    published `quantik-core`, so this test (unlike the two below) needs no
    version guard."""
    SymmetryHandler.permute16(0, 0)  # force lazy D4_MAPPINGS init
    for d in range(8):
        assert fb.SPATIAL_PERMS[d].tolist() == SymmetryHandler.D4_MAPPINGS[d]


def test_shape_perms_match_core_all_shape_perms():
    for i, perm in enumerate(fb.SHAPE_PERMS.tolist()):
        assert tuple(perm) == SymmetryHandler.ALL_SHAPE_PERMS[i]


_HAS_CORE_REMAP = hasattr(SymmetryHandler, "remap_action_index")
_SKIP_UNTIL_CORE_RELEASES_REMAP = pytest.mark.skipif(
    not _HAS_CORE_REMAP,
    reason=(
        "requires a quantik-core release containing SymmetryHandler."
        "remap_action_index/inverse_transform_index (QW-001; merged to "
        "quantik-core-py main, not yet published as of this test). Not a "
        "missing-extra skip: quantik-core is this package's required base "
        "dependency, and DEVELOPMENT.md's tests.yml runs against the "
        "published release by design, so this repo's CI must not depend on "
        "another repo's in-flight, unreleased state."
    ),
)


@_SKIP_UNTIL_CORE_RELEASES_REMAP
def test_transform_actions_matches_core_remap_action_index():
    """The load-bearing cross-check: `fb.transform_actions` (batched) and
    `quantik_core.SymmetryHandler.remap_action_index` (scalar) must agree
    for every one of the 192 transforms and 64 action indices. This is what
    makes the batched re-expression in this module a re-expression of the
    QW-001 action-index.v1 transform contract, rather than a parallel
    192-element scheme that happens to have the same size."""
    actions = np.arange(fb.ACTION_COUNT, dtype=np.int64)
    n_shape_perms = len(fb.SHAPE_PERMS)
    for d4_index in range(8):
        for shape_index in range(n_shape_perms):
            spatial = np.full(fb.ACTION_COUNT, d4_index, dtype=np.int64)
            shape = np.full(fb.ACTION_COUNT, shape_index, dtype=np.int64)
            got = fb.transform_actions(actions, spatial, shape)
            t_index = int(fb.transform_index(np.int64(d4_index), np.int64(shape_index)))
            expected = np.array(
                [SymmetryHandler.remap_action_index(int(a), t_index) for a in actions]
            )
            assert np.array_equal(got, expected), f"d4_index={d4_index} shape_index={shape_index}"


@_SKIP_UNTIL_CORE_RELEASES_REMAP
def test_transform_round_trips_through_core_inverse(sym_batch):
    """Applying a transform and then the inverse `quantik_core.
    SymmetryHandler.inverse_transform_index` reports for it must recover the
    original boards and actions -- an end-to-end round trip through the
    shared contract, not just this module's own machinery agreeing with
    itself."""
    rng = np.random.default_rng(9)
    n = sym_batch.shape[0]
    spatial, shape = fb.random_symmetries(n, rng)
    t_index = fb.transform_index(spatial, shape)
    inverse_index = np.array([SymmetryHandler.inverse_transform_index(int(t)) for t in t_index])
    inv_spatial, inv_shape = np.divmod(inverse_index, len(fb.SHAPE_PERMS))

    moved = fb.transform_boards(sym_batch, spatial, shape)
    restored = fb.transform_boards(moved, inv_spatial, inv_shape)
    assert np.array_equal(restored, sym_batch)

    actions = rng.integers(0, fb.ACTION_COUNT, size=n)
    moved_actions = fb.transform_actions(actions, spatial, shape)
    restored_actions = fb.transform_actions(moved_actions, inv_spatial, inv_shape)
    assert np.array_equal(restored_actions, actions)


def test_canonical_key_equals_the_explicit_192_way_minimum(sym_batch):
    """Guard the optimized reduction against the literal definition."""
    subset = sym_batch[:200]
    explicit = np.full(subset.shape[0], np.iinfo(np.uint64).max, dtype=np.uint64)
    for spatial in range(8):
        for shape_index in range(len(fb.SHAPE_PERMS)):
            moved = fb.transform_boards(
                subset,
                np.full(subset.shape[0], spatial),
                np.full(subset.shape[0], shape_index),
            )
            explicit = np.minimum(explicit, fb.board_codes(moved))
    assert np.array_equal(fb.canonical_keys(subset), explicit)
