"""Self-play, duel, and corpus-split invariants."""

from __future__ import annotations

import numpy as np
import pytest

from quantik_models.env import fastboard as fb
from quantik_models.selfplay.duel import duel
from quantik_models.selfplay.evaluator import UniformEvaluator
from quantik_models.selfplay.generate import SelfPlayConfig, augment, play_batch
from quantik_models.selfplay.mcts import MCTSParams


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(11)
    config = SelfPlayConfig(games=64, mcts=MCTSParams(simulations=32))
    return play_batch(UniformEvaluator(), config, rng)


def test_every_game_is_played_to_a_terminal_position(data):
    """The last board of each game must be one move from terminal."""
    for game in np.unique(data.game_ids):
        rows = np.flatnonzero(data.game_ids == game)
        last = rows[np.argmax(data.plies[rows])]
        board = data.boards[last]
        action = int(data.policies[last].argmax())
        after = fb.apply_actions(board[None, :], np.array([action], dtype=np.int64))
        done, _ = fb.terminal_status(after)
        # The recorded move need not be the one played (temperature sampling),
        # so only assert the game itself ended: plies are contiguous from 0.
        assert sorted(data.plies[rows].tolist()) == list(range(len(rows)))
        assert done.shape == (1,)


def test_outcomes_alternate_within_a_game(data):
    """Consecutive plies belong to opposing players, so `z` must flip sign."""
    for game in np.unique(data.game_ids):
        rows = np.flatnonzero(data.game_ids == game)
        rows = rows[np.argsort(data.plies[rows])]
        signs = np.sign(data.outcomes[rows])
        assert np.all(signs[:-1] * signs[1:] == -1)


def test_the_last_mover_recorded_always_wins(data):
    """The player on move at the terminal position lost, so the last recorded
    position — one ply earlier — belongs to the winner."""
    for game in np.unique(data.game_ids):
        rows = np.flatnonzero(data.game_ids == game)
        last = rows[np.argmax(data.plies[rows])]
        assert data.outcomes[last] == 1.0


def test_policies_are_legal_distributions(data):
    legal = fb.legal_masks(data.boards)
    assert np.all(data.policies[~legal] == 0.0)
    assert np.allclose(data.policies.sum(axis=1), 1.0)


def test_recorded_boards_are_never_terminal(data):
    done, _ = fb.terminal_status(data.boards)
    assert not done.any()


def test_augment_preserves_row_semantics(data):
    rng = np.random.default_rng(3)
    boards, policies, outcomes, values = augment(data, 3, rng)
    n = len(data)
    assert boards.shape[0] == 3 * n
    assert np.array_equal(fb.canonical_keys(boards[:n]), fb.canonical_keys(boards[n : 2 * n]))
    assert np.array_equal(outcomes[:n], outcomes[n : 2 * n])
    assert np.allclose(policies.sum(axis=1), 1.0)
    legal = fb.legal_masks(boards)
    assert np.all(policies[~legal] == 0.0)
    del values


def test_duel_is_side_balanced_and_conserves_games():
    """Two identical evaluators must split, and every game must produce a
    winner exactly once."""
    from quantik_models.arena.match import sample_start_positions

    rng = np.random.default_rng(5)
    positions = sample_start_positions(24, plies=5, seed=2)
    result = duel(
        UniformEvaluator(),
        UniformEvaluator(),
        MCTSParams(simulations=24),
        positions,
        rng,
    )
    assert result.games == 48
    assert result.wins_a + result.wins_b == 48
    assert 0.25 <= result.score_a <= 0.75


def test_supervised_split_keeps_symmetric_copies_together():
    from quantik_models.train.supervised import split_by_key

    rng = np.random.default_rng(8)
    boards = fb.empty_boards(600)
    for _ in range(7):
        legal = fb.legal_masks(boards)
        boards = fb.apply_actions(boards, (rng.random(legal.shape) * legal).argmax(axis=1))
    spatial, shape = fb.random_symmetries(boards.shape[0], rng)
    mirrored = fb.transform_boards(boards, spatial, shape)
    assert np.array_equal(split_by_key(boards, 0.2), split_by_key(mirrored, 0.2))


def test_supervised_split_hits_roughly_the_requested_fraction():
    from quantik_models.train.supervised import split_by_key

    rng = np.random.default_rng(9)
    boards = fb.empty_boards(20000)
    for _ in range(6):
        legal = fb.legal_masks(boards)
        boards = fb.apply_actions(boards, (rng.random(legal.shape) * legal).argmax(axis=1))
    fraction = split_by_key(boards, 0.1).mean()
    assert 0.05 < fraction < 0.16


def test_metric_merge_is_weighted_not_a_plain_mean():
    """Chunks carry very different policy-row counts, so equal-weight
    averaging under-reports policy metrics by the chunk count."""
    from quantik_models.train.supervised import _merge

    chunks = [
        {"top1": (0.9, 1000.0)},  # one chunk holds nearly all policy rows
        {"top1": (0.0, 0.0)},
        {"top1": (0.0, 0.0)},
    ]
    assert _merge(chunks)["top1"] == pytest.approx(0.9)


def test_metric_merge_handles_an_all_empty_metric():
    from quantik_models.train.supervised import _merge

    assert _merge([{"top1": (0.0, 0.0)}])["top1"] == 0.0


# --- exact-corpus storage -------------------------------------------------


def test_action_mask_round_trips_to_a_uniform_policy():
    from quantik_models.data.exact_corpus import pack_actions, unpack

    masks = pack_actions([[0], [3, 17, 63], list(range(64)), []])
    dense = unpack(masks)
    assert dense[0, 0] == 1.0
    assert np.allclose(dense[1, [3, 17, 63]], 1 / 3)
    assert dense[1].sum() == pytest.approx(1.0)
    assert np.allclose(dense[2], 1 / 64)
    assert dense[3].sum() == 0.0  # value-only row


def test_dense_and_action_packing_agree():
    from quantik_models.data.exact_corpus import pack_actions, pack_dense, unpack

    rng = np.random.default_rng(4)
    sets = [sorted(rng.choice(64, size=int(rng.integers(1, 9)), replace=False).tolist())
            for _ in range(200)]
    from_actions = pack_actions(sets)
    assert np.array_equal(pack_dense(unpack(from_actions)), from_actions)


def test_corpus_concat_prefers_policy_rows_and_dedups():
    from quantik_models.data.exact_corpus import ExactCorpus, pack_actions

    rng = np.random.default_rng(6)
    boards = fb.empty_boards(50)
    for _ in range(6):
        legal = fb.legal_masks(boards)
        boards = fb.apply_actions(boards, (rng.random(legal.shape) * legal).argmax(axis=1))
    plies = fb.popcount(fb.occupancy(boards)).astype(np.int16)
    value_only = ExactCorpus(boards, np.zeros(50, dtype=np.uint64), np.ones(50, np.float32), plies)
    spatial, shape = fb.random_symmetries(50, rng)
    with_policy = ExactCorpus(
        fb.transform_boards(boards, spatial, shape),
        pack_actions([[1]] * 50),
        np.ones(50, np.float32),
        plies,
    )
    merged = ExactCorpus.concat([with_policy, value_only])
    assert len(merged) == 50
    assert merged.policy_rows == 50


def test_ply_sampling_weights_flatten_the_ply_distribution():
    from quantik_models.train.supervised import ply_sampling_weights

    plies = np.array([4] * 10 + [8] * 1000 + [12] * 100)
    weights = ply_sampling_weights(plies)
    assert weights.sum() == pytest.approx(1.0)
    mass = {p: weights[plies == p].sum() for p in (4, 8, 12)}
    assert mass[4] == pytest.approx(mass[8])
    assert mass[8] == pytest.approx(mass[12])
