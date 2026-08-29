"""What the `seed` argument on `Agent.select` actually does.

Every agent in this repo takes `select(board, seed)`, and `arena.match`,
`arena.autoplay` and the play service all thread a varying seed through it.
For the network-backed agents that seed was **inert**: `PolicyAgent`
defaults to `temperature=0.0` (argmax over the priors) and `NetMCTSAgent`
returned `visits.argmax()`, while the only consumer of its RNG inside
`BatchedMCTS` is the root Dirichlet noise, which the arena disables. Two
network agents playing from the same position therefore replayed one game
no matter how many were asked for.

These tests pin both halves of the fix: the default stays deterministic,
because every published arena number was measured that way and a silently
stochastic default would change what those numbers mean; and a non-zero
temperature makes the seed decide the move, which is what autoplay and the
browser's engine-vs-engine mode need to produce more than one game.

Torch-free by construction: `UniformEvaluator` is the search's no-network
evaluator and the torch-free install is a tested configuration here.
"""

from __future__ import annotations

import numpy as np

from quantik_models.arena import autoplay
from quantik_models.arena.agents import NetMCTSAgent, PolicyAgent
from quantik_models.env import fastboard as fb
from quantik_models.selfplay.mcts import BatchedMCTS, MCTSParams

# Wide enough that a distribution putting ~90% on its best action still
# shows a second choice; a six-seed sample would report the collapse
# these tests exist to catch about half the time.
SEEDS = tuple(range(24))


class StubEvaluator:
    """A torch-free stand-in for a trained network, spread over its legal
    actions the way a real one is.

    Not `UniformEvaluator`: with uniform priors *and* a flat zero value the
    PUCT search is degenerate — the first child visited has `Q = 0` while
    every unvisited sibling sits at `-fpu_reduction`, and no exploration
    bonus at `c_puct=1.5` and a prior of 1/64 ever closes that gap, so all
    N simulations land on one action and the visit vector is a delta no
    temperature can sample from. Measured: 64 of 64 visits on action 0.
    A real checkpoint spreads its visits (`cpool` at 128 sims: 102 on the
    best action and 26 across nineteen others), which is the distribution
    this stub reproduces.
    """

    def __init__(self, tilt: float = 1.0) -> None:
        self.tilt = tilt

    def __call__(self, boards, legal):
        index = np.arange(fb.ACTION_COUNT, dtype=np.float32)
        shaped = np.exp(-self.tilt * index / fb.ACTION_COUNT)[None, :] * legal
        priors = shaped / shaped.sum(axis=1, keepdims=True).clip(min=1e-9)
        # A value that varies with the position, so sibling branches score
        # differently and the tree actually fans out.
        filled = fb.popcount(fb.occupancy(boards)).astype(np.float32)
        values = np.tanh(0.25 * (filled % 3) - 0.25)
        return priors.astype(np.float32), values.astype(np.float32)


def _params(**kwargs):
    # `fpu_reduction=0.0` rather than the 0.2 default, for the same reason
    # `StubEvaluator` is not `UniformEvaluator`: FPU reduction locks a
    # flat-value search onto its first child, and a search that never fans
    # out is not the shape any trained checkpoint produces. At 0.0 the stub
    # spreads across 25 actions, next to `cpool`'s 20 at 128 simulations.
    kwargs.setdefault("fpu_reduction", 0.0)
    return MCTSParams(simulations=64, leaf_batch=16, dirichlet_weight=0.0, **kwargs)


def _mcts(**kwargs):
    return NetMCTSAgent(StubEvaluator(), params=_params(), name="net", **kwargs)


def _legal(board):
    return fb.legal_masks(board[None, :])[0]


def test_the_mcts_default_is_deterministic_across_seeds() -> None:
    """The arena's contract. Changing this invalidates every published margin."""
    agent = _mcts()
    board = fb.empty_boards(1)[0]
    chosen = {agent.select(board, seed) for seed in SEEDS}
    assert len(chosen) == 1


def test_the_policy_default_is_deterministic_across_seeds() -> None:
    agent = PolicyAgent(StubEvaluator(), name="pol")
    board = fb.empty_boards(1)[0]
    assert len({agent.select(board, seed) for seed in SEEDS}) == 1


def test_a_temperature_makes_the_mcts_seed_decide_the_move() -> None:
    """The reported symptom, at its root: same position, six seeds, one move."""
    agent = _mcts(temperature=1.0)
    board = fb.empty_boards(1)[0]
    assert len({agent.select(board, seed) for seed in SEEDS}) > 1


def test_a_temperature_makes_the_policy_seed_decide_the_move() -> None:
    agent = PolicyAgent(StubEvaluator(), name="pol", temperature=1.0)
    board = fb.empty_boards(1)[0]
    assert len({agent.select(board, seed) for seed in SEEDS}) > 1


def test_temperature_plies_bounds_the_sampling_to_the_opening() -> None:
    """Diversity is wanted in the opening, not in the moves that decide the
    game: past the cut-off the agent must be the same deterministic player
    the arena measured."""
    agent = _mcts(temperature=1.0, temperature_plies=2)
    empty = fb.empty_boards(1)[0]
    assert len({agent.select(empty, seed) for seed in SEEDS}) > 1

    deep = fb.apply_actions(
        np.repeat(empty[None, :], 1, 0), np.array([0], dtype=np.int64)
    )[0]
    deep = fb.apply_actions(deep[None, :], np.array([6], dtype=np.int64))[0]
    assert int(fb.popcount(fb.occupancy(deep[None, :]))[0]) == 2
    assert len({agent.select(deep, seed) for seed in SEEDS}) == 1


def test_sampling_only_ever_returns_a_legal_action() -> None:
    board = fb.apply_actions(fb.empty_boards(1), np.array([0], dtype=np.int64))[0]
    for agent in (_mcts(temperature=1.0), PolicyAgent(StubEvaluator(), temperature=1.0)):
        legal = _legal(board)
        for seed in range(24):
            assert legal[agent.select(board, seed)]


def test_a_tiny_temperature_does_not_overflow_into_a_nan_choice() -> None:
    """`visits ** (1 / 0.01)` is `128 ** 100` — inf, then a nan probability
    vector and a ValueError out of `rng.choice`. The exponent has to be
    taken on weights normalised to their maximum, which caps the base at
    1.0 whatever the temperature is."""
    board = fb.empty_boards(1)[0]
    hot = _mcts(temperature=1.0)
    cold = _mcts(temperature=0.01)
    greedy = _mcts()
    # As T -> 0 the losers underflow to zero and only the joint maximum
    # survives, so a cold pick must land inside the argmax set — a set, not
    # a single action, because the greedy `argmax` breaks a tie by index
    # and sampling breaks it by seed. Neither may raise.
    search = BatchedMCTS(cold.evaluator, cold.params, np.random.default_rng(0))
    visits = search.search(board[None, :], add_noise=False)[0][0]
    best = set(np.flatnonzero(visits == visits.max()).tolist())
    assert {cold.select(board, seed) for seed in SEEDS} <= best
    assert greedy.select(board, 0) in best
    assert hot.select(board, 0) in np.flatnonzero(_legal(board))


def test_autoplay_from_an_empty_board_stops_replaying_one_game() -> None:
    """The end-to-end symptom: `--start-plies 0 --games N` cost N times the
    compute and produced N byte-identical games."""
    specs = [
        {"kind": "uniform-mcts", "name": "a", "temperature": 1.0,
         "params": {"simulations": 32, "leaf_batch": 16,
                    "dirichlet_weight": 0.0, "fpu_reduction": 0.0}},
        {"kind": "uniform-mcts", "name": "b", "temperature": 1.0,
         "params": {"simulations": 16, "leaf_batch": 16,
                    "dirichlet_weight": 0.0, "fpu_reduction": 0.0}},
    ]
    games = autoplay.run(specs, 6, seed=20260829, start_plies=0)
    per_pairing = autoplay.distinct_games(games)
    assert set(per_pairing) == {("a", "b"), ("b", "a")}
    for (mover, responder), counts in per_pairing.items():
        assert counts["distinct"] > 1, f"{mover} vs {responder} replayed one game"


def test_distinct_games_reports_the_collapse_rather_than_hiding_it() -> None:
    """A win rate over N games assumes N games. When the agents are
    deterministic and the start positions repeat, it is over fewer, and the
    interval `arena.pack` prints around it is too narrow. The count has to
    be visible."""
    specs = [
        {"kind": "uniform-mcts", "name": "a",
         "params": {"simulations": 16, "leaf_batch": 16, "dirichlet_weight": 0.0}},
        {"kind": "uniform-mcts", "name": "b",
         "params": {"simulations": 8, "leaf_batch": 16, "dirichlet_weight": 0.0}},
    ]
    games = autoplay.run(specs, 4, seed=1, start_plies=0)
    for counts in autoplay.distinct_games(games).values():
        assert counts == {"games": 4, "distinct": 1}
