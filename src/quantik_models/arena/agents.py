"""Agents playable in the arena, all speaking the 64-slot action index.

The classical agents wrap `quantik_core`'s engines unchanged — they are the
incumbents the network has to beat, so they must be the real thing rather
than a reimplementation. The network agents drive `BatchedMCTS` (or the raw
policy head) over `fastboard`.

Every agent takes a board as an `(8,) uint16` array and returns one action
index. Agents carry a `name` and a `config_label` for reporting, mirroring
the Rust `EngineAdapter` surface.
"""

from __future__ import annotations

import random
from typing import Protocol

import numpy as np
import numpy.typing as npt

from quantik_core import State
from quantik_core.beam_search import BeamSearchConfig, BeamSearchEngine
from quantik_core.mcts import MCTSConfig, MCTSEngine
from quantik_core.minimax import MinimaxConfig, MinimaxEngine
from quantik_core.move import generate_legal_moves_list

from ..env import fastboard as fb
from ..selfplay.mcts import BatchedMCTS, MCTSParams

Board = npt.NDArray[np.uint16]


class Agent(Protocol):
    name: str

    def select(self, board: Board, seed: int) -> int:
        ...

    def config_label(self) -> str:
        ...


def _state(board: Board) -> State:
    return State(tuple(int(v) for v in board))


def _ply(board: Board) -> int:
    return int(fb.popcount(fb.occupancy(board[None, :]))[0])


def _sample(weights: np.ndarray, temperature: float, seed: int) -> int:
    """Pick an index from `weights` at `temperature`, or its argmax at 0.

    `weights` is a non-negative score per action — MCTS visit counts, or
    policy priors — already zero everywhere illegal, so an action with a
    zero score can never be drawn and the legality check upstream stays
    intact.

    The exponent is taken on weights divided by their maximum rather than
    on the raw scores. Both forms are proportional and so describe the same
    distribution, but the raw one overflows: `visits ** (1 / 0.01)` is
    `128 ** 100`, which is `inf`, and one `inf` turns the normalised vector
    into `nan` and `rng.choice` into a `ValueError`. Dividing first caps
    the base at 1.0, so a small temperature underflows the losers to zero
    and converges on the argmax — which is the limit it should converge on.
    """
    if temperature <= 0.0:
        return int(weights.argmax())
    top = float(weights.max())
    if top <= 0.0:
        return int(weights.argmax())
    scaled = np.zeros(weights.shape, dtype=np.float64)
    positive = weights > 0.0
    scaled[positive] = (weights[positive] / top) ** (1.0 / temperature)
    total = scaled.sum()
    if not np.isfinite(total) or total <= 0.0:
        return int(weights.argmax())
    return int(np.random.default_rng(seed).choice(scaled.size, p=scaled / total))


class RandomAgent:
    """Uniform choice among legal actions."""

    def __init__(self, name: str = "random") -> None:
        self.name = name

    def select(self, board: Board, seed: int) -> int:
        legal = np.flatnonzero(fb.legal_masks(board[None, :])[0])
        return int(random.Random(seed).choice(legal.tolist()))

    def config_label(self) -> str:
        return "random"


class MinimaxAgent:
    """`quantik_core.minimax.MinimaxEngine` — iterative-deepening alpha-beta."""

    def __init__(self, time_limit_s: float | None = 0.1, max_depth: int = 16, name: str | None = None):
        self.time_limit_s = time_limit_s
        self.max_depth = max_depth
        self.name = name or (
            f"minimax@{time_limit_s}s" if time_limit_s else f"minimax-d{max_depth}"
        )

    def select(self, board: Board, seed: int) -> int:
        engine = MinimaxEngine(
            MinimaxConfig(
                max_depth=self.max_depth,
                time_limit_s=self.time_limit_s,
                random_seed=seed,
            )
        )
        move = engine.search(_state(board)).best_move
        return move.shape * 16 + move.position

    def config_label(self) -> str:
        return f"minimax(depth={self.max_depth},time={self.time_limit_s})"


class CoreMCTSAgent:
    """`quantik_core.mcts.MCTSEngine` — UCB1 MCTS with random rollouts."""

    def __init__(
        self,
        max_iterations: int = 2000,
        time_limit_s: float | None = 0.1,
        name: str | None = None,
    ):
        self.max_iterations = max_iterations
        self.time_limit_s = time_limit_s
        self.name = name or (
            f"mcts@{time_limit_s}s" if time_limit_s else f"mcts-{max_iterations}"
        )

    def select(self, board: Board, seed: int) -> int:
        engine = MCTSEngine(
            MCTSConfig(
                max_iterations=self.max_iterations,
                time_limit_s=self.time_limit_s,
                random_seed=seed,
            )
        )
        move, _ = engine.search(_state(board))
        return move.shape * 16 + move.position

    def config_label(self) -> str:
        return f"mcts(iters={self.max_iterations},time={self.time_limit_s})"


class BeamAgent:
    """`quantik_core.beam_search.BeamSearchEngine` — level-by-level beam."""

    def __init__(
        self,
        beam_width: int = 64,
        rollouts: int = 8,
        time_limit_s: float | None = 0.1,
        name: str | None = None,
    ):
        self.beam_width = beam_width
        self.rollouts = rollouts
        self.time_limit_s = time_limit_s
        self.name = name or (
            f"beam@{time_limit_s}s" if time_limit_s else f"beam-w{beam_width}"
        )

    def select(self, board: Board, seed: int) -> int:
        engine = BeamSearchEngine(
            BeamSearchConfig(
                beam_width=self.beam_width,
                rollouts_per_candidate=self.rollouts,
                time_limit_s=self.time_limit_s,
                random_seed=seed,
            )
        )
        result = engine.search(_state(board))
        # BeamSearchResult exposes root moves aggregated from its sampled
        # leaves rather than a single best move.
        ranked = result.ranked_root_moves(top_k=1)
        move = ranked[0].move if ranked else generate_legal_moves_list(
            tuple(int(v) for v in board)
        )[0]
        return move.shape * 16 + move.position

    def config_label(self) -> str:
        return f"beam(width={self.beam_width},rollouts={self.rollouts},time={self.time_limit_s})"


class PolicyAgent:
    """The network's policy head alone — one forward pass, zero search."""

    def __init__(
        self,
        evaluator,
        name: str = "net-policy",
        temperature: float = 0.0,
        temperature_plies: int | None = None,
    ):
        self.evaluator = evaluator
        self.name = name
        self.temperature = temperature
        self.temperature_plies = temperature_plies

    def _temperature_at(self, board: Board) -> float:
        """The temperature in force at this position.

        `temperature_plies` bounds sampling to the opening: `None` applies
        it to the whole game, an integer applies it while fewer than that
        many pieces are on the board. The board carries its own ply, so no
        caller has to thread one through `select`.
        """
        if self.temperature <= 0.0:
            return 0.0
        if self.temperature_plies is None:
            return self.temperature
        return self.temperature if _ply(board) < self.temperature_plies else 0.0

    def select(self, board: Board, seed: int) -> int:
        boards = board[None, :]
        legal = fb.legal_masks(boards)
        priors, _ = self.evaluator(boards, legal)
        # `np.where` rather than trusting the evaluator's own masking: a
        # prior that leaked onto an illegal action would otherwise become a
        # drawable outcome once the temperature stops being zero.
        weights = np.where(legal[0], priors[0], 0.0)
        return _sample(weights, self._temperature_at(board), seed)

    def config_label(self) -> str:
        return (
            f"net-policy(temperature={self.temperature},"
            f"plies={self.temperature_plies})"
        )


class NetMCTSAgent:
    """The network inside `BatchedMCTS` — the full AlphaZero-style player.

    Configure `params.time_limit_s` to play on the same clock as a
    time-limited classical engine; otherwise `params.simulations` is the
    budget.
    """

    def __init__(self, evaluator, simulations: int = 128, params: MCTSParams | None = None,
                 name: str | None = None, temperature: float = 0.0,
                 temperature_plies: int | None = None):
        self.evaluator = evaluator
        self.params = params or MCTSParams(simulations=simulations)
        self.temperature = temperature
        self.temperature_plies = temperature_plies
        if name:
            self.name = name
        elif self.params.time_limit_s:
            self.name = f"net-mcts@{self.params.time_limit_s * 1000:.0f}ms"
        else:
            self.name = f"net-mcts-{self.params.simulations}"

    def _temperature_at(self, board: Board) -> float:
        """See `PolicyAgent._temperature_at` — the same schedule, so an
        opponent's opening variety does not depend on which of the two
        kinds it happens to be."""
        if self.temperature <= 0.0:
            return 0.0
        if self.temperature_plies is None:
            return self.temperature
        return self.temperature if _ply(board) < self.temperature_plies else 0.0

    def select(self, board: Board, seed: int) -> int:
        search = BatchedMCTS(self.evaluator, self.params, np.random.default_rng(seed))
        visits, _ = search.search(board[None, :], add_noise=False)
        # Sampling the *visit counts*, not the priors: the visits are what
        # the search converged on, so a temperature here trades strength
        # for variety along the search's own ranking rather than throwing
        # the search away. Root Dirichlet noise (`add_noise`) is the other
        # place variety could come from and is deliberately not it — it
        # perturbs the tree the search is built on, so its cost is spread
        # through every simulation instead of landing on one choice.
        return _sample(visits[0].astype(np.float64), self._temperature_at(board), seed)

    def config_label(self) -> str:
        budget = (
            f"time={self.params.time_limit_s}"
            if self.params.time_limit_s
            else f"sims={self.params.simulations}"
        )
        return (
            f"net-mcts({budget},c_puct={self.params.c_puct},"
            f"leaf_batch={self.params.leaf_batch},"
            f"temperature={self.temperature},plies={self.temperature_plies})"
        )
