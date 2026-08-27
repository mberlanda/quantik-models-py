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

    def __init__(self, evaluator, name: str = "net-policy", temperature: float = 0.0):
        self.evaluator = evaluator
        self.name = name
        self.temperature = temperature

    def select(self, board: Board, seed: int) -> int:
        boards = board[None, :]
        legal = fb.legal_masks(boards)
        priors, _ = self.evaluator(boards, legal)
        if self.temperature <= 0.0:
            return int(priors[0].argmax())
        weights = np.where(legal[0], priors[0] ** (1.0 / self.temperature), 0.0)
        weights = weights / weights.sum()
        return int(np.random.default_rng(seed).choice(fb.ACTION_COUNT, p=weights))

    def config_label(self) -> str:
        return f"net-policy(temperature={self.temperature})"


class NetMCTSAgent:
    """The network inside `BatchedMCTS` — the full AlphaZero-style player."""

    def __init__(self, evaluator, simulations: int = 128, params: MCTSParams | None = None,
                 name: str | None = None):
        self.evaluator = evaluator
        self.params = params or MCTSParams(simulations=simulations)
        self.name = name or f"net-mcts-{self.params.simulations}"

    def select(self, board: Board, seed: int) -> int:
        search = BatchedMCTS(self.evaluator, self.params, np.random.default_rng(seed))
        visits, _ = search.search(board[None, :], add_noise=False)
        return int(visits[0].argmax())

    def config_label(self) -> str:
        return f"net-mcts(sims={self.params.simulations},c_puct={self.params.c_puct})"
