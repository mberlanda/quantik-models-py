"""Paired, side-balanced matches between two agents.

Ported from the Rust `bench::head_to_head` design: for every start position
and seed, two games are played — agent A moving first, then agent B moving
first — so a result never reflects which side a sampled position happened to
favour. Quantik is a decisive game (there are no draws: a player who cannot
move has lost), so a match is fully described by the win split.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ..env import fastboard as fb

Board = npt.NDArray[np.uint16]


def sample_start_positions(
    count: int,
    plies: int | Sequence[int],
    seed: int,
    unique: bool = True,
) -> npt.NDArray[np.uint16]:
    """`count` distinct non-terminal boards reached by `plies` random moves.

    Starting matches from a shared, varied set of openings is what keeps a
    deterministic agent pair from replaying one game over and over — and with
    deterministic agents it is the *only* source of variety, since replaying a
    position under a different seed replays the same game.

    `plies` may be a sequence, in which case the openings are spread evenly
    over those depths. Shallower starts leave more of the game in the
    contested region where engines actually differ.

    Positions are deduplicated **up to symmetry**: two openings related by a
    board rotation or a shape relabeling are the same game, and counting them
    as separate matches would silently narrow the confidence interval.
    """
    if not isinstance(plies, int):
        depths = list(plies)
        per_depth = [count // len(depths)] * len(depths)
        for i in range(count - sum(per_depth)):
            per_depth[i] += 1
        parts = [
            sample_start_positions(n, depth, seed + 1000 * depth, unique)
            for depth, n in zip(depths, per_depth)
            if n > 0
        ]
        return np.concatenate(parts)

    rng = np.random.default_rng(seed)
    collected: list[np.ndarray] = []
    seen: set[int] = set()
    while len(collected) < count:
        batch = fb.empty_boards(max(count * 4, 256))
        for _ in range(plies):
            done, _ = fb.terminal_status(batch)
            batch = batch[~done]
            if batch.shape[0] == 0:
                break
            legal = fb.legal_masks(batch)
            scores = rng.random(legal.shape) * legal
            batch = fb.apply_actions(batch, scores.argmax(axis=1))
        if batch.shape[0] == 0:
            continue
        done, _ = fb.terminal_status(batch)
        live = batch[~done]
        if live.shape[0] == 0:
            continue
        for board, key in zip(live, fb.canonical_keys(live).tolist()):
            if unique and key in seen:
                continue
            seen.add(key)
            collected.append(board)
            if len(collected) == count:
                break
    return np.array(collected, dtype=np.uint16)


def play_game(mover, responder, board: Board, seed: int) -> tuple[int, int]:
    """Play out `board`; return `(winner, plies)` with 0 = mover, 1 = responder.

    Mirrors the Rust harness: the loop checks the terminal conditions before
    each turn, so whoever is on move when the position is dead has lost.
    """
    current = board.copy()
    turn = 0
    plies = 0
    agents = (mover, responder)
    while True:
        done, _ = fb.terminal_status(current[None, :])
        if bool(done[0]):
            return 1 - turn, plies
        action = agents[turn].select(current, seed + plies)
        legal = fb.legal_masks(current[None, :])[0]
        if not legal[action]:
            raise ValueError(f"{agents[turn].name} chose illegal action {action}")
        current = fb.apply_actions(current[None, :], np.array([action], dtype=np.int64))[0]
        turn ^= 1
        plies += 1


@dataclass
class MatchResult:
    agent_a: str
    agent_b: str
    wins_a: int = 0
    wins_b: int = 0
    games: int = 0
    plies: list[int] = field(default_factory=list)
    seconds_a: float = 0.0
    seconds_b: float = 0.0
    moves_a: int = 0
    moves_b: int = 0

    @property
    def score_a(self) -> float:
        """Agent A's win rate in [0, 1]; Quantik has no draws."""
        return self.wins_a / self.games if self.games else 0.0

    @property
    def wilson_ci(self) -> tuple[float, float]:
        """95% Wilson score interval on A's win rate."""
        n = self.games
        if n == 0:
            return (0.0, 0.0)
        z = 1.959963984540054
        p = self.score_a
        denom = 1.0 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        margin = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
        return (max(0.0, centre - margin), min(1.0, centre + margin))

    def summary(self) -> str:
        low, high = self.wilson_ci
        return (
            f"{self.agent_a} vs {self.agent_b}: "
            f"{self.wins_a}-{self.wins_b} ({self.score_a:.1%}, "
            f"95% CI {low:.1%}-{high:.1%}) over {self.games} games"
        )

    def to_dict(self) -> dict:
        low, high = self.wilson_ci
        return {
            "agent_a": self.agent_a,
            "agent_b": self.agent_b,
            "wins_a": self.wins_a,
            "wins_b": self.wins_b,
            "games": self.games,
            "score_a": self.score_a,
            "ci_low": low,
            "ci_high": high,
            "mean_plies": float(np.mean(self.plies)) if self.plies else 0.0,
            "ms_per_move_a": 1000 * self.seconds_a / self.moves_a if self.moves_a else 0.0,
            "ms_per_move_b": 1000 * self.seconds_b / self.moves_b if self.moves_b else 0.0,
        }


class _Timed:
    """Wraps an agent to accumulate its own thinking time and move count."""

    def __init__(self, agent) -> None:
        self.agent = agent
        self.name = agent.name
        self.seconds = 0.0
        self.moves = 0

    def select(self, board: Board, seed: int) -> int:
        start = time.perf_counter()
        action = self.agent.select(board, seed)
        self.seconds += time.perf_counter() - start
        self.moves += 1
        return action


def play_match(
    agent_a,
    agent_b,
    positions: npt.NDArray[np.uint16],
    seeds: tuple[int, ...] = (0,),
    progress=None,
) -> MatchResult:
    """Every position x seed is played twice, once with each agent moving first."""
    timed_a, timed_b = _Timed(agent_a), _Timed(agent_b)
    result = MatchResult(agent_a=agent_a.name, agent_b=agent_b.name)
    total = positions.shape[0] * len(seeds) * 2
    played = 0
    for board in positions:
        for seed in seeds:
            for a_moves_first in (True, False):
                mover, responder = (
                    (timed_a, timed_b) if a_moves_first else (timed_b, timed_a)
                )
                winner, plies = play_game(mover, responder, board, seed)
                a_won = (winner == 0) == a_moves_first
                if a_won:
                    result.wins_a += 1
                else:
                    result.wins_b += 1
                result.games += 1
                result.plies.append(plies)
                played += 1
                if progress is not None:
                    progress(played, total)
    result.seconds_a, result.moves_a = timed_a.seconds, timed_a.moves
    result.seconds_b, result.moves_b = timed_b.seconds, timed_b.moves
    return result


def round_robin(agents, positions, seeds=(0,), progress=None) -> list[MatchResult]:
    """Every unordered pair of agents plays a side-balanced match."""
    results = []
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            results.append(
                play_match(agents[i], agents[j], positions, seeds, progress=progress)
            )
    return results
