"""Net-vs-net matches played in lockstep across all games.

A duel between two networks is the gate for promotion, so it runs every
iteration and its cost matters. Playing games one at a time would put the
search on batch-1 network calls — the slowest possible shape on an
accelerator. Here every game advances together: at each ply the live games
are split by whose turn it is, and each side's positions go through its own
network in a single batched search.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..env import fastboard as fb
from .evaluator import Evaluator
from .mcts import BatchedMCTS, MCTSParams


@dataclass
class DuelResult:
    wins_a: int
    wins_b: int
    games: int
    mean_plies: float

    @property
    def score_a(self) -> float:
        return self.wins_a / self.games if self.games else 0.0


def duel(
    evaluator_a: Evaluator,
    evaluator_b: Evaluator,
    params: MCTSParams,
    start_boards: npt.NDArray[np.uint16],
    rng: np.random.Generator,
    add_noise: bool = False,
) -> DuelResult:
    """Play each start position twice — once with each side assigned to A.

    Returns A's record. Side-balancing is built in, so `score_a` is directly
    comparable to the arena's win rates.
    """
    boards = np.concatenate([start_boards, start_boards])
    n = boards.shape[0]
    half = start_boards.shape[0]
    # `a_side[i]` is the color agent A plays in game i.
    a_side = np.concatenate(
        [fb.side_to_move(start_boards), 1 - fb.side_to_move(start_boards)]
    )
    game_ids = np.arange(n)
    search_a = BatchedMCTS(evaluator_a, params, rng)
    search_b = BatchedMCTS(evaluator_b, params, rng)

    wins_a = 0
    plies_played = []
    for ply in range(fb.SQUARES + 1):
        if boards.shape[0] == 0:
            break
        done, _ = fb.terminal_status(boards)
        if done.any():
            # The mover at a terminal position has lost.
            loser_is_a = fb.side_to_move(boards[done]) == a_side[game_ids[done]]
            wins_a += int((~loser_is_a).sum())
            plies_played.extend([ply] * int(done.sum()))
            keep = ~done
            boards, game_ids = boards[keep], game_ids[keep]
            if boards.shape[0] == 0:
                break

        actions = np.zeros(boards.shape[0], dtype=np.int64)
        turn_is_a = fb.side_to_move(boards) == a_side[game_ids]
        for mask, search in ((turn_is_a, search_a), (~turn_is_a, search_b)):
            if not mask.any():
                continue
            visits, _ = search.search(boards[mask], add_noise=add_noise)
            actions[mask] = visits.argmax(axis=1)
        boards = fb.apply_actions(boards, actions)

    del half
    return DuelResult(
        wins_a=wins_a,
        wins_b=n - wins_a,
        games=n,
        mean_plies=float(np.mean(plies_played)) if plies_played else 0.0,
    )
