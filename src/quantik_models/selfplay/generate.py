"""Self-play game generation.

All games in a batch advance in lockstep — every live game's root search runs
inside one `BatchedMCTS.search` call, so the network sees the whole batch's
leaves at once. A Quantik game is at most 16 plies, so a batch of any size
finishes in at most 16 rounds.

Rows carry both training signals: `outcome` (the AlphaZero `z`, the game's
result from that position's mover's perspective) and `root_value` (the
search's own backed-up estimate). Short games make `z` a high-variance
target, so the trainer blends the two rather than being forced to pick.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ..env import fastboard as fb
from .evaluator import Evaluator
from .mcts import BatchedMCTS, MCTSParams


@dataclass(frozen=True)
class SelfPlayConfig:
    games: int = 256
    mcts: MCTSParams = field(default_factory=MCTSParams)
    # Plies played by sampling from the visit distribution before switching
    # to greedy play. Quantik games are short, so this is most of the game.
    temperature_plies: int = 6
    temperature: float = 1.0
    add_root_noise: bool = True


@dataclass
class SelfPlayData:
    """One batch of self-play rows, one row per position actually played."""

    boards: npt.NDArray[np.uint16]
    policies: npt.NDArray[np.float32]
    outcomes: npt.NDArray[np.float32]
    root_values: npt.NDArray[np.float32]
    plies: npt.NDArray[np.int64]
    game_ids: npt.NDArray[np.int64]

    def __len__(self) -> int:
        return int(self.boards.shape[0])

    def stats(self) -> dict[str, float]:
        _, counts = np.unique(self.game_ids, return_counts=True)
        return {
            "rows": float(len(self)),
            "games": float(counts.size),
            "mean_game_plies": float(counts.mean()),
            "max_game_plies": float(counts.max()),
            "first_player_win_rate": float(
                np.mean(self.outcomes[self.plies == 0] > 0) if (self.plies == 0).any() else 0.0
            ),
            "unique_positions": float(len(set(fb.canonical_keys(self.boards).tolist()))),
            "mean_abs_root_value": float(np.abs(self.root_values).mean()),
        }


def play_batch(
    evaluator: Evaluator,
    config: SelfPlayConfig,
    rng: np.random.Generator,
    start_boards: npt.NDArray[np.uint16] | None = None,
) -> SelfPlayData:
    """Play `config.games` games to completion and return every position seen."""
    search = BatchedMCTS(evaluator, config.mcts, rng)
    boards = (
        fb.empty_boards(config.games) if start_boards is None else start_boards.copy()
    )
    game_ids = np.arange(boards.shape[0], dtype=np.int64)

    rec_boards: list[npt.NDArray[np.uint16]] = []
    rec_policies: list[npt.NDArray[np.float32]] = []
    rec_values: list[npt.NDArray[np.float32]] = []
    rec_plies: list[npt.NDArray[np.int64]] = []
    rec_games: list[npt.NDArray[np.int64]] = []
    # Ply at which each game ended; the mover there is the loser.
    final_ply = np.zeros(boards.shape[0], dtype=np.int64)

    for ply in range(fb.SQUARES + 1):
        if boards.shape[0] == 0:
            break
        done, _ = fb.terminal_status(boards)
        if done.any():
            final_ply[game_ids[done]] = ply
            boards = boards[~done]
            game_ids = game_ids[~done]
            if boards.shape[0] == 0:
                break

        visits, root_values = search.search(boards, add_noise=config.add_root_noise)
        policies = visits / visits.sum(axis=1, keepdims=True)

        rec_boards.append(boards.copy())
        rec_policies.append(policies.astype(np.float32))
        rec_values.append(root_values.astype(np.float32))
        rec_plies.append(np.full(boards.shape[0], ply, dtype=np.int64))
        rec_games.append(game_ids.copy())

        if ply < config.temperature_plies and config.temperature > 0.0:
            weights = visits ** (1.0 / config.temperature)
            weights /= weights.sum(axis=1, keepdims=True)
            actions = np.array(
                [rng.choice(fb.ACTION_COUNT, p=w) for w in weights], dtype=np.int64
            )
        else:
            actions = visits.argmax(axis=1)
        boards = fb.apply_actions(boards, actions)

    all_boards = np.concatenate(rec_boards)
    all_plies = np.concatenate(rec_plies)
    all_games = np.concatenate(rec_games)
    # The mover at `final_ply` lost, so a position's mover won exactly when
    # its ply has the opposite parity.
    outcomes = np.where((final_ply[all_games] - all_plies) % 2 == 0, -1.0, 1.0)

    return SelfPlayData(
        boards=all_boards,
        policies=np.concatenate(rec_policies),
        outcomes=outcomes.astype(np.float32),
        root_values=np.concatenate(rec_values),
        plies=all_plies,
        game_ids=all_games,
    )


def augment(
    data: SelfPlayData, factor: int, rng: np.random.Generator
) -> tuple[npt.NDArray[np.uint16], npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Replay every row under `factor` random symmetries of the 192-element group.

    Returns `(boards, policies, outcomes, root_values)`; the two value
    signals are symmetry-invariant and just repeat.
    """
    n = len(data)
    boards = [data.boards]
    policies = [data.policies]
    for _ in range(max(0, factor - 1)):
        spatial, shape = fb.random_symmetries(n, rng)
        boards.append(fb.transform_boards(data.boards, spatial, shape))
        policies.append(fb.transform_policies(data.policies, spatial, shape))
    reps = len(boards)
    return (
        np.concatenate(boards),
        np.concatenate(policies),
        np.tile(data.outcomes, reps),
        np.tile(data.root_values, reps),
    )
