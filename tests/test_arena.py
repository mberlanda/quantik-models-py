"""Arena correctness: who won, and was the match actually balanced.

Every headline number in this project comes out of `play_match`, so its
bookkeeping is worth pinning down: Quantik's loser is whoever is on move at a
terminal position, and a paired match must attribute results to agents rather
than to colors.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantik_models.arena.agents import RandomAgent
from quantik_models.arena.match import play_game, play_match, sample_start_positions
from quantik_models.env import fastboard as fb


class ScriptedAgent:
    """Plays a fixed action when legal, else the lowest legal one."""

    def __init__(self, name: str, preferred: int | None = None):
        self.name = name
        self.preferred = preferred

    def select(self, board, seed: int) -> int:
        legal = np.flatnonzero(fb.legal_masks(board[None, :])[0])
        if self.preferred is not None and self.preferred in legal:
            return int(self.preferred)
        return int(legal[0])

    def config_label(self) -> str:
        return "scripted"


def test_the_player_on_move_at_a_terminal_position_loses():
    """Replay each game and confirm the reported winner is the other side."""
    positions = sample_start_positions(30, plies=6, seed=4)
    a, b = RandomAgent("a"), RandomAgent("b")
    for board in positions:
        winner, plies = play_game(a, b, board, seed=11)
        # Re-play deterministically with the same seeds to reach the same end.
        current = board.copy()
        turn = 0
        for ply in range(plies):
            action = (a, b)[turn].select(current, 11 + ply)
            current = fb.apply_actions(
                current[None, :], np.array([action], dtype=np.int64)
            )[0]
            turn ^= 1
        done, _ = fb.terminal_status(current[None, :])
        assert bool(done[0]), "the game stopped at a non-terminal position"
        assert winner == 1 - turn, "the winner is not the side that just moved"


def test_a_match_against_yourself_is_balanced():
    """Two identical deterministic agents must split a side-balanced match."""
    positions = sample_start_positions(24, plies=[4, 5], seed=6)
    result = play_match(ScriptedAgent("x"), ScriptedAgent("y"), positions, seeds=(0,))
    assert result.games == 48
    assert result.wins_a == result.wins_b, "identical agents did not split"


def test_wins_are_attributed_to_agents_not_colors():
    """The stronger agent's record must not depend on which side it starts."""
    positions = sample_start_positions(20, plies=5, seed=8)
    strong, weak = ScriptedAgent("strong"), ScriptedAgent("weak")
    result = play_match(strong, weak, positions, seeds=(0,))
    flipped = play_match(weak, strong, positions, seeds=(0,))
    assert result.wins_a == flipped.wins_b
    assert result.wins_b == flipped.wins_a


def test_illegal_moves_are_rejected():
    class Cheater:
        name = "cheater"

        def select(self, board, seed):
            legal = fb.legal_masks(board[None, :])[0]
            return int(np.flatnonzero(~legal)[0])

        def config_label(self):
            return "cheater"

    board = sample_start_positions(1, plies=4, seed=1)[0]
    with pytest.raises(ValueError, match="illegal action"):
        play_game(Cheater(), RandomAgent("r"), board, seed=0)


def test_wilson_interval_brackets_the_point_estimate():
    positions = sample_start_positions(16, plies=5, seed=2)
    result = play_match(RandomAgent("a"), RandomAgent("b"), positions, seeds=(0, 1))
    low, high = result.wilson_ci
    assert low <= result.score_a <= high
    assert 0.0 <= low and high <= 1.0


def test_start_positions_are_symmetry_distinct_and_live():
    positions = sample_start_positions(120, plies=[3, 4, 5], seed=3)
    assert positions.shape[0] == 120
    assert len(set(fb.canonical_keys(positions).tolist())) == 120
    done, _ = fb.terminal_status(positions)
    assert not done.any()
    depths = fb.popcount(fb.occupancy(positions))
    assert set(depths.tolist()) == {3, 4, 5}


def test_parallel_and_serial_matches_agree():
    from quantik_models.arena.parallel import play_match_parallel

    positions = sample_start_positions(8, plies=5, seed=12)
    spec_a = {"kind": "random", "name": "ra"}
    spec_b = {"kind": "minimax", "max_depth": 2, "time_limit_s": None, "name": "mm2"}
    serial = play_match(
        RandomAgent("ra"),
        __import__("quantik_models.arena.agents", fromlist=["MinimaxAgent"]).MinimaxAgent(
            time_limit_s=None, max_depth=2, name="mm2"
        ),
        positions,
        seeds=(0,),
    )
    parallel = play_match_parallel(spec_a, spec_b, positions, seeds=(0,), workers=2)
    assert (serial.wins_a, serial.wins_b) == (parallel.wins_a, parallel.wins_b)


def test_checkpoint_round_trips_through_the_agent_registry(tmp_path):
    """A published checkpoint must rebuild into an identical evaluator.

    The registry reconstructs the architecture by parsing the manifest's
    `architecture` string, so a checkpoint whose shape cannot be recovered
    would fail only at load time, in a worker, mid-match.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")

    from quantik_models.arena.registry import build_agent, load_evaluator
    from quantik_models.export.checkpoint import export_checkpoint
    from quantik_models.model.policy_value_net import PolicyValueNet, PolicyValueNetConfig

    torch.manual_seed(0)
    model = PolicyValueNet(PolicyValueNetConfig(channels=24, blocks=3)).eval()
    export_checkpoint(
        model, out_dir=tmp_path / "ckpt", model_id="test", training_report={"run": "test"}
    )

    evaluator = load_evaluator(tmp_path / "ckpt", "cpu")
    boards = sample_start_positions(8, plies=5, seed=1)
    legal = fb.legal_masks(boards)
    priors, values = evaluator(boards, legal)

    with torch.no_grad():
        logits, expected_values = model(torch.from_numpy(fb.encode_tensors(boards)))
        logits = logits.masked_fill(~torch.from_numpy(legal), torch.finfo(logits.dtype).min)
        expected_priors = torch.softmax(logits, dim=-1).numpy()
    assert np.allclose(priors, expected_priors, atol=1e-6)
    assert np.allclose(values, expected_values.numpy(), atol=1e-6)

    assert np.all(priors[~legal] == 0.0)
    assert np.allclose(priors.sum(axis=1), 1.0, atol=1e-5)

    agent = build_agent(
        {
            "kind": "net-mcts",
            "checkpoint": str(tmp_path / "ckpt"),
            "device": "cpu",
            "params": {"simulations": 16, "dirichlet_weight": 0.0},
            "name": "round-trip",
        }
    )
    action = agent.select(boards[0], seed=0)
    assert legal[0][action]
