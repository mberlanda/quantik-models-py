"""The control agent has to be the same search, not merely a similar one.

`uniform-mcts` exists to answer "how much is the network contributing".
That only works if everything except the evaluator is held fixed — a
control running a different algorithm (the `mcts` kind is UCB1 with random
rollouts, from quantik-core) cannot answer it.
"""

from __future__ import annotations

import pytest

from quantik_models.arena.registry import build_agent
from quantik_models.env import fastboard as fb


def test_uniform_mcts_needs_no_checkpoint() -> None:
    """The point of the control is that there is no network."""
    agent = build_agent({"kind": "uniform-mcts", "params": {"simulations": 8}})
    assert agent.name == "uniform-mcts128" or "mcts" in agent.name


def test_it_is_the_same_search_class_as_the_network_agents() -> None:
    from quantik_models.arena.agents import NetMCTSAgent

    control = build_agent({"kind": "uniform-mcts", "params": {"simulations": 8}})
    assert isinstance(control, NetMCTSAgent), (
        "the control must run the same search; a different algorithm cannot "
        "isolate the network's contribution"
    )


def test_it_plays_legal_moves() -> None:
    agent = build_agent(
        {"kind": "uniform-mcts", "params": {"simulations": 16}, "name": "control"}
    )
    board = fb.empty_boards(1)[0]
    for ply in range(4):
        action = agent.select(board, seed=ply)
        assert fb.legal_masks(board[None, :])[0][action]
        board = fb.apply_actions(
            board[None, :], __import__("numpy").array([action], dtype="int64")
        )[0]


def test_simulation_budget_is_respected() -> None:
    agent = build_agent({"kind": "uniform-mcts", "params": {"simulations": 64}})
    assert agent.params.simulations == 64


def test_unknown_kind_is_still_refused() -> None:
    with pytest.raises(ValueError, match="unknown agent kind"):
        build_agent({"kind": "not-an-agent"})
