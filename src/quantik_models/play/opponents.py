"""The roster a browser dropdown offers: classical engines plus every ready
trained model, each ready to become an `arena.agents.Agent` through
`arena.registry.build_agent`.

Why the names matter: `minimax-d2`, `cpool@128` and `uniform-mcts128` are
not invented here — they are the exact agent names `arena.match` and
`arena.autoplay` already stamp into `runs/eval/*/games.json`. Keeping this
roster's ids and its specs' `name` fields in lockstep with those means a
human game recorded by the play service and a benchmark game recorded by
the arena refer to the same opponent by the same string. That is what
makes them one dataset — comparable rows a later analysis can pool — rather
than two datasets that need reconciling before they can be compared at
all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .registry import PlayModel

# The network's own budget at each opponent: 0 for the bare policy head
# (one forward pass, no search) and 128 for the AlphaZero-style search,
# matching the sweep the arena already ran these checkpoints at.
_NET_MCTS_SIMULATIONS = 128
_NET_MCTS_PARAMS = {
    "simulations": _NET_MCTS_SIMULATIONS,
    "leaf_batch": 32,
    "dirichlet_weight": 0.0,
}


@dataclass(frozen=True)
class Opponent:
    opponent_id: str
    label: str
    kind: str
    spec: dict[str, Any]
    model_id: str | None
    simulations: int | None


CLASSICAL: tuple[Opponent, ...] = (
    Opponent(
        opponent_id="random",
        label="Random",
        kind="random",
        spec={"kind": "random", "name": "random"},
        model_id=None,
        simulations=None,
    ),
    Opponent(
        opponent_id="minimax-d2",
        label="Minimax (depth 2)",
        kind="minimax",
        spec={"kind": "minimax", "time_limit_s": None, "max_depth": 2, "name": "minimax-d2"},
        model_id=None,
        simulations=None,
    ),
    Opponent(
        opponent_id="minimax-d3",
        label="Minimax (depth 3)",
        kind="minimax",
        spec={"kind": "minimax", "time_limit_s": None, "max_depth": 3, "name": "minimax-d3"},
        model_id=None,
        simulations=None,
    ),
    Opponent(
        opponent_id="mcts-1s",
        label="MCTS (1s)",
        kind="mcts",
        spec={
            "kind": "mcts",
            "time_limit_s": 1.0,
            "max_iterations": 1_000_000,
            "name": "mcts-1s",
        },
        model_id=None,
        simulations=None,
    ),
    Opponent(
        opponent_id="beam-w32",
        label="Beam search (width 32)",
        kind="beam",
        spec={"kind": "beam", "time_limit_s": 1.0, "beam_width": 32, "name": "beam-w32"},
        model_id=None,
        simulations=None,
    ),
    Opponent(
        opponent_id="uniform-mcts128",
        label="Uniform MCTS (128 sims, no network)",
        kind="uniform-mcts",
        spec={
            "kind": "uniform-mcts",
            "name": "uniform-mcts128",
            "params": {"simulations": 128, "leaf_batch": 32, "dirichlet_weight": 0.0},
        },
        model_id=None,
        simulations=128,
    ),
)


def neural_opponents(models: list[PlayModel]) -> list[Opponent]:
    """Two opponents per `ready` model: the bare policy head, and a search.

    A refused model is skipped entirely — it has no weights to trust, so
    there is nothing here to offer a player, unlike `scan_models`, which
    reports it anyway for the person staging models to see why.
    """
    out: list[Opponent] = []
    for model in models:
        if model.status != "ready":
            continue
        checkpoint = str(model.path)
        policy_id = f"{model.model_id}@0"
        out.append(
            Opponent(
                opponent_id=policy_id,
                label=f"{model.model_id} (policy only)",
                kind="net-policy",
                spec={
                    "kind": "net-policy",
                    "checkpoint": checkpoint,
                    "device": "cpu",
                    "name": policy_id,
                },
                model_id=model.model_id,
                simulations=0,
            )
        )
        mcts_id = f"{model.model_id}@{_NET_MCTS_SIMULATIONS}"
        out.append(
            Opponent(
                opponent_id=mcts_id,
                label=f"{model.model_id} (MCTS, {_NET_MCTS_SIMULATIONS} sims)",
                kind="net-mcts",
                spec={
                    "kind": "net-mcts",
                    "checkpoint": checkpoint,
                    "device": "cpu",
                    "name": mcts_id,
                    "params": dict(_NET_MCTS_PARAMS),
                },
                model_id=model.model_id,
                simulations=_NET_MCTS_SIMULATIONS,
            )
        )
    return out


def roster(models: list[PlayModel]) -> list[Opponent]:
    """The classical table followed by every ready model's two opponents."""
    return list(CLASSICAL) + neural_opponents(models)
