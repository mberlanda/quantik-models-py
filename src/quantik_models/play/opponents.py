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

# Opening temperature, and the number of plies it applies to.
#
# Without it a network is a deterministic function of the position: the
# same opponent from the same start replays one game, every game. That is
# the right default for the arena, where the question is how a fixed player
# performs, and the wrong one here, where two people watching engine-vs-
# engine see the same game twice and a human who finds one winning line
# wins with it forever.
#
# It is bounded to the opening because that is where it is nearly free.
# The corpus spans plies 6-13, so the policy head was never trained on the
# first few plies and has no opinion there — measured, `cpool` on the empty
# board puts 0.0167 on each legal action, which is 1/60 to three places.
# An `argmax` over that is not a considered choice; it is whichever action
# index sorts first. Sampling it is the more honest reading of a flat
# distribution, and the network still plays its best move from ply 4 on,
# which is where its training starts to bite.
DEFAULT_OPENING_TEMPERATURE = 1.0
DEFAULT_OPENING_PLIES = 4


@dataclass(frozen=True)
class Opponent:
    opponent_id: str
    label: str
    kind: str
    spec: dict[str, Any]
    model_id: str | None
    simulations: int | None
    # Explicit fields rather than a `spec.get`, because these end up on a
    # recorded game's row: a reader of `game_meta` should not have to know
    # the shape of an `arena.registry` spec to find out what it was played
    # against.
    temperature: float = 0.0
    temperature_plies: int | None = None


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
        # 0.25s, not the 1.0s the other clocked engines carry. Beam checks
        # its budget between levels and cannot interrupt one, so a width-32
        # level overruns: measured 3.75s for a declared 1.0s from an empty
        # board, against 1.00s exactly for `mcts` at the same budget. The
        # overshoot scales with width (w8 @1.0s -> 1.49s), so the fix is a
        # budget whose level fits rather than a narrower beam. 0.25s lands at
        # ~0.36s, next to minimax-d2's 0.42s, which keeps every opponent on
        # this roster responsive enough to play against.
        spec={"kind": "beam", "time_limit_s": 0.25, "beam_width": 32, "name": "beam-w32"},
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


def neural_opponents(
    models: list[PlayModel],
    *,
    temperature: float = DEFAULT_OPENING_TEMPERATURE,
    temperature_plies: int = DEFAULT_OPENING_PLIES,
) -> list[Opponent]:
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
                    "temperature": temperature,
                    "temperature_plies": temperature_plies,
                },
                model_id=model.model_id,
                simulations=0,
                temperature=temperature,
                temperature_plies=temperature_plies,
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
                    "temperature": temperature,
                    "temperature_plies": temperature_plies,
                    "params": dict(_NET_MCTS_PARAMS),
                },
                model_id=model.model_id,
                simulations=_NET_MCTS_SIMULATIONS,
                temperature=temperature,
                temperature_plies=temperature_plies,
            )
        )
    return out


def roster(
    models: list[PlayModel],
    *,
    temperature: float = DEFAULT_OPENING_TEMPERATURE,
    temperature_plies: int = DEFAULT_OPENING_PLIES,
) -> list[Opponent]:
    """The classical table followed by every ready model's two opponents."""
    return list(CLASSICAL) + neural_opponents(
        models, temperature=temperature, temperature_plies=temperature_plies
    )
