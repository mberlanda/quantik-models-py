"""The move handler: everything the play service does, minus the HTTP.

This module is deliberately transport-free. `PlayService.choose_move` takes
a decoded `quantik.engine-request.v1` dict and returns a decoded
`quantik.engine-response.v1` dict; a `ServiceError` carries the status code
the eventual server should send. Keeping the rules here rather than in a
request handler means all of it is unit-testable without a socket, and it
is the same split `quantik-api-rust` uses (`validate_request` and
`run_search` are separate from the axum route).

Two things in here are not obvious and are the reason the module exists at
all: the client's legality claim is checked rather than believed, and a
checkpoint that changes underneath a running service is refused rather than
served stale. Both are documented at the functions that implement them.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..env import fastboard as fb
from ..export.digest import file_digest
from . import opponents as op
from . import registry as reg

REQUEST_SCHEMA = "quantik.engine-request.v1"
RESPONSE_SCHEMA = "quantik.engine-response.v1"

_WEIGHTS_NAME = "weights.safetensors"
_SEED_BITS = 32


class ServiceError(Exception):
    """A refusal, carrying the HTTP status the server should return.

    The status is part of the contract, not decoration: a 400 says the
    caller sent nonsense, a 422 says the caller sent a well-formed request
    this position cannot satisfy, and a 500 says the fault is here. The
    tests assert on the number for that reason.
    """

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass
class _CachedAgent:
    """An agent, plus the identity of the weights file it was built from."""

    agent: Any
    mtime_ns: int
    size: int


@dataclass(frozen=True)
class _Position:
    boards: Any  # (1, 8) uint16
    legal: Any  # (64,) bool
    legal_indices: tuple[int, ...]


def _require_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ServiceError(400, f"{field} must be an integer, got {type(value).__name__}")
    return value


def validate_request(request: Any) -> tuple[str, int, tuple[int, ...], dict[str, Any]]:
    """Everything checkable without a board, mirroring the Rust gateway.

    `quantik-api-rust/src/lib.rs::validate_request` rejects exactly these
    four things, and the two services have to agree — the visualizer sends
    one request shape to whichever is listening, so a body one accepts and
    the other rejects is a bug in this file, not a difference of opinion.
    """
    if not isinstance(request, dict):
        raise ServiceError(400, "request body must be a JSON object")
    schema = request.get("schema")
    if schema != REQUEST_SCHEMA:
        raise ServiceError(400, f"schema must be {REQUEST_SCHEMA!r}, got {schema!r}")

    qfen = request.get("qfen")
    if not isinstance(qfen, str) or not qfen:
        raise ServiceError(400, "qfen must be a non-empty string")

    side_to_move = _require_int(request.get("side_to_move"), "side_to_move")
    if side_to_move not in (0, 1):
        raise ServiceError(400, f"side_to_move must be 0 or 1, got {side_to_move}")

    claimed = request.get("legal_action_indices")
    if not isinstance(claimed, list):
        raise ServiceError(400, "legal_action_indices must be a list")
    for index in claimed:
        value = _require_int(index, "legal_action_indices entries")
        if not 0 <= value < fb.ACTION_COUNT:
            raise ServiceError(
                400,
                f"legal_action_indices must contain values from 0 through "
                f"{fb.ACTION_COUNT - 1}, got {value}",
            )

    config = request.get("config")
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise ServiceError(400, "config must be a JSON object")

    return qfen, side_to_move, tuple(sorted(set(claimed))), config


def resolve_position(qfen: str, side_to_move: int, claimed: tuple[int, ...]) -> _Position:
    """Recompute the position from `quantik-core`'s rules, and check the claim.

    The browser computes legality itself so it can grey out squares, and it
    sends what it computed. That list is treated as a claim to be verified,
    never as an input: if the JavaScript rules and the Python rules ever
    disagree, this comparison is the only place in the system that would
    notice, and it turns a silently mis-scored game into a 422 naming the
    exact action indices the two sides disagree about.
    """
    try:
        boards = fb.from_qfen(qfen)
    except ValueError as exc:
        raise ServiceError(400, str(exc)) from exc

    done, _ = fb.terminal_status(boards)
    if bool(done[0]):
        raise ServiceError(422, "position is already decided; there is no move to make")

    actual_side = int(fb.side_to_move(boards)[0])
    if actual_side != side_to_move:
        raise ServiceError(
            422,
            f"side_to_move is {side_to_move}, but quantik-core calculated {actual_side}",
        )

    legal = fb.legal_masks(boards)[0]
    actual = tuple(int(i) for i in np.flatnonzero(legal))
    if actual != claimed:
        extra = sorted(set(claimed) - set(actual))
        missing = sorted(set(actual) - set(claimed))
        raise ServiceError(
            422,
            "legal_action_indices do not match quantik-core: "
            f"claimed but illegal {extra}, legal but omitted {missing}",
        )

    return _Position(boards=boards, legal=legal, legal_indices=actual)


class PlayService:
    """The models on disk, the opponents they make, and one move at a time."""

    def __init__(
        self,
        models_dir: Path,
        *,
        opening_temperature: float = op.DEFAULT_OPENING_TEMPERATURE,
        opening_plies: int = op.DEFAULT_OPENING_PLIES,
    ) -> None:
        self.models_dir = Path(models_dir)
        # Held on the service rather than read at each `refresh`, so a
        # re-scan after a checkpoint is re-staged cannot quietly hand the
        # roster back its defaults.
        self.opening_temperature = opening_temperature
        self.opening_plies = opening_plies
        # One lock covers agent construction *and* `select`. The agents,
        # the MCTS trees they build and `arena.registry`'s evaluator cache
        # are all shared mutable state, and the server this feeds is a
        # ThreadingHTTPServer, so two phones mid-game are two real threads.
        # Serializing inference is not a performance choice; it is the only
        # thing making the shared state safe.
        self._lock = threading.Lock()
        self._agents: dict[str, _CachedAgent] = {}
        self._models: list[reg.PlayModel] = []
        self._opponents: dict[str, op.Opponent] = {}
        self.refresh()

    def refresh(self) -> None:
        models = reg.scan_models(self.models_dir)
        roster = {
            opponent.opponent_id: opponent
            for opponent in op.roster(
                models,
                temperature=self.opening_temperature,
                temperature_plies=self.opening_plies,
            )
        }
        with self._lock:
            self._models = models
            self._opponents = roster
            self._agents.clear()

    def list_models(self) -> list[dict[str, Any]]:
        """Every discovered model, refused ones included.

        A refused model stays in the list with its reason: dropping it
        makes a staging mistake look like a model that was never trained.
        """
        return [
            {
                "model_id": model.model_id,
                "status": model.status,
                "reason": model.reason,
                "architecture": model.architecture,
                "manifest_model_id": model.manifest_model_id,
                "parameter_count": model.parameter_count,
                "weights_hash": model.weights_hash,
            }
            for model in self._models
        ]

    def list_opponents(self) -> list[dict[str, Any]]:
        return [
            {
                "opponent_id": opponent.opponent_id,
                "label": opponent.label,
                "kind": opponent.kind,
                "model_id": opponent.model_id,
                "simulations": opponent.simulations,
            }
            for opponent in self._opponents.values()
        ]

    def opponent(self, opponent_id: str | None):
        """The `Opponent` for an id, or None — never a raise.

        Used when recording a finished game, where the opponent is
        metadata rather than a subject: a game played against a model that
        has since been unstaged is still a game, and losing it to a 404
        would be worse than recording it with the model details blank.
        """
        if opponent_id is None:
            return None
        return self._opponents.get(opponent_id)

    def choose_move(self, opponent_id: str, request: Any) -> dict[str, Any]:
        qfen, side_to_move, claimed, config = validate_request(request)

        opponent = self._opponents.get(opponent_id)
        if opponent is None:
            raise ServiceError(404, f"unknown opponent {opponent_id!r}")

        position = resolve_position(qfen, side_to_move, claimed)

        seed = config.get("seed")
        if seed is None:
            seed = random.randrange(1 << _SEED_BITS)
        else:
            seed = _require_int(seed, "config.seed")

        board = position.boards[0]
        with self._lock:
            agent = self._agent_for(opponent)
            started = time.perf_counter()
            action = int(agent.select(board, seed))
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            assessment = self._assess(opponent, agent, position)

        if action not in position.legal_indices:
            # Not the caller's fault: the request was verified legal above,
            # so an illegal choice here is a defect in this codebase.
            raise ServiceError(
                500,
                f"{opponent_id} returned action {action}, which quantik-core "
                "considers illegal for this position",
            )

        response: dict[str, Any] = {
            "schema": RESPONSE_SCHEMA,
            "action_index": action,
            "engine_kind": opponent.kind,
            # The opponent id, never a filename or a display label: this is
            # the string a recorded game stores as `p*_engine_version`, and
            # it has to be the same string `runs/eval/*/games.json` uses for
            # the same opponent or the two datasets cannot be pooled.
            "engine_version": opponent.opponent_id,
            "elapsed_ms": elapsed_ms,
        }
        response.update(assessment)
        return response

    def _agent_for(self, opponent: op.Opponent):
        """Build or reuse the agent, refusing weights that changed on disk.

        `arena.registry.load_evaluator` caches on `f"{checkpoint}|{device}"`,
        so retraining into a directory this service already served would
        keep the old weights alive for the life of the process while every
        game recorded against them is labelled with the model id the *new*
        weights carry. Nothing downstream could detect that; the game store
        would simply contain rows attributing one network's play to another.

        So the weights file is stat-checked on every request, and a changed
        file is re-digested. A digest that no longer matches the manifest is
        refused outright — there is no warn-and-continue path, because a
        warning on a server nobody is watching is the same as no check.
        """
        cached = self._agents.get(opponent.opponent_id)
        if opponent.model_id is None:
            if cached is None:
                cached = _CachedAgent(_build(opponent.spec), 0, 0)
                self._agents[opponent.opponent_id] = cached
            return cached.agent

        weights = self.models_dir / opponent.model_id / _WEIGHTS_NAME
        try:
            stat = weights.stat()
        except OSError as exc:
            raise ServiceError(409, f"{opponent.model_id}: weights are unreadable: {exc}") from exc

        if cached is not None and (stat.st_mtime_ns, stat.st_size) == (
            cached.mtime_ns,
            cached.size,
        ):
            return cached.agent

        self._agents.pop(opponent.opponent_id, None)
        model = next((m for m in self._models if m.model_id == opponent.model_id), None)
        expected = model.weights_hash if model else None
        actual = file_digest(weights)
        if actual != expected:
            raise ServiceError(
                409,
                f"{opponent.model_id}: weights changed on disk and now digest "
                f"{actual}, which does not match the manifest's {expected!r}. "
                "Re-stage the model and restart, or call refresh.",
            )

        agent = _build(opponent.spec)
        self._agents[opponent.opponent_id] = _CachedAgent(agent, stat.st_mtime_ns, stat.st_size)
        return agent

    def _assess(self, opponent: op.Opponent, agent, position: _Position) -> dict[str, Any]:
        """The network's own read of the position, for opponents that have one.

        This is the *prior* and the value head, from one extra forward
        pass — not the MCTS visit distribution. Reporting visit counts
        would mean changing `NetMCTSAgent.select` to hand back state it
        currently keeps local, and the arena depends on that class being
        exactly what it is; a field that means "the network's prior" for
        both neural kinds is worth more than one that means two different
        things depending on the opponent. Callers wanting visit counts
        should ask for them explicitly in a later revision.

        `uniform-mcts` is excluded on purpose: it has an evaluator, but a
        flat prior and a constant zero say nothing about the position, and
        an overlay drawing them as if they did would be misleading.
        """
        if opponent.model_id is None:
            return {}
        evaluator = getattr(agent, "evaluator", None)
        if evaluator is None:
            return {}
        priors, values = evaluator(position.boards, position.legal[None, :])
        return {
            "policy": [float(p) for p in priors[0]],
            "value": float(values[0]),
        }


def _build(spec: dict[str, Any]):
    """`arena.registry.build_agent`, imported late and on purpose.

    `arena.registry` imports torch at module scope, and this package's test
    suite runs against a torch-free install. A module-level import here
    would make the whole play service unimportable in that configuration —
    including the classical opponents, which need no network at all.
    """
    from ..arena.registry import build_agent

    return build_agent(spec)
