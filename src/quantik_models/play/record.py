"""Turn a browser's finished game into a row that can be trusted.

The client sends the moves it played and its own reading of how the game
ended. The moves are checked, and the reading is not believed: the outcome
stored in `games` is derived here by replaying the moves through
`quantik-core`'s rules, and what the client claimed is kept beside it in
`game_meta.client_*`.

That split is the point of the module. A browser that has miscounted a win
would otherwise write a wrong result into the same table the arena writes
to, and nothing downstream could tell the two apart. Recording both makes
a disagreement a *measurement* — the only signal in this system that the
JavaScript rules and the Python rules have drifted — instead of a silently
corrupted row.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

from ..env import fastboard as fb
from .service import ServiceError

SCHEMA = "game-result.v1"

# The visualizer says "line" where the contract says "win_condition"
# (`quantik-qfen-visualizer/src/game.js` against
# `quantik-core-contracts/docs/game-result-v1.md`). Mapped here rather than
# left to a comment, because `games.terminal_reason` carries a SQL CHECK
# and an unmapped value fails the insert instead of the request.
_CLIENT_TERMINAL_REASONS = {
    "line": "win_condition",
    "win_condition": "win_condition",
    "no_legal_moves": "no_legal_moves",
    "stuck": "no_legal_moves",
}


@dataclass(frozen=True)
class Replay:
    """What the moves actually say, independent of what the client claimed."""

    winner: int
    plies: int
    terminal_reason: str
    initial_position_key: str
    final_qfen: str
    positions: list[dict[str, Any]]


def _canonical_key(boards: np.ndarray) -> str:
    return str(int(fb.canonical_keys(boards)[0]))


def replay(initial_qfen: str, actions: list[int]) -> Replay:
    """Replay the game, deriving the outcome rather than accepting one.

    Raises `ServiceError(422)` for anything that makes the move list
    untrustworthy: an illegal action, a move played after the game was
    already over, or a sequence that ends with play still possible. In all
    three the moves themselves cannot be relied on, so there is nothing
    worth storing — unlike a disagreement about the *label* on a legal
    game, which is recorded rather than refused.
    """
    if not actions:
        raise ServiceError(422, "a game with no moves has no outcome to record")

    try:
        boards = fb.from_qfen(initial_qfen)
    except ValueError as exc:
        raise ServiceError(422, f"initial position does not parse: {exc}") from exc

    initial_key = _canonical_key(boards)
    positions = [{"ply": 0, "qfen": initial_qfen, "canonical_key": initial_key}]

    done, _ = fb.terminal_status(boards)
    if bool(done[0]):
        raise ServiceError(422, "the initial position is already terminal")

    winner = -1
    for ply, action in enumerate(actions, start=1):
        if not isinstance(action, int) or isinstance(action, bool):
            raise ServiceError(422, f"move {ply} is not an integer: {action!r}")
        if not 0 <= action < fb.ACTION_COUNT:
            raise ServiceError(422, f"move {ply} is out of range: {action}")
        if not bool(fb.legal_masks(boards)[0][action]):
            raise ServiceError(
                422, f"move {ply} plays action {action}, which is illegal in {fb.to_qfen(boards[0])}"
            )

        # Both terminal conditions are losses for the side to move, so the
        # player who made the final move is the winner. Captured before the
        # move, because after it `side_to_move` names the loser.
        winner = int(fb.side_to_move(boards)[0])
        boards = fb.apply_actions(boards, np.array([action], dtype=np.int64))

        qfen = fb.to_qfen(boards[0])
        positions.append({"ply": ply, "qfen": qfen, "canonical_key": _canonical_key(boards)})

        done, _ = fb.terminal_status(boards)
        if bool(done[0]) and ply != len(actions):
            raise ServiceError(
                422, f"the game ended at move {ply} but {len(actions)} moves were submitted"
            )

    if not bool(done[0]):
        raise ServiceError(422, "the final position is not terminal; this game is unfinished")

    reason = "win_condition" if bool(fb.has_winning_line(boards)[0]) else "no_legal_moves"
    return Replay(
        winner=winner,
        plies=len(actions),
        terminal_reason=reason,
        initial_position_key=initial_key,
        final_qfen=fb.to_qfen(boards[0]),
        positions=positions,
    )


def _client_reason(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return _CLIENT_TERMINAL_REASONS.get(value, value)


def discrepancies(result: Replay, payload: dict[str, Any]) -> list[str]:
    """Where the client's reading and the replay disagree, in words.

    Empty is the normal case and the one worth stating: it is positive
    evidence that the two rule implementations agree on this game, which is
    the only routine check either of them gets.
    """
    out: list[str] = []
    claimed_winner = payload.get("winner")
    if claimed_winner is not None and claimed_winner != result.winner:
        out.append(f"winner: client says {claimed_winner}, replay says {result.winner}")

    claimed_plies = payload.get("plies")
    if claimed_plies is not None and claimed_plies != result.plies:
        out.append(f"plies: client says {claimed_plies}, replay says {result.plies}")

    claimed_reason = _client_reason(payload.get("terminal_reason"))
    if claimed_reason is not None and claimed_reason != result.terminal_reason:
        out.append(
            f"terminal_reason: client says {claimed_reason}, replay says {result.terminal_reason}"
        )
    return out


def rows_for(
    payload: dict[str, Any],
    result: Replay,
    *,
    contract_version: str,
    service_version: str,
    opponent: Any = None,
    weights_hash: str | None = None,
    manifest_model_id: str | None = None,
    architecture: str | None = None,
    client_user_agent: str | None = None,
    recorded_at: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """The `games`, `game_meta` and `game_positions` dicts `record_game` wants.

    The `games` row carries the replayed outcome only. The client's own
    claims go to `game_meta.client_*`, where a later query can count how
    often the two disagreed without any of those claims ever having been
    treated as fact.
    """
    game = {
        "schema": SCHEMA,
        "contract_version": contract_version,
        "game_id": payload["game_id"],
        "started_at": payload["started_at"],
        "p0_engine_kind": payload["p0_engine_kind"],
        "p0_engine_version": payload["p0_engine_version"],
        "p1_engine_kind": payload["p1_engine_kind"],
        "p1_engine_version": payload["p1_engine_version"],
        "initial_position_key": result.initial_position_key,
        "winner": result.winner,
        "plies": result.plies,
        "terminal_reason": result.terminal_reason,
        "move_action_indices": [int(a) for a in payload["move_action_indices"]],
        "run_id": payload.get("run_id"),
    }
    meta = {
        "recorded_at": recorded_at or datetime.now(timezone.utc).isoformat(),
        "human_seat": payload.get("human_seat"),
        "player_name": payload.get("player_name"),
        "opponent_seat": payload.get("opponent_seat"),
        "opponent_id": getattr(opponent, "opponent_id", None),
        "model_id": getattr(opponent, "model_id", None),
        "simulations": getattr(opponent, "simulations", None),
        "agent_kind": getattr(opponent, "kind", None),
        "weights_hash": weights_hash,
        "manifest_model_id": manifest_model_id,
        "architecture": architecture,
        "service_version": service_version,
        "client_user_agent": client_user_agent,
        "initial_qfen": payload["initial_qfen"],
        "final_qfen": result.final_qfen,
        "client_winner": payload.get("winner"),
        "client_terminal_reason": _client_reason(payload.get("terminal_reason")),
        # A classical opponent has neither, and a `None` here is the right
        # answer for it: `minimax-d2` has no temperature to record.
        "opening_temperature": getattr(opponent, "temperature", None),
        "opening_plies": getattr(opponent, "temperature_plies", None),
    }
    return game, meta, result.positions


REQUIRED_FIELDS = (
    "game_id",
    "started_at",
    "initial_qfen",
    "move_action_indices",
    "p0_engine_kind",
    "p0_engine_version",
    "p1_engine_kind",
    "p1_engine_version",
)


def validate_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ServiceError(400, "request body must be a JSON object")
    schema = payload.get("schema")
    if schema != SCHEMA:
        raise ServiceError(400, f"schema must be {SCHEMA!r}, got {schema!r}")
    missing = [field for field in REQUIRED_FIELDS if payload.get(field) in (None, "")]
    if missing:
        raise ServiceError(400, f"missing required fields: {', '.join(missing)}")
    if not isinstance(payload["move_action_indices"], list):
        raise ServiceError(400, "move_action_indices must be a list")
    # Proves the list survives the round trip the store will put it
    # through, so a value that cannot be serialised fails as a 400 here
    # rather than as an IntegrityError inside the transaction.
    json.dumps(payload["move_action_indices"])
    return payload
