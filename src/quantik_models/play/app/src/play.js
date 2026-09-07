(function exposePlay(global) {
  "use strict";

  const Game = global.QuantikGame;
  const Engines = global.QuantikEngines;

  const RECORD_SCHEMA = "game-result.v1";

  function requestUrl(baseUrl, path) {
    return `${String(baseUrl || "").replace(/\/+$/, "")}${path}`;
  }

  async function readJson(response, what) {
    if (!response.ok) {
      throw new Error(`${what} failed with HTTP ${response.status}.`);
    }
    return response.json();
  }

  async function fetchOpponents({ baseUrl, fetch } = {}) {
    const fetchImplementation = fetch || global.fetch;
    const response = await fetchImplementation(requestUrl(baseUrl, "/api/opponents"), {
      headers: { Accept: "application/json" },
    });
    const payload = await readJson(response, "Loading opponents");
    return (payload.opponents || []).map((opponent) => ({
      id: opponent.opponent_id,
      label: opponent.label,
      kind: opponent.kind,
      modelId: opponent.model_id ?? null,
      simulations: opponent.simulations ?? null,
    }));
  }

  function createOpponentEngine(opponent, { baseUrl, fetch } = {}) {
    // The opponent rides in the URL because createRemoteEngine builds its
    // POST body from a fixed literal with no hook for extra fields. That
    // is why this wraps it rather than changing it: engines.js keeps its
    // contract test, and its request stays exactly the
    // quantik.engine-request.v1 the Rust gateway also accepts.
    const url = requestUrl(baseUrl, `/api/move/${encodeURIComponent(opponent.id)}`);
    return Engines.createRemoteEngine(url, {
      fetch,
      label: opponent.label,
      // Stored as p*_engine_version, so it must be the opponent id and not
      // a display label: a human game and an arena game have to name the
      // same opponent with the same string to be poolable.
      version: opponent.id,
    });
  }

  /**
   * What one opponent's network makes of the position on the board.
   *
   * Returns `null` for a finished game without asking: the service refuses
   * a terminal position with a 422 — correctly, there is no side to move
   * whose prospects mean anything — and asking anyway would turn the end
   * of every game into an error message.
   *
   * The service's `value` and `win_probability` are from the **side to
   * move's** perspective, which it states in `value_perspective`. Both
   * readings are returned: `moverWinProbability` as sent, and
   * `player0WinProbability` converted, because a bar that flips its
   * meaning every ply is unreadable and a bar drawn from the raw number
   * shows the wrong player winning on every odd one.
   */
  async function analysePosition(game, opponentId, { baseUrl, fetch } = {}) {
    if (!game || !opponentId) return null;
    if (Game.getGameStatus(game).phase !== "playing") return null;

    const fetchImplementation = fetch || global.fetch;
    const url = requestUrl(baseUrl, `/api/analyse/${encodeURIComponent(opponentId)}`);
    const response = await fetchImplementation(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        schema: "quantik.engine-request.v1",
        qfen: game.qfen,
        side_to_move: game.sideToMove,
        legal_action_indices: Game.getLegalMoves(game).map((move) =>
          Game.moveToActionIndex(move),
        ),
      }),
    });
    const payload = await readJson(response, "Analysing the position");

    const mover = payload.win_probability ?? null;
    return {
      opponentId: payload.opponent_id,
      sideToMove: payload.side_to_move,
      value: payload.value ?? null,
      moverWinProbability: mover,
      player0WinProbability:
        mover === null ? null : payload.side_to_move === 0 ? mover : 1 - mover,
      topMoves: (payload.top_moves || []).map((move) => ({
        actionIndex: move.action_index,
        shape: move.shape,
        position: move.position,
        prior: move.prior,
      })),
    };
  }

  function createFinishWatcher() {
    let recorded = null;

    return {
      /** True exactly once, when this game first reaches a finished state. */
      check(game) {
        if (!game || Game.getGameStatus(game).phase !== "finished") return false;
        if (recorded === game.id) return false;
        recorded = game.id;
        return true;
      },
      /** A new game: whatever was recorded before is no longer this one. */
      reset() {
        recorded = null;
      },
      /** An imported finished game — already played, not played here. */
      adopt(game) {
        recorded = game?.id ?? null;
      },
    };
  }

  /** The seat the resolved opponent sits in.
   *
   * With a human at the board it is simply the other seat. With no human
   * it cannot be inferred — both seats are engines — so the caller has to
   * say which one it resolved, and `null` means it resolved none.
   */
  function opponentSeatFor({ humanSeat, opponentSeat }) {
    if (opponentSeat === 0 || opponentSeat === 1) return opponentSeat;
    if (humanSeat === 0) return 1;
    if (humanSeat === 1) return 0;
    return null;
  }

  function seatFor(player, { humanSeat, playerName, opponent, opponentSeat, engines }) {
    if (humanSeat === player) {
      return { kind: "human", version: playerName || "anonymous" };
    }
    // Only the seat the opponent was resolved for. Applying it to every
    // non-human seat is what made an engine-vs-engine game record the same
    // model on both sides.
    if (opponent && opponentSeat === player) {
      return { kind: opponent.kind || "remote", version: opponent.id };
    }
    const engine = (engines || [])[player] || {};
    return { kind: engine.kind || "unknown", version: engine.version || "unknown" };
  }

  function buildGameRecord(game, options = {}) {
    const status = Game.getGameStatus(game);
    if (status.phase !== "finished") {
      throw new Error("Only a finished game can be recorded.");
    }

    const humanSeat = options.humanSeat ?? null;
    const opponentSeat = opponentSeatFor({ ...options, humanSeat });
    const p0 = seatFor(0, { ...options, humanSeat, opponentSeat });
    const p1 = seatFor(1, { ...options, humanSeat, opponentSeat });

    return {
      schema: RECORD_SCHEMA,
      game_id: game.id,
      started_at: game.startedAt,
      initial_qfen: game.initialQfen,
      move_action_indices: game.moves.map((record) => record.actionIndex),
      p0_engine_kind: p0.kind,
      p0_engine_version: p0.version,
      p1_engine_kind: p1.kind,
      p1_engine_version: p1.version,
      human_seat: humanSeat,
      player_name: options.playerName || null,
      opponent_id: options.opponent ? options.opponent.id : null,
      // The server reads this into `game_meta.opponent_seat`; it was never
      // sent, so that column was NULL on every row.
      opponent_seat: options.opponent ? opponentSeat : null,
      // Sent, but not authoritative: the service replays the moves and
      // derives its own outcome. A disagreement is the one routine check
      // these rules get against quantik-core's.
      winner: status.winner,
      plies: game.moves.length,
      terminal_reason: status.terminalReason,
    };
  }

  /**
   * What GET /api says about this server, read defensively: anything short
   * of an explicit "false" is read as recording being on. That covers an
   * older server that predates this field (decisions.md#D6 — absent means
   * on) and, deliberately going further, a server that cannot be reached
   * at all: a network failure is not evidence of a storeless server, only
   * evidence that this one request failed. The real POST attempt in
   * recordGame is what eventually finds out, not a capability probe.
   */
  async function fetchCapabilities({ baseUrl, fetch } = {}) {
    const fetchImplementation = fetch || global.fetch;
    try {
      const response = await fetchImplementation(requestUrl(baseUrl, "/api"), {
        headers: { Accept: "application/json" },
      });
      if (!response.ok) return { recording: true };
      const payload = await response.json();
      return { recording: payload.recording === undefined ? true : Boolean(payload.recording) };
    } catch {
      return { recording: true };
    }
  }

  async function recordGame(body, { baseUrl, fetch, recording = true } = {}) {
    if (!recording) {
      // The server said so itself: nothing to post, and nothing wrong
      // either. The caller shows this as a plain fact, not an error.
      return { recorded: false, skipped: true };
    }
    const fetchImplementation = fetch || global.fetch;
    const response = await fetchImplementation(requestUrl(baseUrl, "/api/games"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return readJson(response, "Recording the game");
  }

  global.QuantikPlay = Object.freeze({
    fetchOpponents,
    fetchCapabilities,
    createOpponentEngine,
    createFinishWatcher,
    buildGameRecord,
    recordGame,
    analysePosition,
  });
})(globalThis);
