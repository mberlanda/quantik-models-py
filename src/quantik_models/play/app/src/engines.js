(function exposeEngines(global) {
  "use strict";

  const Game = global.QuantikGame;

  function createLocalEngine(kind, options = {}) {
    if (!['random', 'tactical'].includes(kind)) {
      throw new Error(`Unknown local engine "${kind}".`);
    }

    const random = mulberry32(normalizeSeed(options.seed));
    return Object.freeze({
      id: `local:${kind}`,
      label: kind === "random" ? "Random" : "Tactical",
      kind,
      version: "browser-v1",
      async chooseMove(game) {
        const legalMoves = Game.getLegalMoves(game);
        if (legalMoves.length === 0) return null;

        if (kind === "tactical") {
          const winning = legalMoves.find((move) =>
            Game.getGameStatus(Game.applyMove(game, move, `engine:${kind}`)).phase === "finished",
          );
          if (winning) return winning;
        }

        return legalMoves[Math.floor(random() * legalMoves.length)];
      },
    });
  }

  function createRemoteEngine(endpoint, options = {}) {
    const url = String(endpoint || "").trim();
    if (!url) throw new Error("Remote engine endpoint is required.");
    const fetchImplementation = options.fetch || global.fetch;
    if (typeof fetchImplementation !== "function") {
      throw new Error("This browser does not provide fetch().");
    }

    return Object.freeze({
      id: `remote:${url}`,
      label: options.label || "Remote engine",
      kind: "remote",
      version: options.version || "unknown",
      async chooseMove(game) {
        const legalMoves = Game.getLegalMoves(game);
        if (legalMoves.length === 0) return null;
        const legalActionIndices = legalMoves.map(Game.moveToActionIndex);
        const response = await fetchImplementation(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            schema: "quantik.engine-request.v1",
            qfen: game.qfen,
            side_to_move: game.sideToMove,
            legal_action_indices: legalActionIndices,
          }),
        });
        if (!response.ok) {
          throw new Error(`Engine request failed with HTTP ${response.status}.`);
        }
        const payload = await response.json();
        const actionIndex = payload.action_index;
        if (!legalActionIndices.includes(actionIndex)) {
          throw new Error(`Engine returned illegal action index ${actionIndex}.`);
        }
        return Game.moveFromActionIndex(actionIndex, game.sideToMove);
      },
    });
  }

  function normalizeSeed(value) {
    const number = Number(value);
    return Number.isFinite(number) ? number >>> 0 : Date.now() >>> 0;
  }

  function mulberry32(seed) {
    let state = seed;
    return function random() {
      state |= 0;
      state = (state + 0x6d2b79f5) | 0;
      let result = Math.imul(state ^ (state >>> 15), 1 | state);
      result = (result + Math.imul(result ^ (result >>> 7), 61 | result)) ^ result;
      return ((result ^ (result >>> 14)) >>> 0) / 4294967296;
    };
  }

  global.QuantikEngines = Object.freeze({ createLocalEngine, createRemoteEngine });
})(globalThis);
