(function exposeTrace(global) {
  "use strict";

  const Game = global.QuantikGame;

  function createTrace(game, players = []) {
    const status = Game.getGameStatus(game);
    let replay = Game.createGame(game.initialQfen);
    const positions = game.moves.map((record) => {
      const legalActionIndices = Game.getLegalMoves(replay).map(Game.moveToActionIndex);
      const position = {
        ply: record.ply,
        qfen: record.qfen,
        side_to_move: record.sideToMove,
        legal_action_indices: legalActionIndices,
        action_index: record.actionIndex,
        actor: record.actor,
      };
      replay = Game.applyMove(
        replay,
        Game.moveFromActionIndex(record.actionIndex, record.sideToMove),
        record.actor,
      );
      return position;
    });

    return {
      schema: "quantik.game-trace.v1",
      action_contract: "action-index.v1",
      game_id: game.id || Game.createGameId(game.startedAt),
      started_at: game.startedAt,
      exported_at: new Date().toISOString(),
      players: [0, 1].map((player) => ({
        player,
        kind: players[player]?.kind || "unknown",
        version: players[player]?.version || "unknown",
      })),
      initial_qfen: game.initialQfen,
      positions,
      final_qfen: game.qfen,
      plies: game.moves.length,
      winner: status.winner,
      terminal_reason: status.terminalReason,
    };
  }

  function parseTrace(input) {
    const trace = typeof input === "string" ? JSON.parse(input) : input;
    if (!trace || trace.schema !== "quantik.game-trace.v1" || !Array.isArray(trace.positions)) {
      throw new Error("Unsupported game trace.");
    }

    let game = Game.createGame(trace.initial_qfen, {
      id: trace.game_id,
      startedAt: trace.started_at,
    });
    try {
      for (const position of trace.positions) {
        if (position.qfen !== game.qfen || position.side_to_move !== game.sideToMove) {
          throw new Error("position does not match replay state");
        }
        game = Game.applyMove(
          game,
          Game.moveFromActionIndex(position.action_index, game.sideToMove),
          position.actor || "imported",
        );
      }
    } catch (error) {
      throw new Error(`Invalid trace move: ${error.message}`);
    }

    if (trace.final_qfen !== game.qfen) {
      throw new Error("Invalid trace: final QFEN does not match replay.");
    }
    return game;
  }

  global.QuantikTrace = Object.freeze({ createTrace, parseTrace });
})(globalThis);
