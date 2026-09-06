(function exposeGame(global) {
  "use strict";

  const Qfen = global.QuantikQfen;
  const SHAPES = Qfen.getShapes();
  const EMPTY_QFEN = Qfen.createEmptyQfen();
  const LINES_BY_POSITION = buildLinesByPosition();

  function createGame(qfen = EMPTY_QFEN, { id, startedAt } = {}) {
    const board = Qfen.parseQfen(qfen);
    const counts = countPlayers(board.cells);

    if (counts[0] !== counts[1] && counts[0] !== counts[1] + 1) {
      throw new Error("QFEN does not describe a reachable turn count.");
    }

    const gameStartedAt = startedAt || new Date().toISOString();
    const game = {
      id: id || createGameId(gameStartedAt),
      initialQfen: board.qfen,
      qfen: board.qfen,
      sideToMove: counts[0] === counts[1] ? 0 : 1,
      moves: [],
      startedAt: gameStartedAt,
    };

    return Object.freeze(game);
  }

  function createGameId(startedAt = new Date().toISOString()) {
    const time = String(startedAt).replace(/[^0-9]/g, "").slice(0, 17);
    const random = Math.random().toString(36).slice(2, 10);
    return `browser-${time}-${random}`;
  }

  function getLegalMoves(game) {
    if (getGameStatus(game).phase === "finished") {
      return [];
    }

    return calculateLegalMoves(game);
  }

  function calculateLegalMoves(game) {
    const board = Qfen.parseQfen(game.qfen);
    const inventory = Qfen.summarizeInventory(board).players[game.sideToMove];
    const occupied = new Set(
      board.cells.filter((cell) => cell.player !== null).map((cell) => cell.index),
    );

    return SHAPES.flatMap((shape) => {
      if (inventory.remaining[shape] === 0) {
        return [];
      }

      return board.cells
        .filter((cell) => !occupied.has(cell.index))
        .filter((cell) => isShapeAllowed(board, game.sideToMove, shape, cell.index))
        .map((cell) => ({ player: game.sideToMove, shape, position: cell.index }));
    });
  }

  function applyMove(game, move, actor = "human") {
    const legalMove = getLegalMoves(game).find(
      (candidate) =>
        candidate.player === move.player &&
        candidate.shape === move.shape &&
        candidate.position === move.position,
    );

    if (!legalMove) {
      throw new Error("Move is not legal in the current position.");
    }

    const board = Qfen.parseQfen(game.qfen);
    board.cells[move.position] = {
      ...board.cells[move.position],
      char: move.player === 0 ? move.shape : move.shape.toLowerCase(),
      player: move.player,
      shape: move.shape,
    };
    const resultingQfen = Qfen.qfenFromCells(board.cells);
    const record = Object.freeze({
      ply: game.moves.length,
      sideToMove: game.sideToMove,
      qfen: game.qfen,
      actionIndex: moveToActionIndex(move),
      shape: move.shape,
      position: move.position,
      actor,
      resultingQfen,
    });

    return Object.freeze({
      ...game,
      qfen: resultingQfen,
      sideToMove: 1 - game.sideToMove,
      moves: Object.freeze([...game.moves, record]),
    });
  }

  function getGameStatus(game) {
    const analysis = Qfen.analyzeBoard(Qfen.parseQfen(game.qfen));
    if (analysis.wins.length > 0) {
      return Object.freeze({
        phase: "finished",
        winner: game.moves.length > 0 ? 1 - game.sideToMove : null,
        terminalReason: "line",
      });
    }

    if (calculateLegalMoves(game).length === 0) {
      return Object.freeze({
        phase: "finished",
        winner: 1 - game.sideToMove,
        terminalReason: "no_legal_moves",
      });
    }

    return Object.freeze({ phase: "playing", winner: null, terminalReason: null });
  }

  function moveToActionIndex(move) {
    const shapeIndex = SHAPES.indexOf(move.shape);
    if (shapeIndex < 0 || !Number.isInteger(move.position) || move.position < 0 || move.position > 15) {
      throw new Error("Move must contain shape A-D and position 0-15.");
    }
    return shapeIndex * 16 + move.position;
  }

  function moveFromActionIndex(actionIndex, player) {
    if (!Number.isInteger(actionIndex) || actionIndex < 0 || actionIndex > 63) {
      throw new Error("Action index must be between 0 and 63.");
    }
    return {
      player,
      shape: SHAPES[Math.floor(actionIndex / 16)],
      position: actionIndex % 16,
    };
  }

  function isShapeAllowed(board, player, shape, position) {
    const opponent = 1 - player;
    return LINES_BY_POSITION[position].every((line) =>
      line.every((index) => {
        const cell = board.cells[index];
        return cell.player !== opponent || cell.shape !== shape;
      }),
    );
  }

  function countPlayers(cells) {
    return cells.reduce(
      (counts, cell) => {
        if (cell.player !== null) counts[cell.player] += 1;
        return counts;
      },
      [0, 0],
    );
  }

  function buildLinesByPosition() {
    return Array.from({ length: 16 }, (_, position) => {
      const row = Math.floor(position / 4);
      const col = position % 4;
      const zoneRow = Math.floor(row / 2) * 2;
      const zoneCol = Math.floor(col / 2) * 2;
      return [
        [0, 1, 2, 3].map((offset) => row * 4 + offset),
        [0, 1, 2, 3].map((offset) => offset * 4 + col),
        [0, 1, 4, 5].map((offset) => zoneRow * 4 + zoneCol + offset),
      ];
    });
  }

  global.QuantikGame = Object.freeze({
    applyMove,
    createGame,
    createGameId,
    getGameStatus,
    getLegalMoves,
    moveFromActionIndex,
    moveToActionIndex,
  });
})(globalThis);
