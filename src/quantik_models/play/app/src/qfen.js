(function exposeQfen(global) {
  "use strict";

  const SHAPES = ["A", "B", "C", "D"];
  const MAX_PER_PLAYER_SHAPE = 2;

  const WIN_LINES = [
    { kind: "row", index: 0, cells: [0, 1, 2, 3] },
    { kind: "row", index: 1, cells: [4, 5, 6, 7] },
    { kind: "row", index: 2, cells: [8, 9, 10, 11] },
    { kind: "row", index: 3, cells: [12, 13, 14, 15] },
    { kind: "column", index: 0, cells: [0, 4, 8, 12] },
    { kind: "column", index: 1, cells: [1, 5, 9, 13] },
    { kind: "column", index: 2, cells: [2, 6, 10, 14] },
    { kind: "column", index: 3, cells: [3, 7, 11, 15] },
    { kind: "zone", index: 0, cells: [0, 1, 4, 5] },
    { kind: "zone", index: 1, cells: [2, 3, 6, 7] },
    { kind: "zone", index: 2, cells: [8, 9, 12, 13] },
    { kind: "zone", index: 3, cells: [10, 11, 14, 15] },
  ];

  function parseQfen(input) {
    const qfen = input.trim();
    const ranks = qfen.split("/");

    if (ranks.length !== 4) {
      throw new Error(`QFEN must contain 4 ranks, got ${ranks.length}.`);
    }

    const cells = ranks.flatMap((rank, row) => {
      const chars = [...rank];

      if (chars.length !== 4) {
        throw new Error(
          `Rank ${row + 1} must contain exactly 4 cells, got ${chars.length}.`,
        );
      }

      return chars.map((char, col) => parseCell(char, row, col));
    });

    return { qfen, cells };
  }

  function qfenFromCells(cells) {
    if (!Array.isArray(cells) || cells.length !== 16) {
      throw new Error("Exactly 16 cells are required to serialize QFEN.");
    }

    return [0, 1, 2, 3]
      .map((row) =>
        cells
          .slice(row * 4, row * 4 + 4)
          .map((cell) => cell.char)
          .join(""),
      )
      .join("/");
  }

  function analyzeBoard(board) {
    const wins = WIN_LINES.filter((line) => hasAllShapes(board.cells, line.cells))
      .map((line) => ({ ...line, cells: [...line.cells] }));

    return { wins };
  }

  function summarizeInventory(board) {
    const players = [createPlayerSummary(), createPlayerSummary()];

    for (const cell of board.cells) {
      if (cell.player === null) {
        continue;
      }

      players[cell.player].used[cell.shape] += 1;
    }

    for (const player of players) {
      for (const shape of SHAPES) {
        player.remaining[shape] = Math.max(
          0,
          MAX_PER_PLAYER_SHAPE - player.used[shape],
        );
        player.overused[shape] = Math.max(
          0,
          player.used[shape] - MAX_PER_PLAYER_SHAPE,
        );
      }
    }

    const pieceCount = board.cells.filter((cell) => cell.player !== null).length;

    return { players, pieceCount };
  }

  function createEmptyQfen() {
    return "..../..../..../....";
  }

  function describeWin(win) {
    if (win.kind === "row") {
      return `Row ${win.index + 1}`;
    }

    if (win.kind === "column") {
      return `Column ${win.index + 1}`;
    }

    return `Zone ${win.index + 1}`;
  }

  function getShapes() {
    return [...SHAPES];
  }

  function parseCell(char, row, col) {
    const index = row * 4 + col;

    if (char === ".") {
      return { char, index, row, col, player: null, shape: null };
    }

    const upper = char.toUpperCase();
    if (!SHAPES.includes(upper) || !/[A-Da-d.]/.test(char)) {
      throw new Error(
        `Invalid QFEN character "${char}" at rank ${row + 1}, file ${col + 1}.`,
      );
    }

    return {
      char,
      index,
      row,
      col,
      player: char === upper ? 0 : 1,
      shape: upper,
    };
  }

  function hasAllShapes(cells, indexes) {
    const shapes = new Set();

    for (const index of indexes) {
      const shape = cells[index].shape;
      if (shape !== null) {
        shapes.add(shape);
      }
    }

    return SHAPES.every((shape) => shapes.has(shape));
  }

  function createPlayerSummary() {
    return {
      used: Object.fromEntries(SHAPES.map((shape) => [shape, 0])),
      remaining: Object.fromEntries(
        SHAPES.map((shape) => [shape, MAX_PER_PLAYER_SHAPE]),
      ),
      overused: Object.fromEntries(SHAPES.map((shape) => [shape, 0])),
    };
  }

  global.QuantikQfen = Object.freeze({
    analyzeBoard,
    createEmptyQfen,
    describeWin,
    getShapes,
    parseQfen,
    qfenFromCells,
    summarizeInventory,
  });
})(globalThis);
