/**
 * Positions worth loading, with the exact solver's answer attached.
 *
 * Every non-empty board here was checked against `quantik-core-rust`'s
 * `exact_oracle` and the solved corpus in `quantik-models-py`: `value` is
 * the game-theoretic result for the side to move under perfect play, and
 * `solution`, where present, is the *only* move that achieves it. Quantik
 * has no draws, so every position is a win or a loss and there is no third
 * answer to hedge towards.
 *
 * The previous set was three boards chosen by hand and never checked. One
 * of them had no legal moves at all; another was three-in-a-row with the
 * fourth cell open, so the "tactic" was to play the obvious move and win.
 * `test/examples.test.js` is what stops that recurring.
 *
 * `value` is written from the side to move's perspective, matching the
 * play service's `value_perspective: "side_to_move"`, so a board loaded
 * here and an evaluation bar drawn from the API are describing the same
 * player.
 */
(function attachExamples(global) {
  "use strict";

  const Qfen = global.QuantikQfen;

  const EXAMPLES = [
    {
      label: "Empty",
      qfen: Qfen.createEmptyQfen(),
      sideToMove: 0,
      blurb: "The starting position. Every one of the 64 moves is legal, and only three of them are actually different — the rest are the same move seen through the board's symmetries.",
    },
    {
      label: "One move wins",
      qfen: "...a/..../..../CdB.",
      sideToMove: 0,
      value: 1,
      // A@10 — shape A into position 10. The solver marks it as the only
      // outcome-optimal action out of 36 legal ones.
      solution: 10,
      blurb: "White to play and win. Thirty-six legal moves; exactly one of them wins, and the other thirty-five lose.",
    },
    {
      label: "Quiet trap",
      qfen: ".C.D/..../.d.a/.d.B",
      sideToMove: 0,
      value: 1,
      // C@7 -> action 2 * 16 + 7.
      solution: 39,
      blurb: "White to play and win, from a position that looks like there is time to develop. There is not: thirty-one legal moves, one win.",
    },
    {
      label: "Already lost",
      qfen: ".c.D/..../bAC./....",
      sideToMove: 1,
      value: -1,
      blurb: "Colour to play, and every one of the twenty-nine legal moves loses. Worth loading against a strong engine to see what a lost position feels like from the inside.",
    },
    {
      label: "Endgame",
      qfen: "CaD./bbA./C.BA/d..d",
      sideToMove: 1,
      value: 1,
      // C@14 -> 2 * 16 + 14.
      solution: 46,
      blurb: "Three legal moves left and only one of them wins. The whole game in a single decision.",
    },
  ];

  global.QuantikExamples = Object.freeze({ EXAMPLES: Object.freeze(EXAMPLES) });
})(typeof globalThis !== "undefined" ? globalThis : this);
