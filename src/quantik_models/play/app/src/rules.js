(function exposeRules(global) {
  "use strict";

  const STORAGE_KEY = "quantik-how-to-play-open";

  // Checked against quantik-core-contracts' docs/game-state.md (the
  // placement restriction, stated there as "no row/column/region holds the
  // same shape from both players" → ILLEGAL_PLACEMENT) and, since that file
  // doesn't spell out the win condition in "four different shapes" terms,
  // against quantik-core-rust's game.rs ("any win line contains all 4
  // distinct shapes (regardless of color)") and quantik-core-py's
  // game_utils.py ("a winning line ... with all four [shapes] ... Colors
  // don't matter for winning"). All three agree; this is the same rule
  // stated in the same terms, not a second rule.
  const RULES = Object.freeze({
    title: "How to play",
    points: Object.freeze([
      "Place a shape on any empty square.",
      "You may not place a shape in a row, column, or 2×2 zone where your " +
        "opponent already has that shape. Your own is fine.",
      "You win by completing a row, column, or zone with four different " +
        "shapes, in either colour — whoever places the fourth wins, " +
        "regardless of who owns the others.",
    ]),
  });

  // Unlike layout.js's drawer (an advanced/debug panel, safe to default
  // closed), this fails toward *open*: no stored value means a first
  // visit, and a value that can't be read is treated the same way, because
  // hiding the rules from someone who might need them is the worse failure
  // mode than showing them again to someone who already knows them.
  function readHowToPlayOpen(storage = global.localStorage) {
    if (!storage) {
      return true;
    }

    try {
      const raw = storage.getItem(STORAGE_KEY);
      return raw === null ? true : JSON.parse(raw) === true;
    } catch {
      return true;
    }
  }

  function writeHowToPlayOpen(open, storage = global.localStorage) {
    if (!storage) {
      return;
    }

    try {
      storage.setItem(STORAGE_KEY, JSON.stringify(Boolean(open)));
    } catch {
      // Some browsers restrict localStorage on file-opened pages.
    }
  }

  global.QuantikRules = Object.freeze({
    RULES,
    readHowToPlayOpen,
    writeHowToPlayOpen,
  });
})(globalThis);
