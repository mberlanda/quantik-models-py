(function exposeModes(global) {
  "use strict";

  const MODES = Object.freeze([
    Object.freeze({
      id: "play-a-model",
      label: "Play a model",
      description: "You against a trained model.",
    }),
    Object.freeze({
      id: "watch-two-engines",
      label: "Watch two engines",
      description: "Two engines play each other — step through it or let it autoplay.",
    }),
    Object.freeze({
      id: "two-players",
      label: "Two players",
      description: "Take turns on the same board.",
    }),
    Object.freeze({
      id: "just-the-board",
      label: "Just the board",
      description: "The QFEN and analysis view, with no players assigned for you.",
    }),
  ]);

  const DEFAULT_MODE_ID = "just-the-board";

  // "Just the board" assigns neither player — it is a no-op over whatever
  // controllers are already selected, falling back to the app's own
  // long-standing defaults (human vs. tactical) when nothing is given. An
  // unrecognized mode id resolves here too, which is what makes it "a named
  // default" rather than an arbitrary one: it is a real mode, not a special case.
  function justTheBoard(profile) {
    return {
      player0: profile.player0 || "human",
      player1: profile.player1 || "tactical",
    };
  }

  const ASSIGNMENTS = {
    "play-a-model": () => ({ player0: "human", player1: "service" }),
    "watch-two-engines": () => ({ player0: "tactical", player1: "tactical" }),
    "two-players": () => ({ player0: "human", player1: "human" }),
    "just-the-board": justTheBoard,
  };

  function applyMode(profile = {}, modeId) {
    const resolve = ASSIGNMENTS[modeId] || ASSIGNMENTS[DEFAULT_MODE_ID];
    return resolve(profile);
  }

  global.QuantikModes = Object.freeze({
    MODES,
    DEFAULT_MODE_ID,
    applyMode,
  });
})(globalThis);
