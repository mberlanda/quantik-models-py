(function startApp(global) {
  "use strict";

  const Qfen = global.QuantikQfen;
  const Settings = global.QuantikSettings;
  const Game = global.QuantikGame;
  const Engines = global.QuantikEngines;
  const Play = global.QuantikPlay;
  const Examples = global.QuantikExamples;
  const Trace = global.QuantikTrace;
  const Layout = global.QuantikLayout;
  const Modes = global.QuantikModes;
  const Rules = global.QuantikRules;
  const SHAPES = Qfen.getShapes();
  const SHAPE_META = {
    A: { label: "Cone", className: "piece-cone" },
    B: { label: "Cylinder", className: "piece-cylinder" },
    C: { label: "Cube", className: "piece-cube" },
    D: { label: "Sphere", className: "piece-sphere" },
  };
  const PLAYER_NAMES = ["White / uppercase", "Color / lowercase"];

  const elements = {};
  let game;
  let selectedShape = "A";
  let theme = Settings.readTheme();
  // Empty means no mode has been chosen yet — the chooser shows a neutral
  // prompt and the controller selects keep whatever index.html already has
  // them set to (decisions.md#D7).
  let selectedMode = Settings.readProfile().mode;
  let engines = [];
  let opponents = [];
  // Optimistic until GET /api answers otherwise — decisions.md#D6 and its
  // extension in fetchCapabilities: an older server, an HTTP error, and an
  // unreachable one all mean "assume it records," so the first game of a
  // session is never wrongly announced as unsaved before the real answer
  // arrives.
  let capabilities = { recording: true };
  // The request in flight, so a fast sequence of moves cannot let an
  // earlier answer land after a later one and describe the wrong board.
  let analysisToken = 0;
  const finishWatcher = Play.createFinishWatcher();
  let autoplayTimer = null;
  let engineBusy = false;

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  function init() {
    for (const id of [
      "advanced-drawer", "autoplay-button", "board-grid", "copy-button", "error-message", "examples",
      "export-button", "game-message", "how-to-play", "how-to-play-points", "how-to-play-summary",
      "import-button", "import-file", "inventory",
      "legend", "lowercase-color", "mode-chooser", "move-history", "new-game-button", "piece-count",
      "player-0-controller", "player-1-controller", "ply-count", "qfen-input",
      "opponent-0", "opponent-1", "player-name", "service-base",
      "remote-endpoint-0", "remote-endpoint-1", "reset-button", "reset-colors-button", "seed-input",
      "shape-picker", "speed-select", "status-pill", "step-button", "turn-label",
      "analysis-opponent", "assessment-source", "evaluation-bar", "evaluation-fill",
      "evaluation-caption", "top-moves",
      "undo-button", "uppercase-color", "win-count", "win-list",
    ]) {
      elements[toCamel(id)] = document.querySelector(`#${id}`);
    }

    applyTheme(theme);
    initializeMode();
    renderExamples();
    renderLegend();
    renderShapePicker();
    renderRules();
    bindEvents();
    elements.advancedDrawer.open = Layout.readDrawerOpen();
    elements.advancedDrawer.addEventListener("toggle", () => {
      Layout.writeDrawerOpen(elements.advancedDrawer.open);
    });
    elements.howToPlay.open = Rules.readHowToPlayOpen();
    elements.howToPlay.addEventListener("toggle", () => {
      Rules.writeHowToPlayOpen(elements.howToPlay.open);
    });
    startGame(elements.qfenInput.value.trim());
  }

  function bindEvents() {
    elements.qfenInput.addEventListener("input", loadEditedQfen);
    elements.uppercaseColor.addEventListener("input", updateThemeFromControls);
    elements.lowercaseColor.addEventListener("input", updateThemeFromControls);
    elements.resetColorsButton.addEventListener("click", resetTheme);
    elements.resetButton.addEventListener("click", () => startGame(Qfen.createEmptyQfen()));
    elements.newGameButton.addEventListener("click", () => startGame(Qfen.createEmptyQfen()));
    elements.copyButton.addEventListener("click", copyQfen);
    elements.stepButton.addEventListener("click", stepEngine);
    elements.autoplayButton.addEventListener("click", toggleAutoplay);
    elements.analysisOpponent.addEventListener("change", () => {
      saveProfile();
      refreshAssessment();
    });
    elements.undoButton.addEventListener("click", undoMove);
    elements.exportButton.addEventListener("click", exportTrace);
    elements.importButton.addEventListener("click", () => elements.importFile.click());
    elements.importFile.addEventListener("change", importTrace);
    elements.player0Controller.addEventListener("change", configureEngines);
    elements.player1Controller.addEventListener("change", configureEngines);
    elements.seedInput.addEventListener("change", configureEngines);
    elements.remoteEndpoint0.addEventListener("change", configureEngines);
    elements.remoteEndpoint1.addEventListener("change", configureEngines);
    elements.opponent0.addEventListener("change", configureEngines);
    elements.opponent1.addEventListener("change", configureEngines);
    elements.serviceBase.addEventListener("change", () => {
      saveProfile();
      loadOpponents();
    });
    elements.playerName.addEventListener("change", saveProfile);

    const profile = Settings.readProfile();
    elements.playerName.value = profile.playerName || "";
    elements.serviceBase.value = profile.serviceBase || "";
    loadOpponents();
    loadCapabilities();
  }

  async function loadCapabilities() {
    // fetchCapabilities never rejects — it already turns every failure into
    // the safe "assume it records" default — so there is nothing to catch
    // here beyond that.
    capabilities = await Play.fetchCapabilities({ baseUrl: serviceBase() });
  }

  function saveProfile() {
    Settings.writeProfile({
      playerName: elements.playerName.value,
      analysisOpponent: elements.analysisOpponent.value,
      serviceBase: elements.serviceBase.value,
      mode: selectedMode,
    });
  }

  // Applies a persisted mode's controller assignment before the first
  // render, then draws the chooser. Nothing is forced onto the selects when
  // no mode is stored — index.html's own defaults (human / tactical) stand,
  // exactly as they did before the chooser existed.
  function initializeMode() {
    if (selectedMode) {
      const assignment = Modes.applyMode(currentControllers(), selectedMode);
      elements.player0Controller.value = assignment.player0;
      elements.player1Controller.value = assignment.player1;
    }
    renderModeChooser();
  }

  function currentControllers() {
    return { player0: elements.player0Controller.value, player1: elements.player1Controller.value };
  }

  function renderModeChooser() {
    elements.modeChooser.replaceChildren();
    for (const mode of Modes.MODES) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "mode-button";
      button.dataset.selected = String(mode.id === selectedMode);
      const label = document.createElement("strong");
      label.textContent = mode.label;
      const description = document.createElement("span");
      description.textContent = mode.description;
      button.append(label, description);
      button.addEventListener("click", () => selectMode(mode.id));
      elements.modeChooser.append(button);
    }
  }

  function selectMode(modeId) {
    selectedMode = modeId;
    const assignment = Modes.applyMode(currentControllers(), modeId);
    elements.player0Controller.value = assignment.player0;
    elements.player1Controller.value = assignment.player1;
    configureEngines();
    saveProfile();
    renderModeChooser();
  }

  function serviceBase() {
    // Empty is the default and means "wherever this page came from" — a
    // relative path. That is the normal case: the service serves the app,
    // so a phone that loaded the page reaches the API at the same address
    // with nobody typing anything. The field exists to override that, for
    // the case where the page is opened from a file or another host.
    return elements.serviceBase.value.trim().replace(/\/+$/, "");
  }

  async function loadOpponents() {
    for (const select of [elements.opponent0, elements.opponent1]) {
      select.innerHTML = "";
    }
    try {
      opponents = await Play.fetchOpponents({ baseUrl: serviceBase() });
    } catch (error) {
      opponents = [];
      elements.gameMessage.textContent = `No play service: ${error.message}`;
      renderAnalysisOpponents();
      return;
    }
    renderAnalysisOpponents();
    for (const select of [elements.opponent0, elements.opponent1]) {
      for (const opponent of opponents) {
        const option = document.createElement("option");
        option.value = opponent.id;
        option.textContent = opponent.label;
        select.appendChild(option);
      }
    }
    configureEngines();
  }

  function startGame(qfen) {
    stopAutoplay();
    try {
      game = Game.createGame(qfen);
      finishWatcher.reset();
      elements.qfenInput.value = game.qfen;
      elements.errorMessage.textContent = "";
      elements.qfenInput.removeAttribute("aria-invalid");
      configureEngines();
      render();
      maybeRunEngine();
    } catch (error) {
      showError(error);
    }
  }

  function loadEditedQfen() {
    try {
      game = Game.createGame(elements.qfenInput.value);
      finishWatcher.reset();
      stopAutoplay();
      configureEngines();
      elements.errorMessage.textContent = "";
      elements.qfenInput.removeAttribute("aria-invalid");
      render();
    } catch (error) {
      showError(error);
    }
  }

  function configureEngines() {
    const seed = Number(elements.seedInput.value);
    engines = [elements.player0Controller.value, elements.player1Controller.value]
      .map((controller, player) => createController(controller, seed + player, player));
    if (game) render();
  }

  function createController(controller, seed, player) {
    if (controller === "human") {
      return { id: "human", label: "Human", kind: "human", version: "browser-v1" };
    }
    if (controller === "service") {
      const chosen = opponents.find((o) => o.id === elements[`opponent${player}`].value);
      return chosen
        ? Play.createOpponentEngine(chosen, { baseUrl: serviceBase() })
        : { id: "service:missing", label: "Model (none loaded)", kind: "remote", version: "unknown" };
    }
    if (controller === "remote") {
      const endpoint = elements[`remoteEndpoint${player}`].value.trim();
      return endpoint
        ? Engines.createRemoteEngine(endpoint)
        : { id: "remote:missing", label: "Remote (endpoint required)", kind: "remote", version: "unknown" };
    }
    return Engines.createLocalEngine(controller, { seed });
  }

  function render() {
    renderEverything();
    refreshAssessment();
    // After rendering, not before: recording is a side effect of the game
    // having ended, and the board should be on screen either way.
    if (finishWatcher.check(game)) recordFinishedGame();
  }

  function renderEverything() {
    const board = Qfen.parseQfen(game.qfen);
    const analysis = Qfen.analyzeBoard(board);
    const summary = Qfen.summarizeInventory(board);
    const status = Game.getGameStatus(game);
    const legalMoves = Game.getLegalMoves(game);

    elements.qfenInput.value = game.qfen;
    renderBoard(board, analysis, legalMoves);
    renderWins(analysis);
    renderInventory(summary);
    renderHistory();
    renderTurn(status);
    renderShapePicker(legalMoves, summary);
    elements.pieceCount.textContent = `${summary.pieceCount} ${plural(summary.pieceCount, "piece")}`;
    elements.plyCount.textContent = `${game.moves.length} ${plural(game.moves.length, "ply", "plies")}`;
    setStatus(status.phase === "finished" ? "Finished" : "Playing");
  }

  function renderBoard(board, analysis, legalMoves) {
    const winningCells = new Set(analysis.wins.flatMap((win) => win.cells));
    const playable = new Set(
      legalMoves.filter((move) => move.shape === selectedShape).map((move) => move.position),
    );
    const humanTurn = engines[game.sideToMove]?.kind === "human";
    elements.boardGrid.replaceChildren();

    for (const cell of board.cells) {
      const square = document.createElement("button");
      square.type = "button";
      square.className = "board-cell";
      square.dataset.row = String(cell.row);
      square.dataset.col = String(cell.col);
      square.dataset.zone = String(zoneIndex(cell));
      square.disabled = !humanTurn || !playable.has(cell.index);
      square.setAttribute("aria-label", describeCell(cell, winningCells.has(cell.index), playable.has(cell.index)));
      if (cell.col === 1) square.classList.add("region-right");
      if (cell.row === 1) square.classList.add("region-bottom");
      if (winningCells.has(cell.index)) square.classList.add("is-winning");
      if (humanTurn && playable.has(cell.index)) square.classList.add("is-legal");
      square.addEventListener("click", () => playHumanMove(cell.index));

      const coordinate = document.createElement("span");
      coordinate.className = "coordinate";
      coordinate.textContent = String(cell.index);
      square.append(coordinate);
      if (cell.player !== null) square.append(createPiece(cell));
      elements.boardGrid.append(square);
    }
  }

  function playHumanMove(position) {
    if (engines[game.sideToMove]?.kind !== "human") return;
    try {
      game = Game.applyMove(game, { player: game.sideToMove, shape: selectedShape, position }, "human");
      elements.gameMessage.textContent = "";
      render();
      maybeRunEngine();
    } catch (error) {
      elements.gameMessage.textContent = error.message;
    }
  }

  async function stepEngine() {
    const status = Game.getGameStatus(game);
    const engine = engines[game.sideToMove];
    if (status.phase === "finished" || engineBusy) return;
    if (!engine || typeof engine.chooseMove !== "function") {
      elements.gameMessage.textContent = engine?.kind === "remote"
        ? "Enter a remote endpoint, then reselect the controller."
        : "The current player is human. Choose a piece and square.";
      return;
    }

    engineBusy = true;
    elements.gameMessage.textContent = `${engine.label} is choosing…`;
    try {
      const move = await engine.chooseMove(game);
      if (move) game = Game.applyMove(game, move, `engine:${engine.kind}`);
      elements.gameMessage.textContent = move
        ? `${engine.label} played ${move.shape} on ${move.position}.`
        : `${engine.label} has no legal move.`;
      render();
    } catch (error) {
      stopAutoplay();
      elements.gameMessage.textContent = error.message;
      setStatus("Engine error");
    } finally {
      engineBusy = false;
    }
  }

  function maybeRunEngine() {
    if (engines[game.sideToMove]?.kind !== "human") stepEngine();
  }

  function toggleAutoplay() {
    if (autoplayTimer !== null) {
      stopAutoplay();
      return;
    }
    if (engines.some((engine) => engine.kind === "human")) {
      elements.gameMessage.textContent = "Choose an engine for both players to autoplay.";
      return;
    }
    elements.autoplayButton.textContent = "Pause autoplay";
    autoplayTimer = global.setInterval(async () => {
      if (Game.getGameStatus(game).phase === "finished") {
        stopAutoplay();
        return;
      }
      await stepEngine();
    }, Number(elements.speedSelect.value));
    stepEngine();
  }

  function stopAutoplay() {
    if (autoplayTimer !== null) global.clearInterval(autoplayTimer);
    autoplayTimer = null;
    if (elements.autoplayButton) elements.autoplayButton.textContent = "Start autoplay";
  }

  /** Opponents that actually have a network to ask.
   *
   * `uniform-mcts128` is excluded even though it is an MCTS opponent: the
   * service returns nothing for it on purpose, because a flat prior and a
   * constant zero describe no position in particular. Offering it here
   * would be offering a choice that always answers "no opinion".
   */
  function assessors() {
    return opponents.filter((o) => o.modelId !== null);
  }

  function renderAnalysisOpponents() {
    const previous = Settings.readProfile().analysisOpponent;
    elements.analysisOpponent.replaceChildren();
    const off = document.createElement("option");
    off.value = "";
    off.textContent = "Nobody — no assessment";
    elements.analysisOpponent.append(off);
    for (const opponent of assessors()) {
      const option = document.createElement("option");
      option.value = opponent.id;
      option.textContent = opponent.label;
      elements.analysisOpponent.append(option);
    }
    if (previous && assessors().some((o) => o.id === previous)) {
      elements.analysisOpponent.value = previous;
    }
  }

  async function refreshAssessment() {
    const opponentId = elements.analysisOpponent.value;
    const token = ++analysisToken;

    if (!opponentId) {
      elements.assessmentSource.textContent = "off";
      drawEvaluation(null);
      elements.topMoves.replaceChildren();
      elements.evaluationCaption.textContent =
        "Choose a model to see what it makes of the position.";
      return;
    }

    elements.assessmentSource.textContent = opponentId;
    try {
      const analysis = await Play.analysePosition(game, opponentId, {
        baseUrl: serviceBase(),
      });
      // A later position has already been asked about, so this answer
      // describes a board that is no longer on screen.
      if (token !== analysisToken) return;
      drawEvaluation(analysis);
      renderTopMoves(analysis);
    } catch (error) {
      if (token !== analysisToken) return;
      drawEvaluation(null);
      elements.topMoves.replaceChildren();
      elements.evaluationCaption.textContent = `No assessment: ${error.message}`;
    }
  }

  function drawEvaluation(analysis) {
    const share = analysis ? analysis.player0WinProbability : null;
    if (share === null || share === undefined) {
      elements.evaluationFill.style.width = "50%";
      elements.evaluationBar.dataset.known = "false";
      elements.evaluationBar.setAttribute("aria-label", "No assessment available");
      if (analysis) {
        elements.evaluationCaption.textContent =
          "That opponent has no value head, so it has no opinion on the position.";
      }
      return;
    }
    elements.evaluationBar.dataset.known = "true";
    elements.evaluationFill.style.width = `${(share * 100).toFixed(1)}%`;
    const leader = share >= 0.5 ? 0 : 1;
    const chance = Math.round((leader === 0 ? share : 1 - share) * 100);
    const label = `Player ${leader} · ${PLAYER_NAMES[leader]} at ${chance}%`;
    elements.evaluationBar.setAttribute("aria-label", label);
    // "estimate" is doing real work in this sentence. Quantik is solved,
    // so the true answer is a win or a loss and never 63% — this number
    // describes the network's confidence, not the position.
    elements.evaluationCaption.textContent = `${label} — the model's estimate, not the solver's.`;
  }

  function renderTopMoves(analysis) {
    elements.topMoves.replaceChildren();
    if (!analysis || !analysis.topMoves.length) return;
    for (const move of analysis.topMoves.slice(0, 5)) {
      const item = document.createElement("li");
      const piece = createMiniPiece(move.shape, game.sideToMove);
      const where = document.createElement("span");
      // Row and column rather than the raw index: the board shows
      // coordinates, and 0-15 is an implementation detail of the action
      // encoding.
      where.textContent = `${"abcd"[move.position % 4]}${Math.floor(move.position / 4) + 1}`;
      const share = document.createElement("span");
      share.className = "top-move-prior";
      share.textContent = `${Math.round(move.prior * 100)}%`;
      item.append(piece, where, share);
      elements.topMoves.append(item);
    }
  }

  async function recordFinishedGame() {
    const humanSeat = engines.findIndex((engine) => engine.kind === "human");
    // With a human at the board the opponent is the other seat. With none,
    // both seats are engines and there is no "the opponent" to resolve —
    // the two engine versions carry both identities, so nothing is lost.
    const opponentSeat = humanSeat === 0 ? 1 : humanSeat === 1 ? 0 : null;
    const opponent =
      opponentSeat === null
        ? null
        : opponents.find((o) => o.id === engines[opponentSeat]?.version) || null;
    try {
      const body = Play.buildGameRecord(game, {
        playerName: elements.playerName.value.trim(),
        humanSeat: humanSeat === -1 ? null : humanSeat,
        opponent,
        opponentSeat,
        engines,
      });
      const result = await Play.recordGame(body, {
        baseUrl: serviceBase(),
        recording: capabilities.recording,
      });
      // A disagreement is reported, not swallowed: it is the only signal
      // that these rules and quantik-core's have drifted apart. A skip is
      // reported too, but as a plain fact — this server was never asked,
      // so there is no HTTP status to show and nothing went wrong.
      elements.gameMessage.textContent = result.skipped
        ? "Not saved — this server keeps no record of games."
        : result.discrepancies?.length
          ? `Recorded, but the service disagreed: ${result.discrepancies.join("; ")}`
          : result.recorded
            ? "Game recorded."
            : "Game was already recorded.";
    } catch (error) {
      elements.gameMessage.textContent = `Not recorded: ${error.message}`;
    }
  }

  function undoMove() {
    if (!game.moves.length) return;
    stopAutoplay();
    const records = game.moves.slice(0, -1);
    // The id and start time survive an undo. Without them every undo mints
    // a fresh game, which defeats the already-recorded guard and would let
    // one game be stored several times under different ids.
    let replay = Game.createGame(game.initialQfen, { id: game.id, startedAt: game.startedAt });
    for (const record of records) {
      replay = Game.applyMove(
        replay,
        Game.moveFromActionIndex(record.actionIndex, replay.sideToMove),
        record.actor,
      );
    }
    game = replay;
    render();
  }

  function exportTrace() {
    const trace = Trace.createTrace(game, engines);
    const blob = new Blob([`${JSON.stringify(trace, null, 2)}\n`], { type: "application/json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `${trace.game_id}.json`;
    link.click();
    URL.revokeObjectURL(link.href);
    elements.gameMessage.textContent = `Exported ${trace.plies} plies.`;
  }

  async function importTrace() {
    const file = elements.importFile.files[0];
    if (!file) return;
    try {
      stopAutoplay();
      game = Trace.parseTrace(await file.text());
      // Adopted, not recorded: an imported game was played somewhere else.
      finishWatcher.adopt(game);
      configureEngines();
      render();
      elements.gameMessage.textContent = `Imported ${game.moves.length} plies.`;
    } catch (error) {
      elements.gameMessage.textContent = error.message;
    } finally {
      elements.importFile.value = "";
    }
  }

  function renderTurn(status = Game.getGameStatus(game)) {
    if (!game || !elements.turnLabel) return;
    if (status.phase === "finished") {
      elements.turnLabel.textContent = status.winner === null
        ? "Position is terminal"
        : `Player ${status.winner} wins · ${status.terminalReason.replaceAll("_", " ")}`;
      return;
    }
    elements.turnLabel.textContent = `Player ${game.sideToMove} · ${engines[game.sideToMove]?.label || "Human"}`;
  }

  function renderShapePicker(legalMoves = game ? Game.getLegalMoves(game) : [], summary = null) {
    elements.shapePicker.replaceChildren();
    const side = game ? game.sideToMove : 0;
    const remaining = summary ? summary.players[side].remaining : null;
    for (const shape of SHAPES) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "shape-button";
      button.dataset.selected = String(shape === selectedShape);
      button.disabled = !legalMoves.some((move) => move.shape === shape) || engines[side]?.kind !== "human";

      const label = document.createElement("span");
      label.className = "shape-label";
      label.textContent = SHAPE_META[shape].label;
      button.append(createMiniPiece(shape, side), label);

      if (remaining) {
        // How many of that shape the side to move still holds. Two per
        // shape per player, and on a phone the label is hidden and this is
        // the only text left — which is the more useful of the two, since
        // "0 left" is the difference between a shape you may not want and
        // one you cannot have.
        const count = document.createElement("span");
        count.className = "shape-remaining";
        count.textContent = `${remaining[shape]}`;
        count.title = `${remaining[shape]} left`;
        button.append(count);
      }

      button.addEventListener("click", () => {
        selectedShape = shape;
        render();
      });
      elements.shapePicker.append(button);
    }
  }

  function renderHistory() {
    elements.moveHistory.replaceChildren();
    if (!game.moves.length) {
      const item = document.createElement("li");
      item.className = "empty-state";
      item.textContent = "No moves yet";
      elements.moveHistory.append(item);
      return;
    }
    for (const move of game.moves) {
      const item = document.createElement("li");
      item.textContent = `${move.ply + 1}. P${move.sideToMove} ${move.shape}@${move.position}`;
      const actor = document.createElement("small");
      actor.textContent = move.actor;
      item.append(actor);
      elements.moveHistory.append(item);
    }
  }

  function renderWins(analysis) {
    elements.winCount.textContent = `${analysis.wins.length} ${plural(analysis.wins.length, "win")}`;
    elements.winList.replaceChildren();
    if (!analysis.wins.length) {
      const item = document.createElement("li");
      item.className = "empty-state";
      item.textContent = "No complete shape set";
      elements.winList.append(item);
    }
    for (const win of analysis.wins) {
      const item = document.createElement("li");
      item.textContent = Qfen.describeWin(win);
      elements.winList.append(item);
    }
  }

  function renderInventory(summary) {
    elements.inventory.replaceChildren();
    summary.players.forEach((player, playerIndex) => {
      const group = document.createElement("div");
      group.className = "inventory-player";
      const title = document.createElement("h3");
      title.textContent = PLAYER_NAMES[playerIndex];
      group.append(title);
      for (const shape of SHAPES) {
        const row = document.createElement("div");
        row.className = "inventory-row";
        if (player.overused[shape] > 0) row.classList.add("is-overused");
        row.append(createMiniPiece(shape, playerIndex), textNode(SHAPE_META[shape].label), strongNode(String(player.used[shape])), textNode(`${player.remaining[shape]} left`));
        group.append(row);
      }
      elements.inventory.append(group);
    });
  }

  function renderRules() {
    elements.howToPlaySummary.textContent = Rules.RULES.title;
    elements.howToPlayPoints.replaceChildren();
    for (const point of Rules.RULES.points) {
      const item = document.createElement("li");
      item.textContent = point;
      elements.howToPlayPoints.append(item);
    }
  }

  function renderLegend() {
    elements.legend.replaceChildren();
    for (const shape of SHAPES) {
      const item = document.createElement("div");
      item.className = "legend-item";
      item.append(createMiniPiece(shape, 0), textNode(shape), strongNode(SHAPE_META[shape].label));
      elements.legend.append(item);
    }
  }

  function renderExamples() {
    elements.examples.replaceChildren();
    for (const example of Examples.EXAMPLES) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = example.label;
      // The blurb, not the QFEN: the tooltip is the only place the reason
      // for loading a position can be said, and a raw QFEN says nothing a
      // person could act on.
      button.title = example.blurb;
      button.addEventListener("click", () => {
        startGame(example.qfen);
        elements.gameMessage.textContent = example.blurb;
      });
      elements.examples.append(button);
    }
  }

  function createPiece(cell) {
    const piece = document.createElement("span");
    piece.className = `piece ${SHAPE_META[cell.shape].className}`;
    piece.dataset.player = String(cell.player);
    piece.dataset.shape = cell.shape;
    const face = document.createElement("span");
    face.className = "piece-face";
    const label = document.createElement("span");
    label.className = "piece-label";
    label.textContent = cell.char;
    face.append(label);
    piece.append(face);
    return piece;
  }

  function createMiniPiece(shape, player) {
    const piece = document.createElement("span");
    piece.className = `mini-piece ${SHAPE_META[shape].className}`;
    piece.dataset.player = String(player);
    piece.textContent = shape;
    return piece;
  }

  function updateThemeFromControls() {
    theme = Settings.normalizeTheme({ uppercase: elements.uppercaseColor.value, lowercase: elements.lowercaseColor.value });
    applyTheme(theme);
    Settings.writeTheme(theme);
  }

  function resetTheme() {
    theme = Settings.normalizeTheme();
    applyTheme(theme);
    Settings.writeTheme(theme);
  }

  function applyTheme(nextTheme) {
    for (const [name, value] of Object.entries(Settings.themeToCssVariables(nextTheme))) {
      document.documentElement.style.setProperty(name, value);
    }
    elements.uppercaseColor.value = nextTheme.uppercase.color;
    elements.lowercaseColor.value = nextTheme.lowercase.color;
  }

  async function copyQfen() {
    try {
      await navigator.clipboard.writeText(game.qfen);
      setStatus("Copied");
    } catch {
      elements.qfenInput.select();
      setStatus("Selected");
    }
  }

  function showError(error) {
    elements.errorMessage.textContent = error.message;
    elements.qfenInput.setAttribute("aria-invalid", "true");
    setStatus("Invalid");
  }

  function describeCell(cell, winning, playable) {
    const prefix = `Square ${cell.index}, row ${cell.row + 1}, column ${cell.col + 1}`;
    if (cell.player === null) return `${prefix}, empty${playable ? ", legal move" : ""}${winning ? ", winning line" : ""}`;
    return `${prefix}, player ${cell.player}, ${SHAPE_META[cell.shape].label}${winning ? ", winning line" : ""}`;
  }

  function setStatus(text) {
    elements.statusPill.textContent = text;
    elements.statusPill.dataset.state = text.toLowerCase().replaceAll(" ", "-");
  }

  function zoneIndex(cell) { return Math.floor(cell.row / 2) * 2 + Math.floor(cell.col / 2); }
  function textNode(text) { const node = document.createElement("span"); node.textContent = text; return node; }
  function strongNode(text) { const node = document.createElement("strong"); node.textContent = text; return node; }
  function plural(count, singular, pluralWord = `${singular}s`) { return count === 1 ? singular : pluralWord; }
  function toCamel(value) { return value.replace(/-([a-z0-9])/g, (_, char) => char.toUpperCase()); }
})(globalThis);
