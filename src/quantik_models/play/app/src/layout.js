(function exposeLayout(global) {
  "use strict";

  const STORAGE_KEY = "quantik-advanced-drawer-open";

  // The controls relocated into the bottom drawer, in the order they appear
  // there. Order matters here — app.js and index.html both read this as the
  // definition of "everything advanced", not just a membership set.
  const ADVANCED_CONTROL_IDS = Object.freeze([
    "qfen-input",
    "copy-button",
    "reset-button",
    "seed-input",
    "speed-select",
    "player-0-controller",
    "player-1-controller",
    "remote-endpoint-0",
    "remote-endpoint-1",
    "service-base",
    "export-button",
    "import-button",
    "import-file",
  ]);

  function isAdvancedControl(id) {
    return ADVANCED_CONTROL_IDS.includes(id);
  }

  function readDrawerOpen(storage = global.localStorage) {
    if (!storage) {
      return false;
    }

    try {
      return JSON.parse(storage.getItem(STORAGE_KEY) || "false") === true;
    } catch {
      return false;
    }
  }

  function writeDrawerOpen(open, storage = global.localStorage) {
    if (!storage) {
      return;
    }

    try {
      storage.setItem(STORAGE_KEY, JSON.stringify(Boolean(open)));
    } catch {
      // Some browsers restrict localStorage on file-opened pages.
    }
  }

  global.QuantikLayout = Object.freeze({
    ADVANCED_CONTROL_IDS,
    isAdvancedControl,
    readDrawerOpen,
    writeDrawerOpen,
  });
})(globalThis);
