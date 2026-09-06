(function exposeSettings(global) {
  "use strict";

  const STORAGE_KEY = "quantik-qfen-theme";
  const PROFILE_STORAGE_KEY = "quantik-play-profile";
  const HEX_COLOR = /^#[0-9a-fA-F]{6}$/;
  const PLAYER_NAME_MAX_LENGTH = 40;
  const DEFAULT_THEME = Object.freeze({
    uppercase: Object.freeze({ label: "White", color: "#f8f4e8", text: "#17242a" }),
    lowercase: Object.freeze({ label: "Color", color: "#b83d4b", text: "#ffffff" }),
  });
  const DEFAULT_PROFILE = Object.freeze({
    playerName: "",
    modelId: "",
    simulations: 0,
    analysisOpponent: "",
    // Empty means "wherever this page came from", which is the normal
    // case: the service serves the app, so a phone that loaded the page
    // reaches the API at the same address without anyone typing it. A
    // default of localhost here would work on exactly one machine.
    serviceBase: "",
  });

  function normalizeTheme(input = {}) {
    return {
      uppercase: {
        label: DEFAULT_THEME.uppercase.label,
        color: normalizeColor(input.uppercase, DEFAULT_THEME.uppercase.color),
        text: normalizeColor(input.uppercaseText, DEFAULT_THEME.uppercase.text),
      },
      lowercase: {
        label: DEFAULT_THEME.lowercase.label,
        color: normalizeColor(input.lowercase, DEFAULT_THEME.lowercase.color),
        text: normalizeColor(input.lowercaseText, DEFAULT_THEME.lowercase.text),
      },
    };
  }

  function themeToCssVariables(theme) {
    const normalized = normalizeTheme({
      uppercase: theme?.uppercase?.color,
      lowercase: theme?.lowercase?.color,
      uppercaseText: theme?.uppercase?.text,
      lowercaseText: theme?.lowercase?.text,
    });

    return {
      "--uppercase-piece": normalized.uppercase.color,
      "--uppercase-piece-text": normalized.uppercase.text,
      "--lowercase-piece": normalized.lowercase.color,
      "--lowercase-piece-text": normalized.lowercase.text,
    };
  }

  function readTheme(storage = global.localStorage) {
    if (!storage) {
      return normalizeTheme();
    }

    try {
      return normalizeTheme(JSON.parse(storage.getItem(STORAGE_KEY) || "{}"));
    } catch {
      return normalizeTheme();
    }
  }

  function writeTheme(theme, storage = global.localStorage) {
    if (!storage) {
      return;
    }

    try {
      storage.setItem(STORAGE_KEY, JSON.stringify(themeToStorage(theme)));
    } catch {
      // Some browsers restrict localStorage on file-opened pages.
    }
  }

  function themeToStorage(theme) {
    const normalized = normalizeTheme({
      uppercase: theme?.uppercase?.color,
      lowercase: theme?.lowercase?.color,
      uppercaseText: theme?.uppercase?.text,
      lowercaseText: theme?.lowercase?.text,
    });

    return {
      uppercase: normalized.uppercase.color,
      lowercase: normalized.lowercase.color,
      uppercaseText: normalized.uppercase.text,
      lowercaseText: normalized.lowercase.text,
    };
  }

  function normalizeColor(value, fallback) {
    if (typeof value !== "string") {
      return fallback;
    }

    const color = value.trim();
    return HEX_COLOR.test(color) ? color.toLowerCase() : fallback;
  }

  function normalizeProfile(input = {}) {
    return {
      playerName: normalizePlayerName(input.playerName),
      modelId: normalizeModelId(input.modelId),
      simulations: normalizeSimulations(input.simulations),
      analysisOpponent: normalizeModelId(input.analysisOpponent),
      serviceBase: normalizeServiceBase(input.serviceBase),
    };
  }

  function normalizePlayerName(value) {
    return String(value ?? "").trim().slice(0, PLAYER_NAME_MAX_LENGTH);
  }

  function normalizeModelId(value) {
    return String(value ?? "").trim();
  }

  function normalizeServiceBase(value) {
    // Trailing slashes are stripped here rather than at every call site,
    // since the request paths all start with one and `//api/...` is a
    // different URL that no server routes.
    return String(value ?? "").trim().replace(/\/+$/, "");
  }

  function normalizeSimulations(value) {
    const simulations = Number(value);
    if (!Number.isInteger(simulations) || simulations < 0) {
      return DEFAULT_PROFILE.simulations;
    }
    return simulations;
  }

  function readProfile(storage = global.localStorage) {
    if (!storage) {
      return normalizeProfile();
    }

    try {
      return normalizeProfile(JSON.parse(storage.getItem(PROFILE_STORAGE_KEY) || "{}"));
    } catch {
      return normalizeProfile();
    }
  }

  function writeProfile(profile, storage = global.localStorage) {
    if (!storage) {
      return;
    }

    try {
      storage.setItem(PROFILE_STORAGE_KEY, JSON.stringify(normalizeProfile(profile)));
    } catch {
      // Some browsers restrict localStorage on file-opened pages.
    }
  }

  global.QuantikSettings = Object.freeze({
    DEFAULT_PROFILE,
    DEFAULT_THEME,
    normalizeProfile,
    normalizeTheme,
    readProfile,
    readTheme,
    themeToCssVariables,
    writeProfile,
    writeTheme,
  });
})(globalThis);
