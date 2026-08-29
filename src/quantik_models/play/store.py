"""SQLite storage for games humans play against trained Quantik networks.

This is the storage layer only: no HTTP, no request handling, no game
logic. A later service imports `record_game` from a request handler and
`head_to_head` / `distinct_positions` from a dashboard or a solver-queue
script.

`games` carries exactly the columns of `game-result.v1`
(`quantik-core-contracts/schemas/game-result-v1.json`), in contract order,
so a later parquet export is a plain `SELECT * FROM games` — no reshaping,
no column renaming, nothing to get out of sync as the contract evolves.
Everything the service needs that the contract has no column for —
who the human was, what serving config faced them, the raw client report —
lives in `game_meta` instead, keyed by `game_id`. Keeping the two apart is
what keeps the export clean: a contract reader should never have to know
`player_name` or `weights_hash` exist.

`game_positions` holds one row per position visited. **Only positions ever
leave this database for training.** A human game's `winner` is not a label
by itself — Quantik positions are labelled by exact search, the same way
autoplay's positions are, in `data.exact_corpus` and `data.merge_corpus`.
Recording a human's win does not tell the solver anything an autoplay game
against the same opening wouldn't; what a human game adds is which
positions were actually *reached*, which is exactly what `game_positions`
and `distinct_positions` exist to queue.

The contract's `winner` field has no draw value because Quantik has no
draws — a player who cannot move has lost. That means a terminal position
pasted in as a game's start, or any other path that reaches this store
without a winner, is not a valid completed game; the `CHECK` on `winner`
refuses it rather than defaulting it to 0 and quietly recording a fake
result.

`terminal_reason` is constrained to the two values `quantik-core-contracts`
documents this service as producing (`docs/game-result-v1.md`):
`win_condition` and `no_legal_moves`. The browser client reports a third
value, `"line"`, for the first of these — that mapping is a request-handler
concern, not a storage concern, and the `CHECK` here is what stops the
client's vocabulary leaking into the contract's.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

GAME_COLUMNS = (
    "schema",
    "contract_version",
    "game_id",
    "started_at",
    "p0_engine_kind",
    "p0_engine_version",
    "p1_engine_kind",
    "p1_engine_version",
    "initial_position_key",
    "winner",
    "plies",
    "terminal_reason",
    "move_action_indices",
    "run_id",
)

META_COLUMNS = (
    "game_id",
    "recorded_at",
    "human_seat",
    "player_name",
    "opponent_seat",
    "opponent_id",
    "model_id",
    "simulations",
    "agent_kind",
    "weights_hash",
    "manifest_model_id",
    "architecture",
    "service_version",
    "client_user_agent",
    "initial_qfen",
    "final_qfen",
    "client_winner",
    "client_terminal_reason",
    # The opening temperature the served network played at. `cpool@128`
    # names the same opponent in this store and in the arena's
    # `runs/eval/*/games.json`, and `play.opponents` justifies reusing the
    # arena's names on the grounds that the rows then pool. A sampled
    # opening breaks that — same name, different player — so the setting
    # belongs on the row rather than in the operator's memory of which
    # flags the service was started with.
    "opening_temperature",
    "opening_plies",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    schema                 TEXT    NOT NULL CHECK (schema = 'game-result.v1'),
    contract_version       TEXT    NOT NULL,
    game_id                TEXT    PRIMARY KEY,
    started_at             TEXT    NOT NULL,
    p0_engine_kind         TEXT    NOT NULL,
    p0_engine_version      TEXT    NOT NULL,
    p1_engine_kind         TEXT    NOT NULL,
    p1_engine_version      TEXT    NOT NULL,
    initial_position_key   TEXT    NOT NULL,
    winner                 INTEGER NOT NULL CHECK (winner IN (0, 1)),
    plies                  INTEGER NOT NULL CHECK (plies BETWEEN 1 AND 16),
    terminal_reason        TEXT    NOT NULL
        CHECK (terminal_reason IN ('win_condition', 'no_legal_moves')),
    move_action_indices    TEXT    NOT NULL CHECK (json_valid(move_action_indices)),
    run_id                 TEXT
);

CREATE INDEX IF NOT EXISTS idx_games_started_at ON games (started_at);

CREATE TABLE IF NOT EXISTS game_meta (
    game_id                   TEXT    PRIMARY KEY
        REFERENCES games (game_id) ON DELETE CASCADE,
    recorded_at                TEXT    NOT NULL,
    -- Nullable, because the approved plan records *every* game, and a
    -- model-vs-model one has no human in it. NOT NULL here would have
    -- silently narrowed the store to human games only, and the shape of
    -- that failure is an IntegrityError at the end of somebody's game.
    human_seat                 INTEGER CHECK (human_seat IS NULL OR human_seat IN (0, 1)),
    player_name                TEXT,
    opponent_seat               INTEGER,
    opponent_id                  TEXT,
    model_id                     TEXT,
    simulations                  INTEGER,
    agent_kind                   TEXT,
    weights_hash                 TEXT,
    manifest_model_id            TEXT,
    architecture                 TEXT,
    service_version              TEXT,
    client_user_agent            TEXT,
    initial_qfen                 TEXT,
    final_qfen                   TEXT,
    client_winner                INTEGER,
    client_terminal_reason       TEXT,
    opening_temperature          REAL,
    opening_plies                INTEGER
);

CREATE INDEX IF NOT EXISTS idx_game_meta_player_opponent
    ON game_meta (player_name, opponent_id);
CREATE INDEX IF NOT EXISTS idx_game_meta_opponent ON game_meta (opponent_id);

CREATE TABLE IF NOT EXISTS game_positions (
    game_id          TEXT    NOT NULL REFERENCES games (game_id) ON DELETE CASCADE,
    ply              INTEGER NOT NULL,
    qfen             TEXT    NOT NULL,
    canonical_key    TEXT    NOT NULL,
    PRIMARY KEY (game_id, ply)
);

CREATE INDEX IF NOT EXISTS idx_game_positions_canonical_key
    ON game_positions (canonical_key);
"""


def connect(path: Path) -> sqlite3.Connection:
    """Open the store, applying the pragmas correctness here depends on.

    `foreign_keys = ON` is not the default in SQLite even when the schema
    declares `REFERENCES ... ON DELETE CASCADE` — without it, cascade
    deletes silently do nothing and `game_meta` / `game_positions` rows
    outlive the `games` row they belong to. `journal_mode = WAL` lets a
    dashboard read while a game is being recorded, which a request-serving
    process will be doing constantly. `busy_timeout` turns a concurrent
    writer's lock into a short wait instead of an immediate
    `OperationalError`.
    """
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.executescript(_SCHEMA)
    _add_missing_meta_columns(conn)
    conn.row_factory = sqlite3.Row
    return conn


# The `game_meta` columns this module knows how to create on a table that
# already exists. `games` is deliberately absent: it mirrors
# `game-result.v1` column for column, so a change there is a contract
# change that deserves a considered migration rather than a silent
# `ALTER TABLE`. `game_meta` is this service's own annotation space, where
# a new nullable column costs nothing and losing one costs a game.
_META_COLUMN_TYPES = {
    "opening_temperature": "REAL",
    "opening_plies": "INTEGER",
}


def _add_missing_meta_columns(conn: sqlite3.Connection) -> None:
    """Bring an existing `game_meta` forward to the current column set.

    `CREATE TABLE IF NOT EXISTS` does nothing at all to a table that is
    already there, so a database written before a column existed keeps its
    old shape and every insert naming the new column fails with "table
    game_meta has no column named ..." — at the end of somebody's game,
    which is the worst moment to discover it. These columns are all
    nullable with no default, which is what makes adding them safe.
    """
    existing = {row[1] for row in conn.execute("PRAGMA table_info(game_meta)")}
    if not existing:
        return
    with conn:
        for column, sql_type in _META_COLUMN_TYPES.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE game_meta ADD COLUMN {column} {sql_type}")


def record_game(
    conn: sqlite3.Connection,
    game: dict,
    meta: dict,
    positions: list[dict],
) -> bool:
    """Insert one finished game, its meta, and its positions atomically.

    Returns `False` without writing anything when `game["game_id"]` is
    already present. A browser can POST the same finished game twice — a
    page reload after the result screen renders, or a client retry after a
    response that succeeded but looked like a timeout — and the second
    call must be a no-op rather than a duplicate row or a raised
    `IntegrityError`, since the client has no reliable way to tell "already
    recorded" from "never sent" apart.

    All three inserts share one transaction: if `positions` or `meta`
    violates a constraint, the `games` row that would otherwise have gone
    in alone is rolled back with it, so a game is never left half-recorded
    for a retry to trip over.

    `game["move_action_indices"]` is a Python list of action indices; it is
    serialised to JSON here so callers build the same dict shape whether
    they end up writing it to SQLite or to a `game-result.v1` Parquet file.
    """
    values = tuple(
        json.dumps(game["move_action_indices"])
        if column == "move_action_indices"
        else game.get(column)
        for column in GAME_COLUMNS
    )
    placeholders = ", ".join("?" for _ in GAME_COLUMNS)
    with conn:
        cursor = conn.execute(
            f"INSERT INTO games ({', '.join(GAME_COLUMNS)}) "
            f"VALUES ({placeholders}) "
            "ON CONFLICT (game_id) DO NOTHING",
            values,
        )
        if cursor.rowcount == 0:
            return False

        meta_values = tuple(
            game["game_id"] if column == "game_id" else meta.get(column)
            for column in META_COLUMNS
        )
        conn.execute(
            f"INSERT INTO game_meta ({', '.join(META_COLUMNS)}) "
            f"VALUES ({', '.join('?' for _ in META_COLUMNS)})",
            meta_values,
        )

        conn.executemany(
            "INSERT INTO game_positions (game_id, ply, qfen, canonical_key) "
            "VALUES (?, ?, ?, ?)",
            [
                (game["game_id"], position["ply"], position["qfen"], position["canonical_key"])
                for position in positions
            ],
        )
    return True


def game_count(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) AS n FROM games").fetchone()
    return int(row["n"])


def head_to_head(conn: sqlite3.Connection, player_name: str | None = None) -> list[dict]:
    """Human record against each opponent, split by seat.

    Mirrors `arena.pack.head_to_head`'s reason for splitting: from most
    openings the mover has a real edge, so a human's win rate against one
    opponent pools two different questions — "is the human better than
    this opponent" and "did the human move first" — into one number.
    Splitting by `human_seat` keeps those apart. A player who wins every
    game moving first and loses every game moving second is two rows, one
    at 1.0 and one at 0.0, not one row at 0.5.

    Rows are ordered by `(opponent_id, human_seat)` rather than by win
    rate, so the two seats against the same opponent read as a pair.
    """
    query = """
        SELECT
            game_meta.opponent_id AS opponent_id,
            game_meta.human_seat AS human_seat,
            COUNT(*) AS games,
            SUM(CASE WHEN games.winner = game_meta.human_seat THEN 1 ELSE 0 END)
                AS human_wins
        FROM games
        JOIN game_meta ON game_meta.game_id = games.game_id
        WHERE (:player_name IS NULL OR game_meta.player_name = :player_name)
        GROUP BY game_meta.opponent_id, game_meta.human_seat
        ORDER BY game_meta.opponent_id, game_meta.human_seat
    """
    rows = conn.execute(query, {"player_name": player_name}).fetchall()
    out = []
    for row in rows:
        games = int(row["games"])
        human_wins = int(row["human_wins"])
        out.append(
            {
                "opponent_id": row["opponent_id"],
                "human_seat": row["human_seat"],
                "games": games,
                "human_wins": human_wins,
                "win_rate": human_wins / games if games else 0.0,
            }
        )
    return out


def distinct_positions(conn: sqlite3.Connection, max_ply: int = 6) -> list[tuple[str, str]]:
    """`(qfen, canonical_key)` pairs at or below `max_ply`, one per key.

    Two games that share an opening reach the same canonical position
    through separate rows, one per game; sending both to the solver queue
    wastes the solver the same way an unmerged autoplay run does
    (`arena.pack.merge_qfens`). `MIN(rowid)` is SQLite's documented way to
    pick a deterministic representative row per group without a subquery:
    with exactly one `MIN()`/`MAX()` aggregate in the query, SQLite takes
    the plain columns from the row that produced it, so `qfen` comes from
    whichever row was inserted first for that `canonical_key`.
    """
    rows = conn.execute(
        """
        SELECT qfen, canonical_key, MIN(rowid)
        FROM game_positions
        WHERE ply <= ?
        GROUP BY canonical_key
        ORDER BY canonical_key
        """,
        (max_ply,),
    ).fetchall()
    return [(row["qfen"], row["canonical_key"]) for row in rows]
