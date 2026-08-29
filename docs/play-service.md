# The play service

The local service that lets a person play the trained networks from a
browser and lands the finished games in a store the training pipeline can
read. This document covers the move handler — the part that decides a move
and refuses a request. `frontend-play.md` covers what the browser does with
it; the finished-game store is documented in its own module docstring,
`play/store.py`, until it has enough surface to need a page.

Everything here is implemented in `src/quantik_models/play/`, which has
four pieces and no HTTP in three of them:

| module | what it owns |
| --- | --- |
| `registry.py` | which checkpoints on disk are playable, and why the others are not |
| `opponents.py` | the roster — classical engines and neural ones, under the arena's own names |
| `service.py` | validation, legality, the move itself |
| `record.py` | replaying a submitted game, and refusing to take its word |
| `store.py` | finished games |
| `server.py` | the HTTP surface, the static app, the LAN address |
| `__main__.py` | the CLI |

`service.PlayService` takes a decoded request dict and returns a decoded
response dict. It is transport-free on purpose: the rules are testable
without a socket, and the split matches `quantik-api-rust`, where
`validate_request` and `run_search` sit apart from the axum route.

## The request

The body is `quantik.engine-request.v1`, unchanged — the same object the
visualizer already sends to the Rust gateway:

```json
{
  "schema": "quantik.engine-request.v1",
  "qfen": "A.../..../..../....",
  "side_to_move": 1,
  "legal_action_indices": [0, 1, 2],
  "config": {"seed": 12345}
}
```

`config.seed` is optional and makes a move reproducible. Without it the
service draws a fresh seed per request; a fixed default would make
`random` and both MCTS opponents replay one game forever.

The opponent is chosen by the route, not the body: `POST
/api/move/{opponent_id}`, where `opponent_id` comes from
`GET /api/opponents`. That keeps `src/engines.js` in the visualizer
unchanged: it builds its POST body from a fixed literal with no hook for
extra fields, so anything the server needs beyond the position has to ride
in the URL. That keeps `src/engines.js` in the visualizer
unchanged — it builds its POST body from a fixed literal with no hook for
extra fields, so anything the server needs to know beyond the position has
to ride in the URL.

## The response

```json
{
  "schema": "quantik.engine-response.v1",
  "action_index": 17,
  "engine_kind": "net-mcts",
  "engine_version": "cpool@128",
  "elapsed_ms": 214,
  "policy": [0.01, "... 64 floats ..."],
  "value": 0.42
}
```

`engine_version` is the opponent id and nothing else — not a filename, not
a display label. It is the string a recorded game stores as
`p*_engine_version`, and it has to be the same string `runs/eval/*/games.json`
uses for the same opponent, or a human game and a benchmark game cannot be
pooled into one dataset.

`policy` and `value` appear only for opponents backed by a real network,
and they are the **network's prior and value head**, from one extra forward
pass — not the MCTS visit distribution. Reporting visit counts would mean
changing `NetMCTSAgent.select` to hand back state it currently keeps local,
and the arena depends on that class being exactly what it is. A field
meaning "the network's prior" for both neural kinds is worth more than one
meaning two different things depending on the opponent. `uniform-mcts` is
excluded even though it has an evaluator: a flat prior and a constant zero
say nothing about the position, and an overlay drawing them as though they
did would mislead.

## What gets refused

| status | when |
| --- | --- |
| 400 | the schema is not `quantik.engine-request.v1`; `side_to_move` is not 0 or 1; an action index falls outside 0–63; `qfen` is absent, empty, or not a string; `config` is not an object; the QFEN does not parse |
| 404 | no such opponent |
| 422 | the position is already decided; `side_to_move` disagrees with `quantik-core`; the claimed legal set does not match the computed one |
| 409 | the weights file changed under a running service |
| 500 | the chosen action is illegal — a defect here, not bad input |

The status is part of the contract, not decoration, and the tests assert on
the number. A 400 tells the client it sent nonsense; a 422 tells it the
request was well-formed but this position cannot satisfy it.

The first three 400s and the `side_to_move` and legal-set 422s mirror
`quantik-api-rust/src/lib.rs` exactly. The two services have to agree: the
visualizer sends one request shape to whichever is listening, so a body one
accepts and the other rejects is a bug, not a difference of opinion.

## Two things that would otherwise fail silently

### The client's legality is checked, never believed

The browser computes legality itself so it can grey out squares, and it
sends what it computed. The service treats that list as a claim and
recomputes the legal set from `fastboard`. Any difference is a 422 naming
the exact indices — which were claimed legal but are not, and which are
legal but were omitted.

This is the only place in the system that would notice a divergence between
the JavaScript rules and the Python rules. Without it, a rules bug on
either side does not raise anything; it produces a game that was scored
under two different rulebooks and stored as though it were one.

### A checkpoint that changes on disk is refused

`arena.registry.load_evaluator` caches on `f"{checkpoint}|{device}"`. So
retraining into a directory the service has already served keeps the *old*
network alive for the life of the process, while every game recorded
against it is labelled with the model id the *new* weights carry. Nothing
downstream could detect that: the store would simply hold rows attributing
one network's play to another, and they would go on to be pooled with arena
results as if they were comparable.

So the weights file is `stat`-checked on every request, and a changed
`st_mtime_ns` or `st_size` drops the cached agent and re-digests the file.
A digest that no longer matches the manifest is refused with a 409. There
is no warn-and-continue path — a warning on a server nobody is watching is
the same as no check at all.

The freshness check runs **before** the agent is built, which has a useful
side effect: a tampered checkpoint is refused without anything trying to
load it, so the guard is covered in the torch-free CI job.

## The networks vary their opening, on purpose

A network at the arena's default temperature is a deterministic function of
the position, so the same opponent from the same start replays one game
forever. Two people watching engine-versus-engine see the same game twice,
and a human who finds one winning line wins with it every time. That is the
right default for the arena, where the question is how a fixed player
performs, and the wrong one here.

Served opponents therefore sample their first four plies:
`--opening-temperature 1.0`, `--opening-plies 4`. Setting the temperature
to `0` restores the arena's exact behaviour.

**Why the opening is where this is nearly free.** The corpus spans plies
6-13, so the policy head was never trained on the first few plies and holds
no opinion there — measured, `cpool` on an empty board puts `0.0167` on
each legal action, which is 1/60 to three places. An argmax over that is
not a considered choice; it is whichever action index sorts first. Sampling
is the more honest reading of a flat distribution. The board also has only
**three distinct canonical positions** after any first move, so the variety
costs less than the move count suggests. From ply 4 the network plays its
best move, which is where its training starts to bite.

**What it costs, stated plainly.** `cpool@128` names the same opponent here
and in the arena's `runs/eval/*/games.json`, and the roster reuses the
arena's names precisely so the rows pool into one dataset. A sampled
opening breaks that — same name, different player. `game_meta` therefore
carries `opening_temperature` and `opening_plies` on every row, so a later
analysis can separate the two rather than discover the difference by
noticing the numbers do not fit. A classical opponent stores `NULL` for
both; `minimax-d2` has no temperature to record.

`uniform-mcts128` is the one opponent the temperature does not help. Its
search is degenerate — uniform priors and a flat zero value put nearly
every visit on one action — so there is no visit distribution to sample
from. See `autoplay.md`, "the uniform control barely searches".

## Concurrency

One `threading.Lock` covers agent construction and `select`. The agents,
the MCTS trees they build, and `arena.registry`'s evaluator cache are all
shared mutable state, and the server this feeds is a `ThreadingHTTPServer` —
two phones mid-game are two real threads. Serializing inference is not a
performance decision; it is the only thing that makes the shared state
safe. Static files and the listing routes stay responsive because they
never take the lock.

## Running it

```bash
.venv/bin/python -m quantik_models.play --models staging
```

Prints what it found and where it is:

```
quantik play service 0.1.0
  models     staging  (4 ready of 5 found)
    ok cpool-c191-b6
    -- attn-c96-b4  (weights hash 3f9c... does not match manifest 'a12e...')
  opponents  14
  store      /Users/you/.local/share/quantik/games.db
  app        /Users/you/Code/quantik-ns/quantik-qfen-visualizer

  local      http://127.0.0.1:8000
  this WiFi  http://192.168.4.27:8000
```

The second address is the one to type into a phone. It is resolved by
opening a UDP socket toward TEST-NET-1 and reading back the local address
the OS chose — nothing is sent. `gethostname` would be simpler and
resolves to 127.0.0.1 on macOS: the answer that looks right and does not
work from another device.

A refused model is printed with its reason rather than omitted. Silence is
the failure mode here — a model missing from the dropdown looks like one
that was never trained.

The store defaults to `~/.local/share/quantik/games.db`, deliberately not
under `runs/`. A checkpoint can be retrained; a game somebody played
cannot be replayed, and `runs/` is gitignored and routinely deleted
wholesale.

### Routes

```
GET  /api/opponents          the dropdown
GET  /api/models             manifest detail: hashes, architecture, refusals
POST /api/move/{opponent_id} a move
POST /api/games              record a finished game
GET  /api/games?player=      counts and head-to-head
GET  /*                      the visualizer
```

`POST /api/games` answers **201** when it wrote the game and **200** when
the id was already present, so a page reload after the result screen is a
no-op rather than a duplicate. It always returns a `discrepancies` list,
empty when the client's reading of the game matched the replay — a caller
has to be able to tell "the two rule implementations agreed" from "this
server does not report disagreements".

### Why `http.server` and not a framework

The base dependency of this package is numpy alone, with torch in an
extra, and that posture is deliberate. The work a framework would do here
is five routes, JSON in and out, and a static directory. The one thing it
would genuinely buy — async concurrency — buys nothing, because inference
is serialized behind a lock regardless: the evaluator and the MCTS state
are shared.

What does matter is that the server is *threading*. A 128-simulation move
takes a second or more, and on a single-threaded server one player's move
would freeze the board on every other device in the house.
`ThreadingHTTPServer` keeps the static files and the listing routes
answering while a move computes.

One SQLite connection is opened per request rather than shared, because
`sqlite3` connections are not safe to move between threads and this server
has several. Opening one costs microseconds against a move that costs a
second.

## Verifying it

```bash
.venv/bin/python -m pytest -q tests/test_play_service.py
```

The suite runs without a network for everything except one test: the
neural path is guarded by `pytest.importorskip("torch")` **inside** the
test function. At module scope that import would fail collection for the
whole file in the torch-free job, turning one skip into a red build — which
has happened three times in this repository.
