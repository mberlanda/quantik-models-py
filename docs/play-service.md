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
| `store.py` | finished games |

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

## Concurrency

One `threading.Lock` covers agent construction and `select`. The agents,
the MCTS trees they build, and `arena.registry`'s evaluator cache are all
shared mutable state, and the server this feeds is a `ThreadingHTTPServer` —
two phones mid-game are two real threads. Serializing inference is not a
performance decision; it is the only thing that makes the shared state
safe. Static files and the listing routes stay responsive because they
never take the lock.

## Verifying it

```bash
.venv/bin/python -m pytest -q tests/test_play_service.py
```

The suite runs without a network for everything except one test: the
neural path is guarded by `pytest.importorskip("torch")` **inside** the
test function. At module scope that import would fail collection for the
whole file in the torch-free job, turning one skip into a red build — which
has happened three times in this repository.
