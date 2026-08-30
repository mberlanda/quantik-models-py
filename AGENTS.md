# AGENTS.md

Instructions for humans and AI agents working in this repository.

## What this is

The `quantik-models` package: training, dataset materialization, autoplay, the
arena, checkpoint export, evaluation, and the play service. It consumes
`quantik-core-contracts`, `quantik-core-rust` and `quantik-core-py`.

`README.md` covers what the package does. `docs/README.md` is the reading order
for the twenty-three documents in `docs/` — **start there**, not by grepping.
This file covers tooling and workflow.

## Environment

Everything runs from the repo's own virtualenv. There is no venv at the
workspace root, and a bare `python` is the system one.

```sh
[ -d .venv ] || python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev,arrow,torch,onnx,viz]" \
  --extra-index-url https://download.pytorch.org/whl/cpu
```

Optional extras are real boundaries, not packaging decoration: `torch`,
`onnx`, `arrow` and `viz` are each excluded from the base install on purpose.
The base dependency is `numpy` alone.

## Test

```sh
.venv/bin/python -m pytest -q          # 513 tests
```

**`mypy` is declared in `[dev]` but is not wired into CI or any script.** It is
available; it is not a gate. Do not claim a type check passed unless you ran it.

### The torch-free install is a tested configuration

`e2e-data-pipeline.yml` installs `[dev,arrow]` — **no torch** — and runs the
unit suite against it. So the package has a torch layer and a torch-free layer,
and the boundary is load-bearing:

- **`model/*` and `train/*` are the torch layer.** A module-scope `import torch`
  is correct there.
- **`env`, `selfplay`, `arena`, `play` and `data` must stay importable without
  torch** — verified: all twelve of their modules import clean with torch
  blocked. Import torch lazily inside the function that needs it;
  `arena.registry.load_evaluator` is the pattern to copy.
- **In tests, never a bare `import torch`** — that fails *collection* for the
  whole file rather than skipping. Use `pytest.importorskip("torch")`: at module
  scope when every test in the file needs it (`test_checkpoint_roundtrip.py`),
  inside the function when only some do (`test_arena.py:133`).

### CI fails when a test skips for a missing dependency

`tests.yml` installs every extra, then parses `junit.xml` and **fails if any
test skipped with "could not import"**. Skipping is the failure mode this guard
exists for: an uninstalled extra turns `importorskip` into a green run with a
third of the suite silently unexercised.

Skips for local corpus data under `runs/` are legitimate and allowed — that data
is gitignored and can never exist on a runner. Match on the reason, not the
count.

### The three workflows

| workflow | what it protects |
|---|---|
| `tests.yml` | the unit suite on 3.12 and 3.13, every extra installed, against the **published** `quantik-core` from PyPI — deliberately not a sibling checkout, so this repo's green never waits on another repo's in-flight work |
| `train-smoke.yml` | a real tiny training run end to end, and that the **ONNX export still matches torch** |
| `e2e-data-pipeline.yml` | contracts → Rust data → model view, and the torch-free install |

## Workflow conventions

- **One PR per change**, own branch, merged when CI is green. Don't stack open
  work.
- **Atomic commits**; the message explains *why*, the diff already shows what.
- **Commit as the repository owner only.** No `Co-Authored-By:` and no
  `Claude-Session:` trailers.
- Branches: `feat/…`, `fix/…`, `chore/…`, `docs/…`.
- **Focused unit tests for every new behavior** — success, validation/error, and
  any compatibility fallback. An end-to-end smoke test is not a substitute.
- Documentation lands in the same PR as the change it describes.
- When you recommend something, write down the alternatives you rejected.

## Before anything expensive

**Smoke-test the assumptions first.** Training runs and arenas cost hours, and
this repo has paid for skipping that step more than once.

- **Never plan around a timing taken under load.** It is an upper bound, often
  a wild one: `minimax-d2` was recorded at 1.1 s/move while a solver and a sweep
  saturated the machine, and is actually 0.28 s. A `beam-w16` figure was
  discarded for the same reason. Re-measure on a quiet machine.
- **An average from a differently-shaped workload does not transfer.** Measure
  per bucket.
- Write the projected wall-clock down before starting. If it exceeds a night,
  stop and ask rather than launching.

## Measurement discipline

These are the standing conclusions that keep being re-derived the hard way.
`docs/corpus-v3.md` and `docs/learning-rate-sweep.md` carry the evidence.

- **The arena is the ranking that matters.** Held-out accuracy has now failed to
  predict play strength four separate times. When validation top-1 and the arena
  disagree, say so plainly rather than splitting the difference.
- **The seat dwarfs the model.** Mover win rates run 68–88%, responder 15–39%.
  Two networks a point apart are being compared inside an effect forty times
  larger. Keep the side-balancing; never quote an unbalanced win rate.
- **Watch the multiple comparisons.** Twenty intervals were computed in one
  study and six excluded 50% — roughly one is expected by chance. The pattern
  across rows carries an argument; a single cell does not.
- **A hyperparameter inherited from another architecture is a bug.** A shared
  learning rate produced three plausible, detailed, statistically significant
  conclusions that were all false. A shared epoch budget is the same flaw and is
  still partly unfixed — see `briefs/lineup-under-patience.md` at the workspace
  root.

## Invariants that are not negotiable

- **Game outcomes never become labels.** Only positions travel to the corpus,
  and they get their labels from the exact oracle like everything else. This
  holds for autoplay and for human games. See `docs/labeling-strategy.md`.
- **Contracts are the source of truth.** Schemas live in
  `quantik-core-contracts`; code is validated against them, not the reverse.
- **Legality masking lives outside the model, by design.** The rules are exact
  in `quantik-core`, so the network never has to approximate them.
- **`tensor-board.v1` is ambiguous — two incompatible encodings share the
  name.** Everything in training uses `fastboard.encode_tensors`, which is
  **mover-relative**. `quantik_core.ml_data.qfen_to_tensor` and
  `fastboard.to_core_tensor` are colour-ordered and are used by nothing here.
  Building to the wrong one gives a model that plays legally and confidently
  wrong on half of all positions, with nothing indicating a fault. The
  discriminating fixture is `"A.../..../..../...."`: one piece, so
  `side_to_move == 1`, and mover-relative puts the 1.0 at channel 4.
- **`runs/` is gitignored** and holds every checkpoint and corpus. Nothing in it
  can be verified from a fresh clone, and nothing in it is published.
- **Versioning is lockstep** with contracts and `quantik-core-rust`. Tag
  contracts first; tag py before rust.

## Where work is tracked

`WORKSTREAMS.md` at the workspace root, one level up. Delegation briefs for
self-contained tasks live in `briefs/` beside it.
