# Developing `quantik-models`

Everything about working *on* this package. Using it is
[`README.md`](README.md).

## The workspace

This repository is one of several that make up Quantik, and three of them are
contracts or engines this one consumes. You do **not** need the workspace to
use the published package — `pip install quantik-models` pulls
`quantik-core` from PyPI like any other dependency. You need it to run the
data pipeline end to end, because that calls the Rust engine and the
contracts validators directly.

```bash
export QUANTIK_NS="$HOME/Code/quantik-ns"
mkdir -p "$QUANTIK_NS" && cd "$QUANTIK_NS"

git clone https://github.com/mberlanda/quantik-core-contracts.git
git clone https://github.com/mberlanda/quantik-core-rust.git
git clone https://github.com/mberlanda/quantik-core-py.git
git clone https://github.com/mberlanda/quantik-models-py.git
```

| repository | what it owns |
|---|---|
| `quantik-core-contracts` | schemas, fixtures, validators, the shared GitHub Actions. **The source of truth**: code is validated against the schemas, not the reverse. |
| `quantik-core-rust` | search, opening-book generation, the exact oracle |
| `quantik-core-py` | artifact readers, QFEN and bitboard helpers, manifest validation |
| `quantik-models-py` | this repository |

## Environment

Everything runs from this repository's own virtualenv. There is no venv at
the workspace root, and a bare `python` is the system one.

```bash
cd "$QUANTIK_NS/quantik-models-py"
[ -d .venv ] || python3 -m venv .venv
.venv/bin/python -m pip install -e ".[all]" \
  --extra-index-url https://download.pytorch.org/whl/cpu
```

To work against a sibling checkout of the core rather than the published
release — needed when a contract change is in flight in both repositories:

```bash
.venv/bin/python -m pip install -e "../quantik-core-py[arrow]"
```

**The optional extras are real boundaries, not packaging decoration.** The
base dependency is `numpy` and `quantik-core`, and it stays that way:

| extra | what it adds | who needs it |
|---|---|---|
| `torch` | torch, safetensors | training, and the torch evaluator |
| `onnx` | the ONNX *exporter* and runtime | producing `model.onnx` |
| `serve` | onnxruntime only | running a `model.onnx` — the Docker image |
| `hub` | huggingface-hub | `quantik_models.hub` |
| `arrow` | pyarrow | Parquet corpus IO |
| `viz` | matplotlib | the benchmark figures |
| `dev` | pytest, mypy, build, twine | this document |

### The torch-free layer is a tested configuration

`e2e-data-pipeline.yml` installs `[dev,arrow]` — **no torch** — and runs the
unit suite against it, and the storeless Docker image installs `[serve]` and
no torch at all. So the package has a torch layer and a torch-free layer, and
the boundary is load-bearing:

- **`model/*` and `train/*` are the torch layer.** A module-scope
  `import torch` is correct there.
- **`env`, `selfplay`, `arena`, `play`, `data` and `hub` must stay importable
  without torch.** Import it lazily inside the function that needs it;
  `arena.registry.load_evaluator` is the pattern to copy.
- **In tests, never a bare `import torch`** — that fails *collection* for the
  whole file rather than skipping it. Use `pytest.importorskip("torch")`: at
  module scope when every test in the file needs it
  (`tests/test_checkpoint_roundtrip.py`), inside the function when only some
  do (`tests/test_arena.py`).

## Test and check

```bash
.venv/bin/python -m pytest -q     # the unit suite
.venv/bin/python -m mypy          # a gate, not advice — CI runs it
```

`mypy` is configured in `pyproject.toml` and deliberately not strict.
`disallow_untyped_defs` across a package that predates the checker fails on
sixty files, and a flag that fails on sixty files is a flag somebody turns
off. What is enabled is what has actually caught bugs here.

### CI fails when a test *skips* for a missing dependency

`tests.yml` installs every extra, then parses `junit.xml` and fails if any
test skipped with "could not import". Skipping is the failure mode that guard
exists for: an uninstalled extra turns `importorskip` into a green run with a
third of the suite silently unexercised, which is how six of fourteen test
modules went unrun for weeks.

Skips for local corpus data under `runs/` are legitimate — that data is
gitignored and can never exist on a runner. Match on the reason, not the
count.

### The five workflows

| workflow | what it protects |
|---|---|
| `tests.yml` | the unit suite on 3.12 and 3.13 with every extra, plus `mypy`. Runs against the **published** `quantik-core`, deliberately not a sibling checkout, so this repo's green never waits on another repo's in-flight work. |
| `build.yml` | that the sdist and wheel build, that their contents are publishable, and that the wheel installs and works on six OS/Python combinations **with no extras** |
| `publish.yml` | the release itself — see below |
| `train-smoke.yml` | a real tiny training run end to end, and that the ONNX export still matches torch |
| `e2e-data-pipeline.yml` | contracts → Rust data → model view, and the torch-free install |

## The smoke pipeline

Needs the full workspace, because it drives the Rust engine and the contracts
validators.

```bash
scripts/run_smoke_pipeline.sh
```

It validates contracts, asks Rust for a depth-6 opening book, generates
positions, observations, H2H rows and MCTS self-play rows, converts contract
rows to Parquet where supported, and materializes `.npz` training views.

## Where the data lives

**`runs/` is gitignored** and holds every checkpoint and corpus — nothing in
it can be verified from a fresh clone and nothing in it ships. It is staged
to two Hugging Face dataset repositories instead; see
[`docs/dev-data.md`](docs/dev-data.md) for how to restore it, and
[`docs/corpora.md`](docs/corpora.md) for what each corpus contains.

The 80 KB smoke checkpoint under `tests/fixtures/` is the one exception, and
it is committed precisely so the fixture is exercised rather than rotting.

## Before anything expensive

Training runs and arenas cost hours, and this repository has paid for
skipping this step more than once.

```bash
# ~1 min/arch, projects wall-clock, runs the real code paths
python -m quantik_models.train.preflight --preset medium --epochs 16
```

- **Never plan around a timing taken under load.** It is an upper bound,
  often a wild one: `minimax-d2` was recorded at 1.1 s/move while a solver
  and a sweep saturated the machine, and is actually 0.28 s. A `beam-w16`
  figure was discarded for the same reason. Re-measure on a quiet machine.
- **An average from a differently-shaped workload does not transfer.**
  Measure per bucket.
- Write the projected wall-clock down before starting. If it exceeds a
  night, stop and ask rather than launching.

## Contributing

- **One PR per change**, own branch, merged when CI is green. Don't stack
  open work.
- **Atomic commits.** The message explains *why*; the diff already shows
  what.
- **Commit as the repository owner only.** No `Co-Authored-By:` and no
  `Claude-Session:` trailers.
- Branches: `feat/…`, `fix/…`, `chore/…`, `docs/…`.
- **Focused unit tests for every new behavior** — success, validation and
  error, and any compatibility fallback. An end-to-end smoke test is not a
  substitute.
- Documentation lands in the same PR as the change it describes.
- When you recommend something, write down the alternatives you rejected.

## Cutting a release

Versioning policy, and what counts as a breaking change here, is
[`docs/decisions/0002-versioning-and-release.md`](docs/decisions/0002-versioning-and-release.md).

1. **Bump `__version__`** in `src/quantik_models/__init__.py`. That is the
   only place the number appears — `pyproject.toml` reads it statically.
2. **Move the `Unreleased` section of `CHANGELOG.md`** under the new version
   with today's date. `tests/test_packaging.py` fails if the released
   version has no entry.
3. **Verify locally**, exactly as the release workflow will:

   ```bash
   .venv/bin/python -m pytest -q
   .venv/bin/python -m mypy
   rm -rf dist build
   .venv/bin/python -m build
   .venv/bin/python -m twine check dist/*
   .venv/bin/python scripts/check_dist.py dist/
   ```

   `check_dist.py` is the one that matters most. `twine check` validates that
   the metadata renders and says nothing about the payload, and the payload
   mistake here is one-way: an sdist that picked up `runs/` uploads
   gigabytes of CC-BY-NC weights to PyPI under an MIT package, and **PyPI
   never lets a filename be reused**.
4. **Rehearse against TestPyPI** if anything about the packaging changed:
   run `publish.yml` via `workflow_dispatch` with target `testpypi`, then
   install from there into a clean venv.
5. **Merge, then publish a GitHub Release** tagged `vX.Y.Z`. The release
   event triggers `publish.yml`, which refuses to build if the tag and
   `__version__` disagree, runs the suite and the type check, and uploads
   through **trusted publishing** — there is no long-lived PyPI token stored
   in this repository.

Releasing the *weights* is a separate process with a separate licence; see
[`docs/publishing-to-hugging-face.md`](docs/publishing-to-hugging-face.md).

## Invariants that are not negotiable

- **Game outcomes never become labels.** Only positions travel to the
  corpus, and they get their labels from the exact oracle like everything
  else. This holds for autoplay and for human games.
  [`docs/labeling-strategy.md`](docs/labeling-strategy.md).
- **Contracts are the source of truth.** Schemas live in
  `quantik-core-contracts`.
- **Legality masking lives outside the model, by design.**
- **`tensor-board.v1` is ambiguous — two incompatible encodings share the
  name.** Everything here uses `fastboard.encode_tensors`, which is
  mover-relative. Building to the other one gives a model that plays legally
  and is confidently wrong on half of all positions, with nothing indicating
  a fault. [`docs/models.md`](docs/models.md) has the discriminating fixture.
- **The arena is the ranking that matters.** Held-out accuracy has failed to
  predict play strength four separate times. When the two disagree, say so
  plainly rather than splitting the difference.
- **The seat dwarfs the model.** Mover win rates run 68-88%. Keep the
  side-balancing; never quote an unbalanced win rate.

## Where work is tracked

`quantik-workspace` is the control plane — initiatives under
`tasks/active/QW-NNN/`, repository packets under `context/repositories/`.
