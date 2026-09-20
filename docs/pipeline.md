# Pipeline

This repository owns the model-training side of the Quantik pipeline. The
contracts repository remains the source of truth for artifact IDs and field
semantics.

> **Two parts.** Everything down to and including "GitHub Actions proof run"
> describes what the scripts do **today**. [Pipeline profiles](#pipeline-profiles-design)
> after it is a **design, not yet implemented**: it replaces the environment
> variables described above with committed, versioned profiles.

## Clone The Workspace

```bash
export QUANTIK_NS="$HOME/Code/quantik-ns"
mkdir -p "$QUANTIK_NS"
cd "$QUANTIK_NS"

git clone https://github.com/mberlanda/quantik-core-contracts.git
git clone https://github.com/mberlanda/quantik-core-rust.git
git clone https://github.com/mberlanda/quantik-core-py.git
git clone https://github.com/mberlanda/quantik-models-py.git
```

## One-command smoke

```bash
export QUANTIK_NS="${QUANTIK_NS:-$HOME/Code/quantik-ns}"
export CONTRACTS="$QUANTIK_NS/quantik-core-contracts"
export RUST="$QUANTIK_NS/quantik-core-rust"
export CORE_PY="$QUANTIK_NS/quantik-core-py"
export MODELS="$QUANTIK_NS/quantik-models-py"
cd "$MODELS"
scripts/run_smoke_pipeline.sh
```

The script runs:

1. contract validation,
2. Rust depth-6 opening-book generation,
3. Rust position generation using the book for exact references,
4. Rust observation generation across engines,
5. Rust H2H generation and report rendering,
6. Rust MCTS self-play export,
7. Python materialization into `.npz` training views.

Large runs should override counts with environment variables such as
`OPENING_POSITIONS`, `MCTS_ITERATIONS`, `SELFPLAY_GAMES`, and `OUT`.

The chain does not stop at materialization: `examples/train_smoke.sh` wraps
`scripts/run_smoke_pipeline.sh` with two more stages —

8. training the `smoke` preset with `quantik-models-train` against the
   materialized `.npz` views,
9. checkpoint export to `weights.safetensors`, `training-report.json`, and a
   `model-checkpoint.v1` `manifest.json`.

See `docs/scaling-guide.md` for scaling that same trainer invocation from
`smoke` to `small` and `target`.

## GitHub Actions proof run

The `E2E Data Pipeline` workflow runs a tiny proof version of steps 1-7 on
pushes, pull requests, and manual dispatch. The `train-smoke` workflow extends
the same tiny-pipeline configuration through steps 8-9: it runs the smoke
pipeline, trains the `smoke` preset with `quantik-models-train`, inspects the
resulting checkpoint with `examples/inspect_checkpoint.py`, and uploads the
checkpoint as a build artifact. Both workflows intentionally use small counts
and debug Rust builds:

- opening book depth: `1` by default,
- positions: one per phase, with `POSITIONS_USE_BOOK=1` so the searched
  depth-book is read through (and written back to) by position generation,
- engines: `random,minimax`,
- H2H positions/seeds: `1`,
- self-play games: `1`,
- MCTS iterations: `8`.

The goal is not model strength; it is to prove that contracts validation, Rust
data generation, row export, Python materialization, training, checkpoint
export, and artifact verification all still connect end to end. The E2E Data
Pipeline workflow uploads the generated smoke corpus as
`quantik-e2e-data-pipeline`; the train-smoke workflow uploads the trained
checkpoint as `smoke-checkpoint`.

---

# Pipeline profiles (design)

**Status: design, no code.** Nothing below exists yet; the scripts and
workflows above are unchanged. This part is the input to the implementing
work items, and it is written so that a reviewer can disagree with a single
decision without unpicking the rest.

Run tiers are assembled today from 35 environment variables, 21 hardcoded
values and three copies of the same 25-assignment configuration. This part
decides what replaces them. Every decision carries the alternatives that were
rejected and why, because that is the part of a decision record that ages best
(see [`decisions/`](decisions/)).

Citations are `file:line` against the repository at `1d80429`.

## 0. The inventory

The design starts from what the scripts do, because a schema that cannot
express something they do today is a regression. Sources read:
`scripts/run_smoke_pipeline.sh`, `scripts/verify_smoke_outputs.py`,
`scripts/run_patience_lineup.sh`, `src/quantik_models/train/provenance.py`,
`.github/workflows/e2e-data-pipeline.yml`, `.github/workflows/train-smoke.yml`,
plus `examples/train_smoke.sh` and `src/quantik_models/train/trainer.py`,
which the docs above name as part of the chain and which turned out to carry
a third copy of the configuration and a second set of defaults.

One rule splits the inventory in two tiers, and the whole design follows from
it:

- **Knob tier** — changes *what is computed*. Same value, same result (up to
  the time-budget caveat in section 5). These become profile fields.
- **Location tier** — says *where things are or which machine runs them*.
  These are never profile fields: a committed profile that names
  `/Users/someone/Code` is wrong on every other machine, and a `RUN_ID` in a
  profile would make every run of it collide.

### 0.1 Location tier — 10 variables, none become profile fields

`OUT` and `RUN_ID` also get CLI flags (`--out`, `--run-id`). The precedence
for this tier is unchanged from the shell: CLI flag, then environment
variable, then the derived default. These names are **not deprecated** — CI
and the examples set them and there is nothing to migrate them to.

| # | Variable | Where | Destination |
| --- | --- | --- | --- |
| L1 | `QUANTIK_NS` | `run_smoke_pipeline.sh:6`; workflows `e2e:21`, `train-smoke:33` | stays env; root for the four below when they are unset |
| L2 | `CONTRACTS` | `run_smoke_pipeline.sh:8`; `e2e:22`, `train-smoke:34` | stays env |
| L3 | `RUST` | `run_smoke_pipeline.sh:9`; `e2e:23`, `train-smoke:35` | stays env |
| L4 | `CORE_PY` | `run_smoke_pipeline.sh:10`; `e2e:24`, `train-smoke:36` | stays env |
| L5 | `MODELS` | `run_smoke_pipeline.sh:11`; `e2e:25`, `train-smoke:37` | stays env |
| L6 | `RUN_ID` | `run_smoke_pipeline.sh:12` | stays env, plus `--run-id`; recorded in the run record, never in the resolved-profile hash |
| L7 | `OUT` | `run_smoke_pipeline.sh:13`; `e2e:26`, `train-smoke:38`; `examples/train_smoke.sh:16` | stays env, plus `--out` |
| L8 | `PYTHON` | `run_smoke_pipeline.sh:120,124`; `scripts/run_patience_lineup.sh:12` | stays env, plus `--python` |
| L9 | `PYTHONPATH` | inline `run_smoke_pipeline.sh:120,124`; `train-smoke:121,128,158` | stays environment plumbing; the runner sets it for children from L4/L5 |
| L10 | `CKPT` | `examples/train_smoke.sh:17` | superseded by `<OUT>/checkpoint` (see `train-smoke.yml:119`); the example script is rewritten, not migrated |

### 0.2 Knob tier — 25 environment variables, all become profile fields

All are read in `run_smoke_pipeline.sh` with `${VAR:-default}` — an *empty*
variable counts as unset, and the migration preserves that. The defaults in
the "script default" column are the ones baked into the script.

| # | Variable | Script line(s) | Script default | CI value (both workflows) | Profile field |
| --- | --- | --- | --- | --- | --- |
| K1 | `RUST_PROFILE` | 14, validated 16-19 | `release` | `debug` (`e2e:27`) | `rust.build_mode` |
| K2 | `BOOK_DEPTH` | 32, 36 | `6` | `1` | `generate.book.depth` |
| K3 | `OPENING_BOOK_EXTRA_ARGS` | 38 (unquoted, word-split) | empty | `--max-positions 16 --batch-size 16 --quiet` | `generate.book.max_positions`, `.batch_size`, `.quiet` (see 4.3) |
| K4 | `OPENING_POSITIONS` | 43 | `8` | `1` | `generate.positions.opening` |
| K5 | `EARLY_MID_POSITIONS` | 44 | `8` | `1` | `generate.positions.early_mid` |
| K6 | `LATE_MID_POSITIONS` | 45 | `8` | `1` | `generate.positions.late_mid` |
| K7 | `ENDGAME_POSITIONS` | 46 | `8` | `1` | `generate.positions.endgame` |
| K8 | `SOLVE_BUDGET` | 47 | `30` | `0.05` | `generate.positions.solve_budget` |
| K9 | `POSITIONS_USE_BOOK` | 50 (`== "1"`) | `1` | `1` | `generate.positions.use_book` |
| K10 | `ENGINES` | 61 | `mcts,minimax,beam` | `random,minimax` | `generate.observations.engines` |
| K11 | `MCTS_ITERATIONS` | 62 **and** 88 | `512` | `8` | `generate.engines.mcts_iterations` (one field, two call sites) |
| K12 | `MCTS_DEPTH` | 63 and 89 | `16` | `4` | `generate.engines.mcts_depth` |
| K13 | `MINIMAX_DEPTH` | 64 and 90 | `5` | `1` | `generate.engines.minimax_depth` |
| K14 | `MINIMAX_TIME` | 65 and 91 | `0.2` | `0.01` | `generate.engines.minimax_time` |
| K15 | `BEAM_WIDTH` | 66 and 92 | `32` | `4` | `generate.engines.beam_width` |
| K16 | `BEAM_DEPTH` | 67 and 93 | `16` | `4` | `generate.engines.beam_depth` |
| K17 | `OBSERVATION_SEEDS` | 68, and as the fallback at 87 | `2` | `1` | `generate.observations.seeds` |
| K18 | `WORKERS` | 69 and 94 | `1` | `1` | `generate.workers` |
| K19 | `H2H_ENGINES` | 84 | `mcts,minimax` | `random,minimax` | `generate.h2h.engines` |
| K20 | `H2H_POSITIONS` | 85 | `4` | `1` | `generate.h2h.positions` |
| K21 | `H2H_SEEDS` | 86 | `1` | `1` | `generate.h2h.seeds` |
| K22 | `H2H_OBSERVATION_SEEDS` | 87, defaulting to `OBSERVATION_SEEDS` | `2` (via K17) | `1` | `generate.h2h.observation_seeds` |
| K23 | `SELFPLAY_GAMES` | 107 and 113 | `8` | `1` | `generate.selfplay.games` |
| K24 | `SELFPLAY_ITERATIONS` | 108 and 114 | `512` | `8` | `generate.selfplay.iterations` |
| K25 | `SELFPLAY_SEED` | 109 and 115 | `20260713` | `20260713` | `generate.selfplay.seed` |

Two things in that table are worth stating rather than leaving to be noticed.

**The script named "smoke" does not have smoke defaults.** Its defaults
(depth-6 book, 8 positions per phase, 512 MCTS iterations, three engines) are
a much larger run than anything CI does; the tiny values live *only* in the
environment blocks of the two workflows and of `examples/train_smoke.sh:21-29`.
So "the smoke run" is not defined anywhere as one thing — it is defined three
times, by copy.

**The three copies are byte-identical, today.** The `env:` blocks of
`e2e-data-pipeline.yml:20-55` and `train-smoke.yml:32-67` do not differ at
all (`diff` of the two ranges is empty), and `examples/train_smoke.sh:21-29`
repeats the same 25 assignments. They will drift; the seed `20260715` already
lives in three places (`trainer.py:49,267`, `train-smoke.yml:118`,
`examples/train_smoke.sh:39`).

### 0.3 Hardcoded values that are the same kind of thing

Not environment variables, but configuration all the same — and the ones a
profile has to be able to hold, or the lineup and the CI checkpoint step would
regress.

| # | Value | Where | Destination |
| --- | --- | --- | --- |
| H1 | training inputs: the two `.npz` views | `train-smoke.yml:115-116`; `examples/train_smoke.sh:34-35` | derived by the runner from stage outputs; not a field |
| H2 | preset `smoke` | `train-smoke.yml:117`; `examples/train_smoke.sh:36` | `train.preset` |
| H3 | epochs `3` (CI) / `5` (example) / `2` (trainer default, `trainer.py:44,262`) | as listed | `train.epochs` — CI and the example already disagree |
| H4 | batch size `16` (CI, example) / `64` (trainer default, `trainer.py:45,263`) | as listed | `train.batch_size` |
| H5 | seed `20260715` | `train-smoke.yml:118`; `examples/train_smoke.sh:39`; `trainer.py:49,267` | `train.seed` |
| H6 | device `cpu` (CI) / `auto` (example, trainer default `trainer.py:50,268`) | as listed | `train.device` |
| H7 | checkpoint dir `$OUT/checkpoint` (CI) / `$CKPT` (example) / `outputs/checkpoint` (default) | `train-smoke.yml:119`; `examples/train_smoke.sh:17,41`; `trainer.py:51,269` | derived: `<OUT>/checkpoint`, not a field |
| H8 | `inspect_checkpoint.py` invocation and its `--npz` | `train-smoke.yml:125-126`; `examples/train_smoke.sh:44-45` | stage `evaluate` (4.3) |
| H9 | ONNX parity: arch `resnet`, preset `smoke`, sample batch `4`, `atol=1e-5`, input name `board` | `train-smoke.yml:145,147,152-154` | `verify.onnx_parity` fields; arch and preset **derived from `train.*`**, not restated — the workflow hardcodes them independently of the `--preset smoke` on line 117 |
| H10 | lineup corpus `runs/oracle/corpus/exact-sampled.npz` | `run_patience_lineup.sh:13` | `train.corpus` (repo-relative input, identified by hash) |
| H11 | lineup seed `20260828` | `run_patience_lineup.sh:14` | `train.seed` |
| H12 | lineup epochs `60` | `run_patience_lineup.sh:15` | `train.epochs` |
| H13 | lineup patience `5` | `run_patience_lineup.sh:16` | `train.patience` |
| H14 | lineup arch loop `resnet mlp cpool attn` | `run_patience_lineup.sh:18` | **not a field**: one profile is one run; the loop stays in the caller with `--set train.arch=<x>` (see 1.4) |
| H15 | lineup preset `medium` | `run_patience_lineup.sh:23` | `train.preset` |
| H16 | lineup run name `patience-$ARCH`, out root `runs/train` | `run_patience_lineup.sh:21,25` | `--run-id` / `--out`; location tier |
| H17 | selfplay cargo command duplicated for release and debug | `run_smoke_pipeline.sh:104-116` | disappears: `rust.build_mode` selects the flag once |
| H18 | required output files | `verify_smoke_outputs.py:12-21` | verify-internal, derived from which stages ran (3.3) |
| H19 | required npz arrays, `(n, 9, 4, 4)` tensors, 64-wide policy | `verify_smoke_outputs.py:54-62,69,71` | verify-internal; these are contract facts, not knobs |
| H20 | Python `3.12` and pip extras | `e2e:84,94-95`; `train-smoke:96,106-107` | workflow provisioning; stays in the workflows |
| H21 | `train-smoke.yml` never runs `verify_smoke_outputs.py`; `e2e-data-pipeline.yml:110-112` does | both | the `ci` profile turns `verify` on for both (behaviour change; see 3.3) |

**Inventory count: 35 environment variables (10 location, 25 knob) and 21
hardcoded items, every one with a stated destination.**

### 0.4 Deliberately outside the inventory

Named so a reviewer does not have to wonder whether they were missed:

- `QUANTIK_HF_NAMESPACE` (`src/quantik_models/export/huggingface.py:118`) — a
  publishing knob for a separate tool; publishing is not a pipeline stage.
- `OMP_NUM_THREADS` / `MKL_NUM_THREADS` (`src/quantik_models/arena/parallel.py:33-34`)
  — set by the arena's worker pool as an implementation detail, not read from
  the caller.
- The `pytest` step of `e2e-data-pipeline.yml:102-104` — test hygiene that
  runs before the pipeline, not a stage of it. It stays a workflow step.
- `scripts/evaluate_lineup.sh`, `evaluate_opening_arena.sh`, autoplay and the
  arena — evaluation *of trained models by play*. The `evaluate` stage below
  is what exists in the smoke chain today (checkpoint inspection); arena
  evaluation is a later extension of the same schema, not part of this one.

## 1. Profile schema

### 1.1 Decision

A profile is a **TOML file** with a required schema tag, a name that must equal
the file stem, and one table per stage. Every field is typed; **unknown keys
are an error**, not ignored.

```toml
schema      = "quantik.pipeline-profile.v1"   # required
name        = "smoke"                          # required; == file stem
description = "Tiny end-to-end run: proves the stages connect."   # required
extends     = "smoke"                          # optional; one level only

[stages]                      # all required booleans; see section 3
generate    = true
materialize = true
train       = true
evaluate    = true
verify      = true

[rust]
build_mode = "debug"          # "release" | "debug"  (was RUST_PROFILE)

[generate]
workers = 1

[generate.book]
depth = 1
max_positions = 16            # optional; omitted = the tool's own default
batch_size    = 16            # optional
quiet         = true          # optional, default false

[generate.positions]
opening = 1
early_mid = 1
late_mid = 1
endgame = 1
solve_budget = 0.05
use_book = true

[generate.engines]            # shared by observations and h2h, as today
mcts_iterations = 8
mcts_depth = 4
minimax_depth = 1
minimax_time = 0.01
beam_width = 4
beam_depth = 4

[generate.observations]
engines = ["random", "minimax"]
seeds = 1

[generate.h2h]
engines = ["random", "minimax"]
positions = 1
seeds = 1
observation_seeds = 1

[generate.selfplay]
games = 1
iterations = 8
seed = 20260713

[train]
trainer    = "views"          # "views" | "supervised"  (see 1.3)
preset     = "smoke"          # a *model* preset; see 1.2
epochs     = 3
batch_size = 16
seed       = 20260715
device     = "cpu"
# optional, forwarded to the trainer when present: lr, weight_decay,
# value_loss_weight, channels + blocks (together), patience, arch, corpus, ...

[evaluate]
inspect_checkpoint = true

[verify]
outputs     = true            # the verify_smoke_outputs checks
onnx_parity = true            # the inline check in train-smoke.yml:130-158
```

**Required vs optional.** Every field of the `generate` and `materialize`
tables is **required**, in every profile that enables the stage, and there is
**no schema default** for any of them. In `[train]`, the fields the profiles
actually differ on (`trainer`, `preset`, `epochs`, `batch_size`, `seed`,
`device`) are required; the rest are optional and, when absent, the trainer's
own default applies **and the resolved value is written to the run record**
("resolved, not requested" — the rule `train/provenance.py` and
`docs/reproducibility.md` already follow for `lr` and `device`).

**What a profile may not contain.** Any of these is a validation error:

1. Anything from the location tier: absolute paths, `OUT`, `RUN_ID`,
   interpreter, sibling-checkout locations.
2. Secrets or credentials of any kind — profiles are committed.
3. Raw command-line fragments (`extra_args = "--quiet"`). Only structured,
   typed fields.
4. Free-form shell, or anything evaluated.
5. Unknown keys, and unknown table names.
6. A field belonging to a stage the profile disables *and* that no other
   stage needs — a disabled `[generate]` table full of values is a profile
   that lies about what it runs. (Exception: `extends` inheritance, where the
   parent's tables are legitimately present until overridden.)

Repo-relative **input** paths are allowed (`train.corpus`), because an input is
identified by the `sha256` the run records, not by where it was found.

**Where profiles live.** `src/quantik_models/pipeline/profiles/<name>.toml`, as
package data, so an installed wheel carries them; `--profile` also accepts a
path for a profile outside the package. (`pyproject.toml:134-136` lists
package data explicitly and only names `py.typed` and the play app, so the
implementing item must add this line.) `tomllib` is in the standard library on
the `>=3.12` this package requires (`pyproject.toml:24`), so the format costs no
dependency.

### 1.2 The four profiles

`smoke`, `ci`, `small` and `target` all ship. **The names overlap the model
presets and are only nominal** — `train.preset` is an explicit field, and
nothing infers one from the other.

- `smoke` — the values in the `env:` blocks of the workflows today (K1-K25
  "CI value"), `train.preset = "smoke"`, `epochs = 5`, `device = "auto"` (the
  example script's values, H3/H6). What a person runs locally.
- `ci` — `extends = "smoke"`, overriding `train.epochs = 3` and
  `train.device = "cpu"` (H3/H6, the workflows' values), and setting
  `verify.*` on for both workflows (H21). It exists as its own profile so that
  the difference between "local" and "CI" is two named lines rather than an
  environment that happens to differ.
- `small` — `train.preset = "small"` and a corpus sized per
  `scaling-guide.md` (">= 100k rows"); generation counts sized to produce it.
- `target` — `train.preset = "target"`, the depth-6 book generation, large
  self-play. `scaling-guide.md` is the reference for what "target" needs; this
  design fixes the *shape* of the profile, the implementing item fixes the
  numbers and must measure them (see `estimates-dont-survive-reshaping`: an
  average from a different workload is not an estimate).

`medium` — the published lineup size (`policy_value_net.py:34-40`) — has a
model preset but no shipped pipeline profile. The four are the **minimum set,
not a closed set**; the schema imposes no list of names.

### 1.3 Two trainers, one tag

There are two trainers with different flag surfaces and **different defaults
for the same-named flags**: `quantik-models-train` (`trainer.py`, ResNet only,
`--npz` views; epochs 2, batch 64, lr `1e-3`, seed `20260715`) and
`python -m quantik_models.train.supervised` (four architectures, `--corpus`,
`--patience`; epochs 30, batch 1024, `lr` optional, seed `20260827` —
`supervised.py:51-91`). The pipeline uses the first; the lineup uses the
second. `train.trainer = "views" | "supervised"` selects one, and the
trainer-specific optional fields are validated against that trainer's own
config dataclass. The lineup is then expressible as: `trainer = "supervised"`,
`stages.generate = false`, `stages.materialize = false`, `train.corpus = ...`,
`train.preset = "medium"`, `train.epochs = 60`, `train.patience = 5`,
`train.seed = 20260828`.

### 1.4 Rejected

- **A single flat file with a shared defaults table.** It reproduces the
  disease being cured: `run_smoke_pipeline.sh` is exactly "shared defaults
  plus per-caller overrides", and its defaults are not the smoke values (see
  0.2). The defaults were the source of the drift.
- **Schema defaults for `generate.*`.** Same reason, and worse: a default in
  the schema is a value a run can silently take because a profile forgot a
  line. A missing field is an error that names the field.
- **YAML.** PyYAML is not a dependency; adding one to parse six small files is
  the wrong trade. **JSON.** No comments, and this repo's habit of writing the
  reason beside the number (`policy_value_net.py:29-37`) is worth keeping.
- **Multi-level `extends`, or a profile matrix (`arch = [...]`).** A chain of
  three files is the same "where did this value come from" problem as three
  environment layers, without the tooling to answer it; a list-valued `arch`
  would make one profile several runs and break "one profile, one run record".
  One level is enough for `ci` and `smoke`; the lineup loop stays in the
  caller.
- **A free-form `extra_args` string.** It is what `OPENING_BOOK_EXTRA_ARGS`
  is today, it is unvalidated, and it cannot be recorded as fields. If a real
  need appears the answer is a new typed field.
- **Putting `RUST_PROFILE` in the location tier because it is "just a build
  flag".** Rejected in section 5: it changes results.
- **Inferring `train.preset` from the profile name.** `ci` has no preset and
  `medium` has no profile; the coupling would be wrong on day one.

## 2. Override precedence

### 2.1 Decision

For the knob tier, **exactly three layers, lowest to highest**:

1. **The committed profile** (after resolving its one-level `extends`).
2. **Legacy environment variables** — the 25 names of 0.2 only, honoured with
   a warning during the migration window (section 4).
3. **CLI**: `--set <table.key>=<value>`, repeatable, typed by the schema.

Later layers win, per field, atomically. There is **no other environment
namespace** — no `QUANTIK_PIPELINE__…` — and no other override channel: stage
skipping is `--set stages.train=false`, not a separate `--skip` flag.

This preserves what the shell does today. Every knob is `${VAR:-default}`
(`run_smoke_pipeline.sh:43-69`), so the environment already beats the baked-in
value, and a person who exports `MCTS_ITERATIONS=64` and runs the script
already expects 64. The design puts the most specific, most deliberate layer —
the CLI — last, which is also what `argparse` callers expect.

Values from the environment are strings and are coerced by the field's schema
type; a value that does not coerce (`MCTS_ITERATIONS=8.5`) is an error, never a
truncation. List fields take a comma-separated string on both layers
(`ENGINES=random,minimax` is already that). An empty legacy variable is
treated as unset, exactly as `${VAR:-x}` does.

**Every override is visible, twice.** At start the runner prints one line per
non-profile value:

```
generate.engines.mcts_iterations: 256  (cli --set) > 64 (env MCTS_ITERATIONS) > 8 (profile smoke)
```

and the run record stores the same trail (section 5). A conflict is **not an
error** — the higher layer wins and says so — because an error would make
`MCTS_ITERATIONS=64 runner --set …=256` unusable during the migration window.

### 2.2 Worked example: all three layers set one field

The field is `generate.engines.mcts_iterations` (K11). Take `--profile ci`:

| layer | source | value |
| --- | --- | --- |
| 0. profile parent | `smoke.toml` (`extends` target): `mcts_iterations = 8` | `8` |
| 1. committed profile | `ci.toml` does not set it, so the parent's stands | `8` |
| 2. legacy env | `MCTS_ITERATIONS=64` exported in the caller's shell | `"64"` → `64` |
| 3. CLI | `--set generate.engines.mcts_iterations=256` | `256` |

Resolved value: **`256`**. It feeds both call sites that read `MCTS_ITERATIONS`
today (`run_smoke_pipeline.sh:62` and `:88`) because they are one field. The
run record carries:

```json
"resolved": {"generate": {"engines": {"mcts_iterations": 256}}},
"resolution": {
  "generate.engines.mcts_iterations": {
    "value": 256, "source": "cli",
    "trail": [
      {"layer": "profile", "ref": "smoke", "value": 8},
      {"layer": "legacy_env", "name": "MCTS_ITERATIONS", "value": 64},
      {"layer": "cli", "arg": "--set generate.engines.mcts_iterations=256", "value": 256}
    ]
  }
}
```

If the CLI layer is dropped, the same run resolves to `64` and the record says
`"source": "legacy_env"`; with neither, `8` and the field is absent from
`resolution` (profile values are not overrides). A second trap the example
avoids stating in prose: `OBSERVATION_SEEDS` (K17) is the fallback for
`H2H_OBSERVATION_SEEDS` (K22) at `run_smoke_pipeline.sh:87`. In the profile
they are separate required fields. The legacy-env mapper therefore reproduces
the chain: `OBSERVATION_SEEDS=5` with `H2H_OBSERVATION_SEEDS` unset sets **both**
fields to 5, and the trail for `generate.h2h.observation_seeds` names
`OBSERVATION_SEEDS`.

### 2.3 Rejected

- **Environment above CLI.** Would mean a stale `export` in a long-lived shell
  silently defeats an explicit flag. The most recently typed, most specific
  input must win.
- **Error on any cross-layer conflict.** Cleaner, and unusable while the same
  variables are still set in three call sites.
- **A new namespaced env channel (`QUANTIK_PIPELINE__generate__…`)** for CI
  to use in place of the legacy names. Two env channels is one more than can
  be explained; CI should use `--profile ci` and, for a real one-off,
  `--set`.
- **Profile above env** ("profiles are authoritative"). Breaks every existing
  invocation of the form `VAR=x scripts/run_smoke_pipeline.sh`, which is the
  documented way to scale a run ("override counts with environment variables",
  above).
- **Several env vars per field with implicit priority.** One legacy name maps
  to one field; the one shared-fallback case (K17/K22) is handled explicitly.

## 3. Stages

### 3.1 Decision

Five stages, in this order. Each is a boolean under `[stages]`.

| # | Stage | Contains (today's step) | Where it is today | Skippable |
| --- | --- | --- | --- | --- |
| 1 | `generate` | contracts validation; opening book; positions; observations and row export; H2H and game-result rows; MCTS self-play | `run_smoke_pipeline.sh:23-30`, `32-38`, `40-53`, `55-75`, `77-100`, `102-116` | yes |
| 2 | `materialize` | Python materialization of the two `.npz` views | `run_smoke_pipeline.sh:118-127` | yes |
| 3 | `train` | training, checkpoint export (safetensors, ONNX, `manifest.json`, `training-report.json`) | `train-smoke.yml:112-121`; export is called from inside `trainer.py:245-250` | yes |
| 4 | `evaluate` | checkpoint inspection against the self-play view | `train-smoke.yml:123-128`; `examples/train_smoke.sh:43-45` | yes; requires `train` |
| 5 | `verify` | output-set check; ONNX-versus-torch parity | `e2e:110-112` (`verify_smoke_outputs.py`); `train-smoke.yml:130-158` | yes |

"Data generation, materialization, optional training, evaluation,
verification" — the five in the brief — are these five; the nine steps the
docs number above map onto them as steps 1-6 → `generate`, 7 → `materialize`,
8-9 → `train`. Steps 8 and 9 are one stage because export is a call inside
`train()`; there is no seam to skip between them without a code change.

### 3.2 How a profile expresses a skip

`[stages]` lists all five, each `true` or `false`. There is no implicit
default and no auto-detection. Three consequences:

- A disabled stage's outputs must already exist in `--out`, or the runner
  **fails before doing any work**, naming what is missing. This is the
  `preflight.py` ethos (fail in seconds, not after the expensive part).
- The run record marks such a stage `"status": "inherited"` and records the
  digest of the artifact it found and, when the directory has one, the
  `pipeline-run.json` that produced it — so a training-only run on last week's
  corpus is traceable to last week's generation.
- Hard dependencies are validated at load: `evaluate` requires `train`;
  `train` requires `materialize` *or* `train.corpus` (the lineup case);
  `materialize` requires `generate` *or* existing outputs.

`train` is optional in the brief's sense: the data-only tier (`e2e-data-pipeline.yml`
today) is `train = false`, `evaluate = false`.

### 3.3 Verification is new surface, not migration

`verify_smoke_outputs.py` takes exactly one positional argument
(`verify_smoke_outputs.py:83-86`) and hardcodes its required-file list
(`:12-21`) — there is no configuration to migrate. The design adds
`verify.outputs` and `verify.onnx_parity` as **switches only**. What
`outputs` checks is **derived from which stages produced files in this run**
(plus inherited ones): a run with `generate = false` does not demand
`opening-book.sqlite` be *fresh*, but does demand it be present if
`materialize` ran from it. The literal list in `:12-21` becomes a table keyed by
stage. The ONNX check reads architecture and preset from the resolved
`train.*` fields instead of restating `resnet`/`smoke` (H9) — the workflow's
copy is exactly the kind of independent restatement that goes stale when
someone changes `--preset`.

`train-smoke.yml` does not run the output verifier today (H21), so the `ci`
profile makes it run there: a strictly stronger check than CI performs now.

### 3.4 Rejected

- **Splitting `generate` into book / positions / observations / h2h /
  selfplay stages.** They are one script whose steps share `OUT` files
  (`positions-v1.json` feeds three others) and none is useful skipped alone
  except the book (`use_book = false` already covers it). Five stages is
  what the brief asked for; sub-stage toggles are a schema that needs
  dependency validation for no caller.
- **Contracts validation as its own stage.** It guards the generators
  (`run_smoke_pipeline.sh:23-30` runs it against the contracts checkout, not
  against data), so skipping `generate` skips it too; a stage that is always
  on together with another is a step.
- **Treating H2H as `evaluate`.** It reads like evaluation but is generated
  data (game-result rows, `run_smoke_pipeline.sh:77-100`) between *classical
  engines*; no trained model is involved. `evaluate` is what checks the model.
- **Auto-skip when outputs exist**, which `examples/train_smoke.sh:19` does
  (tests for one file). It is convenient and silently reuses a stale corpus
  whose generating profile is no longer the one asked for. The rewrite of that
  script makes the skip explicit (`--set stages.generate=false`), which is a
  behaviour change and is called out as such.
- **`pytest` as a stage.** It runs before the pipeline and tests code, not
  data (0.4).

## 4. Migration

### 4.1 Decision

**A caller that still sets an old variable is honoured, with a warning, and
after a stated removal point the same caller gets a hard error. It is never
ignored.**

- **Mapping**: the `Profile field` column of 0.2 is the mapping. It is
  data, not prose — the implementing item owns it as a single table in code and
  a test asserts that all 25 names are present.
- **The warning**: one line on stderr per variable that took effect,
  `warning: MCTS_ITERATIONS is deprecated; use --set generate.engines.mcts_iterations=64`
  — and the value is recorded in the run record's `resolution` trail as
  `legacy_env`, so a result never depends on an invisible environment.
- **Removal point**: the first release after both workflows **and**
  `examples/train_smoke.sh` stop setting the variables (that is, after the
  call sites are migrated to `--profile`). From then on any of the 25 names
  set in the environment is an error naming the replacement field. This design
  does not put a date on it; the migrating item states the release.
- **Semantics preserved in the mapper** (each is a place the old shell would
  otherwise silently differ): empty = unset (`${VAR:-…}`); `POSITIONS_USE_BOOK`
  is true only for the literal `"1"` (`run_smoke_pipeline.sh:50`), anything
  else false; `ENGINES` and `H2H_ENGINES` are comma lists; the K17→K22
  fallback of 2.2.
- **`OPENING_BOOK_EXTRA_ARGS`** is a free-form string and has no equal in a
  typed schema. The mapper accepts exactly the three flags CI uses —
  `--max-positions N`, `--batch-size N`, `--quiet` — and maps them to
  `generate.book.max_positions`, `.batch_size`, `.quiet`. Any other token is a
  **hard error naming the token**, even during the window: honouring an
  argument the schema cannot record would produce a run whose record does not
  describe it.
- **Location-tier names** (0.1) are unchanged and not warned about.

### 4.2 Why not ignore

The variables are load-bearing in currently-green CI: both workflows set all
25 (`e2e:27-55`, `train-smoke:39-67`) and `examples/train_smoke.sh:21-29`
sets them a third time. If the new runner ignored them, those three call
sites would silently revert to the script defaults — depth-6 book, 8 positions
per phase, 512 MCTS iterations, `release` build — before the migrating item
lands. Nothing would fail; a 30-minute-timeout job would simply become a much
longer one, or time out. The only ignore-outright behaviour that is safe is
one that fails loudly, which is what the post-window error is.

### 4.3 Rejected

- **Ignore, with a note in the changelog.** Above. It converts a documentation
  problem into silent wrong runs, the exact failure this design exists to
  prevent.
- **Honour forever.** Keeps a second override channel alive that has to be
  explained and tested in perpetuity, and leaves precedence permanently
  three-layered where two would do.
- **Hard error from day one.** Correct as an end state, and considered: it
  forces the three call sites to migrate in the same change as the runner. It
  is rejected only because it couples two independently reviewable changes;
  the window is the price of not doing that, and it is bounded.
- **Passing unknown `OPENING_BOOK_EXTRA_ARGS` tokens through anyway.** See
  above: the record would not describe the run.

## 5. Reproducibility metadata

### 5.1 What already exists, and its level

`train/provenance.py` records one *training run*: `capture(corpus=, device=)`
returns `training-provenance.v1` with `code` (commit, dirty, branch, remote,
`commit_url`, repository root), `hardware` (the device *used*), `versions`
(python, torch, numpy, quantik-core, quantik-models, onnxruntime) and
`corpus` (path, resolved path, `sha256`, size)
(`provenance.py:152-172`). `supervised.py:178-179` and `alphazero.py:189-191`
write it to `provenance.json`.

Two facts about that mechanism shape the design:

1. **It records the models repository only.** `code_provenance(root=…)` takes
   a root (`provenance.py:78`), defaulting to this repo. The data was
   generated by `quantik-core-rust`, validated by `quantik-core-contracts` and
   read through `quantik-core-py`; none of their commits is recorded anywhere
   today.
2. **`quantik-models-train` does not call it.** `trainer.py` has no
   `provenance` import; only the `supervised` and `alphazero` trainers do. The
   smoke checkpoint that CI uploads (`smoke-checkpoint`, `train-smoke.yml:160-164`)
   therefore carries a `training-report.json` but **no `provenance.json`** —
   no commit, no hardware, no versions, no corpus hash.

### 5.2 Decision: a run record one level up, that references and never restates

A pipeline run writes **`<OUT>/pipeline-run.json`**, schema `pipeline-run.v1`,
containing:

- `run_id`, `started_at`, `finished_at` (UTC ISO, as `captured_at` is).
- `profile`: name, source path, `file_sha256`, and the same for its `extends`
  parent.
- `resolved`: the full resolved knob-tier configuration after all layers, and
  `resolved_sha256` — the digest of its canonical JSON (sorted keys). Two runs
  with equal `resolved_sha256` were configured identically regardless of how
  the layers got them there. The location tier is **excluded** from this hash
  by construction, since it is not in the resolved profile.
- `resolution`: for each field a non-profile layer changed, the trail shown in
  2.2; the verbatim `argv`; the names of legacy variables that took effect.
- `stages`: per stage, `status` (`ran`, `skipped`, `inherited`), the reason,
  timestamps, and `file_digest` (from `provenance.py:99`) of each output.
  `inherited` carries the digest of what was found and the source run record's
  hash.
- `code`: one `code_provenance(root)` result for each of models, contracts,
  rust, core-py (four calls to the existing function, with the four roots
  from the location tier), plus the contracts `VERSION` file content.
- `training`: a **reference** — the path to the trainer's `provenance.json`
  and its `sha256` — not a copy of it. Code, hardware and versions of the
  *training* live where `provenance.py` already puts them; the run record does
  not recompute them.
- `environment`: the resolved location tier (paths, interpreter). Present for
  a human debugging a run, and excluded from `resolved_sha256`.
- `nondeterministic_inputs`: the resolved fields that are wall-clock budgets
  (below), so a reader knows which of two "identical" runs may legitimately
  differ.

**How it relates to `provenance.py`, in one sentence:** the pipeline record
answers *which profile and overrides, over which repositories, produced which
files*; `provenance.json` answers *what one training run ran on*; the first
points at the second and reuses its functions (`code_provenance`,
`file_digest`) rather than defining any.

### 5.3 Two gaps the design surfaces, and its position on each

- **The `views` trainer records no provenance (5.1, fact 2).** The fix belongs
  in `trainer.py` — one `capture_provenance(...)` call mirroring
  `supervised.py:178-179` — **not** in the pipeline runner. A runner that
  captured on the trainer's behalf would create two writers for one file and
  leave `quantik-models-train` run by hand exactly as unrecorded as now. This
  is a code change outside this document; it needs its own work item and the
  design depends on it. Until it lands, `training.provenance` is `null` with a
  `reason`, in the style of `provenance.py`'s own missing-field convention.
- **`capture(corpus=…)` takes one path; the `views` trainer takes several
  `.npz`.** Until `capture` learns a list, the pipeline record's per-output
  digests (`stages.materialize.outputs`) are the complete corpus identity and
  `training.provenance.corpus` is the partial one. Extending `capture` is
  preferred to a second digest mechanism.

### 5.4 What this does not make reproducible

Say it in the record rather than let it be discovered. `SOLVE_BUDGET` (K8) and
`MINIMAX_TIME` (K14) are **wall-clock budgets**: on a slower or busier machine a
search reaches a shallower depth in the allotted time and returns different
rows. A profile fixes the *configuration* exactly; it does not make generated
data bit-identical across machines. `nondeterministic_inputs` lists them, and
`hardware` (from `provenance.py`) is what a reader compares. The same
discipline applies to timings taken under load (`DEVELOPMENT.md`): a number
measured that way is an upper bound.

This is also why `RUST_PROFILE` (K1) is in the profile as `rust.build_mode`,
**not** left in the location tier as a build detail. Debug and release builds
differ in speed by an order of magnitude, and under a time budget speed changes
results. It is renamed because *profile* now means this document's noun and the
old name would collide with it; the legacy name still maps to it.

### 5.5 Rejected

- **Extending `training-provenance.v1` into the run record.** The training
  record is written by the trainer, per run, next to the checkpoint that leaves
  the machine (`reproducibility.md`); a pipeline-level record that lives in
  `OUT` has a different lifetime and audience. One schema doing both would
  force the checkpoint to carry the whole pipeline's paths.
- **Recording only the profile name.** A name is not an identity; a profile
  file changes. This is the same argument `provenance.py` makes for corpora
  (a filename is not an identity, a hash is), applied to the config.
- **Hashing the environment or the locations into the identity.** It would make
  identical configurations on two machines look different for reasons that do
  not affect the result.
- **Recording the commit of the models repo only.** The data path runs through
  three other repositories; recording one is the current gap, not a design.
- **Runner-side provenance capture for the `views` trainer.** See 5.3.

## Decided without an obvious right answer

The short list a reviewer should spend their attention on. Everything else in
this part follows from the inventory.

1. **Legacy variables are honoured with a warning, then become a hard error;
   never ignored (4.1).** The removal point is "after the three in-repo call
   sites migrate", not a date or a version. If you would rather couple the
   runner and the call-site change and skip the window, 4.3 has the argument.
2. **No schema defaults for `generate.*`; every field required (1.4).** Makes
   profiles verbose and a forgotten field a hard error. The alternative is
   less to write and reintroduces the drift in 0.2.
3. **One-level `extends` (1.1, 1.4).** It is what lets `ci` be two lines
   rather than a fourth copy. It is also a second place a value can come from;
   the trail in the run record is what makes that tolerable.
4. **`RUST_PROFILE` is a knob, not a location, and is renamed `build_mode`
   (5.4).** Rests on the claim that time-budgeted search makes debug and
   release results differ. That is reasoning from `SOLVE_BUDGET`/`MINIMAX_TIME`,
   not a measurement. `WORKERS` is assumed **non-semantic** (it changes speed,
   not rows) and is in the profile only because it is not a location; if it
   turns out to change output ordering it needs a place in the identity, and
   that has not been verified.
5. **Stage boundaries (3.1, 3.4).** Contracts validation lives inside
   `generate`; H2H is `generate`, not `evaluate`; export is inside `train`;
   `evaluate` is checkpoint inspection only because nothing else evaluates in
   the smoke chain today. Any of these could reasonably be drawn differently.
6. **Explicit skip replaces `examples/train_smoke.sh:19`'s auto-skip (3.4).**
   A behaviour change for anyone who relied on "reuse if the file is there".
7. **The lineup is expressed by `trainer = "supervised"`, not by folding the
   two trainers together (1.3).** The two have different defaults for the same
   flag names; unifying them is a much larger question than this one. It also
   leaves `medium` without a shipped profile (1.2).
8. **The `views` trainer's missing provenance is a fix in `trainer.py`, not in
   the runner (5.3).** This design depends on a code change it cannot make.
9. **Location-tier names are not deprecated (0.1)** — the rule that splits the
   tiers is "knob changes the result, location does not", and `RUST_PROFILE`
   is the one case where that rule and the variable's name disagree.
10. **Format and location: TOML, in package data (1.1).** Requires a
    `pyproject.toml` package-data edit by the implementing item; a
    profile-file-outside-the-package escape hatch (`--profile PATH`) is
    included because the lineup profile is not something to ship in a wheel.

## Not decided here

Numbers for `small` and `target` (they need measuring, not designing); the
arena evaluation stage; the exact removal release for the legacy names;
whether `capture` should take a list of corpora (5.3); and the flag surface of
the runner beyond `--profile`, `--set`, `--out`, `--run-id`, `--python`.
