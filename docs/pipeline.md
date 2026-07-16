# Pipeline

This repository owns the model-training side of the Quantik pipeline. The
contracts repository remains the source of truth for artifact IDs and field
semantics.

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
