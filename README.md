# quantik-models-py

`quantik-models-py` owns Quantik model training, dataset materialization,
autoplay experiments, checkpoint export, and evaluation. It consumes
`quantik-core-py`, `quantik-core-rust`, and `quantik-core-contracts`; it does
not replace them.

Core libraries stay small and stable:

- `quantik-core-contracts`: artifact IDs, schemas, docs, validators.
- `quantik-core-rust`: search, opening-book generation, observations, H2H,
  self-play producers.
- `quantik-core-py`: artifact readers, QFEN/bitboard/action helpers, tensor
  encoders, checkpoint manifest validation.
- `quantik-models-py`: training views, model architecture, training loops,
  exported checkpoints, calibration reports.

## Clone The Workspace

Choose any parent directory for the Quantik repositories:

```bash
export QUANTIK_NS="$HOME/Code/quantik-ns"
mkdir -p "$QUANTIK_NS"
cd "$QUANTIK_NS"

git clone https://github.com/mberlanda/quantik-core-contracts.git
git clone https://github.com/mberlanda/quantik-core-rust.git
git clone https://github.com/mberlanda/quantik-core-py.git
git clone https://github.com/mberlanda/quantik-models-py.git
```

## Setup

```bash
export QUANTIK_NS="${QUANTIK_NS:-$HOME/Code/quantik-ns}"
export CONTRACTS="$QUANTIK_NS/quantik-core-contracts"
export RUST="$QUANTIK_NS/quantik-core-rust"
export CORE_PY="$QUANTIK_NS/quantik-core-py"
export MODELS="$QUANTIK_NS/quantik-models-py"

cd "$MODELS"
test -d .venv || python -m venv .venv
.venv/bin/python -m pip install -e "${CORE_PY}[arrow]"
.venv/bin/python -m pip install -e ".[dev,arrow]"
```

## Smoke Pipeline

```bash
cd "$MODELS"
scripts/run_smoke_pipeline.sh
```

The script validates contracts, asks Rust to build a depth-6 opening book,
generates positions, observations, H2H rows, and MCTS self-play rows, converts
contract rows to Parquet where supported, and materializes `.npz` training
views.

## CI Data Pipeline

`.github/workflows/e2e-data-pipeline.yml` runs a tiny version of the same flow
on GitHub Actions. It checks out contracts, Rust core, Python core, and this
repository; generates a small book/dataset/observation/H2H/self-play corpus;
materializes training views; verifies the output arrays; and uploads the smoke
artifacts.

## Materialize A Training View

From observations:

```bash
quantik-models-materialize \
  --observations-jsonl /path/to/observations-v1.jsonl \
  --output-npz /path/to/training-view-observations.npz
```

From self-play:

```bash
quantik-models-materialize \
  --selfplay-jsonl /path/to/selfplay-v1.jsonl \
  --output-npz /path/to/training-view-selfplay.npz
```

See `docs/model-report.md`, `docs/pipeline.md`, `docs/tensor-structure.md`,
`docs/labeling-strategy.md`, `docs/autoplay-training.md`, and
`docs/frontend-play.md`.
