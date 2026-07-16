#!/usr/bin/env bash
# End-to-end smoke demo: generate a tiny data corpus with the existing
# pipeline, train the `smoke` preset for a few epochs, export a
# model-checkpoint.v1 checkpoint, and inspect it.
#
# Prereqs:
#   - sibling checkouts per README (quantik-core-contracts/-rust/-py)
#   - pip install -e .[dev,torch]
#
# Usage:
#   examples/train_smoke.sh                 # generates tiny data first
#   OUT=/path/to/existing-data examples/train_smoke.sh   # reuse data
set -euo pipefail

MODELS="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-$MODELS/outputs/smoke-demo-data}"
CKPT="${CKPT:-$MODELS/outputs/smoke-demo-checkpoint}"

if [[ ! -f "$OUT/training-view-observations.npz" ]]; then
  echo "== generating tiny pipeline data into $OUT =="
  OUT="$OUT" RUST_PROFILE=debug BOOK_DEPTH=1 POSITIONS_USE_BOOK=1 \
  OPENING_BOOK_EXTRA_ARGS='--max-positions 16 --batch-size 16 --quiet' \
  OPENING_POSITIONS=1 EARLY_MID_POSITIONS=1 LATE_MID_POSITIONS=1 \
  ENDGAME_POSITIONS=1 SOLVE_BUDGET=0.05 ENGINES=random,minimax \
  OBSERVATION_SEEDS=1 MINIMAX_DEPTH=1 MINIMAX_TIME=0.01 \
  MCTS_ITERATIONS=8 MCTS_DEPTH=4 BEAM_WIDTH=4 BEAM_DEPTH=4 WORKERS=1 \
  H2H_ENGINES=random,minimax H2H_POSITIONS=1 H2H_SEEDS=1 \
  H2H_OBSERVATION_SEEDS=1 SELFPLAY_GAMES=1 SELFPLAY_ITERATIONS=8 \
  SELFPLAY_SEED=20260713 "$MODELS/scripts/run_smoke_pipeline.sh"
fi

echo "== training smoke preset =="
quantik-models-train \
  --npz "$OUT/training-view-observations.npz" \
  --npz "$OUT/training-view-selfplay.npz" \
  --preset smoke \
  --epochs 5 \
  --batch-size 16 \
  --seed 20260715 \
  --device auto \
  --out-dir "$CKPT"

echo "== inspecting checkpoint =="
python "$MODELS/examples/inspect_checkpoint.py" "$CKPT" \
  --npz "$OUT/training-view-selfplay.npz"

echo "checkpoint: $CKPT"
