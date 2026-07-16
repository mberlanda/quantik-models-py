#!/usr/bin/env bash
# Train the `small` preset (64 channels x 4 blocks, single-digit MB) on
# a locally generated corpus. Meant for a laptop: a few minutes on
# Apple Silicon (MPS) or modern x86 CPU for the default settings.
#
# The smoke corpus is far too small for this preset to be meaningful;
# generate a bigger dataset first (see docs/scaling-guide.md), e.g.:
#
#   OUT=$PWD/outputs/small-corpus \
#   OPENING_POSITIONS=64 EARLY_MID_POSITIONS=64 LATE_MID_POSITIONS=96 \
#   ENDGAME_POSITIONS=64 MCTS_ITERATIONS=512 SELFPLAY_GAMES=32 \
#   SELFPLAY_ITERATIONS=256 scripts/run_smoke_pipeline.sh
#
# Usage:
#   DATA=$PWD/outputs/small-corpus examples/train_small_local.sh
set -euo pipefail

MODELS="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${DATA:?set DATA to a directory containing training-view-*.npz}"
CKPT="${CKPT:-$MODELS/outputs/small-checkpoint}"

quantik-models-train \
  --npz "$DATA/training-view-observations.npz" \
  --npz "$DATA/training-view-selfplay.npz" \
  --preset small \
  --epochs 20 \
  --batch-size 128 \
  --lr 5e-4 \
  --seed 20260715 \
  --device auto \
  --out-dir "$CKPT"

python "$MODELS/examples/inspect_checkpoint.py" "$CKPT" \
  --npz "$DATA/training-view-selfplay.npz"
