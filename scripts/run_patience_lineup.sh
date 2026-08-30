#!/usr/bin/env bash
# Train all four lineup architectures to convergence under --patience,
# on the published lineup's own corpus (exact-sampled.npz) so this is a
# budget change and nothing else. See briefs/lineup-under-patience.md.
#
# patience-cpool-v2 and -v3 are NOT reused: they trained on a different
# corpus (exact-sampled-v2.npz / -v3.npz) than the published lineup, which
# would confound corpus with budget in the one arm this exercise exists to
# isolate. See the brief's 2026-08-30 correction.
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
CORPUS="runs/oracle/corpus/exact-sampled.npz"
SEED=20260828
EPOCHS=60
PATIENCE=5

for ARCH in resnet mlp cpool attn; do
  echo "=== $(date -u +%FT%TZ) starting $ARCH ==="
  "$PYTHON" -m quantik_models.train.supervised \
    --name "patience-$ARCH" \
    --corpus "$CORPUS" \
    --arch "$ARCH" --preset medium \
    --epochs "$EPOCHS" --patience "$PATIENCE" \
    --seed "$SEED" --out runs/train
  echo "=== $(date -u +%FT%TZ) finished $ARCH ==="
done

echo "=== $(date -u +%FT%TZ) ALL FOUR DONE ==="
