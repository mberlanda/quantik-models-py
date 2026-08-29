#!/usr/bin/env bash
# Regenerate every number the architecture comparison rests on.
#
# ADR 0001 requires the comparison to be reproducible, and it had not been:
# the shift evaluation, the arena at three start depths, and the MCTS arena
# were each run by hand, so restating a margin meant remembering four
# invocations and their flags. This is those invocations.
#
# It exists because the margins genuinely needed restating. `cpool` was
# trained at 2e-3 — a rate inherited from the ResNet — and prefers 6e-4 by
# 1.6 points of validation top-1, so every comparison involving it was
# measured against an understated model.
#
#   scripts/evaluate_lineup.sh runs/eval/2026-08-29 \
#     resnet=runs/train/lineup-resnet/best \
#     cpool=runs/train/swept-cpool/best \
#     mlp=runs/train/lineup-mlp/best
#
# Writes into the output directory and prints nothing it does not write.
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "usage: $0 OUT_DIR NAME=CHECKPOINT [NAME=CHECKPOINT ...]" >&2
  exit 2
fi

OUT="$1"; shift
PYTHON="${PYTHON:-.venv/bin/python}"
# Games per ordered pairing. The published arena used 400 at ply 3 and 300
# at plies 6 and 9; 300 keeps every depth comparable to the others.
GAMES="${GAMES:-300}"
MCTS_SIMS="${MCTS_SIMS:-128}"
# Arena seed. Deliberately settable, and deliberately not a training seed:
# the training runs used 20260827, 20260828 and 20260901, and reusing one
# here would make any seed-linked bias invisible rather than absent. Vary it
# across runs and compare.
SEED="${SEED:-20260829}"

mkdir -p "$OUT"
AGENTS_POLICY="$OUT/agents-policy.json"
AGENTS_MCTS="$OUT/agents-mcts.json"

# Build the agent specs from the NAME=CHECKPOINT arguments rather than
# keeping a checked-in JSON per lineup: a stale spec pointing at a
# superseded checkpoint is exactly how a comparison silently measures the
# wrong model.
{
  printf '['
  sep=""
  for pair in "$@"; do
    name="${pair%%=*}"; ckpt="${pair#*=}"
    [ -d "$ckpt" ] || { echo "no such checkpoint: $ckpt" >&2; exit 1; }
    printf '%s\n  {"kind": "net-policy", "checkpoint": "%s", "device": "cpu", "name": "%s"}' \
      "$sep" "$ckpt" "$name"
    sep=","
  done
  printf '\n]\n'
} > "$AGENTS_POLICY"

{
  printf '['
  sep=""
  for pair in "$@"; do
    name="${pair%%=*}"; ckpt="${pair#*=}"
    printf '%s\n  {"kind": "net-mcts", "checkpoint": "%s", "device": "cpu", "name": "%s-mcts%s",' \
      "$sep" "$ckpt" "$name" "$MCTS_SIMS"
    printf ' "params": {"simulations": %s, "leaf_batch": 32, "dirichlet_weight": 0.0}}' "$MCTS_SIMS"
    sep=","
  done
  # The control, without which "these networks are close" cannot be told
  # apart from "the search is doing all the work".
  printf '%s\n  {"kind": "uniform-mcts", "name": "uniform-mcts%s",' "$sep" "$MCTS_SIMS"
  printf ' "params": {"simulations": %s, "leaf_batch": 32, "dirichlet_weight": 0.0}}\n]\n' "$MCTS_SIMS"
} > "$AGENTS_MCTS"

echo "== shift evaluation =="
$PYTHON -m quantik_models.eval.shift \
  $(for pair in "$@"; do printf -- '--checkpoint %s ' "${pair#*=}"; done) \
  --out "$OUT/shift.json" | tee "$OUT/shift.md"

# Three start depths, because the arena ranking depends on where play
# starts: the ResNet leads from ply 3 and `cpool` from ply 6, and a result
# at one depth says little about another.
for ply in 3 6 9; do
  echo "== arena, net-policy, start ply $ply =="
  $PYTHON -m quantik_models.arena.autoplay \
    --agents "$AGENTS_POLICY" --games "$GAMES" --start-plies "$ply" \
    --out "$OUT/policy-p$ply" --seed "$SEED" | grep -v '^  [0-9]*/'
done

for ply in 3 6; do
  echo "== arena, net-mcts $MCTS_SIMS sims, start ply $ply =="
  $PYTHON -m quantik_models.arena.autoplay \
    --agents "$AGENTS_MCTS" --games "$GAMES" --start-plies "$ply" \
    --out "$OUT/mcts-p$ply" --seed "$SEED" | grep -v '^  [0-9]*/'
done

echo
echo "wrote $OUT"
