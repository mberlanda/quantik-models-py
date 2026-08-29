#!/usr/bin/env bash
# Play the network field against quantik-core's minimax, and pack the result.
#
# Every win rate this project publishes is against another network or against
# the uniform-prior control. That answers "which of these is better" and says
# nothing about "are any of these good", because the floor moves with the
# field. This is the fixed opponent: an exact classical engine at a fixed
# depth, which is the same player in every run.
#
#   scripts/oracle_benchmark.sh runs/eval/oracle-today \
#     cpool=runs/train/swept-cpool/best \
#     attn=runs/train/swept-attn/best
#
# Depth 2 is the affordable oracle and the choice is measured, not assumed —
# see docs/oracle-benchmark.md for the per-move timings. Depth 4 costs 16 s a
# move from a ply-3 start, which is three orders of magnitude off the budget.
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "usage: $0 OUT_DIR NAME=CHECKPOINT [NAME=CHECKPOINT ...]" >&2
  exit 2
fi

OUT="$1"; shift
PYTHON="${PYTHON:-.venv/bin/python}"
GAMES="${GAMES:-1000}"
DEPTH="${MINIMAX_DEPTH:-2}"
# Three seeds and two start depths. One seed cannot separate a result from a
# seed-linked bias, which is the whole reason the oracle is fixed — and none
# of these is a training seed (training used 20260827, 20260828, 20260901).
SEEDS="${SEEDS:-20260902 20260903 20260904}"
START_PLIES="${START_PLIES:-3}"
# `${EXTRA_PLY-6}`, not `:-`: passing EXTRA_PLY= is the documented way to
# skip the second depth, and `:-` treats an explicit empty string as unset,
# so the run happened anyway.
EXTRA_PLY="${EXTRA_PLY-6}"

mkdir -p "$OUT"
AGENTS="$OUT/agents.json"

{
  printf '['
  for pair in "$@"; do
    name="${pair%%=*}"; ckpt="${pair#*=}"
    [ -d "$ckpt" ] || { echo "no such checkpoint: $ckpt" >&2; exit 1; }
    printf '\n  {"kind": "net-policy", "checkpoint": "%s", "device": "cpu", "name": "%s"},' \
      "$ckpt" "$name"
  done
  # `time_limit_s: null` selects the fixed-depth engine. The fixed-*clock*
  # configuration is not usable as an oracle: iterative deepening cannot
  # interrupt a level, so a 10 ms budget spends 157 ms from a ply-3 start and
  # reaches exactly the depth this does. A budget the engine ignores is not a
  # budget.
  printf '\n  {"kind": "minimax", "time_limit_s": null, "max_depth": %s, "name": "minimax-d%s"}\n]\n' \
    "$DEPTH" "$DEPTH"
} > "$AGENTS"

RUNS=()
for seed in $SEEDS; do
  for ply in $START_PLIES; do
    echo "== seed $seed, start ply $ply =="
    $PYTHON -m quantik_models.arena.autoplay \
      --agents "$AGENTS" --games "$GAMES" --start-plies "$ply" \
      --against "minimax-d$DEPTH" --seed "$seed" \
      --out "$OUT/s$seed-p$ply" | grep -v '^  [0-9]*/'
    RUNS+=("$OUT/s$seed-p$ply")
  done
done

if [ -n "$EXTRA_PLY" ]; then
  seed="${SEEDS%% *}"
  echo "== seed $seed, start ply $EXTRA_PLY =="
  $PYTHON -m quantik_models.arena.autoplay \
    --agents "$AGENTS" --games "$GAMES" --start-plies "$EXTRA_PLY" \
    --against "minimax-d$DEPTH" --seed "$seed" \
    --out "$OUT/s$seed-p$EXTRA_PLY" | grep -v '^  [0-9]*/'
  RUNS+=("$OUT/s$seed-p$EXTRA_PLY")
fi

$PYTHON -m quantik_models.arena.pack "$OUT/packed" "${RUNS[@]}"
