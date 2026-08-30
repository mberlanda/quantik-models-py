#!/usr/bin/env bash
# The arena from the position a human game actually starts in: the empty board.
#
# Every arena on disk starts at ply 3, 4, 5, 6 or 9. Nothing has ever been
# measured from ply 0, so the phase a player opens in is unmeasured for every
# checkpoint in the family — which is why the skill-level mapping (QW-010) is
# blocked on this run.
#
# THE TRAP THIS SCRIPT EXISTS TO AVOID
#
# Network agents are deterministic at the default temperature, and ply 0 has
# exactly ONE start position. So a ply-0 arena between two deterministic agents
# replays the same game every time. Measured: 8 games, `distinct games: worst
# pairing 1/8`, and a 100.0% leaderboard that is one game's result reported as
# sixteen. With `temperature 1.0` over the first 4 plies the same pairing gives
# 40/40 distinct and 52.5% — a different answer, and a real one.
#
# Two details that cost a run each to find:
#
#   * `temperature` and `temperature_plies` are TOP-LEVEL spec keys. Putting
#     them under `params` raises `PolicyAgent.__init__() got an unexpected
#     keyword argument 'params'` for net-policy, and is silently accepted as an
#     MCTSParams field for net-mcts.
#   * `dirichlet_weight` alone does NOT restore diversity. Measured with
#     `dirichlet_weight 0.25` and no temperature: still 1/12 distinct.
#
#   scripts/evaluate_opening_arena.sh runs/eval/opening-2026-08-30 \
#     cpool=runs/train/swept-cpool/best \
#     attn=runs/train/swept-attn/best \
#     resnet=runs/train/lineup-resnet/best \
#     mlp=runs/train/lineup-mlp/best \
#     patience-cpool=runs/train/patience-cpool/best \
#     patience-cpool-v2=runs/train/patience-cpool-v2/best
#
# Writes into the output directory and prints nothing it does not write.
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "usage: $0 OUT_DIR NAME=CHECKPOINT [NAME=CHECKPOINT ...]" >&2
  exit 2
fi

OUT="$1"; shift
PYTHON="${PYTHON:-.venv/bin/python}"
# Games per ordered pairing. Six agents is 30 ordered pairings.
GAMES="${GAMES:-300}"
MCTS_SIMS="${MCTS_SIMS:-128}"
# Not 20260829 (epoch-test) and not 20260909 (lineup): both are spent, and
# reusing one makes a seed-linked bias invisible rather than absent.
SEED="${SEED:-20261001}"
# The opening sampling window. 1.0 over the first 4 plies was measured to take
# a policy pairing from 1/8 distinct games to 40/40. Raising TEMP_PLIES samples
# deeper into the game and measures less of the opening; lowering it toward 1
# collapses back toward the single start position.
TEMP="${TEMP:-1.0}"
TEMP_PLIES="${TEMP_PLIES:-4}"
# Start plies. 0 is the point of this script; 1 is included because ply 0 has
# one position and ply 1 has three, so the two answer slightly different
# questions about how early the differences appear.
START_PLIES="${START_PLIES:-0 1}"

mkdir -p "$OUT"
AGENTS_POLICY="$OUT/agents-policy.json"
AGENTS_MCTS="$OUT/agents-mcts.json"

# Built from the arguments rather than checked in: a stale spec pointing at a
# superseded checkpoint is how a comparison silently measures the wrong model.
{
  printf '['
  sep=""
  for pair in "$@"; do
    name="${pair%%=*}"; ckpt="${pair#*=}"
    [ -d "$ckpt" ] || { echo "no such checkpoint: $ckpt" >&2; exit 1; }
    printf '%s\n  {"kind": "net-policy", "checkpoint": "%s", "device": "cpu", "name": "%s",' \
      "$sep" "$ckpt" "$name"
    printf ' "temperature": %s, "temperature_plies": %s}' "$TEMP" "$TEMP_PLIES"
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
    printf ' "temperature": %s, "temperature_plies": %s,' "$TEMP" "$TEMP_PLIES"
    printf ' "params": {"simulations": %s, "leaf_batch": 32, "dirichlet_weight": 0.25}}' "$MCTS_SIMS"
    sep=","
  done
  # The control: the same search with uniform priors and a value of zero.
  # Without it "these networks are close at ply 0" cannot be told apart from
  # "no network knows anything at ply 0", which is the live hypothesis here —
  # every checkpoint is uniform to three decimals on the empty board.
  printf '%s\n  {"kind": "uniform-mcts", "name": "uniform-mcts%s",' "$sep" "$MCTS_SIMS"
  printf ' "temperature": %s, "temperature_plies": %s,' "$TEMP" "$TEMP_PLIES"
  printf ' "params": {"simulations": %s, "leaf_batch": 32, "dirichlet_weight": 0.25}}\n]\n' "$MCTS_SIMS"
} > "$AGENTS_MCTS"

for ply in $START_PLIES; do
  echo "== arena, net-policy, start ply $ply =="
  $PYTHON -m quantik_models.arena.autoplay \
    --agents "$AGENTS_POLICY" --games "$GAMES" --start-plies "$ply" \
    --out "$OUT/policy-p$ply" --seed "$SEED" | grep -v '^  [0-9]*/'
done

for ply in $START_PLIES; do
  echo "== arena, net-mcts $MCTS_SIMS sims, start ply $ply =="
  $PYTHON -m quantik_models.arena.autoplay \
    --agents "$AGENTS_MCTS" --games "$GAMES" --start-plies "$ply" \
    --out "$OUT/mcts-p$ply" --seed "$SEED" | grep -v '^  [0-9]*/'
done

echo
echo "wrote $OUT"
echo
echo "CHECK THIS BEFORE READING ANY WIN RATE:"
echo "  every run above prints 'distinct games: worst pairing N/$GAMES'."
echo "  If N is much below $GAMES, the rates rest on fewer independent games"
echo "  than were played and every interval is too narrow. Raise TEMP or"
echo "  TEMP_PLIES and rerun; do not report the numbers as they stand."
