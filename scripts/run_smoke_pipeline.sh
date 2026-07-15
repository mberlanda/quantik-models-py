#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DEFAULT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
QUANTIK_NS="${QUANTIK_NS:-$(CDPATH= cd -- "$MODELS_DEFAULT/.." && pwd)}"

CONTRACTS="${CONTRACTS:-$QUANTIK_NS/quantik-core-contracts}"
RUST="${RUST:-$QUANTIK_NS/quantik-core-rust}"
CORE_PY="${CORE_PY:-$QUANTIK_NS/quantik-core-py}"
MODELS="${MODELS:-$MODELS_DEFAULT}"
RUN_ID="${RUN_ID:-smoke-$(date +%Y%m%d-%H%M%S)}"
OUT="${OUT:-$MODELS/outputs/$RUN_ID}"

mkdir -p "$OUT"

echo "== validate contracts =="
cd "$CONTRACTS"
python3 scripts/validate_contracts.py \
  --manifest contracts.json \
  --version-file VERSION \
  --schema-glob 'schemas/*.json' \
  --fixture-glob 'fixtures/**/*.jsonl' \
  --expected-release "$(cat VERSION)"

echo "== opening book depth 6 =="
cd "$RUST"
scripts/generate_opening_book.sh search \
  --depth "${BOOK_DEPTH:-6}" \
  --db "$OUT/opening-book.sqlite" \
  ${OPENING_BOOK_EXTRA_ARGS:-}

echo "== positions with book references =="
scripts/generate_positions.sh \
  --opening "${OPENING_POSITIONS:-8}" \
  --early-mid "${EARLY_MID_POSITIONS:-8}" \
  --late-mid "${LATE_MID_POSITIONS:-8}" \
  --endgame "${ENDGAME_POSITIONS:-8}" \
  --solve-budget "${SOLVE_BUDGET:-30}" \
  --book "$OUT/opening-book.sqlite" \
  --output "$OUT/positions-v1.json"

echo "== observations =="
scripts/generate_observations.sh \
  --dataset "$OUT/positions-v1.json" \
  --output "$OUT/observations-bundle.json" \
  --checkpoint-dir "$OUT/observations-ckpt" \
  --engines "${ENGINES:-mcts,minimax,beam}" \
  --mcts-iterations "${MCTS_ITERATIONS:-512}" \
  --minimax-depth "${MINIMAX_DEPTH:-5}" \
  --beam-width "${BEAM_WIDTH:-32}" \
  --seeds "${OBSERVATION_SEEDS:-2}" \
  --workers "${WORKERS:-1}"

scripts/export_contract_rows.sh \
  --input "$OUT/observations-ckpt" \
  --dataset "$OUT/positions-v1.json" \
  --observations-output "$OUT/observations-v1.jsonl"

echo "== h2h report and game-result rows =="
scripts/generate_h2h_stats.sh run \
  --dataset "$OUT/positions-v1.json" \
  --output "$OUT/h2h-bundle.json" \
  --checkpoint-dir "$OUT/h2h-ckpt" \
  --report-output "$OUT/h2h-report.md" \
  --engines "${H2H_ENGINES:-mcts,minimax}" \
  --h2h-positions "${H2H_POSITIONS:-4}" \
  --h2h-seeds "${H2H_SEEDS:-1}" \
  --mcts-iterations "${MCTS_ITERATIONS:-512}" \
  --minimax-depth "${MINIMAX_DEPTH:-5}" \
  --workers "${WORKERS:-1}"

scripts/export_contract_rows.sh \
  --input "$OUT/h2h-ckpt" \
  --dataset "$OUT/positions-v1.json" \
  --games-output "$OUT/game-results-v1.jsonl"

echo "== self-play =="
cargo run --release --example selfplay_export -- \
  --games "${SELFPLAY_GAMES:-8}" \
  --iterations "${SELFPLAY_ITERATIONS:-512}" \
  --seed "${SELFPLAY_SEED:-20260713}" \
  --out "$OUT/selfplay-v1.jsonl"

echo "== materialize model views =="
cd "$MODELS"
PYTHONPATH="$CORE_PY/src:$MODELS/src" "${PYTHON:-python3}" \
  -m quantik_models.data.materialize \
  --observations-jsonl "$OUT/observations-v1.jsonl" \
  --output-npz "$OUT/training-view-observations.npz"
PYTHONPATH="$CORE_PY/src:$MODELS/src" "${PYTHON:-python3}" \
  -m quantik_models.data.materialize \
  --selfplay-jsonl "$OUT/selfplay-v1.jsonl" \
  --output-npz "$OUT/training-view-selfplay.npz"

echo "pipeline output: $OUT"
