# NN Quest — Resume Checkpoint

**Goal:** the first neural network that beats the existing Quantik strategies
(minimax / MCTS / beam / random) head-to-head.

**Status:** PHASE 2 — exact corpus generating; supervised distillation validated
on the deep slice. AlphaZero-from-scratch already *ties* minimax.

Last updated: 2026-08-27.

---

## 1. How to resume

```bash
export NNQ=/Users/mauroberlanda/Code/quantik-ns/quantik-models-py
cd "$NNQ" && git checkout nn-beats-baselines
.venv/bin/python -m pytest -q            # 40 tests, all should pass
cat docs/nn-quest/JOURNAL.md             # full narrative, findings and dead ends
```

Environment: venv at `$NNQ/.venv` (python 3.13.14 via pyenv), editable
`quantik-core-py[arrow]` + `quantik-models[dev,arrow,torch]`, torch 2.13.0,
MPS available. Machine: Apple M5 Pro, 18 cores, 48 GB.

Rust oracle lives in a sibling repo on its own branch:
`../quantik-core-rust` @ branch `exact-oracle`, binary
`target/release/examples/exact_oracle` (rebuild: `cargo build --release --example exact_oracle`).

---

## 2. The one number that matters

**Minimax@100ms is the incumbent champion: 90.1% win rate in the round-robin,
and 97.2% outcome accuracy against exact truth.**

Critically, its accuracy is **100% from ply 8 onward** and drops only in the
opening:

| ply | 4 | 5 | 6 | 7 | 8+ |
|---|---|---|---|---|---|
| minimax@100ms accuracy | 84.8% | 90.9% | 92.2% | 97.1% | 100% |

**All the headroom is in plies 4-7.** Effort spent on endgame strength is
wasted; the network must match minimax there and beat it in the opening.

---

## 3. What has been built

| module | purpose | tests |
|---|---|---|
| `env/fastboard.py` | vectorized rules over `(n,8) uint16` batches; 192-symmetry transforms; canonical keys | `tests/test_fastboard.py` (16) |
| `selfplay/mcts.py` | batched AlphaZero MCTS, descent vectorized across games, leaf batching w/ virtual loss | `tests/test_batched_mcts.py` (11) |
| `selfplay/evaluator.py` | `UniformEvaluator`, `NetEvaluator` (masking outside the model, per contract) | — |
| `selfplay/generate.py` | lockstep self-play, both value signals, symmetry augmentation | `tests/test_selfplay.py` (11) |
| `selfplay/duel.py` | lockstep net-vs-net matches (used for gating) | ↑ |
| `arena/` | side-balanced paired matches vs `quantik_core` engines, Wilson CIs, multiprocess | — |
| `train/alphazero.py` | self-play / learn / gate loop, resumable | — |
| `train/supervised.py` | distillation from exact labels, weighted metric merge | ↑ |
| `../quantik-core-rust` `examples/exact_oracle.rs` | exact solver → JSONL labels | `tests/test_oracle_corpus.py` (4) |

Key speedups measured: rules **377x** over `quantik_core` (8.1M vs 21.6k
positions/s); single-position 800-sim search **855 ms → 266 ms** via leaf batching.

---

## 4. Runbook

```bash
# Baseline round-robin (~6 min on 16 workers)
.venv/bin/python scripts/run_arena.py --out runs/arena/baseline-100ms \
  --label "baselines @ 100 ms/move" --time-limit 0.1 --positions 32 --seeds 2 --workers 16

# Exact-truth probe of any agent set
.venv/bin/python scripts/oracle_probe.py --agents runs/oracle/<agents>.json \
  --out runs/oracle/<name>.json

# Exact corpus (long; resumable — existing ply*.jsonl are reused)
.venv/bin/python scripts/build_oracle_corpus.py

# Supervised distillation
.venv/bin/python -m quantik_models.train.supervised --name <run> \
  --corpus runs/oracle/corpus/exact.npz --channels 128 --blocks 6 --epochs 40

# AlphaZero self-play (resumes from runs/train/<name>/state.json)
.venv/bin/python -m quantik_models.train.alphazero --name <run> --preset small
```

---

## 5. Results so far

| agent | overall outcome accuracy | value MAE |
|---|---|---|
| `minimax@100ms` | 97.2% | — |
| `az-v1-mcts800` (pure AlphaZero, 60 iters) | 97.2% | 0.727 |
| `az-v1-mcts128` | 93.3% | 0.727 |
| `beam@100ms` | 87.6% | — |
| `mcts@100ms` | 77.7% | — |
| `az-v1-policy` (no search) | 70.6% | 0.727 |
| `random` | 44.4% | — |

Supervised on the **deep-only** slice (plies 8-13, 1.26M positions, 5 epochs):

| net | params | s/epoch | val top-1 | value MAE | value sign |
|---|---|---|---|---|---|
| c64 b4 (`small`) | 304,711 | 18 | 92.0% | 0.306 | 89.1% |
| c128 b6 | 1,786,823 | 61 | 91.2% | **0.232** | **91.8%** |

Exact labels take value MAE from 0.727 → 0.232. **That is the whole thesis:
AlphaZero's value head was learning from its own undertrained estimate; the
oracle breaks the circularity.**

---

## 6. In flight / next actions

1. **[running]** `scripts/build_oracle_corpus.py` — plies 12→3.
   Done: 12, 11, 10, 9, 8, 7, 6. Running: 5. Remaining: 4, 3. ~1.8 h left.
   Log: `runs/oracle/corpus-build.log`. Safe to kill and restart; finished
   `ply*.jsonl` files are reused.
2. Rebuild `runs/oracle/corpus/exact.npz` from all plies once solving finishes.
3. Train `c64b4` and `c128b6` on the **full** corpus (40+ epochs) and probe both.
4. AlphaZero fine-tune from the supervised checkpoint (`--init-from`) to sharpen
   the opening, which is the only region that decides the match.
5. Final arena: net vs all four baselines, reporting measured ms/move.
6. Publish the write-up as an artifact.

## 7. Gotchas already hit (do not re-learn these)

- `quantik_core`'s nominal time limits overshoot: minimax checks its clock
  between deepening iterations (196 ms actual for a 100 ms budget), beam only
  between beam levels (465-623 ms). **Always report measured ms/move.**
- `BeamSearchResult` has no `best_move`; use `ranked_root_moves(top_k=1)`.
- Validation metrics must merge **weighted** — the corpus stores all
  policy-labelled rows before all value-only rows, so equal-weight chunk
  averaging under-reported policy metrics 8x.
- Requiring the search to match the solver's single best move is the wrong
  test: in a lost position every move loses and the solver ranks by mate
  distance, which a win/loss-valued search cannot express. Use
  **outcome-optimality** on **won** positions.
- The 640-position probe (`runs/oracle/probe.jsonl`) is held out — the corpus
  builder excludes its canonical keys. Keep it that way or accuracy numbers
  become training scores.
