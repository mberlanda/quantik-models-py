# NN Quest — Resume Checkpoint

**Goal:** the first neural network that beats the existing Quantik strategies
(minimax / MCTS / beam / random) head-to-head.

**Status: DONE.** The network tops the leaderboard at 80.6% against minimax's
72.6%, wins the direct match 60.5%, and beats it 55.8% while spending 3.2x
less time per move. Verified against exact truth on 8,440 held-out solved
positions with a paired exact test at p = 8e-49.

Merged: `quantik-models-py` #4 and #5, `quantik-core-rust` #38.

Last updated: 2026-08-27.

## 1. How to resume


```bash
export NNQ=/Users/mauroberlanda/Code/quantik-ns/quantik-models-py
cd "$NNQ" && git checkout main   # everything is merged
.venv/bin/python -m pytest -q            # 91 tests, all should pass
cat docs/nn-quest/JOURNAL.md             # full narrative, findings and dead ends
cat docs/nn-quest/COVERAGE.md            # how much of the game it saw, and the power analysis
open docs/nn-quest/report.html           # the published write-up
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

## 5. Headline result

Round-robin, 1,800 side-balanced games per agent, one CPU thread each;
accuracy on 8,440 exactly-solved held-out positions (7,030 provably won):

| agent | arena win rate | ms/move | exact-truth accuracy |
|---|---|---|---|
| **`qnet@200ms`** | **80.6%** | 213 | **98.98%** |
| `minimax@100ms` | 72.6% | 218 | 95.86% |
| `alphazero@200ms` | 67.2% | 206 | 93.27% |
| `qnet-policy` (one forward pass) | 58.4% | **1** | 93.76% |
| `beam@100ms` | 45.2% | 451 | 80.84% |
| `mcts@100ms` | 23.2% | 111 | 66.05% |
| `random` | 2.8% | 0 | 34.79% |

Direct matches: 60.5% over 1,200 games (CI 57.7-63.2%); 55.8% at a 50 ms budget
while spending 63 ms/move against minimax's 203.

Paired exact test on accuracy: **p = 8e-49** (239 positions only the
network gets right, 20 only minimax). Every disagreement is at plies 4-7;
across 2,880 won positions from ply 8 on, neither engine errs.

Coverage: the model trained on 5.02% of the 61,495,314 canonical positions
through ply 9 — and **none at plies 0-5**, where it beats minimax by the widest
margin. See `COVERAGE.md`.

## 5b. Results log

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

## 6. Data assets

| file | contents |
|---|---|
| `runs/oracle/probe.jsonl` | **held out.** 640 exactly-solved positions, plies 4-12. Never train on this. |
| `runs/oracle/corpus/exact-sampled.npz` | 3,087,356 unique positions (plies 6-13), 250,000 with exact policy. Sampled, full-oracle. |
| `runs/oracle/opening5/opening-exact.npz` | complete exact solution, plies 0-5 values + plies 0-4 policies. |
| `runs/oracle/opening/opening-exact.npz` | **[in flight]** complete exact solution, plies 0-6 values + plies 0-5 policies. |
| `runs/oracle/opening*/level*.npy` | enumerated canonical live positions per ply (reusable cache). |

## 7. Optional follow-up (not required — the goal is met)

**[stopped 2026-08-27, 64,000 of 105,632 banked — 61%]**
`scripts/solve_opening.py --frontier 5 --threads 10 --out runs/oracle/opening5`
— root-only exact solve of all 105,632 canonical ply-5 positions. Stopped
because it was taking ~10 CPU cores for an optional experiment after the goal
was already met.

It **resumes**: results stream to `runs/oracle/opening5/frontier.jsonl` and a
re-run skips what is already there, so restarting picks up at 61% done. Use
`--threads N` to bound it; at 10 threads the remaining 67,632 positions were
running at roughly 4,000 per 13 minutes; 41,632 remain.

Why it exists: the coverage analysis showed the model trained on **zero**
positions at plies 4-5, and those are exactly where its remaining error lives
(96.4% at ply 4, 97.8% at ply 5, 100% from ply 8). Completing this solve yields
100% exact move coverage at ply 4 and 100% exact values at ply 5 — the material
to test whether filling that gap moves ply-4 accuracy.

To run the experiment once it finishes:

```bash
# induction completes automatically and writes opening-exact.npz
.venv/bin/python -c "
from quantik_models.data.exact_corpus import ExactCorpus
merged = ExactCorpus.concat([
    ExactCorpus.load('runs/oracle/opening5/opening-exact.npz'),
    ExactCorpus.load('runs/oracle/corpus/sampled.npz')])
merged.save('runs/oracle/corpus/combined.npz')"
.venv/bin/python -m quantik_models.train.supervised --name qnet-v2 \
  --corpus runs/oracle/corpus/combined.npz --channels 128 --blocks 6 --epochs 16
.venv/bin/python scripts/coverage_report.py --agents runs/oracle/coverage-agents.json
```

To stop it instead: `kill $(pgrep -f solve_opening)` — nothing else depends on it.

## 8. Superseded (kept so the history reads correctly)

1. **[running]** `scripts/solve_opening.py --frontier 5 --threads 12 --out runs/oracle/opening5`
   — root-only solve of all 105,632 canonical ply-5 positions (~1.7 h).
   Yields complete exact values for plies 0-5 and complete exact policies for
   plies 0-4. **Resumable**: streams to `runs/oracle/opening5/frontier.jsonl`
   and skips what is already there.
2. **[queued]** sampled full oracle on 15,000 ply-5 positions
   (`runs/oracle/corpus/ply05.qfen`, 14.2% coverage, probe positions excluded)
   → ply-5 policy labels plus ply-6 values.
3. **[running]** `sup-sampled-c128b6` — supervised on the sampled corpus alone.
   Evaluate it on the probe as soon as it finishes; it may already beat minimax
   without any opening data.
4. Combine: `ExactCorpus.concat([opening, ply5_sampled, sampled])`, opening first.
5. Train the final net with `--balance-plies`, probe it, then AlphaZero
   fine-tune from it (`--init-from`).
6. `scripts/final_evaluation.py` on an **idle machine**.
7. Publish the write-up as an artifact.

### Why frontier 5, not 6

Frontier 6 (901,916 root solves) was measured at ~7.5 h and was abandoned.
The same coverage comes far cheaper:

| need | source | cost |
|---|---|---|
| exact policy, plies 0-4 | frontier-5 solve (105,632 roots) | ~1.7 h |
| exact policy, ply 5 | sampled full oracle, 15,000 positions | ~1.2 h |
| exact policy, plies 6-7 | already have (40k + 60k full-oracle rows) | done |
| exact values, plies 8-13 | already have (3.09M rows) | done |

10,000 ply-6 values from the abandoned frontier-6 run survive in
`runs/oracle/opening/frontier.jsonl` — the streaming fix's first dividend.

## 8. The report

`docs/nn-quest/report.html` — a self-contained page (open it directly in a
browser; no server, no assets). Also published at
<https://claude.ai/code/artifact/75b8b0de-a91e-49ba-8ae1-4c01062c46ee>.

It is **generated, not hand-written**:

```bash
.venv/bin/python scripts/build_report.py
```

reads `runs/arena/final.json`, `runs/arena/showdown.json` and
`runs/arena/handicap.json` and rebuilds every number, chart coordinate and
table cell. `docs/nn-quest/report.head.html` holds the styling; edit that for
design changes and the generator for content. Rerun the evaluation and the
report updates itself — no number in it is ever transcribed by hand.
