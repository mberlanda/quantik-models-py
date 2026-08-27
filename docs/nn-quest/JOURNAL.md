# NN Quest — Journal

## 2026-08-27 — Phase 0 kickoff

Surveyed the six-repo Quantik namespace. `quantik-models-py` already ships a
policy/value ResNet, a supervised trainer over `.npz` views, and a
model-checkpoint.v1 exporter, but **no self-play loop, no NN-backed engine
adapter, and no evaluation against the classical engines**. That gap is the
quest.

Created `.venv` (3.13.14) with editable `quantik-core-py` + `quantik-models`
and torch 2.13.0 (MPS available).

### Probe 1 — why the reference engine can't drive self-play

`experiments/probe_solver.py`: an exact `MinimaxEngine.solve` (depth 16) from a
random ply-5 position averages **107 s** in Python (max 272 s). Ply 6 averages
8.9 s, ply 8 averages 0.7 s. Generating exact labels for a training corpus in
Python is therefore off the table; and every sampled position was a forced win
for the side to move (scores 9995-9999).

`experiments/probe_primitives.py` found the real bottleneck:

| primitive | throughput |
|---|---|
| `quantik_core` random playout | 1,598 games/s |
| `quantik_core.generate_legal_moves_list` | 21,657 /s |
| `qfen_to_tensor` | 230,030 /s |
| torch `small` net fwd, MPS, batch 512 | 630,262 pos/s |

The net can evaluate 630k positions/s but the rules can only produce 21k/s.
Self-play would be **97% rule-bound**. So the first build had to be a
vectorized rules engine.

### Build 1 — `quantik_models.env.fastboard`

Re-expressed the rules as NumPy ops over an `(n, 8) uint16` batch (same
channel order as `quantik_core.commons.Bitboard`):

- `legal_masks` — `(n, 64)` from a precomputed `REGION[pos]` table
  (row | column | zone) plus a 64 KiB uint16 popcount table.
- `has_winning_line` — `(n, 4 shapes, 12 lines)` presence reduction.
- `apply_actions`, `terminal_status`, `encode_tensors`, QFEN I/O.

Both Quantik terminal conditions (opponent completed a line / mover has no
legal reply) are losses for the side to move, so `terminal_status` returns a
flat `-1.0` — no winner bookkeeping needed anywhere downstream.

`encode_tensors` is **mover-relative** (channels 0-3 = side to move's shapes)
rather than color-ordered like `qfen_to_tensor`. That is what lets a single
value head use one sign convention. `to_core_tensor` keeps the color-ordered
layout for contract interop, and a test asserts the two differ only by a
channel permutation.

**Correctness:** `tests/test_fastboard.py` replays 3,000 positions from random
playouts and asserts legality masks, win detection, side to move, move
application, QFEN round-trip, both tensor encodings, and terminal status all
match `quantik_core` exactly. 9/9 pass.

**Speedup:**

| primitive | core-py | fastboard | factor |
|---|---|---|---|
| legal moves | 21,657 /s | 8,163,562 /s | **377x** |
| random playouts | 1,598 games/s | 138,024 games/s | **86x** |
| tensor encode | 230,030 /s | 6,146,648 /s | 27x |

Rules are no longer the bottleneck: 8.1M/s rules vs 630k/s net.

### Build 2 — batched MCTS (`quantik_models.selfplay.mcts`)

AlphaZero MCTS where **the descent is vectorized across games rather than
within a tree**. All `g` games in flight walk their trees in lockstep: at each
level the PUCT scores for every active game are one `(g, 64)` NumPy op, so a
simulation round costs a fixed handful of array ops no matter how many games
run, and every leaf in the round reaches the network in a single batch.

Trees live in one flat `(max_nodes, 64)` edge arena, rebuilt per move instead
of re-rooted — re-rooting would need `games x sims x 16` nodes (~577 MB at
g=256, s=128); rebuilding needs `games x sims + games` (~36 MB).

Added FPU reduction (unvisited edge inherits parent Q minus 0.2) so a node
whose explored children all lose still tries a fresh sibling.

**Correctness (`tests/test_batched_mcts.py`, 7 passing):** judged against
`MinimaxEngine.solve`, which is a true depth-16 solver. The search never
visits an illegal action, conserves visit counts, always takes an immediate
win, and — the load-bearing one — **never throws away a provably won root** at
plies 9/10/11 with 512 simulations, using only a *uniform* evaluator.

One test had to be rewritten: an initial version demanded agreement with the
solver's single best move and failed 16/24 at ply 10. That was not a bug. In a
lost position every move loses, and the solver ranks them by mate distance
(`win - ply`), which a win/loss-valued search cannot express. The honest bar is
outcome-optimality, which is what the test now asserts.

### Build 3 — the arena (`quantik_models.arena`)

Ported the Rust `bench::head_to_head` design to Python so the network and the
incumbents can be measured on the same board: every start position x seed is
played twice, once with each agent moving first, so no result reflects which
side a sampled position happened to favour. Quantik has no draws, so a match is
fully described by its win split; reported with a 95% Wilson interval.

The classical agents wrap `quantik_core`'s engines **unchanged** — they are the
incumbents, so they must be the real thing. Agents are built from JSON-able
specs (`arena.registry`) so games can be farmed across the 18 cores.

### Baseline — the number to beat

`runs/arena/baseline-100ms.md`, 128 side-balanced games per pairing from 32
unique ply-4 openings:

| agent | win rate |
|---|---|
| `minimax@100ms` | **90.1%** |
| `beam@100ms` | 65.6% |
| `mcts@100ms` | 40.9% |
| `random` | 3.4% |

**Minimax is the incumbent champion at 90.1%** — that is the bar.

Caveat worth recording: the nominal 100 ms budget is not what the engines
actually spend. Minimax checks its clock between iterative-deepening
iterations (196 ms/move actual) and beam only between beam levels (465-623
ms/move actual). Both overshoot by design, documented in their sources. The
arena therefore reports measured ms/move alongside every result, and the
network will be judged on measured time, not nominal budget.

### Build 4 — symmetry, self-play, and the training loop

**Symmetry (`fastboard`).** Quantik is invariant under 8 dihedral board
symmetries composed with 24 shape relabelings — 192 in all. Implemented as a
single `8 x 65536` uint16 lookup table (1 MiB) that maps a whole board word
through a dihedral element in one fancy-index, plus channel reordering for the
shape permutation. `canonical_keys` packs a board into one uint64 (16 nibbles,
one per square) and minimizes over all 192 images, so dedup is a plain
vectorized reduction. Seven more tests assert the transforms preserve
legality, outcome and mover, commute with `apply_actions`, and that
`transform_actions` and `transform_policies` agree. 16/16 pass.

This is 192x free data augmentation, and it stops the net memorizing board
orientation.

**Self-play (`selfplay/generate.py`).** All games advance in lockstep, so every
live game's root search is one `BatchedMCTS.search` call and the network sees
the whole batch's leaves at once. A game is at most 16 plies, so any batch
finishes in at most 16 rounds. Measured with the `small` net on MPS:

| batch | simulations | wall | throughput |
|---|---|---|---|
| 256 games | 96 | 5.5 s | 46.6 games/s |
| 512 games | 96 | 7.3 s | 70.1 games/s |

Rows keep **both** value signals — the game result `z` and the search's backed-up
root value `q`. Quantik games average ~8 plies, which makes `z` a very coarse
label for an opening position, so the trainer blends them rather than picking.

**Training (`train/alphazero.py`).** Self-play, learn, gate, repeat. A new
generation only becomes the self-play actor if it beats the incumbent
head-to-head at >= 55%, so a bad iteration cannot poison the replay buffer.
Every iteration writes `latest.pt`, a `model-checkpoint.v1` export of the
current best, and a `metrics.jsonl` line; `state.json` makes the run
resumable.

**Gating had to be batched.** The first version played gate games one at a
time through the arena, putting the search on batch-1 network calls — the
slowest possible shape on an accelerator (1,580 fwd/s vs 630k pos/s at batch
512). `selfplay/duel.py` plays all gate games in lockstep instead: at each ply
the live games are split by whose turn it is and each side's positions go
through its own network in one batched search.

### Build 5 — an exact oracle, and the finding that reframed the quest

Python cannot solve Quantik at useful speed, but Rust can. Added
`crates/quantik-core/examples/exact_oracle.rs` to `quantik-core-rust`
(branch `exact-oracle`): read QFENs on stdin, emit per line the root's exact
score, every legal move's exact value, and the **outcome-optimal** move set
(moves that preserve the best achievable result — the fair bar, since a
win/loss-valued engine cannot rank mate distance).

Measured cost per position (full root + all children, rayon over 18 cores):

| ply | 4 | 5 | 6 | 7 | 8+ |
|---|---|---|---|---|---|
| seconds | 1.5 | 0.32 | 0.055 | 0.010 | ~0.003 |

Solved a 640-position probe set spanning plies 4-12 and scored the incumbents
on **outcome accuracy** — over positions the mover provably wins, how often the
agent picks a move that keeps the win:

| agent | ply 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | overall |
|---|---|---|---|---|---|---|---|---|---|---|
| `minimax@100ms` | 84.8% | 90.9% | 92.2% | 97.1% | 100% | 100% | 100% | 100% | 100% | **97.2%** |
| `beam@100ms` | 42.4% | 48.5% | 64.1% | 92.8% | 98.4% | 98.6% | 100% | 100% | 100% | 87.6% |
| `mcts@100ms` | 24.2% | 24.2% | 56.2% | 89.9% | 93.8% | 84.9% | 81.2% | 90.8% | 98.6% | 77.7% |
| `random` | 18.2% | 33.3% | 18.8% | 23.2% | 34.4% | 43.8% | 54.7% | 72.3% | 81.2% | 44.4% |

**This reframes the whole problem.** Minimax at 100 ms is *perfect from ply 8
onward* — it simply solves the endgame. It is beatable only in the opening,
where it scores 84.8% at ply 4 and 90.9% at ply 5.

So the network's requirement is precise: **match minimax in the endgame
(~100% from ply 8) and beat it at plies 4-7**. Any effort spent making the net
better at ply 10 is wasted; all of the headroom is in the opening.

Two more facts from the probe worth keeping: 534/640 positions (83%) are wins
for the side to move, so Quantik strongly favours the mover; and a won
position has on average 3.6 outcome-optimal moves out of 12.1 legal ones,
which is why random still scores 44%.

### Run 1 — pure AlphaZero from scratch (`az-small-v1`)

60 iterations, `small` preset (304,711 params), 512 games/iteration at 96
simulations, 400 train steps of batch 512, 4x symmetry augmentation, replay
buffer of 10 generations. ~12 s/iteration, ~12 minutes total on MPS.

Policy loss 3.40 -> 2.11, policy top-1 17.8% -> 46.0%, mean game length
7.9 -> 9.3 plies (both sides learning to avoid quick losses).

Probe result — outcome accuracy against exact truth:

| agent | ply 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | overall | value MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `az-v1-policy` (no search) | 72.7% | 60.6% | 50.0% | 63.8% | 67.2% | 72.6% | 71.9% | 84.6% | 87.0% | 70.6% | 0.727 |
| `az-v1-mcts128` | 78.8% | 84.8% | 79.7% | 85.5% | 100% | 100% | 98.4% | 100% | 100% | 93.3% | 0.727 |
| `az-v1-mcts800` | 84.8% | 87.9% | 92.2% | 98.6% | 100% | 100% | 100% | 100% | 100% | **97.2%** | 0.727 |
| `minimax@100ms` | 84.8% | 90.9% | 92.2% | 97.1% | 100% | 100% | 100% | 100% | 100% | **97.2%** | — |

From-scratch AlphaZero **draws level with minimax** at 800 simulations — same
97.2% overall, better at ply 7, worse at ply 5. Level is not the goal.

**The value head is the bottleneck, and the probe says so precisely: value MAE
0.727 against a ±1 truth.** Always predicting +1 would score 0.34, so the value
head is outputting near-zero — it does not know who is winning. That is what
caps MCTS: with a flat value the search does almost all the work, which is why
128 sims scores 93.3% while 800 scores 97.2%.

The cause is structural, not a bug: the value target was
`0.5 * game_result + 0.5 * search_root_value`, and both terms are weak early —
the result is a single bit at the end of an ~8-ply game, and the root value
comes from the same undertrained net. Exact labels break that circularity.

### Build 6 — supervised training on exact labels

`train/supervised.py`. Two things make it cheap:

* **Free child labels.** Solving a position solves all its children, so the
  corpus carries ~10 value-labelled rows per policy-labelled one. Rows with no
  policy target contribute to the value loss only, via `policy_weight`.
* **On-the-fly symmetry.** Every batch is transformed by a fresh draw from the
  192-element group, so the net effectively never sees the same board twice
  without paying for the storage.

The train/val split hashes the **canonical** key, so a rotated copy of a
training board cannot leak into validation.

**Bug found and fixed: validation metrics were being under-reported ~8x.**
`rows_from_oracle` writes every policy-labelled row before every value-only
row, so a sorted validation index put 7,520 of 7,520 policy rows in the first
8,192-row chunk and none in the other seven. Averaging chunks equally then
divided the policy metrics by the chunk count: a real 89% top-1 was printing
as 11%, and 1.40 policy loss as 0.175. Fixed by pairing every metric with the
weight it averages over and merging with a weighted mean (`_merge`), with a
regression test. Worth recording because the wrong numbers looked *plausible*
— a low loss next to a low accuracy — and would have sent the next few hours
chasing the policy head instead of finishing the corpus.

**Ground truth is cross-validated.** `tests/test_oracle_corpus.py` re-solves
sampled oracle rows with `quantik_core`'s Python solver — a different language
and a different codebase — and asserts agreement on who wins, on the
outcome-optimal move set, and on legal-action coverage. 4/4 pass.

First supervised result, on the deep-only slice of the corpus (plies 8-13,
1.26M positions, 150k with policy labels, 3 epochs, `small` preset):
val top-1 **88.2%**, value MAE **0.387** — versus 0.727 for the AlphaZero net.
The value head starts working the moment it is given real labels.

### Build 7 — leaf-batched MCTS

The arena plays one position at a time, which made every simulation a batch-1
network call. Descents within a round are now separated by virtual loss and
their leaves evaluated together; duplicate leaf edges in a round are collapsed
so a position never gets two nodes. 800 simulations on a single position:
**855 ms -> 266 ms** at `leaf_batch=64`.

### Build 8 — solving the opening outright

The probe said all the headroom is at plies 4-7. Rather than sample that
region, solve it completely.

The trick is that **solving one level makes every shallower level free**. A
position's value is the best of its children's negated values, and its optimal
moves are exactly those leading to a child the opponent loses. So instead of
paying the full oracle (one solve per legal move) at every level, pay a
*root-only* solve — 25x cheaper, added as `--roots-only` — at one deep level
and back-induce everything above.

First, enumerate the canonical tree with `fastboard`. The counts came out:

| ply | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| canonical live positions | 3 | 51 | 726 | 10,946 | 105,632 | 901,916 |

These match `quantik-core-py/GAME_TREE_ANALYSIS.md` **exactly** at every level —
an independent validation of the vectorized rules and the 192-symmetry
canonical key against a table computed by entirely different code.

`scripts/solve_opening.py --frontier 6` therefore yields exact values for every
canonical position at plies 0-6 and exact optimal-move sets for plies 0-5: the
complete solution of the region where minimax is beatable.

Killed the sampled-corpus job after ply 6, since the complete solve subsumes
plies 0-6. Kept its plies 6-12 output: **3,087,356** unique exactly-labelled
positions, **250,000** with exact policy targets.

### Build 9 — compact corpus storage

The first full-corpus training run sat for **12 minutes before its first
epoch**, all of it decompressing and canonicalizing. Cause: a
`(3.09M, 64) float32` dense policy array is 790 MB, and 92% of its rows are
value-only — all zeros.

Every policy target the oracle produces is uniform over a set of
outcome-optimal actions, so the whole target is really a 64-bit set.
`data/exact_corpus.py` stores it as one `uint64` mask (8 bytes/row instead of
256) and expands it per batch; an empty mask means "value label only", which
also replaces the separate `policy_weight` column. `ExactCorpus.load` still
reads the old dense format so earlier runs stay reproducible.

### Build 10 — the final evaluation harness

`scripts/final_evaluation.py` runs the two measurements a claim of "beats the
incumbents" actually needs, into one report: the side-balanced arena **with
measured ms/move**, and per-ply outcome accuracy against exact truth. Winning
the arena while being slower per move, or while being less accurate, would not
be the same result.

Note for whoever runs it: the machine must be otherwise idle. Under load from
the corpus solve, minimax measured 772 ms/move against its clean 196 ms — the
timings are only meaningful on a quiet machine.

### Incident — the ply-6 solve, and what it taught

The user noticed `exact_oracle` eating 1800% CPU. Three real defects behind that:

1. **No thread bound.** The default rayon pool takes every core. Fine for a
   dedicated batch run, wrong when anything else needs the machine — and I had
   briefly run *two* copies (a frontier-5 hedge alongside frontier-6), driving
   load average to 145 and slowing both.
2. **No streaming.** `par_iter().collect()` buffered all 901,916 results until
   the end, so an interrupt lost the entire multi-hour run.
3. **No resume.** Restarting meant redoing everything.

Then I made it concrete: `pkill -f "roots-only"` was over-broad and killed the
main solve along with the hedge, losing ~1 hour — exactly the failure the
buffering made unrecoverable.

Fixes: `--threads N` sizes the pool (now 12 of 18); positions solve in chunks
of 2,000 with each chunk flushed; `--append-to` skips QFENs already in the
output so a killed run resumes. Verified by running the same 3,000 positions
twice — the second run reported "resuming: 3000 of 3000 already solved" and
did no work.

Separately, the shared `quantik-core-rust` checkout was switched to `main` by
other work mid-run, which made the oracle source vanish from under the build.
The oracle now builds from a dedicated git worktree (`.oracle-worktree` on
branch `exact-oracle`) instead of competing for the branch.

**Lesson worth keeping:** a job measured in hours needs to be interruptible by
construction. Buffer-then-write is fine at 200 positions and indefensible at a
million.

### Result — the hypothesis holds: deep exact values generalize upward

Probed `sweep-c128b6` — a net trained for only **5 epochs on the deep-only
slice** (plies 8-13, no opening data at all) — driving MCTS at 400 simulations:

| agent | ply 4 | 5 | 6 | 7 | 8 | 9-12 | overall |
|---|---|---|---|---|---|---|---|
| `sup-deep-mcts400` | **90.9%** | **93.9%** | **95.3%** | **98.6%** | 100% | 100% | **98.3%** |
| `minimax@100ms` | 84.8% | 90.9% | 92.2% | 97.1% | 100% | 100% | 97.2% |
| `sup-deep-policy` (no search) | 63.6% | 60.6% | 67.2% | 81.2% | 85.9% | 86-96% | 83.1% |

**The network beats minimax at every opening ply**, with no opening training
data whatsoever. Exact values at plies 8-13 plus two plies of search is enough
to out-play a solver that cannot see that far in 100 ms. This is the first
configuration to pass minimax on the exact-truth measure.

### But the arena said 47.7% — and that is a measurement problem, not a fluke

Clean paired match (background jobs SIGSTOPped, resumed after), 128
side-balanced games from 32 ply-4 openings:

| | win rate | 95% CI | ms/move |
|---|---|---|---|
| `net-mcts400` | 47.7% | 39.2%-56.3% | 228 |
| `minimax@100ms` | 52.3% | — | 198 |

Two structural reasons the arena under-discriminates, both worth fixing before
the final run:

1. **Side-balanced pairing pins the ceiling near 50%.** From a fixed position,
   two near-perfect players each win the game where they hold the winning side.
   With conversion rates of 91% vs 85%, the expected score is only ~52% — well
   inside a 128-game confidence interval. The arena needs **many more start
   positions**, not more seeds.
2. **Seeds add nothing here.** Both agents are deterministic (`add_noise=False`
   for the net; minimax's seed only tweaks move ordering), so replaying a
   position under a second seed replays the same game. Diversity has to come
   from distinct openings.

Also: games from ply-4 starts last only ~4.9 more plies, so most of the game
happens in the zone where both players are already exact. Starting earlier
puts more decisions in the contested region.

**Final evaluation design, revised:** ~500 distinct openings at plies 3-5, one
seed, giving ~1,000 games and a ~±3% interval — and a network with a genuinely
larger opening edge, which is what the exact opening corpus is for.

### Run 2 — supervised distillation on the sampled corpus (`sup-sampled-c128b6`)

c128/b6 (1,786,823 params), 3,087,356 exactly-labelled positions (plies 6-13,
250,000 with exact optimal-move sets), ply-balanced draws, per-batch symmetry
augmentation, ~185 s/epoch on MPS.

| epoch | val top-1 | value MAE | value sign | ply-6 top-1 | ply-7 top-1 |
|---|---|---|---|---|---|
| 0 | 87.8% | 0.377 | 87.8% | 82.0% | 84.2% |
| 1 | 91.3% | 0.199 | 93.0% | 86.3% | 88.1% |
| 2 | 92.8% | 0.159 | 94.2% | 89.0% | 90.1% |
| 4 | 94.4% | 0.128 | 95.4% | — | — |
| 6 | 95.1% | 0.107 | 96.1% | — | — |
| 7 | 95.3% | 0.103 | 96.3% | — | — |

Value MAE against a ±1 truth: **0.727 (AlphaZero) → 0.103**. The value head
now knows who is winning, which is precisely what the probe said was missing.

### Decision — no AlphaZero fine-tune on top

The obvious next step would be to fine-tune the distilled net with self-play.
Not doing it, deliberately: self-play labels are the net's own bootstrapped
estimates, and every position it would generate already has an *exact* label
available for less compute. Fine-tuning would replace a perfect teacher with an
imperfect one. AlphaZero-from-scratch remains in the results as the documented
baseline it is — it tied minimax at 97.2%, and diagnosing *why it stalled there*
is what produced the oracle in the first place.

## RESULT — the network beats minimax

`sup-sampled-c128b6` (16 epochs, final val top-1 96.3%, value MAE 0.084)
driving the batched MCTS on a wall clock, CPU-only, one thread — the same
hardware the classical engines get.

**Exact-truth accuracy** (640 held-out solved positions):

| agent | ply 4 | 5 | 6 | 7 | 8-12 | overall |
|---|---|---|---|---|---|---|
| **`qnet@200ms`** | **97.0%** | **97.0%** | **100%** | **100%** | 100% | **99.6%** |
| `alphazero@200ms` | 84.8% | 90.9% | 93.8% | 97.1% | 100% | 97.4% |
| `minimax@100ms` | 84.8% | 90.9% | 92.2% | 97.1% | 100% | 97.2% |
| `qnet-policy` (one forward pass) | 87.9% | 90.9% | 89.1% | 89.9% | 94-100% | 94.9% |
| `beam@100ms` | 42.4% | 48.5% | 64.1% | 92.8% | 98-100% | 87.6% |
| `mcts@100ms` | 24.2% | 24.2% | 56.2% | 89.9% | 81-99% | 77.7% |
| `random` | 18.2% | 33.3% | 18.8% | 23.2% | 34-81% | 44.4% |

**Head-to-head**, 1,200 side-balanced games from 600 symmetry-distinct
openings at plies 3-5:

| match | win rate | 95% CI | ms/move (net vs minimax) |
|---|---|---|---|
| `qnet@200ms` vs `minimax@100ms` | **60.5%** | 57.7-63.2% | 211 vs 194 |
| `qnet@50ms` vs `minimax@100ms` | **55.8%** | 52.9-58.5% | **63 vs 203** |

The second row is the one that settles it: the network beats the incumbent
champion **while spending 3.2x less time per move**, on the same single CPU
thread. Both intervals sit entirely above 50%.

Note how well the two measurements agree with the theory. Predicted score from
conversion rates alone was
`½·[P(net converts) + P(minimax errs)·P(net punishes)]` ≈ 0.5·[0.99 + 0.15·0.95]
≈ 57% at ply 4, rising for earlier starts — and the measured 60.5% from plies
3-5 lands right there. The arena's ~50% pull is real; the network clears it
because its opening error rate is genuinely near zero.

### Where the win comes from

Exactly where the probe predicted at the very start: **the opening**. Minimax
is perfect from ply 8; so is the network. The entire margin is plies 4-7,
where the network scores 97-100% against minimax's 84.8-97.1%. The network is
not out-searching the solver — it is out-*knowing* it, carrying distilled exact
values into positions minimax cannot reach in 100 ms.

### Final leaderboard

Full round-robin, 300 side-balanced games per pairing from 150
symmetry-distinct openings at plies 3-5 (1,800 games per agent), every agent
on one CPU thread:

| agent | win rate | ms/move | outcome accuracy |
|---|---|---|---|
| **`qnet@200ms`** | **80.6%** | 213 | **99.6%** |
| `minimax@100ms` | 72.6% | 218 | 97.2% |
| `alphazero@200ms` | 67.2% | 206 | 97.2% |
| `qnet-policy` (one forward pass) | 58.4% | **1** | 94.9% |
| `beam@100ms` | 45.2% | 451 | 87.8% |
| `mcts@100ms` | 23.2% | 111 | 77.5% |
| `random` | 2.8% | 0 | 44.4% |

**The network tops the table**, and the direct match is `qnet@200ms` 60.3% vs
`minimax@100ms` (39.7% for minimax) at near-identical think time.

Three things in this table are worth more than the headline:

1. **`qnet-policy` places fourth at 1 ms/move** — a single forward pass, no
   search at all, beating `beam@100ms` (451 ms/move) and `mcts@100ms`. Most of
   the classical engines' compute is buying knowledge the network simply has.
2. **`alphazero@200ms` ties minimax on accuracy (97.2% each) but loses the
   match 42% vs 58%.** Equal average accuracy is not equal strength: the two
   make their mistakes in different places, and minimax's are concentrated
   where the network from scratch also happens to be weak, so it fails to
   punish them.
3. **`beam@100ms` spends 451 ms/move** — 4.5x its nominal budget, because it
   only checks its clock between beam levels — and still places fifth. Nominal
   budgets would have made this table meaningless; every number here is
   measured.
