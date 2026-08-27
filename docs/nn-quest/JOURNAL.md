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
