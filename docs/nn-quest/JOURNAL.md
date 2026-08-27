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
