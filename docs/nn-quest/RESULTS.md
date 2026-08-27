# Beating Quantik's classical engines with a neural network

*Working results document — numbers are filled in as runs complete. See
`JOURNAL.md` for the narrative and `CHECKPOINT.md` for how to resume.*

## The question, made precise

"Beat the existing strategies" needs a definition, because Quantik is a solved
game: `quantik_core.minimax.MinimaxEngine.solve` is a depth-16 search and no
Quantik game exceeds 16 plies, so it plays perfectly given unlimited time. The
meaningful contest is therefore **at a fixed budget**, which is how the
project's own Rust `bench::head_to_head` harness frames it.

Two measurements are reported for every agent:

- **Arena win rate** — side-balanced paired games (each opening played twice,
  once from each side) against `random`, `minimax`, `mcts`, `beam`, with
  **measured** ms/move. Measured matters: `quantik_core`'s engines overshoot
  their nominal budgets by design, so "minimax@100ms" really spends ~196 ms.
- **Outcome accuracy** — on 640 held-out exactly-solved positions, the share of
  provably won positions where the agent picks a move that keeps the win.
  Accuracy says *where* an agent is wrong; win rate only says that it lost.

## Why the arena alone is a weak instrument

Side-balanced pairing bounds what any agent can score. From a fixed position,
two near-perfect players each win the game where they hold the winning side, so
the score tends to 50%. An agent only gains by punishing mistakes:

```
expected score ≈ ½ · [ P(I convert my won side) + P(you blunder) · P(I punish) ]
```

With conversion rates of 91% vs 85% that is only ~52% — inside the confidence
interval of a few hundred games. So the arena is run with many
**symmetry-distinct** openings (extra seeds add nothing: both agents are
deterministic) started **early** (plies 3-5), where more of the game falls in
the region the engines actually disagree about.

## The incumbents, measured against exact truth

| agent | ply 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | overall |
|---|---|---|---|---|---|---|---|---|---|---|
| `minimax@100ms` | 84.8% | 90.9% | 92.2% | 97.1% | 100% | 100% | 100% | 100% | 100% | **97.2%** |
| `beam@100ms` | 42.4% | 48.5% | 64.1% | 92.8% | 98.4% | 98.6% | 100% | 100% | 100% | 87.6% |
| `mcts@100ms` | 24.2% | 24.2% | 56.2% | 89.9% | 93.8% | 84.9% | 81.2% | 90.8% | 98.6% | 77.7% |
| `random` | 18.2% | 33.3% | 18.8% | 23.2% | 34.4% | 43.8% | 54.7% | 72.3% | 81.2% | 44.4% |

**Minimax is exact from ply 8 onward.** It is beatable only in the opening.
That single fact determined everything that followed.

## Method

1. **Vectorized rules** (`env/fastboard.py`) — the reference engine runs at
   21.6k positions/s and the network evaluates 630k/s, so self-play would have
   been 97% rule-bound. Re-expressed as NumPy over `(n, 8) uint16` batches:
   8.1M positions/s, **377x**. Cross-checked against `quantik_core` on 3,000
   sampled positions.
2. **Batched MCTS** (`selfplay/mcts.py`) — descent vectorized *across games*
   rather than within a tree, plus leaf batching behind virtual loss for
   single-position play (800 sims: 855 ms → 266 ms).
3. **Exact oracle** (`quantik-core-rust`, `examples/exact_oracle.rs`) — solves
   positions to game-theoretic truth; `--roots-only` is 25x cheaper and enough
   to reconstruct optimal moves by backward induction over a whole level.
4. **Distillation** (`train/supervised.py`) — supervised on exact values and
   exact optimal-move sets, with 192-fold symmetry augmentation applied per
   batch and training draws balanced across plies.

## Results

*(filled in on completion)*

## Reproducing

See `CHECKPOINT.md` §4.
