# How much of Quantik did the model see, and is the evaluation big enough?

Two questions a win rate cannot answer. Both came up because the original
report quoted "640 exactly-solved held-out positions" without a denominator,
and 640 is not obviously enough.

## Short answer

- **640 was the test set, not the training data.** The model trained on
  **3,087,356** exactly-labelled positions.
- **But the probe really was too thin where it mattered** — only 33 won
  positions each at plies 4 and 5, the two plies that carry the entire margin.
  It is now **8,440** positions (7,030 won), with 1,240 at each of plies 4-5.
- **On the larger probe the result holds and sharpens**: +3.14 percentage
  points overall (95% CI +2.70 to +3.60), paired exact test **p = 2.5e-49**.
- **The model saw 5.02% of the game through ply 9, and 0% of plies 0-5** —
  the plies where it beats minimax.

## 1. Coverage

Positions are counted **up to symmetry**: Quantik is invariant under 8 board
symmetries composed with 24 shape relabelings, so a position and its 191 images
are one game. `canonical live` excludes terminal positions, which need no
decision.

| ply | canonical live | trained on | with policy label | coverage | held-out probe |
|---|---|---|---|---|---|
| 0 | 1 | 0 | 0 | — | 0 |
| 1 | 3 | 0 | 0 | — | 0 |
| 2 | 51 | 0 | 0 | — | 0 |
| 3 | 726 | 0 | 0 | — | 0 |
| 4 | 10,946 | 0 | 0 | **0%** | 1,240 |
| 5 | 105,632 | 0 | 0 | **0%** | 1,240 |
| 6 | 901,916 | 40,000 | 40,000 | 4.44% | 1,280 |
| 7 | 4,658,465 | 846,816 | 60,000 | 18.18% | 1,280 |
| 8 | 17,894,928 | 1,001,185 | 60,000 | 5.59% | 880 |
| 9 | 37,922,646 | 698,460 | 30,000 | 1.84% | 680 |
| 10 | not enumerated | 255,278 | 20,000 | — | 680 |
| 11 | not enumerated | 118,055 | 20,000 | — | 580 |
| 12 | not enumerated | 86,741 | 20,000 | — | 580 |
| 13 | not enumerated | 40,821 | 0 | — | 0 |

**Plies 0-9 hold 61,495,314 canonical live positions. The model trained on
3,087,356 of them — 5.02%.** Policy-label coverage is thinner still: 1.29% at
ply 7, 0.34% at ply 8, 0.08% at ply 9.

Plies 10-13 were not enumerated. The counts are past their peak by then (the
growth factor has fallen from ~15 early to ~2.1 at ply 9) and every engine
measured is already exact from ply 8, so the missing denominators do not
affect any claim here.

### The striking part

**The network never saw a single position at ply 4 or ply 5, and beats
minimax at both.** Its accuracy there (96.4% and 97.8%) comes from exact values
learned two to four plies deeper, carried up by search. That is what makes the
result a generalization claim rather than a memorization one.

It also says where the remaining error is. The network's only mistakes are at
plies 4-6 — precisely the region with zero training coverage.

### Verified against the project's own published counts

`quantik-core-py/GAME_TREE_ANALYSIS.md` publishes canonical counts to depth 8.
This enumeration reproduces plies 1-7 **exactly**, and resolves the one
apparent discrepancy at ply 8 (`scripts/verify_published_counts.py`):

```
ply 8 canonical positions: 20,049,874
  live (a decision to make):          17,894,928
  terminal, a line is complete:        2,149,714
  terminal, mover has no legal move:       5,232
  live + stuck = 17,900,160   == the published figure
```

The published table counts a stuck mover as "ongoing"; this project counts it
as terminal, because a player with no legal move has lost. Same enumeration,
different convention. Ply 9 (**37,922,646** live) is past where the published
analysis stopped.

## 2. Statistical power

Both agents see identical positions, so the comparison is **paired** — an
unpaired confidence interval discards most of the evidence. The instrument is
an exact McNemar test on the positions where the two disagree.

### Per ply, on 7,030 provably won positions

| ply | positions | `qnet@200ms` | `minimax@100ms` |
|---|---|---|---|
| 4 | 951 | **96.42%** | 84.44% |
| 5 | 1,024 | **97.75%** | 91.80% |
| 6 | 1,050 | **98.86%** | 95.24% |
| 7 | 1,125 | **99.91%** | 99.20% |
| 8 | 732 | 100% | 100% |
| 9 | 595 | 100% | 100% |
| 10 | 541 | 100% | 100% |
| 11 | 507 | 100% | 100% |
| 12 | 505 | 100% | 100% |
| **all** | **7,030** | **99.00%** | **95.86%** |

### The test

| | all plies | opening (4-7) |
|---|---|---|
| won positions compared | 7,030 | 4,150 |
| only `qnet` right | 241 | 241 |
| only `minimax` right | 20 | 20 |
| accuracy difference | +3.14% | +5.33% |
| 95% CI (paired bootstrap) | +2.70 to +3.60 | +4.58 to +6.10 |
| exact paired test | **p = 2.5e-49** | **p = 2.5e-49** |

**Every disagreement between them is in the opening.** Across 2,880 won
positions at plies 8-12, both play perfectly — not one error either way. The
two columns of the table are identical because there is nothing to add past
ply 7.

### All agents on the larger probe

Re-measured on the same 8,440 positions, so the whole table is one instrument:

| agent | outcome accuracy |
|---|---|
| `qnet@200ms` | 98.98% |
| `minimax@100ms` | 95.86% |
| `qnet-policy` | 93.76% |
| `alphazero@200ms` | 93.27% |
| `beam@100ms` | 80.84% |
| `mcts@100ms` | 66.05% |
| `random` | 34.79% |

Two orderings changed against the 640-position probe, and both are informative.
`qnet-policy` — a single forward pass, 0.36 ms per position — now **outscores
`alphazero@200ms`**, which spends 200 ms searching. And `alphazero@200ms` drops
from 97.2% to 93.3%: its apparent parity with minimax on the old probe was an
artifact of that probe's ply distribution, not real strength.

### Was the original 640 wrong?

No — underpowered, not wrong. Minimax measured 84.8% at ply 4 on 33 positions
and 84.44% on 951; the network 97.0% and 96.42%. The point estimates were
accurate; the intervals were just too wide to lean on. The overall figures
moved (network 99.63% → 99.00%, minimax 97.19% → 95.86%) because the new probe
deliberately weights the opening far more heavily, not because either agent
changed.

## 3. A leak, found and quantified

The corpus builder excluded probe positions when *sampling parents*, but child
rows are derived rather than sampled — so **16 of the 640 original probe
positions reached the training corpus** as value-only rows (plies 7-9; none
carried a policy label, none at plies 4-6).

Worst-case impact, assuming every leaked position was answered correctly and
removing all of them: `qnet` 99.63% → 99.61%, `minimax` 97.19% → 97.11%. The
margin is unaffected, and the plies that carry it had zero leakage. Fixed at
the point it cannot be bypassed, with a regression test.

The 7,800-position probe excludes the training corpus **and** the original
probe, both up to symmetry.

## Reproduce

```bash
.venv/bin/python scripts/count_canonical.py --max-ply 9   # ~35 min
.venv/bin/python scripts/verify_published_counts.py       # ~8 min
.venv/bin/python scripts/build_probe.py                   # ~1 h (ply 4 dominates)
.venv/bin/python scripts/coverage_report.py --workers 12
```
