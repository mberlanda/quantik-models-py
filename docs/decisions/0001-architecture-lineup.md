# 1. Which architectures to train, and which to only write down

**Status:** accepted, 2026-08-28
**Supersedes:** nothing. The ResNet was never chosen against alternatives; it
was the first thing that worked.

## The question

We have a labelled corpus of 3,087,356 canonical positions carrying exact
policy and value targets from the solver, and one trained network —
`resnet-c128-b6`, 1,786,823 parameters. The goal is at least three trained
architectures so that a comparison is possible at all, plus a written record
of the ones we considered and declined.

The declining is the part worth writing down. A lineup that lists only what
was built reads like everything else was overlooked.

## The constraint that shapes everything

Quantik is played on a 4x4 board. That single fact removes most of the
reasons one normally prefers one architecture over another:

- **Receptive field is not an argument.** The ResNet's stem plus its first
  residual block already sees all sixteen cells. "This one models long-range
  structure and that one does not" is a statement about 19x19 Go, not about
  this game. Any claim that an architecture wins because it is global is
  false here, and an earlier version of the attention rationale made exactly
  that mistake.
- **Depth buys capacity, not reach.** Adding blocks to the ResNet adds
  parameters and nonlinearity. It does not let the network see further,
  because there is nowhere further to see.
- **The interesting structure is not spatial adjacency.** Quantik's rule is
  that a shape may not be placed where the *opponent* already played that
  shape in the same row, column, or 2x2 zone. Twelve overlapping groups —
  four rows, four columns, four zones — and every cell belongs to exactly
  three of them. A 3x3 convolution does not align with any of those groups:
  a zone is a 2x2 block, a row is a 1x4 strip, and a 3x3 kernel centred on a
  cell straddles the boundaries of both.

So the honest hypothesis space is not "how far can each model see" but "how
naturally does each model express a twelve-group constraint over four
shapes".

## Candidates

### Trained

**MLP baseline.** Flatten the 144 inputs, two or three hidden layers, the
same two heads. It exists to answer a question no amount of argument
settles: on a board this small, does spatial structure buy anything at all?
If a parameter-matched MLP comes within a point of the ResNet, then the
convolutions are decoration and every subsequent architectural argument is
about a rounding error. Cheap to build, and the result is informative
whichever way it falls.

**ResNet (incumbent).** Convolutional residual trunk, kept as the reference
point. It is the only architecture with an existing trained checkpoint and
the only one whose numbers we already trust — but it will be **retrained
from scratch** on the current split, because the published checkpoint was
trained before the split fix and its validation number is not comparable to
anything produced now.

**ConstraintPoolNet.** Aggregate over the twelve constraint groups
explicitly: for each row, column and zone, pool the cell features belonging
to it, transform the pooled summary, and scatter it back to the member
cells. That is the game's rule structure written directly into the wiring,
and it is the one architecture in this lineup that can express "this shape
is blocked here" in a single layer rather than approximating it with stacked
kernels that do not align to the groups.

This is the candidate most likely to beat the ResNet per parameter, and the
one whose failure would be most informative — if hard-wiring the actual
rules does not help, the task is not constraint-shaped and we should stop
theorising about it.

### Written down, not built

**Attention encoder over 16 cell tokens.** Kept as an optional fourth, and
re-framed: the argument for it is *not* range, it is **content-dependent
interaction**. Convolution applies the same kernel regardless of what is on
the board; attention lets the weight between two cells depend on what
occupies them, which is closer to how the blocking rule actually works. It
is a weaker version of the ConstraintPoolNet hypothesis with more parameters
and no structural prior, which is exactly why it is fourth and not third.

**D4 x S4 equivariant network.** Quantik has 192 symmetries — 8 board
symmetries times 24 shape relabellings — and a network equivariant to all of
them would need to learn each position class once instead of 192 times.
In principle the largest sample-efficiency win available.

Declined because the corpus is already canonicalised *and* augmented, so
most of that win has already been bought with data rather than with
architecture; because a bug in the equivariance construction produces a
network that is quietly only approximately equivariant, which is very hard
to detect from a loss curve; and because the implementation cost is
comparable to everything else in this lineup combined. Worth revisiting if
data, not capacity, turns out to be the binding constraint.

**Constraint hypergraph network.** Cells and the twelve groups as a
bipartite graph, with message passing between them. ConstraintPoolNet *is*
this, restricted to one round of messages and a fixed group structure.
Building the general version first would mean paying for machinery before
knowing whether the restricted version already captures the win.

**Recurrent constraint propagation.** Iterate the message passing to a
fixed point, so effective depth adapts to how constrained a position is.
Elegant, and a plausible follow-up to a successful ConstraintPoolNet, but it
introduces a convergence question and a variable-latency forward pass into a
lineup that is currently trying to answer a much simpler comparison.

**Factorized shape-position policy head.** The 64 actions are 4 shapes x 16
positions; predict the two factors and combine them rather than emitting 64
independent logits. This is not an architecture — it is a head, orthogonal
to every trunk above, and it belongs in an ablation on the winner rather
than as a fourth entry in a trunk comparison.

**Engineered-feature baseline.** Hand-crafted features (blocked-shape
counts per group, threat counts, mobility) into a gradient-boosted tree.
Would establish how much of the task is trivially computable and would make
every neural result interpretable relative to a floor. Genuinely valuable,
genuinely out of scope for a comparison of *networks*; recorded here so that
it is a deferral rather than an oversight.

**Colour-ordered input encoding.** Already settled, and settled the other
way: everything is mover-relative — planes 0-3 belong to the side to move.
See `architectures.md`. Recorded here only because both layouts exist in the
codebase under the same contract name and the ambiguity has already caused
one wrong document.

## Decision

Train **MLP**, **ResNet** and **ConstraintPoolNet**. Add the **attention
encoder** as a fourth if the first three land cleanly.

## How the comparison is run

The lineup is worthless if the models are not comparable, so:

- **Every architecture is trained from scratch**, on the same corpus, the
  same split, the same optimizer and schedule, and the same budget. The
  incumbent ResNet checkpoint does not get a free pass on the strength of
  numbers produced under a different split.
- **`init_from` and layer freezing stay out of the comparison entirely.**
  They are retrain and fine-tune utilities, useful in their own right, and a
  model that started from someone else's weights is not evidence about its
  architecture.
- **Presets are parameter-matched to the ResNet**, so a difference is about
  shape rather than about who was allowed more capacity.
- **The split is keyed on the canonical position** (see PR #12). Before that
  fix, symmetric images of one position could land on both sides of the
  split, which inflates validation accuracy by an unknown amount and makes
  any cross-architecture difference unreadable.
- **Two evaluations, reported separately.** An IID canonical holdout, and a
  depth-shift set drawn from the shallow end of the game.

  The shift set is not something we have to construct — the corpus already
  has that shape. Per `runs/coverage.md`, plies 0-5 contain **zero** training
  positions, and ply 6 reaches 4.44% of its 901,916 canonical live positions.
  Plies 4 and 5 are therefore not "held out" in any meaningful sense; they
  were never in, which makes them a natively out-of-distribution probe rather
  than a constructed one. The existing held-out probe sets (1,240 positions
  at ply 4, 1,240 at ply 5, 1,280 at ply 6) are exactly that evaluation.

  This matters because the IID number flatters every model equally: it is
  measured on plies 6-13, which is not the regime an engine playing from the
  opening actually operates in. A model can be excellent on the holdout and
  useless for its first five moves, and only the second evaluation would say
  so.

## Outcome, 2026-08-28

All three trained at `medium` for 16 epochs on `exact-sampled.npz`. Both
evaluations ran; `shift-evaluation.md` has the detail.

| model | IID top-1 | shift, shallow (4-6) | shift, deep (7-12) |
|---|---|---|---|
| `resnet-c128-b6` | 0.9701 | **0.9126** | 0.9720 |
| `mlp-h455-b4` | 0.9516 | 0.8843 | 0.9578 |
| `cpool-c191-b6` | **0.9851** | 0.9092 | **0.9883** |

And a third, 2,400 games of autoplay (`autoplay.md`):

| start ply | 1st | 2nd | 3rd |
|---|---|---|---|
| 3 | `resnet` 53.7% | `cpool` 49.9% | `mlp` 46.4% |
| 6 | **`cpool` 53.9%** | `resnet` 48.8% | `mlp` 47.2% |
| 9 | `cpool` 51.2% | `resnet` 50.9% | `mlp` 47.9% |

Three things this settles, against what the section above predicted:

**The MLP loses, so the spatial prior is real.** At matched parameters it
trails by ~1.9 points of IID top-1 and considerably more on the value head.
The first branch below — "if the MLP matches the ResNet" — did not happen,
so `ConstraintPoolNet` is judged against the ResNet as planned.

**ConstraintPoolNet wins clearly**, which by the second branch promotes the
hypergraph and recurrent variants from "written down" to "next".

**The arena says the ranking depends on where the game is played.** At
ply-3 starts the ResNet leads and `cpool` beats nobody. At ply-6 starts
`cpool` beats the ResNet significantly (328-272, 54.7%, CI [50.7%, 58.6%])
and the ResNet's edge over the MLP evaporates. At ply-9 starts nothing is
significant — the positions are close enough to decided that move quality
stops mattering.

Each network's advantage is real and lives at a specific depth, and the
two measurements agree: accuracy said the ResNet is the better shallow
evaluator and `cpool` the better deep one; the arena says whoever is
stronger where the game is fought wins it.

So **"which architecture is better" is not well posed for this project**.
It depends on where play starts, which is a property of the deployment.
Any future architecture claim needs a game result at a stated start depth,
not only a validation number.

**And under search the differences largely vanish.** The same three
checkpoints inside `BatchedMCTS` at 128 simulations produce *no*
significant head-to-head result at either start depth — both of the
significant `net-policy` gaps disappear, and the ply-6 leaderboard
scrambles into noise. The network supplies a prior and a leaf value;
search corrects both, and 128 simulations on a 4x4 board is already past
the point where architecture is the binding constraint.

That has a direct consequence for this lineup: **the comparison is a
statement about the raw evaluator, not about the engine.** It is the right
comparison for choosing what to train and what to publish, and the wrong
one for predicting how a searching engine will play. `autoplay.md` records
both, and names the control (`uniform-mcts`, the same search with no
network) that separates "search washes out the differences" from "search
does all the work".

**And the third branch happened too, which qualifies the second.** `cpool`
wins the IID holdout and the deep probes and *loses the shallow ones*. The
prediction attached to that pattern — that the wiring "helps memorise the
trained distribution rather than generalise the rule" — is wrong as stated:
`cpool` generalises better to unseen positions, which is what the deep
probes are. What it does not do is extrapolate to **unseen plies**. Those
are different failures and the record should not have conflated them.

## What would change this decision

If the MLP matches the ResNet, the spatial prior is worth nothing here and
ConstraintPoolNet should be judged against the MLP, not the ResNet. If
ConstraintPoolNet wins clearly, the hypergraph and recurrent variants move
from "written down" to "next". If all three land within noise of each other,
the binding constraint is the corpus, not the architecture, and the next
work is data — which is what the autoplay loop is for.
