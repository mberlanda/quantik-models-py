# ConstraintPoolNet: wiring the rules into the network

`cpool-c191-b6` — 1,780,253 parameters, matched to `resnet-c128-b6`'s
1,786,823.

## The hypothesis

Quantik's rule has nothing to do with spatial adjacency. A shape may not be
placed where the **opponent** already has that shape in the same row,
column, or 2x2 zone. That is twelve overlapping groups — four rows, four
columns, four zones — with every cell belonging to exactly three of them.
The same twelve groups are also the win conditions: a group holding all
four shapes ends the game.

The board is small enough that a convolutional trunk can see all of it, so
the usual reason to prefer one architecture over another does not apply
here. What differs is how naturally each one can *express* a group-wise
predicate.

A 3x3 convolution aligns to none of the twelve. A row is a 1x4 strip; a
zone is a 2x2 block; a kernel centred on a cell covers part of three rows,
part of three columns, and parts of up to four zones. To compute "does the
opponent have shape C anywhere in my row", a convolutional network has to
compose several layers of local evidence into something that happens to
sum over exactly the right four cells and no others — learnable, certainly,
but learned rather than given.

This architecture gives it. Each block pools cell features into the twelve
groups, transforms them there, and scatters the result back to the member
cells: one round of bipartite message passing over the game's own
constraint structure.

**This is the candidate most likely to beat the ResNet per parameter, and
the one whose failure would be most informative.** If hard-wiring the
actual rules does not help, the task is not constraint-shaped and we should
stop theorising about it and go get more data instead.

## Where the constraint structure comes from

The twelve groups are not restated by hand. `model.spec.constraint_groups()`
derives them from the board size, and `tests/test_constraint_groups.py`
asserts the result is the same twelve sets as `env.fastboard.WIN_MASKS` —
the masks the engine itself uses for legality and for win detection.

That test exists because the alternative is a network wired to a rule the
engine no longer plays by, with nothing failing to say so.

## The layers, one at a time

Input `(B, 9, 4, 4)`, mover-relative per `tensor-board.v1`.

### Stem: cells as tokens

```
flatten(2).transpose(1, 2)     (B, 9, 4, 4) -> (B, 16, 9)
Linear(9 -> C)                 9C + C parameters
```

The board becomes sixteen tokens, each carrying its nine plane values.
From here on there is no 2-D grid: position is meaningful only through
group membership, which is the point.

### Membership matrices

Two constant `(12, 16)` and `(16, 12)` matrices, registered as
non-persistent buffers rather than parameters — the constraint structure is
the game's, not something to learn. Being buffers, they follow the model
onto whatever device it moves to and are baked into the ONNX graph as
constants.

Both are **mean**-reducing. Every group has four cells and every cell three
groups, so mean and sum differ only by a constant here; mean keeps the
pooled summaries on the same scale as the cell features they are about to
be concatenated with, which matters because a LayerNorm sits upstream and
not between.

### Trunk: `blocks` x constraint block

```
h      = LayerNorm(x)                                (B, 16, C)
groups = mean over member cells of h                 (B, 12, C)
groups = groups + kind_embedding[kind]               (B, 12, C)
groups = Linear(C->C) -> GELU -> Linear(C->C)        2C^2 + 2C
back   = mean over a cell's three groups             (B, 16, C)
x      = x + Linear(2C -> C)([h, back])              2C^2 + C
x      = x + FFN(LayerNorm(x))                       2 * E * C^2 + ...
```

Per block, roughly `(4 + 2E) C^2` parameters, with `E = 2` the FFN
expansion. Two design choices are load-bearing:

**The group transform is shared across all twelve groups.** What
distinguishes them is a learned embedding indexed by *kind*, added to the
pooled summary. Twelve independent group networks would be twelve little
models rather than one statement about constraint structure, and the
parameter count would scale with the number of groups for no reason —
there is nothing special about the third row that is not also true of the
first.

**There are two kinds, not three.** Rows and columns share one embedding;
zones have their own. Transposition is an element of D4, it exchanges rows
with columns, and it maps the zone partition onto itself — so tying rows
and columns is consistent with a symmetry the game actually has, not merely
economical. `test_transposing_the_board_swaps_lines_and_preserves_zones`
asserts exactly that, so if the geometry ever changed, the tying would stop
being justified loudly rather than silently.

**LayerNorm, not BatchNorm.** Two reasons. It removes the batch-of-one
failure that only ever appears in serving, where a single position is
evaluated and a `BatchNorm` in train mode has no variance to divide by. And
opset 17 has native LayerNorm, so the exported graph is one op rather than
a decomposed and noisier subgraph.

### Policy head, and the transpose that is easy to get wrong

```
Linear(C -> 4)                 per-cell logits, (B, 16, 4)
transpose(1, 2).reshape        -> (B, 64)
```

Actions are indexed `action_index = shape * 16 + position`, so **shape is
the outer axis**. A head that emits `(cell, shape)` and flattens it
directly produces `position * 4 + shape` — a tensor of exactly the right
shape and dtype, with every logit on the wrong action. The network would
train against a permuted target and either learn the permutation or fail
to, and no shape assertion anyone would think to write would catch it.

This is why the flatten lives in a named function,
`flatten_cell_shape_logits`, with two tests: one asserting the mapping cell
by cell, and one asserting that the naive flatten differs from it *and is
the same multiset*, which is precisely why the mistake survives every
casual check.

The ResNet does not have this problem: its policy head produces the 64
logits directly from a flattened trunk, so there is no cell/shape axis to
get backwards.

### Value head

```
mean over the 16 cells -> Linear(C -> 64) -> GELU -> Linear(64 -> 1) -> Tanh
```

Mean-pooled rather than read from a designated token, so the value is
invariant to any relabelling of cells that the trunk has not already broken
— a value estimate should not depend on which cell is "first".

Tanh, so the output is in `[-1, 1]` as the contract requires, and
mover-relative like the input: `+1` is good for the side to move.

## Presets

| preset | channels | blocks | parameters | matched against |
|---|---|---|---|---|
| `smoke` | 16 | 2 | 5,893 | CI only |
| `small` | 96 | 4 | 307,333 | `resnet-c64-b4`, 304,711 (+0.9%) |
| `medium` | 191 | 6 | 1,780,253 | `resnet-c128-b6`, 1,786,823 (-0.4%) |

Depth mirrors the ResNet's — four blocks at `small`, six at `medium` — so
the width is the only free variable and is solved for the parameter match.

## An ONNX trap this architecture hit

The pooling started life as `pool @ h`, a 2-D constant matmul against a 3-D
activation. Torch traces that to a `Gemm` with the batch dimension folded
into a hard-coded `Reshape`, so the exported graph ran correctly at the
batch size it was traced with and failed on every other one:

```
Reshape node ... Input shape:{4,12,16}, requested shape:{12,16}
```

Rewritten as `F.linear` on a transposed view, which contracts the last axis
and leaves the batch dimension dynamic. `test_onnx_export_matches_torch`
runs a batch of four against a graph traced with a batch of one, which is
what caught it — a round-trip test at the traced batch size would have
passed.

> **Provenance.** Everything below was measured at `--lr 2e-3`, the
> trainer's old global default — chosen for the ResNet, the only
> architecture that existed when it was set. Until the learning-rate sweep
> finishes these are comparisons at *the ResNet's* preferred setting. See
> `attention-negative-result.md`.

## What happened

The three falsification conditions this section used to list in the future
tense have all been tested.

| | IID top-1 | shift 4-6 | shift 7-12 | arena @ply3 | arena @ply6 |
|---|---|---|---|---|---|
| `resnet-c128-b6` | 0.9701 | **0.9126** | 0.9720 | **53.7%** | 48.8% |
| `mlp-h455-b4` | 0.9516 | 0.8843 | 0.9578 | 46.4% | 47.2% |
| `cpool-c191-b6` | **0.9851** | 0.9092 | **0.9883** | 49.9% | **53.9%** |

**It did not tie the MLP.** The gap is 3.4 points of IID top-1 and more
than double on the value head, so the group structure is buying something
144 flat features do not give.

**It did not tie the ResNet either — it beat it, on the deep probes.**
1.63 points on held-out positions at trained plies, with less than half the
value error at ply 6 and deeper.

**And the third condition happened**, which is the interesting one: it wins
the IID holdout and the deep probes and **loses the shallow ones**. The
explanation attached to that prediction — that the wiring helps memorise
rather than generalise — turns out to be wrong. `cpool` generalises *better*
to unseen positions; the deep probes are entirely held out and it wins them
clearly. What it fails at is extrapolating to unseen **plies**. Those are
different failures and only the second is happening.

## Where the shallow deficit actually lives

The original explanation for that deficit was that sparse groups carry no
signal, and it named the check: the advantage should track group occupancy
rather than ply.

Group occupancy is how many of the twelve groups hold at least one piece.
At a fixed ply it measures concentration inversely — every piece belongs to
three groups, so `3 x ply` memberships spread across more groups leaves
fewer pieces in each. Four pieces touching all twelve leaves one per group;
four pieces touching five leaves nearly two and a half.

Splitting each ply at its own median occupancy and comparing against the
ResNet with an exact McNemar test — paired, because only the positions where
the two disagree carry information:

| ply | bucket | n | `resnet` | `cpool` | difference | p |
|---|---|---|---|---|---|---|
| 4 | occ<=9 | 687 | 0.8967 | 0.8908 | -0.0058 | 0.74 |
| 4 | **occ>9** | 231 | 0.8268 | 0.7273 | **-0.0996** | **0.0006** |
| 5 | occ<=10 | 690 | 0.9232 | 0.9275 | +0.0043 | 0.79 |
| 5 | **occ>10** | 301 | 0.9037 | 0.8439 | **-0.0598** | **0.0064** |
| 6 | occ<=11 | 728 | 0.9547 | 0.9698 | +0.0151 | 0.09 |
| 6 | **occ>11** | 258 | 0.8953 | 0.9767 | **+0.0814** | **<0.0001** |
| 7 | occ<=11 | 544 | 0.9669 | 0.9908 | **+0.0239** | **0.0023** |
| 7 | occ>11 | 512 | 0.9414 | 0.9727 | **+0.0312** | **0.0052** |

**Occupancy is a strong moderator**, much stronger than the per-ply numbers
suggested. Within ply 4, `cpool` is statistically tied with the ResNet on
low-occupancy positions and loses by ten points on high-occupancy ones. The
ply-4 deficit is not spread across ply 4; it is concentrated in a quarter
of it.

**But the direction is not constant.** The prediction was that thin pooled
summaries hurt `cpool` — the high-occupancy case at fixed ply — and that
holds at plies 4 and 5. It *reverses* by ply 6, where `cpool`'s advantage is
largest precisely on high-occupancy positions. No single "sparse groups
carry no signal" story produces both signs, so the one-line explanation was
too simple.

**The useful part survives: the shallow weakness is not general.** `cpool`
is not worse in the opening. It is worse in a specific, identifiable kind of
opening position — four or five pieces scattered thin, roughly one to a
group. Those are also the positions every model finds hardest, since the
ResNet drops from 0.8967 to 0.8268 across the same split. `cpool` degrades
faster there and is otherwise its equal.

What explains the sign flip is not established. One untested possibility: at
ply 6 a thinly populated group still holds 1.5 pieces on average against 1.0
at ply 4, so there may be a threshold below which a pooled summary carries
nothing. That is a story, not a finding, and it is recorded only to say what
has not been ruled out.

```bash
python -m quantik_models.eval.shift --checkpoint runs/train/lineup-cpool/best
# occupancy_split() in the same module produces the table above
```

## Under search, none of this survives

At 128 MCTS simulations the `cpool`-versus-`resnet` difference disappears
entirely — 51.8% from ply-6 starts, not significant, against a significant
54.7% without search. The network still matters enormously (the same
search with uniform priors loses 99.5% of games), but the *margin between
these architectures* is below what search can resolve. See `autoplay.md`.


## Training it

```bash
.venv/bin/python -m quantik_models.train.supervised \
  --arch cpool --preset medium \
  --corpus runs/oracle/corpus/exact-sampled.npz \
  --name sup-cpool-c191b6
```

Same corpus, same split, same budget as every other architecture — see
`decisions/0001-architecture-lineup.md`.
