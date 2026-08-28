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

## What would falsify the hypothesis

- **ConstraintPoolNet ties the MLP.** Then the group structure is not
  buying anything the network could not infer from 144 flat features, and
  the whole line of reasoning in this document is decoration.
- **ConstraintPoolNet ties the ResNet.** Weaker evidence: it would mean the
  convolutional trunk already approximates the group predicates well
  enough at this corpus size, which is a statement about the corpus as much
  as about the architectures.
- **It wins on the IID holdout but not on the shallow probes.** The most
  interesting failure. Plies 0-5 carry no training positions at all, so
  that pattern would say the group wiring helps memorise the trained
  distribution rather than generalise the rule — the opposite of what an
  inductive bias is supposed to do.

## Training it

```bash
.venv/bin/python -m quantik_models.train.supervised \
  --arch cpool --preset medium \
  --corpus runs/oracle/corpus/exact-sampled.npz \
  --name sup-cpool-c191b6
```

Same corpus, same split, same budget as every other architecture — see
`decisions/0001-architecture-lineup.md`.
