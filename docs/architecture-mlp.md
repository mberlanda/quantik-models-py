# The MLP baseline: what a 4x4 board owes to convolution

`mlp-h455-b4` — 1,788,343 parameters, matched to `resnet-c128-b6`'s
1,786,823.

## Why this network exists

Convolution is justified by two properties of the input: **locality**,
meaning that nearby cells interact more than distant ones, and **weight
sharing**, meaning that the same pattern is worth detecting wherever it
appears. Both are overwhelmingly true of a 19x19 Go board. Neither is
obviously true of a 4x4 one.

On 4x4, locality is nearly vacuous. Every cell is within two steps of every
other, and the ResNet's stem plus its first residual block already covers
the entire board — after that, depth adds capacity, not reach. Weight
sharing is a stronger argument, but a translated pattern on a 4x4 grid has
at most a handful of positions to be translated to, and Quantik's rule is
not translation-invariant in the way Go's is: the 2x2 zones are fixed to a
particular quartering of the board, so a pattern that matters in the
top-left zone does not mean the same thing shifted one cell right.

So the convolutional prior might be doing real work here, or it might be a
habit inherited from AlphaZero. That is an empirical question, and this
network is how it gets asked. It discards spatial structure entirely: the
`(9, 4, 4)` input is flattened to 144 features and pushed through dense
residual blocks. Whatever the ResNet gains from convolution is the gap.

**Either result is worth having.** If the MLP matches, the convolutions are
decoration and every subsequent architectural argument on this board is
about a rounding error — which would also mean `ConstraintPoolNet` should
be judged against the MLP rather than the ResNet. If the MLP loses badly,
the spatial prior is load-bearing, and we have quantified by how much
instead of assuming it.

## The layers, one at a time

Input `(B, 9, 4, 4)`, mover-relative per `tensor-board.v1`.

### Stem

```
Flatten                144 = 9 planes x 16 cells
Linear(144 -> h)       bias=False, 144h parameters
BatchNorm1d(h)         2h parameters (scale, shift) + 2h buffers
ReLU
```

`bias=False` throughout, because a batch norm immediately downstream has
its own shift and would make the bias unidentifiable — the two parameters
would be free to drift against each other with no effect on the output.

The flatten is the whole architectural claim. After this line, the network
has no idea that feature 17 and feature 21 were vertically adjacent cells,
or that features 0-15 were one shape plane. It has to relearn any structure
it needs from the data — which, with three million labelled positions, is
not obviously a losing proposition.

### Trunk: `blocks` x pre-activation dense residual block

```
x -> Linear(h -> h, bias=False) -> BatchNorm1d(h) -> ReLU
  -> Linear(h -> h, bias=False) -> BatchNorm1d(h)
  -> add x -> ReLU
```

Per block: `2h^2 + 4h` parameters. This is where essentially all the
capacity sits, and it is why the width that parameter-matches a given
ResNet is not a number anyone would guess: total parameters grow as
`2 * blocks * h^2`, so matching 1.79M at four blocks lands on `h = 455`
rather than on a round 512.

The residual connection is not decoration. Without it, four stacked dense
layers with batch norm train visibly worse, and any observed gap would then
be confounded by an optimisation difference rather than an architectural
one — the network would be answering a question about trainability instead
of about inductive bias. Residual here, residual in the ResNet, same
justification.

### Policy head

```
Linear(h -> 64)        64h + 64 parameters
```

Sixty-four logits, indexed `action_index = shape * 16 + position` with
`position = row * 4 + col`. No transpose is needed: the head emits the 64
actions directly in contract order, unlike an architecture that produces
per-cell features and has to be careful about which axis it flattens.

The head is a single linear layer on purpose. Adding a hidden layer here
would move capacity out of the trunk and into the head, which is a
different experiment than the one this network is for.

### Value head

```
Linear(h -> 64) -> ReLU -> Linear(64 -> 1) -> Tanh -> squeeze
```

Tanh, so the output is in `[-1, 1]` as `policy-logits-64+value-tanh`
requires and as a runtime may rely on. The value is mover-relative like the
input: `+1` means good for the side to move, not good for player 0. That is
what the mover-relative encoding buys — a single value head with one sign
convention, rather than a head that must learn to read plane 8 and flip its
own answer.

## Presets

| preset | hidden | blocks | parameters | matched against |
|---|---|---|---|---|
| `smoke` | 32 | 1 | 11,137 | nothing; CI only |
| `small` | 178 | 4 | 305,285 | `resnet-c64-b4`, 304,711 (+0.2%) |
| `medium` | 455 | 4 | 1,788,343 | `resnet-c128-b6`, 1,786,823 (+0.1%) |

`tests/test_parameter_matching.py` asserts the match to within 5% and
re-derives the ResNet's own counts, so a preset cannot drift out of
comparability silently. The first draft of these presets was off by 2x in
exactly that way.

## What this network cannot tell us

It is a control, not a contender. Three limits worth stating before anyone
reads too much into its number:

- **A tie is not proof of equivalence.** Matched parameters is not matched
  effective capacity, and a dense layer uses its parameters differently from
  a convolution. A tie means the convolutional prior did not pay for itself
  *at this corpus size*; more data could change that in either direction.
- **It says nothing about the constraint hypothesis.** `ConstraintPoolNet`
  asks whether Quantik's twelve-group structure is worth wiring in. The MLP
  has no more access to that structure than the ResNet does, so it is a
  baseline for both, not a comparison against either.
- **It is not a serving candidate.** At 1.79M dense parameters it is
  arithmetically comparable to the ResNet but has no locality to exploit,
  which makes it a worse fit for the small-batch, single-position inference
  a game server actually does.

## Batch norm and a batch of one

`BatchNorm1d` in training mode raises on a batch of one — there is no
variance to normalise by. Serving evaluates single positions, so the model
must be in eval mode, where the folded running statistics are used instead.
`export_onnx` handles this by exporting in eval mode, and
`test_batch_size_one_works` guards it for every registered architecture.
This is not specific to the MLP, but it is the failure that only ever
appears in production and never in a batched training loop.

## Training it

```bash
.venv/bin/python -m quantik_models.train.supervised \
  --arch mlp --preset medium \
  --corpus runs/oracle/corpus/exact-sampled.npz \
  --name sup-mlp-h455b4
```

Same corpus, same split, same budget as every other architecture in the
lineup — see `decisions/0001-architecture-lineup.md` for why that matters
and what would invalidate the comparison.
