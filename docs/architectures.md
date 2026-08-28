# Architectures

Every model in this package answers the same question in a different way,
and they are all interchangeable because they agree on one contract:

```
input   (B, 9, 4, 4) float32      tensor-board.v1
output  (B, 64) policy logits     policy-logits-64+value-tanh
        (B,)    value in [-1, 1]
```

Nine input planes and sixty-four output logits, the latter indexed
`action_index = shape * 16 + position` with `position = row * 4 + col`.

**The plane order is mover-relative, and this matters more than it looks.**
`fastboard.encode_tensors` — which is what both `train/supervised.py` and
`selfplay/evaluator.py` feed the network — puts the side-to-move's four
shapes in channels 0-3, the opponent's in 4-7, and the side-to-move flag in
channel 8. The plane order therefore swaps with parity.

`quantik_core.ml_data.qfen_to_tensor` orders the same nine planes by colour
instead: player 0 first, always. Both encodings call themselves
`tensor-board.v1` and both are valid readings of its 9x4x4 shape, but they
are not interchangeable. A runtime that builds the colour-ordered tensor
and feeds it to a checkpoint trained on the mover-relative one gets the two
players swapped on every position where player 1 is to move — half of them
— and the result is a model that plays legally and badly, with nothing to
indicate anything is wrong.

`fastboard.to_core_tensor` produces the colour-ordered layout for interop.
Nothing in the training or serving path currently uses it.

Those constants live in `quantik_models.model.spec` so that no architecture
restates them, and an architecture that emits per-cell logits has to
transpose before flattening or every logit lands on the wrong action.

**Legality masking is not part of any model.** It is applied outside, by
`masked_log_softmax`, using the same code path in training and at
inference. The rules are already exact in `quantik-core`; a network that
learned them from data would approximate something that can be computed
perfectly. This is the standard arrangement — AlphaZero, Leela and NNUE all
do it — and it means no engine in this project can return an illegal move,
because the mask sets illegal logits to the dtype minimum and softmax sends
them to exactly zero.

## The registry

`quantik_models.model.registry` maps a short name to a constructor, a
config type and a preset table. Training, export and evaluation all go
through it, which is what lets a checkpoint be loaded without the caller
knowing in advance which architecture produced it: the manifest names the
architecture, the registry resolves it.

```python
from quantik_models.model import registry

registry.architectures()          # ('cpool', 'mlp', 'resnet')
registry.presets("resnet")        # ('medium', 'small', 'smoke', 'target')

model = registry.build("resnet", preset="small")
model = registry.build("resnet", preset="small", channels=128, blocks=6)
model.architecture                # 'resnet-c128-b6'
model.model_family                # 'quantik-policy-value-resnet'
```

`architecture` and `model_family` are read straight into the
`model-checkpoint.v1` manifest, so they are part of the published contract
rather than incidental metadata. Adding an architecture therefore requires
nothing of the exporter.

### Adding one

1. Write the module under `quantik_models/model/`, taking its shape
   constants from `spec` and exposing `architecture` and `model_family`.
2. Register it with `registry.register(...)`, giving it a preset table.
   Parameter-match the presets against the ResNet so a comparison is about
   shape rather than size.
3. That is all. `tests/test_architecture_registry.py` runs over everything
   registered — forward shapes, value bounds, batch-of-one, identity
   strings, and ONNX/torch agreement — so a new architecture is covered the
   moment it is added.

## Training with one

`--arch` selects the architecture; `--preset` selects its size; `--channels`
and `--blocks` override the preset for ablations. An architecture that has
no notion of `channels` simply ignores it, so one CLI drives all of them.

```bash
.venv/bin/python -m quantik_models.train.supervised \
  --arch resnet --preset small --channels 128 --blocks 6 \
  --corpus runs/oracle/corpus/exact-sampled.npz \
  --name sup-sampled-c128b6
```

## Export: two artifacts, both hashed

`export_checkpoint` writes both:

| file | what it is | why |
|---|---|---|
| `weights.safetensors` | named tensors, no graph | the primary; what `weights_hash` covers |
| `model.onnx` | graph **and** weights | executable by a runtime that has never seen this package |

The ONNX artifact is what makes a checkpoint consumable from Rust without
reimplementing the architecture there. Five details are load-bearing, and
three of them were learned the hard way:

- **Exported in eval mode**, so batch norm folds its running statistics.
  A graph exported in train mode gives different answers for the same
  position depending on what else is in the batch.
- **`external_data=False`**, so the graph and its weights stay in one file.
  The exporter otherwise spills tensors into a sibling `model.onnx.data`
  that `onnx_hash` would not cover.
- **Traced at batch 2, not batch 1.** `torch.export` specializes any
  dimension of size 0 or 1. A graph traced with a single example is frozen
  at batch one *while still advertising a symbolic batch dimension* — the
  input signature says `['batch', 9, 4, 4]` and an internal `Reshape`
  says otherwise.
- **`dynamic_shapes`, not `dynamic_axes`.** The dynamo exporter ignores
  `dynamic_axes` and warns that it does.
- **Opset 18**, which is the dynamo exporter's floor. Requesting 17 gets a
  graph at 18 followed by a down-conversion that fails *silently* for some
  architectures, leaving a file at 18. `onnx_opset` in the manifest is
  therefore read back from the exported file rather than recorded from the
  request — the two disagreed for `cpool`.

`weights_format` stays `"safetensors"` because the contract admits a single
value. The ONNX artifact is recorded beside it as `onnx_export`, with its
own `onnx_hash`, so a runtime can verify whichever one it actually loads —
a manifest that named only the safetensors while a server ran the ONNX
would be describing something other than what it serves.

The ONNX exporter needs `onnxscript`, which the `torch` extra does not
pull in:

```bash
pip install -e ".[dev,torch,onnx]"
```

Pass `with_onnx=False` to skip it where torch alone is available.

## After a run: two evaluations, not one

The validation number a training run prints is measured on the trained
distribution. `docs/shift-evaluation.md` measures what survives leaving it,
on 7,800 exactly-solved positions sharing no canonical key with the corpus:

```bash
.venv/bin/python -m quantik_models.eval.shift \
  --checkpoint runs/train/lineup-cpool/best --out runs/eval/shift.json
```

Report both. On this lineup the two rank the architectures differently in
the shallow plies, which is the regime an engine opens from.

And then a third, because neither predicts playing strength: `docs/autoplay.md`
has the arena result, where the architecture leading both accuracy tables
wins no games.

## Before a long run: the preflight

```bash
.venv/bin/python -m quantik_models.train.preflight \
  --corpus runs/oracle/corpus/exact-sampled.npz \
  --preset medium --epochs 16
```

About a minute per architecture, running the real code paths —
`load_corpus`, `split_by_key`, `_forward_losses`, `export_checkpoint` — on
a handful of batches. It checks that the corpus carries what it claims,
that the split leaks no canonical key, that every parameter receives
gradient, that the loss falls on a fixed batch, that the masked argmax of
a single position is legal, and that the exported graph agrees with torch
at three batch sizes. Then it projects a wall-clock per architecture, so a
run is budgeted before it is started.

It earned its keep on the first invocation: `cpool` at `medium` exported a
graph that failed at any batch but the traced one, which the unit tests had
missed because they only exercised the `smoke` preset.

## Registered architectures

### `resnet` — the incumbent

Convolutional residual trunk. A 3x3 stem lifts the 9 input planes to `C`
channels, `B` residual blocks follow, and two heads read the shared trunk:
a 1x1 convolution to 2 channels then a linear layer to 64 logits, and a 1x1
convolution to 1 channel then a small MLP to a tanh value.

| preset | channels | blocks | parameters |
|---|---|---|---|
| `smoke` | 16 | 2 | 13,991 |
| `small` | 64 | 4 | 304,711 |
| `medium` | 128 | 6 | 1,786,823 |
| `target` | 256 | 13 | 15,374,023 |

`resnet-c128-b6` is the published model, and `medium` is it. It used to be
reachable only as `--preset small --channels 128 --blocks 6`, which made
the project's flagship size an incantation rather than a name.

See `decisions/0001-architecture-lineup.md` for which architectures are in
the comparison, which were declined and why, and how the comparison is
kept fair.

Note for anyone reasoning about receptive fields: on a 4x4 board the stem
plus one residual block already covers the whole position. Depth here buys
representational capacity, not reach — which is why "this architecture sees
globally and that one does not" is never a meaningful distinction on this
game, and why alternatives have to be justified on different grounds.

See `policy-value-training-paper.md` for how it is trained and why it is
distilled from search rather than learned from self-play.

### `mlp` — the control

Flattened dense trunk: the `(9, 4, 4)` input becomes 144 features and goes
through pre-activation residual dense blocks, with the same two heads. It
throws spatial structure away deliberately, to make "convolution is worth
having on a 4x4 board" a falsifiable claim rather than an assumption.

| preset | hidden | blocks | parameters | matched against |
|---|---|---|---|---|
| `smoke` | 32 | 1 | 11,137 | CI only |
| `small` | 178 | 4 | 305,285 | `resnet-c64-b4` (+0.2%) |
| `medium` | 455 | 4 | 1,788,343 | `resnet-c128-b6` (+0.1%) |

Widths are solved against the ResNet rather than chosen for roundness —
dense parameters grow as `2 * blocks * hidden^2`, so the matching width is
never a round number, and `tests/test_parameter_matching.py` keeps it
honest. A control carrying twice the incumbent's capacity would measure
capacity rather than architecture, which is what the first draft of these
presets did.

See `architecture-mlp.md` for the layer-by-layer account and for what this
network cannot tell us.

### `cpool` — the constraint model

Message passing over Quantik's twelve constraint groups (four rows, four
columns, four 2x2 zones). Each block pools the sixteen cell tokens into the
groups they belong to, transforms them there, and scatters the result back
to the member cells — the game's rule structure written into the wiring
rather than approximated by stacked kernels that align to none of it.

| preset | channels | blocks | parameters | matched against |
|---|---|---|---|---|
| `smoke` | 16 | 2 | 5,893 | CI only |
| `small` | 96 | 4 | 307,333 | `resnet-c64-b4` (+0.9%) |
| `medium` | 191 | 6 | 1,780,253 | `resnet-c128-b6` (-0.4%) |

The groups are derived in `model.spec`, not restated by hand, and
`tests/test_constraint_groups.py` asserts they are the same twelve sets as
the engine's `WIN_MASKS`. Rows and columns share one set of group weights
because transposition is in D4; zones have their own.

This architecture emits **per-cell** logits, so it is the one that has to
transpose before flattening — `flatten_cell_shape_logits`, tested directly,
because `position * 4 + shape` is the right shape and dtype and completely
wrong.

See `architecture-constraint-pool.md` for the layer-by-layer account and
for what would falsify the hypothesis.
