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

registry.architectures()          # ('resnet',)
registry.presets("resnet")        # ('small', 'smoke', 'target')

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
reimplementing the architecture there. Three details are load-bearing:

- **Exported in eval mode**, so batch norm folds its running statistics.
  A graph exported in train mode gives different answers for the same
  position depending on what else is in the batch.
- **`external_data=False`**, so the graph and its weights stay in one file.
  The exporter otherwise spills tensors into a sibling `model.onnx.data`
  that `onnx_hash` would not cover.
- **Opset 17**, which has native LayerNorm; older opsets decompose it into
  a noisier subgraph.

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
| `target` | 256 | 13 | 15,374,023 |

`resnet-c128-b6` (1,786,823 parameters) is the published model and is
reached with `--preset small --channels 128 --blocks 6`.

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
