# The published models

Four networks, one contract, four Hugging Face repositories. This page is
what you need to pick one and run it. The design arguments are in
[`architectures.md`](architectures.md) and the per-architecture documents;
how the numbers were produced is in [`benchmarks.md`](benchmarks.md).

| model | Hub repository | parameters | what it is |
|---|---|---|---|
| `cpool` | [`brpoplpush/quantik-cpool-c191-b6`](https://huggingface.co/brpoplpush/quantik-cpool-c191-b6) | 1,780,253 | Constraint pooling. **The default.** |
| `attn` | [`brpoplpush/quantik-attn-d192-b6`](https://huggingface.co/brpoplpush/quantik-attn-d192-b6) | 1,800,709 | Transformer encoder over the 16 cells. |
| `resnet` | [`brpoplpush/quantik-resnet-c128-b6`](https://huggingface.co/brpoplpush/quantik-resnet-c128-b6) | 1,786,823 | Convolutional residual trunk. The incumbent. |
| `mlp` | [`brpoplpush/quantik-mlp-h455-b4`](https://huggingface.co/brpoplpush/quantik-mlp-h455-b4) | 1,788,343 | Dense control. Discards spatial structure. |

The four are parameter-matched within 1.2%, which is what makes comparing
them an architecture comparison rather than a capacity one.
`tests/test_parameter_matching.py` enforces it.

## Loading one

```bash
pip install 'quantik-models[torch,hub]'
```

```python
from quantik_models import hub
from quantik_models.env import fastboard as fb

evaluator = hub.load_evaluator("cpool")

boards = fb.empty_boards(1)               # (1, 8) uint16
legal = fb.legal_masks(boards)            # (1, 64) bool
policy, value = evaluator(boards, legal)  # (1, 64) priors, (1,) value
```

An evaluator is **callable**, and the legality mask is an argument rather
than something it works out for itself — the rules live in `quantik-core`,
and a caller running a search already has the mask. The priors it returns
are masked with it: probability on an illegal action is exactly zero, and
the legal entries sum to one.

`load_evaluator` downloads the repository, verifies the artifact the chosen
runtime will actually load against the matching digest in `manifest.json`
(`weights_hash` for safetensors, `onnx_hash` for the graph), and rebuilds
the network from `architecture_spec`. It caches, so calling it again in the
same process costs nothing.

**Without torch**, using the ONNX graph every repository ships:

```bash
pip install 'quantik-models[serve,hub]'    # onnxruntime is 80 MB, torch is 529
```

```python
evaluator = hub.load_evaluator("cpool", runtime="onnx")
```

The two runtimes are held to the same output by
`tests/test_onnx_evaluator_agreement.py`. They are two ways of running one
set of weights, not two models.

**Pin a revision for anything you report.** `main` can move:

```python
evaluator = hub.load_evaluator("cpool", revision="a6a122ef")
```

## The contract every model implements

    input   (B, 9, 4, 4) float32      tensor-board.v1, mover-relative
    output  (B, 64) policy logits     action_index = shape * 16 + position
            (B,)    value in [-1, 1]  +1 = good for the side to move

Two things about this are load-bearing and easy to get wrong:

**The encoding is mover-relative, not colour-ordered.** Planes 0-3 belong to
the side to move. `quantik_models.env.fastboard.encode_tensors` produces it;
`quantik_core.ml_data.qfen_to_tensor` produces the colour-ordered variant and
is *not* what these weights were trained on. Both are called
`tensor-board.v1`. Feeding the wrong one gives a model that plays legally and
is confidently wrong on every position with player 1 to move, with nothing
indicating a fault. The discriminating fixture is `"A.../..../..../...."`:
one piece, so `side_to_move == 1`, and mover-relative puts the 1.0 at
channel 4.

**Legality masking happens outside the model.** The rules are exact in
`quantik-core`, so the network never has to approximate them — and no engine
in this package can return an illegal move. `evaluator.evaluate` applies the
mask for you. If you run the raw ONNX graph yourself, the mask is your job;
`fastboard.legal_masks` computes it.

## How they compare

These are the numbers for **the published weights**, all measured in one
evaluation run at swept learning rates, `--preset medium`, 16 epochs.

| model | IID top-1 | shift, plies 4-6 | shift, plies 7-12 | arena @ply3 | vs `minimax-d2` |
|---|---|---|---|---|---|
| `cpool` | **0.9893** | **0.9295** | **0.9919** | **57.2%** | **49.4%** |
| `attn` | 0.9879 | 0.9102 | 0.9914 | 54.2% | 43.1% |
| `resnet` | 0.9701 | 0.9126 | 0.9720 | 47.8% | 36.5% |
| `mlp` | 0.9516 | 0.8843 | 0.9578 | 40.8% | 31.9% |

Four columns because they disagree, and the disagreement is the finding:

1. **IID top-1** — validation accuracy on the training distribution.
2. **Shift** — accuracy on solved positions the corpus never saw, held out
   up to all 192 symmetries. [`shift-evaluation.md`](shift-evaluation.md).
3. **Arena** — games against the other three networks.
   [`autoplay.md`](autoplay.md).
4. **vs `minimax-d2`** — games against a fixed two-ply alpha-beta search.
   [`oracle-benchmark.md`](oracle-benchmark.md). The only column whose floor
   does not move with the field, and the only one that answers "is any of
   this good" rather than "which of these is better".

`cpool` playing raw policy, one forward pass per move, is even with a
two-ply search. The other three lose to it.

### Three things to know before quoting any of this

**Held-out accuracy does not predict play strength here.** It has now failed
to four separate times. When validation top-1 and the arena disagree, the
arena is the one that answers the question anyone actually has.

**The seat dwarfs the model.** Mover win rates run 68-88%, responder 15-39%.
Two networks a point apart are being compared inside an effect forty times
larger, which is why every number above is side-balanced. Never quote an
unbalanced win rate.

**Newer checkpoints exist and are not what is published.** The
`patience-{arch}` runs (`--patience 5 --epochs 60`, 2026-08-30) score higher
on validation top-1 — `cpool` 0.9916 against 0.9893 — and are *not* on the
Hub. What is published is the fixed-16-epoch lineup the table above
describes, so the table and the weights match.
[`decisions/0001-architecture-lineup.md`](decisions/0001-architecture-lineup.md)
carries the patience results and what they did and did not change.

## Licensing

**The weights and the code are under different licences, deliberately.**

- This package, and everything in this repository: **MIT**.
- The weights in the four Hub repositories: **CC BY-NC 4.0** —
  non-commercial, and not an OSI-approved licence.

So a commercial application may use the training pipeline, the rules engine
and the evaluation harness freely, and may not ship these weights. Train
your own on the same pipeline and they are yours.

## Loading a checkpoint you trained

Nothing above is Hub-specific. A local `runs/train/*/best` directory is the
same thing:

```python
from quantik_models.arena.registry import load_evaluator

evaluator = load_evaluator("runs/train/my-run/best")
```

The two directory layouts differ in one detail — a Hub repository names its
weights `model.safetensors` and a local checkpoint names them
`weights.safetensors` — and the loader reads either. Publishing your own is
[`publishing-to-hugging-face.md`](publishing-to-hugging-face.md).
