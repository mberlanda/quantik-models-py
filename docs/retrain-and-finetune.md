# Retraining and fine-tuning

> **About the numbers in this document.** Every measurement here comes from
> a specific run of a specific checkpoint, and the checkpoints live under
> `runs/`, which is gitignored — so nothing here can be verified from a
> fresh clone alone. Two things are worth checking before trusting a figure:
> **which learning rate it was measured at**, and **when**. Anything dated
> before 2026-08-30 was measured at `--lr 2e-3`, a rate chosen for the
> ResNet and inherited by every architecture added later; two of the four
> were being trained at the wrong one, and correcting that reversed several
> conclusions rather than merely shifting decimals. See
> `learning-rate-sweep.md`. Regenerate everything with
> `scripts/evaluate_lineup.sh`.


Three ways to start a run, in increasing order of how much of a previous
model they keep.

| | `--init-from` | `--freeze` | what it is for |
|---|---|---|---|
| **from scratch** | no | no | architecture comparisons; a fresh corpus |
| **retrain / warm start** | yes | no | more or better data, same architecture |
| **fine-tune** | yes | yes | adapt one part; keep the rest exactly |

## From scratch

The default, and what every architecture comparison must use. A model that
started from someone else's weights is not evidence about its architecture
— see `decisions/0001-architecture-lineup.md`.

```bash
.venv/bin/python -m quantik_models.train.supervised \
  --arch cpool --preset medium \
  --corpus runs/oracle/corpus/exact-sampled.npz --name lineup-cpool
```

## Retrain: warm start on the full dataset

Keeps the weights, trains everything. The natural move when the corpus
grows — for instance after autoplay positions have been solved and folded
in with `python -m quantik_models.data.merge_corpus` (see `autoplay.md`).

```bash
.venv/bin/python -m quantik_models.train.supervised \
  --arch resnet --preset medium \
  --corpus runs/oracle/corpus/exact-sampled-v2.npz \
  --init-from runs/train/lineup-resnet/best \
  --name resnet-v2 --lr 5e-4
```

A lower learning rate than the original run is usually right: the cosine
schedule restarts from `--lr`, and the default 2e-3 will walk a converged
model a long way from where it was before it starts improving.

`--init-from` loads the state dict strictly, so the architecture must
match. That is deliberate — a warm start that silently dropped
mismatched tensors would train a model that is part new and part stale,
and report nothing.

## Fine-tune: hold most of it fixed

```bash
.venv/bin/python -m quantik_models.train.supervised \
  --arch resnet --preset medium \
  --corpus runs/oracle/corpus/shallow.npz \
  --init-from runs/train/lineup-resnet/best \
  --freeze stem,trunk --name resnet-shallow-heads --lr 5e-4
```

`--freeze` takes a comma-separated list of **dotted module prefixes**:

| pattern | freezes |
|---|---|
| `stem` | the input projection |
| `trunk` | every residual block (`resnet`, `mlp`) |
| `blocks` | every constraint block (`cpool`) |
| `trunk.0,trunk.1` | the first two blocks only |
| `policy_head` | the policy head, training the trunk and value head |

Prefixes rather than globs: the useful unit is a submodule, and a prefix
names one exactly without inviting a pattern language nobody wants to
debug. `stem` matches `stem.0.weight` and does not match `stemx.0.weight`.

The run prints what it did, and it is worth reading:

```
froze 1,772,032 of 1,786,823 parameters (99.2%) matching ['stem', 'trunk'];
14,791 trainable, 13 normalisation module(s) held in eval
```

### Two silent failures this prevents

**A pattern that matches nothing.** `--freeze trunk` against `cpool`, whose
blocks are called `blocks`, would freeze nothing and train normally — which
looks exactly like a successful fine-tune. Unmatched patterns raise, and
the error lists the modules that do exist.

**Normalisation layers that keep tracking.** This is the subtle one.
`requires_grad = False` stops the gradient; it does **not** stop a
`BatchNorm` in training mode from updating its running mean and variance
from every batch it sees. A "frozen" trunk whose batch norms are still
tracking computes a different function after one epoch, and nothing in the
loss curve says so — the weights really are unchanged, so a check on the
weights passes.

Frozen normalisation modules are therefore also held in eval mode. Because
`model.train()` recurses over every submodule and would undo that at the
top of each epoch, the trainer calls `freezing.set_train_mode` instead.
`tests/test_freezing.py` asserts both halves: that the frozen running mean
does not move, and that a plain `model.train()` really would have moved it.

Normalisation layers *outside* the frozen set keep tracking, as they must.

**`--freeze` without `--init-from` is refused.** Freezing randomly
initialised weights trains a model around noise it can never correct.

### What frozen means for the optimizer

Only trainable parameters are passed to AdamW. Otherwise it carries moment
buffers for tensors that never receive a gradient — wasted memory, and an
optimizer state dict that implies it is training more than it is.

## Verifying a fine-tune did what you asked

The end-to-end guarantee is in `tests/test_finetune_cli.py`: after a
fine-tune with `--freeze stem,trunk`, every frozen tensor in the exported
checkpoint is **byte-identical** to the one it started from — including the
batch-norm running buffers — while the heads have moved. If the heads have
*not* moved, the fine-tune did nothing, and that is asserted too.

To check by hand:

```python
from safetensors.torch import load_file
import torch

before = load_file("runs/train/lineup-resnet/best/weights.safetensors")
after = load_file("runs/train/resnet-shallow-heads/best/weights.safetensors")
moved = [k for k in before if not torch.equal(before[k], after[k])]
print(sorted(moved))
```

## A note on when fine-tuning is the right tool

`shift-evaluation.md` shows every architecture is weakest at plies 4-6, and
`autoplay.md` produces exactly those positions with exact labels. It is
tempting to fine-tune the heads on that shallow corpus alone.

That is worth trying and worth watching: training only on shallow positions
can trade away the deep accuracy the model already has, and the shift
evaluation would show it immediately. Warm-starting on the *combined*
corpus is the more conservative move, and freezing is the tool for when
that turns out not to be enough.
