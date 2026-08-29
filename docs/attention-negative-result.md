# The attention encoder does not train at the lineup's learning rate

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


**Status: resolved — it is the learning rate.** An earlier version of this
document called it unresolved and listed "learning rate alone" among the
things ruled out. That was wrong, and it was wrong for an instructive
reason: the claim rested on a single epoch. Given three, the picture
reverses completely.

| epochs at `--lr 3e-4` | 1 | 2 | 3 |
|---|---|---|---|
| val top-1 | 0.5380 | 0.6454 | **0.7271** |

At 3e-4 the network learns steadily and was still climbing when the probe
ended. At the lineup's shared 2e-3 it is flat at 0.51 for sixteen epochs.
One epoch could not distinguish "marginally better" from "learning"; three
could.

`attn-d192-b6` was trained at `medium` on the same corpus, split, budget
and seed as the other three architectures, per the methodology in
`decisions/0001-architecture-lineup.md`.

| model | val top-1 | val policy loss | val MAE |
|---|---|---|---|
| `cpool-c191-b6` | 0.9851 | 1.2217 | 0.0435 |
| `resnet-c128-b6` | 0.9701 | 1.3090 | 0.0710 |
| `mlp-h455-b4` | 0.9516 | 1.3945 | 0.1116 |
| **`attn-d192-b6`** | **0.5130** | **2.1299** | **0.6464** |

That is not a weaker result. It is a model that did not learn the task.

## The curve is flat from the first epoch

```
ep   lr        train_policy  val_policy  train_top1  val_top1
 0   1.98e-03  2.2674        2.1384      0.5273      0.5031
 4   1.56e-03  2.2544        2.1422      0.5351      0.5034
 8   8.11e-04  2.2529        2.1422      0.5372      0.5008
12   1.78e-04  2.2437        2.1314      0.5424      0.5079
15   1.00e-05  2.2400        2.1299      0.5428      0.5114
```

Train loss moves 2.2674 → 2.2400 across sixteen epochs and 45,808 steps.
Train and validation track each other closely throughout, so this is not
overfitting — it is a model stuck immediately and never moving. For
comparison, the ResNet reaches 0.9055 validation top-1 after its *first*
epoch.

## What the trained model actually outputs

Evaluated on 200 reachable ply-7 positions:

| | `resnet` | `cpool` | `attn` |
|---|---|---|---|
| mean max prior | 0.3695 | 0.3758 | **0.0856** |
| value std | 0.6934 | 0.7002 | **0.2063** |
| value range | [−0.999, +1.000] | [−1.000, +1.000] | **[+0.227, +0.984]** |

The policy is nearly uniform over legal moves — a uniform distribution
over roughly two dozen legal actions would give about 0.042, so 0.0856 is
barely peaked at all. The value head **never predicts a loss**: its whole
output range sits above zero, which is close to just reporting the base
rate of won positions.

It does respond to its input — 53 distinct argmaxes across the 200
positions, against 58 and 63 for the others — so it is not producing a
constant. It is producing something very close to one.

## What has been ruled out

**Vanishing or exploding gradients.** Per-module gradient RMS on a real
batch, against `cpool` as the closest comparable architecture (same
token-per-cell layout, same heads, same LayerNorm-only normalisation):

| module | `cpool` | `attn` |
|---|---|---|
| stem | 2.97e-02 | 4.12e-02 |
| trunk blocks | 1.06e-02 | 1.25e-02 |
| norm_out | 1.99e-02 | 2.66e-02 |
| policy_head | 5.77e-02 | 7.46e-02 |
| value_head | 5.91e-02 | 8.80e-02 |

`attn`'s gradients are marginally *larger* than `cpool`'s at every layer.
Nothing is starved.

**Dead parameters.** The preflight's gradient check passes: every
parameter receives a non-zero gradient.

**A wrong action layout.** The policy head reuses
`flatten_cell_shape_logits`, the same directly-tested function `cpool` uses,
so the transpose trap is not in play.

**A broken export.** ONNX agrees with torch to 8.34e-07 across batch sizes
1, 5 and 64. The graph computes what the model computes; the model is the
problem.

Not ruled out, and in fact the cause: **the learning rate**. See the
correction at the top. 2e-3 is roughly seven times higher than the range
usually recommended for transformers, and at 3e-4 the same network
trains.

## An improvement that was tried and rejected

The obvious response is to strengthen the preflight so it catches this. Its
fixed-batch check — "the loss falls over twelve steps" — passed `attn`
before the run started.

Extending that check to 120 steps and requiring a substantial reduction
does **not** work, and the measurement is worth recording so nobody tries
it again:

| architecture | loss reduction over 120 steps on a fixed batch |
|---|---|
| `mlp` | 69.9% |
| `resnet` | 65.6% |
| `attn` | 36.8% |
| `cpool` | **26.4%** |

`cpool` — the best model in the lineup by every real measure — overfits a
fixed batch *less* than `attn` does. Any threshold that flags `attn` would
have blocked `cpool` first. The fixed-batch overfit test does not predict
trainability across these architectures, so the preflight keeps its weak
version, which at least catches a frozen trunk or a detached graph.

## The methodology was the problem, not the architecture

ADR 0001 requires every architecture to be trained with the same
optimizer, schedule and budget, so that none is quietly tuned harder than
the others. That is the right instinct and it was implemented the wrong
way, which this failure makes visible.

**A shared learning-rate *value* is not equal treatment.** 2e-3 is the
trainer's default, and it is the default because it was chosen for the
ResNet — the only architecture that existed when the default was set. Every
subsequent architecture has been evaluated at a learning rate tuned for a
convolutional network. That does not privilege "no architecture"; it
privileges the incumbent.

The MLP and `ConstraintPoolNet` happened to tolerate it. The attention
encoder does not, and would have been recorded as a failed architecture on
the strength of a hyperparameter inherited from a different one.

**The fix is a shared protocol rather than a shared value.** Each
architecture gets the same small learning-rate sweep and the same budget,
and the best validation result is what enters the comparison. That is
equal treatment — every architecture receives exactly the same tuning
effort — and it removes the incumbent's advantage without letting anyone
tune one model harder than the rest.

## What follows

1. **Train `attn` to completion at 3e-4** and report it.
2. **Sweep the other three at the same grid.** Not optional, and not a
   courtesy to attention: if `cpool` or the ResNet also improves at a
   different learning rate, then every comparison in
   `shift-evaluation.md` and `autoplay.md` is a comparison at *the
   ResNet's* preferred setting, and their margins need restating.
3. Only then is there a defensible claim about attention on this task.

Until step 1 lands, `attn` stays registered and out of every comparison
table.

## Things that were ruled out and stay ruled out

Vanishing gradients, dead parameters, the action layout, and the ONNX
export are all genuinely clear — those diagnostics were sound. The error
was stopping the learning-rate probe after one epoch and writing the
conclusion from it.
