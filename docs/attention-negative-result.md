# The attention encoder does not train on this task

**Status: unresolved.** This document records a failure and the
diagnostics run against it. It does not claim to explain it.

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

**Learning rate alone.** A probe at `--lr 3e-4` — roughly seven times
lower, and in the range usually recommended for transformers — reaches
0.5380 validation top-1 after its first epoch, against 0.5031 at 2e-3.
Better, and nowhere near the 0.90 the ResNet reaches in the same epoch. A
further probe at 1e-4 is running.

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

## What this does and does not say about attention

**It does not say attention is a bad architecture for Quantik.** The
lineup methodology gives every architecture the same optimizer, schedule
and budget, deliberately, so that no architecture is quietly tuned harder
than the others. That rule has a failure mode, and this is it: when one
architecture needs different hyperparameters to train *at all*, holding
them fixed produces a result about the hyperparameters rather than about
the architecture.

The honest claim is therefore narrow: **at the lineup's shared settings,
and at 3e-4, this attention encoder does not learn the task, and the usual
suspects have been ruled out.**

## What to try next, in order

1. **Finish the learning-rate sweep** — 1e-4 is running; 3e-5 is the next
   rung down.
2. **Warmup.** Pre-norm blocks were chosen specifically so that no warmup
   schedule would be needed, which is the standard justification. If a
   linear warmup fixes this, that justification was wrong for this setting
   and the module docstring needs correcting.
3. **Exclude LayerNorm and the positional embedding from weight decay.**
   Standard practice for transformers, not currently done — the trainer
   passes every parameter to AdamW with `weight_decay=1e-4`. Small, but it
   is free to test.
4. **The `small` preset.** If `attn-d96-b4` trains and `attn-d192-b6` does
   not, the problem is scale, not architecture.
5. **Only then** conclude anything about attention on this task.

Until one of those lands, `attn` stays registered and out of every
comparison table. A number produced by a model that did not train is worse
than no number.
