# The ResNet, layer by layer

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


`resnet-c128-b6` — 1,786,823 parameters. The project's incumbent, and the
model the Part VI and VII articles describe.

`policy-value-training-paper.md` covers *why* this network is trained the
way it is: distillation from an exact solver rather than self-play, the
loss, the sharding, the checkpoint contract. This document covers what is
actually in it, one layer at a time, because none of that was written down
and "a 3×3 stem, `B` residual blocks, two heads" is not enough to
reimplement or to reason about.

## Where the parameters are

| group | parameters | share |
|---|---|---|
| `stem` | 10,624 | 0.59% |
| `trunk` (6 blocks) | 1,772,544 | 99.20% |
| `policy_head` | 2,372 | 0.13% |
| `value_head` | 1,283 | 0.07% |
| **total** | **1,786,823** | |

**Ninety-nine percent of this network is trunk.** Both heads together are
3,655 parameters — one fifth of one percent. That is worth internalising
before reading anything else here: essentially every design question about
capacity is a question about `channels` and `blocks`, and essentially
nothing that happens in the heads matters for size.

It also explains the preset arithmetic. Each residual block is
`2 × (9C² + 2C)` — two 3×3 convolutions plus two batch norms — so
parameters grow as roughly `18 · B · C²`. Doubling width quadruples the
model; doubling depth merely doubles it.

## Input

`(B, 9, 4, 4)` float32, **mover-relative**: planes 0–3 are the side to
move's shapes A–D, planes 4–7 the opponent's, plane 8 a constant carrying
the side to move. See `architectures.md` — the colour-ordered layout under
the same contract name is used by nothing in training, and specifying the
wrong one has already cost this project one incorrect document.

## Stem

```
Conv2d(9 -> C, kernel 3, padding 1, bias=False)    9·C·9 = 81C parameters
BatchNorm2d(C)                                     2C parameters + 2C buffers
ReLU
```

At C=128: 10,368 + 256 = 10,624.

`padding=1` keeps the spatial extent at 4×4 throughout the network. There
is no pooling and no stride anywhere: the board is 4×4, and downsampling it
would throw away the only spatial structure there is.

`bias=False` because the batch norm immediately downstream has its own
shift. A bias here would be unidentifiable — the two parameters could drift
against each other with no effect on the output.

**After this single layer plus the first residual block, the receptive
field already covers the whole board.** A 3×3 kernel on a 4×4 grid reaches
every cell within two applications. Everything after that adds capacity and
nonlinearity, not reach. This is the fact that makes "sees globally" a
meaningless distinction on this game, and it is why the alternatives in
`decisions/0001-architecture-lineup.md` are justified on other grounds.

## Trunk: `B` × residual block

```
x ──┬─ Conv2d(C→C, 3, pad 1, bias=False) ─ BatchNorm2d ─ ReLU
    │                                                        │
    │  Conv2d(C→C, 3, pad 1, bias=False) ─ BatchNorm2d ──────┘
    │                          │
    └──────────── + ───────────┘
                  │
                ReLU
```

Post-activation, in the original ResNet arrangement: the skip is added
*before* the final ReLU rather than after. (The MLP baseline uses
pre-activation blocks; the difference is not load-bearing at this depth and
neither ordering was chosen for a reason worth defending.)

Per block: `2 · (9C² + 2C)` = 295,424 at C=128. Six blocks: 1,772,544.

The residual connection is doing ordinary work here — at six blocks the
network would still train without it, but the skip makes depth free to add
and removes the question of whether a depth difference between
architectures is measuring optimisation rather than capacity.

## Policy head

```
Conv2d(C -> 2, kernel 1, bias=False)     2C = 256
BatchNorm2d(2)                           4
ReLU
Flatten                                  2 × 16 = 32 features
Linear(32 -> 64)                         2,112
```

Total 2,372. The 1×1 convolution to **2 channels** is the AlphaZero
arrangement, and it is a severe bottleneck: the entire 128-channel trunk
representation is compressed to 32 numbers before the linear layer produces
all 64 action logits.

This is worth flagging as an inherited choice rather than a justified one.
In AlphaZero the equivalent head sits on a 19×19 board, where 2 channels
still means 722 features. Here it means 32, for 64 outputs — fewer inputs
than outputs. Whether widening it would help is a cheap ablation that has
never been run; at 0.13% of parameters, the cost of trying is nil.

Output ordering needs no transpose: the linear layer emits the 64 actions
directly in contract order, `action_index = shape · 16 + position`. An
architecture producing per-cell logits does have to transpose — see
`architecture-constraint-pool.md`, where getting it wrong yields a tensor
of exactly the right shape with every logit on the wrong action.

## Value head

```
Conv2d(C -> 1, kernel 1, bias=False)     C = 128
BatchNorm2d(1)                           2
ReLU
Flatten                                  16 features
Linear(16 -> 64)                         1,088
ReLU
Linear(64 -> 1)                          65
Tanh
squeeze(-1)                              -> (B,)
```

Total 1,283 — the smallest part of the network by a wide margin, and a
single channel is an even tighter bottleneck than the policy head's two.

The value is **mover-relative**: `+1` means good for the side to move, not
good for player 0. That is what the mover-relative input encoding buys — a
single head with one sign convention, instead of a head that must read
plane 8 and flip its own answer. `Tanh` bounds the output to `[-1, 1]` as
`policy-logits-64+value-tanh` requires and as a runtime may rely on.

`shift-evaluation.md` measures this head against exact truth: MAE 0.1148
across plies 4–12, ranging from 0.2307 at ply 4 to 0.0217 at ply 12. It
knows who is winning once the position is developed, and much less well in
the opening — where, on the empty board, it returns +0.77 with no training
position anywhere near it.

## Legality masking is not here

There is no mask in this module, deliberately. `masked_log_softmax` applies
it outside, using the same code path in training and at inference: illegal
logits are filled with the dtype's most negative finite value, so softmax
sends them to exactly zero.

The rules are already exact in `quantik-core`; a network that learned them
from data would approximate something computable perfectly. This is the
standard arrangement — AlphaZero, Leela and NNUE all do it — and it means
no engine in this project can return an illegal move.

One sharp edge, documented in the source and repeated here: an all-illegal
mask row yields a *uniform* distribution over all 64 actions rather than
`NaN`, because `finfo.min` keeps the arithmetic finite. Callers must not
pass terminal positions expecting an error.

## Presets

| preset | C | B | parameters | float32 safetensors |
|---|---|---|---|---|
| `smoke` | 16 | 2 | 13,991 | 68 KB |
| `small` | 64 | 4 | 304,711 | 1.2 MB |
| `medium` | 128 | 6 | 1,786,823 | 6.8 MB |
| `target` | 256 | 13 | 15,374,023 | ~61 MB |

`target` exists to sit inside the 50–100 MB contract envelope, which sizes
the artifact a runtime must accept. It is not a claim that Quantik needs
15M parameters, and it has never been trained — §4.2 of the training paper
discusses that tension.

## Batch norm and a batch of one

`BatchNorm2d` in training mode raises on a batch of one: there is no
variance to normalise by. Serving evaluates single positions, so the model
must be in eval mode, where the folded running statistics are used instead.
`export_onnx` handles it by exporting in eval mode, and
`test_batch_size_one_works` guards it for every registered architecture.

This is the failure that only ever appears in production and never in a
batched training loop, which is why it is tested rather than remembered.

It also has a second edge that matters for fine-tuning: `requires_grad =
False` does **not** stop a batch norm from updating its running statistics.
See `retrain-and-finetune.md`.

## How it actually performs

Three measurements, and they disagree until you condition on depth. All
were taken at `--lr 2e-3` — a rate chosen for *this* architecture, which is
worth knowing before comparing them to anything else:

| | value |
|---|---|
| IID validation top-1 (plies 6–13) | 0.9701 |
| shift, plies 4–6 | **0.9126** — best of the three |
| shift, plies 7–12 | 0.9720 — beaten by `cpool`'s 0.9883 |
| arena from ply-3 starts | **53.7%** — best of the three |
| arena from ply-6 starts | 48.8% — beaten by `cpool` |

**Corrected 2026-08-30.** This section used to conclude that the ResNet was
"the best opening player in the lineup and not the best midgame player".
That was wrong, and it was wrong for a reason worth keeping: `cpool` was
trained at 2e-3 — *this* architecture's preferred rate — and prefers 6e-4.
Retrained, it beats the ResNet at every depth, and the ResNet is third at
every arena start depth.

| | resnet | best of the lineup |
|---|---|---|
| IID top-1 | 0.9701 | `cpool` 0.9893 |
| shift, plies 4-6 | 0.9126 | `cpool` 0.9295 |
| shift, plies 7-12 | 0.9720 | `cpool` 0.9919 |
| arena @ply3 | 47.8% | `cpool` 57.2% |
| arena @ply6 | 48.8% | `attn` 54.3% |

The ResNet's only remaining lead is ply 4 on the shift probe, 0.8791
against `cpool`'s 0.8780 — a gap of 0.0011.

`shift-evaluation.md` and `autoplay.md` have the detail;
`decisions/0001-architecture-lineup.md` has what it means for the
comparison, and `learning-rate-sweep.md` has why the earlier numbers were
what they were.
