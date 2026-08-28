# Where the accuracy goes when the distribution changes

Every training run in this project prints a validation number measured on
the distribution it trained on: plies 6 to 13, where the corpus is dense.
That is not the regime an engine playing from the opening operates in. Per
`runs/coverage.md`, plies 0-5 contain **zero** training positions and ply 6
reaches 4.44% of its 901,916 canonical live positions.

So the number that matters is not how accurate a network is, but how much
of its accuracy survives leaving the distribution.

```bash
.venv/bin/python -m quantik_models.eval.shift \
  --checkpoint runs/train/lineup-resnet/best \
  --checkpoint runs/train/lineup-mlp/best \
  --checkpoint runs/train/lineup-cpool/best \
  --out runs/eval/shift-lineup.json
```

## The probe

`runs/oracle/probe-large.jsonl` — 7,800 exactly-solved positions spanning
plies 4 to 12, each carrying the full outcome-optimal action set from the
solver. It shares **zero canonical keys** with `exact-sampled.npz`, checked
on every run and asserted rather than assumed: a probe that overlapped the
corpus would measure recall, report it as generalisation, and nothing
downstream would look wrong.

Two conventions, both inherited from `scripts/oracle_probe.py`:

- **Accuracy is over positions the mover provably wins.** In a lost
  position every move loses, so nothing is being tested there.
- **Value truth is the game-theoretic outcome**, +1 or -1 from the side to
  move.

## The result

| model | params | shallow (4-6) | deep (7-12) | all | value MAE | value sign |
|---|---|---|---|---|---|---|
| `resnet-c128-b6` | 1,786,823 | **0.9126** | 0.9720 | 0.9455 | 0.1148 | 0.9559 |
| `mlp-h455-b4` | 1,788,343 | 0.8843 | 0.9578 | 0.9250 | 0.1659 | 0.9414 |
| `cpool-c191-b6` | 1,780,253 | 0.9092 | **0.9883** | **0.9530** | **0.0915** | **0.9574** |

Policy accuracy by ply:

| ply | `resnet` | `mlp` | `cpool` |
|---|---|---|---|
| 4 * | **0.8791** | 0.8682 | 0.8497 |
| 5 * | **0.9173** | 0.8809 | 0.9021 |
| 6 * | 0.9391 | 0.9026 | **0.9716** |
| 7 | 0.9545 | 0.9205 | **0.9820** |
| 8 | 0.9596 | 0.9521 | **0.9790** |
| 9 | 0.9674 | 0.9655 | **0.9866** |
| 10 | 0.9916 | 0.9790 | **0.9979** |
| 11 | 0.9932 | 0.9864 | **1.0000** |
| 12 | 0.9954 | 0.9954 | **0.9977** |

Value MAE by ply — the same shape, more sharply:

| ply | `resnet` | `mlp` | `cpool` |
|---|---|---|---|
| 4 * | **0.2307** | 0.2458 | 0.2483 |
| 5 * | 0.1505 | 0.2013 | **0.1398** |
| 6 * | 0.1225 | 0.2007 | **0.0526** |
| 8 | 0.0883 | 0.1279 | **0.0591** |
| 10 | 0.0577 | 0.1243 | **0.0490** |
| 12 | 0.0217 | 0.1151 | **0.0095** |

## Reading it

**ConstraintPoolNet wins overall, and the win is not where the IID number
said it was.** On the trained distribution it led the ResNet by 1.50 points
of top-1 (0.9851 vs 0.9701). Here it leads by 1.63 points on the deep
probes — held-out positions at *trained plies* — and **loses by 0.34 points
on the shallow set**, with the gap widening to 2.94 points at ply 4, where
it is the worst of the three.

The value head tells the same story with less noise. At ply 6 and deeper,
`cpool` is decisively better — less than half the ResNet's error at ply 6,
and less than half again at ply 12. At ply 5 it is marginally ahead. At ply
4 it is last, and its sign accuracy (0.8725) is beaten by the MLP's
(0.9175) — the weakest model everywhere else.

There is a clean crossover at ply 5.

ADR 0001 named this pattern in advance as the interesting failure:

> **It wins on the IID holdout but not on the shallow probes.** The most
> interesting failure. Plies 0-5 carry no training positions at all, so
> that pattern would say the group wiring helps memorise the trained
> distribution rather than generalise the rule.

That framing needs one correction now that the numbers are in. `cpool`
generalises *better* than the ResNet to unseen positions — the deep probes
are entirely held out and it wins them clearly. What it does not do is
**extrapolate to unseen plies**. Those are different failures, and only the
second one is happening.

## The shallow deficit is narrower than it looks

`cpool`'s ply-4 deficit is not spread across ply 4. Splitting each ply at
its own median **group occupancy** — the number of the twelve constraint
groups holding at least one piece — and testing paired with exact McNemar,
because only the positions where the two models disagree carry information:

| ply | bucket | n | `resnet` | `cpool` | difference | p |
|---|---|---|---|---|---|---|
| 4 | occ≤9 | 687 | 0.8967 | 0.8908 | −0.0058 | 0.74 |
| 4 | **occ>9** | 231 | 0.8268 | 0.7273 | **−0.0996** | **0.0006** |
| 5 | occ≤10 | 690 | 0.9232 | 0.9275 | +0.0043 | 0.79 |
| 5 | **occ>10** | 301 | 0.9037 | 0.8439 | **−0.0598** | **0.0064** |
| 6 | **occ>11** | 258 | 0.8953 | 0.9767 | **+0.0814** | **<0.0001** |

At plies 4 and 5, `cpool` is statistically **tied** with the ResNet on
low-occupancy positions and loses heavily on high-occupancy ones. It is not
worse in the opening generally — it is worse on a specific, identifiable
quarter of it, where the pieces are scattered thin, roughly one to a group.
Those are also the positions every model finds hardest: the ResNet itself
drops from 0.8967 to 0.8268 across the same split. `cpool` degrades faster
on them.

By ply 6 the effect reverses, and `cpool`'s advantage is *largest* on
high-occupancy positions. `architecture-constraint-pool.md` has the full
table and what it does and does not settle — including that the original
one-line explanation offered for the shallow deficit was too simple, since
no single "sparse groups carry no signal" story produces both signs.


## What this does not settle

- **Single seed, single run per architecture.** The resnet/mlp gap is wide
  enough to trust. The shallow resnet/cpool gap is 0.34 points on 3,600
  positions and is not.
- **Plies 0-3 are not evaluated at all.** The probe starts at 4. Ply 0 is
  the position an engine actually opens from, and the three networks
  disagree about it flatly: on the empty board the ResNet returns +0.77,
  the MLP +0.59, and `cpool` -0.997. At most one of them is right.
- **This measures move choice, not play.** An engine that picks the second
  best move in one position may still win the game. That is what autoplay
  is for.

## What it does settle

**The IID number was flattering everybody, and it was flattering them
unequally.** Ranking the three architectures on `val_top1` alone would have
overstated `cpool`'s advantage in the regime that matters and hidden that
the ResNet is still the better opening evaluator. Any future architecture
in this project gets both numbers.
