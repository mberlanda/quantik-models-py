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

> **Restated 2026-08-30 at swept learning rates.** The tables below are the
> corrected ones. `cpool` was trained at 2e-3, a rate chosen for the ResNet,
> and prefers 6e-4; retraining it changed the *conclusions*, not just the
> decimals. What the earlier version claimed, and why it was wrong, is at
> the end of this document rather than deleted.

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

| model | lr | shallow (4-6) | deep (7-12) | all | value MAE | value sign |
|---|---|---|---|---|---|---|
| `cpool-c191-b6` | 6e-4 | **0.9295** | **0.9919** | **0.9641** | **0.0777** | **0.9646** |
| `attn-d192-b6` | 6e-4 | 0.9102 | 0.9914 | 0.9552 | 0.0881 | 0.9587 |
| `resnet-c128-b6` | 2e-3 | 0.9126 | 0.9720 | 0.9455 | 0.1148 | 0.9559 |
| `mlp-h455-b4` | 2e-3 | 0.8843 | 0.9578 | 0.9250 | 0.1659 | 0.9414 |

Policy accuracy by ply:

| ply | `resnet` | `mlp` | `cpool` | `attn` |
|---|---|---|---|---|
| 4 * | 0.8791 | 0.8682 | 0.8780 | 0.8693 |
| 5 * | 0.9173 | 0.8809 | **0.9324** | 0.8971 |
| 6 * | 0.9391 | 0.9026 | **0.9746** | 0.9615 |
| 7 | 0.9545 | 0.9205 | **0.9915** | 0.9848 |
| 8 | 0.9596 | 0.9521 | 0.9835 | **0.9910** |
| 9 | 0.9674 | 0.9655 | 0.9904 | **0.9923** |
| 10 | 0.9916 | 0.9790 | 0.9937 | **0.9958** |
| 11 | 0.9932 | 0.9864 | **0.9977** | 0.9932 |
| 12 | 0.9954 | 0.9954 | **1.0000** | **1.0000** |

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

**`cpool` is best at every depth**, once trained at its own learning rate.
Shallow, deep and overall; policy and value. There is no longer a regime
where another architecture leads it.

**`attn` is second and close** — 0.9914 deep against `cpool`'s 0.9919 — and
it was still improving when the epoch budget ran out. On the shallow set it
is weaker (0.9102), which is the one place the constraint prior still looks
like it is doing work.

**The ResNet is third**, and its remaining strength is exactly one cell:
ply 4, where it leads at 0.8791 against `cpool`'s 0.8780 — a gap of 0.0011,
which is nothing.

## What the earlier version of this document claimed, and why it was wrong

Before `cpool` was retrained, this file reported a much more interesting
story: that the ResNet was the better *shallow* evaluator (0.9126 against
0.9092), that `cpool` "wins the IID holdout and the deep probes and loses
the shallow ones", and that the two architectures had advantages living at
different depths.

**All of that was an artifact of the learning rate.** `cpool` was trained
at 2e-3, a value chosen for the ResNet; at 6e-4 its shallow accuracy goes
from 0.9092 to 0.9295 and it passes the ResNet's 0.9126. The ply-4 deficit
that looked like a real architectural weakness — 0.8497 against 0.8791 —
becomes 0.8780 against 0.8791.

The occupancy analysis built on top of that story went the same way. It
reported that `cpool` lost specifically on high-occupancy shallow
positions, by 10.0 points at ply 4 with p = 0.0006. Re-run against the
correctly trained model:

| ply | bucket | n | `resnet` | `cpool` | diff | p |
|---|---|---|---|---|---|---|
| 4 | occ<=9 | 687 | 0.8967 | 0.9039 | +0.0073 | 0.65 |
| 4 | occ>9 | 231 | 0.8268 | 0.8009 | **-0.0260** | **0.42** |
| 5 | occ<=10 | 690 | 0.9232 | 0.9522 | +0.0290 | 0.0078 |
| 5 | occ>10 | 301 | 0.9037 | 0.8870 | -0.0166 | 0.49 |
| 6 | occ>11 | 258 | 0.8953 | 0.9845 | +0.0891 | <0.0001 |
| 7 | occ>11 | 512 | 0.9414 | 0.9863 | +0.0449 | <0.0001 |

The ply-4 high-occupancy deficit is now **-0.0260 at p = 0.42** — not
significant, and a quarter of its previous size. What survives is that
`cpool`'s *advantage* at plies 6 and 7 is largest on high-occupancy
positions, which is the opposite half of the original finding.

This is kept rather than deleted because the failure mode is the point: a
hyperparameter inherited from another architecture produced a plausible,
detailed, statistically significant story about architectural behaviour,
and it was measuring the learning rate.

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
