# The learning-rate sweep, and why it had to happen

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


## The problem it fixes

`SupervisedConfig.lr` defaulted to `2e-3`. That value was chosen for the
ResNet, because the ResNet was the only architecture in the project when it
was set. Every architecture added afterwards inherited it **by omission** —
nothing in the code or the CLI made the inheritance visible.

ADR 0001 required every architecture to be trained with the same optimizer,
schedule and budget, so that none was tuned harder than the others. The
instinct is right. The implementation was not: **a shared learning-rate
*value* is not equal treatment.** It does not privilege nobody; it
privileges whichever architecture the value was chosen for.

The failure that exposed it was total. `attn-d192-b6` at 2e-3 sits flat at
0.51 validation top-1 for sixteen epochs and 45,808 steps — not a weaker
model, a model that does not learn. It was one commit away from being
written up as a failed architecture on the strength of a hyperparameter
belonging to a different one.

The rule is now a shared **protocol** rather than a shared value: the same
grid, the same budget, for every architecture, with the best validation
result entering the comparison. Equal tuning effort for everyone, no
incumbent advantage.

## The sweep

Four architectures, three rates, three epochs each, all at `--preset medium`
on `exact-sampled.npz` with seed 20260828. Twelve runs.

Three epochs because that is where the signal is. The attention failure was
**unambiguous by epoch 3 and invisible at epoch 1** — a single epoch showed
0.5380 against 0.5031 and read as "marginally better", which is how it was
briefly written up as unrelated to the learning rate.

| architecture | 2e-3 | 6e-4 | 2e-4 | best |
|---|---|---|---|---|
| `mlp-h455-b4` | **0.9261** | 0.9155 | 0.8312 | 2e-3 |
| `resnet-c128-b6` | **0.9300** | 0.9282 | 0.9097 | 2e-3 |
| `cpool-c191-b6` | 0.9622 | **0.9781** | 0.9666 | **6e-4** |
| `attn-d192-b6` | 0.5096 | **0.7938** | 0.5404 | **6e-4** |

Three epochs of validation top-1. Higher is better.

## What it found

**Two of four architectures were trained at the wrong rate.**

`mlp` and `resnet` prefer 2e-3, so their published numbers stand. The MLP's
curve is monotone and clean. The ResNet's top two differ by **0.0018**,
which is not a result — that is "no change", not "2e-3 confirmed", and a
sixteen-epoch run could order them either way.

`cpool` prefers **6e-4 by 1.6 points**, with a clean inverted-U rather than
a monotone trend. That is a real preference, and it means every number
published for `cpool` — the 0.9851 IID top-1, the 1.63-point lead on the
deep probes, the halved value error, the occupancy analysis — was measured
on a model trained at the wrong rate. **They are floors, not results.**

`attn` needs 6e-4 and the window is narrow in both directions:

| rate | 3-epoch val top-1 | |
|---|---|---|
| 2e-3 | 0.5096 | flat; does not learn |
| **6e-4** | **0.7938** | best measured |
| 3e-4 | 0.7271 | learns steadily |
| 2e-4 | 0.5404 | stuck, and *declining* over the three epochs |
| 1e-4 | 0.5409 | stuck, rising imperceptibly |

The transition is not gradual. Between 2e-4 and 3e-4 the network goes from
sitting at 0.54 to reaching 0.73 — a factor of 1.5 in the rate separating
"does not learn" from "learns well". At the top end 2e-3 fails too, so the
usable window is roughly a single octave, `3e-4` to `1e-3`.

Note the failures at either end are not the same failure. At 2e-3 and 2e-4
the curve is flat or declining; at 1e-4 it rises, just far too slowly. Only
the last of those is "too small a step" in the ordinary sense.

## What this sweep does not establish

**The peak is at the middle of the grid, so the true optimum is unlocated.**
`cpool` and `attn` both chose 6e-4 out of {2e-3, 6e-4, 2e-4}. The real
optimum lies somewhere between 2e-4 and 2e-3 and could be meaningfully
better than 6e-4. A finer grid would find it; this one only establishes
that 2e-3 was wrong for both.

**Three epochs rank, they do not settle.** A lower rate often looks worse
at three epochs and better at sixteen, having had less time to converge.
The gaps here are large enough to act on for `cpool` (1.6 points) and
`attn` (2.8 points), and too small to act on for the ResNet (0.0018). Each
architecture's chosen rate is confirmed by a full-length run before its
numbers are restated.

**One seed.** Everything above is seed 20260828.

## The registry default is not a guess any more

The rate is now a property of the architecture —
`registry.ArchitectureEntry.default_lr` — so a new architecture must state
its own rather than inherit the incumbent's by silence.

Worth recording that the first version of that field was itself wrong.
`attn` was set to 3e-4 on the strength of a single probe, which is the same
"chosen once, then trusted" reasoning that caused the original problem, in
miniature. The sweep says 6e-4. The values are now swept rather than
guessed, and the ResNet's and MLP's are swept-and-unchanged rather than
merely unexamined.

## Reproducing it

```bash
for arch in mlp resnet cpool attn; do
  for lr in 2e-3 6e-4 2e-4; do
    python -m quantik_models.train.supervised \
      --arch "$arch" --preset medium --lr "$lr" \
      --corpus runs/oracle/corpus/exact-sampled.npz \
      --name "sweep-$arch-$lr" --epochs 3 --seed 20260828 --out runs/lrsweep
  done
done
```

Then `scripts/evaluate_lineup.sh` regenerates every downstream number in
one command, which is what makes restating the margins tractable.
