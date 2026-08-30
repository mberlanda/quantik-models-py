# The corpora: what is in them, how to make more, and what you can tune

Every trained checkpoint in this project reads one `.npz` from
`runs/oracle/corpus/`. `runs/` is gitignored, so this document is what survives
the directory. It covers what a corpus contains, how the ones on record were
built, what you can change, and which changes are known to buy nothing.

Related: [`labeling-strategy.md`](labeling-strategy.md) (what a label is here),
[`pipeline.md`](pipeline.md) (solver to trainer), [`corpus-v3.md`](corpus-v3.md)
(the v3 result and its correction), [`reproducibility.md`](reproducibility.md).

## What a corpus file is

One row per **canonical** position — deduplicated up to the 192 board
symmetries, so two positions that are the same game state under relabelling are
one row.

Two schemas exist, and they differ only in how the policy label is stored:

| | `exact-sampled.npz` (v1) | `exact-sampled-v2/v3.npz` |
| --- | --- | --- |
| `boards` | `uint16 (N, 8)` | `uint16 (N, 8)` |
| `value_target` | `float32 (N,)` | `float32 (N,)` |
| `plies` | `int16 (N,)` | `int16 (N,)` |
| policy label | `policy_target float32 (N, 64)` + `policy_weight float32 (N,)` | `optimal_mask uint64 (N,)` |
| bytes/row | **282** | **30** |

`optimal_mask` is a 64-bit set of optimal `action_index` values. It carries the
same information as the dense row — the dense row is uniform over the optimal
moves and sums to 1 — in 8 bytes instead of 256. **The conversion mask → dense
is exact; dense → mask is exact too, for these files, because every stored
distribution is uniform over its support.**

### The values are only ±1, and that is not a simplification

`value_target` takes exactly two values. Quantik has no draws, and both terminal
conditions are losses for the side to move, so every solved position is a win or
a loss for the mover. `win_probability = (value + 1) / 2` is therefore exact
rather than a calibration.

Across v1: **76.6% of rows are wins** for the side to move, 23.4% losses.

### Only about 8% of rows carry a policy label

This surprises people, so it is worth stating plainly.

| corpus | rows | policy-labelled | share |
| --- | --- | --- | --- |
| v1 | 3,087,356 | 250,000 | 8.1% |
| v2 | 3,196,958 | 255,058 | 8.0% |
| v3 | 3,520,526 | 271,676 | 7.7% |

The other ~92% are **value-only** rows: positions that were labelled as a
by-product of solving something else, so their outcome is known but which move
is optimal was not recorded. The trainer masks the policy loss on them
(`policy_weight = 0` in v1, a zero mask in v2/v3) — they train the value head
only.

The density is **the same across all three schemas**, which matters: the v1 → v2
change is a storage change, not a change in how much policy supervision exists.
Anyone re-examining the v1-vs-v2 finding in [`corpus-v3.md`](corpus-v3.md) should
know that this is *not* a hidden second variable.

Among labelled rows, the mean number of optimal moves is **4.22**, and the
maximum is 31.

## Distribution: where the positions actually are

Rows by ply. The three corpora are **strictly nested** — `v1 ⊂ v2 ⊂ v3`, zero
canonical keys lost at either step — and **plies 8-13 are byte-identical across
all three**. Every difference between them lives at plies 3-7:

| ply | v1 | v2 | v3 |
| --- | --- | --- | --- |
| 0-2 | 0 | 0 | 0 |
| 3 | 0 | 664 | 726 |
| 4 | 0 | 9,664 | 9,758 |
| 5 | 0 | 22,655 | 29,905 |
| 6 | 40,000 | 86,631 | 170,766 |
| 7 | 846,816 | 876,804 | 1,108,831 |
| 8 | 1,001,185 | 1,001,185 | 1,001,185 |
| 9-13 | identical across all three | | |

Two things to read off this:

1. **No corpus contains a single position from plies 0, 1 or 2.** Not one. This
   is why every checkpoint is uniform to three decimal places on the empty board
   — it has never been shown one.
2. **Coverage is a rounding error where it matters.** Plies 0-6 hold 1,019,275
   canonical live positions in total. v1 covers 40,000 of them — under 4% — and
   v3 covers 211,155, about 21%. Meanwhile ply 8 alone contributes a million
   rows. The corpus is dense exactly where the game is nearly forced.

**Full opening coverage is cheaper than what already exists.** All of plies 0-6
is 1,019,275 positions, against the 3,087,356 rows v1 already trains on.

## How the corpora on record were built

The enumerations are already computed and on disk: `runs/canonical/level01.npy`
through `level08.npy` — every canonical live position at that ply, up to
symmetry. Only labelling is missing below ply 6.

Two sources feed a corpus, and they are not interchangeable:

**1. Exhaustive solve of an enumerated level.** Take `levelNN.npy`, run it
through the exact oracle in `quantik-core-rust`, get a label per position. This
is what produced ply 8 and above. Deterministic; no seed; run it twice and get
the same file.

**2. Autoplay, then solve.** `arena.autoplay` plays engines against each other,
collects the distinct positions visited up to a ply cut-off, and writes
`to-solve.qfen`. The oracle labels those, and `data.merge_corpus` folds them in.
This is what produced v2 and v3 — the shallow rows are positions strong engines
actually reached, not a uniform sample of the ply.

```bash
# 1. play, and collect the positions
python -m quantik_models.arena.autoplay \
  --agents agents.json --games 300 --start-plies 3 \
  --out runs/autoplay/mine --seed <seed>

# 2. label them exactly
../quantik-core-rust/target/release/examples/exact_oracle \
  < runs/autoplay/mine/to-solve.qfen > runs/autoplay/mine/solved.jsonl

# 3. fold them into an existing corpus
python -m quantik_models.data.merge_corpus \
  --corpus runs/oracle/corpus/exact-sampled.npz \
  --solved runs/autoplay/mine/solved.jsonl \
  --out runs/oracle/corpus/mine.npz
```

## Your questions, answered

### Can I merge two corpora generated separately?

**Yes, and `merge_corpus` is the only way you should.** It enforces two
invariants that a naive concatenation breaks:

- **The probe stays held out.** Exclusion is applied to the *merged result*, not
  just the incoming rows — because solving a position also labels its children,
  so a probe position can arrive as somebody's child without ever being sampled.
  That is exactly how sixteen probe positions reached the first corpus.
- **One row per canonical position, preferring the policy-labelled copy.** A
  position can appear as a solved parent in one file and a value-only child in
  another. Keeping both silently reweights it in the loss.

Two cautions specific to merging:

- **Across schemas**, a v1 file and a v2/v3 file do not concatenate — one has
  `policy_target`/`policy_weight`, the other `optimal_mask`. Convert first;
  the mapping is exact in both directions for these files.
- **Merging changes the ply distribution**, and the ply distribution is a
  hyperparameter in disguise. Two corpora that merge cleanly can still produce a
  model that is worse in a band you care about. Re-measure with
  `python -m quantik_models.eval.shift`, which uses the shared probe and is
  therefore common ground; **per-corpus validation splits are not** — comparing
  them is how [`corpus-v3.md`](corpus-v3.md) reached a wrong conclusion.

### Can I combine the opening book to ply 6, then a stochastic corpus beyond?

**Yes, and this is the recommended shape.** They are different instruments:

- **Plies 0-6 are small enough to solve completely** — 1,019,275 positions,
  fewer than the corpus already has. Exhaustive enumeration is available on disk
  (`runs/canonical/`), so this region needs no sampling and no cleverness. For
  opening *play*, prefer the exact book over the network outright: the region is
  solved, and the network is least informed there.
- **Beyond ply 6 the space is too large to enumerate**, so sampling is the only
  option, and autoplay sampling is better than uniform sampling because it
  concentrates on positions strong play actually reaches.

The measured warning attached: **densifying an already-covered band buys
nothing.** The v2 → v3 step added 323,568 more positions of the same shallow
distribution, and it is null on every probe band, every value metric, and all
five arena conditions at matched budget. The v1 → v2 step, which reached *new*
plies, bought ~5 points of ply-4 accuracy and cut value MAE threefold. **Extend
to new plies; do not densify old ones.**

### What is the impact of using different seeds?

Honestly: **unknown, and it is the largest gap in this project's evidence.**

- **Training seed.** Every checkpoint on record is seed `20260828`. The
  run-to-run spread has never been measured, so **no margin in any lineup table
  has an error bar under it**. A one-point difference between two architectures
  and a one-point difference between two seeds of the same architecture are
  currently indistinguishable. Measuring this is cheap relative to what it would
  settle.
- **Arena seed.** Genuinely does move results — the arena is seeded separately
  and deliberately never reuses a training seed, so a seed-linked bias shows up
  rather than hiding. `20260829` and `20260909` are **spent**; use a third.
- **Corpus-generation seed.** Changes *which* positions autoplay reaches. Since
  positions are deduplicated on canonical key and labels are exact, two seeds
  produce overlapping but different row sets — merging them is legitimate and is
  a reasonable way to broaden coverage without changing the ply band.

The seat effect dwarfs all of this: mover win rates run 68-88% and responder
rates 15-39%, so any arena comparison must be side-balanced or it is unreadable.

### Which hyperparameters were used, and what can I tune?

The exact values for a given run are in `runs/train/<name>/config.json` and on
every published model card. The published family used:

| | |
| --- | --- |
| architecture | `mlp`, `resnet`, `attn`, `cpool`, preset `medium` |
| learning rate | **6e-4** for `cpool`/`attn`, **2e-3** for `resnet`/`mlp` |
| batch size | 1024 |
| weight decay | 1e-4 |
| epochs | 16 fixed (published) — see below |
| seed | 20260828 |
| val fraction | 0.05, split by canonical key |
| augmentation | symmetry augmentation on, ply-balanced sampling on |

**What is worth tuning, roughly in order of measured payoff:**

1. **Ply coverage of the corpus** — the largest measured effect in this project.
2. **The epoch budget.** `--patience N` stops on a stale combined validation
   loss and turns `--epochs` into a cap. Sixteen epochs is **not** a defensible
   budget for `cpool`; the converged run stopped at 43. Note that more epochs
   buys *deep-band* accuracy and costs shallow accuracy — it is not a free win.
3. **The learning rate, per architecture.** A single shared rate is not equal
   treatment; it privileges whichever architecture it was chosen for. Treating a
   shared hyperparameter as neutral has produced a wrong published conclusion
   here once already. See [`learning-rate-sweep.md`](learning-rate-sweep.md).
4. **`value_loss_weight`.** Untuned. The value head and policy head have very
   different label densities (~92% vs ~8% of rows), so the default 1.0 is a
   guess, not a finding.
5. **Ply-balanced sampling.** On by default. Given the distribution table above,
   it is doing a lot of work, and how much has not been isolated.

**What is known not to be worth it:** densifying a covered ply band (above), and
reading held-out accuracy as a proxy for play strength — it has failed to
predict the arena **five times**. Rank in the arena.

## Reproducing and expanding: the short version

```bash
# what you have, by content rather than by name
shasum -a 256 runs/oracle/corpus/*.npz

# what is in it
python -c "
import numpy as np, collections
d = np.load('runs/oracle/corpus/exact-sampled.npz')
print({k: (str(d[k].dtype), d[k].shape) for k in d})
print(sorted(collections.Counter(d['plies'].tolist()).items()))"
```

A corpus is identified by its **hash**, never by its filename.
`exact-sampled.npz` and `exact-sampled-v2.npz` are different corpora whose names
differ by one character, and confusing them produced a wrong published
conclusion. Since [`reproducibility.md`](reproducibility.md), every training run
records the corpus hash it read in `provenance.json`.
