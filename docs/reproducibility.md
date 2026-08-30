# Reproducing a training run

Every published number in this repository comes from a run on one laptop. This
document says what is recorded about each run, what that is enough for, and
what it is not enough for.

## What is recorded, and where

A supervised run writes four files into `runs/train/<name>/`:

| file | carries |
| --- | --- |
| `config.json` | every hyperparameter, **resolved** — the seed, the learning rate, and the device that was actually used |
| `provenance.json` | the commit, the machine, the dependency versions, and the corpus by **hash** |
| `metrics.jsonl` | one JSON object per epoch |
| `best/` | the checkpoint, its `manifest.json`, and `training-report.json` |

`best/training-report.json` carries a **copy** of the resolved config and the
full provenance record, so a checkpoint that leaves the machine — to the Hub, to
a serving container — takes its provenance with it. `runs/` is gitignored; that
copy is what a downloader actually gets.

### Resolved, not requested

`config.json` records what ran, not what was asked for.

- The learning rate is recorded resolved. A config that says `null` does not
  reproduce the run it describes.
- The **device** is recorded resolved. It used to be recorded as `"auto"` — the
  request. A run on MPS and a run on CPU are not the same run, and `"auto"`
  says which was asked for, not which happened.

### The corpus is recorded by hash

`config.json` records a filename. `provenance.json` records a `sha256`.

This is not belt-and-braces. `exact-sampled.npz` and `exact-sampled-v2.npz` are
different corpora with names one character apart, and confusing them is exactly
what produced the wrong conclusion corrected in [`corpus-v3.md`](corpus-v3.md):
a published document read a v1 baseline as a v2 one and drew a causal claim from
a comparison that moved two variables. **A filename is not an identity.**

### The commit, and a link to it

`provenance.json` records the commit, the branch, the remote, and a browsable
`commit_url`. Every published model card carries the same, and its install
snippet is **pinned to that commit** rather than tracking `main` — an unpinned
`git+https://…` gives a reader code the card does not describe.

`dirty` is recorded too, and it is not a footnote. **A dirty tree means the
recorded commit does not describe the code that ran**, and no permalink fixes
that. The card says so in the provenance table rather than quoting the commit as
though it were authoritative.

## Reproducing a published checkpoint

Every value below comes from the model card's "Reproducing this checkpoint"
table, or from `training-report.json` in the same repository.

```bash
pip install 'quantik-models[torch] @ git+https://github.com/mberlanda/quantik-models-py@<commit>'

# Verify you have the corpus the run used — not one with the right name.
shasum -a 256 runs/oracle/corpus/<corpus>

python -m quantik_models.train.supervised \
  --name repro --corpus runs/oracle/corpus/<corpus> \
  --arch <arch> --preset <preset> \
  --epochs <epochs> --batch-size <batch> --lr <lr> --seed <seed>
```

Then compare `best/manifest.json`'s `weights_hash` against the card's.

## What this is not enough for

**Bit-identical weights are not promised, and probably will not happen.**
Seeding makes a run deterministic on the same machine with the same versions.
It does not survive a change of accelerator, a change of `torch`, or in general
a change of thread count — floating-point reduction order is not fixed across
any of those. What the record gives you is the ability to know *which* of those
changed, which is the difference between a reproduction that disagrees and one
that cannot be interpreted at all.

**One seed.** Every checkpoint in the published family is seed `20260828`. The
spread across seeds has never been measured, so no margin in any lineup table
has a run-to-run error bar under it. This is stated on every model card and it
is the largest single caveat on every number here.

**Held-out accuracy does not rank play strength.** It has failed to, five times.
Reproducing a checkpoint's validation numbers does not reproduce its standing in
the arena, and the arena is the ranking that matters. See
[`shift-evaluation.md`](shift-evaluation.md) and [`benchmarks.md`](benchmarks.md).

**Checkpoints trained before this record existed do not have it.** The four
published models predate `provenance.json`. Their cards carry the seed and every
hyperparameter, which was already recorded, but not the commit, the machine, or
the corpus hash. Restaging them fills in everything except the commit, which is
not recoverable after the fact — the honest options are to leave it absent or to
retrain, and leaving it absent is what the card does.

## Hardware, for the runs on record

All published training ran on one machine: Apple silicon, `arm64`, macOS, with
`torch` on the **MPS** backend. `provenance.json` records the resolved platform
string and accelerator per run rather than relying on this paragraph, because
this paragraph will eventually be wrong.
