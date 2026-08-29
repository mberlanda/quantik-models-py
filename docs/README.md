# Reading these documents

Twenty-three documents accumulate quickly. This is the order they make sense in,
and what each one is for.

Two conventions worth knowing before you start.

**Every number here is dated, and some of it is superseded on purpose.** When a
conclusion turned out to be wrong, the old reading is kept beside the correction
rather than deleted — the superseded run is usually the evidence for the fix.
Anything measured before 2026-08-30 used a learning rate chosen for the ResNet
and inherited by everything else; `learning-rate-sweep.md` explains what that
cost.

**Nothing here can be verified from a fresh clone.** The checkpoints and corpora
live under `runs/`, which is gitignored. The committed figures in `figures/` are
the only form in which the data leaves this machine.

---

## Start here

| | |
|---|---|
| [`decisions/0001-architecture-lineup.md`](decisions/0001-architecture-lineup.md) | **The umbrella document.** Which architectures are trained, which six were declined and why, and the methodology the whole comparison rests on. Everything below hangs off it. |
| [`benchmarks.md`](benchmarks.md) | **The pictures.** Six committed figures and what each does and does not establish. The fastest way to see the shape of the results. |

## The models, one document each

Layer by layer, with the design argument and the parameter accounting.

- [`architectures.md`](architectures.md) — the registry, the presets, the ONNX export invariants. Read before any individual model.
- [`architecture-resnet.md`](architecture-resnet.md) — the incumbent, and the one every hyperparameter was originally chosen for.
- [`architecture-mlp.md`](architecture-mlp.md) — the control that makes "convolution is worth having" falsifiable.
- [`architecture-constraint-pool.md`](architecture-constraint-pool.md) — the constraint prior wired into the architecture.
- [`attention-negative-result.md`](attention-negative-result.md) — the same bet without the prior, and the failure that turned out to be a hyperparameter.
- [`policy-value-training-paper.md`](policy-value-training-paper.md) — why the ResNet is trained the way it is. The oldest of these and the shallowest; the others set a standard it has not been raised to.

## How the models are measured

In the order the measurements get harder to fool.

1. [`shift-evaluation.md`](shift-evaluation.md) — accuracy on solved positions the corpus never saw. Held out up to the 192 symmetries.
2. [`autoplay.md`](autoplay.md) — the arena. Networks against each other, and why autoplay generates positions rather than labels.
3. [`oracle-benchmark.md`](oracle-benchmark.md) — **the only non-relative measurement.** The field against a fixed classical engine, which is what answers "is any of this good" rather than "which of these is better".

## What went wrong, and what it cost

These are the documents to read if you want to know how much to trust the rest.

- [`learning-rate-sweep.md`](learning-rate-sweep.md) — a rate chosen for one architecture and inherited by silence. It reversed three published conclusions.
- [`corpus-v3.md`](corpus-v3.md) — a corpus change that moved play strength a great deal and held-out accuracy not at all, and the longer training run that showed the change was compensating for an epoch budget rather than adding information. The document that puts a question mark over the metric everything above ranks with.

## Pipeline and operations

- [`pipeline.md`](pipeline.md) — how data gets from the solver to a trained model.
- [`labeling-strategy.md`](labeling-strategy.md) — what a label is here, and why none of them come from game outcomes.
- [`tensor-structure.md`](tensor-structure.md) — the `(B, 9, 4, 4)` input contract.
- [`autoplay-training.md`](autoplay-training.md) — feeding arena positions back into the corpus.
- [`retrain-and-finetune.md`](retrain-and-finetune.md) — `--init-from`, `--freeze`, `--patience`, and two silent failures.
- [`scaling-guide.md`](scaling-guide.md) — what changes when the presets grow.
- [`model-report.md`](model-report.md) — the generated per-run report.

## Publishing and playing

- [`publishing-to-hugging-face.md`](publishing-to-hugging-face.md) — what a Hub repository requires, the version-control model, and the documentation a published model needs. Every mistake in it publishes cleanly.
- [`play-service.md`](play-service.md) — the local move handler: the request contract, what it refuses, and the two checks that exist because their failures are silent.
- [`frontend-play.md`](frontend-play.md) — playing against these models from the browser.

## Archives

`nn-quest/` and `superpowers/` are dated journals and specs from earlier work.
They describe what was true when they were written and are deliberately not kept
current — the cross-reference check exempts them for that reason.
