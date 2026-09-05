# Reading these documents

Every document here, in the order they make sense in. The index is
complete by test rather than by discipline — `test_docs_crossrefs.py`
fails if a document exists that this page does not link.

Two conventions worth knowing before you start.

**Every measurement is dated, and some of it is superseded on purpose.** When
a conclusion turned out to be wrong the old reading is kept beside the
correction, because the superseded run is usually the evidence for the fix.
Anything measured before 2026-08-30 used a learning rate chosen for the ResNet
and inherited by everything else; [`learning-rate-sweep.md`](learning-rate-sweep.md)
explains what that cost. Documents that carry measurements say so at the top;
documents that do not, do not.

**Nothing here can be verified from a fresh clone.** The checkpoints and
corpora live under `runs/`, which is gitignored. The committed figures in
[`figures/`](figures/) are the only form in which the data leaves this
machine — plus the four published models, which anyone can download.

---

## Start here

| | |
|---|---|
| [`models.md`](models.md) | **The published models.** How to load one, what the numbers mean, and the two things about the input contract that are easy to get silently wrong. Start here if you want to *use* a model. |
| [`decisions/0001-architecture-lineup.md`](decisions/0001-architecture-lineup.md) | **The umbrella document.** Which architectures are trained, which six were declined and why, and the methodology the comparison rests on. Start here if you want to know whether to believe it. |
| [`benchmarks.md`](benchmarks.md) | **The pictures.** Six committed figures and what each does and does not establish. |

## The models, one document each

Layer by layer, with the design argument and the parameter accounting.

- [`architectures.md`](architectures.md) — the registry, the presets, the ONNX export invariants. Read before any individual model.
- [`architecture-constraint-pool.md`](architecture-constraint-pool.md) — `cpool`, the constraint prior wired into the architecture. The strongest of the four.
- [`attention-negative-result.md`](attention-negative-result.md) — `attn`, the same bet without the prior, and the failure that turned out to be a hyperparameter.
- [`architecture-resnet.md`](architecture-resnet.md) — `resnet`, the incumbent, and the one every hyperparameter was originally chosen for.
- [`architecture-mlp.md`](architecture-mlp.md) — `mlp`, the control that makes "convolution is worth having" falsifiable.
- [`policy-value-training-paper.md`](policy-value-training-paper.md) — why the ResNet is trained the way it is. The oldest of these and the shallowest; the others set a standard it has not been raised to.

## How the models are measured

In the order the measurements get harder to fool.

1. [`shift-evaluation.md`](shift-evaluation.md) — accuracy on solved positions the corpus never saw. Held out up to all 192 symmetries.
2. [`autoplay.md`](autoplay.md) — the arena. Networks against each other, and why autoplay generates positions rather than labels.
3. [`oracle-benchmark.md`](oracle-benchmark.md) — **the only non-relative measurement.** The field against a fixed classical engine, which is what answers "is any of this good" rather than "which of these is better".

## What went wrong, and what it cost

The documents to read if you want to know how much to trust the rest.

- [`learning-rate-sweep.md`](learning-rate-sweep.md) — a rate chosen for one architecture and inherited by silence. It reversed three published conclusions.
- [`corpus-v3.md`](corpus-v3.md) — a corpus change that moved play strength a great deal and held-out accuracy not at all, and the longer run that showed it was compensating for an epoch budget. The document that puts a question mark over the metric everything above ranks with.

## Pipeline and operations

- [`pipeline.md`](pipeline.md) — how data gets from the solver to a trained model.
- [`corpora.md`](corpora.md) — what is in each corpus, the ply distribution, how to merge and extend one.
- [`labeling-strategy.md`](labeling-strategy.md) — what a label is here, and why none of them come from game outcomes.
- [`tensor-structure.md`](tensor-structure.md) — the `(B, 9, 4, 4)` input contract, and the fixture that tells the two encodings apart.
- [`retrain-and-finetune.md`](retrain-and-finetune.md) — `--init-from`, `--freeze`, `--patience`, and two silent failures.
- [`reproducibility.md`](reproducibility.md) — what each run records, and what it is not enough to promise.
- [`scaling-guide.md`](scaling-guide.md) — what changes when the presets grow.
- [`model-report.md`](model-report.md) — the generated per-run report.
- [`dev-data.md`](dev-data.md) — the dataset repositories `runs/` is staged to, and how to restore it.

## Publishing and playing

- [`publishing-to-hugging-face.md`](publishing-to-hugging-face.md) — what a Hub repository requires and the version-control model. Every mistake in it publishes cleanly.
- [`play-service.md`](play-service.md) — the local move handler: the request contract, what it refuses, and the two checks that exist because their failures are silent.
- [`frontend-play.md`](frontend-play.md) — playing against these models from the browser.

## Decisions

- [`decisions/0001-architecture-lineup.md`](decisions/0001-architecture-lineup.md) — the architecture lineup and the methodology.
- [`decisions/0002-versioning-and-release.md`](decisions/0002-versioning-and-release.md) — why this package versions independently of `quantik-core`, and what counts as a breaking change.

---

Working on the repository rather than reading about it:
[`../DEVELOPMENT.md`](../DEVELOPMENT.md).
