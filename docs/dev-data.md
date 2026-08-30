# Development data: what `runs/` holds and git does not

`runs/` is gitignored, is about 1.3 GB, and lives on one machine. It holds days
of solver time, hours of training, and every number this project has published.
Losing it means recomputing all of it.

The published model repositories do not help. **A model repo carries weights and
a card — not the corpus the weights were fitted to, nor the probe they were
scored against.** So a second repository exists for the development artefacts:
a Hugging Face **dataset** repo, `quantik-dev-data`.

## Staging it

```bash
python -m quantik_models.export.devdata runs/devdata
python -m quantik_models.export.devdata runs/devdata --only corpora --only probe
```

Copies, never moves. **Never uploads** — staging and publishing are separate
steps, because publishing is not reversible the way a local directory is. The
command prints the upload line rather than running it:

```bash
huggingface-cli upload-large-folder --repo-type dataset \
  <namespace>/quantik-dev-data runs/devdata
```

## What is in it

Six groups, each a directory with its own `README.md` saying what it is, what it
cost, how to reproduce it and how to extend it.

| group | what | why it is expensive |
| --- | --- | --- |
| `corpora` | the training corpora | days of exact-solver time |
| `enumerations` | every canonical live position, plies 1-8 | hours of search; `level08.npy` is 273 MB |
| `probe` | **held out** — 7,800 solved positions | small, but irreplaceable as a reference |
| `opening-book` | exhaustively solved shallow positions | the expensive shallow work |
| `checkpoints` | trained weights, config, metrics, provenance | hours per run |
| `evaluations` | arena games and shift output | hours of CPU per arena |

## Restoring

Paths are kept relative to the repository root, so a group restores to where the
tooling already expects it:

```bash
huggingface-cli download --repo-type dataset <namespace>/quantik-dev-data \
  --local-dir /tmp/devdata
cp -r /tmp/devdata/corpora/runs/ .
```

Then check what you got against `MANIFEST.json`, which carries a sha256 per
file.

## Two rules

**Identify a file by its hash, not its name.** `exact-sampled.npz` and
`exact-sampled-v2.npz` are different corpora whose names differ by one
character, and confusing them produced a wrong published conclusion — see
[`corpus-v3.md`](corpus-v3.md). Since [`reproducibility.md`](reproducibility.md),
every training run records the corpus hash it read.

**The probe is held out.** Never merge `probe/` into a corpus.
`data.merge_corpus` excludes probe keys from the *merged result*, not just from
incoming rows, because solving a position also labels its children. Sixteen
probe positions reached the first corpus exactly that way.

## Not restarting from scratch

Two mechanisms, and they are the reason `checkpoints` is worth hosting:

- **`--init-from`** warm-starts a new run from a staged checkpoint, and
  **`--freeze`** holds layers while later ones train. See
  [`retrain-and-finetune.md`](retrain-and-finetune.md), which documents two
  silent failure modes in that path.
- **`data.merge_corpus`** folds newly solved positions into an existing corpus,
  so extending coverage never means re-solving what is already labelled.

## Licence

Data and weights are **CC BY-NC 4.0**; the code that produced them is MIT.
Deliberately not an OSI licence: every OSI licence permits royalty-free
commercial use, which is the one thing this reserves.
