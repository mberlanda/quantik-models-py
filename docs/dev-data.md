# Development data: what `runs/` holds and git does not

`runs/` is gitignored, is about 1.3 GB, and lives on one machine. It holds days
of solver time, hours of training, and every number this project has published.
Losing it means recomputing all of it.

The published model repositories do not help. **A model repo carries weights and
a card — not the corpus the weights were fitted to, nor the probe they were
scored against.** So two Hugging Face **dataset** repositories exist for the
development artefacts.

## Two repositories, split by churn

| repo | holds | size | rewritten |
| --- | --- | --- | --- |
| [`brpoplpush/quantik-dev-data`](https://huggingface.co/datasets/brpoplpush/quantik-dev-data) | solver output, corpora, enumerations, probe, opening book | 562 MB | never — appended to only |
| `brpoplpush/quantik-dev-runs` | checkpoints, sweeps, autoplay, evaluations | 882 MB | every training generation |

**The split is by churn, not by subject, and the reason is that Hub history is
permanent.** Every re-push of a file adds an LFS object, and the only way to
reclaim that space is `super_squash_history` — destructive and irreversible. In
one repository, the churny half inflates the history of the irreplaceable half,
and reclaiming it means running a destructive operation across data that took
days of solver time to produce. Split, `quantik-dev-runs` can be squashed freely
and `quantik-dev-data` never has to be.

It is also the public/private line. `quantik-dev-data` is the half that would
have community value if it were ever opened up; `quantik-dev-runs` contains
smoke runs and duplicates of the four published model repos and would not.

*Alternatives considered.* **One repository** — simpler to clone, but couples the
two lifetimes as above. **One repository per group (nine)** — makes every
restore a nine-way clone for no gain, since groups are never restored
individually in practice. **No dataset repo, rely on local disk** — this is the
status quo the audit rejected.

## Staging

```bash
python -m quantik_models.export.devdata runs/devdata --repo quantik-dev-data
python -m quantik_models.export.devdata runs/devruns --repo quantik-dev-runs
python -m quantik_models.export.devdata runs/devdata --only corpora   # one group
```

Copies, never moves. **Never uploads** — staging and publishing are separate
steps, because publishing is not reversible the way a local directory is. The
command prints the upload line rather than running it.

`--prune` clears each group directory first. Use it whenever a source file has
been renamed: staging never deletes what it did not write, so the old copy would
otherwise linger, and **a stale file in a backup is worse than a missing one
because it still hashes fine.** `--prune` never touches `.git`.

## Keeping it in sync

The staging directory can be a clone of the dataset repo, so the sync is a
normal git cycle:

```bash
cd quantik-models-py
git lfs install                                       # once
mkdir -p runs
git clone git@hf.co:datasets/brpoplpush/quantik-dev-data runs/devdata
git clone git@hf.co:datasets/brpoplpush/quantik-dev-runs runs/devruns

python -m quantik_models.export.devdata runs/devdata --repo quantik-dev-data --prune
cd runs/devdata && git status          # review before pushing
git add -A && git commit -m "..." && git push
```

Staging is non-destructive to anything it did not write, so `.git` survives a
re-stage. `scripts/sync_dev_data.sh` does the clone-or-pull and the re-stage,
and **stops before the push** so a diff is always reviewed by a person.

For a first upload, or when a clone would be slow, `hf upload-large-folder`
skips git entirely:

```bash
hf upload-large-folder --repo-type dataset brpoplpush/quantik-dev-data runs/devdata
```

## What is in it

Nine groups, each a directory with its own `README.md` saying what it is, what
it cost, how to reproduce it and how to extend it.

| group | repo | what | why it is expensive |
| --- | --- | --- | --- |
| `solver-output` | data | raw per-ply oracle output | **days of CPU** — the most expensive artefact here |
| `corpora` | data | the training corpora packed from it | derived, but only from the above |
| `enumerations` | data | every canonical live position, plies 1-8 | hours of search; `level08.npy` is 273 MB |
| `probe` | data | **held out** — 7,800 solved positions | small, but irreplaceable as a reference |
| `opening-book` | data | exhaustively solved shallow positions | the expensive shallow work |
| `checkpoints` | runs | weights, config, metrics, provenance, **resume state** | hours per run |
| `sweeps` | runs | learning-rate and hyperparameter grids | hours per architecture |
| `autoplay` | runs | self-play games and nominated positions | cheap to regenerate |
| `evaluations` | runs | arena games and shift output | hours of CPU per arena |

`checkpoints` carries `latest.pt` and `state.json` — the optimizer state that
lets an interrupted run resume instead of restarting at epoch zero. That is the
artefact that motivated this repository, and the first catalogue missed it.

Between the two repos, **every file under `runs/` is covered.** A gap check is
one command:

```bash
python - <<'EOF'
import json
from pathlib import Path
staged = set()
for d in ("runs/devdata", "runs/devruns"):
    for a in json.load(open(f"{d}/MANIFEST.json"))["artefacts"]:
        staged |= {f["path"] for f in a["files"]}
missing = [p for p in Path("runs").rglob("*") if p.is_file()
           and not str(p).startswith(("runs/devdata", "runs/devruns"))
           and str(p) not in staged]
print(f"unstaged: {sum(p.stat().st_size for p in missing)/1e6:.2f} MB in {len(missing)} files")
EOF
```

## Restoring

Paths are kept relative to the repository root, so a group restores to where the
tooling already expects it:

```bash
hf download --repo-type dataset brpoplpush/quantik-dev-data --local-dir /tmp/devdata
cp -r /tmp/devdata/corpora/runs/ .
```

**A clone is not a working tree.** The dataset layout is `<group>/runs/...`, so
nothing under `runs/devdata/` is on a path the trainer looks at — the `cp` above
is what makes the data usable, and it is not optional.

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
