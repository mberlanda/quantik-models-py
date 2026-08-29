# Publishing to Hugging Face

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

A visual version of this document, organised around the fact that every
mistake here publishes cleanly:
<https://claude.ai/code/artifact/7f644a2a-24c3-4163-831b-747b422516a6>

Nothing in this repository uploads anything. Staging and pushing are
separate steps on purpose: a push is authenticated, public, and awkward to
undo, and a function that both prepares and publishes makes the dry run
impossible.

```bash
python -m quantik_models.export.huggingface \
  runs/train/swept-cpool/best staging/quantik-cpool-c191-b6 \
  --repo-id <your-org>/quantik-cpool-c191-b6 \
  --shift runs/eval/swept-2026-08-30/shift.json \
  --arena runs/eval/swept-2026-08-30/policy-p3/games.json
```

That writes a directory. The `model-index` numbers come out of the
evaluation artifacts rather than being typed onto the card, which is how a
card stops describing a checkpoint that was retrained after it was written.
What happens to the directory afterwards is below.

## What a model repository actually needs

A Hugging Face repo is a git repo with three files the Hub treats as
**structural rather than decorative**. Getting them wrong does not produce
an error; it produces a model page that works and that nobody finds.

**`README.md` with YAML front matter.** On the Hub the front matter is
metadata, not documentation. `license` gates the download button —
without it the Hub shows an "unlicensed" warning and some downstream
tooling refuses the repo outright. `pipeline_tag` and `library_name`
decide which widget the page renders and which snippet it offers.
`model-index` is what puts a number on the card and in search; a card that
states its accuracy only in prose is unsearchable by that accuracy.

**`config.json`.** There is no `AutoModel` for this architecture, so this
file is not a loading contract — it is the spec, readable without loading
anything. It carries `architecture_spec` verbatim plus the input and output
contracts. It deliberately has **no `auto_map`**: advertising a
`trust_remote_code` path that does not exist is worse than advertising
none, because it fails at the point where a user has already trusted you.

**`.gitattributes`.** Weight files must be LFS-tracked in the commit that
**first contains them**. This is the one mistake on this page that cannot
be fixed with a follow-up commit — a 7 MB `.safetensors` committed as a
plain blob is in the history for good, and the repair is a history
rewrite. The Hub's default `.gitattributes` covers `*.safetensors` and
**does not cover `*.onnx`**, which is exactly the trap here: the weights
are handled for you and the graph beside them silently is not. The staged
directory writes both patterns.

## What this project publishes beside the weights

| file | why it is there |
|---|---|
| `model.safetensors` | renamed from `weights.safetensors`; the Hub's viewers look for `model.` |
| `model.onnx` | opset 18, dynamic batch. Runs without torch and without this package. |
| `manifest.json` | the `model-checkpoint.v1` record, carried over unchanged |
| `training-report.json` | the epoch that produced these weights, and its metrics |
| `config.json`, `README.md`, `.gitattributes` | generated |

The manifest travels with the model because it is the only file that says
which contract version the weights speak, and a checkpoint whose contract
is unknown is a checkpoint nobody can safely load.

## The digests are checked before the push, not after

`stage()` recomputes the SHA-256 of every weight file and compares it to
the manifest, and refuses to return a directory that fails. The card
publishes those digests, and **a card whose digest disagrees with the file
beside it is worse than a card with no digest**: it invites a check that
fails, and the reader cannot tell a bad upload from a false claim.

## Version control

The Hub is git, and this is the part that most often gets treated as if it
were not.

**Every push is a commit; nothing is overwritten.** A repo keeps its whole
history, so a superseded checkpoint stays reachable at its old revision
even after `main` moves. That is the mechanism this project needs, given
that two of its four models were retrained after a methodology fix and the
old numbers are kept in the docs rather than deleted.

**Tag the revision that a paper or article cites.** `main` is a moving
target; a Hub tag (`v1.2.0`) is not. Every `from_pretrained` /
`snapshot_download` call takes a `revision`, and a citation without one
points at whatever is there when the reader arrives.

**Match the tag to `contract_version`.** This workspace versions in
lockstep — contracts, rust and py share one number — and a model repo that
invents its own scheme breaks the one property that makes the lockstep
useful: being able to read a version and know which schema the artifact
speaks.

**One repo per architecture, not one per run.** `swept-cpool` and
`lineup-cpool` are the same architecture at two learning rates. They belong
in one repo as two revisions with the story in between, not in two repos
that a reader has to know to compare. Which is exactly what this project
learned the hard way: the superseded run is not noise, it is the evidence
for the methodology fix.

## What the documentation has to say

The generated card covers the part that must be exact — shapes, contract,
hashes, parameter count. Three things it cannot generate, and that a person
has to write:

**What the model is bad at.** For these networks that is the shallow end.
Accuracy at plies 4-5 is 0.88-0.93 against 0.99+ at ply 12, and the corpus
holds no positions at all at plies 4-5. A card that reports one number
hides the only region where the model is genuinely uncertain.

**The masking requirement, in the first screen.** These models emit logits
over all 64 actions including illegal ones. An unmasked `argmax` plays
illegal moves. It is in the generated card because it is the single thing a
user of these weights can get silently wrong — silently, because an illegal
move looks like a bad move rather than like a bug.

**Where the numbers come from and what they do not cover.** Every win rate
this project publishes is against another network or a uniform-prior
control, not against a classical engine. That is a real limit and it
belongs on the card, not only in `benchmarks.md`.

## Pushing, when you decide to

Nothing here does this for you.

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli upload <your-org>/quantik-cpool-c191-b6 \
  staging/quantik-cpool-c191-b6 . --commit-message "cpool-c191-b6 at 6e-4"
```

Check the rendered card before tagging: front matter that fails to parse is
displayed as body text rather than rejected, so a malformed `model-index`
looks like a page that published fine.
