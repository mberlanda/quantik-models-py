# 0002 — Versioning and release policy

**Status:** accepted, 2026-09-05
**Context:** the first PyPI release of `quantik-models`

## Decision

`quantik-models` follows [semantic versioning](https://semver.org/spec/v2.0.0.html)
on **its own version number**, independent of `quantik-core`. The first
public release is **1.0.0**.

## Why 1.0.0 rather than 0.x

Semantic versioning says nothing at all below 1.0.0 — `0.y.z` explicitly
means "anything may change at any time", which would be a false statement
about this package. The public surface has been stable across four trained
architectures and four published model repositories:

- the `(B, 9, 4, 4)` mover-relative input contract,
- the 64-logit policy and tanh value output,
- `model-checkpoint.v1` and the architecture registry that reads it,
- the arena and evaluator interfaces the play service is built on.

Four sets of weights are already published against that surface. Their model
cards tell readers to install this package and call into it; a `0.x` version
would advertise that those instructions may break arbitrarily, which is not
what is intended and not what will happen.

## Why not lockstep with `quantik-core`

`quantik-core-contracts`, `quantik-core-rust` and `quantik-core-py` share one
version number — tag contracts first, then py, then rust. That lockstep
exists because those three implement *one contract* and a version skew
between them is a correctness bug.

`quantik-models` is a **consumer** of that contract, not a member of it.
Joining the lockstep at 1.2.0 was considered and rejected:

- It would assert a correspondence that does not exist. `quantik-models`
  1.2.0 would imply "the 1.2.0 contract release", which is already what
  `quantik-core` 1.2.0 means.
- The two release on unrelated cadences. This package changes when a model,
  a trainer or an evaluation changes; the contract changes when a schema
  does. The first breaking change on either side would break the lockstep
  anyway, and a lockstep broken once is worse than one never claimed.
- Dependency ranges express the real relationship better and are checkable:
  `quantik-core>=1.2,<2`.

The **policy** is shared, and that is what "follow `quantik-core`" means
here: semantic versioning, a `CHANGELOG.md` with a section per release,
annotated `vX.Y.Z` tags, and publication through trusted publishing on a
GitHub Release.

## What counts as a breaking change

MAJOR, because code or a checkpoint that worked stops working:

- a change to the input encoding, the output layout, or the action index
  convention (`shape * 16 + position`);
- dropping support for a `model-checkpoint.v1` field that published
  checkpoints carry, or requiring a field they lack;
- removing or renaming a public function, a console script, or an extra;
- raising the `quantik-core` floor across a major boundary;
- raising the minimum Python version.

MINOR:

- a new architecture in the registry, a new evaluator runtime, a new
  endpoint on the play service, a new extra;
- a new optional argument with a default that preserves existing behaviour.

PATCH: fixes that do not change a documented interface.

**Retraining a published model is not a version change of this package.**
Weights are versioned by their Hub revision and released separately, under a
different licence. A new revision of `brpoplpush/quantik-cpool-c191-b6` bumps
nothing here.

## Consequences

- `__version__` in `src/quantik_models/__init__.py` is the single source of
  truth; `pyproject.toml` reads it statically and `publish.yml` refuses a
  release whose tag disagrees with it.
- The version and its changelog entry are checked by
  `tests/test_packaging.py`, so a release with no written-down diff fails
  before it is built.
- The release procedure is in [`DEVELOPMENT.md`](../../DEVELOPMENT.md).

## Rejected alternatives

| option | why not |
|---|---|
| **0.2.0**, staying pre-1.0 | Understates a surface that four published models already depend on, and postpones the first honest semver commitment indefinitely. |
| **1.2.0**, joining the contract lockstep | Claims a correspondence with the contract release that does not exist, and breaks on the first independent change either side makes. |
| **CalVer** (`2026.9.0`) | Says when a release happened and nothing about whether it is safe to take. The whole question a consumer of an ML package has is whether the model interface moved. |
