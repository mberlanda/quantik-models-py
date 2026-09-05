# Changelog

All notable changes to `quantik-models` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
What counts as a breaking change here is written down in
[`docs/decisions/0002-versioning-and-release.md`](docs/decisions/0002-versioning-and-release.md).

## Unreleased

## 1.0.0 - 2026-09-05

First release on PyPI. The package has been usable from a checkout for
months; what changes is that it is now installable, versioned and documented
as a library rather than as this workspace's training directory.

### Added

- **`quantik_models.hub`** — load the four published networks from the
  Hugging Face Hub in one call:

  ```python
  from quantik_models import hub
  evaluator = hub.load_evaluator("cpool")               # torch
  evaluator = hub.load_evaluator("cpool", runtime="onnx")  # no torch
  ```

  Short names resolve through a table of the published repo ids rather than
  a Hub query, a full `owner/repo` passes through unresolved so a fork needs
  no code change, and the downloaded artifact is checked against the digest
  in `manifest.json` for the runtime that will load it — `weights_hash` for
  safetensors, `onnx_hash` for the graph. The module is importable without
  torch.

  Every fetch failure is re-raised as `hub.HubError` carrying the remedy —
  the cache path and the command to run when offline with a cold cache, the
  terms link for a gated repo, the valid names for a typo, the commit list
  for an unknown revision — with the Hub's own exception kept as `__cause__`.
  A digest mismatch is re-fetched once before it is raised, because the usual
  cause is a truncated cache entry that would otherwise fail identically
  forever. `resolve()` reports the commit `main` resolved to, so a run made
  without an explicit `revision` is still reproducible.
- **`quantik-models-fetch` console script** (`hub.prefetch()` from Python) —
  fills the Hugging Face cache without loading anything, so a container build
  or an air-gapped host can be prepared before torch or onnxruntime exist.
- **`hub` extra** (`pip install 'quantik-models[hub]'`) carrying
  `huggingface-hub`. Kept out of the base install because nothing in
  training, evaluation or the arena fetches anything.
- **`quantik-models-play` console script**, so the play service starts the
  same way from an installed package as from a checkout.
- **`py.typed`**: the package now ships its type information.
- **`arena.registry.weights_path`**, the resolver behind the loader fix
  below.
- **[`DEVELOPMENT.md`](DEVELOPMENT.md)**, [`CHANGELOG.md`](CHANGELOG.md), and
  [`docs/models.md`](docs/models.md) — the published models, their numbers,
  and how to load them.

### Fixed

- **`load_evaluator` could not read a downloaded Hub repository.** It
  required `weights.safetensors`; `export.huggingface.stage` renames the
  file to `model.safetensors` on the way to the Hub, and that is the name in
  all four published repositories. The result was that the primary Python
  snippet on every published model card raised `FileNotFoundError`. The
  loader now resolves both names and, when it finds neither, says so naming
  both candidates instead of pointing at a file that was never meant to
  exist.
- **The build declared `setuptools>=68` while using PEP 639 metadata that
  needs 77.** A resolver that honoured the stated floor produced a wheel
  with no licence, or failed outright.
- **The generated model cards told readers `quantik-models` was not on PyPI**
  and to install from a git ref. They now name the release.
- **The card's Python snippet called a method that does not exist.**
  `evaluator.evaluate(boards)` is wrong twice over — an evaluator is
  callable, and the legality mask is a required argument with no default —
  so a reader following the card got an `AttributeError` on the one line the
  card exists to provide. Both live snippets were verified against a real
  download from the Hub before this release, and
  `tests/test_documented_snippets.py` now executes the documented call and
  fails if `Evaluator.__call__` and the documents disagree.

### Changed

- Version **0.1.0 → 1.0.0**. The package's public surface — the `(B, 9, 4, 4)`
  mover-relative input contract, the 64-logit policy and tanh value output,
  the architecture registry and the checkpoint manifest — has been stable
  across four trained architectures and four published model repositories.
  `0.1.0` understated that, and semantic versioning only says anything once
  the first stable release exists.
- `quantik-core` dependency pinned to `>=1.2,<2` rather than `>=1.2`, so a
  future major release of the core cannot silently satisfy this constraint.
- `mypy` is now configured in `pyproject.toml` and **enforced in CI**. It was
  previously declared in the `dev` extra and wired into nothing.
- Packaging metadata filled in: authors, classifiers, keywords, project URLs
  (including the Hub namespace), and a `MANIFEST.in` that ships the tests
  with the fixtures they need and excludes `runs/`, `docs/` and `staging/`.

### Documentation

- `README.md` is now about using the library and the models. Everything
  about checking out the workspace, the smoke pipeline, the corpora and the
  release process moved to `DEVELOPMENT.md`.
- The twenty-three documents in `docs/` were consolidated. Superseded
  narrative and dated working journals were removed in favour of the
  standing conclusions, which are what a reader can act on. See
  `docs/README.md`.
