# Policy/Value Trainer, Checkpoint Export, and Companion Paper — Design

Date: 2026-07-15
Repo: quantik-models-py
Status: approved (brainstorm 2026-07-15)

## Goal

Implement the first PyTorch policy/value training slice for Quantik:
dataset loader over materialized `.npz` training views, a scalable
AlphaZero-style network, a training CLI, a smoke-training test, and
checkpoint export as safetensors weights plus a `model-checkpoint.v1`
manifest. Ship with runnable example scripts, a smoke-to-target scaling
guide, and a research-grade companion paper with verified arXiv
references.

Contract constraints (from quantik-core-contracts
`docs/policy-value-model-project.md`):

- input tensor `(9, 4, 4)` float32 (8 bitboard planes + side-to-move),
- policy head: 64 logits over `action_index = shape * 16 + position`,
  masked by `legal_action_mask`,
- value head: scalar `tanh` in `[-1, 1]`,
- 50-100 MB checkpoint envelope for the eventual target model,
- weights detached from core libraries; manifest is the handshake.

## Decisions (from brainstorm)

1. **Scope**: trainer + smoke-training test + safetensors export with
   `model-checkpoint.v1` manifest in one slice.
2. **Architecture**: configurable residual network, tiny default.
   Presets scale the same architecture from CI-fast to the contract
   envelope.
3. **Compute/format**: CPU-first (MPS/CUDA opportunistic, never
   required); safetensors first, ONNX deferred to a later slice.
4. **Paper**: Markdown at `docs/policy-value-training-paper.md`,
   arXiv-preprint-shaped, references verified via web search.
5. **Extras**: runnable well-documented example scripts; explicit
   scaling documentation (smoke -> small -> target).

## Package layout

All under `src/quantik_models/`; torch and safetensors live behind an
optional `[torch]` extra so the base install stays NumPy-only.

### `data/dataset.py`

- `TrainingSample`: tensors `(9,4,4)` f32, `policy_target (64,)` f32,
  `value_target` f32, `sample_weight` f32, `legal_mask (64,)` bool,
  `source_tag` str.
- `QuantikNpzDataset(paths: list[Path], split: Split, ...)`:
  - loads one or more `.npz` training views (observation + selfplay
    views produced by `quantik-models-materialize`) and concatenates,
  - expands the `uint64` `legal_action_mask` into a `(64,)` bool mask
    (bit i == action_index i),
  - deterministic sharding: split assignment is
    `sha1(tensor_bytes || policy_bytes || source_tag) mod 100` mapped to
    train `[0,80)`, val `[80,90)`, test `[90,100)`. Stable across row
    order, file grouping, and machines. Split fractions configurable but
    default 80/10/10.
- Plain NumPy internally; a thin `TorchViewDataset` adapter implements
  `torch.utils.data.Dataset` (import guarded behind the extra).

### `model/policy_value_net.py`

- `PolicyValueNetConfig(channels, blocks, value_hidden)` dataclass.
- `PolicyValueNet(nn.Module)`:
  - stem: 3x3 conv `9 -> channels` + BatchNorm + ReLU,
  - trunk: `blocks` pre-activation residual blocks (3x3 conv pairs),
  - policy head: 1x1 conv -> flatten -> linear -> 64 logits,
  - value head: 1x1 conv -> flatten -> linear -> ReLU -> linear -> tanh.
- Presets (named, exact sizes asserted by tests within a tolerance):
  - `smoke`: channels=16, blocks=2 (well under 1 MB) — CI and examples,
  - `small`: channels=64, blocks=4 (single-digit MB) — local baseline,
  - `target`: channels/blocks solved to land the safetensors file in
    50-100 MB (e.g. ~384 channels x ~18 blocks; final numbers fixed by a
    size test at implementation time).
- `forward(x) -> (policy_logits, value)`; masking is NOT baked into the
  module — a separate `masked_log_softmax(logits, legal_mask)` helper
  applies `-inf` to illegal logits so the training loss and any engine
  adapter share one implementation.

### `train/` + CLI `quantik-models-train`

- Loss: `sample_weight`-weighted soft-target cross-entropy between
  `masked_log_softmax` outputs and `policy_target`, plus MSE on value,
  plus weight decay via AdamW. Total = `policy + value_loss_weight *
  value` (default 1.0).
- Metrics per epoch (train and val): policy loss, value MSE, top-1
  agreement with argmax of `policy_target`, pre-mask illegal probability
  mass (softmax without mask, summed over illegal actions) — the
  contract gate wants this near zero after masking, we also track how
  much the raw network learns legality.
- Determinism: `--seed` seeds Python/NumPy/torch; dataloader workers=0
  by default.
- Device: `--device auto|cpu|mps|cuda`, auto prefers cuda > mps > cpu.
- CLI inputs: `--npz` (repeatable), `--preset` or explicit
  `--channels/--blocks`, `--epochs --batch-size --lr --value-loss-weight
  --seed --device --out-dir`.
- Outputs to `--out-dir`: `weights.safetensors`, `manifest.json`
  (model-checkpoint.v1), `training-report.json` (final metrics, dataset
  sizes, split counts, config, elapsed).

### `export/checkpoint.py`

- `export_checkpoint(model, config, metrics, out_dir)`:
  - writes `weights.safetensors` (flat state dict),
  - writes `model-checkpoint.v1` manifest with the exact field surface
    of the contract schema (release, inputs = tensor-board.v1 /
    action-index.v1 / legal mask, outputs, `weights_format:
    "safetensors"`, architecture config, calibration/metrics metadata,
    data provenance = source `.npz` names + row counts),
  - manifest field names follow `quantik-core-contracts`
    `schemas/model-checkpoint-v1.json`; implementation reads the schema
    and the core-py reader to match exactly.
- Acceptance: the manifest round-trips through the existing
  `quantik-core-py` checkpoint manifest reader/validator in a test.

## Example scripts (`examples/`)

Well-commented, runnable end-to-end on a laptop:

- `examples/train_smoke.sh` — generates tiny pipeline data via
  `scripts/run_smoke_pipeline.sh` (or accepts `OUT` pointing at existing
  data), then trains the `smoke` preset for a few epochs and exports a
  checkpoint. Prints where everything landed.
- `examples/inspect_checkpoint.py` — loads `manifest.json` +
  `weights.safetensors`, prints architecture/config/metrics, runs a
  forward pass on the empty board and on one sample from the dataset,
  prints top-5 policy actions with `shape:position` decoding and the
  value estimate, demonstrating legality masking.
- `examples/train_small_local.sh` — the `small` preset on a larger
  locally generated dataset; documents expected wall-clock on CPU/MPS.

## Scaling documentation (`docs/scaling-guide.md`)

Explicit smoke -> small -> target path:

- table of presets: channels, blocks, parameter count, safetensors
  size, expected CPU/MPS epoch time on the smoke corpus,
- what to change when scaling (only `--preset`; data volume guidance:
  smoke corpus is for plumbing, `small` wants >=100k rows, `target`
  wants the full depth-6 book + self-play corpus per the contracts
  data milestones),
- overfitting warning: parameters vs distinct reachable states;
  reference to the paper's discussion,
- how the `target` preset satisfies the 50-100 MB contract envelope and
  how to verify (`ls -l weights.safetensors`, manifest metadata),
- pointer to `model-checkpoint.v1` acceptance gates.

## Companion paper (`docs/policy-value-training-paper.md`)

arXiv-preprint-shaped Markdown:

1. Abstract.
2. Introduction: Quantik, the book-vs-model motivation.
3. Background: rules, state space, symmetry (canonical keys, orbit
   sizes), existing pipeline artifacts.
4. Method: data (label precedence, sample weighting, visit-count
   distillation), architecture, masked policy loss, deterministic
   sharding, checkpoint contract.
5. Tradeoffs: distillation-from-search vs pure self-play;
   overparameterization in a small domain vs the 50-100 MB envelope;
   masking at loss time vs architecture time; book memorization vs
   generalization; calibration.
6. Planned experiments: fixed-budget search improvement, book-frontier
   H2H, illegal-mass, active-learning loop.
7. Applications in other domains/games: Go/chess/shogi (AlphaZero
   line), Hex (ExIt), KataGo efficiency improvements, AlphaTensor
   (matrix multiplication), AlphaDev (sorting), chip floorplanning,
   combinatorial optimization, theorem proving.
8. References: every arXiv ID verified via web search before inclusion;
   no from-memory citations.

## Testing

- Unit: dataset load/concat/mask-expansion/split-determinism (same rows
  -> same split regardless of order); model forward shapes and preset
  size bounds; masked_log_softmax puts zero probability on illegal
  actions; export writes loadable safetensors + schema-valid manifest.
- Smoke training test (marked, still fast): train `smoke` preset ~50
  steps on committed-or-generated tiny `.npz`, assert train loss
  decreases and manifest validates through the quantik-core-py reader.
- Existing test suite stays green; torch-dependent tests skip cleanly
  when the `[torch]` extra is absent.

## CI

- New workflow `train-smoke.yml` (separate from `e2e-data-pipeline.yml`
  to keep that job fast): checks out the four repos, installs the
  `[torch]` extra (CPU torch), reuses the tiny pipeline to produce
  `.npz`, runs `quantik-models-train --preset smoke`, runs
  `examples/inspect_checkpoint.py` against the result, uploads the
  checkpoint as an artifact.

## Docs updates in the same slice

- `README.md`: train/export quickstart + examples pointer.
- `docs/pipeline.md`: extend the flow past materialization into
  train/export.
- `docs/model-report.md`: mark trainer/exporter slices as implemented.
- quantik-core-contracts `docs/implementation-status.md`:
  `model-checkpoint.v1` gains a Produced surface (separate PR).
- Progress board.

## Execution

- Coding tasks dispatched to sonnet subagents once the plan is written
  (dataset/model/tests and trainer/export/CLI as parallelizable units).
- Paper written in the main thread with web-verified references.
- PR per repo, Copilot review with 2/5/10/20 backoff, no co-author
  trailers, squash on green.

## Out of scope

- ONNX export, model-guided engine adapter, Rust book-guided autoplay,
  active-learning loop, frontend. These follow in later slices.
