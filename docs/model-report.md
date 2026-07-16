# Quantik Policy/Value Model Report

This report captures the first model-project target for `quantik-models-py`.
The repository consumes Quantik core artifacts; it does not vendor model weights
inside `quantik-core-py` or `quantik-core-rust`.

## Goal

Build a portable 50-100 MB policy/value model that can be loaded by Quantik
engine adapters and evaluated against the extensive opening book. The model is
intended to generalize beyond static opening-book lookup while preserving a
stable artifact boundary with the core libraries.

## Repository Boundary

| Repository | Responsibility |
| --- | --- |
| `quantik-core-contracts` | Contract IDs, schemas, docs, fixtures, validators, release bundles. |
| `quantik-core-rust` | High-throughput opening-book, search, observation, H2H, and self-play producers. |
| `quantik-core-py` | Artifact readers, QFEN/bitboard/action helpers, tensor helpers, checkpoint manifest validation. |
| `quantik-models-py` | Training views, model architecture, training loops, autoplay experiments, exported checkpoints. |

The core libraries should expose APIs for reading, validating, probing, and
using model artifacts. The model repository owns the design and training of the
artifacts themselves.

## Clone The Workspace

```bash
export QUANTIK_NS="$HOME/Code/quantik-ns"
mkdir -p "$QUANTIK_NS"
cd "$QUANTIK_NS"

git clone https://github.com/mberlanda/quantik-core-contracts.git
git clone https://github.com/mberlanda/quantik-core-rust.git
git clone https://github.com/mberlanda/quantik-core-py.git
git clone https://github.com/mberlanda/quantik-models-py.git
```

## Current Artifact Flow

```text
contracts validate
  -> rust opening book, depth 6
  -> rust position generation with book references
  -> rust observations and H2H results
  -> rust self-play rows
  -> python artifact readers/tensor helpers
  -> quantik-models-py materialized training view
  -> future trainer/checkpoint exporter
```

Run the smoke flow with:

```bash
cd "$QUANTIK_NS/quantik-models-py"
scripts/run_smoke_pipeline.sh
```

The script accepts environment overrides such as `OUT`, `BOOK_DEPTH`,
`OPENING_POSITIONS`, `MCTS_ITERATIONS`, `SELFPLAY_GAMES`, and `WORKERS`.

## Training Tensor

The first training view stores NumPy arrays in a compressed `.npz` file:

| Field | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| `tensors` | `(n, 9, 4, 4)` | `float32` | 8 bitboard planes plus side-to-move plane. |
| `policy_target` | `(n, 64)` | `float32` | Normalized visit distribution over `action-index.v1`. |
| `value_target` | `(n,)` | `float32` | Side-to-move value in `[-1, 1]`. |
| `sample_weight` | `(n,)` | `float32` | Source confidence and priority. |
| `legal_action_mask` | `(n,)` | `uint64` | Runtime legality mask. |
| `side_to_move` | `(n,)` | `uint8` | Player to move. |
| `source_tags` | `(n,)` | string | Provenance and source labels. |

The policy action index is always:

```text
action_index = shape * 16 + position
```

Legal move filtering remains an engine/rules invariant. The model may learn
legality, but engine adapters must still apply `legal_action_mask`.

## Label Strategy

Policy labels come from normalized visit counts:

- `selfplay.v1` root MCTS visits,
- `observation.v1` dense `policy_visits[64]`,
- future `search-summary.v1` only after root visits/Q-values are contracted.

Value labels use this precedence:

1. exact/tablebase/opening-book labels,
2. bounded solver labels,
3. strong search labels,
4. generic MCTS/minimax/beam search labels,
5. self-play outcomes,
6. H2H calibration evidence,
7. heuristic or synthetic labels.

Single-visit observation policies are retained for smoke/debug data and tagged
as `policy:single-visit`; they are weak targets for real training.

## Autoplay Strategy

The staged plan is:

1. supervised bootstrap from depth-6 opening-book references, observations, and
   MCTS self-play,
2. book-guided self-play that starts from opening-book guidance up to depth 6
   and then mixes MCTS, beam, minimax, and model-guided search,
3. active learning from disagreements between model-guided engines and stronger
   search/reference engines.

The immediate Rust gap is a configurable self-play/autoplay runner that accepts
`--book PATH`, an opening policy, engine pairs, and provenance tags while still
exporting contract-compatible rows.

## First Acceptance Gates

- Contract validation passes in `quantik-core-contracts`.
- Rust/Python artifact parity remains green for `observation.v1`,
  `game-result.v1`, `selfplay.v1`, and Parquet surfaces.
- `quantik-models-materialize` roundtrips the smoke rows and writes stable
  `.npz` views.
- A future checkpoint exporter emits `model-checkpoint.v1` manifests and keeps
  weights detached from the core libraries.
- The trained model can be evaluated as an engine adapter against the opening
  book and existing baselines.

## Open Implementation Slices

Implemented:

- `quantik-models-train`, the PyTorch policy/value training CLI, at the
  `smoke`/`small`/`target` presets (see `docs/scaling-guide.md`).
- Deterministic content-hash train/validation/test sharding.
- Checkpoint export to `weights.safetensors` plus a `model-checkpoint.v1`
  `manifest.json` and `training-report.json`.

Open:

- Add core-library checkpoint loading/probe APIs without bundling weights.
- Add Rust book-guided autoplay/self-play export up to depth 6.
- Add cross-stack evaluation reports comparing model-guided engines against
  opening-book, MCTS, beam, and minimax baselines.
- Add a frontend/game-session layer for human-vs-human, human-vs-CPU, and
  CPU-vs-CPU autoplay using the same core rules and engine adapters.
