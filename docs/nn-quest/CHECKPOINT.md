# NN Quest — Resume Checkpoint

**Goal:** Train the first neural network that beats the existing Quantik
strategies (minimax / MCTS / beam / random) head-to-head.

**Status:** PHASE 0 — environment + baseline recon

## How to resume

```bash
export NNQ=/Users/mauroberlanda/Code/quantik-ns/quantik-models-py
cd "$NNQ"
.venv/bin/python -V          # 3.13.14
.venv/bin/python -c "import torch, quantik_core, quantik_models"
cat docs/nn-quest/JOURNAL.md # full narrative log
```

## Environment (established 2026-08-27)

- Workspace root: `/Users/mauroberlanda/Code/quantik-ns`
- Repos: `quantik-core-contracts`, `quantik-core-rust`, `quantik-core-py`,
  `quantik-models-py`, `quantik-qfen-visualizer`, `quantik-workspace`
- Venv: `quantik-models-py/.venv` (python 3.13.14, pyenv), editable installs of
  `quantik-core-py[arrow]` and `quantik-models[dev,arrow,torch]`.
- torch 2.13.0, MPS available.

## Key facts discovered

- Tensor contract: `(9, 4, 4)` float32 — 8 channels for player x shape,
  channel 8 = side to move (broadcast constant). `quantik_core.ml_data.qfen_to_tensor`.
- Action index: `shape * 16 + position`, 64 actions total.
- Net: `quantik_models.model.policy_value_net.PolicyValueNet`, presets
  `smoke` (c16/b2), `small` (c64/b4), `target` (c256/b13).
- Existing engines (python `quantik_core`, mirrored in rust):
  `minimax.MinimaxEngine` (exact solver at max_depth=16), `mcts.MCTSEngine`,
  `beam_search.BeamSearchEngine`, plus random.
- `MinimaxEngine.solve` == exact game solve, so "beating" baselines must be
  measured at a **fixed per-move time budget** (the same framing the rust
  `bench::head_to_head` + `fixed_time_adapters` harness uses).

## Next actions

1. Probe exact-solve cost from the empty board (feasibility of perfect labels).
2. Build the python arena (side-balanced paired games) + baseline table.
