# Policy/Value Trainer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** First PyTorch policy/value trainer for quantik-models-py: `.npz` dataset loader with deterministic sharding, scalable residual network, training CLI, safetensors + `model-checkpoint.v1` export, smoke test, examples, scaling guide, and CI workflow.

**Architecture:** Pure-NumPy dataset layer reusing `quantik_models.data.materialize.load_npz`; torch-dependent model/train/export modules importable only with the `[torch]` extra; export validated against the `quantik-core-py` manifest reader. Spec: `docs/superpowers/specs/2026-07-15-policy-value-trainer-design.md`.

**Tech Stack:** Python >=3.12, numpy>=2, torch>=2.4 (optional extra), safetensors, pytest, mypy.

## Global Constraints

- `QUANTIK_NS` is the sibling-checkout namespace root per README (e.g. `export QUANTIK_NS="$HOME/Code/quantik-ns"`); repos live at `$QUANTIK_NS/quantik-models-py` and `$QUANTIK_NS/quantik-core-py`.
- `$VENV` is any Python >=3.12 virtualenv with `pip install -e ".[dev,torch]"`.
- Repo: `$QUANTIK_NS/quantik-models-py`, branch `policy-value-trainer` (exists).
- Base install stays numpy-only: torch/safetensors imports ONLY inside `[torch]`-extra modules; torch-dependent tests `pytest.importorskip("torch")`.
- Commit messages: NO Co-Authored-By trailer, no tool prefixes.
- Action index is always `shape * 16 + position` (0..63).
- Input tensor `(9, 4, 4)` float32; policy 64 logits; value tanh scalar.
- Presets: smoke=(16 ch, 2 blocks), small=(64, 4), target=(256, 13) — target safetensors must land in [50MB, 100MB].
- Determinism: sharding by content hash (below); training seeded.
- Run tests with: `cd $QUANTIK_NS/quantik-models-py && python -m pytest tests/<file> -q` (use `$VENV`; `pip install -e ".[dev,torch]"` plus `pip install safetensors` there first if not present).
- `quantik_core` (sibling checkout `$QUANTIK_NS/quantik-core-py/src` on PYTHONPATH) is used by export-validation tests; skip cleanly when unavailable.

---

### Task 1: Dataset module with deterministic sharding

**Files:**
- Create: `src/quantik_models/data/dataset.py`
- Test: `tests/test_dataset.py`
- Modify: `pyproject.toml` (extend `torch` extra with safetensors)

**Interfaces:**
- Consumes: `quantik_models.data.materialize.load_npz(path) -> TrainingDatasetView` (existing; fields `tensors (n,9,4,4) f32`, `policy_target (n,64) f32`, `value_target (n,) f32`, `sample_weight (n,) f32`, `legal_action_mask (n,) u64`, `side_to_move (n,) u8`, `source_tags: tuple[tuple[str,...],...]`).
- Produces:
  - `Split = Literal["train", "val", "test"]`
  - `expand_legal_mask(mask: np.ndarray) -> np.ndarray` — `(n,) uint64 -> (n, 64) bool`, bit i (LSB first) = action i legal.
  - `split_assignments(tensors, policy_target, source_tags, *, train_pct=80, val_pct=10) -> np.ndarray[str]` — per-row `"train"/"val"/"test"`.
  - `class LoadedTrainingData` dataclass: `tensors (m,9,4,4) f32`, `policy_target (m,64) f32`, `value_target (m,) f32`, `sample_weight (m,) f32`, `legal_mask (m,64) bool`, `source_tags: tuple[str, ...]` (joined with `|`).
  - `load_training_data(paths: Sequence[str | Path], split: Split | None = None, *, train_pct=80, val_pct=10) -> LoadedTrainingData` — concatenates all files then filters by split (`None` = all rows).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_dataset.py
from __future__ import annotations

from pathlib import Path

import numpy as np

from quantik_models.data.dataset import (
    LoadedTrainingData,
    expand_legal_mask,
    load_training_data,
    split_assignments,
)


def _write_view(path: Path, n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    tensors = rng.random((n, 9, 4, 4), dtype=np.float32)
    policy = rng.random((n, 64), dtype=np.float32)
    policy /= policy.sum(axis=1, keepdims=True)
    np.savez_compressed(
        path,
        tensors=tensors,
        policy_target=policy,
        value_target=rng.uniform(-1, 1, n).astype(np.float32),
        sample_weight=np.ones(n, dtype=np.float32),
        legal_action_mask=rng.integers(1, 2**63, n, dtype=np.uint64),
        side_to_move=rng.integers(0, 2, n, dtype=np.uint8),
        source_tags=np.asarray([f"tag{i}" for i in range(n)], dtype=np.str_),
    )


def test_expand_legal_mask_bits() -> None:
    mask = np.asarray([0b101, 1 << 63], dtype=np.uint64)
    expanded = expand_legal_mask(mask)
    assert expanded.shape == (2, 64)
    assert expanded.dtype == np.bool_
    assert expanded[0, 0] and not expanded[0, 1] and expanded[0, 2]
    assert expanded[1, 63] and not expanded[1, 0]
    assert expanded[0].sum() == 2 and expanded[1].sum() == 1


def test_split_assignments_deterministic_and_order_independent(tmp_path: Path) -> None:
    _write_view(tmp_path / "a.npz", 200, seed=1)
    data = load_training_data([tmp_path / "a.npz"])
    a1 = split_assignments(data.tensors, data.policy_target, data.source_tags)
    a2 = split_assignments(data.tensors, data.policy_target, data.source_tags)
    assert (a1 == a2).all()
    perm = np.random.default_rng(0).permutation(len(a1))
    a3 = split_assignments(
        data.tensors[perm],
        data.policy_target[perm],
        tuple(data.source_tags[i] for i in perm),
    )
    assert (a3 == a1[perm]).all()  # assignment travels with the row
    # all three splits are non-empty at n=200 and fractions are sane
    frac_train = float((a1 == "train").mean())
    assert 0.7 < frac_train < 0.9
    assert (a1 == "val").any() and (a1 == "test").any()


def test_load_training_data_concat_and_split(tmp_path: Path) -> None:
    _write_view(tmp_path / "a.npz", 60, seed=2)
    _write_view(tmp_path / "b.npz", 40, seed=3)
    full = load_training_data([tmp_path / "a.npz", tmp_path / "b.npz"])
    assert isinstance(full, LoadedTrainingData)
    assert full.tensors.shape == (100, 9, 4, 4)
    assert full.legal_mask.shape == (100, 64)
    assert full.legal_mask.dtype == np.bool_
    parts = [
        load_training_data([tmp_path / "a.npz", tmp_path / "b.npz"], split=s)
        for s in ("train", "val", "test")
    ]
    assert sum(p.tensors.shape[0] for p in parts) == 100
    # rows in val/test do not depend on how files are listed
    swapped = load_training_data([tmp_path / "b.npz", tmp_path / "a.npz"], split="val")
    assert {t.tobytes() for t in swapped.tensors} == {t.tobytes() for t in parts[1].tensors}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_dataset.py -q`
Expected: FAIL / ERROR with `ModuleNotFoundError: No module named 'quantik_models.data.dataset'`

- [ ] **Step 3: Implement `src/quantik_models/data/dataset.py`**

```python
"""Training-data loading over materialized `.npz` views.

Pure NumPy: this module must import without torch so the base install
can inspect datasets. Sharding is content-addressed so a row's split is
stable across file order, file grouping, and machines.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

from .materialize import load_npz

Split = Literal["train", "val", "test"]

_TAG_JOIN = "|"


def expand_legal_mask(mask: np.ndarray) -> np.ndarray:
    """Expand `(n,)` uint64 bitmasks into `(n, 64)` bool, LSB = action 0."""
    if mask.dtype != np.uint64:
        raise ValueError(f"legal mask must be uint64, got {mask.dtype}")
    bits = np.arange(64, dtype=np.uint64)
    return ((mask[:, None] >> bits[None, :]) & np.uint64(1)).astype(np.bool_)


def split_assignments(
    tensors: np.ndarray,
    policy_target: np.ndarray,
    source_tags: Sequence[str],
    *,
    train_pct: int = 80,
    val_pct: int = 10,
) -> np.ndarray:
    """Deterministic per-row split labels.

    bucket = sha1(tensor_bytes || policy_bytes || tag) mod 100:
    [0, train_pct) -> train, [train_pct, train_pct+val_pct) -> val,
    the rest -> test.
    """
    if not 0 < train_pct + val_pct < 100:
        raise ValueError("train_pct + val_pct must be in (0, 100)")
    labels = np.empty(len(source_tags), dtype=object)
    for i, tag in enumerate(source_tags):
        digest = hashlib.sha1(
            tensors[i].tobytes() + policy_target[i].tobytes() + tag.encode("utf-8")
        ).digest()
        bucket = int.from_bytes(digest[:8], "big") % 100
        if bucket < train_pct:
            labels[i] = "train"
        elif bucket < train_pct + val_pct:
            labels[i] = "val"
        else:
            labels[i] = "test"
    return labels.astype(np.str_)


@dataclass(frozen=True)
class LoadedTrainingData:
    """One concatenated (and optionally split-filtered) training view."""

    tensors: np.ndarray
    policy_target: np.ndarray
    value_target: np.ndarray
    sample_weight: np.ndarray
    legal_mask: np.ndarray
    source_tags: tuple[str, ...]


def load_training_data(
    paths: Sequence[str | Path],
    split: Split | None = None,
    *,
    train_pct: int = 80,
    val_pct: int = 10,
) -> LoadedTrainingData:
    """Load and concatenate `.npz` views, optionally filtering to a split."""
    if not paths:
        raise ValueError("at least one .npz path is required")
    views = [load_npz(path) for path in paths]
    tensors = np.concatenate([v.tensors for v in views])
    policy = np.concatenate([v.policy_target for v in views])
    value = np.concatenate([v.value_target for v in views])
    weight = np.concatenate([v.sample_weight for v in views])
    mask = expand_legal_mask(np.concatenate([v.legal_action_mask for v in views]))
    tags = tuple(
        _TAG_JOIN.join(row_tags) for v in views for row_tags in v.source_tags
    )
    if split is not None:
        labels = split_assignments(
            tensors, policy, tags, train_pct=train_pct, val_pct=val_pct
        )
        keep = labels == split
        tensors, policy, value = tensors[keep], policy[keep], value[keep]
        weight, mask = weight[keep], mask[keep]
        tags = tuple(t for t, k in zip(tags, keep) if k)
    return LoadedTrainingData(
        tensors=tensors,
        policy_target=policy,
        value_target=value,
        sample_weight=weight,
        legal_mask=mask,
        source_tags=tags,
    )
```

Also in `pyproject.toml`, extend the existing extra:

```toml
torch = [
  "torch>=2.4",
  "safetensors>=0.4",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_dataset.py -q`
Expected: 3 passed

- [ ] **Step 5: Run the full suite and mypy**

Run: `python -m pytest -q && python -m mypy src/quantik_models/data/dataset.py`
Expected: all pass, no mypy errors

- [ ] **Step 6: Commit**

```bash
git add src/quantik_models/data/dataset.py tests/test_dataset.py pyproject.toml
git commit -m "Add training dataset loader with deterministic content-hash sharding"
```

---

### Task 2: PolicyValueNet, presets, masked log-softmax

**Files:**
- Create: `src/quantik_models/model/__init__.py` (empty docstring module)
- Create: `src/quantik_models/model/policy_value_net.py`
- Test: `tests/test_policy_value_net.py`

**Interfaces:**
- Consumes: nothing from Task 1 (torch-only module).
- Produces:
  - `@dataclass(frozen=True) PolicyValueNetConfig(channels: int, blocks: int, value_hidden: int = 64)`
  - `PRESETS: dict[str, PolicyValueNetConfig]` with keys `"smoke"` (16, 2), `"small"` (64, 4), `"target"` (256, 13)
  - `class PolicyValueNet(torch.nn.Module)`: `__init__(config)`, `forward(x: Tensor(b,9,4,4)) -> tuple[Tensor(b,64), Tensor(b,)]` (logits, tanh value)
  - `masked_log_softmax(logits: Tensor(b,64), legal_mask: Tensor(b,64) bool) -> Tensor(b,64)`
  - `parameter_count(model) -> int`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_policy_value_net.py
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from quantik_models.model.policy_value_net import (  # noqa: E402
    PRESETS,
    PolicyValueNet,
    PolicyValueNetConfig,
    masked_log_softmax,
    parameter_count,
)


def test_forward_shapes_and_value_range() -> None:
    model = PolicyValueNet(PRESETS["smoke"])
    x = torch.rand(5, 9, 4, 4)
    logits, value = model(x)
    assert logits.shape == (5, 64)
    assert value.shape == (5,)
    assert value.abs().max().item() <= 1.0


def test_masked_log_softmax_zeroes_illegal() -> None:
    logits = torch.zeros(2, 64)
    mask = torch.zeros(2, 64, dtype=torch.bool)
    mask[0, :4] = True
    mask[1, 63] = True
    logp = masked_log_softmax(logits, mask)
    probs = logp.exp()
    assert torch.allclose(probs[0, :4], torch.full((4,), 0.25), atol=1e-6)
    assert probs[0, 4:].max().item() == pytest.approx(0.0, abs=1e-6)
    assert probs[1, 63].item() == pytest.approx(1.0, abs=1e-6)
    # masked entries must not produce NaN gradients
    loss = (probs[0, :4]).sum()
    loss.backward()


def test_preset_sizes() -> None:
    smoke = parameter_count(PolicyValueNet(PRESETS["smoke"]))
    small = parameter_count(PolicyValueNet(PRESETS["small"]))
    target = parameter_count(PolicyValueNet(PRESETS["target"]))
    assert smoke < 100_000
    assert small < 2_000_000
    # 4 bytes/param must land inside the 50-100 MB contract envelope
    assert 50 * 2**20 <= target * 4 <= 100 * 2**20


def test_config_round_trip() -> None:
    cfg = PolicyValueNetConfig(channels=16, blocks=2)
    model = PolicyValueNet(cfg)
    assert model.config == cfg
    assert math.isfinite(parameter_count(model))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_policy_value_net.py -q`
Expected: ERROR `ModuleNotFoundError: No module named 'quantik_models.model'`

- [ ] **Step 3: Implement `src/quantik_models/model/policy_value_net.py`**

```python
"""AlphaZero-style policy/value network for Quantik.

Torch-only module: import it behind the `[torch]` extra. Legality
masking is deliberately NOT part of the module — `masked_log_softmax`
is the single shared implementation used by both the training loss and
engine adapters, per the model-checkpoint.v1 note that runtimes must
apply legal action masks outside the model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class PolicyValueNetConfig:
    channels: int
    blocks: int
    value_hidden: int = 64


PRESETS: dict[str, PolicyValueNetConfig] = {
    # CI-fast preset for smoke tests and examples (<1 MB).
    "smoke": PolicyValueNetConfig(channels=16, blocks=2),
    # Laptop baseline (single-digit MB).
    "small": PolicyValueNetConfig(channels=64, blocks=4),
    # Sized so float32 safetensors lands in the 50-100 MB contract
    # envelope (~15.4M parameters ~= 61 MB).
    "target": PolicyValueNetConfig(channels=256, blocks=13),
}


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + x)


class PolicyValueNet(nn.Module):
    """Shared trunk with a 64-logit policy head and a tanh value head."""

    def __init__(self, config: PolicyValueNetConfig) -> None:
        super().__init__()
        self.config = config
        c = config.channels
        self.stem = nn.Sequential(
            nn.Conv2d(9, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(),
        )
        self.trunk = nn.Sequential(*[_ResidualBlock(c) for _ in range(config.blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(c, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 16, 64),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(c, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16, config.value_hidden),
            nn.ReLU(),
            nn.Linear(config.value_hidden, 1),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        trunk = self.trunk(self.stem(x))
        return self.policy_head(trunk), self.value_head(trunk).squeeze(-1)


def masked_log_softmax(logits: Tensor, legal_mask: Tensor) -> Tensor:
    """Log-softmax with illegal logits filled with the dtype's most negative finite value (near-zero probability)."""
    masked = logits.masked_fill(~legal_mask, torch.finfo(logits.dtype).min)
    return torch.log_softmax(masked, dim=-1)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
```

`src/quantik_models/model/__init__.py`:

```python
"""Model architectures (torch extra required)."""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_policy_value_net.py -q`
Expected: 4 passed (or skipped if torch missing — must pass in the venv which has torch)

- [ ] **Step 5: Full suite + mypy, commit**

Run: `python -m pytest -q && python -m mypy src/quantik_models/model/policy_value_net.py`

```bash
git add src/quantik_models/model tests/test_policy_value_net.py
git commit -m "Add scalable policy/value residual network with legality-masked log-softmax"
```

---

### Task 3: Checkpoint export (safetensors + model-checkpoint.v1 manifest)

**Files:**
- Create: `src/quantik_models/export/__init__.py`
- Create: `src/quantik_models/export/checkpoint.py`
- Test: `tests/test_export_checkpoint.py`

**Interfaces:**
- Consumes: `PolicyValueNet`, `PolicyValueNetConfig`, `parameter_count` from Task 2.
- Produces:
  - `export_checkpoint(model: PolicyValueNet, *, out_dir: Path, model_id: str, training_report: dict, contract_version: str = "1.1.0") -> Path` — writes `weights.safetensors`, `training-report.json`, `manifest.json`; returns manifest path.
  - Manifest fields (exact, validated by `quantik_core.artifact_data.load_model_checkpoint_manifest`): `schema="model-checkpoint.v1"`, `contract_version`, `model_id`, `model_family="quantik-policy-value-resnet"`, `created_at` (UTC ISO 8601), `input_contracts=["tensor-board.v1", "bitboard.v1", "action-index.v1"]`, `output_contract="policy-logits-64+value-tanh"`, `weights_format="safetensors"`, `weights_hash="sha256:<hex>"`, `size_bytes=<weights file size>`, `training_data_manifest="training-report.json"`, `calibration_report="training-report.json"`, `parameter_count`, `architecture="resnet-c{channels}-b{blocks}"`, `legal_action_mask_required=true`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_export_checkpoint.py
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from quantik_models.export.checkpoint import export_checkpoint  # noqa: E402
from quantik_models.model.policy_value_net import (  # noqa: E402
    PRESETS,
    PolicyValueNet,
)


def _export(tmp_path: Path) -> Path:
    model = PolicyValueNet(PRESETS["smoke"])
    return export_checkpoint(
        model,
        out_dir=tmp_path,
        model_id="quantik-pv-test",
        training_report={"final": {"policy_loss": 1.0}},
    )


def test_export_writes_weights_manifest_report(tmp_path: Path) -> None:
    manifest_path = _export(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    weights = tmp_path / "weights.safetensors"
    assert weights.is_file()
    assert manifest["schema"] == "model-checkpoint.v1"
    assert manifest["weights_format"] == "safetensors"
    assert manifest["size_bytes"] == weights.stat().st_size
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    assert manifest["weights_hash"] == f"sha256:{digest}"
    assert manifest["legal_action_mask_required"] is True
    assert manifest["architecture"] == "resnet-c16-b2"
    assert (tmp_path / "training-report.json").is_file()


def test_weights_round_trip(tmp_path: Path) -> None:
    from safetensors.torch import load_file

    model = PolicyValueNet(PRESETS["smoke"])
    export_checkpoint(
        model,
        out_dir=tmp_path,
        model_id="quantik-pv-test",
        training_report={},
    )
    restored = PolicyValueNet(PRESETS["smoke"])
    restored.load_state_dict(load_file(tmp_path / "weights.safetensors"))
    x = torch.rand(2, 9, 4, 4)
    model.eval()
    restored.eval()
    with torch.no_grad():
        a, av = model(x)
        b, bv = restored(x)
    assert torch.allclose(a, b) and torch.allclose(av, bv)


def test_manifest_validates_through_core_py(tmp_path: Path) -> None:
    artifact_data = pytest.importorskip("quantik_core.artifact_data")
    manifest_path = _export(tmp_path)
    parsed = artifact_data.load_model_checkpoint_manifest(manifest_path)
    assert parsed.weights_format == "safetensors"
    assert parsed.parameter_count is not None and parsed.parameter_count > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$QUANTIK_NS/quantik-core-py/src python -m pytest tests/test_export_checkpoint.py -q`
Expected: ERROR `ModuleNotFoundError: No module named 'quantik_models.export'`

- [ ] **Step 3: Implement `src/quantik_models/export/checkpoint.py`**

```python
"""Checkpoint export: safetensors weights + model-checkpoint.v1 manifest.

The manifest is the contract handshake with the core libraries: it is
validated in tests through quantik-core-py's
`load_model_checkpoint_manifest`, and weights stay detached from core
per the policy/value model project doc.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from safetensors.torch import save_file

from ..model.policy_value_net import PolicyValueNet, parameter_count

_WEIGHTS_NAME = "weights.safetensors"
_MANIFEST_NAME = "manifest.json"
_REPORT_NAME = "training-report.json"


def export_checkpoint(
    model: PolicyValueNet,
    *,
    out_dir: Path,
    model_id: str,
    training_report: dict[str, Any],
    contract_version: str = "1.1.0",
) -> Path:
    """Write weights, training report, and manifest; return manifest path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_path = out_dir / _WEIGHTS_NAME
    save_file(model.state_dict(), str(weights_path))
    weights_bytes = weights_path.read_bytes()

    report_path = out_dir / _REPORT_NAME
    report_path.write_text(json.dumps(training_report, indent=2, sort_keys=True))

    config = model.config
    manifest = {
        "schema": "model-checkpoint.v1",
        "contract_version": contract_version,
        "model_id": model_id,
        "model_family": "quantik-policy-value-resnet",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_contracts": ["tensor-board.v1", "bitboard.v1", "action-index.v1"],
        "output_contract": "policy-logits-64+value-tanh",
        "weights_format": "safetensors",
        "weights_hash": f"sha256:{hashlib.sha256(weights_bytes).hexdigest()}",
        "size_bytes": weights_path.stat().st_size,
        "training_data_manifest": _REPORT_NAME,
        "calibration_report": _REPORT_NAME,
        "parameter_count": parameter_count(model),
        "architecture": f"resnet-c{config.channels}-b{config.blocks}",
        "legal_action_mask_required": True,
    }
    manifest_path = out_dir / _MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest_path
```

`src/quantik_models/export/__init__.py`:

```python
"""Checkpoint export (torch extra required)."""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$QUANTIK_NS/quantik-core-py/src python -m pytest tests/test_export_checkpoint.py -q`
Expected: 3 passed

- [ ] **Step 5: Full suite + mypy, commit**

Run: `python -m pytest -q && python -m mypy src/quantik_models/export/checkpoint.py`

```bash
git add src/quantik_models/export tests/test_export_checkpoint.py
git commit -m "Add safetensors checkpoint export with model-checkpoint.v1 manifest"
```

---

### Task 4: Trainer and `quantik-models-train` CLI

**Files:**
- Create: `src/quantik_models/train/__init__.py`
- Create: `src/quantik_models/train/trainer.py`
- Modify: `pyproject.toml` (add console script)
- Test: `tests/test_trainer.py`

**Interfaces:**
- Consumes: `load_training_data`, `LoadedTrainingData` (Task 1); `PolicyValueNet`, `PRESETS`, `PolicyValueNetConfig`, `masked_log_softmax`, `parameter_count` (Task 2); `export_checkpoint` (Task 3).
- Produces:
  - `@dataclass TrainConfig(npz_paths: list[Path], preset: str = "smoke", channels: int | None = None, blocks: int | None = None, epochs: int = 2, batch_size: int = 64, lr: float = 1e-3, value_loss_weight: float = 1.0, weight_decay: float = 1e-4, seed: int = 20260715, device: str = "auto", out_dir: Path = Path("outputs/checkpoint"), model_id: str | None = None)`
  - `train(config: TrainConfig) -> dict` — runs training, exports checkpoint, returns the training report dict (also written to disk by export).
  - Report dict shape: `{"config": {...}, "dataset": {"train_rows": int, "val_rows": int, "sources": [...]}, "epochs": [{"epoch": int, "train_policy_loss": float, "train_value_mse": float, "val_policy_loss": float, "val_value_mse": float, "val_top1_agreement": float, "val_illegal_mass_premask": float}], "final": {<last epochs entry>}, "elapsed_seconds": float}`
  - `main(argv: list[str] | None = None) -> int` — argparse CLI mapping flags `--npz` (repeatable, required), `--preset`, `--channels`, `--blocks`, `--epochs`, `--batch-size`, `--lr`, `--value-loss-weight`, `--weight-decay`, `--seed`, `--device`, `--out-dir`, `--model-id` onto `TrainConfig`.
  - Console script in pyproject: `quantik-models-train = "quantik_models.train.trainer:main"`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_trainer.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from quantik_models.train.trainer import TrainConfig, main, train  # noqa: E402


def _write_learnable_view(path: Path, n: int = 256) -> None:
    """Synthetic but learnable data: policy always action 0, value = +1
    when plane-0 mean exceeds the median (so loss can decrease)."""
    rng = np.random.default_rng(7)
    tensors = rng.random((n, 9, 4, 4), dtype=np.float32)
    policy = np.zeros((n, 64), dtype=np.float32)
    policy[:, 0] = 1.0
    signal = tensors[:, 0].mean(axis=(1, 2))
    value = np.where(signal > np.median(signal), 1.0, -1.0).astype(np.float32)
    np.savez_compressed(
        path,
        tensors=tensors,
        policy_target=policy,
        value_target=value,
        sample_weight=np.ones(n, dtype=np.float32),
        legal_action_mask=np.full(n, np.uint64(0xFFFF), dtype=np.uint64),
        side_to_move=np.zeros(n, dtype=np.uint8),
        source_tags=np.asarray(["synthetic"] * n, dtype=np.str_),
    )


def test_train_smoke_loss_decreases_and_exports(tmp_path: Path) -> None:
    npz = tmp_path / "view.npz"
    _write_learnable_view(npz)
    config = TrainConfig(
        npz_paths=[npz],
        preset="smoke",
        epochs=3,
        batch_size=32,
        seed=1,
        device="cpu",
        out_dir=tmp_path / "ckpt",
    )
    report = train(config)
    epochs = report["epochs"]
    assert len(epochs) == 3
    assert epochs[-1]["train_policy_loss"] < epochs[0]["train_policy_loss"]
    assert (tmp_path / "ckpt" / "manifest.json").is_file()
    assert (tmp_path / "ckpt" / "weights.safetensors").is_file()
    saved = json.loads((tmp_path / "ckpt" / "training-report.json").read_text())
    assert saved["final"] == epochs[-1]
    # legality masking: probability mass on illegal actions is ~0 post-mask
    assert epochs[-1]["val_illegal_mass_premask"] >= 0.0


def test_train_is_seeded_deterministic(tmp_path: Path) -> None:
    npz = tmp_path / "view.npz"
    _write_learnable_view(npz)

    def run(out: Path) -> float:
        cfg = TrainConfig(
            npz_paths=[npz], preset="smoke", epochs=1, batch_size=32,
            seed=42, device="cpu", out_dir=out,
        )
        return float(train(cfg)["final"]["train_policy_loss"])

    assert run(tmp_path / "a") == pytest.approx(run(tmp_path / "b"))


def test_cli_maps_flags(tmp_path: Path) -> None:
    npz = tmp_path / "view.npz"
    _write_learnable_view(npz, n=128)
    rc = main(
        [
            "--npz", str(npz),
            "--preset", "smoke",
            "--epochs", "1",
            "--batch-size", "32",
            "--seed", "3",
            "--device", "cpu",
            "--out-dir", str(tmp_path / "cli-ckpt"),
        ]
    )
    assert rc == 0
    assert (tmp_path / "cli-ckpt" / "manifest.json").is_file()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_trainer.py -q`
Expected: ERROR `ModuleNotFoundError: No module named 'quantik_models.train'`

- [ ] **Step 3: Implement `src/quantik_models/train/trainer.py`**

```python
"""Training loop and CLI for the Quantik policy/value network.

Loss = sample-weighted soft-target cross-entropy over legality-masked
log-probabilities, plus MSE on the tanh value, optimized with AdamW.
CPU-first: `--device auto` prefers cuda > mps > cpu but never requires
an accelerator.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from ..data.dataset import LoadedTrainingData, load_training_data
from ..export.checkpoint import export_checkpoint
from ..model.policy_value_net import (
    PRESETS,
    PolicyValueNet,
    PolicyValueNetConfig,
    masked_log_softmax,
)


@dataclass
class TrainConfig:
    npz_paths: list[Path]
    preset: str = "smoke"
    channels: int | None = None
    blocks: int | None = None
    epochs: int = 2
    batch_size: int = 64
    lr: float = 1e-3
    value_loss_weight: float = 1.0
    weight_decay: float = 1e-4
    seed: int = 20260715
    device: str = "auto"
    out_dir: Path = field(default_factory=lambda: Path("outputs/checkpoint"))
    model_id: str | None = None

    def net_config(self) -> PolicyValueNetConfig:
        if self.channels is not None or self.blocks is not None:
            if self.channels is None or self.blocks is None:
                raise ValueError("--channels and --blocks must be given together")
            return PolicyValueNetConfig(channels=self.channels, blocks=self.blocks)
        if self.preset not in PRESETS:
            raise ValueError(f"unknown preset {self.preset!r}")
        return PRESETS[self.preset]


def _resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _to_tensors(
    data: LoadedTrainingData, device: torch.device
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    return (
        torch.from_numpy(data.tensors).to(device),
        torch.from_numpy(data.policy_target).to(device),
        torch.from_numpy(data.value_target).to(device),
        torch.from_numpy(data.sample_weight).to(device),
        torch.from_numpy(data.legal_mask).to(device),
    )


def _losses(
    model: PolicyValueNet,
    x: Tensor,
    policy_target: Tensor,
    value_target: Tensor,
    weight: Tensor,
    legal_mask: Tensor,
    value_loss_weight: float,
) -> tuple[Tensor, Tensor, Tensor]:
    logits, value = model(x)
    logp = masked_log_softmax(logits, legal_mask)
    # Soft-target CE; rows can carry zero policy mass on illegal actions
    # by construction of the training view.
    policy_loss = -(policy_target * logp).sum(dim=-1)
    value_mse = (value - value_target) ** 2
    norm = weight.sum().clamp_min(1e-8)
    policy_loss = (policy_loss * weight).sum() / norm
    value_loss = (value_mse * weight).sum() / norm
    return policy_loss + value_loss_weight * value_loss, policy_loss, value_loss


@torch.no_grad()
def _validate(
    model: PolicyValueNet,
    x: Tensor,
    policy_target: Tensor,
    value_target: Tensor,
    weight: Tensor,
    legal_mask: Tensor,
    value_loss_weight: float,
) -> dict[str, float]:
    model.eval()
    _, policy_loss, value_loss = _losses(
        model, x, policy_target, value_target, weight, legal_mask, value_loss_weight
    )
    logits, _ = model(x)
    top1 = (
        (logits.argmax(dim=-1) == policy_target.argmax(dim=-1)).float().mean().item()
    )
    premask_probs = torch.softmax(logits, dim=-1)
    illegal_mass = premask_probs.masked_fill(legal_mask, 0.0).sum(dim=-1).mean().item()
    model.train()
    return {
        "val_policy_loss": float(policy_loss.item()),
        "val_value_mse": float(value_loss.item()),
        "val_top1_agreement": float(top1),
        "val_illegal_mass_premask": float(illegal_mass),
    }


def train(config: TrainConfig) -> dict[str, Any]:
    started = time.monotonic()
    _seed_everything(config.seed)
    device = _resolve_device(config.device)

    train_data = load_training_data(config.npz_paths, split="train")
    val_data = load_training_data(config.npz_paths, split="val")
    if train_data.tensors.shape[0] == 0:
        raise ValueError("training split is empty")
    # Tiny corpora can shard into an empty val split; fall back to train.
    if val_data.tensors.shape[0] == 0:
        val_data = train_data

    model = PolicyValueNet(config.net_config()).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    x, pt, vt, w, m = _to_tensors(train_data, device)
    vx, vpt, vvt, vw, vm = _to_tensors(val_data, device)

    epochs_report: list[dict[str, float]] = []
    n = x.shape[0]
    generator = torch.Generator().manual_seed(config.seed)
    for epoch in range(config.epochs):
        perm = torch.randperm(n, generator=generator)
        policy_sum = value_sum = 0.0
        batches = 0
        for start in range(0, n, config.batch_size):
            idx = perm[start : start + config.batch_size]
            optimizer.zero_grad()
            total, policy_loss, value_loss = _losses(
                model, x[idx], pt[idx], vt[idx], w[idx], m[idx],
                config.value_loss_weight,
            )
            total.backward()
            optimizer.step()
            policy_sum += float(policy_loss.item())
            value_sum += float(value_loss.item())
            batches += 1
        entry: dict[str, float] = {
            "epoch": float(epoch),
            "train_policy_loss": policy_sum / max(batches, 1),
            "train_value_mse": value_sum / max(batches, 1),
        }
        entry.update(
            _validate(model, vx, vpt, vvt, vw, vm, config.value_loss_weight)
        )
        epochs_report.append(entry)

    report: dict[str, Any] = {
        "config": {
            "preset": config.preset,
            "channels": model.config.channels,
            "blocks": model.config.blocks,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "lr": config.lr,
            "value_loss_weight": config.value_loss_weight,
            "weight_decay": config.weight_decay,
            "seed": config.seed,
            "device": str(device),
        },
        "dataset": {
            "train_rows": int(train_data.tensors.shape[0]),
            "val_rows": int(val_data.tensors.shape[0]),
            "sources": [str(p) for p in config.npz_paths],
        },
        "epochs": epochs_report,
        "final": epochs_report[-1],
        "elapsed_seconds": time.monotonic() - started,
    }

    model_id = config.model_id or (
        f"quantik-pv-{config.preset}-seed{config.seed}"
    )
    export_checkpoint(
        model.cpu(),
        out_dir=Path(config.out_dir),
        model_id=model_id,
        training_report=report,
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train a Quantik policy/value model from .npz training views."
    )
    parser.add_argument("--npz", action="append", required=True, type=Path)
    parser.add_argument("--preset", default="smoke", choices=sorted(PRESETS))
    parser.add_argument("--channels", type=int, default=None)
    parser.add_argument("--blocks", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/checkpoint"))
    parser.add_argument("--model-id", default=None)
    args = parser.parse_args(argv)

    config = TrainConfig(
        npz_paths=list(args.npz),
        preset=args.preset,
        channels=args.channels,
        blocks=args.blocks,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        value_loss_weight=args.value_loss_weight,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        out_dir=args.out_dir,
        model_id=args.model_id,
    )
    report = train(config)
    final = report["final"]
    print(
        "trained "
        f"preset={args.preset} epochs={args.epochs} "
        f"train_policy_loss={final['train_policy_loss']:.4f} "
        f"val_top1={final['val_top1_agreement']:.3f} "
        f"-> {config.out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

`src/quantik_models/train/__init__.py`:

```python
"""Training loop and CLI (torch extra required)."""
```

pyproject console script section becomes:

```toml
[project.scripts]
quantik-models-materialize = "quantik_models.data.materialize:main"
quantik-models-train = "quantik_models.train.trainer:main"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_trainer.py -q`
Expected: 3 passed

- [ ] **Step 5: Full suite + mypy, commit**

Run: `python -m pytest -q && python -m mypy src/quantik_models/train/trainer.py`

```bash
git add src/quantik_models/train tests/test_trainer.py pyproject.toml
git commit -m "Add policy/value trainer with quantik-models-train CLI"
```

---

### Task 5: Example scripts

**Files:**
- Create: `examples/train_smoke.sh` (chmod +x)
- Create: `examples/train_small_local.sh` (chmod +x)
- Create: `examples/inspect_checkpoint.py`

**Interfaces:**
- Consumes: `quantik-models-train` CLI (Task 4), `scripts/run_smoke_pipeline.sh` (existing), manifest/weights layout (Task 3), `PolicyValueNetConfig`/`PolicyValueNet`/`masked_log_softmax` (Task 2).
- Produces: runnable demos referenced by README and the scaling guide.

- [ ] **Step 1: Write `examples/train_smoke.sh`**

```bash
#!/usr/bin/env bash
# End-to-end smoke demo: generate a tiny data corpus with the existing
# pipeline, train the `smoke` preset for a few epochs, export a
# model-checkpoint.v1 checkpoint, and inspect it.
#
# Prereqs:
#   - sibling checkouts per README (quantik-core-contracts/-rust/-py)
#   - pip install -e ".[dev,torch]"
#
# Usage:
#   examples/train_smoke.sh                 # generates tiny data first
#   OUT=/path/to/existing-data examples/train_smoke.sh   # reuse data
set -euo pipefail

MODELS="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-$MODELS/outputs/smoke-demo-data}"
CKPT="${CKPT:-$MODELS/outputs/smoke-demo-checkpoint}"

if [[ ! -f "$OUT/training-view-observations.npz" ]]; then
  echo "== generating tiny pipeline data into $OUT =="
  OUT="$OUT" RUST_PROFILE=debug BOOK_DEPTH=1 POSITIONS_USE_BOOK=1 \
  OPENING_BOOK_EXTRA_ARGS='--max-positions 16 --batch-size 16 --quiet' \
  OPENING_POSITIONS=1 EARLY_MID_POSITIONS=1 LATE_MID_POSITIONS=1 \
  ENDGAME_POSITIONS=1 SOLVE_BUDGET=0.05 ENGINES=random,minimax \
  OBSERVATION_SEEDS=1 MINIMAX_DEPTH=1 MINIMAX_TIME=0.01 \
  MCTS_ITERATIONS=8 MCTS_DEPTH=4 BEAM_WIDTH=4 BEAM_DEPTH=4 WORKERS=1 \
  H2H_ENGINES=random,minimax H2H_POSITIONS=1 H2H_SEEDS=1 \
  H2H_OBSERVATION_SEEDS=1 SELFPLAY_GAMES=1 SELFPLAY_ITERATIONS=8 \
  SELFPLAY_SEED=20260713 "$MODELS/scripts/run_smoke_pipeline.sh"
fi

echo "== training smoke preset =="
quantik-models-train \
  --npz "$OUT/training-view-observations.npz" \
  --npz "$OUT/training-view-selfplay.npz" \
  --preset smoke \
  --epochs 5 \
  --batch-size 16 \
  --seed 20260715 \
  --device auto \
  --out-dir "$CKPT"

echo "== inspecting checkpoint =="
python "$MODELS/examples/inspect_checkpoint.py" "$CKPT" \
  --npz "$OUT/training-view-selfplay.npz"

echo "checkpoint: $CKPT"
```

- [ ] **Step 2: Write `examples/inspect_checkpoint.py`**

```python
#!/usr/bin/env python3
"""Inspect an exported Quantik policy/value checkpoint.

Loads manifest.json + weights.safetensors, prints the architecture and
training metrics, then runs a forward pass on the empty board and (when
--npz is given) on the first dataset row, decoding the top-5 policy
actions as shape:position pairs. Demonstrates that legality masking is
applied OUTSIDE the model, as required by model-checkpoint.v1.

Usage:
    python examples/inspect_checkpoint.py OUT_DIR [--npz view.npz]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from quantik_models.data.dataset import load_training_data
from quantik_models.model.policy_value_net import (
    PolicyValueNet,
    PolicyValueNetConfig,
    masked_log_softmax,
)


def _decode_action(index: int) -> str:
    shape, position = divmod(index, 16)
    row, col = divmod(position, 4)
    return f"shape={shape} pos={position} (r{row}c{col})"


def _model_from_manifest(out_dir: Path) -> tuple[PolicyValueNet, dict]:
    manifest = json.loads((out_dir / "manifest.json").read_text())
    match = re.fullmatch(r"resnet-c(\d+)-b(\d+)", manifest["architecture"])
    if match is None:
        raise SystemExit(f"unsupported architecture: {manifest['architecture']}")
    config = PolicyValueNetConfig(
        channels=int(match.group(1)), blocks=int(match.group(2))
    )
    model = PolicyValueNet(config)
    model.load_state_dict(load_file(out_dir / "weights.safetensors"))
    model.eval()
    return model, manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--npz", type=Path, default=None)
    args = parser.parse_args()

    model, manifest = _model_from_manifest(args.out_dir)
    print(f"model_id      : {manifest['model_id']}")
    print(f"architecture  : {manifest['architecture']}")
    print(f"parameters    : {manifest['parameter_count']:,}")
    print(f"weights       : {manifest['size_bytes']:,} bytes "
          f"({manifest['weights_hash'][:19]}...)")
    report = json.loads((args.out_dir / "training-report.json").read_text())
    if report.get("final"):
        final = report["final"]
        print(f"final metrics : {json.dumps(final, sort_keys=True)}")

    # Empty board: all 9 planes zero except side-to-move plane = 0 too
    # (player 0 to move); every action is legal.
    empty = torch.zeros(1, 9, 4, 4)
    all_legal = torch.ones(1, 64, dtype=torch.bool)
    with torch.no_grad():
        logits, value = model(empty)
        probs = masked_log_softmax(logits, all_legal).exp()[0]
    print("\nempty board:")
    print(f"  value estimate: {value.item():+.3f}")
    for rank, index in enumerate(probs.topk(5).indices.tolist(), start=1):
        print(f"  top{rank}: p={probs[index]:.3f} {_decode_action(index)}")

    if args.npz is not None:
        data = load_training_data([args.npz])
        x = torch.from_numpy(data.tensors[:1])
        mask = torch.from_numpy(data.legal_mask[:1])
        with torch.no_grad():
            logits, value = model(x)
            probs = masked_log_softmax(logits, mask).exp()[0]
        illegal_mass = probs[~mask[0]].sum().item()
        print(f"\nfirst dataset row ({data.source_tags[0]}):")
        print(f"  value estimate: {value.item():+.3f}")
        print(f"  legal actions : {int(mask.sum())}/64")
        print(f"  illegal mass after masking: {illegal_mass:.2e}")
        for rank, index in enumerate(probs.topk(5).indices.tolist(), start=1):
            legal = "legal" if mask[0, index] else "ILLEGAL"
            print(f"  top{rank}: p={probs[index]:.3f} {_decode_action(index)} [{legal}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Write `examples/train_small_local.sh`**

```bash
#!/usr/bin/env bash
# Train the `small` preset (64 channels x 4 blocks, single-digit MB) on
# a locally generated corpus. Meant for a laptop: a few minutes on
# Apple Silicon (MPS) or modern x86 CPU for the default settings.
#
# The smoke corpus is far too small for this preset to be meaningful;
# generate a bigger dataset first (see docs/scaling-guide.md), e.g.:
#
#   OUT=$PWD/outputs/small-corpus \
#   OPENING_POSITIONS=64 EARLY_MID_POSITIONS=64 LATE_MID_POSITIONS=96 \
#   ENDGAME_POSITIONS=64 MCTS_ITERATIONS=512 SELFPLAY_GAMES=32 \
#   SELFPLAY_ITERATIONS=256 scripts/run_smoke_pipeline.sh
#
# Usage:
#   DATA=$PWD/outputs/small-corpus examples/train_small_local.sh
set -euo pipefail

MODELS="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${DATA:?set DATA to a directory containing training-view-*.npz}"
CKPT="${CKPT:-$MODELS/outputs/small-checkpoint}"

quantik-models-train \
  --npz "$DATA/training-view-observations.npz" \
  --npz "$DATA/training-view-selfplay.npz" \
  --preset small \
  --epochs 20 \
  --batch-size 128 \
  --lr 5e-4 \
  --seed 20260715 \
  --device auto \
  --out-dir "$CKPT"

python "$MODELS/examples/inspect_checkpoint.py" "$CKPT" \
  --npz "$DATA/training-view-selfplay.npz"
```

- [ ] **Step 4: Verify the smoke example end-to-end**

Run (venv with torch+safetensors active, sibling repos present):
```bash
chmod +x examples/train_smoke.sh examples/train_small_local.sh
PYTHONPATH=$QUANTIK_NS/quantik-core-py/src:src \
OUT="${TMPDIR:-/tmp}/quantik-e2e-book" \
  bash examples/train_smoke.sh
```
Expected: trains 5 epochs, prints checkpoint summary including top-5 actions and `illegal mass after masking: ~e-08` or smaller, exits 0. (Uses the existing generated data at that OUT; regenerates if absent.)

- [ ] **Step 5: Commit**

```bash
git add examples/
git commit -m "Add runnable training and checkpoint-inspection examples"
```

---

### Task 6: Scaling guide + docs updates

**Files:**
- Create: `docs/scaling-guide.md`
- Modify: `README.md` (add Training section after the materializer section)
- Modify: `docs/pipeline.md` (extend flow past materialization)
- Modify: `docs/model-report.md` (mark trainer/export slices implemented)

**Interfaces:**
- Consumes: presets and CLI from Tasks 2/4, examples from Task 5.
- Produces: documentation only.

- [ ] **Step 1: Write `docs/scaling-guide.md`**

```markdown
# Scaling From Smoke To The Target Model

The trainer exposes one architecture (`PolicyValueNet`) at three named
presets. Scaling is a one-flag change; what actually needs to grow with
the model is the data.

## Presets

| Preset | Channels | Blocks | Parameters | float32 safetensors | Intended use |
| --- | --- | --- | --- | --- | --- |
| `smoke` | 16 | 2 | ~24k | ~0.1 MB | CI, examples, plumbing checks |
| `small` | 64 | 4 | ~330k | ~1.3 MB | Laptop baselines, ablations |
| `target` | 256 | 13 | ~15.4M | ~61 MB | The 50-100 MB contract model |

(Exact parameter counts are asserted by `tests/test_policy_value_net.py`;
the `target` preset must land inside the 50-100 MB envelope from
quantik-core-contracts `docs/policy-value-model-project.md`.)

## What changes when you scale

Only the preset flag:

    quantik-models-train --npz ... --preset small
    quantik-models-train --npz ... --preset target

Everything else (loss, masking, sharding, export, manifest) is
identical at every scale. `--channels/--blocks` override the presets
for ablations.

## What must grow: data

| Preset | Minimum sensible corpus |
| --- | --- |
| `smoke` | the tiny pipeline corpus (tens of rows) — plumbing only |
| `small` | >= 100k rows: full observation runs across MCTS/minimax/beam plus self-play (see the data milestones in the contracts model project doc) |
| `target` | the full depth-6 book corpus: book-backed positions, multi-engine observations, large self-play, autoplay rounds |

Quantik's reachable state space is small compared to `target`'s 15.4M
parameters. Training `target` on a small corpus will memorize it
perfectly and generalize poorly; the paper
(`docs/policy-value-training-paper.md`) discusses why the envelope is
still useful (headroom for regularization, distillation targets, and
future feature channels) and when a smaller deployed model is the
better trade.

## Wall-clock expectations

Measured on the smoke corpus (tens of rows), batch 16, per epoch:
`smoke` sub-second on CPU; `small` ~1-2 s CPU, sub-second MPS;
`target` minutes on CPU — use MPS/CUDA (`--device auto` picks them up)
and a real corpus.

## Verifying the envelope

    ls -l <out-dir>/weights.safetensors     # 50-100 MB for target
    python - <<'PY'
    import json; m = json.load(open("<out-dir>/manifest.json"))
    print(m["size_bytes"] / 2**20, "MiB", m["parameter_count"], "params")
    PY

The exported manifest must validate through quantik-core-py
(`load_model_checkpoint_manifest`) — the export test does this on every
CI run.

## Acceptance gates beyond size

A checkpoint is only useful if it clears the gates in the contracts
model project doc: manifest validation, fixed-budget search
improvement, book-frontier H2H, near-zero illegal mass after masking,
reproducible training data. Size is the cheapest gate; H2H is the one
that matters.
```

- [ ] **Step 2: Update `README.md`**

Add after the materializer section (adapt heading level to the file):

```markdown
## Training and checkpoint export

Install the training extra and train the smoke preset on materialized
views:

    pip install -e ".[dev,torch]"
    quantik-models-train \
      --npz outputs/smoke/training-view-observations.npz \
      --npz outputs/smoke/training-view-selfplay.npz \
      --preset smoke --epochs 5 --out-dir outputs/checkpoint

This exports `weights.safetensors`, `training-report.json`, and a
`model-checkpoint.v1` `manifest.json` (validated against
quantik-core-py in tests). See `examples/train_smoke.sh` for the full
end-to-end demo, `examples/inspect_checkpoint.py` to poke at a
checkpoint, and `docs/scaling-guide.md` for the smoke -> small ->
target path. The design/tradeoff discussion lives in
`docs/policy-value-training-paper.md`.
```

- [ ] **Step 3: Update `docs/pipeline.md` and `docs/model-report.md`**

`docs/pipeline.md`: extend the flow description so the chain ends with
`quantik-models-train` and checkpoint export instead of stopping at
materialization; mention the `train-smoke` workflow alongside the E2E
workflow paragraph.

`docs/model-report.md`: in "Open Implementation Slices", change the two
bullets "Add the first PyTorch training module and CLI." and "Add
checkpoint export and manifest generation." plus "Add deterministic
train/validation/test sharding." into a short "Implemented" list noting
`quantik-models-train`, content-hash sharding, and safetensors +
manifest export (keep the remaining open bullets).

- [ ] **Step 4: Commit**

```bash
git add docs/scaling-guide.md README.md docs/pipeline.md docs/model-report.md
git commit -m "Document training quickstart and smoke-to-target scaling"
```

---

### Task 7: train-smoke CI workflow

**Files:**
- Create: `.github/workflows/train-smoke.yml`

**Interfaces:**
- Consumes: everything above; mirrors the checkout pattern of `.github/workflows/e2e-data-pipeline.yml` (read it first and copy its checkout/env structure exactly, including the four-repo checkout and the tiny pipeline env block).

- [ ] **Step 1: Write the workflow**

Copy the checkout + Rust toolchain + tiny-pipeline env structure from
`e2e-data-pipeline.yml` (same env values, `POSITIONS_USE_BOOK: "1"`),
then replace the materialize/verify steps with:

```yaml
      - name: Install quantik-models with training extra
        run: |
          python -m pip install --upgrade pip
          pip install -e "quantik-models-py[dev,torch]" \
            --extra-index-url https://download.pytorch.org/whl/cpu
      - name: Run tiny pipeline
        run: quantik-models-py/scripts/run_smoke_pipeline.sh
      - name: Train smoke preset
        run: |
          quantik-models-train \
            --npz "$OUT/training-view-observations.npz" \
            --npz "$OUT/training-view-selfplay.npz" \
            --preset smoke --epochs 3 --batch-size 16 \
            --seed 20260715 --device cpu \
            --out-dir "$OUT/checkpoint"
        env:
          PYTHONPATH: ${{ github.workspace }}/quantik-core-py/src
      - name: Inspect checkpoint
        run: |
          python quantik-models-py/examples/inspect_checkpoint.py \
            "$OUT/checkpoint" --npz "$OUT/training-view-selfplay.npz"
        env:
          PYTHONPATH: ${{ github.workspace }}/quantik-core-py/src
      - name: Upload checkpoint artifact
        uses: actions/upload-artifact@v4
        with:
          name: smoke-checkpoint
          path: ${{ env.OUT }}/checkpoint
```

Trigger: `on: [push, pull_request, workflow_dispatch]` scoped with
`paths` covering `src/**`, `examples/**`, `.github/workflows/train-smoke.yml`,
`pyproject.toml` (models-py repo paths). Single ubuntu-latest job.

- [ ] **Step 2: Validate YAML locally**

Run: `python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/train-smoke.yml'))" || pip install pyyaml`
Expected: no exception.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/train-smoke.yml
git commit -m "Add train-smoke CI workflow exercising trainer and checkpoint export"
```

---

### Task 8 (main thread, not subagent): Companion paper

`docs/policy-value-training-paper.md` per the spec's paper outline, with
every arXiv ID verified by web search before inclusion. Written by the
orchestrator; committed separately. Not detailed here because it is
prose, not code, and reference verification requires web access.

---

## Verification (whole slice, before PR)

1. `python -m pytest -q` — everything green (torch tests run in the venv).
2. `python -m mypy src/quantik_models` — clean.
3. `PYTHONPATH=core-py-src:src bash examples/train_smoke.sh` against the scratchpad corpus — exits 0, illegal mass ~0.
4. `pip install -e .` in a fresh venv WITHOUT torch: `import quantik_models.data.dataset` works; `pytest -q` skips torch tests cleanly.
5. PR loop: push branch `policy-value-trainer`, PR, Copilot with 2/5/10/20 backoff, no co-author trailers, squash on green.
```
