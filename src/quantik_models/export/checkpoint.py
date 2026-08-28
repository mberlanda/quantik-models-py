"""Checkpoint export: safetensors weights, an ONNX graph, and a
`model-checkpoint.v1` manifest.

The manifest is the contract handshake with the core libraries: it is
validated in tests through quantik-core-py's
`load_model_checkpoint_manifest`, and weights stay detached from core per
the policy/value model project doc.

Two artifacts, deliberately:

* `weights.safetensors` is the primary. It is what the trainer produces,
  what `weights_hash` covers, and what a Python runtime loads into a model
  it already knows how to build.
* `model.onnx` carries the computation graph as well as the weights, so a
  runtime that has never seen this package can execute it. That is what
  makes a checkpoint consumable from Rust without reimplementing the
  architecture there.

Both are hashed. A manifest that named only the safetensors while a server
ran the ONNX would be describing something other than what it serves.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from torch import nn

from ..model.policy_value_net import parameter_count
from ..model.spec import BOARD_SIZE, INPUT_PLANES

_WEIGHTS_NAME = "weights.safetensors"
_ONNX_NAME = "model.onnx"
_MANIFEST_NAME = "manifest.json"
_REPORT_NAME = "training-report.json"

# The opset needs to cover the ops every registered architecture uses;
# 17 brings native LayerNorm, which the attention trunk needs and which
# older opsets decompose into a noisier subgraph.
_ONNX_OPSET = 17


def _supported_contract_version() -> str:
    """The contracts release this checkpoint claims compatibility with.

    Read from quantik-core-py rather than written down here. A literal
    default silently stamped every checkpoint `1.1.0` and kept doing so
    after contracts moved to 1.2.0, which made the exports unloadable by
    the very validator this manifest exists to satisfy.
    """
    try:
        from quantik_core.contracts import SUPPORTED_CONTRACTS_RELEASE
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "quantik-core is required to stamp a contracts release on a "
            "checkpoint manifest; install it (see the README) or pass an "
            "explicit contract_version="
        ) from exc
    return str(SUPPORTED_CONTRACTS_RELEASE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def export_onnx(model: nn.Module, path: Path) -> None:
    """Trace the model to ONNX with a dynamic batch dimension.

    Exported in eval mode so batch norm folds its running statistics rather
    than recomputing them per batch — a graph exported in train mode gives
    different answers for the same position depending on what else is in
    the batch, which would be a genuinely nasty bug to chase in a server.
    """
    was_training = model.training
    model.eval()
    example = torch.zeros(1, INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)
    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                example,
                str(path),
                input_names=["board"],
                output_names=["policy_logits", "value"],
                dynamic_axes={
                    "board": {0: "batch"},
                    "policy_logits": {0: "batch"},
                    "value": {0: "batch"},
                },
                opset_version=_ONNX_OPSET,
                # Keep the graph and its weights in one file. The exporter
                # otherwise spills tensors into a sibling `.onnx.data`, which
                # `onnx_hash` would not cover — exactly the drift this
                # manifest exists to prevent.
                external_data=False,
            )
    finally:
        if was_training:
            model.train()


def export_checkpoint(
    model: nn.Module,
    *,
    out_dir: Path,
    model_id: str,
    training_report: dict[str, Any],
    contract_version: str | None = None,
    with_onnx: bool = True,
) -> Path:
    """Write weights, ONNX graph, training report, and manifest.

    Returns the manifest path. `contract_version` defaults to the release
    quantik-core-py supports; `architecture` and `model_family` come from
    the model itself, so a new architecture records itself correctly
    without touching this function.
    """
    contract_version = contract_version or _supported_contract_version()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_path = out_dir / _WEIGHTS_NAME
    # safetensors serializes CPU tensors; the model may live on an accelerator.
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    save_file(state_dict, str(weights_path))

    report_path = out_dir / _REPORT_NAME
    report_path.write_text(json.dumps(training_report, indent=2, sort_keys=True))

    manifest = {
        "schema": "model-checkpoint.v1",
        "contract_version": contract_version,
        "model_id": model_id,
        "model_family": model.model_family,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_contracts": ["tensor-board.v1", "bitboard.v1", "action-index.v1"],
        "output_contract": "policy-logits-64+value-tanh",
        "weights_format": "safetensors",
        "weights_hash": _sha256(weights_path),
        "size_bytes": weights_path.stat().st_size,
        "training_data_manifest": _REPORT_NAME,
        "calibration_report": _REPORT_NAME,
        "parameter_count": parameter_count(model),
        "architecture": model.architecture,
        "legal_action_mask_required": True,
    }

    if with_onnx:
        # ONNX export needs the model on CPU: the exporter traces with a CPU
        # example input, and a model still on MPS or CUDA raises a device
        # mismatch rather than silently moving anything.
        device = next(model.parameters()).device
        model.to("cpu")
        try:
            onnx_path = out_dir / _ONNX_NAME
            export_onnx(model, onnx_path)
        finally:
            model.to(device)
        # `weights_format` stays "safetensors" because the contract admits a
        # single value and safetensors is what `weights_hash` covers. The
        # ONNX artifact is recorded beside it with its own hash so a runtime
        # can verify whichever one it actually loads.
        manifest["onnx_export"] = _ONNX_NAME
        manifest["onnx_hash"] = _sha256(onnx_path)
        manifest["onnx_opset"] = _ONNX_OPSET
        manifest["onnx_size_bytes"] = onnx_path.stat().st_size

    manifest_path = out_dir / _MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest_path
