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
from torch.export import Dim

from ..model import registry
from ..model.policy_value_net import parameter_count
from ..model.spec import BOARD_SIZE, INPUT_PLANES

_WEIGHTS_NAME = "weights.safetensors"
_ONNX_NAME = "model.onnx"
_MANIFEST_NAME = "manifest.json"
_REPORT_NAME = "training-report.json"

# The dynamo exporter's floor is 18; asking for anything lower makes it
# export at 18 and then attempt a down-conversion that fails silently for
# some graphs — leaving a file at 18 while the caller believes it got what
# it asked for. 18 also has native LayerNorm, which `cpool` uses and which
# older opsets decompose into a noisier subgraph.
_ONNX_OPSET = 18

# torch.export specializes any dimension of size 0 or 1, so a graph traced
# with a batch of one is frozen at one no matter what dynamic axes are
# declared — the declared symbolic dimension survives in the input
# signature while an internal Reshape hard-codes the batch. Trace with two.
_TRACE_BATCH = 2


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
    example = torch.zeros(_TRACE_BATCH, INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)
    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                (example,),
                str(path),
                input_names=["board"],
                output_names=["policy_logits", "value"],
                # `dynamic_shapes`, not `dynamic_axes`: the dynamo exporter
                # ignores the latter, and warns that it does. With
                # `dynamic_axes` the input signature said "batch" while the
                # graph body was specialized — a lie that only surfaced at
                # a batch size nobody had tested.
                dynamic_shapes=({0: Dim("batch")},),
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


def onnx_opset(path: Path) -> int:
    """The opset the file actually declares.

    Read back rather than assumed. `opset_version` is a request, and a
    failed down-conversion leaves a graph at a different version with no
    exception raised — so a manifest that recorded the request would be
    describing a file that does not exist.
    """
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            return int(opset.version)
    raise ValueError(f"{path} declares no default-domain opset")


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
        # `architecture` is the human-readable name; this is the machine
        # one. `resnet-c128-b6` and `mlp-h455-b4` do not share a grammar,
        # so a loader that parsed the string would be guessing at which
        # architecture it was even looking at.
        "architecture_spec": registry.spec_for(model),
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
        manifest["onnx_opset"] = onnx_opset(onnx_path)
        manifest["onnx_size_bytes"] = onnx_path.stat().st_size

    manifest_path = out_dir / _MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest_path
