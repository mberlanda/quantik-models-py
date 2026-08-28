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


def export_checkpoint(
    model: PolicyValueNet,
    *,
    out_dir: Path,
    model_id: str,
    training_report: dict[str, Any],
    contract_version: str | None = None,
) -> Path:
    """Write weights, training report, and manifest; return manifest path.

    `contract_version` defaults to the release quantik-core-py supports.
    """
    contract_version = contract_version or _supported_contract_version()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_path = out_dir / _WEIGHTS_NAME
    # safetensors serializes CPU tensors; the model may live on an accelerator.
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    save_file(state_dict, str(weights_path))
    weights_digest = hashlib.sha256()
    with weights_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            weights_digest.update(chunk)

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
        "weights_hash": f"sha256:{weights_digest.hexdigest()}",
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
