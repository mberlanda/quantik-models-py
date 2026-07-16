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
