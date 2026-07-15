#!/usr/bin/env python3
"""Verify the small end-to-end Quantik data pipeline output directory."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REQUIRED_FILES = (
    "opening-book.sqlite",
    "positions-v1.json",
    "observations-v1.jsonl",
    "game-results-v1.jsonl",
    "selfplay-v1.jsonl",
    "training-view-observations.npz",
    "training-view-selfplay.npz",
    "h2h-report.md",
)


def _require_file(root: Path, name: str) -> Path:
    path = root / name
    if not path.is_file():
        raise SystemExit(f"missing expected pipeline output: {path}")
    if path.stat().st_size <= 0:
        raise SystemExit(f"empty pipeline output: {path}")
    return path


def _jsonl_count(path: Path, expected_schema: str) -> int:
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if record.get("schema") != expected_schema:
                raise SystemExit(
                    f"{path}:{line_number} schema={record.get('schema')!r}, "
                    f"expected {expected_schema!r}"
                )
            count += 1
    if count == 0:
        raise SystemExit(f"no rows found in {path}")
    return count


def _verify_npz(path: Path) -> int:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "tensors",
            "policy_target",
            "value_target",
            "sample_weight",
            "legal_action_mask",
            "side_to_move",
            "source_tags",
        }
        missing = required.difference(data.files)
        if missing:
            raise SystemExit(f"{path} missing arrays: {sorted(missing)}")
        tensors = data["tensors"]
        policy = data["policy_target"]
        values = data["value_target"]
        if tensors.ndim != 4 or tensors.shape[1:] != (9, 4, 4):
            raise SystemExit(f"{path} tensors shape must be (n, 9, 4, 4): {tensors.shape}")
        if policy.ndim != 2 or policy.shape[1] != 64:
            raise SystemExit(f"{path} policy_target shape must be (n, 64): {policy.shape}")
        if values.shape != (tensors.shape[0],):
            raise SystemExit(f"{path} value_target shape mismatch: {values.shape}")
        if tensors.shape[0] <= 0:
            raise SystemExit(f"{path} contains no rows")
        sums = policy.sum(axis=1)
        if not np.allclose(sums, 1.0):
            raise SystemExit(f"{path} policy rows must sum to 1.0: {sums}")
        return int(tensors.shape[0])


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise SystemExit("usage: verify_smoke_outputs.py PIPELINE_OUTPUT_DIR")
    root = Path(argv[1])
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    for name in REQUIRED_FILES:
        _require_file(root, name)

    observation_rows = _jsonl_count(root / "observations-v1.jsonl", "observation.v1")
    game_rows = _jsonl_count(root / "game-results-v1.jsonl", "game-result.v1")
    selfplay_rows = _jsonl_count(root / "selfplay-v1.jsonl", "selfplay.v1")
    observation_view_rows = _verify_npz(root / "training-view-observations.npz")
    selfplay_view_rows = _verify_npz(root / "training-view-selfplay.npz")

    print(
        "verified e2e pipeline outputs: "
        f"observations={observation_rows}, games={game_rows}, "
        f"selfplay={selfplay_rows}, observation_view={observation_view_rows}, "
        f"selfplay_view={selfplay_view_rows}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
