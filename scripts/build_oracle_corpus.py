#!/usr/bin/env python
"""Sample positions, solve them exactly, and materialize a training corpus.

The probe (`runs/oracle/probe.jsonl`) is the held-out yardstick, so every
canonical position appearing in it is **excluded** from the corpus — otherwise
the reported accuracy would be a training-set score.

Each solved parent yields two kinds of row:

* a **policy row** — the parent, with probability spread uniformly over the
  outcome-optimal moves and the exact outcome as its value;
* one **value row per child** — free, because solving the parent means solving
  every child, and a child's exact value is the negation of its move's value.

Rows are deduplicated by canonical key, keeping the policy row where both
exist.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from quantik_models.env import fastboard as fb

DEFAULT_PLAN = {
    # ply: positions to solve. Weighted toward the opening, which is the only
    # place the incumbent minimax is beatable (it is exact from ply 8 on).
    12: 20_000,
    11: 20_000,
    10: 20_000,
    9: 30_000,
    8: 60_000,
    7: 60_000,
    6: 40_000,
    5: 15_000,
    4: 3_000,
    3: 300,
}


def sample_ply(
    ply: int, count: int, rng: np.random.Generator, exclude: set[int]
) -> np.ndarray:
    """Canonically distinct, non-terminal positions at `ply`, minus `exclude`."""
    collected: list[np.ndarray] = []
    seen: set[int] = set()
    attempts = 0
    while len(collected) < count and attempts < 60:
        attempts += 1
        batch = fb.empty_boards(max(4 * count, 4096))
        for _ in range(ply):
            done, _ = fb.terminal_status(batch)
            batch = batch[~done]
            if batch.shape[0] == 0:
                break
            legal = fb.legal_masks(batch)
            batch = fb.apply_actions(batch, (rng.random(legal.shape) * legal).argmax(axis=1))
        if batch.shape[0] == 0:
            continue
        done, _ = fb.terminal_status(batch)
        batch = batch[~done]
        keys = fb.canonical_keys(batch)
        for board, key in zip(batch, keys.tolist()):
            if key in seen or key in exclude:
                continue
            seen.add(key)
            collected.append(board)
            if len(collected) == count:
                break
    return np.array(collected, dtype=np.uint16)


def run_oracle(oracle_bin: Path, qfen_path: Path, out_path: Path) -> None:
    with qfen_path.open("rb") as stdin, out_path.open("wb") as stdout:
        subprocess.run(
            [str(oracle_bin)], stdin=stdin, stdout=stdout, stderr=subprocess.DEVNULL, check=True
        )


def rows_from_oracle(paths: list[Path]) -> dict[str, np.ndarray]:
    """Turn oracle JSONL into deduplicated training arrays."""
    policy_boards: list[np.ndarray] = []
    policy_targets: list[np.ndarray] = []
    policy_values: list[float] = []
    value_boards: list[np.ndarray] = []
    value_targets: list[float] = []

    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            board = fb.from_qfen(record["qfen"])[0]
            optimal = record["outcome_optimal"]
            target = np.zeros(fb.ACTION_COUNT, dtype=np.float32)
            target[optimal] = 1.0 / len(optimal)
            policy_boards.append(board)
            policy_targets.append(target)
            policy_values.append(1.0 if record["won"] else -1.0)

            parent = board[None, :]
            for action, value in record["action_values"].items():
                child = fb.apply_actions(parent, np.array([int(action)], dtype=np.int64))[0]
                value_boards.append(child)
                # `value` is from the parent mover's view; the child's own
                # mover sees the negation. Terminal children are -1 already.
                value_targets.append(-1.0 if -float(value) < 0 else 1.0)

    boards = np.array(policy_boards + value_boards, dtype=np.uint16)
    n_policy = len(policy_boards)
    policy = np.concatenate(
        [
            np.array(policy_targets, dtype=np.float32),
            np.zeros((len(value_boards), fb.ACTION_COUNT), dtype=np.float32),
        ]
    )
    policy_weight = np.concatenate(
        [np.ones(n_policy, dtype=np.float32), np.zeros(len(value_boards), dtype=np.float32)]
    )
    values = np.array(policy_values + value_targets, dtype=np.float32)

    # Dedup by canonical key, preferring rows that carry a policy target.
    keys = fb.canonical_keys(boards)
    order = np.argsort(-policy_weight, kind="stable")
    _, first = np.unique(keys[order], return_index=True)
    keep = np.sort(order[first])
    return {
        "boards": boards[keep],
        "policy_target": policy[keep],
        "policy_weight": policy_weight[keep],
        "value_target": values[keep],
        "plies": fb.popcount(fb.occupancy(boards[keep])).astype(np.int16),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/oracle/corpus"))
    parser.add_argument(
        "--oracle-bin",
        type=Path,
        default=Path("../quantik-core-rust/target/release/examples/exact_oracle"),
    )
    parser.add_argument("--probe", type=Path, default=Path("runs/oracle/probe.jsonl"))
    parser.add_argument("--plan", type=Path, help="JSON {ply: count}; defaults to DEFAULT_PLAN")
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--skip-solve", action="store_true", help="only rebuild the npz")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    exclude: set[int] = set()
    if args.probe.exists():
        probe_boards = np.concatenate(
            [
                fb.from_qfen(json.loads(line)["qfen"])
                for line in args.probe.read_text().splitlines()
                if line.strip()
            ]
        )
        exclude = set(fb.canonical_keys(probe_boards).tolist())
        print(f"excluding {len(exclude)} held-out probe positions from the corpus")

    plan = (
        {int(k): int(v) for k, v in json.loads(args.plan.read_text()).items()}
        if args.plan
        else DEFAULT_PLAN
    )

    jsonl_paths = []
    # Cheapest plies first so the corpus is usable long before the expensive
    # openings finish.
    for ply in sorted(plan, reverse=True):
        count = plan[ply]
        qfen_path = args.out / f"ply{ply:02d}.qfen"
        jsonl_path = args.out / f"ply{ply:02d}.jsonl"
        jsonl_paths.append(jsonl_path)
        if args.skip_solve and jsonl_path.exists():
            continue
        if not qfen_path.exists():
            boards = sample_ply(ply, count, rng, exclude)
            qfen_path.write_text("\n".join(fb.to_qfen(b) for b in boards))
            print(f"ply {ply}: sampled {boards.shape[0]}", flush=True)
        if not jsonl_path.exists():
            started = time.perf_counter()
            run_oracle(args.oracle_bin, qfen_path, jsonl_path)
            solved = sum(1 for _ in jsonl_path.open())
            print(
                f"ply {ply}: solved {solved} in {time.perf_counter() - started:.0f}s",
                flush=True,
            )

    arrays = rows_from_oracle([p for p in jsonl_paths if p.exists()])
    npz_path = args.out / "exact.npz"
    np.savez_compressed(npz_path, **arrays)
    n_policy = int(arrays["policy_weight"].sum())
    print(
        f"wrote {npz_path}: {arrays['boards'].shape[0]:,} unique positions "
        f"({n_policy:,} with exact policy targets)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
