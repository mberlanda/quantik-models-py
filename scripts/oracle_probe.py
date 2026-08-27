#!/usr/bin/env python
"""Score an agent against exact game-theoretic truth.

Win rates tell you an agent beat another agent; they do not tell you how
close it is to *correct*. This probe compares an agent's move choice on a
fixed set of solved positions against the exact solver's verdict (produced by
`quantik-core-rust`'s `exact_oracle` example).

Two numbers are reported:

* **outcome accuracy** — over positions the mover provably wins, how often the
  agent picks a move that keeps the win. This is the honest bar: in a lost
  position every move loses, so nothing is being tested there.
* **value error** — mean |predicted value - true outcome| for network agents,
  which says whether the value head actually knows who is winning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from quantik_models.arena.registry import build_agent
from quantik_models.env import fastboard as fb


def load_probe(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def evaluate(agent, probe: list[dict], seed: int = 0) -> dict:
    won_total = won_correct = 0
    lost_total = lost_correct = 0
    by_ply: dict[int, list[int]] = {}
    for index, row in enumerate(probe):
        board = fb.from_qfen(row["qfen"])[0]
        action = agent.select(board, seed + index)
        optimal = set(row["outcome_optimal"])
        correct = action in optimal
        ply = int(fb.popcount(fb.occupancy(board[None, :]))[0])
        if row["won"]:
            won_total += 1
            won_correct += int(correct)
            bucket = by_ply.setdefault(ply, [0, 0])
            bucket[0] += int(correct)
            bucket[1] += 1
        else:
            lost_total += 1
            lost_correct += int(correct)
    return {
        "agent": agent.name,
        "won_positions": won_total,
        "outcome_accuracy": won_correct / won_total if won_total else 0.0,
        "lost_positions": lost_total,
        "lost_position_agreement": lost_correct / lost_total if lost_total else 0.0,
        "accuracy_by_ply": {
            str(ply): {"correct": c, "total": t, "accuracy": c / t}
            for ply, (c, t) in sorted(by_ply.items())
        },
    }


def value_error(spec: dict, probe: list[dict]) -> dict | None:
    """Mean |value head - true outcome| over the probe, for network agents."""
    if spec.get("kind") not in {"net-policy", "net-mcts"}:
        return None
    from quantik_models.arena.registry import load_evaluator

    evaluator = load_evaluator(
        spec["checkpoint"], spec.get("device", "cpu"), spec.get("eval_batch_size", 4096)
    )
    boards = np.concatenate([fb.from_qfen(row["qfen"]) for row in probe])
    _, values = evaluator(boards, fb.legal_masks(boards))
    truth = np.array([1.0 if row["won"] else -1.0 for row in probe], dtype=np.float32)
    return {
        "value_mae": float(np.abs(values - truth).mean()),
        "value_sign_accuracy": float(np.mean(np.sign(values) == np.sign(truth))),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, default=Path("runs/oracle/probe.jsonl"))
    parser.add_argument("--agents", type=Path, required=True, help="JSON list of agent specs")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    probe = load_probe(args.probe)
    specs = json.loads(args.agents.read_text())
    report = {"probe": str(args.probe), "positions": len(probe), "results": []}
    print(f"probe: {len(probe)} solved positions")
    for spec in specs:
        row = evaluate(build_agent(spec), probe, seed=args.seed)
        extra = value_error(spec, probe)
        if extra:
            row.update(extra)
        report["results"].append(row)
        print(
            f"  {row['agent']:<24} outcome accuracy {row['outcome_accuracy']:.1%} "
            f"({row['won_positions']} won positions)"
            + (f"  value MAE {row['value_mae']:.3f}" if extra else "")
        )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
