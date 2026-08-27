#!/usr/bin/env python
"""Build a held-out, exactly-solved evaluation probe.

Two disjointness rules make the resulting accuracy numbers honest, and both
are enforced **up to symmetry** — a rotated copy of a training board is the
same position:

* nothing already in the training corpus,
* nothing already in an existing probe (so probes can be extended, not
  silently re-drawn).

Sampling is weighted toward the opening. That is where the engines actually
differ — from ply 8 onward every engine measured so far is exact — so
evaluation precision is worth paying for there and nowhere else.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from quantik_models.data.exact_corpus import ExactCorpus
from quantik_models.env import fastboard as fb

# Positions per ply. Heavily front-loaded: at plies 4-5 a hundred positions
# is the difference between "probably better" and "demonstrably better",
# while at ply 12 every engine already scores 100%.
DEFAULT_PLAN = {4: 1200, 5: 1200, 6: 1200, 7: 1200, 8: 800, 9: 600, 10: 600, 11: 500, 12: 500}


def excluded_keys(corpus: Path | None, probes: list[Path]) -> set[int]:
    keys: set[int] = set()
    if corpus and corpus.exists():
        keys |= set(ExactCorpus.load(corpus).canonical_keys().tolist())
        print(f"excluding {len(keys):,} training positions", flush=True)
    for path in probes:
        if not path.exists():
            continue
        boards = np.concatenate(
            [fb.from_qfen(json.loads(l)["qfen"]) for l in path.read_text().splitlines() if l.strip()]
        )
        before = len(keys)
        keys |= set(fb.canonical_keys(boards).tolist())
        print(f"excluding {len(keys) - before:,} positions from {path.name}", flush=True)
    return keys


def sample_ply(ply: int, count: int, rng: np.random.Generator, exclude: set[int]) -> np.ndarray:
    """Canonically distinct, non-terminal, unseen positions at `ply`."""
    collected: list[np.ndarray] = []
    seen: set[int] = set()
    for _ in range(400):
        if len(collected) >= count:
            break
        batch = fb.empty_boards(20_000)
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
        for board, key in zip(batch, fb.canonical_keys(batch).tolist()):
            if key in seen or key in exclude:
                continue
            seen.add(key)
            collected.append(board)
            if len(collected) == count:
                break
    return np.array(collected, dtype=np.uint16)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/oracle/probe-large.jsonl"))
    parser.add_argument("--work", type=Path, default=Path("runs/oracle/probe-large"))
    parser.add_argument("--corpus", type=Path, default=Path("runs/oracle/corpus/sampled.npz"))
    parser.add_argument("--exclude-probe", type=Path, nargs="*", default=[Path("runs/oracle/probe.jsonl")])
    parser.add_argument(
        "--oracle-bin", type=Path,
        default=Path("../.oracle-worktree/target/release/examples/exact_oracle"))
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--seed", type=int, default=777001)
    args = parser.parse_args(argv)

    args.work.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    exclude = excluded_keys(args.corpus, list(args.exclude_probe))

    # Cheapest plies first, so the probe is usable before the openings finish.
    for ply in sorted(DEFAULT_PLAN, reverse=True):
        qfen_path = args.work / f"ply{ply:02d}.qfen"
        jsonl_path = args.work / f"ply{ply:02d}.jsonl"
        if not qfen_path.exists():
            boards = sample_ply(ply, DEFAULT_PLAN[ply], rng, exclude)
            qfen_path.write_text("\n".join(fb.to_qfen(b) for b in boards))
            print(f"ply {ply}: sampled {boards.shape[0]:,}", flush=True)
        wanted = sum(1 for _ in qfen_path.open())
        have = sum(1 for _ in jsonl_path.open()) if jsonl_path.exists() else 0
        if have < wanted:
            started = time.perf_counter()
            command = [str(args.oracle_bin), "--append-to", str(jsonl_path),
                       "--threads", str(args.threads)]
            with qfen_path.open("rb") as stdin:
                subprocess.run(command, stdin=stdin, check=True)
            print(f"ply {ply}: solved {wanted:,} in {time.perf_counter() - started:.0f}s", flush=True)

    merged = []
    for ply in sorted(DEFAULT_PLAN):
        path = args.work / f"ply{ply:02d}.jsonl"
        if path.exists():
            merged += [l for l in path.read_text().splitlines() if l.strip()]
    args.out.write_text("\n".join(merged) + "\n")
    won = sum(json.loads(l)["won"] for l in merged)
    print(f"wrote {args.out}: {len(merged):,} solved positions, {won:,} won by the mover")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
