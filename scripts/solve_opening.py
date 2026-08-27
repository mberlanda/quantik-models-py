#!/usr/bin/env python
"""Solve the entire Quantik opening exactly, and emit it as training data.

Solving every position at one ply makes every shallower ply free: a position's
value is the best of its children's negated values, and its optimal moves are
exactly those leading to a child the opponent loses. So instead of paying the
full oracle (one solve per legal move) at every level, this pays a *root-only*
solve — 25x cheaper — at one deep level and back-induces everything above it.

With `--frontier 6` that means:

* exact values for **every** canonical position at plies 0-6,
* exact optimal-move sets for **every** canonical position at plies 0-5,

which is the complete solution of the region where the incumbent minimax is
actually beatable (it is already perfect from ply 8).

The canonical counts this enumerates (3, 51, 726, 10 946, 105 632, 901 916)
match `quantik-core-py/GAME_TREE_ANALYSIS.md` exactly, which independently
validates the vectorized engine.
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


def enumerate_levels(frontier_ply: int, cache: Path) -> list[np.ndarray]:
    """Canonical, non-terminal positions at each ply from 0 to `frontier_ply`.

    Expanding from canonical representatives is sound: if a child X is
    reachable from Y, then sigma(X) is reachable from Y's representative
    sigma(Y), and sigma(X) shares X's canonical key.
    """
    levels = [fb.empty_boards(1)]
    for ply in range(1, frontier_ply + 1):
        path = cache / f"level{ply:02d}.npy"
        if path.exists():
            levels.append(np.load(path))
            print(f"ply {ply}: {levels[-1].shape[0]:,} canonical (cached)", flush=True)
            continue
        started = time.perf_counter()
        legal = fb.legal_masks(levels[-1])
        rows, actions = np.nonzero(legal)
        children = fb.apply_actions(levels[-1][rows], actions)
        keys = fb.canonical_keys(children)
        _, first = np.unique(keys, return_index=True)
        children = children[first]
        done, _ = fb.terminal_status(children)
        live = children[~done]
        np.save(path, live)
        levels.append(live)
        print(
            f"ply {ply}: {live.shape[0]:,} canonical live "
            f"({int(done.sum()):,} terminal) [{time.perf_counter() - started:.0f}s]",
            flush=True,
        )
    return levels


def solve_frontier(
    boards: np.ndarray, oracle_bin: Path, workdir: Path, threads: int | None
) -> np.ndarray:
    """Root-only exact solve of every frontier position; returns a `won` mask.

    Delegates to the oracle in append mode so the solve streams to disk and
    resumes: re-running skips the QFENs already in `frontier.jsonl`. A
    million-position solve is hours long and must survive an interrupt.
    """
    qfen_path = workdir / "frontier.qfen"
    jsonl_path = workdir / "frontier.jsonl"
    if not qfen_path.exists():
        qfen_path.write_text("\n".join(fb.to_qfen(b) for b in boards))
    solved = sum(1 for _ in jsonl_path.open()) if jsonl_path.exists() else 0
    if solved < boards.shape[0]:
        print(
            f"solving {boards.shape[0] - solved:,} of {boards.shape[0]:,} "
            f"frontier positions...",
            flush=True,
        )
        started = time.perf_counter()
        command = [str(oracle_bin), "--roots-only", "--append-to", str(jsonl_path)]
        if threads:
            command += ["--threads", str(threads)]
        with qfen_path.open("rb") as stdin:
            subprocess.run(command, stdin=stdin, check=True)
        print(f"  done in {time.perf_counter() - started:.0f}s", flush=True)

    won_by_key: dict[int, bool] = {}
    chunk: list[str] = []

    def flush(lines: list[str]) -> None:
        if not lines:
            return
        records = [json.loads(line) for line in lines]
        keys = fb.canonical_keys(np.concatenate([fb.from_qfen(r["qfen"]) for r in records]))
        for key, record in zip(keys.tolist(), records):
            won_by_key[key] = bool(record["won"])

    with jsonl_path.open() as handle:
        for line in handle:
            if line.strip():
                chunk.append(line)
            if len(chunk) >= 20000:
                flush(chunk)
                chunk = []
    flush(chunk)

    keys = fb.canonical_keys(boards)
    missing = [k for k in keys.tolist() if k not in won_by_key]
    if missing:
        raise RuntimeError(f"{len(missing)} frontier positions were not solved")
    return np.array([won_by_key[k] for k in keys.tolist()], dtype=bool)


def induct(boards: np.ndarray, child_keys: np.ndarray, child_won: np.ndarray):
    """One backward-induction step.

    Returns `(won, policy_target)` for `boards`, given a lookup of exact
    outcomes for every live position one ply deeper. A move wins when it
    leaves the opponent lost — either at a terminal position, or at a solved
    child the opponent loses.
    """
    legal = fb.legal_masks(boards)
    rows, actions = np.nonzero(legal)
    children = fb.apply_actions(boards[rows], actions)
    done, _ = fb.terminal_status(children)

    # Value of the move, from the mover's perspective.
    move_wins = np.zeros(rows.shape[0], dtype=bool)
    move_wins[done] = True  # the child is dead on arrival: the mover just won
    live = ~done
    if live.any():
        keys = fb.canonical_keys(children[live])
        slot = np.searchsorted(child_keys, keys)
        if np.any(slot >= child_keys.shape[0]) or np.any(child_keys[np.minimum(slot, child_keys.shape[0] - 1)] != keys):
            raise RuntimeError("a live child was missing from the deeper level")
        move_wins[live] = ~child_won[slot]

    n = boards.shape[0]
    any_win = np.zeros(n, dtype=bool)
    np.logical_or.at(any_win, rows, move_wins)

    # Outcome-optimal moves: from a won position, only the winning moves; from
    # a lost one every move loses, so all of them are equally optimal.
    keep = np.where(any_win[rows], move_wins, True)
    mask = np.zeros(n, dtype=np.uint64)
    np.bitwise_or.at(
        mask, rows[keep], (np.uint64(1) << actions[keep].astype(np.uint64))
    )
    return any_win, mask


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier", type=int, default=6)
    parser.add_argument("--out", type=Path, default=Path("runs/oracle/opening"))
    parser.add_argument(
        "--oracle-bin",
        type=Path,
        default=Path("../.oracle-worktree/target/release/examples/exact_oracle"),
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=14,
        help="rayon threads for the solver; leave headroom so the machine stays usable",
    )
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    levels = enumerate_levels(args.frontier, args.out)
    frontier_won = solve_frontier(
        levels[args.frontier], args.oracle_bin, args.out, args.threads
    )
    print(
        f"frontier ply {args.frontier}: "
        f"{frontier_won.mean():.1%} are wins for the side to move",
        flush=True,
    )

    boards_out: list[np.ndarray] = []
    mask_out: list[np.ndarray] = []
    value_out: list[np.ndarray] = []

    # The frontier itself contributes exact values but no policy (its children
    # were never solved).
    boards_out.append(levels[args.frontier])
    mask_out.append(np.zeros(frontier_won.shape[0], dtype=np.uint64))
    value_out.append(np.where(frontier_won, 1.0, -1.0).astype(np.float32))

    child_keys = fb.canonical_keys(levels[args.frontier])
    order = np.argsort(child_keys)
    child_keys, child_won = child_keys[order], frontier_won[order]

    for ply in range(args.frontier - 1, -1, -1):
        started = time.perf_counter()
        boards = levels[ply]
        won, mask = induct(boards, child_keys, child_won)
        boards_out.append(boards)
        mask_out.append(mask)
        value_out.append(np.where(won, 1.0, -1.0).astype(np.float32))
        print(
            f"ply {ply}: {boards.shape[0]:,} positions, {won.mean():.1%} won by the mover "
            f"[{time.perf_counter() - started:.0f}s]",
            flush=True,
        )
        keys = fb.canonical_keys(boards)
        order = np.argsort(keys)
        child_keys, child_won = keys[order], won[order]

    boards = np.concatenate(boards_out)
    corpus = ExactCorpus(
        boards=boards,
        optimal_mask=np.concatenate(mask_out),
        value_target=np.concatenate(value_out),
        plies=fb.popcount(fb.occupancy(boards)).astype(np.int16),
    )
    path = corpus.save(args.out / "opening-exact.npz")
    print(
        f"wrote {path}: {len(corpus):,} exactly-solved positions "
        f"({corpus.policy_rows:,} with exact policy)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
