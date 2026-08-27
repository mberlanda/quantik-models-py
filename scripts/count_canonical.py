#!/usr/bin/env python
"""Count Quantik's canonical positions at every ply, exactly.

`quantik-core-py/GAME_TREE_ANALYSIS.md` publishes these counts to depth 8.
This recomputes them from scratch and continues deeper, which is what a
coverage claim needs: "we solved N positions at ply p" means nothing without
the denominator.

Positions are counted **up to symmetry** — Quantik is invariant under 8
dihedral board symmetries composed with 24 shape relabelings, so a position
and its 191 images are one game, and counting them separately would inflate
every denominator by up to 192x.

Levels are expanded in chunks with periodic compaction, because the child
multiset at the deeper plies runs to hundreds of millions of boards and
materializing it whole would not fit. Each level is cached to `.npy`, so a
killed run resumes at the last completed ply.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from quantik_models.env import fastboard as fb

# Boards expanded per chunk. Each parent yields up to ~30 children, so this
# caps the working set at a few million boards regardless of level size.
CHUNK = 150_000
# Compact the running unique-key array every this many chunks; more often
# costs sorts, less often costs memory.
COMPACT_EVERY = 12


def expand_level(
    parents: np.ndarray, keep_boards: bool
) -> tuple[int, int, np.ndarray | None]:
    """Return `(live, terminal, boards)` for the ply below `parents`.

    `boards` is one representative per canonical class, or None when the
    caller only needs counts (the deepest level asked for).
    """
    key_parts: list[np.ndarray] = []
    board_parts: list[np.ndarray] = []
    seen_keys = np.empty(0, dtype=np.uint64)
    seen_boards = np.empty((0, 8), dtype=np.uint16)

    def compact() -> None:
        nonlocal seen_keys, seen_boards, key_parts, board_parts
        keys = np.concatenate([seen_keys] + key_parts) if key_parts else seen_keys
        if keep_boards:
            boards = np.concatenate([seen_boards] + board_parts) if board_parts else seen_boards
            keys, index = np.unique(keys, return_index=True)
            seen_boards = boards[index]
        else:
            keys = np.unique(keys)
        seen_keys = keys
        key_parts, board_parts = [], []

    for start in range(0, parents.shape[0], CHUNK):
        chunk = parents[start : start + CHUNK]
        legal = fb.legal_masks(chunk)
        rows, actions = np.nonzero(legal)
        children = fb.apply_actions(chunk[rows], actions)
        keys = fb.canonical_keys(children)
        keys, index = np.unique(keys, return_index=True)
        key_parts.append(keys)
        if keep_boards:
            board_parts.append(children[index])
        if len(key_parts) >= COMPACT_EVERY:
            compact()
    compact()

    # Terminal children are counted but never expanded further.
    if keep_boards:
        done, _ = fb.terminal_status(seen_boards)
        return int((~done).sum()), int(done.sum()), seen_boards[~done]
    # Without boards we still need the live/terminal split, so re-derive it
    # from the keys by unpacking them back into boards in slices.
    live = terminal = 0
    for start in range(0, seen_keys.shape[0], 2_000_000):
        boards = unpack_codes(seen_keys[start : start + 2_000_000])
        done, _ = fb.terminal_status(boards)
        terminal += int(done.sum())
        live += int((~done).sum())
    return live, terminal, None


def unpack_codes(codes: np.ndarray) -> np.ndarray:
    """Invert `fb.board_codes`: nibble `pos` = channel + 1, 0 = empty."""
    boards = np.zeros((codes.shape[0], 8), dtype=np.uint16)
    for pos in range(fb.SQUARES):
        nibble = ((codes >> np.uint64(4 * pos)) & np.uint64(0xF)).astype(np.int64)
        for channel in range(8):
            hit = nibble == channel + 1
            boards[hit, channel] |= np.uint16(1 << pos)
    return boards


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-ply", type=int, default=9)
    parser.add_argument("--out", type=Path, default=Path("runs/canonical"))
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    counts_path = args.out / "counts.json"

    counts = json.loads(counts_path.read_text()) if counts_path.exists() else {}
    level = fb.empty_boards(1)
    counts.setdefault("0", {"live": 1, "terminal": 0})

    for ply in range(1, args.max_ply + 1):
        cache = args.out / f"level{ply:02d}.npy"
        if str(ply) in counts and (cache.exists() or ply == args.max_ply):
            print(f"ply {ply}: {counts[str(ply)]['live']:,} live (cached)", flush=True)
            if cache.exists():
                level = np.load(cache)
                continue
            break
        started = time.perf_counter()
        keep = ply < args.max_ply
        live, terminal, boards = expand_level(level, keep_boards=keep)
        counts[str(ply)] = {"live": live, "terminal": terminal}
        counts_path.write_text(json.dumps(counts, indent=2, sort_keys=True))
        print(
            f"ply {ply}: {live:,} live + {terminal:,} terminal = {live + terminal:,} "
            f"canonical  [{time.perf_counter() - started:.0f}s]",
            flush=True,
        )
        if boards is None:
            break
        np.save(cache, boards)
        level = boards
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
