"""Reconcile this enumeration with quantik-core-py's published depth-8 count.

`GAME_TREE_ANALYSIS.md` reports **17,900,160** unique canonical states at
depth 8. `count_canonical.py` finds **17,894,928 live** — a 5,232 gap, where
plies 1-7 matched exactly.

This resolves it: splitting ply-8 terminal positions by *why* they are
terminal shows the gap is exactly the positions where the mover has no legal
reply but no line is complete. Those are losses, so this project counts them
as terminal; the published table counts them as ongoing. Same enumeration,
different convention.

    ply 8 canonical positions: 20,049,874
      live (a decision to make):          17,894,928
      terminal, a line is complete:        2,149,714
      terminal, mover has no legal move:       5,232
      live + stuck = 17,900,160  == published figure

Run: `.venv/bin/python scripts/verify_published_counts.py`
(needs `runs/canonical/level07.npy` from `count_canonical.py`; ~8 minutes).
"""
import numpy as np
from quantik_models.env import fastboard as fb

parents = np.load("runs/canonical/level07.npy")
print(f"expanding {parents.shape[0]:,} ply-7 positions", flush=True)
won = stuck = live = 0
seen = np.empty(0, dtype=np.uint64)
key_parts, class_parts = [], []

def compact():
    global seen, key_parts, class_parts, won, stuck, live
    keys = np.concatenate(key_parts); cls = np.concatenate(class_parts)
    keys, index = np.unique(keys, return_index=True)
    cls = cls[index]
    fresh = ~np.isin(keys, seen, assume_unique=True)
    keys, cls = keys[fresh], cls[fresh]
    seen = np.union1d(seen, keys)
    won += int((cls == 1).sum()); stuck += int((cls == 2).sum()); live += int((cls == 0).sum())
    key_parts.clear(); class_parts.clear()

for start in range(0, parents.shape[0], 150_000):
    chunk = parents[start:start + 150_000]
    legal = fb.legal_masks(chunk)
    rows, actions = np.nonzero(legal)
    children = fb.apply_actions(chunk[rows], actions)
    has_win = fb.has_winning_line(children)
    no_move = ~fb.legal_masks(children).any(axis=1)
    cls = np.where(has_win, 1, np.where(no_move, 2, 0)).astype(np.int8)
    keys = fb.canonical_keys(children)
    keys, index = np.unique(keys, return_index=True)
    key_parts.append(keys); class_parts.append(cls[index])
    if len(key_parts) >= 12:
        compact()
compact()
print(f"\nply 8 canonical positions: {won + stuck + live:,}")
print(f"  live (a decision to make): {live:,}")
print(f"  terminal, a line is complete: {won:,}")
print(f"  terminal, mover has no legal move: {stuck:,}")
print(f"\nlive + stuck = {live + stuck:,}   (published depth-8 figure: 17,900,160)")
