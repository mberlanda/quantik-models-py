#!/usr/bin/env python
"""Back-induct exact labels for plies 0-2 from v3's ply-3 rows (QW-027/W3).

Plies 0, 1 and 2 hold 1, 3 and 51 canonical positions (55 in all) and no corpus
reaches them. Their exact values and optimal-move sets follow from the ply below by
`scripts.solve_opening.induct`. v3 holds 726 rows at ply 3, the complete canonical
level, so the induction has the precondition it needs.

**Back-induction from an incomplete level is silently wrong**: a missing sibling makes
a lost position look won or drops an optimal move, and nothing downstream notices. So
before inducting, the set of canonical keys of v3's ply-3 rows is compared, as a
*set*, with the complete `runs/canonical/level03.npy`. A count can match while the sets
differ, so counts are never the check. On any difference the run aborts and names the
missing keys.

Labels come only from exact induction over the oracle's ply-3 values
(`canonical-invariants.md#I4`); nothing here reads a game outcome. Nothing is written
into an existing corpus: the output is a standalone ``.npz`` (boards, optimal_mask,
value_target, plies) for W6 to merge.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantik_models.data.exact_corpus import ExactCorpus  # noqa: E402
from quantik_models.env import fastboard as fb  # noqa: E402
from scripts.solve_opening import induct  # noqa: E402

FRONTIER_PLY = 3


class IncompleteLevelError(RuntimeError):
    """The ply-3 rows do not cover exactly the canonical ply-3 level."""


def _fmt(keys: np.ndarray, limit: int = 10) -> str:
    shown = ", ".join(f"{int(k):#018x}" for k in keys[:limit])
    return shown + (f", ... ({len(keys)} in all)" if len(keys) > limit else "")


def assert_complete_level(corpus_boards: np.ndarray, canonical_boards: np.ndarray) -> None:
    """Raise `IncompleteLevelError` unless the two board sets have equal canonical-key *sets*.

    Names the keys missing from the corpus, and any it holds that are not canonical
    ply-3 positions. Duplicate rows for one position are not an error here (the set is
    what matters) but the caller must not rely on row counts.
    """
    have = np.unique(fb.canonical_keys(corpus_boards))
    want = np.unique(fb.canonical_keys(canonical_boards))
    missing = np.setdiff1d(want, have)
    extra = np.setdiff1d(have, want)
    if len(missing) or len(extra):
        parts = []
        if len(missing):
            parts.append(f"{len(missing)} canonical ply-{FRONTIER_PLY} key(s) missing from the corpus: {_fmt(missing)}")
        if len(extra):
            parts.append(f"{len(extra)} corpus key(s) that are not canonical ply-{FRONTIER_PLY} positions: {_fmt(extra)}")
        raise IncompleteLevelError(
            "refusing to back-induct from an incomplete level: " + "; ".join(parts)
        )


def _canonical_children(boards: np.ndarray) -> np.ndarray:
    """One representative per canonical class of live children of `boards`."""
    legal = fb.legal_masks(boards)
    rows, actions = np.nonzero(legal)
    children = fb.apply_actions(boards[rows], actions)
    done, _ = fb.terminal_status(children)
    children = children[~done]
    _, first = np.unique(fb.canonical_keys(children), return_index=True)
    return children[first]


def shallow_levels() -> list[np.ndarray]:
    """Canonical live positions at plies 0, 1, 2, expanded from the empty board."""
    levels = [fb.empty_boards(1)]
    for _ in range(2):
        levels.append(_canonical_children(levels[-1]))
    return levels


def frontier_rows(corpus: ExactCorpus) -> tuple[np.ndarray, np.ndarray]:
    """Ply-3 boards and their won-by-mover flags, from the corpus's exact value targets."""
    at3 = corpus.plies == FRONTIER_PLY
    values = corpus.value_target[at3]
    if not np.all(np.abs(values) == 1.0):
        raise ValueError("ply-3 value_target is not exactly +1/-1 on every row")
    return corpus.boards[at3], values > 0


def induct_shallow(corpus: ExactCorpus, level03: np.ndarray) -> tuple[ExactCorpus, dict[int, int]]:
    """Guarded induction of plies 2, 1, 0 from `corpus`'s ply-3 rows."""
    boards3, won3 = frontier_rows(corpus)
    assert_complete_level(boards3, level03)

    keys = fb.canonical_keys(boards3)
    order = np.argsort(keys)
    child_keys, child_won = keys[order], won3[order]
    # A duplicated position must agree with itself, or the lookup would pick one silently.
    same = child_keys[1:] == child_keys[:-1]
    if np.any(same & (child_won[1:] != child_won[:-1])):
        raise ValueError("ply-3 rows disagree about the value of one canonical position")

    levels = shallow_levels()
    boards_out, mask_out, value_out = [], [], []
    for ply in (2, 1, 0):
        boards = levels[ply]
        won, mask = induct(boards, child_keys, child_won)
        boards_out.append(boards)
        mask_out.append(mask)
        value_out.append(np.where(won, 1.0, -1.0).astype(np.float32))
        keys = fb.canonical_keys(boards)
        order = np.argsort(keys)
        child_keys, child_won = keys[order], won[order]

    boards_all = np.concatenate(boards_out)
    result = ExactCorpus(
        boards=boards_all,
        optimal_mask=np.concatenate(mask_out),
        value_target=np.concatenate(value_out),
        plies=fb.popcount(fb.occupancy(boards_all)).astype(np.int16),
    )
    counts = {p: int((result.plies == p).sum()) for p in (0, 1, 2)}
    return result, counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, default=Path("runs/oracle/corpus/exact-sampled-v3.npz"), help="corpus holding the exact ply-3 rows")
    parser.add_argument("--level03", type=Path, default=Path("runs/canonical/level03.npy"), help="the complete canonical ply-3 level")
    parser.add_argument("--out", type=Path, default=Path("runs/oracle/shallow-induced.npz"), help="standalone artefact for W6 (no existing corpus is touched)")
    args = parser.parse_args(argv)

    corpus = ExactCorpus.load(args.corpus)
    level03 = np.load(args.level03)
    try:
        result, counts = induct_shallow(corpus, level03)
    except IncompleteLevelError as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        return 2
    print(f"completeness: ply-3 key set equals {args.level03} ({len(level03)} canonical positions)")
    for ply, n in counts.items():
        print(f"ply {ply}: {n} positions")
    print(f"total: {len(result)} positions, {result.policy_rows} with an optimal-move set")
    path = result.save(args.out)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
