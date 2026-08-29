"""Fold newly solved positions into an existing corpus.

Autoplay produces positions the corpus does not have (`arena.autoplay`),
and the exact solver labels them. This is the last step: merge the labelled
rows into the corpus a trainer reads, without breaking the two invariants
everything downstream assumes.

**The probe must stay held out.** Every evaluation number in this project
compares against `probe-large.jsonl`, and a probe position that leaks into
the corpus turns generalisation into recall while every report still looks
fine. Exclusion is applied to the *merged* result, not just to the new
rows, because solving a position also labels its children — a probe
position can arrive as somebody's child without ever having been sampled.
That is exactly how sixteen probe positions reached the first corpus.

**One row per canonical position, preferring policy labels.** A position
can appear as a solved parent in one file and a value-only child in
another. Keeping both would double-count it in the loss and split it across
the train/val boundary is not the risk — `split_by_key` handles that — but
the duplicate would silently reweight the position.

    python -m quantik_models.data.merge_corpus \\
      --corpus runs/oracle/corpus/exact-sampled.npz \\
      --solved runs/autoplay/lineup-p3/solved.jsonl \\
      --out runs/oracle/corpus/exact-sampled-v2.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..env import fastboard as fb
from .exact_corpus import ExactCorpus


def probe_keys(paths: list[Path]) -> set[int]:
    """Canonical keys of every position in the held-out probes."""
    keys: set[int] = set()
    for path in paths:
        qfens = [
            json.loads(line)["qfen"]
            for line in path.read_text().splitlines()
            if line.strip()
        ]
        if not qfens:
            continue
        boards = np.concatenate([fb.from_qfen(q) for q in qfens])
        keys.update(int(k) for k in fb.canonical_keys(boards))
    return keys


def merge(corpora: list[ExactCorpus], exclude: set[int] | None = None) -> ExactCorpus:
    """Concatenate and deduplicate on the canonical key.

    Order matters for the tie-break: rows carrying a policy target win over
    value-only rows for the same position, regardless of which corpus they
    came from, because a policy label is strictly more information.
    """
    corpora = [c for c in corpora if len(c)]
    if not corpora:
        raise ValueError("nothing to merge")

    boards = np.concatenate([c.boards for c in corpora])
    masks = np.concatenate([c.optimal_mask for c in corpora])
    values = np.concatenate([c.value_target for c in corpora])

    keys = fb.canonical_keys(boards)
    # Stable sort puts policy-labelled rows first within each key, so
    # `np.unique`'s first occurrence is the one worth keeping.
    order = np.argsort(masks == 0, kind="stable")
    _, first = np.unique(keys[order], return_index=True)
    keep = np.sort(order[first])

    if exclude:
        held_out = np.fromiter(
            (int(k) in exclude for k in keys[keep].tolist()), dtype=bool, count=len(keep)
        )
        if held_out.any():
            print(f"dropped {int(held_out.sum()):,} held-out probe positions")
        keep = keep[~held_out]

    kept = boards[keep]
    return ExactCorpus(
        boards=kept,
        optimal_mask=masks[keep],
        value_target=values[keep],
        plies=fb.popcount(fb.occupancy(kept)).astype(np.int16),
    )


def describe(corpus: ExactCorpus, label: str) -> str:
    plies, counts = np.unique(corpus.plies, return_counts=True)
    span = ", ".join(f"{p}:{c:,}" for p, c in zip(plies.tolist(), counts.tolist()))
    return (
        f"{label}: {len(corpus):,} rows, {corpus.policy_rows:,} policy-labelled\n"
        f"  by ply: {span}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, action="append", default=None)
    parser.add_argument(
        "--solved", type=Path, action="append", default=None,
        help="exact_oracle JSONL, e.g. runs/autoplay/*/solved.jsonl",
    )
    parser.add_argument(
        "--probe", type=Path, action="append",
        default=None,
        help="held-out probes to exclude; defaults to probe-large.jsonl",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    probes = args.probe or [Path("runs/oracle/probe-large.jsonl")]
    probes = [p for p in probes if p.exists()]
    exclude = probe_keys(probes)
    print(f"holding out {len(exclude):,} canonical keys from {len(probes)} probe file(s)")

    corpora = []
    for path in args.corpus or []:
        corpus = ExactCorpus.load(path)
        print(describe(corpus, str(path)))
        corpora.append(corpus)

    if args.solved:
        # Reuse the corpus builder's oracle reader so the merged rows are
        # produced exactly the way the original corpus's were — including
        # the free child value labels and the sign convention on them.
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from scripts.build_oracle_corpus import rows_from_oracle

        existing = [p for p in args.solved if p.exists() and p.stat().st_size]
        missing = [p for p in args.solved if p not in existing]
        if missing:
            print(f"skipping {len(missing)} missing or empty solve file(s): {missing}")
        if existing:
            fresh = rows_from_oracle(existing)
            print(describe(fresh, "newly solved"))
            corpora.append(fresh)

    merged = merge(corpora, exclude)
    print(describe(merged, "merged"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.save(args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
