"""Export the play store's positions to a solver queue.

Every recorded game already has its positions in `game_positions`
(`play/store.py`), reached by a human actually playing rather than by an
engine self-playing. Nothing has consumed them: autoplay is the only source
that has ever fed the corpus, so the positions people actually reach never
get there. This closes that loop by producing the same artifact autoplay
already produces — a `to-solve.qfen.gz` the exact solver eats and
`data/merge_corpus.py` folds in. No new format, no new consumer.

**Human game outcomes are never labels. Only positions travel.** A human
game's `winner` says which of two fallible players won, which is not the
value of a position — that comes from the exact oracle alone, the same
discipline autoplay follows (`docs/labeling-strategy.md`,
`docs/autoplay.md`). This module never reads `games.winner`; it reads
`game_positions` through `play.store.distinct_positions` and nothing else.

## The trap this module exists to avoid

`game_positions.canonical_key` is `str(int(fb.canonical_keys(boards)[0]))`
(`play/record.py:_canonical_key`) — a **decimal string**. `ExactCorpus`
gives `fb.canonical_keys(corpus.boards)` — a **numpy `uint64` array**.
Comparing the two without converting finds no overlap: the filter drops
nothing, the queue looks a plausible size, and hours of solver time go to
positions the corpus already has. That is not hypothetical — the first
oracle runs made exactly this mistake against a superseded corpus and paid
for it in real solver time (`arena.pack.merge_qfens`'s docstring). Every
key comparison here goes through the decimal-string form for that reason.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..arena.pack import write_gzip
from ..env import fastboard as fb
from . import store


def _known_canonical_keys(corpus_path: Path) -> set[str]:
    """The corpus's canonical keys, as the decimal strings the store uses.

    Converting once, here, keeps the only place that has to remember the
    format mismatch between `game_positions.canonical_key` (a decimal
    string) and `ExactCorpus`'s `uint64` array.
    """
    from ..data.exact_corpus import ExactCorpus

    corpus = ExactCorpus.load(corpus_path)
    return {str(int(key)) for key in fb.canonical_keys(corpus.boards)}


def export_queue(db: Path, out: Path, corpus: Path | None = None, max_ply: int = 6) -> dict:
    """Write `out/to-solve.qfen.gz` and `out/summary.json`.

    Read-only against `db`: only `store.distinct_positions` is called, never
    anything that writes a row. `corpus` is optional — with none given,
    every distinct position is exported and `summary.json` says so via a
    null `filtered_against`, rather than silently exporting nothing.
    """
    conn = store.connect(db)
    try:
        positions = store.distinct_positions(conn, max_ply=max_ply)
    finally:
        conn.close()

    known = _known_canonical_keys(corpus) if corpus is not None else set()
    kept = [(qfen, key) for qfen, key in positions if key not in known]

    out.mkdir(parents=True, exist_ok=True)
    write_gzip("\n".join(qfen for qfen, _ in kept) + "\n", out / "to-solve.qfen.gz")

    summary = {
        "source_db": str(db),
        "filtered_against": str(corpus) if corpus else None,
        "max_ply": max_ply,
        "positions_found": len(positions),
        # The number that tells you the filter worked at all — the twelve-hour
        # incident this module exists to prevent was invisible precisely
        # because nobody was counting it.
        "positions_dropped_known": len(positions) - len(kept),
        "positions_written": len(kept),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, required=True, help="the play service's games.db")
    parser.add_argument(
        "--out", type=Path, required=True, help="directory for to-solve.qfen.gz and summary.json"
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="drop positions this corpus already has; omit to export every distinct position",
    )
    parser.add_argument(
        "--max-ply",
        type=int,
        default=6,
        help="ply cut-off, matching autoplay's --max-solve-ply (default: 6)",
    )
    args = parser.parse_args(argv)

    if not args.db.exists():
        print(f"no database at {args.db}", file=sys.stderr)
        return 1

    summary = export_queue(args.db, args.out, args.corpus, args.max_ply)
    print(json.dumps(summary, indent=2))
    print(
        f"{summary['positions_found']:,} positions found, "
        f"{summary['positions_dropped_known']:,} already known, "
        f"{summary['positions_written']:,} written -> {args.out / 'to-solve.qfen.gz'}"
    )
    # An empty queue means every position played is already in the corpus —
    # a legitimate, reportable outcome, not a failure a cron job should page on.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
