"""Puzzles mined from the exact corpus, by theme.

The examples in the visualizer were four positions picked by hand. Four is
not a set you can come back to, and picking by hand is how three of the
previous four turned out to be poor — one had no legal moves at all. This
generates them instead, from `data.exact_corpus`, where every position
already carries its game-theoretic value and its full set of
outcome-optimal actions.

**The solver is not the risk; the selection is.** A position offered as
"find the only move" that actually has two winning moves, or a "double
threat" whose threat the opponent can simply block, is a wrong answer
stated with authority, and a player has no way to tell. So each theme is a
predicate that has to hold, checked against the corpus rather than assumed
from it.

The themes, in the order they get harder:

`mate-in-1`
    A move that ends the game is available. Both of Quantik's terminal
    conditions are losses for the side to move, so the last mover always
    wins and "ends the game" is exactly "wins" — including the win by
    suffocation, which a line check would miss.

`only-move`
    Winning, with exactly one outcome-optimal action and no immediate win.
    Every other legal move throws the game.

`double-threat`
    The one the board game is actually about: the single winning move
    leaves a threat the opponent cannot answer, and the win does not always
    land on the same square. Blocking one combination concedes the other.
    Rejected when a move simply wins on the spot — that is a mate in one
    under a name promising something harder.

`endgame`
    Few legal moves left and only one of them wins.

`already-lost`
    Every legal move loses. Not a puzzle — a study, for seeing what a lost
    position looks like from the inside, and the one position an evaluation
    bar has an unambiguous right answer for.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..env import fastboard as fb

SCHEMA = "quantik-play.puzzle-pack.v1"
SHAPES = "ABCD"

#: Above this, "find the only move" is a search rather than a puzzle; below
#: it, `endgame` is the better label for the same position.
_ONLY_MOVE_MIN_LEGAL = 12
_ENDGAME_MAX_LEGAL = 8


def immediate_wins(board: np.ndarray) -> np.ndarray:
    """Legal actions that end the game, which in Quantik means win it.

    Both terminal conditions — a completed line, and a side with no legal
    move — are losses for the side to move, so whoever moved last has won.
    Checking for terminality rather than for a line is therefore both
    simpler and more correct: it catches the win by suffocation too.
    """
    single = board.reshape(1, -1)
    legal = np.flatnonzero(fb.legal_masks(single)[0])
    if not legal.size:
        return legal.astype(np.int64)
    after = fb.apply_actions(np.repeat(single, legal.size, 0), legal.astype(np.int64))
    done, _ = fb.terminal_status(after)
    return legal[done].astype(np.int64)


def describe_double_threat(board: np.ndarray, action: int) -> dict[str, Any] | None:
    """`action` as a fork, or `None` if it is not one.

    A fork here is the thing a player would recognise: after the move,
    *every* legal reply loses at once, and the win does not always land on
    the same square. The second half is what separates a real double threat
    from a single threat that happens to be unstoppable for some unrelated
    reason — without it the theme name would be a claim the position does
    not support.

    A move that wins on the spot is refused: that is a mate in one, and
    filing it here would put a trivial position under a theme promising a
    hard one.
    """
    single = board.reshape(1, -1)
    if immediate_wins(board).size:
        return None

    child = fb.apply_actions(single, np.array([action], dtype=np.int64))[0]
    done, _ = fb.terminal_status(child.reshape(1, -1))
    if bool(done[0]):
        return None

    replies = np.flatnonzero(fb.legal_masks(child.reshape(1, -1))[0])
    if replies.size < 2:
        return None

    squares: set[int] = set()
    for reply in replies:
        grandchild = fb.apply_actions(
            child.reshape(1, -1), np.array([reply], dtype=np.int64)
        )[0]
        wins = immediate_wins(grandchild)
        if not wins.size:
            return None
        squares.update(int(w) % 16 for w in wins)

    if len(squares) < 2:
        return None
    return {"replies": int(replies.size), "winning_squares": sorted(squares)}


def _optimal_actions(mask: int) -> list[int]:
    return [bit for bit in range(fb.ACTION_COUNT) if (mask >> bit) & 1]


def _decode(action: int) -> dict[str, Any]:
    return {
        "action_index": int(action),
        "shape": SHAPES[int(action) // 16],
        "position": int(action) % 16,
    }


def _puzzle(board, ply, side, theme, solutions, legal_count, extra=None):
    puzzle = {
        "qfen": fb.to_qfen(board),
        "ply": int(ply),
        "side_to_move": int(side),
        "theme": theme,
        "solutions": [int(a) for a in solutions],
        "moves": [_decode(a) for a in solutions],
        "legal_moves": int(legal_count),
    }
    if extra:
        puzzle.update(extra)
    return puzzle


def classify(board: np.ndarray, value: float, mask: int, ply: int) -> dict[str, Any] | None:
    """The theme this position belongs to, or `None` if it is not a puzzle.

    Order matters: a position that wins on the spot is a mate in one
    whatever else is true of it, and `double-threat` is tried before
    `only-move` because every double threat is also an only-move position
    and the more specific name is the more useful one.
    """
    single = board.reshape(1, -1)
    done, _ = fb.terminal_status(single)
    if bool(done[0]):
        return None

    legal = np.flatnonzero(fb.legal_masks(single)[0])
    if not legal.size:
        return None
    side = int(fb.popcount(fb.occupancy(single))[0] % 2)

    if value < 0:
        # Every legal move loses, which is what `value = -1` means in a game
        # with no draws. Offered as a study, not a puzzle.
        return _puzzle(board, ply, side, "already-lost", [], legal.size)

    wins = immediate_wins(board)
    if wins.size:
        return _puzzle(board, ply, side, "mate-in-1", wins.tolist(), legal.size)

    optimal = _optimal_actions(mask)
    if len(optimal) != 1:
        return None

    fork = describe_double_threat(board, optimal[0])
    if fork is not None:
        return _puzzle(board, ply, side, "double-threat", optimal, legal.size, fork)
    if legal.size <= _ENDGAME_MAX_LEGAL:
        return _puzzle(board, ply, side, "endgame", optimal, legal.size)
    if legal.size >= _ONLY_MOVE_MIN_LEGAL:
        return _puzzle(board, ply, side, "only-move", optimal, legal.size)
    return None


THEMES = ("mate-in-1", "only-move", "double-threat", "endgame", "already-lost")


def generate(
    corpus,
    *,
    per_theme: int = 40,
    seed: int = 0,
    scan: int = 60_000,
) -> dict[str, Any]:
    """Mine `corpus` for up to `per_theme` puzzles of each theme.

    `scan` bounds the work: the double-threat check replays every reply to
    every candidate, so classifying the whole corpus would cost far more
    than the handful of positions a pack needs. The sample is drawn with a
    seeded generator, so a pack can be regenerated exactly.

    Deduplicated on the canonical key. Quantik has 192 symmetries, and two
    puzzles that are reflections of each other are one puzzle shown twice —
    which is what makes a generated set feel worse than a hand-picked one.
    """
    rng = np.random.default_rng(seed)
    total = len(corpus.boards)
    order = rng.permutation(total)[: min(scan, total)]

    found: dict[str, list] = {theme: [] for theme in THEMES}
    seen_keys: set[int] = set()

    for index in order:
        if all(len(found[theme]) >= per_theme for theme in THEMES):
            break
        board = corpus.boards[index]
        key = int(fb.canonical_keys(board.reshape(1, -1))[0])
        if key in seen_keys:
            continue
        puzzle = classify(
            board,
            float(corpus.value_target[index]),
            int(corpus.optimal_mask[index]),
            int(corpus.plies[index]),
        )
        if puzzle is None or len(found[puzzle["theme"]]) >= per_theme:
            continue
        seen_keys.add(key)
        found[puzzle["theme"]].append(puzzle)

    packed = [puzzle for theme in THEMES for puzzle in found[theme]]
    return {
        "schema": SCHEMA,
        "seed": seed,
        "scanned": int(len(order)),
        "counts": {theme: len(found[theme]) for theme in THEMES},
        "puzzles": packed,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    from ..data.exact_corpus import ExactCorpus

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-theme", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--scan", type=int, default=60_000)
    args = parser.parse_args(argv)

    pack = generate(
        ExactCorpus.load(args.corpus),
        per_theme=args.per_theme,
        seed=args.seed,
        scan=args.scan,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(pack, indent=2) + "\n")
    for theme, count in pack["counts"].items():
        print(f"  {theme:<14} {count}")
    print(f"wrote {len(pack['puzzles'])} puzzles to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
