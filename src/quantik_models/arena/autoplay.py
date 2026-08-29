"""Engine-vs-engine games, recorded as positions worth solving.

Autoplay is usually described as a way to generate *labels*: play a game,
take the result, train on it. That is not what it is for here, and the
distinction matters enough to state up front.

This project already has better labels than any game can produce. The
corpus carries exact policy and value targets from the solver — the true
game-theoretic outcome and the full outcome-optimal action set. A game
result is a far weaker signal: the value is contaminated by both players'
mistakes, and the "policy target" is a single move that may simply be
wrong. The AlphaZero run in this repo already showed what that costs — its
value head learned almost nothing, because the target was a blend of a game
result and its own undertrained estimate.

What autoplay uniquely provides is **positions**. The corpus spans plies
6-13; games played from the opening spend their first moves at plies 0-5,
where there is not one training position and where `shift-evaluation.md`
shows every architecture is at its weakest. Those positions are reachable
in real play, which sampling the canonical space uniformly does not
guarantee.

So the pipeline is: play games, keep the positions, and hand them to the
exact solver. Game outcomes are recorded too — they cost nothing and make
the arena result auditable — but they are provenance, not targets.

    python -m quantik_models.arena.autoplay \\
      --agents runs/arena/lineup-agents.json \\
      --games 200 --out runs/autoplay/lineup
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import permutations
from pathlib import Path

import numpy as np

from ..env import fastboard as fb


@dataclass
class Game:
    """One complete game, keeping every position that was actually played."""

    mover: str
    responder: str
    boards: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    winner: int = -1  # 0 = mover, 1 = responder

    @property
    def plies(self) -> int:
        return len(self.actions)


def play_recorded(mover, responder, board: np.ndarray, seed: int) -> Game:
    """`arena.match.play_game`, but keeping the trajectory.

    Deliberately a separate function rather than an option on the existing
    one: the arena plays tens of thousands of games and has no use for the
    boards, and making it accumulate them would cost memory in the one
    place that cares most about throughput.
    """
    game = Game(mover=mover.name, responder=responder.name)
    current = board.copy()
    turn = 0
    agents = (mover, responder)
    while True:
        done, _ = fb.terminal_status(current[None, :])
        if bool(done[0]):
            game.winner = 1 - turn
            return game
        action = agents[turn].select(current, seed + game.plies)
        legal = fb.legal_masks(current[None, :])[0]
        if not legal[action]:
            raise ValueError(f"{agents[turn].name} chose illegal action {action}")
        game.boards.append(current.copy())
        game.actions.append(int(action))
        current = fb.apply_actions(
            current[None, :], np.array([action], dtype=np.int64)
        )[0]
        turn ^= 1


def positions_from(games: list[Game], max_ply: int | None = None) -> np.ndarray:
    """Every distinct position visited, deduplicated by canonical key.

    Deduplication is on the canonical key rather than the raw board because
    two games reaching the same position by different move orders — or by
    symmetric ones — are one position to solve, and solving is the
    expensive step.
    """
    boards = [b for game in games for b in game.boards]
    if not boards:
        return np.zeros((0, 8), dtype=np.uint16)
    stacked = np.stack(boards)
    if max_ply is not None:
        stacked = stacked[fb.popcount(fb.occupancy(stacked)) <= max_ply]
    if not len(stacked):
        return stacked
    _, first = np.unique(fb.canonical_keys(stacked), return_index=True)
    return stacked[np.sort(first)]


def novel_positions(candidates: np.ndarray, corpus_boards: np.ndarray) -> np.ndarray:
    """Drop anything the training corpus already covers.

    The point of autoplay here is reach, so a position the corpus already
    holds is not worth a solver call.
    """
    if not len(candidates):
        return candidates
    known = np.unique(fb.canonical_keys(corpus_boards))
    keys = fb.canonical_keys(candidates)
    return candidates[~np.isin(keys, known)]


def ply_histogram(boards: np.ndarray) -> dict[int, int]:
    if not len(boards):
        return {}
    return dict(sorted(Counter(fb.popcount(fb.occupancy(boards)).tolist()).items()))


def pairings(names: Iterable[str], against: str | None = None) -> list[tuple[str, str]]:
    """The ordered pairings to play.

    Ordered, not unordered: moving first is a real advantage in Quantik, so
    a pairing that only ever ran one way round would attribute that
    advantage to the agent rather than to the seat.

    `against` restricts the schedule to pairings involving that one agent,
    both ways round. Measuring a field against a common oracle is a
    different experiment from a round robin, and running the full round
    robin to extract it spends most of the budget replaying games that are
    already measured — with four networks and one oracle, 12 of the 20
    ordered pairings are network-versus-network.
    """
    ordered = list(permutations(sorted(names), 2))
    if against is None:
        return ordered
    if against not in set(names):
        raise ValueError(f"no agent named {against!r} among {sorted(names)}")
    return [pair for pair in ordered if against in pair]


def run(
    specs: list[dict],
    games_per_pairing: int,
    *,
    seed: int = 0,
    start_plies: int = 0,
    against: str | None = None,
    progress=None,
) -> list[Game]:
    """Play `games_per_pairing` games over the schedule `pairings` returns."""
    from .registry import build_agent

    agents = {spec.get("name", spec["kind"]): build_agent(dict(spec)) for spec in specs}
    games: list[Game] = []
    rng = np.random.default_rng(seed)
    for mover_name, responder_name in pairings(agents, against):
        for index in range(games_per_pairing):
            board = fb.empty_boards(1)[0]
            for _ in range(start_plies):
                legal = fb.legal_masks(board[None, :])[0]
                actions = np.flatnonzero(legal)
                if not actions.size:
                    break
                choice = int(rng.choice(actions))
                board = fb.apply_actions(
                    board[None, :], np.array([choice], dtype=np.int64)
                )[0]
            games.append(
                play_recorded(
                    agents[mover_name],
                    agents[responder_name],
                    board,
                    seed=seed + 1000 * index,
                )
            )
            if progress is not None:
                progress(len(games))
    return games


def leaderboard(games: list[Game]) -> list[dict]:
    tally: dict[str, list[int]] = {}
    for game in games:
        for name in (game.mover, game.responder):
            tally.setdefault(name, [0, 0])
        winner = game.mover if game.winner == 0 else game.responder
        tally[winner][0] += 1
        tally[game.mover][1] += 1
        tally[game.responder][1] += 1
    return sorted(
        (
            {"agent": n, "wins": w, "games": g, "win_rate": w / g if g else 0.0}
            for n, (w, g) in tally.items()
        ),
        key=lambda r: -r["win_rate"],
    )


def write_qfens(boards: np.ndarray, path: Path) -> Path:
    """The solver's input format: one QFEN per line, on stdin."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(fb.to_qfen(b) for b in boards) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--agents", type=Path, required=True)
    parser.add_argument("--games", type=int, default=50, help="per ordered pairing")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument(
        "--start-plies",
        type=int,
        default=0,
        help="random plies before the engines take over; 0 starts from empty",
    )
    parser.add_argument(
        "--max-solve-ply",
        type=int,
        default=6,
        help="only keep positions this shallow for solving; deeper ones are "
        "already well covered by the corpus",
    )
    parser.add_argument(
        "--against",
        default=None,
        help="restrict play to pairings involving this agent, both ways round",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("runs/oracle/corpus/exact-sampled.npz"),
        help="positions already in here are not worth a solver call",
    )
    args = parser.parse_args(argv)

    specs = json.loads(args.agents.read_text())
    names = [spec.get("name", spec["kind"]) for spec in specs]
    schedule = pairings(names, args.against)
    total = len(schedule) * args.games
    print(f"{len(specs)} agents, {len(schedule)} ordered pairings, {total} games")

    def progress(n: int) -> None:
        if n % max(1, total // 20) == 0:
            print(f"  {n}/{total}", flush=True)

    games = run(
        specs,
        args.games,
        seed=args.seed,
        start_plies=args.start_plies,
        against=args.against,
        progress=progress,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    board = leaderboard(games)
    print("\nleaderboard")
    for row in board:
        print(f"  {row['agent']:<24} {row['win_rate']:.1%}  ({row['wins']}/{row['games']})")

    visited = positions_from(games)
    shallow = positions_from(games, max_ply=args.max_solve_ply)
    print(f"\npositions visited: {len(visited):,} distinct "
          f"({len(shallow):,} at ply <= {args.max_solve_ply})")
    print(f"  by ply: {ply_histogram(visited)}")

    if args.corpus.exists():
        from ..data.exact_corpus import ExactCorpus

        novel = novel_positions(shallow, ExactCorpus.load(args.corpus).boards)
        print(f"  novel (not in {args.corpus.name}): {len(novel):,}")
        print(f"  novel by ply: {ply_histogram(novel)}")
    else:
        novel = shallow

    qfen_path = write_qfens(novel, args.out / "to-solve.qfen")
    (args.out / "games.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "games": len(games),
                "leaderboard": board,
                "results": [
                    {
                        "mover": g.mover,
                        "responder": g.responder,
                        "winner": g.mover if g.winner == 0 else g.responder,
                        "plies": g.plies,
                        "actions": g.actions,
                    }
                    for g in games
                ],
            },
            indent=2,
        )
    )
    print(f"\nwrote {qfen_path} and {args.out / 'games.json'}")
    print(
        "\nNext: label them exactly, then rebuild the corpus.\n"
        f"  ../quantik-core-rust/target/release/examples/exact_oracle "
        f"< {qfen_path} > {args.out / 'solved.jsonl'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
