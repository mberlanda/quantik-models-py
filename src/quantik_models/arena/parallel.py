"""Run arena matches across processes.

Games are independent, so the unit of work is one game. Workers rebuild
their agents from specs once per process (see `registry`) and reuse them.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import numpy.typing as npt

from .match import MatchResult, _Timed, play_game
from .registry import build_agent

_WORKER_AGENTS: dict[str, Any] = {}


def _init_worker() -> None:
    """One compute thread per worker, for every agent alike.

    The classical engines are single-threaded Python. Torch defaults to one
    thread per core, so a network agent in each of 16 workers would quietly
    claim many times the CPU its opponent gets — and the reported ms/move
    would be measuring that, not the agent. Pinning makes the head-to-head
    timing an apples-to-apples number.
    """
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        # torch is optional, and set_num_interop_threads raises if the pool
        # has already started — neither is worth failing a match over.
        pass


def _agent_for(spec: dict[str, Any]):
    key = repr(sorted(spec.items(), key=lambda kv: kv[0]))
    if key not in _WORKER_AGENTS:
        _WORKER_AGENTS[key] = build_agent(spec)
    return _WORKER_AGENTS[key]


def _play_one(job) -> tuple[bool, int, float, float, int, int]:
    spec_a, spec_b, board_bytes, seed, a_moves_first = job
    board = np.frombuffer(board_bytes, dtype=np.uint16).copy()
    timed_a = _Timed(_agent_for(spec_a))
    timed_b = _Timed(_agent_for(spec_b))
    mover, responder = (timed_a, timed_b) if a_moves_first else (timed_b, timed_a)
    winner, plies = play_game(mover, responder, board, seed)
    a_won = (winner == 0) == a_moves_first
    return (a_won, plies, timed_a.seconds, timed_b.seconds, timed_a.moves, timed_b.moves)


def play_match_parallel(
    spec_a: dict[str, Any],
    spec_b: dict[str, Any],
    positions: npt.NDArray[np.uint16],
    seeds: tuple[int, ...] = (0,),
    workers: int | None = None,
    progress=None,
) -> MatchResult:
    """Side-balanced match, one game per task."""
    name_a = build_agent(spec_a).name
    name_b = build_agent(spec_b).name
    jobs = [
        (spec_a, spec_b, board.tobytes(), seed, a_first)
        for board in positions
        for seed in seeds
        for a_first in (True, False)
    ]
    result = MatchResult(agent_a=name_a, agent_b=name_b)
    workers = workers or min(os.cpu_count() or 4, 16)
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        for done, (a_won, plies, sec_a, sec_b, mv_a, mv_b) in enumerate(
            pool.map(_play_one, jobs, chunksize=1), start=1
        ):
            result.wins_a += int(a_won)
            result.wins_b += int(not a_won)
            result.games += 1
            result.plies.append(plies)
            result.seconds_a += sec_a
            result.seconds_b += sec_b
            result.moves_a += mv_a
            result.moves_b += mv_b
            if progress is not None:
                progress(done, len(jobs))
    return result
