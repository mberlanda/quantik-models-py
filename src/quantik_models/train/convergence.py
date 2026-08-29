"""When to stop training.

Its own module, importing nothing heavy, because the torch-free install is a
tested configuration in this project: `supervised.py` imports torch at module
scope, so a stopping rule living there could only be tested where torch is
installed — and the rule is the part most worth testing everywhere.
"""

from __future__ import annotations

from collections.abc import Sequence


def epochs_since_best(history: Sequence[float]) -> int:
    """How many epochs have passed since the lowest value in `history`.

    Ties count as "no improvement", matching the checkpoint rule one line
    away in the training loop: `best/` is only rewritten on a strict
    decrease, so an epoch that merely equals the best did not produce the
    weights on disk and should not buy more epochs either.
    """
    if not history:
        return 0
    best = min(range(len(history)), key=history.__getitem__)
    return len(history) - 1 - best
