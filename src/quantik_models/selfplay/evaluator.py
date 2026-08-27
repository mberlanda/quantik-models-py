"""Position evaluators feeding the batched MCTS.

An evaluator maps a batch of raw `(n, 8) uint16` boards to
`(policy_priors, values)`, where priors are already restricted to legal
actions and values are from the side-to-move's perspective. Keeping this
behind a protocol lets the search run against a random-play baseline
(`UniformEvaluator`) with no torch import, which is how the search itself
is tested independently of any trained weights.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt

from ..env import fastboard as fb

Boards = npt.NDArray[np.uint16]
Priors = npt.NDArray[np.float32]
Values = npt.NDArray[np.float32]


class Evaluator(Protocol):
    def __call__(self, boards: Boards, legal: npt.NDArray[np.bool_]) -> tuple[Priors, Values]:
        ...


class UniformEvaluator:
    """Uniform priors over legal actions and a value of 0.

    With this evaluator PUCT reduces to visit-count-driven exploration, so
    the search behaves like plain MCTS without rollouts — useful as a
    control and for testing the search machinery.
    """

    def __call__(self, boards: Boards, legal: npt.NDArray[np.bool_]) -> tuple[Priors, Values]:
        counts = legal.sum(axis=1, keepdims=True).clip(min=1)
        priors = (legal.astype(np.float32) / counts).astype(np.float32)
        return priors, np.zeros(boards.shape[0], dtype=np.float32)


class NetEvaluator:
    """Policy/value network evaluator with legality-masked softmax.

    Per `model-checkpoint.v1`, legal-action masking lives outside the model,
    so it is applied here — the same masking the training loss uses.
    """

    def __init__(self, model, device, batch_size: int = 4096) -> None:
        import torch

        self._torch = torch
        self.model = model.to(device).eval()
        self.device = device
        self.batch_size = batch_size

    def __call__(self, boards: Boards, legal: npt.NDArray[np.bool_]) -> tuple[Priors, Values]:
        torch = self._torch
        n = boards.shape[0]
        if n == 0:
            return (
                np.zeros((0, fb.ACTION_COUNT), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
            )
        priors = np.empty((n, fb.ACTION_COUNT), dtype=np.float32)
        values = np.empty(n, dtype=np.float32)
        with torch.no_grad():
            for start in range(0, n, self.batch_size):
                stop = min(start + self.batch_size, n)
                x = torch.from_numpy(fb.encode_tensors(boards[start:stop])).to(self.device)
                mask = torch.from_numpy(legal[start:stop]).to(self.device)
                logits, value = self.model(x)
                logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
                priors[start:stop] = torch.softmax(logits, dim=-1).cpu().numpy()
                values[start:stop] = value.float().cpu().numpy()
        return priors, values
