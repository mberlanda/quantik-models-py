"""`action_index = shape * 16 + position`, and the trap in getting there.

An architecture that emits per-cell logits has to transpose before
flattening. `position * 4 + shape` produces a tensor of exactly the right
shape and dtype containing exactly the right values in the wrong order —
so it survives every casual check, and the network trains against a
permuted target.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from quantik_models.model.constraint_pool_net import (  # noqa: E402
    flatten_cell_shape_logits,
)
from quantik_models.model.spec import (  # noqa: E402
    ACTION_COUNT,
    CELL_COUNT,
    SHAPE_COUNT,
)


def test_per_cell_logits_flatten_to_contract_action_order() -> None:
    # A value that encodes where it came from: cell c, shape s -> c * 10 + s.
    per_cell = torch.tensor(
        [[[c * 10 + s for s in range(SHAPE_COUNT)] for c in range(CELL_COUNT)]],
        dtype=torch.float32,
    )
    flat = flatten_cell_shape_logits(per_cell)
    assert flat.shape == (1, ACTION_COUNT)

    for shape in range(SHAPE_COUNT):
        for cell in range(CELL_COUNT):
            action = shape * CELL_COUNT + cell
            assert flat[0, action].item() == cell * 10 + shape


def test_the_naive_flatten_would_have_been_wrong() -> None:
    """Guards the guard: the trap this protects against is real."""
    torch.manual_seed(0)
    per_cell = torch.randn(1, CELL_COUNT, SHAPE_COUNT)
    correct = flatten_cell_shape_logits(per_cell)
    naive = per_cell.reshape(1, ACTION_COUNT)
    assert not torch.allclose(correct, naive)
    # Same multiset, different order — which is exactly why the mistake
    # survives every shape, dtype and range check anyone would write.
    assert np.allclose(np.sort(correct.numpy(), axis=1), np.sort(naive.numpy(), axis=1))


def test_the_batch_dimension_survives() -> None:
    per_cell = torch.randn(5, CELL_COUNT, SHAPE_COUNT)
    assert flatten_cell_shape_logits(per_cell).shape == (5, ACTION_COUNT)
