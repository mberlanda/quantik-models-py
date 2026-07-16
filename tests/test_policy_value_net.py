from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from quantik_models.model.policy_value_net import (  # noqa: E402
    PRESETS,
    PolicyValueNet,
    PolicyValueNetConfig,
    masked_log_softmax,
    parameter_count,
)


def test_forward_shapes_and_value_range() -> None:
    model = PolicyValueNet(PRESETS["smoke"])
    x = torch.rand(5, 9, 4, 4)
    logits, value = model(x)
    assert logits.shape == (5, 64)
    assert value.shape == (5,)
    assert value.abs().max().item() <= 1.0


def test_masked_log_softmax_zeroes_illegal() -> None:
    logits = torch.zeros(2, 64, requires_grad=True)
    mask = torch.zeros(2, 64, dtype=torch.bool)
    mask[0, :4] = True
    mask[1, 63] = True
    logp = masked_log_softmax(logits, mask)
    probs = logp.exp()
    assert torch.allclose(probs[0, :4], torch.full((4,), 0.25), atol=1e-6)
    assert probs[0, 4:].max().item() == pytest.approx(0.0, abs=1e-6)
    assert probs[1, 63].item() == pytest.approx(1.0, abs=1e-6)
    # masked entries must not produce NaN gradients
    loss = (probs[0, :4]).sum()
    loss.backward()


def test_preset_sizes() -> None:
    smoke = parameter_count(PolicyValueNet(PRESETS["smoke"]))
    small = parameter_count(PolicyValueNet(PRESETS["small"]))
    target = parameter_count(PolicyValueNet(PRESETS["target"]))
    assert smoke < 100_000
    assert small < 2_000_000
    # 4 bytes/param must land inside the 50-100 MB contract envelope
    assert 50 * 2**20 <= target * 4 <= 100 * 2**20


def test_config_round_trip() -> None:
    cfg = PolicyValueNetConfig(channels=16, blocks=2)
    model = PolicyValueNet(cfg)
    assert model.config == cfg
    assert math.isfinite(parameter_count(model))
