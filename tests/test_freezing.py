"""Freezing has to actually freeze, and say so when it cannot.

Both failure modes here are silent: a pattern that matches nothing trains
normally and looks like a working fine-tune, and a "frozen" batch norm in
training mode keeps updating its running statistics so the module computes
a different function after one epoch with nothing in the loss to show it.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from quantik_models.model import registry  # noqa: E402
from quantik_models.train import freezing  # noqa: E402


@pytest.mark.parametrize("name", registry.architectures())
def test_freezing_the_stem_stops_its_gradients(name: str) -> None:
    model = registry.build(name, preset="smoke")
    report = freezing.freeze(model, ["stem"])
    assert report.frozen_parameters > 0
    assert report.trainable_parameters > 0

    policy, value = model(torch.zeros(4, 9, 4, 4))
    (policy.sum() + value.sum()).backward()

    for tensor in report.frozen_tensors:
        param = dict(model.named_parameters())[tensor]
        assert param.grad is None, f"{tensor} received a gradient while frozen"
    assert any(p.grad is not None for p in freezing.trainable_parameters(model))


@pytest.mark.parametrize("name", registry.architectures())
def test_counts_add_up(name: str) -> None:
    model = registry.build(name, preset="smoke")
    report = freezing.freeze(model, ["stem"])
    assert report.total_parameters == sum(p.numel() for p in model.parameters())


def test_a_pattern_matching_nothing_is_refused() -> None:
    """The silent failure: freeze nothing, train normally, look successful."""
    with pytest.raises(ValueError, match="match no parameter"):
        freezing.freeze(registry.build("cpool", preset="smoke"), ["trunk"])


def test_the_error_names_the_modules_that_do_exist() -> None:
    with pytest.raises(ValueError, match="blocks"):
        freezing.freeze(registry.build("cpool", preset="smoke"), ["trunk"])


def test_prefix_matching_does_not_match_a_partial_name() -> None:
    """`stem` must not match a hypothetical `stem_extra`."""
    assert freezing._matches("stem.0.weight", ("stem",))
    assert freezing._matches("stem", ("stem",))
    assert not freezing._matches("stemx.0.weight", ("stem",))


def test_a_nested_prefix_freezes_one_block_only() -> None:
    model = registry.build("resnet", preset="small")
    report = freezing.freeze(model, ["trunk.0"])
    assert all(t.startswith("trunk.0.") for t in report.frozen_tensors)
    assert report.frozen_parameters < sum(p.numel() for p in model.parameters())


def test_frozen_batch_norm_stops_tracking() -> None:
    """The bug this module exists to prevent.

    `requires_grad = False` does not stop a BatchNorm in training mode from
    updating its running mean and variance. A trunk that is still tracking
    is not frozen — and after one epoch it computes a different function,
    with nothing in the loss to say so.
    """
    model = registry.build("resnet", preset="smoke")
    report = freezing.freeze(model, ["stem"])
    norms = freezing.frozen_norm_modules(model, report)
    assert norms, "the resnet stem has a batch norm; it must be held"

    freezing.set_train_mode(model, norms)
    stem_norm = norms[0]
    assert isinstance(stem_norm, nn.BatchNorm2d)
    before = stem_norm.running_mean.clone()

    torch.manual_seed(0)
    model(torch.randn(32, 9, 4, 4))
    torch.testing.assert_close(stem_norm.running_mean, before)


def test_plain_train_would_have_broken_it() -> None:
    """Guards the guard: `model.train()` really does undo the eval state."""
    model = registry.build("resnet", preset="smoke")
    report = freezing.freeze(model, ["stem"])
    norms = freezing.frozen_norm_modules(model, report)

    freezing.set_train_mode(model, norms)
    assert not norms[0].training
    model.train()  # what a caller would write by accident
    assert norms[0].training

    before = norms[0].running_mean.clone()
    torch.manual_seed(0)
    model(torch.randn(32, 9, 4, 4))
    assert not torch.allclose(norms[0].running_mean, before)


def test_unfrozen_norms_still_track() -> None:
    """Freezing must not accidentally stop the parts still being trained."""
    model = registry.build("resnet", preset="smoke")
    report = freezing.freeze(model, ["stem"])
    freezing.set_train_mode(model, freezing.frozen_norm_modules(model, report))

    trunk_norm = model.trunk[0].bn1
    before = trunk_norm.running_mean.clone()
    torch.manual_seed(0)
    model(torch.randn(32, 9, 4, 4))
    assert not torch.allclose(trunk_norm.running_mean, before)


def test_no_patterns_freezes_nothing() -> None:
    model = registry.build("resnet", preset="smoke")
    report = freezing.freeze(model, [])
    assert report.frozen_parameters == 0
    assert all(p.requires_grad for p in model.parameters())
    assert "nothing frozen" in report.summary()
