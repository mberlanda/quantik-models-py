"""Presets that claim to be comparable have to actually be comparable.

The architecture comparison is only readable if a difference in accuracy
is attributable to shape. A control that quietly carries twice the
capacity of the incumbent measures capacity, not architecture — and the
first draft of the MLP presets did exactly that, at 2x, because dense
layers scale as `2 * blocks * hidden^2` and the widths had been chosen by
eye. This test is the guard against choosing them by eye again.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from quantik_models.model import registry  # noqa: E402
from quantik_models.model.policy_value_net import parameter_count  # noqa: E402

# The two reference sizes every other architecture matches against: the
# published `resnet-c128-b6`, and the smaller `resnet-c64-b4` it was scaled
# up from.
REFERENCES = {
    "small": 304_711,  # resnet-c64-b4
    "medium": 1_786_823,  # resnet-c128-b6
}

TOLERANCE = 0.05


def _matched_presets() -> list[tuple[str, str, int]]:
    cases = []
    for arch in registry.architectures():
        for preset in registry.presets(arch):
            if preset in REFERENCES:
                cases.append((arch, preset, REFERENCES[preset]))
    return cases


def test_the_resnet_references_are_still_what_we_think() -> None:
    """If the ResNet's own counts drift, every match above is stale."""
    assert parameter_count(registry.build("resnet", preset="small")) == 304_711
    assert (
        parameter_count(registry.build("resnet", preset="small", channels=128, blocks=6))
        == 1_786_823
    )


@pytest.mark.parametrize("arch,preset,reference", _matched_presets())
def test_matched_presets_are_within_tolerance(
    arch: str, preset: str, reference: int
) -> None:
    count = parameter_count(registry.build(arch, preset=preset))
    drift = count / reference - 1
    assert abs(drift) <= TOLERANCE, (
        f"{arch}/{preset} has {count:,} parameters against the reference "
        f"{reference:,} ({drift:+.1%}); re-solve the preset width rather "
        f"than widening the tolerance"
    )
