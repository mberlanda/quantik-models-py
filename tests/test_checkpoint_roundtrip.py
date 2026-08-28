"""A checkpoint has to say enough about itself to be reloaded.

`load_evaluator` used to parse `"resnet-c{channels}-b{blocks}"` out of the
manifest's `architecture` field. That worked while there was one
architecture and broke silently in the worst way once there were three:
`mlp-h455-b4` does not match the pattern, and `cpool-c191-b6` matches it
closely enough to build a *ResNet* of the right width and then fail on the
state dict — or, with unlucky naming, not fail at all.

So the manifest now records the registry name and the config outright, and
these tests cover the round trip for every registered architecture.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from quantik_models.arena.registry import _model_from_manifest  # noqa: E402
from quantik_models.export.checkpoint import export_checkpoint  # noqa: E402
from quantik_models.model import registry  # noqa: E402


@pytest.mark.parametrize("name", registry.architectures())
def test_manifest_rebuilds_the_same_architecture(name: str, tmp_path) -> None:
    preset = "smoke" if "smoke" in registry.presets(name) else registry.presets(name)[0]
    model = registry.build(name, preset=preset)
    manifest = json.loads(
        export_checkpoint(
            model, out_dir=tmp_path, model_id=f"{name}-test", training_report={}
        ).read_text()
    )

    assert manifest["architecture_spec"]["arch"] == name
    rebuilt = _model_from_manifest(manifest, registry)
    assert rebuilt.architecture == model.architecture
    assert rebuilt.model_family == model.model_family

    # The real test: the weights have to load into it.
    from safetensors.torch import load_file

    rebuilt.load_state_dict(load_file(str(tmp_path / "weights.safetensors")))


@pytest.mark.parametrize("name", registry.architectures())
def test_rebuilt_model_computes_the_same_function(name: str, tmp_path) -> None:
    preset = "smoke" if "smoke" in registry.presets(name) else registry.presets(name)[0]
    torch.manual_seed(0)
    model = registry.build(name, preset=preset).eval()
    manifest = json.loads(
        export_checkpoint(
            model, out_dir=tmp_path, model_id=f"{name}-test", training_report={}
        ).read_text()
    )

    from safetensors.torch import load_file

    rebuilt = _model_from_manifest(manifest, registry)
    rebuilt.load_state_dict(load_file(str(tmp_path / "weights.safetensors")))
    rebuilt.eval()

    sample = torch.randn(4, 9, 4, 4)
    with torch.no_grad():
        want = model(sample)
        got = rebuilt(sample)
    torch.testing.assert_close(got[0], want[0])
    torch.testing.assert_close(got[1], want[1])


def test_legacy_resnet_manifest_still_loads() -> None:
    """Checkpoints written before `architecture_spec` are all ResNets."""
    model = _model_from_manifest({"architecture": "resnet-c64-b4"}, registry)
    assert model.architecture == "resnet-c64-b4"


def test_legacy_manifest_for_a_non_resnet_is_refused_loudly() -> None:
    """The one case that must not guess.

    Silently building a ResNet for an `mlp-` manifest would either fail on
    the state dict with an unhelpful error, or — worse — succeed.
    """
    with pytest.raises(ValueError, match="architecture_spec"):
        _model_from_manifest({"architecture": "mlp-h455-b4"}, registry)
