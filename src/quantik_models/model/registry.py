"""The architecture registry: one name, one constructor, one preset table.

Training, export and evaluation all need to turn a short string into a
model, and a model back into a string that a `model-checkpoint.v1` manifest
can record. Keeping that mapping in one place is what lets a checkpoint be
loaded without the caller knowing in advance which architecture produced
it — the manifest names it, the registry resolves it.

Every registered architecture satisfies the same contract, defined by
`spec`: it consumes `(B, 9, 4, 4)` float32 and returns
`(policy_logits[B, 64], value[B])` with the value already through a tanh.
That is what makes them substitutable behind one evaluator, and comparable
to each other on the same corpus.

Torch-only module: import it behind the `[torch]` extra.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Protocol

from torch import nn

from . import constraint_pool_net, mlp_net, policy_value_net


class PolicyValueModel(Protocol):
    """What every architecture in this package promises.

    `architecture` and `model_family` are read straight into the checkpoint
    manifest, so they are part of the published contract rather than
    incidental metadata.
    """

    @property
    def architecture(self) -> str: ...

    @property
    def model_family(self) -> str: ...


@dataclass(frozen=True)
class ArchitectureEntry:
    build: Callable[[Any], nn.Module]
    config_type: type
    presets: dict[str, Any]
    summary: str


_REGISTRY: dict[str, ArchitectureEntry] = {
    "resnet": ArchitectureEntry(
        build=policy_value_net.PolicyValueNet,
        config_type=policy_value_net.PolicyValueNetConfig,
        presets=policy_value_net.PRESETS,
        summary="Convolutional residual trunk; the project's incumbent.",
    ),
    "mlp": ArchitectureEntry(
        build=mlp_net.MLPNet,
        config_type=mlp_net.MLPNetConfig,
        presets=mlp_net.PRESETS,
        summary="Flattened dense trunk; the control for whether 4x4 spatial "
        "structure is worth modelling at all.",
    ),
    "cpool": ArchitectureEntry(
        build=constraint_pool_net.ConstraintPoolNet,
        config_type=constraint_pool_net.ConstraintPoolNetConfig,
        presets=constraint_pool_net.PRESETS,
        summary="Message passing over Quantik's twelve constraint groups; "
        "the game's rule structure written into the wiring.",
    ),
}


def register(name: str, entry: ArchitectureEntry) -> None:
    """Add an architecture. Re-registering a name is a programming error."""
    if name in _REGISTRY:
        raise ValueError(f"architecture already registered: {name}")
    _REGISTRY[name] = entry


def architectures() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def entry(name: str) -> ArchitectureEntry:
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(architectures())
        raise ValueError(f"unknown architecture {name!r}; known: {known}") from None


def presets(name: str) -> tuple[str, ...]:
    return tuple(sorted(entry(name).presets))


def name_for(model: nn.Module) -> str:
    """The registry name that built this model.

    Keyed on the config type, which is one-to-one with an architecture, so
    a checkpoint can record how to rebuild itself without anyone having to
    keep a second mapping in sync.
    """
    config_type = type(getattr(model, "config"))
    for name, spec in _REGISTRY.items():
        if spec.config_type is config_type:
            return name
    raise ValueError(f"no registered architecture builds {config_type.__name__}")


def spec_for(model: nn.Module) -> dict[str, Any]:
    """`{"arch": ..., "config": {...}}` — enough to rebuild this model.

    Written into the checkpoint manifest so a loader does not have to parse
    the human-readable `architecture` string. That string is for people;
    `resnet-c128-b6` and `mlp-h455-b4` do not share a grammar, and a loader
    that tried to parse both would be guessing.
    """
    return {"arch": name_for(model), "config": asdict(getattr(model, "config"))}


def build_from_spec(spec: dict[str, Any]) -> nn.Module:
    """Rebuild a model from the spec `spec_for` produced."""
    entry_ = entry(spec["arch"])
    return entry_.build(entry_.config_type(**spec["config"]))


def build(name: str, *, preset: str | None = None, **overrides: Any) -> nn.Module:
    """Construct an architecture from a preset, with optional overrides.

    Overrides are applied on top of the preset so that an ablation can vary
    one dimension — width, depth, head count — without restating the rest.
    """
    spec = entry(name)
    if preset is not None and preset not in spec.presets:
        known = ", ".join(sorted(spec.presets))
        raise ValueError(f"unknown preset {preset!r} for {name}; known: {known}")
    base = spec.presets[preset] if preset is not None else None

    if base is None and not overrides:
        raise ValueError(f"{name}: pass a preset or explicit config overrides")

    fields = {} if base is None else asdict(base)
    fields.update({k: v for k, v in overrides.items() if v is not None})
    return spec.build(spec.config_type(**fields))
