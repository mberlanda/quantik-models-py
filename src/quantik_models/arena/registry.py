"""Build agents from plain-dict specs so matches can be farmed out to workers.

Agents hold engines, RNGs, and (for the network) torch modules, none of which
survive pickling cleanly. A spec is a small JSON-able dict; workers rebuild
the agent locally from it. This is also the format the arena reports and
run manifests record, so a result always names exactly what played.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .agents import (
    BeamAgent,
    CoreMCTSAgent,
    MinimaxAgent,
    NetMCTSAgent,
    PolicyAgent,
    RandomAgent,
)

_CACHE: dict[str, Any] = {}

# A checkpoint directory names its weights one of two ways, and both are
# canonical somewhere:
#
# * `weights.safetensors` is what `export.checkpoint` writes and what every
#   `runs/train/*/best` on disk holds.
# * `model.safetensors` is what `export.huggingface.stage` renames it to,
#   because the Hub's viewers and half its tooling look for that name.
#
# Resolving both is not leniency for its own sake: the four published model
# cards tell a reader to `load_evaluator(snapshot_download(repo_id))`, and a
# downloaded Hub repo is the *second* layout. Accepting only the first made
# that snippet raise `FileNotFoundError` on every published model.
_WEIGHT_FILENAMES = ("weights.safetensors", "model.safetensors")


def weights_path(checkpoint: str | Path) -> Path:
    """The safetensors file inside a checkpoint directory.

    Raises `FileNotFoundError` naming both candidates, because "no such file
    or directory: .../weights.safetensors" sends a reader looking for a file
    that was never supposed to be there.
    """
    path = Path(checkpoint)
    for filename in _WEIGHT_FILENAMES:
        candidate = path / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"no safetensors weights in {path}: expected one of "
        f"{', '.join(_WEIGHT_FILENAMES)}"
    )


def load_evaluator(checkpoint: str | Path, device: str = "cpu", batch_size: int = 4096):
    """Load a `model-checkpoint.v1` directory into a `NetEvaluator` (cached).

    Accepts both a local `runs/train/*/best` directory and the path returned
    by `huggingface_hub.snapshot_download` for a published Quantik model.

    Workers replay the same spec for every game, so caching keeps the weights
    off the critical path after the first call in each process.
    """
    import json

    from safetensors.torch import load_file
    import torch

    from ..model import registry as model_registry
    from ..selfplay.evaluator import NetEvaluator

    key = f"{checkpoint}|{device}"
    if key in _CACHE:
        return _CACHE[key]
    path = Path(checkpoint)

    manifest = json.loads((path / "manifest.json").read_text())
    model = _model_from_manifest(manifest, model_registry)
    model.load_state_dict(load_file(str(weights_path(path))))
    resolved = torch.device(device)
    evaluator = NetEvaluator(model, resolved, batch_size=batch_size)
    _CACHE[key] = evaluator
    return evaluator


def load_onnx_evaluator(checkpoint: str | Path, batch_size: int = 4096):
    """Load a `model-checkpoint.v1` directory's `model.onnx` into an
    `OnnxEvaluator` (cached, torch-free).

    Deliberately does not share `load_evaluator`'s cache key or code path:
    the two runtimes can be loaded side by side (the agreement test does
    exactly that), and neither import statement runs unless its runtime is
    actually chosen.
    """
    from ..selfplay.evaluator import OnnxEvaluator

    key = f"onnx|{checkpoint}"
    if key in _CACHE:
        return _CACHE[key]
    evaluator = OnnxEvaluator(Path(checkpoint) / "model.onnx", batch_size=batch_size)
    _CACHE[key] = evaluator
    return evaluator


def _model_from_manifest(manifest: dict[str, Any], model_registry):
    """Rebuild the architecture a checkpoint was trained with.

    Prefers `architecture_spec`, which records the registry name and the
    config outright. Falls back to parsing the human-readable
    `architecture` string for checkpoints written before that field
    existed — all of which are ResNets, because it was the only
    architecture at the time.
    """
    spec = manifest.get("architecture_spec")
    if spec is not None:
        return model_registry.build_from_spec(spec)

    architecture = manifest["architecture"]
    if not architecture.startswith("resnet-c"):
        raise ValueError(
            f"checkpoint records architecture {architecture!r} but no "
            "`architecture_spec`; re-export it with a current "
            "quantik-models to make it loadable"
        )
    channels, blocks = architecture.removeprefix("resnet-c").split("-b")
    return model_registry.build(
        "resnet", preset="small", channels=int(channels), blocks=int(blocks)
    )


def build_agent(spec: dict[str, Any]):
    """Instantiate one agent from its spec dict."""
    spec = dict(spec)
    kind = spec.pop("kind")
    name = spec.pop("name", None)
    if kind == "random":
        return RandomAgent(name=name or "random")
    if kind == "minimax":
        return MinimaxAgent(name=name, **spec)
    if kind == "mcts":
        return CoreMCTSAgent(name=name, **spec)
    if kind == "beam":
        return BeamAgent(name=name, **spec)
    if kind == "uniform-mcts":
        # The control the network agents are missing: the *same* PUCT
        # search, at the same budget, with the network replaced by uniform
        # priors and a value of zero. `mcts` is a different algorithm
        # entirely (UCB1 with random rollouts, from quantik-core), so it
        # cannot answer "how much is the network contributing" — this can,
        # because everything except the evaluator is held fixed.
        from ..selfplay.evaluator import UniformEvaluator
        from ..selfplay.mcts import MCTSParams

        params = MCTSParams(**spec.pop("params", {"simulations": 128}))
        return NetMCTSAgent(
            UniformEvaluator(), params=params, name=name or "uniform-mcts", **spec
        )
    if kind in {"net-policy", "net-mcts"}:
        checkpoint = spec.pop("checkpoint")
        device = spec.pop("device", "cpu")
        batch_size = spec.pop("eval_batch_size", 4096)
        runtime = spec.pop("runtime", "torch")
        if runtime == "torch":
            evaluator = load_evaluator(checkpoint, device, batch_size)
        elif runtime == "onnx":
            evaluator = load_onnx_evaluator(checkpoint, batch_size)
        else:
            raise ValueError(f"unknown evaluator runtime {runtime!r}; want 'torch' or 'onnx'")
        if kind == "net-policy":
            return PolicyAgent(evaluator, name=name or "net-policy", **spec)
        from ..selfplay.mcts import MCTSParams

        params = MCTSParams(**spec.pop("params", {"simulations": spec.pop("simulations", 128)}))
        return NetMCTSAgent(evaluator, params=params, name=name, **spec)
    raise ValueError(f"unknown agent kind {kind!r}")


BASELINE_SPECS: dict[str, dict[str, Any]] = {
    "random": {"kind": "random"},
    "minimax": {"kind": "minimax"},
    "mcts": {"kind": "mcts"},
    "beam": {"kind": "beam"},
}


def fixed_time_baselines(time_limit_s: float) -> list[dict[str, Any]]:
    """The four incumbent strategies, each on the same per-move clock.

    `random` ignores the clock by nature; it is the floor of the table.
    """
    return [
        {"kind": "random", "name": "random"},
        {
            "kind": "minimax",
            "time_limit_s": time_limit_s,
            "name": f"minimax@{time_limit_s * 1000:.0f}ms",
        },
        {
            "kind": "mcts",
            "time_limit_s": time_limit_s,
            "max_iterations": 1_000_000,
            "name": f"mcts@{time_limit_s * 1000:.0f}ms",
        },
        {
            "kind": "beam",
            "time_limit_s": time_limit_s,
            "beam_width": 64,
            "name": f"beam@{time_limit_s * 1000:.0f}ms",
        },
    ]
