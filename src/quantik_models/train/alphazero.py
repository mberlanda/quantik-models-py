"""AlphaZero training loop for Quantik: self-play, learn, gate, repeat.

Every iteration writes a full checkpoint and appends a metrics line, so a run
can be killed at any point and resumed from `runs/<name>/state.json` with no
loss beyond the iteration in flight.

Three details are Quantik-specific rather than stock AlphaZero:

* **Value blending.** Games last ~8 plies, so the final result `z` is an
  extremely coarse label for an opening position. The value target is
  `(1 - lambda) * z + lambda * q`, mixing in the search's own backed-up root
  value.
* **Symmetry augmentation.** The rules are invariant under 192 symmetries
  (8 dihedral x 24 shape relabelings), so each row is replayed under several
  of them — free data, and it stops the net memorizing board orientation.
* **Gating against the incumbent.** A new generation only becomes the
  self-play actor if it beats the current best head-to-head, which keeps a
  bad iteration from poisoning the replay buffer.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from ..env import fastboard as fb
from ..export.checkpoint import export_checkpoint
from ..model.policy_value_net import (
    PRESETS,
    PolicyValueNet,
    PolicyValueNetConfig,
    masked_log_softmax,
    parameter_count,
)
from ..selfplay.evaluator import NetEvaluator
from ..selfplay.generate import SelfPlayConfig, augment, play_batch
from ..selfplay.mcts import MCTSParams
from .provenance import capture as capture_provenance


@dataclass
class AlphaZeroConfig:
    name: str = "az"
    preset: str = "small"
    channels: int | None = None
    blocks: int | None = None
    iterations: int = 40
    games_per_iteration: int = 256
    simulations: int = 96
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.6
    dirichlet_weight: float = 0.25
    fpu_reduction: float = 0.2
    temperature_plies: int = 6
    augment_factor: int = 4
    buffer_generations: int = 8
    train_steps: int = 300
    batch_size: int = 512
    lr: float = 2e-3
    weight_decay: float = 1e-4
    value_loss_weight: float = 1.0
    # Weight on the search's root value vs the final game result.
    value_blend: float = 0.5
    device: str = "auto"
    seed: int = 20260827
    # Gating: a new net replaces the self-play actor only above this win rate.
    gate_games: int = 96
    gate_threshold: float = 0.55
    gate_simulations: int = 64
    eval_every: int = 5

    def net_config(self) -> PolicyValueNetConfig:
        if self.channels is not None and self.blocks is not None:
            return PolicyValueNetConfig(channels=self.channels, blocks=self.blocks)
        return PRESETS[self.preset]

    def mcts_params(self, simulations: int | None = None) -> MCTSParams:
        return MCTSParams(
            simulations=simulations or self.simulations,
            c_puct=self.c_puct,
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_weight=self.dirichlet_weight,
            fpu_reduction=self.fpu_reduction,
        )


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ReplayBuffer:
    """Fixed number of recent self-play generations, sampled uniformly."""

    def __init__(self, generations: int) -> None:
        self._chunks: deque[dict[str, np.ndarray]] = deque(maxlen=generations)

    def add(self, boards, policies, values) -> None:
        self._chunks.append({"boards": boards, "policies": policies, "values": values})

    def __len__(self) -> int:
        return sum(c["boards"].shape[0] for c in self._chunks)

    def sample(self, size: int, rng: np.random.Generator):
        boards = np.concatenate([c["boards"] for c in self._chunks])
        policies = np.concatenate([c["policies"] for c in self._chunks])
        values = np.concatenate([c["values"] for c in self._chunks])
        idx = rng.integers(0, boards.shape[0], size=size)
        return boards[idx], policies[idx], values[idx]


def _batch_loss(
    model: PolicyValueNet,
    boards: np.ndarray,
    policies: np.ndarray,
    values: np.ndarray,
    device: torch.device,
    value_loss_weight: float,
) -> tuple[Tensor, float, float, float]:
    x = torch.from_numpy(fb.encode_tensors(boards)).to(device)
    target_p = torch.from_numpy(policies).to(device)
    target_v = torch.from_numpy(values).to(device)
    mask = torch.from_numpy(fb.legal_masks(boards)).to(device)

    logits, value = model(x)
    logp = masked_log_softmax(logits, mask)
    policy_loss = -(target_p * logp).sum(dim=-1).mean()
    value_loss = ((value - target_v) ** 2).mean()
    total = policy_loss + value_loss_weight * value_loss
    with torch.no_grad():
        top1 = (logp.argmax(dim=-1) == target_p.argmax(dim=-1)).float().mean().item()
    return total, policy_loss.item(), value_loss.item(), top1


def _gate(
    challenger: PolicyValueNet,
    incumbent: PolicyValueNet,
    config: AlphaZeroConfig,
    device: torch.device,
    rng: np.random.Generator,
) -> float:
    """Side-balanced match between two nets; returns the challenger's win rate.

    Both sides play from the same sampled openings with the same seeds, and
    each opening is played twice with the roles swapped, so the result is not
    an artifact of which side a position favours.
    """
    from ..arena.match import sample_start_positions
    from ..selfplay.duel import duel

    positions = sample_start_positions(
        max(1, config.gate_games // 2), plies=4, seed=int(rng.integers(1 << 30))
    )
    return duel(
        NetEvaluator(challenger, device),
        NetEvaluator(incumbent, device),
        config.mcts_params(config.gate_simulations),
        positions,
        rng,
    ).score_a


def train(config: AlphaZeroConfig, out_root: Path, resume: bool = True) -> Path:
    run_dir = out_root / config.name
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "state.json"
    metrics_path = run_dir / "metrics.jsonl"

    device = resolve_device(config.device)
    # The resolved device, not "auto" — see supervised.train for why.
    (run_dir / "config.json").write_text(
        json.dumps(asdict(config) | {"device": str(device)}, indent=2)
    )
    # Self-play has no corpus file; the rest of the record applies unchanged.
    (run_dir / "provenance.json").write_text(
        json.dumps(capture_provenance(device=str(device)), indent=2)
    )
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    model = PolicyValueNet(config.net_config()).to(device)
    best = PolicyValueNet(config.net_config()).to(device)
    best.load_state_dict(model.state_dict())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    start_iteration = 0
    if resume and state_path.exists():
        state = json.loads(state_path.read_text())
        blob = torch.load(run_dir / "latest.pt", map_location=device, weights_only=True)
        model.load_state_dict(blob["model"])
        best.load_state_dict(blob["best"])
        optimizer.load_state_dict(blob["optimizer"])
        start_iteration = state["iteration"]
        rng = np.random.default_rng(config.seed + start_iteration)
        print(f"resumed {config.name} at iteration {start_iteration}", flush=True)

    buffer = ReplayBuffer(config.buffer_generations)
    print(
        f"{config.name}: {parameter_count(model):,} params on {device}, "
        f"{config.iterations} iterations",
        flush=True,
    )

    for iteration in range(start_iteration, config.iterations):
        started = time.perf_counter()
        # --- self-play with the current best actor -----------------------
        actor = NetEvaluator(best, device)
        sp_config = SelfPlayConfig(
            games=config.games_per_iteration,
            mcts=config.mcts_params(),
            temperature_plies=config.temperature_plies,
        )
        data = play_batch(actor, sp_config, rng)
        selfplay_seconds = time.perf_counter() - started

        boards, policies, outcomes, root_values = augment(
            data, config.augment_factor, rng
        )
        targets = (
            (1.0 - config.value_blend) * outcomes + config.value_blend * root_values
        ).astype(np.float32)
        buffer.add(boards, policies, targets)

        # --- learn -------------------------------------------------------
        model.train()
        train_started = time.perf_counter()
        losses = []
        for _ in range(config.train_steps):
            b, p, v = buffer.sample(config.batch_size, rng)
            loss, policy_loss, value_loss, top1 = _batch_loss(
                model, b, p, v, device, config.value_loss_weight
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append((policy_loss, value_loss, top1))
        train_seconds = time.perf_counter() - train_started
        mean_policy, mean_value, mean_top1 = (float(np.mean(x)) for x in zip(*losses))

        # --- gate --------------------------------------------------------
        model.eval()
        gate_score = _gate(model, best, config, device, rng)
        promoted = gate_score >= config.gate_threshold
        if promoted:
            best.load_state_dict(model.state_dict())

        record: dict[str, Any] = {
            "iteration": iteration,
            "policy_loss": mean_policy,
            "value_loss": mean_value,
            "policy_top1": mean_top1,
            "gate_score": gate_score,
            "promoted": promoted,
            "buffer_rows": len(buffer),
            "selfplay_seconds": selfplay_seconds,
            "train_seconds": train_seconds,
            "total_seconds": time.perf_counter() - started,
            **data.stats(),
        }
        with metrics_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        print(
            f"[{config.name} {iteration:03d}] "
            f"policy={mean_policy:.3f} value={mean_value:.3f} top1={mean_top1:.1%} "
            f"gate={gate_score:.1%}{' PROMOTED' if promoted else ''} "
            f"rows={len(buffer):,} "
            f"({selfplay_seconds:.0f}s sp + {train_seconds:.0f}s train)",
            flush=True,
        )

        torch.save(
            {
                "model": model.state_dict(),
                "best": best.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            run_dir / "latest.pt",
        )
        state_path.write_text(
            json.dumps({"iteration": iteration + 1, "config": asdict(config)}, indent=2)
        )
        export_checkpoint(
            best,
            out_dir=run_dir / "best",
            model_id=f"{config.name}-best",
            training_report={"run": config.name, "history": record},
        )

    return run_dir / "best"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/train"))
    parser.add_argument("--no-resume", action="store_true")
    for field_name, value in asdict(AlphaZeroConfig()).items():
        flag = "--" + field_name.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(flag, action="store_true")
        elif value is None:
            parser.add_argument(flag, type=int, default=None)
        else:
            parser.add_argument(flag, type=type(value), default=value)
    args = parser.parse_args(argv)
    fields = set(asdict(AlphaZeroConfig()))
    config = AlphaZeroConfig(**{k: v for k, v in vars(args).items() if k in fields})
    path = train(config, args.out, resume=not args.no_resume)
    print(f"best checkpoint: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
