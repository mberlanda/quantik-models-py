"""Training loop and CLI for the Quantik policy/value network.

Loss = sample-weighted soft-target cross-entropy over legality-masked
log-probabilities, plus MSE on the tanh value, optimized with AdamW.
`--device auto` prefers cuda > mps > cpu; CPU is always sufficient —
no accelerator is required.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from ..data.dataset import LoadedTrainingData, load_training_data
from ..export.checkpoint import export_checkpoint
from ..model.policy_value_net import (
    PRESETS,
    PolicyValueNet,
    PolicyValueNetConfig,
    masked_log_softmax,
)


@dataclass
class TrainConfig:
    npz_paths: list[Path]
    preset: str = "smoke"
    channels: int | None = None
    blocks: int | None = None
    epochs: int = 2
    batch_size: int = 64
    lr: float = 1e-3
    value_loss_weight: float = 1.0
    weight_decay: float = 1e-4
    seed: int = 20260715
    device: str = "auto"
    out_dir: Path = field(default_factory=lambda: Path("outputs/checkpoint"))
    model_id: str | None = None

    def net_config(self) -> PolicyValueNetConfig:
        if self.channels is not None or self.blocks is not None:
            if self.channels is None or self.blocks is None:
                raise ValueError("--channels and --blocks must be given together")
            return PolicyValueNetConfig(channels=self.channels, blocks=self.blocks)
        if self.preset not in PRESETS:
            raise ValueError(f"unknown preset {self.preset!r}")
        return PRESETS[self.preset]


def _resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _to_tensors(
    data: LoadedTrainingData, device: torch.device
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    return (
        torch.from_numpy(data.tensors).to(device),
        torch.from_numpy(data.policy_target).to(device),
        torch.from_numpy(data.value_target).to(device),
        torch.from_numpy(data.sample_weight).to(device),
        torch.from_numpy(data.legal_mask).to(device),
    )


def _losses(
    model: PolicyValueNet,
    x: Tensor,
    policy_target: Tensor,
    value_target: Tensor,
    weight: Tensor,
    legal_mask: Tensor,
    value_loss_weight: float,
    logits_value: tuple[Tensor, Tensor] | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    logits, value = logits_value if logits_value is not None else model(x)
    logp = masked_log_softmax(logits, legal_mask)
    # Soft-target CE; rows can carry zero policy mass on illegal actions
    # by construction of the training view.
    policy_loss = -(policy_target * logp).sum(dim=-1)
    value_mse = (value - value_target) ** 2
    norm = weight.sum().clamp_min(1e-8)
    policy_loss = (policy_loss * weight).sum() / norm
    value_loss = (value_mse * weight).sum() / norm
    return policy_loss + value_loss_weight * value_loss, policy_loss, value_loss


@torch.no_grad()
def _validate(
    model: PolicyValueNet,
    x: Tensor,
    policy_target: Tensor,
    value_target: Tensor,
    weight: Tensor,
    legal_mask: Tensor,
    value_loss_weight: float,
) -> dict[str, float]:
    model.eval()
    logits, value = model(x)
    _, policy_loss, value_loss = _losses(
        model,
        x,
        policy_target,
        value_target,
        weight,
        legal_mask,
        value_loss_weight,
        logits_value=(logits, value),
    )
    # Weight the per-row metrics the same way the losses are weighted so all
    # validation numbers describe the same distribution.
    norm = weight.sum().clamp_min(1e-8)
    masked_logp = masked_log_softmax(logits, legal_mask)
    top1_hits = (masked_logp.argmax(dim=-1) == policy_target.argmax(dim=-1)).float()
    top1 = ((top1_hits * weight).sum() / norm).item()
    premask_probs = torch.softmax(logits, dim=-1)
    illegal_rows = premask_probs.masked_fill(legal_mask, 0.0).sum(dim=-1)
    illegal_mass = ((illegal_rows * weight).sum() / norm).item()
    model.train()
    return {
        "val_policy_loss": float(policy_loss.item()),
        "val_value_mse": float(value_loss.item()),
        "val_top1_agreement": float(top1),
        "val_illegal_mass_premask": float(illegal_mass),
    }


def train(config: TrainConfig) -> dict[str, Any]:
    started = time.monotonic()
    _seed_everything(config.seed)
    device = _resolve_device(config.device)

    train_data = load_training_data(config.npz_paths, split="train")
    val_data = load_training_data(config.npz_paths, split="val")
    if train_data.tensors.shape[0] == 0:
        raise ValueError("training split is empty")
    # Tiny corpora can shard into an empty val split; fall back to train.
    if val_data.tensors.shape[0] == 0:
        val_data = train_data

    model = PolicyValueNet(config.net_config()).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    x, pt, vt, w, m = _to_tensors(train_data, device)
    vx, vpt, vvt, vw, vm = _to_tensors(val_data, device)

    epochs_report: list[dict[str, Any]] = []
    n = x.shape[0]
    generator = torch.Generator().manual_seed(config.seed)
    for epoch in range(config.epochs):
        perm = torch.randperm(n, generator=generator)
        policy_sum = value_sum = weight_sum = 0.0
        for start in range(0, n, config.batch_size):
            idx = perm[start : start + config.batch_size]
            batch_weight_sum = w[idx].sum().item()
            optimizer.zero_grad()
            total, policy_loss, value_loss = _losses(
                model, x[idx], pt[idx], vt[idx], w[idx], m[idx],
                config.value_loss_weight,
            )
            total.backward()
            optimizer.step()
            policy_sum += float(policy_loss.item()) * batch_weight_sum
            value_sum += float(value_loss.item()) * batch_weight_sum
            weight_sum += batch_weight_sum
        entry: dict[str, Any] = {
            "epoch": epoch,
            "train_policy_loss": policy_sum / max(weight_sum, 1e-8),
            "train_value_mse": value_sum / max(weight_sum, 1e-8),
        }
        entry.update(
            _validate(model, vx, vpt, vvt, vw, vm, config.value_loss_weight)
        )
        epochs_report.append(entry)

    report: dict[str, Any] = {
        "config": {
            "preset": config.preset,
            "channels": model.config.channels,
            "blocks": model.config.blocks,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "lr": config.lr,
            "value_loss_weight": config.value_loss_weight,
            "weight_decay": config.weight_decay,
            "seed": config.seed,
            "device": str(device),
        },
        "dataset": {
            "train_rows": int(train_data.tensors.shape[0]),
            "val_rows": int(val_data.tensors.shape[0]),
            "sources": [str(p) for p in config.npz_paths],
        },
        "epochs": epochs_report,
        "final": epochs_report[-1],
        "elapsed_seconds": time.monotonic() - started,
    }

    net_config = model.config
    model_id = config.model_id or (
        f"quantik-pv-c{net_config.channels}-b{net_config.blocks}-seed{config.seed}"
    )
    export_checkpoint(
        model.cpu(),
        out_dir=Path(config.out_dir),
        model_id=model_id,
        training_report=report,
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train a Quantik policy/value model from .npz training views."
    )
    parser.add_argument("--npz", action="append", required=True, type=Path)
    parser.add_argument("--preset", default="smoke", choices=sorted(PRESETS))
    parser.add_argument("--channels", type=int, default=None)
    parser.add_argument("--blocks", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/checkpoint"))
    parser.add_argument("--model-id", default=None)
    args = parser.parse_args(argv)

    config = TrainConfig(
        npz_paths=list(args.npz),
        preset=args.preset,
        channels=args.channels,
        blocks=args.blocks,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        value_loss_weight=args.value_loss_weight,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        out_dir=args.out_dir,
        model_id=args.model_id,
    )
    report = train(config)
    final = report["final"]
    print(
        "trained "
        f"preset={args.preset} epochs={args.epochs} "
        f"train_policy_loss={final['train_policy_loss']:.4f} "
        f"val_top1={final['val_top1_agreement']:.3f} "
        f"-> {config.out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
