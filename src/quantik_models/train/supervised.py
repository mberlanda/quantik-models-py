"""Supervised training on exactly-solved positions.

The AlphaZero run showed the value head learning almost nothing: its target
was a blend of an 8-ply game result and its own undertrained root estimate,
so the signal was circular. Exact labels from `exact_oracle` break that —
every row here carries the true game-theoretic outcome, and most carry the
true optimal-move set as well.

Two things make this cheap:

* **Free child labels.** Solving a position solves all its children, so the
  corpus has roughly ten value-labelled rows per policy-labelled one. Rows
  without a policy target contribute to the value loss only, via
  `policy_weight`.
* **On-the-fly symmetry.** Each batch is transformed by a fresh random draw
  from the 192-element symmetry group, so the net effectively never sees the
  same board twice without paying for the storage.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from ..env import fastboard as fb
from ..export.checkpoint import export_checkpoint
from ..model.policy_value_net import (
    PRESETS,
    PolicyValueNet,
    PolicyValueNetConfig,
    masked_log_softmax,
    parameter_count,
)
from .alphazero import resolve_device


@dataclass
class SupervisedConfig:
    name: str = "sup"
    corpus: str = "runs/oracle/corpus/exact.npz"
    preset: str = "small"
    channels: int | None = None
    blocks: int | None = None
    epochs: int = 30
    batch_size: int = 1024
    lr: float = 2e-3
    min_lr: float = 1e-5
    weight_decay: float = 1e-4
    value_loss_weight: float = 1.0
    val_fraction: float = 0.05
    augment: bool = True
    device: str = "auto"
    seed: int = 20260827
    init_from: str | None = None

    def net_config(self) -> PolicyValueNetConfig:
        if self.channels is not None and self.blocks is not None:
            return PolicyValueNetConfig(channels=self.channels, blocks=self.blocks)
        return PRESETS[self.preset]


def load_corpus(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def split_by_key(boards: np.ndarray, val_fraction: float) -> np.ndarray:
    """Deterministic train/val mask keyed on the *canonical* position.

    Hashing the canonical key rather than the row keeps a position and its
    symmetric images on the same side of the split, so validation cannot be
    contaminated by a rotated copy of a training board.
    """
    keys = fb.canonical_keys(boards)
    bucket = (keys * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(40)  # 24-bit spread
    return (bucket / float(1 << 24)) < val_fraction


def _forward_losses(model, boards, policy, policy_weight, value, device, value_loss_weight):
    """Loss plus per-batch metrics, each paired with the weight it averages
    over so callers can aggregate exactly (see `_evaluate`)."""
    x = torch.from_numpy(fb.encode_tensors(boards)).to(device)
    mask = torch.from_numpy(fb.legal_masks(boards)).to(device)
    target_p = torch.from_numpy(policy).to(device)
    weight_p = torch.from_numpy(policy_weight).to(device)
    target_v = torch.from_numpy(value).to(device)

    logits, predicted = model(x)
    logp = masked_log_softmax(logits, mask)
    per_row = -(target_p * logp).sum(dim=-1)
    policy_loss = (per_row * weight_p).sum() / weight_p.sum().clamp_min(1.0)
    value_loss = ((predicted - target_v) ** 2).mean()
    total = policy_loss + value_loss_weight * value_loss

    with torch.no_grad():
        chosen = logp.argmax(dim=-1)
        # Credit any outcome-optimal move, not just the argmax of the target:
        # the target spreads mass evenly over all of them.
        hit = torch.gather(target_p, 1, chosen[:, None]).squeeze(1) > 0
        top1 = ((hit.float() * weight_p).sum() / weight_p.sum().clamp_min(1.0)).item()
        value_mae = (predicted - target_v).abs().mean().item()
        sign = (torch.sign(predicted) == torch.sign(target_v)).float().mean().item()
    return total, {
        "policy_loss": (policy_loss.item(), float(weight_p.sum())),
        "value_loss": (value_loss.item(), float(target_v.numel())),
        "top1": (top1, float(weight_p.sum())),
        "value_mae": (value_mae, float(target_v.numel())),
        "value_sign": (sign, float(target_v.numel())),
    }


def _merge(chunks: list[dict[str, tuple[float, float]]]) -> dict[str, float]:
    """Weighted mean per metric.

    A plain mean over chunks is wrong here: the corpus stores every
    policy-labelled row before every value-only row, so a sorted validation
    index puts nearly all policy rows in one chunk and none in the rest.
    Averaging those chunks equally divided the policy metrics by the chunk
    count — 89% top-1 was being reported as 11%.
    """
    out: dict[str, float] = {}
    for key in chunks[0]:
        total = sum(value * weight for value, weight in (c[key] for c in chunks))
        denominator = sum(weight for _, weight in (c[key] for c in chunks))
        out[key] = total / denominator if denominator else 0.0
    return out


def train(config: SupervisedConfig, out_root: Path) -> Path:
    run_dir = out_root / config.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))
    metrics_path = run_dir / "metrics.jsonl"

    device = resolve_device(config.device)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    corpus = load_corpus(config.corpus)
    is_val = split_by_key(corpus["boards"], config.val_fraction)
    train_idx = np.flatnonzero(~is_val)
    val_idx = np.flatnonzero(is_val)
    print(
        f"corpus {config.corpus}: {len(train_idx):,} train / {len(val_idx):,} val, "
        f"{int(corpus['policy_weight'].sum()):,} policy-labelled",
        flush=True,
    )

    model = PolicyValueNet(config.net_config()).to(device)
    if config.init_from:
        from safetensors.torch import load_file

        model.load_state_dict(load_file(str(Path(config.init_from) / "weights.safetensors")))
        print(f"initialized from {config.init_from}", flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    steps_per_epoch = max(1, len(train_idx) // config.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs * steps_per_epoch, eta_min=config.min_lr
    )
    print(
        f"{config.name}: {parameter_count(model):,} params on {device}, "
        f"{config.epochs} epochs x {steps_per_epoch} steps",
        flush=True,
    )

    def batch_arrays(idx: np.ndarray, augment: bool):
        boards = corpus["boards"][idx]
        policy = corpus["policy_target"][idx]
        if augment:
            spatial, shape = fb.random_symmetries(idx.shape[0], rng)
            boards = fb.transform_boards(boards, spatial, shape)
            policy = fb.transform_policies(policy, spatial, shape)
        return boards, policy, corpus["policy_weight"][idx], corpus["value_target"][idx]

    best_val = float("inf")
    for epoch in range(config.epochs):
        started = time.perf_counter()
        model.train()
        order = rng.permutation(train_idx)
        stats = []
        for step in range(steps_per_epoch):
            idx = order[step * config.batch_size : (step + 1) * config.batch_size]
            loss, metrics = _forward_losses(
                model, *batch_arrays(idx, config.augment), device, config.value_loss_weight
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            stats.append(metrics)
        train_metrics = _merge(stats)

        model.eval()
        with torch.no_grad():
            val_stats = []
            for start in range(0, len(val_idx), 8192):
                idx = val_idx[start : start + 8192]
                _, metrics = _forward_losses(
                    model, *batch_arrays(idx, False), device, config.value_loss_weight
                )
                val_stats.append(metrics)
            val_metrics = _merge(val_stats)

        record = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            "train_policy_loss": train_metrics["policy_loss"],
            "train_value_loss": train_metrics["value_loss"],
            "train_top1": train_metrics["top1"],
            "val_policy_loss": val_metrics["policy_loss"],
            "val_value_loss": val_metrics["value_loss"],
            "val_top1": val_metrics["top1"],
            "val_value_mae": val_metrics["value_mae"],
            "val_value_sign": val_metrics["value_sign"],
            "seconds": time.perf_counter() - started,
        }
        with metrics_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        print(
            f"[{config.name} {epoch:03d}] "
            f"train p={record['train_policy_loss']:.3f} v={record['train_value_loss']:.3f} "
            f"| val p={record['val_policy_loss']:.3f} v={record['val_value_loss']:.3f} "
            f"top1={record['val_top1']:.1%} vMAE={record['val_value_mae']:.3f} "
            f"sign={record['val_value_sign']:.1%} ({record['seconds']:.0f}s)",
            flush=True,
        )

        combined = record["val_policy_loss"] + record["val_value_loss"]
        if combined < best_val:
            best_val = combined
            export_checkpoint(
                model,
                out_dir=run_dir / "best",
                model_id=f"{config.name}-best",
                training_report={"run": config.name, "epoch": epoch, "metrics": record},
            )
    export_checkpoint(
        model,
        out_dir=run_dir / "final",
        model_id=f"{config.name}-final",
        training_report={"run": config.name, "epochs": config.epochs},
    )
    return run_dir / "best"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/train"))
    defaults = asdict(SupervisedConfig())
    for field_name, value in defaults.items():
        flag = "--" + field_name.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(flag, action="store_true", default=value)
            parser.add_argument("--no-" + field_name.replace("_", "-"), dest=field_name, action="store_false")
        elif value is None:
            parser.add_argument(flag, default=None)
        else:
            parser.add_argument(flag, type=type(value), default=value)
    args = parser.parse_args(argv)
    config = SupervisedConfig(**{k: v for k, v in vars(args).items() if k in defaults})
    path = train(config, args.out)
    print(f"best checkpoint: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
