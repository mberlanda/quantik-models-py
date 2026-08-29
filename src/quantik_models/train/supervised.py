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
from typing import get_args, get_type_hints

import numpy as np
import torch

from ..data.exact_corpus import (
    ExactCorpus,
    policy_weight,
    ply_sampling_weights,
    split_by_key,
    unpack,
)
from .metrics import merge_weighted as _merge
from ..env import fastboard as fb
from ..export.checkpoint import export_checkpoint
from ..model import registry
from ..model.policy_value_net import masked_log_softmax, parameter_count
from . import freezing
from .convergence import epochs_since_best
from .alphazero import resolve_device


@dataclass
class SupervisedConfig:
    name: str = "sup"
    corpus: str = "runs/oracle/corpus/exact.npz"
    arch: str = "resnet"
    preset: str = "small"
    channels: int | None = None
    blocks: int | None = None
    # A cap, not a budget, once `patience` is set. A *shared* epoch count is
    # not equal treatment for the same reason a shared learning rate was not:
    # sixteen was chosen when the ResNet was the only architecture, and the
    # attention encoder is still climbing when it runs out. See
    # `docs/learning-rate-sweep.md` for how the same mistake played out on
    # the rate.
    epochs: int = 30
    # Stop when the combined validation loss has not improved for this many
    # consecutive epochs. None keeps the fixed-length behaviour, so every
    # published run still reproduces exactly.
    patience: int | None = None
    batch_size: int = 1024
    # None means "ask the architecture". A shared default is not neutral —
    # 2e-3 was chosen for the ResNet, and every architecture added later
    # inherited it silently. The attention encoder does not train at 2e-3
    # at all. See `registry.ArchitectureEntry.default_lr`.
    lr: float | None = None
    min_lr: float = 1e-5
    weight_decay: float = 1e-4
    value_loss_weight: float = 1.0
    val_fraction: float = 0.05
    augment: bool = True
    # Draw training rows so every ply gets equal attention instead of
    # attention proportional to how many positions that ply happens to
    # contribute. The corpus is ~75% plies 7-13, but the match is decided at
    # plies 4-7 — the only region where the incumbent minimax is beatable.
    balance_plies: bool = True
    device: str = "auto"
    seed: int = 20260827
    init_from: str | None = None
    # Dotted module prefixes to hold fixed, e.g. "stem,trunk". Only
    # meaningful with `init_from`: freezing randomly initialised weights
    # trains a model around noise it can never correct.
    freeze: str | None = None

    def resolved_lr(self) -> float:
        """The learning rate this run will actually use."""
        return self.lr if self.lr is not None else registry.default_lr(self.arch)

    def build_model(self):
        """Resolve `arch` + `preset` + overrides into a model.

        Width and depth are passed as overrides rather than baked into a
        config type, so an architecture that has no notion of `channels`
        simply ignores it and the same CLI drives all of them.
        """
        overrides = {}
        if self.channels is not None:
            overrides["channels"] = self.channels
        if self.blocks is not None:
            overrides["blocks"] = self.blocks
        return registry.build(self.arch, preset=self.preset, **overrides)


def load_corpus(path: str | Path) -> dict[str, np.ndarray]:
    """Corpus arrays in the shape the training loop wants.

    Policy targets are stored as a 64-bit optimal-action mask (see
    `data.exact_corpus`); they are expanded per batch rather than up front,
    because a dense `(n, 64) float32` array for a 3M-row corpus is 790 MB of
    mostly zeros and dominated startup time.
    """
    corpus = ExactCorpus.load(path)
    return {
        "boards": corpus.boards,
        "optimal_mask": corpus.optimal_mask,
        "policy_weight": policy_weight(corpus.optimal_mask),
        "value_target": corpus.value_target,
        "plies": corpus.plies,
    }


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


def train(config: SupervisedConfig, out_root: Path) -> Path:
    run_dir = out_root / config.name
    run_dir.mkdir(parents=True, exist_ok=True)
    # Record the resolved learning rate, not `null`: a config that says
    # "None" does not reproduce the run it describes.
    recorded = asdict(config) | {"lr": config.resolved_lr()}
    (run_dir / "config.json").write_text(json.dumps(recorded, indent=2))
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

    model = config.build_model().to(device)
    if config.init_from:
        from safetensors.torch import load_file

        model.load_state_dict(load_file(str(Path(config.init_from) / "weights.safetensors")))
        print(f"initialized from {config.init_from}", flush=True)

    patterns = [p.strip() for p in config.freeze.split(",") if p.strip()] if config.freeze else []
    if patterns and not config.init_from:
        raise ValueError(
            "--freeze without --init-from would train a model around frozen "
            "random weights it can never correct"
        )
    freeze_report = freezing.freeze(model, patterns)
    frozen_norms = freezing.frozen_norm_modules(model, freeze_report)
    if patterns:
        print(freeze_report.summary(), flush=True)

    # Only the trainable parameters go to the optimizer. AdamW would
    # otherwise carry moment buffers for tensors that never receive a
    # gradient, which costs memory and makes the state dict misleading.
    lr = config.resolved_lr()
    optimizer = torch.optim.AdamW(
        freezing.trainable_parameters(model), lr=lr, weight_decay=config.weight_decay
    )
    sampling = (
        ply_sampling_weights(corpus["plies"][train_idx]) if config.balance_plies else None
    )
    steps_per_epoch = max(1, len(train_idx) // config.batch_size)
    # T_max is the *cap*, not the epoch the run turns out to stop at — the
    # schedule has to be fixed before the first step. So an early-stopped run
    # ends part-way down the cosine and never reaches `min_lr`, which
    # understates it slightly against a run that used its whole budget. That
    # is an argument for a generous patience, not for rescaling the schedule
    # to a length that is not known yet.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs * steps_per_epoch, eta_min=config.min_lr
    )
    source = "explicit" if config.lr is not None else f"{config.arch} default"
    print(
        f"{config.name}: {parameter_count(model):,} params on {device}, "
        f"{config.epochs} epochs x {steps_per_epoch} steps"
        + (f" (patience {config.patience})" if config.patience is not None else "")
        + f", lr {lr:g} ({source})",
        flush=True,
    )

    def batch_arrays(idx: np.ndarray, augment: bool):
        boards = corpus["boards"][idx]
        policy = unpack(corpus["optimal_mask"][idx])
        if augment:
            spatial, shape = fb.random_symmetries(idx.shape[0], rng)
            boards = fb.transform_boards(boards, spatial, shape)
            policy = fb.transform_policies(policy, spatial, shape)
        return boards, policy, corpus["policy_weight"][idx], corpus["value_target"][idx]

    best_val = float("inf")
    val_history: list[float] = []
    stopped_early = False
    for epoch in range(config.epochs):
        started = time.perf_counter()
        # Not `model.train()`: that recurses and puts frozen batch norms
        # back into training mode, where they keep updating running
        # statistics from the batch. A trunk that is still tracking is not
        # frozen, and nothing in the loss curve would say so.
        freezing.set_train_mode(model, frozen_norms)
        if sampling is None:
            order = rng.permutation(train_idx)
        else:
            order = rng.choice(
                train_idx, size=steps_per_epoch * config.batch_size, p=sampling
            )
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
            per_ply: dict[int, list[dict[str, tuple[float, float]]]] = {}
            for start in range(0, len(val_idx), 8192):
                idx = val_idx[start : start + 8192]
                _, metrics = _forward_losses(
                    model, *batch_arrays(idx, False), device, config.value_loss_weight
                )
                val_stats.append(metrics)
            val_metrics = _merge(val_stats)
            # Where the net is wrong matters more than how often: the arena is
            # decided in the opening, so track accuracy per ply.
            for ply in np.unique(corpus["plies"][val_idx]):
                rows = val_idx[corpus["plies"][val_idx] == ply]
                policy_rows = rows[corpus["policy_weight"][rows] > 0][:8192]
                if policy_rows.size == 0:
                    continue
                _, metrics = _forward_losses(
                    model, *batch_arrays(policy_rows, False), device, config.value_loss_weight
                )
                per_ply[int(ply)] = [metrics]

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
            "top1_by_ply": {
                str(ply): _merge(chunks)["top1"] for ply, chunks in sorted(per_ply.items())
            },
            "value_mae_by_ply": {
                str(ply): _merge(chunks)["value_mae"] for ply, chunks in sorted(per_ply.items())
            },
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
        opening = {p: v for p, v in record["top1_by_ply"].items() if int(p) <= 7}
        if opening:
            print(
                "        opening top1: "
                + "  ".join(f"ply{p}={v:.1%}" for p, v in sorted(opening.items(), key=lambda kv: int(kv[0]))),
                flush=True,
            )

        combined = float(record["val_policy_loss"] + record["val_value_loss"])
        val_history.append(combined)
        if combined < best_val:
            best_val = combined
            export_checkpoint(
                model,
                out_dir=run_dir / "best",
                model_id=f"{config.name}-best",
                training_report={"run": config.name, "epoch": epoch, "metrics": record},
            )
        stale = epochs_since_best(val_history)
        if config.patience is not None and stale >= config.patience:
            stopped_early = True
            print(
                f"        stopping: no improvement for {stale} epochs "
                f"(patience {config.patience}); best was epoch "
                f"{epoch - stale}",
                flush=True,
            )
            break
    export_checkpoint(
        model,
        out_dir=run_dir / "final",
        model_id=f"{config.name}-final",
        training_report={
            "run": config.name,
            # What actually ran, not what was asked for — a report saying 60
            # epochs for a run that stopped at 22 describes a different run.
            "epochs": len(val_history),
            "epoch_cap": config.epochs,
            "patience": config.patience,
            "stopped_early": stopped_early,
        },
    )
    return run_dir / "best"


# Resolved once: `from __future__ import annotations` makes every
# annotation a string, so the optional-field types below need the real
# objects rather than the source text.
_HINTS = get_type_hints(SupervisedConfig)


def build_parser() -> argparse.ArgumentParser:
    """The CLI, derived from `SupervisedConfig`.

    Separate from `main` so the flag types can be tested without running a
    training loop — which is how the `--lr`-as-string bug got past every
    test the first time.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/train"))
    defaults = asdict(SupervisedConfig())
    for field_name, value in defaults.items():
        flag = "--" + field_name.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(flag, action="store_true", default=value)
            parser.add_argument("--no-" + field_name.replace("_", "-"), dest=field_name, action="store_false")
        elif value is None:
            # Optional fields have no runtime value to infer a type from, so
            # read it off the annotation. This was a hardcoded name list
            # ({"channels", "blocks"} -> int, everything else -> str) until
            # `lr` became optional, was not on the list, and started arriving
            # as the string "2e-3" — which AdamW rejects with a TypeError
            # about comparing float and str, twelve runs into a sweep.
            annotation = _HINTS[field_name]
            inner = [a for a in get_args(annotation) if a is not type(None)]
            parser.add_argument(flag, type=inner[0] if inner else str, default=None)
        else:
            parser.add_argument(flag, type=type(value), default=value)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    fields = asdict(SupervisedConfig())
    config = SupervisedConfig(**{k: v for k, v in vars(args).items() if k in fields})
    path = train(config, args.out)
    print(f"best checkpoint: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
