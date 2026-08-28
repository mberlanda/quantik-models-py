"""Confirm the assumptions a long training run rests on, in about a minute.

Training `resnet-c128-b6` on the sampled corpus took roughly 45 minutes on
MPS. Three architectures at that size is most of an afternoon, and the
failure modes that waste it are all cheap to detect up front: a corpus that
does not carry the key you expected, a split that leaks, an architecture
whose gradients do not reach half its parameters, an ONNX export that only
works at the batch size it was traced with.

So this runs the *real* code paths — `load_corpus`, `split_by_key`,
`_forward_losses`, `export_checkpoint` — on a handful of batches, checks
what it can check, and reports a projected wall-clock per architecture so
the run can be budgeted before it is started rather than after.

    python -m quantik_models.train.preflight \\
      --corpus runs/oracle/corpus/exact-sampled.npz \\
      --arch resnet mlp cpool --preset medium --epochs 16

Exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..data.exact_corpus import split_by_key, unpack
from ..env import fastboard as fb
from ..export.checkpoint import export_checkpoint
from ..model import registry
from ..model.policy_value_net import masked_log_softmax, parameter_count
from .alphazero import resolve_device
from .supervised import SupervisedConfig, _forward_losses, load_corpus

# Enough steps to see a loss trend and to time a step past warm-up, few
# enough that the whole preflight stays under a minute per architecture.
_STEPS = 12


@dataclass
class Check:
    name: str
    ok: bool
    detail: str

    def render(self) -> str:
        return f"  [{'ok ' if self.ok else 'FAIL'}] {self.name}: {self.detail}"


def check_corpus(corpus: dict[str, np.ndarray], path: str) -> list[Check]:
    checks = []
    rows = len(corpus["boards"])
    labelled = int((corpus["policy_weight"] > 0).sum())
    checks.append(
        Check(
            "corpus loads",
            rows > 0,
            f"{rows:,} rows from {path}, {labelled:,} policy-labelled "
            f"({labelled / max(rows, 1):.1%})",
        )
    )

    # The boards have to survive a round trip through the canonical key, or
    # the split below is keyed on something other than the position.
    keys = fb.canonical_keys(corpus["boards"][: min(rows, 50_000)])
    checks.append(
        Check(
            "canonical keys are dense",
            len(np.unique(keys)) > 0.5 * len(keys),
            f"{len(np.unique(keys)):,} distinct in the first {len(keys):,} rows",
        )
    )

    plies, counts = np.unique(corpus["plies"], return_counts=True)
    span = ", ".join(f"{p}:{c:,}" for p, c in zip(plies.tolist(), counts.tolist()))
    checks.append(Check("ply coverage", True, span))
    return checks


def check_split(corpus: dict[str, np.ndarray], val_fraction: float) -> list[Check]:
    """The check that matters most, because leakage is invisible downstream.

    A leaked split does not crash, does not warn, and reports a better
    validation number than it earned — so it is worth paying to verify on
    the real corpus rather than trusting the unit test on synthetic boards.
    """
    boards = corpus["boards"]
    is_val = split_by_key(boards, val_fraction)
    keys = fb.canonical_keys(boards)
    train_keys = np.unique(keys[~is_val])
    val_keys = np.unique(keys[is_val])
    shared = np.intersect1d(train_keys, val_keys, assume_unique=True)

    actual = is_val.mean()
    return [
        Check(
            "split is leak-free",
            shared.size == 0,
            f"{shared.size:,} canonical keys on both sides "
            f"({len(train_keys):,} train / {len(val_keys):,} val)",
        ),
        Check(
            "split proportion",
            abs(actual - val_fraction) < 0.25 * val_fraction,
            f"{actual:.2%} validation against a requested {val_fraction:.2%}",
        ),
    ]


def _sample_batch(
    corpus: dict[str, np.ndarray], rng: np.random.Generator, size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A batch drawn from the policy-labelled rows.

    Policy-labelled specifically: an unlabelled batch has zero policy
    weight, the policy loss is then a division by a clamped zero, and the
    gradient check below would pass for the wrong reason.
    """
    labelled = np.flatnonzero(corpus["policy_weight"] > 0)
    idx = rng.choice(labelled, size=min(size, labelled.size), replace=False)
    return (
        corpus["boards"][idx],
        unpack(corpus["optimal_mask"][idx]),
        corpus["policy_weight"][idx],
        corpus["value_target"][idx],
    )


def check_architecture(
    arch: str,
    config: SupervisedConfig,
    corpus: dict[str, np.ndarray],
    device: torch.device,
    steps_per_epoch: int,
) -> tuple[list[Check], float]:
    checks: list[Check] = []
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    model = config.build_model().to(device)
    params = parameter_count(model)
    checks.append(Check(f"{arch}: builds", True, f"{model.architecture}, {params:,} params"))

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    batch = _sample_batch(corpus, rng, config.batch_size)

    # One backward pass before timing anything: lazy kernel compilation and
    # the first allocation on MPS are not representative of a steady step.
    model.train()
    loss, _ = _forward_losses(model, *batch, device, config.value_loss_weight)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    starved = [n for n, p in model.named_parameters() if p.grad is None or not p.grad.any()]
    checks.append(
        Check(
            f"{arch}: every parameter gets gradient",
            not starved,
            "all reached" if not starved else f"{len(starved)} starved, e.g. {starved[:3]}",
        )
    )
    optimizer.step()

    losses = []
    started = time.perf_counter()
    for _ in range(_STEPS):
        loss, _ = _forward_losses(model, *batch, device, config.value_loss_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(loss.item())
    if device.type in ("mps", "cuda"):
        torch.mps.synchronize() if device.type == "mps" else torch.cuda.synchronize()
    per_step = (time.perf_counter() - started) / _STEPS

    # Overfitting one fixed batch is the cheapest signal that the model can
    # learn at all. It says nothing about generalisation and is not meant to
    # — it catches a frozen trunk, a detached graph, or an learning rate
    # that is wrong by orders of magnitude.
    checks.append(
        Check(
            f"{arch}: loss falls on a fixed batch",
            losses[-1] < losses[0],
            f"{losses[0]:.4f} -> {losses[-1]:.4f} over {_STEPS} steps",
        )
    )

    model.eval()
    with torch.no_grad():
        boards = batch[0][:1]
        x = torch.from_numpy(fb.encode_tensors(boards)).to(device)
        mask = torch.from_numpy(fb.legal_masks(boards)).to(device)
        logits, value = model(x)
        logp = masked_log_softmax(logits, mask)
        chosen = int(logp.argmax(dim=-1).item())
        legal = bool(mask[0, chosen].item())
    checks.append(
        Check(
            f"{arch}: single position, masked argmax is legal",
            legal and -1.0 <= float(value.item()) <= 1.0,
            f"action {chosen}, legal={legal}, value={float(value.item()):+.3f}",
        )
    )

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        export_checkpoint(model, out_dir=out, model_id=f"preflight-{arch}", training_report={})
        onnx_ok, onnx_detail = _check_onnx(model, out)
    checks.append(Check(f"{arch}: ONNX matches torch", onnx_ok, onnx_detail))

    projected = per_step * steps_per_epoch * config.epochs
    checks.append(
        Check(
            f"{arch}: projected wall-clock",
            True,
            f"{per_step * 1000:.0f} ms/step x {steps_per_epoch:,} steps x "
            f"{config.epochs} epochs = {projected / 60:.0f} min "
            f"(training only, excludes per-epoch validation)",
        )
    )
    return checks, projected


def _check_onnx(model: torch.nn.Module, out: Path) -> tuple[bool, str]:
    """Run the exported graph at a batch size it was not traced with.

    The trace uses a batch of one. A graph that specialised on that — which
    is what a 2-D-by-3-D matmul does — passes a round trip at batch one and
    fails everywhere else, so the batch size here has to differ.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        return False, "onnxruntime not installed; install the [onnx] extra"

    was_on = next(model.parameters()).device
    model.to("cpu")
    try:
        session = ort.InferenceSession(
            str(out / "model.onnx"), providers=["CPUExecutionProvider"]
        )
        drift = 0.0
        # Several batch sizes, none of them the traced one. A graph can
        # advertise a symbolic batch dimension and still carry an internal
        # Reshape frozen at the batch it was traced with — which is exactly
        # what `cpool` did, and what a single round trip would not catch.
        for batch in (1, 5, 64):
            sample = torch.randn(batch, 9, 4, 4)
            with torch.no_grad():
                want_policy, want_value = model(sample)
            got_policy, got_value = session.run(None, {"board": sample.numpy()})
            drift = max(
                drift,
                float(np.abs(got_policy - want_policy.numpy()).max()),
                float(np.abs(got_value - want_value.numpy()).max()),
            )
    except Exception as exc:  # noqa: BLE001 - a failed check is a result
        return False, f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
    finally:
        model.to(was_on)

    return drift < 1e-4, f"max abs drift {drift:.2e} across batches 1, 5 and 64"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", default="runs/oracle/corpus/exact-sampled.npz")
    parser.add_argument("--arch", nargs="+", default=list(registry.architectures()))
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    print(f"device: {device}")

    corpus = load_corpus(args.corpus)
    checks = check_corpus(corpus, args.corpus)
    checks += check_split(corpus, args.val_fraction)
    print("\ncorpus and split")
    for check in checks:
        print(check.render())

    train_rows = int((~split_by_key(corpus["boards"], args.val_fraction)).sum())
    steps_per_epoch = max(1, train_rows // args.batch_size)

    total = 0.0
    for arch in args.arch:
        config = SupervisedConfig(
            name=f"preflight-{arch}",
            corpus=args.corpus,
            arch=arch,
            preset=args.preset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            val_fraction=args.val_fraction,
            device=args.device,
            seed=args.seed,
        )
        print(f"\n{arch} @ {args.preset}")
        arch_checks, projected = check_architecture(
            arch, config, corpus, device, steps_per_epoch
        )
        for check in arch_checks:
            print(check.render())
        checks += arch_checks
        total += projected

    failed = [c for c in checks if not c.ok]
    print(f"\nprojected total: {total / 3600:.1f} h across {len(args.arch)} architectures")
    if failed:
        print(f"\n{len(failed)} check(s) failed:")
        for check in failed:
            print(check.render())
        return 1
    print(f"all {len(checks)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
