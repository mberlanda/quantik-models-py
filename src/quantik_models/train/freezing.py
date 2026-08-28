"""Freeze part of a network for fine-tuning.

Warm-starting from an existing checkpoint (`init_from`) answers "start
from what we already learned". Freezing answers the next question: keep
most of it and adapt only a part — retrain the heads on a new label
distribution, or hold the heads and let the trunk move.

Two things make this less trivial than `requires_grad_(False)`, and both
fail silently:

**Normalisation layers keep updating.** `requires_grad = False` stops the
gradient; it does not stop a `BatchNorm` in training mode from updating its
running mean and variance from the batch it sees. A "frozen" trunk whose
batch norms are still tracking is not frozen — it computes a different
function after one epoch, and nothing in the loss curve says so. Frozen
modules are therefore also held in eval mode, and re-held after every
`model.train()`.

**A pattern that matches nothing is a no-op.** `--freeze trunk` against an
architecture whose trunk is called `blocks` freezes nothing and trains
normally, which looks exactly like a successful fine-tune. Unmatched
patterns raise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from torch import nn

# Modules whose behaviour depends on batch statistics they accumulate in
# training mode, independently of any gradient.
_STATEFUL_NORMS = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
)


@dataclass
class FreezeReport:
    patterns: tuple[str, ...]
    frozen_parameters: int = 0
    trainable_parameters: int = 0
    frozen_tensors: list[str] = field(default_factory=list)
    held_in_eval: list[str] = field(default_factory=list)

    @property
    def total_parameters(self) -> int:
        return self.frozen_parameters + self.trainable_parameters

    def summary(self) -> str:
        if not self.patterns:
            return f"nothing frozen; {self.trainable_parameters:,} trainable"
        share = self.frozen_parameters / max(self.total_parameters, 1)
        return (
            f"froze {self.frozen_parameters:,} of {self.total_parameters:,} "
            f"parameters ({share:.1%}) matching {list(self.patterns)}; "
            f"{self.trainable_parameters:,} trainable, "
            f"{len(self.held_in_eval)} normalisation module(s) held in eval"
        )


def _matches(name: str, patterns: tuple[str, ...]) -> bool:
    """Prefix match on the dotted parameter or module path.

    Prefixes rather than globs because the useful unit is a submodule —
    `trunk`, `trunk.2`, `policy_head` — and a prefix names exactly that
    without inviting a pattern language nobody wants to debug.
    """
    return any(name == p or name.startswith(p + ".") for p in patterns)


def freeze(model: nn.Module, patterns: list[str] | tuple[str, ...]) -> FreezeReport:
    """Freeze every parameter under `patterns`; return what happened.

    Raises if a pattern matches nothing, because a fine-tune that silently
    froze nothing is indistinguishable from one that worked.
    """
    patterns = tuple(patterns)
    report = FreezeReport(patterns=patterns)

    if patterns:
        known = {name for name, _ in model.named_parameters()}
        unmatched = [p for p in patterns if not any(_matches(n, (p,)) for n in known)]
        if unmatched:
            tops = sorted({name.split(".")[0] for name in known})
            raise ValueError(
                f"freeze pattern(s) {unmatched} match no parameter; "
                f"top-level modules are {tops}"
            )

    for name, param in model.named_parameters():
        if patterns and _matches(name, patterns):
            param.requires_grad_(False)
            report.frozen_parameters += param.numel()
            report.frozen_tensors.append(name)
        else:
            report.trainable_parameters += param.numel()

    for name, module in model.named_modules():
        if name and isinstance(module, _STATEFUL_NORMS) and _matches(name, patterns):
            report.held_in_eval.append(name)

    return report


def frozen_norm_modules(model: nn.Module, report: FreezeReport) -> list[nn.Module]:
    lookup = dict(model.named_modules())
    return [lookup[name] for name in report.held_in_eval]


def set_train_mode(model: nn.Module, frozen_norms: list[nn.Module]) -> None:
    """`model.train()`, then put the frozen normalisation layers back.

    Call this instead of `model.train()` anywhere a frozen model is
    trained. `model.train()` recurses over every submodule, so it undoes
    the eval state on frozen norms every epoch — which is precisely the
    bug this module exists to prevent, and it leaves no trace in the loss.
    """
    model.train()
    for module in frozen_norms:
        module.eval()


def trainable_parameters(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]
