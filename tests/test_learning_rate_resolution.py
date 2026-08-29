"""A shared learning-rate default is not neutral.

2e-3 was the trainer's default because it was chosen for the ResNet, the
only architecture that existed at the time. Every architecture added later
inherited it by omission, and the attention encoder does not train at 2e-3
at all — flat at 0.5130 for sixteen epochs, against 0.7271 and climbing
after three at 3e-4. It was one commit from being recorded as a failed
architecture on the strength of a hyperparameter belonging to a different
one.

These tests hold the fix in place: every architecture states its own rate,
an explicit `--lr` still wins, and the value written to disk is the one
actually used.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from quantik_models.model import registry  # noqa: E402
from quantik_models.train.supervised import SupervisedConfig  # noqa: E402


@pytest.mark.parametrize("name", registry.architectures())
def test_every_architecture_states_a_learning_rate(name: str) -> None:
    lr = registry.default_lr(name)
    assert 0 < lr < 1, f"{name} has an implausible default learning rate {lr}"


def test_the_attention_encoder_does_not_inherit_the_resnets_rate() -> None:
    """The specific regression: 2e-3 does not train `attn`."""
    assert registry.default_lr("attn") != registry.default_lr("resnet")
    assert registry.default_lr("attn") < registry.default_lr("resnet")


@pytest.mark.parametrize("name", registry.architectures())
def test_config_resolves_to_the_architectures_rate(name: str) -> None:
    assert SupervisedConfig(arch=name).resolved_lr() == registry.default_lr(name)


@pytest.mark.parametrize("name", registry.architectures())
def test_an_explicit_rate_still_wins(name: str) -> None:
    """Sweeps pass `--lr` and must not be silently overridden."""
    assert SupervisedConfig(arch=name, lr=1.5e-5).resolved_lr() == 1.5e-5


def test_an_unknown_architecture_is_refused_not_defaulted() -> None:
    with pytest.raises(ValueError, match="unknown architecture"):
        SupervisedConfig(arch="does-not-exist").resolved_lr()


def test_the_written_config_records_the_rate_actually_used(tmp_path) -> None:
    """A config saying `null` does not reproduce the run it describes."""
    import numpy as np

    from quantik_models.data.exact_corpus import ExactCorpus
    from quantik_models.env import fastboard as fb
    from quantik_models.train.supervised import train

    from boards import random_positions

    boards = random_positions(128, seed=2, plies=7)
    legal = fb.legal_masks(boards)
    mask = np.zeros(len(boards), dtype=np.uint64)
    for row in range(len(boards)):
        for action in np.flatnonzero(legal[row])[:2]:
            mask[row] |= np.uint64(1) << np.uint64(int(action))
    corpus_path = tmp_path / "corpus.npz"
    ExactCorpus(
        boards=boards,
        optimal_mask=mask,
        value_target=np.zeros(len(boards), dtype=np.float32),
        plies=np.full(len(boards), 7, dtype=np.int16),
    ).save(corpus_path)

    config = SupervisedConfig(
        name="lr-record",
        corpus=str(corpus_path),
        arch="attn",
        preset="smoke",
        epochs=1,
        batch_size=32,
        device="cpu",
        val_fraction=0.2,
        balance_plies=False,
    )
    train(config, tmp_path / "out")
    written = json.loads((tmp_path / "out" / "lr-record" / "config.json").read_text())
    assert written["lr"] == registry.default_lr("attn")


def test_the_preflight_uses_the_resolved_rate(tmp_path) -> None:
    """The regression this fix introduced, caught by the suite.

    `preflight` passed `config.lr` straight to AdamW. Once that field
    became None-by-default, the preflight raised a TypeError — and had it
    silently defaulted instead, it would have been checking a learning rate
    the real run never uses, which is worse: the fixed-batch check is
    precisely a check on the learning rate.
    """
    import inspect

    from quantik_models.train import preflight

    source = inspect.getsource(preflight.check_architecture)
    assert "config.resolved_lr()" in source
    assert "lr=config.lr" not in source


def test_optional_cli_flags_parse_to_their_annotated_type() -> None:
    """The bug that broke a running sweep.

    The CLI is generated from the dataclass, and optional fields have no
    runtime value to infer a type from. That inference was a hardcoded name
    list — `{"channels", "blocks"}` to int, everything else to str — so
    when `lr` became optional it started arriving as the string `"2e-3"`.
    Nothing failed until AdamW compared a float to a str, twelve runs into
    a sweep.

    Reading the annotation instead means a new optional field cannot
    silently land in the wrong type.
    """
    from quantik_models.train.supervised import build_parser

    args = build_parser().parse_args(
        ["--lr", "2e-3", "--channels", "64", "--blocks", "4", "--freeze", "stem"]
    )
    assert isinstance(args.lr, float) and args.lr == 2e-3
    assert isinstance(args.channels, int) and args.channels == 64
    assert isinstance(args.blocks, int) and args.blocks == 4
    assert isinstance(args.freeze, str)


def test_omitted_optional_flags_stay_none() -> None:
    from quantik_models.train.supervised import build_parser

    args = build_parser().parse_args([])
    assert args.lr is None and args.channels is None and args.freeze is None
