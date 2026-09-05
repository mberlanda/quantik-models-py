"""The code in the documentation has to be the code that exists.

Two snippets shipped wrong to four public Hugging Face model cards, both of
the same shape: prose asserted against prose, never against the API. The
first named a weights file that staging renames. The second called
`evaluator.evaluate(boards)` — wrong method, and missing the legality mask
the real signature requires — so a reader following the card got an
AttributeError on the line the card exists to provide.

Nothing else in this suite reads the README, `docs/models.md` or a generated
card as code. These tests do.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "checkpoints" / "smoke-best"

# Every place the loading snippet is published.
DOCUMENTS = ("README.md", "docs/models.md")


# The committed smoke checkpoint predates `architecture_spec`, which the card
# generator requires, so the card is rendered from a manifest shaped like the
# ones the published repositories actually carry.
CARD_MANIFEST = {
    "schema": "model-checkpoint.v1",
    "architecture": "cpool-c191-b6",
    "architecture_spec": {"arch": "cpool", "config": {"blocks": 6, "channels": 191}},
    "contract_version": "1.2.0",
    "created_at": "2026-08-29T05:37:24+00:00",
    "input_contracts": ["tensor-board.v1", "bitboard.v1", "action-index.v1"],
    "output_contract": "policy-logits-64+value-tanh",
    "legal_action_mask_required": True,
    "parameter_count": 1780253,
    "weights_hash": "sha256:deadbeef",
}

SOURCES = (*DOCUMENTS, "generated model card")


def _text(source: str) -> str:
    if source == "generated model card":
        from quantik_models.export import huggingface as hf

        return hf.model_card(CARD_MANIFEST)
    return (REPO / source).read_text()


@pytest.mark.parametrize("source", SOURCES)
def test_no_document_calls_a_method_the_evaluator_does_not_have(source) -> None:
    text = _text(source)
    assert ".evaluate(" not in text, (
        f"{source} calls `.evaluate(...)`; an evaluator is callable — "
        "`evaluator(boards, legal)` — and has no `evaluate` method"
    )


@pytest.mark.parametrize("source", SOURCES)
def test_every_document_passes_the_legality_mask(source) -> None:
    """The mask is a required argument, not an optional refinement.

    A snippet that omits it does not merely return unmasked priors — it
    raises, because the signature has no default. This is the check that
    would have caught the published version.
    """
    text = _text(source)
    if "evaluator(" not in text:
        pytest.skip(f"{source} does not show an evaluator call")
    assert "legal_masks(" in text and "evaluator(boards, legal)" in text, (
        f"{source} calls the evaluator without computing a legality mask"
    )


def test_the_documented_call_matches_the_real_signature() -> None:
    """Pin the signature the documents are written against.

    If `__call__` grows or loses a required parameter, this fails and the
    documents get updated in the same change — rather than a year later,
    by a reader.
    """
    from quantik_models.selfplay.evaluator import Evaluator

    parameters = list(inspect.signature(Evaluator.__call__).parameters)
    assert parameters == ["self", "boards", "legal"]


def test_the_documented_call_actually_runs() -> None:
    """Execute the snippet against the committed smoke checkpoint.

    The 68 KB fixture is here precisely so this can run with no network and
    no `runs/` directory.
    """
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")

    import numpy as np

    from quantik_models.arena.registry import load_evaluator
    from quantik_models.env import fastboard as fb

    evaluator = load_evaluator(FIXTURE, "cpu")

    boards = fb.empty_boards(1)
    legal = fb.legal_masks(boards)
    policy, value = evaluator(boards, legal)

    assert policy.shape == (1, 64)
    assert value.shape == (1,)
    # The masking claim the documents make, checked rather than asserted in
    # prose: zero probability on an illegal action, and the legal ones sum
    # to one.
    assert policy[~legal].sum() == pytest.approx(0.0, abs=1e-6)
    assert policy[legal].sum() == pytest.approx(1.0, abs=1e-4)
    assert -1.0 <= float(value[0]) <= 1.0
    assert np.isfinite(policy).all()
