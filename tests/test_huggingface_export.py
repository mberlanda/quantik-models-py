"""Staging a checkpoint as a Hugging Face model repository.

Nothing here uploads. The failures this guards against are the ones that
are expensive *after* a push: front matter that publishes but does not
index, a `.gitattributes` written too late to keep a 7 MB blob out of the
history, and a card whose stated digest does not match the file beside it.
"""

from __future__ import annotations

import json

import pytest

from quantik_models.export import huggingface as hf

MANIFEST = {
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
    "onnx_export": "model.onnx",
    "onnx_hash": "sha256:feedface",
    "onnx_opset": 18,
}


def write_checkpoint(root, *, with_onnx=True, manifest=None):
    d = root / "best"
    d.mkdir(parents=True)
    (d / "weights.safetensors").write_bytes(b"weights")
    (d / "training-report.json").write_text("{}")
    payload = dict(manifest or MANIFEST)
    if with_onnx:
        (d / "model.onnx").write_bytes(b"onnx")
    else:
        payload.pop("onnx_export", None)
        payload.pop("onnx_hash", None)
    payload["weights_hash"] = hf.file_digest(d / "weights.safetensors")
    if with_onnx:
        payload["onnx_hash"] = hf.file_digest(d / "model.onnx")
    (d / "manifest.json").write_text(json.dumps(payload))
    return d


def test_gitattributes_tracks_onnx_which_the_hubs_default_does_not():
    """The trap: safetensors is handled for you, the graph beside it is not.

    LFS tracking has to exist in the commit that first contains the file —
    a follow-up commit leaves the blob in the history for good.
    """
    text = hf.gitattributes()
    assert "*.safetensors filter=lfs" in text
    assert "*.onnx filter=lfs" in text


def test_config_is_the_hub_facing_view_not_a_copy_of_the_manifest():
    config = hf.hf_config(MANIFEST)
    assert config["arch"] == "cpool"
    assert config["config"]["channels"] == 191
    assert config["legal_action_mask_required"] is True
    # No trust_remote_code path exists here; advertising one is worse than
    # advertising none.
    assert "auto_map" not in config
    # The manifest's own bookkeeping does not belong on the Hub side.
    assert "weights_hash" not in config


def test_config_refuses_a_checkpoint_with_no_architecture_spec():
    """Older checkpoints record only the name string, which cannot be rebuilt."""
    stale = {k: v for k, v in MANIFEST.items() if k != "architecture_spec"}
    with pytest.raises(ValueError, match="architecture_spec"):
        hf.hf_config(stale)


def test_the_card_front_matter_carries_the_metadata_the_hub_indexes():
    card = hf.model_card(
        MANIFEST,
        repo_id="me/quantik-cpool",
        metrics=[{"type": "accuracy", "name": "IID top-1", "value": 0.9893}],
    )
    assert card.startswith("---\n")
    head = card.split("---", 2)[1]
    # The weights are CC BY-NC 4.0, not the MIT the code carries: every OSI
    # licence permits royalty-free commercial use, which is the one thing
    # this reserves. Lowercase, from the Hub's fixed table.
    assert "license: cc-by-nc-4.0" in head
    assert "pipeline_tag: reinforcement-learning" in head
    assert "model-index:" in head and "0.9893" in head
    assert "  - cpool" in head


def test_the_card_installs_from_the_release_and_pins_the_source_underneath():
    """The install line a reader runs must be the published package.

    Both halves matter and they answer different questions: the release is
    how you run the model, the pinned commit is what produced the numbers on
    the card. The unpinned `git+https://...` form is neither — it tracks
    `main` and stops describing the card the first time main moves.
    """
    from quantik_models import __version__

    card = hf.model_card(MANIFEST, provenance={"code": {"commit": "abc1234"}})
    assert f"pip install 'quantik-models[torch,hub]>={__version__}'" in card
    assert "@abc1234" in card
    # `quantik-core` *is* published, on both registries.
    assert "pip install quantik-core" in card
    assert "cargo add quantik-core" in card


def test_the_card_snippet_loads_the_directory_this_module_stages(tmp_path):
    """The card's Python snippet, checked against the layout `stage` writes.

    This is the regression that shipped: the card said
    `load_evaluator(snapshot_download(repo_id))`, `stage` renames
    `weights.safetensors` to `model.safetensors`, and the loader only read
    the first name — so the primary snippet on all four published cards
    raised `FileNotFoundError`. Asserting the prose and the staged bytes
    separately is what let that through; this ties them together.
    """
    from quantik_models.arena.registry import weights_path

    checkpoint = write_checkpoint(tmp_path)
    out = tmp_path / "staged"
    hf.stage(checkpoint, out)

    card = (out / "README.md").read_text()
    assert 'hub.load_evaluator("cpool")' in card
    assert "from quantik_models import hub" in card

    # The name the card's call resolves to, and the file it will look for.
    from quantik_models import hub

    assert hub.repo_id("cpool") == hf.repo_id_for(MANIFEST["architecture"])
    assert weights_path(out).name == "model.safetensors"


def test_the_card_says_the_mask_is_the_callers_job():
    """The one thing a user of these weights can get silently wrong."""
    card = hf.model_card(MANIFEST)
    assert "Legality masking happens outside this model" in card
    assert "illegal moves" in card


def test_stage_writes_a_hub_ready_directory(tmp_path):
    checkpoint = write_checkpoint(tmp_path)
    out = hf.stage(checkpoint, tmp_path / "hub", repo_id="me/quantik-cpool")

    names = {p.name for p in out.iterdir()}
    # `weights.safetensors` is renamed: the Hub's viewers look for `model.`
    assert "model.safetensors" in names and "weights.safetensors" not in names
    assert names >= {"README.md", "config.json", ".gitattributes", "manifest.json", "model.onnx"}
    # Copies, never moves — the run directory is the record.
    assert (checkpoint / "weights.safetensors").exists()
    assert "me/quantik-cpool" in (out / "README.md").read_text()


def test_stage_handles_a_checkpoint_with_no_onnx(tmp_path):
    checkpoint = write_checkpoint(tmp_path, with_onnx=False)
    out = hf.stage(checkpoint, tmp_path / "hub")
    assert not (out / "model.onnx").exists()
    assert "model.onnx" not in (out / "README.md").read_text()


def test_verify_catches_a_file_that_does_not_match_its_published_digest(tmp_path):
    checkpoint = write_checkpoint(tmp_path)
    out = hf.stage(checkpoint, tmp_path / "hub")
    (out / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="manifest says"):
        hf.verify_staged(out)


def test_verify_refuses_a_directory_with_no_weights(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps(MANIFEST))
    with pytest.raises(ValueError, match="no weight file"):
        hf.verify_staged(tmp_path)


def test_staging_is_verified_before_it_returns(tmp_path, monkeypatch):
    """A staged directory that fails its own check must not be handed back."""
    calls = []
    real = hf.verify_staged
    monkeypatch.setattr(hf, "verify_staged", lambda d: calls.append(d) or real(d))
    checkpoint = write_checkpoint(tmp_path)
    hf.stage(checkpoint, tmp_path / "hub")
    assert calls == [tmp_path / "hub"]


SHIFT = [
    {
        "architecture": "cpool-c191-b6",
        "by_ply": {
            "4": {"won_positions": 100, "correct": 88, "accuracy": 0.88},
            "5": {"won_positions": 100, "correct": 93, "accuracy": 0.93},
            "6": {"won_positions": 200, "correct": 195, "accuracy": 0.975},
            "7": {"won_positions": 100, "correct": 99, "accuracy": 0.99},
        },
    },
    {"architecture": "mlp-h455-b4", "by_ply": {"4": {"won_positions": 100, "correct": 80}}},
]


def test_card_metrics_weights_plies_by_their_position_count():
    """Not the mean of the per-ply accuracies.

    Ply 6 carries twice the positions of ply 4 here; averaging the rates
    would report 0.928 where the actual accuracy over the band is 0.940.
    """
    metrics = hf.card_metrics(SHIFT, [], "cpool-c191-b6", "cpool")
    shallow = next(m for m in metrics if "4-6" in m["name"])
    assert shallow["value"] == pytest.approx((88 + 93 + 195) / 400, abs=1e-4)


def test_card_metrics_reports_shallow_and_deep_separately():
    """The corpus holds nothing at plies 4-5, so a pooled figure hides the
    only band that measures generalisation rather than recall."""
    names = [m["name"] for m in hf.card_metrics(SHIFT, [], "cpool-c191-b6", "cpool")]
    assert any("4-6" in n for n in names) and any("7-12" in n for n in names)


def test_card_metrics_adds_the_arena_row_when_the_agent_is_present():
    board = [{"agent": "cpool", "wins": 1029, "games": 1800, "win_rate": 0.5717}]
    metrics = hf.card_metrics(SHIFT, board, "cpool-c191-b6", "cpool")
    win = next(m for m in metrics if m["type"] == "win_rate")
    assert win["value"] == pytest.approx(0.5717)
    assert "1800 games" in win["name"]


def test_card_metrics_omits_the_arena_row_rather_than_inventing_one():
    metrics = hf.card_metrics(SHIFT, [{"agent": "attn", "wins": 1, "games": 2}], "cpool-c191-b6", "cpool")
    assert all(m["type"] != "win_rate" for m in metrics)


def test_card_metrics_refuses_an_architecture_the_evaluation_does_not_cover():
    """Silently publishing another model's accuracy is the failure to avoid."""
    with pytest.raises(ValueError, match="attn-d192-b6"):
        hf.card_metrics(SHIFT, [], "attn-d192-b6", "attn")


def test_repo_name_carries_the_project_prefix():
    """On the Hub a repo name sits alone in search, with no directory around it."""
    assert hf.repo_name_for("cpool-c191-b6") == "quantik-cpool-c191-b6"
    assert hf.repo_name_for("mlp-h455-b4") == "quantik-mlp-h455-b4"


def test_repo_name_is_not_prefixed_twice():
    assert hf.repo_name_for("quantik-attn-d192-b6") == "quantik-attn-d192-b6"


def test_repo_name_refuses_something_the_hub_would_not_take():
    with pytest.raises(ValueError, match="not a usable"):
        hf.repo_name_for("cpool c191")
    with pytest.raises(ValueError, match="cannot derive"):
        hf.repo_name_for("")


def test_repo_id_defaults_to_the_project_namespace(monkeypatch):
    """A repo id assembled by hand each time is how one model in a family
    ends up under a different account than the rest — and a Hub repo cannot
    be renamed without breaking every link that already points at it."""
    monkeypatch.delenv("QUANTIK_HF_NAMESPACE", raising=False)
    assert hf.repo_id_for("cpool-c191-b6") == f"{hf.DEFAULT_NAMESPACE}/quantik-cpool-c191-b6"


def test_repo_id_honours_the_environment_then_the_argument(monkeypatch):
    monkeypatch.setenv("QUANTIK_HF_NAMESPACE", "from-env")
    assert hf.repo_id_for("mlp-h455-b4") == "from-env/quantik-mlp-h455-b4"
    assert hf.repo_id_for("mlp-h455-b4", "explicit") == "explicit/quantik-mlp-h455-b4"


def test_repo_id_refuses_a_namespace_containing_a_slash():
    with pytest.raises(ValueError, match="must not contain a slash"):
        hf.repo_id_for("cpool-c191-b6", "org/extra")


def test_the_card_ships_a_real_repo_id_not_a_placeholder(monkeypatch):
    """A card carrying `<your-org>` teaches the reader to edit the snippet
    before running it, and most will not."""
    monkeypatch.delenv("QUANTIK_HF_NAMESPACE", raising=False)
    card = hf.model_card(MANIFEST)
    assert "<your-org>" not in card
    assert f"{hf.DEFAULT_NAMESPACE}/quantik-cpool-c191-b6" in card


def test_stage_derives_the_repo_id_from_the_namespace(tmp_path, monkeypatch):
    monkeypatch.delenv("QUANTIK_HF_NAMESPACE", raising=False)
    checkpoint = write_checkpoint(tmp_path)
    out = hf.stage(checkpoint, tmp_path / "hub", namespace="an-org")
    assert "an-org/quantik-cpool-c191-b6" in (out / "README.md").read_text()


def test_the_card_states_the_licence_split_rather_than_leaving_it_to_inference():
    """A reader who sees MIT on the GitHub repository will otherwise assume
    it covers the download. Weights and code are licensed differently here,
    and that is exactly the kind of thing nobody checks."""
    card = hf.model_card(MANIFEST)
    assert "cc-by-nc-4.0" in card.split("---", 2)[1]
    body = card.split("---", 2)[2]
    assert "non-commercial" in body.lower()
    assert "MIT" in body


def test_the_card_carries_the_architecture_diagram():
    """A Hub repo is read on its own, with no README beside it."""
    card = hf.model_card(MANIFEST)
    assert "```mermaid" in card
    assert "constraint block" in card  # the cpool diagram specifically


def test_every_registered_architecture_has_a_diagram_and_a_summary():
    """A new architecture must not be able to ship a card that silently
    omits its own diagram.

    Needs torch only because `model.registry` imports it to build the
    networks; the diagrams themselves are plain strings. The torch-free
    install is a tested configuration here, so this guards rather than
    failing collection for the whole job.
    """
    pytest.importorskip("torch")
    from quantik_models.export import cards
    from quantik_models.model import registry

    for name in registry.architectures():
        assert cards.diagram_for(name), f"{name} has no diagram"
        assert cards.summary_for(name), f"{name} has no summary"


def test_the_card_links_the_project_writeup():
    card = hf.model_card(MANIFEST)
    assert "mauroberlanda.substack.com/t/quantik" in card
