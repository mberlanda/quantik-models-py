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
    assert "license: apache-2.0" in head
    assert "pipeline_tag: reinforcement-learning" in head
    assert "model-index:" in head and "0.9893" in head
    assert "  - cpool" in head


def test_the_card_says_the_mask_is_the_callers_job():
    """The one thing a user of these weights can get silently wrong."""
    card = hf.model_card(MANIFEST)
    assert "Legality masking happens outside this model" in card
    assert "illegal moves" in card


def test_the_card_uses_a_placeholder_that_looks_like_one():
    card = hf.model_card(MANIFEST)
    assert "<your-org>/cpool-c191-b6" in card


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
