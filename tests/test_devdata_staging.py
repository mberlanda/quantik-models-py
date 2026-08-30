"""Staging the artefacts `runs/` holds and git does not."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quantik_models.export import devdata


def _tree(root: Path) -> None:
    (root / "runs/oracle/corpus").mkdir(parents=True)
    (root / "runs/oracle/corpus/exact-sampled.npz").write_bytes(b"rows")
    (root / "runs/oracle/corpus/exact-sampled-v2.npz").write_bytes(b"more rows")
    (root / "runs/oracle").mkdir(exist_ok=True)
    (root / "runs/oracle/probe-large.jsonl").write_text('{"qfen": "..../..../..../...."}\n')


def test_stage_is_content_addressed_and_copies_never_moves(tmp_path: Path) -> None:
    root, out = tmp_path / "repo", tmp_path / "out"
    root.mkdir()
    _tree(root)
    devdata.stage(root, out, only=("corpora", "probe"))

    # The source is untouched: this is a backup, not a move.
    assert (root / "runs/oracle/corpus/exact-sampled.npz").exists()

    manifest = json.loads((out / "MANIFEST.json").read_text())
    assert manifest["schema"] == "dev-data-manifest.v1"
    corpora = next(g for g in manifest["artefacts"] if g["name"] == "corpora")
    assert corpora["file_count"] == 2
    # Every file carries a hash, because a corpus is identified by content and
    # not by a filename that differs from another by one character.
    assert all(entry["sha256"].startswith("sha256:") for entry in corpora["files"])
    digests = {entry["sha256"] for entry in corpora["files"]}
    assert len(digests) == 2


def test_paths_stay_relative_to_the_repository_root(tmp_path: Path) -> None:
    root, out = tmp_path / "repo", tmp_path / "out"
    root.mkdir()
    _tree(root)
    devdata.stage(root, out, only=("corpora",))
    # So `cp -r corpora/runs/ <checkout>/` restores to where tooling expects it.
    assert (out / "corpora/runs/oracle/corpus/exact-sampled.npz").is_file()


def test_every_group_gets_a_card_that_says_how_to_reproduce_and_expand(tmp_path: Path) -> None:
    root, out = tmp_path / "repo", tmp_path / "out"
    root.mkdir()
    _tree(root)
    devdata.stage(root, out, only=("corpora", "probe"))
    for name in ("corpora", "probe"):
        card = (out / name / "README.md").read_text()
        assert "## Reproducing it" in card
        assert "## Extending it" in card
    # The probe's card leads with the fact that determines how it may be used.
    assert "HELD OUT" in (out / "probe/README.md").read_text()


def test_lfs_is_configured_for_every_binary_pattern_staged(tmp_path: Path) -> None:
    root, out = tmp_path / "repo", tmp_path / "out"
    root.mkdir()
    _tree(root)
    devdata.stage(root, out, only=("corpora",))
    attributes = (out / ".gitattributes").read_text()
    # A file committed as a plain blob cannot be fixed by a later commit.
    for pattern in ("*.npz", "*.npy", "*.safetensors", "*.onnx"):
        assert f"{pattern} filter=lfs" in attributes


def test_an_unknown_group_is_refused_rather_than_silently_staging_nothing(tmp_path: Path) -> None:
    root, out = tmp_path / "repo", tmp_path / "out"
    root.mkdir()
    _tree(root)
    with pytest.raises(ValueError, match="unknown artefact"):
        devdata.stage(root, out, only=("corpra",))


def test_a_group_with_no_matching_files_stages_empty_rather_than_failing(tmp_path: Path) -> None:
    root, out = tmp_path / "repo", tmp_path / "out"
    root.mkdir()
    (root / "runs").mkdir()
    devdata.stage(root, out, only=("checkpoints",))
    group = json.loads((out / "MANIFEST.json").read_text())["artefacts"][0]
    # A machine that has not trained anything still stages its corpora.
    assert group["file_count"] == 0
    assert (out / "checkpoints/README.md").is_file()
