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


def test_every_artefact_belongs_to_a_known_repo():
    """The two repos are the split; a typo in `repo=` would silently orphan a group."""
    assert {a.repo for a in devdata.CATALOGUE} == {devdata.DATA_REPO, devdata.RUNS_REPO}


def test_repo_filter_selects_only_that_repos_groups(tmp_path):
    root = tmp_path / "root"
    (root / "runs/oracle/corpus").mkdir(parents=True)
    (root / "runs/oracle/corpus/exact.npz").write_bytes(b"corpus")
    (root / "runs/eval/run").mkdir(parents=True)
    (root / "runs/eval/run/games.json").write_text("{}")

    data = devdata.stage(root, tmp_path / "data", repo=devdata.DATA_REPO)
    runs = devdata.stage(root, tmp_path / "runs", repo=devdata.RUNS_REPO)

    assert (data / "corpora").is_dir()
    assert not (data / "evaluations").exists()
    assert (runs / "evaluations").is_dir()
    assert not (runs / "corpora").exists()


def test_unknown_repo_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown repo"):
        devdata.stage(tmp_path, tmp_path / "out", repo="quantik-nope")


def test_prune_clears_a_stale_file_but_keeps_dot_git(tmp_path):
    """The staging directory may be a clone of the dataset repo.

    A stale file in a backup is worse than a missing one — it still hashes
    fine — so `--prune` has to remove it. It must not remove `.git` with it.
    """
    root = tmp_path / "root"
    (root / "runs/oracle/corpus").mkdir(parents=True)
    (root / "runs/oracle/corpus/exact.npz").write_bytes(b"corpus")
    out = tmp_path / "out"

    devdata.stage(root, out, repo=devdata.DATA_REPO)
    (out / ".git").mkdir()
    stale = out / "corpora/runs/oracle/corpus/renamed-away.npz"
    stale.write_bytes(b"stale")

    devdata.stage(root, out, repo=devdata.DATA_REPO, prune=True)

    assert not stale.exists()
    assert (out / ".git").is_dir()
    assert (out / "corpora/runs/oracle/corpus/exact.npz").exists()


def test_resume_state_is_staged():
    """`latest.pt` is the file that makes an interrupted run resumable.

    It was missing from the first catalogue, which defeated the stated purpose
    of the repository for exactly the artefact that motivated it.
    """
    checkpoints = next(a for a in devdata.CATALOGUE if a.name == "checkpoints")
    assert "runs/train/*/latest.pt" in checkpoints.sources
    assert "runs/train/*/state.json" in checkpoints.sources
    assert "runs/train/*/final" in checkpoints.sources


def test_first_sentence_does_not_split_inside_bold():
    """`**HELD OUT.**` split on '.' leaves an unterminated marker on the card."""
    assert devdata.first_sentence("**HELD OUT.** Seven thousand. And more.") == (
        "**HELD OUT.** Seven thousand."
    )
    assert devdata.first_sentence("One sentence only.") == "One sentence only."
