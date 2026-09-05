"""What a run records so it can be reproduced later, not merely described."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from quantik_models.train import provenance as prov


def _git(path: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(path), "-c", "user.name=T", "-c", "user.email=t@t.invalid", *args],
        check=True,
        capture_output=True,
    )


def test_capture_never_raises_and_always_names_its_schema(tmp_path: Path) -> None:
    record = prov.capture(corpus=tmp_path / "does-not-exist.npz", device="cpu")
    assert record["schema"] == "training-provenance.v1"
    # A missing input is recorded with its reason, not omitted: an absent key
    # reads as "not checked", a null with a reason reads as "checked, absent".
    assert record["corpus"]["sha256"] is None
    assert record["corpus"]["reason"]


def test_file_digest_hashes_content_not_name(tmp_path: Path) -> None:
    same_a = tmp_path / "exact-sampled.npz"
    same_b = tmp_path / "exact-sampled-v2.npz"
    same_a.write_bytes(b"rows")
    same_b.write_bytes(b"rows")
    differs = tmp_path / "exact-sampled-v3.npz"
    differs.write_bytes(b"other rows")
    # Confusable names, identical content -> identical digest. This is the
    # failure the record exists to prevent: a filename is not an identity.
    assert prov.file_digest(same_a)["sha256"] == prov.file_digest(same_b)["sha256"]
    assert prov.file_digest(differs)["sha256"] != prov.file_digest(same_a)["sha256"]
    assert prov.file_digest(same_a)["size_bytes"] == 4


def test_code_provenance_records_commit_dirtiness_and_a_permalink(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "remote", "add", "origin", "git@github.com:owner/repo.git")
    (repo / "f.txt").write_text("one")
    _git(repo, "add", "f.txt")
    _git(repo, "commit", "-q", "-m", "one")

    clean = prov.code_provenance(repo)
    assert clean["commit"] and len(clean["commit"]) == 40
    assert clean["dirty"] is False
    assert clean["branch"] == "main"
    # An ssh remote becomes a browsable link, so the recorded commit is
    # reachable by a reader rather than merely quoted at them.
    assert clean["commit_url"] == f"https://github.com/owner/repo/commit/{clean['commit']}"

    (repo / "f.txt").write_text("two")
    assert prov.code_provenance(repo)["dirty"] is True


def test_code_provenance_outside_a_checkout_says_so(tmp_path: Path) -> None:
    record = prov.code_provenance(tmp_path)
    assert record["commit"] is None
    assert "not a git checkout" in record["reason"]


@pytest.mark.parametrize(
    "remote,expected",
    [
        ("git@github.com:o/r.git", "https://github.com/o/r/commit/abc"),
        ("https://github.com/o/r.git", "https://github.com/o/r/commit/abc"),
        ("https://github.com/o/r", "https://github.com/o/r/commit/abc"),
        # No link at all beats a wrong one: a reader cannot tell a bad guess
        # from a deleted commit.
        ("/srv/git/local.git", None),
        (None, None),
    ],
)
def test_commit_url_refuses_to_guess(remote: str | None, expected: str | None) -> None:
    assert prov._commit_url(remote, "abc") == expected


def test_versions_records_a_null_for_an_absent_package() -> None:
    versions = prov.version_provenance(("numpy", "definitely-not-installed"))
    assert versions["python"]
    assert versions["numpy"]
    assert versions["definitely-not-installed"] is None


def test_hardware_records_the_resolved_device_not_the_request() -> None:
    record = prov.hardware_provenance("cpu")
    assert record["device"] == "cpu"
    assert record["platform"] and record["cpu_count"]


def test_card_pins_the_install_to_the_commit_that_trained_the_weights() -> None:
    from quantik_models import __version__
    from quantik_models.export import huggingface as hf

    pinned = "\n".join(hf._install_lines({"code": {"commit": "a" * 40}}))
    # The release is what a reader runs; the commit is what reproduces the
    # numbers. Both, in that order.
    assert f"quantik-models[torch,hub]>={__version__}" in pinned
    assert "@" + "a" * 40 in pinned

    # Without a commit the pinned line is *omitted* rather than degraded to an
    # unpinned `git+https://...`, which would track main and stop describing
    # the card the first time main moved. The card's provenance table is then
    # absent too, which is the visible signal that nothing was recorded.
    unpinned = "\n".join(hf._install_lines(None))
    assert f"quantik-models[torch,hub]>={__version__}" in unpinned
    assert "git+https" not in unpinned


def test_card_flags_a_dirty_tree_rather_than_quoting_the_commit_plainly() -> None:
    from quantik_models.export import huggingface as hf

    dirty = "\n".join(hf._provenance_section({"code": {"commit": "b" * 40, "dirty": True}}))
    clean = "\n".join(hf._provenance_section({"code": {"commit": "b" * 40, "dirty": False}}))
    assert "does not describe the code that ran" in dirty
    assert "does not describe the code that ran" not in clean
    assert not hf._provenance_section(None)


def test_run_provenance_prefers_the_copy_that_travels(tmp_path: Path) -> None:
    from quantik_models.export import huggingface as hf

    run = tmp_path / "run"
    best = run / "best"
    best.mkdir(parents=True)
    (run / "provenance.json").write_text(json.dumps({"code": {"commit": "stale"}}))
    (best / "training-report.json").write_text(
        json.dumps({"provenance": {"code": {"commit": "travelling"}}})
    )
    assert hf.run_provenance(best)["code"]["commit"] == "travelling"

    # Falls back to the run directory when the report carries none...
    (best / "training-report.json").write_text(json.dumps({"run": "r"}))
    assert hf.run_provenance(best)["code"]["commit"] == "stale"
    # ...and to None for a checkpoint trained before any of this existed.
    (run / "provenance.json").unlink()
    assert hf.run_provenance(best) is None
