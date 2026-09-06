"""Vendor app assets are complete and resolvable.

Checks that the vendored `index.html` and every resource it references
actually exist inside the package. The goal is to catch a partial sync
that would ship a blank page.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from quantik_models.play.__main__ import DEFAULT_STATIC


def test_default_static_exists() -> None:
    assert DEFAULT_STATIC.is_dir(), f"DEFAULT_STATIC is not a directory: {DEFAULT_STATIC}"


def test_index_html_present() -> None:
    idx = DEFAULT_STATIC / "index.html"
    assert idx.is_file(), f"index.html missing from {DEFAULT_STATIC}"


def test_source_json_parses() -> None:
    src = DEFAULT_STATIC / "SOURCE.json"
    assert src.is_file(), f"SOURCE.json missing from {DEFAULT_STATIC}"
    data = json.loads(src.read_text())
    assert "commit" in data
    assert "repository" in data
    assert "synced" in data


def test_source_json_commit_is_sha() -> None:
    src = DEFAULT_STATIC / "SOURCE.json"
    data = json.loads(src.read_text())
    commit = data["commit"]
    assert len(commit) == 40, f"expected 40-char commit, got {len(commit)}"
    assert re.fullmatch(r"[0-9a-f]{40}", commit), f"commit is not a hex SHA: {commit}"


def test_vendored_script_refs_resolve() -> None:
    """Every <script src> in index.html points to a real file."""
    idx = DEFAULT_STATIC / "index.html"
    html = idx.read_text()

    srcs = re.findall(r'<script\b[^>]+src=["\']([^"\']+)["\']', html)
    for ref in srcs:
        target = DEFAULT_STATIC / ref
        assert target.is_file(), f"script ref {ref!r} not found at {target}"


def test_vendored_link_refs_resolve() -> None:
    """Every <link href> in index.html points to a real file."""
    idx = DEFAULT_STATIC / "index.html"
    html = idx.read_text()

    hrefs = re.findall(r'<link\b[^>]+href=["\']([^"\']+)["\']', html)
    for ref in hrefs:
        target = DEFAULT_STATIC / ref
        assert target.is_file(), f"link ref {ref!r} not found at {target}"
