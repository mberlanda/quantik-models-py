"""Every document this project points at has to exist.

A merged PR shipped `shift-evaluation.md` saying "architecture-constraint-pool.md
has the full table" when that section had never been written — the edit sat
behind a failed assertion in a multi-file script, the script exited, and the
files it *had* written were correct, so the commit looked clean.

Prose is not type-checked and nothing else in this suite reads it. This is the
cheapest guard that would have caught it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"
REPO = DOCS.parent

# References this repo cannot verify, and must not pretend to.
#
# `runs/` is gitignored training output and `quantik-*/` are sibling repos
# in the workspace. Both exist on a developer's machine and neither exists
# on a runner, so resolving them against the filesystem makes the check
# pass locally and fail in CI — which a first version of this file did,
# and which is a worse failure than the one it was written to catch.
#
# The line is drawn at "is this a path into this repository": those are
# checkable anywhere, and nothing else is.
UNVERIFIABLE = ("runs/", "quantik-")

# Dated design records and journals from past work. They describe what was
# true when they were written, so a path that has since moved is part of the
# record rather than a defect — correcting them would falsify the history
# they exist to preserve. Live documentation is checked.
ARCHIVAL = ("superpowers/", "nn-quest/")

# `name.md` in backticks, or a markdown link target. Deliberately not a full
# markdown parser: the failure mode is a stale filename, and a regex over
# backticked names catches it without pulling in a dependency.
BACKTICKED = re.compile(r"`([A-Za-z0-9._/-]+\.md)`")
LINKED = re.compile(r"\]\(([A-Za-z0-9._/-]+\.md)\)")
SOURCE_PATH = re.compile(r"`((?:src|tests|scripts)/[A-Za-z0-9._/-]+\.py)`")
# Embedded images. The figures are generated into `docs/figures/` and
# committed, so a renamed figure breaks a document exactly the way a renamed
# module does, and just as invisibly — the alt text still reads fine.
IMAGE = re.compile(r"!\[[^\]]*\]\(([A-Za-z0-9._/-]+\.(?:svg|png))\)")




def _markdown_files() -> list[Path]:
    return sorted(
        doc
        for doc in DOCS.rglob("*.md")
        if not any(part in doc.relative_to(DOCS).as_posix() for part in ARCHIVAL)
    )


def test_there_are_docs_to_check() -> None:
    """Guards the guard: a glob that matches nothing passes everything."""
    assert len(_markdown_files()) >= 5


@pytest.mark.parametrize("doc", _markdown_files(), ids=lambda p: p.name)
def test_referenced_documents_exist(doc: Path) -> None:
    text = doc.read_text()
    referenced = set(BACKTICKED.findall(text)) | set(LINKED.findall(text))

    checkable = [n for n in sorted(referenced) if not n.startswith(UNVERIFIABLE)]
    bases = (doc.parent, DOCS, REPO)
    missing = [n for n in checkable if not any((b / n).exists() for b in bases)]

    assert not missing, (
        f"{doc.relative_to(REPO)} references {missing}, which do not exist "
        f"relative to the file, docs/, or the repo root"
    )


@pytest.mark.parametrize("doc", _markdown_files(), ids=lambda p: p.name)
def test_referenced_source_paths_exist(doc: Path) -> None:
    """The same check for `src/...` paths quoted in prose.

    These go stale the same way and are just as invisible — a renamed module
    leaves the docs pointing at nothing.
    """
    missing = [
        path
        for path in sorted(set(SOURCE_PATH.findall(doc.read_text())))
        if not (REPO / path).exists()
    ]
    assert not missing, f"{doc.relative_to(REPO)} references missing files: {missing}"


@pytest.mark.parametrize("doc", _markdown_files(), ids=lambda p: p.name)
def test_embedded_figures_exist(doc: Path) -> None:
    """Every `![...](...)` target resolves.

    A figure is generated output that happens to be committed, so it can go
    missing in a way prose cannot: regenerate under a new name and the old
    document still renders, just with a broken image where the evidence was.
    """
    bases = (doc.parent, DOCS, REPO)
    missing = [
        target
        for target in sorted(set(IMAGE.findall(doc.read_text())))
        if not any((base / target).exists() for base in bases)
    ]
    assert not missing, f"{doc.relative_to(REPO)} embeds missing figures: {missing}"


def test_the_skip_rule_does_not_swallow_the_check() -> None:
    """Guard the guard.

    `UNVERIFIABLE` exists so the check behaves the same on a runner as on a
    developer's machine. A prefix list that grew until it matched
    everything would achieve that by checking nothing.
    """
    referenced = set()
    for doc in _markdown_files():
        text = doc.read_text()
        referenced |= set(BACKTICKED.findall(text)) | set(LINKED.findall(text))

    checked = [n for n in referenced if not n.startswith(UNVERIFIABLE)]
    assert len(checked) >= 10, (
        f"only {len(checked)} of {len(referenced)} references are being "
        "checked; the skip list has grown too broad"
    )
