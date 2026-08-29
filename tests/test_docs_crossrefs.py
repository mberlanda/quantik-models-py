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
# This repo sits in a multi-repo workspace and its docs legitimately point
# at siblings — `quantik-core-contracts/docs/policy-value-model-project.md`,
# for instance. Resolving against the workspace root as well is what makes
# this check correct rather than merely strict; a first version without it
# flagged five valid references.
WORKSPACE = REPO.parent

# `name.md` in backticks, or a markdown link target. Deliberately not a full
# markdown parser: the failure mode is a stale filename, and a regex over
# backticked names catches it without pulling in a dependency.
BACKTICKED = re.compile(r"`([A-Za-z0-9._/-]+\.md)`")
LINKED = re.compile(r"\]\(([A-Za-z0-9._/-]+\.md)\)")
SOURCE_PATH = re.compile(r"`((?:src|tests|scripts)/[A-Za-z0-9._/-]+\.py)`")


# Dated design records and journals from past work. They describe what was
# true when they were written, so a path that has since moved is part of the
# record rather than a defect — correcting them would falsify the history
# they exist to preserve. Live documentation is checked.
ARCHIVAL = ("superpowers/", "nn-quest/")


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

    bases = (doc.parent, DOCS, REPO, WORKSPACE)
    missing = [n for n in sorted(referenced) if not any((b / n).exists() for b in bases)]

    assert not missing, (
        f"{doc.relative_to(REPO)} references {missing}, which do not exist "
        f"relative to the file, docs/, the repo root, or the workspace root"
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
