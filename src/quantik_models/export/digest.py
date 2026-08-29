"""Content hashing for exported artifacts.

Its own module because `checkpoint.py` imports torch at module scope, and
neither staging a directory for the Hub nor verifying a published digest
needs a tensor library — both are file operations, and requiring torch for
them would make the check unavailable exactly where it matters, on a
machine that only has the files.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def file_digest(path: Path) -> str:
    """`sha256:<hex>` over the file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"
