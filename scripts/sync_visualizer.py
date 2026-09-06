"""Sync the browser app from the sibling visualizer checkout.

Copies index.html and src/ into src/quantik_models/play/app/ so the
installed package can serve the app without a sibling checkout.

Usage:

    python scripts/sync_visualizer.py [source-dir]

The default source is ../quantik-qfen-visualizer (valid inside the
quantik-ns workspace root).

The script refuses to sync from a dirty tree — a recorded commit hash
that does not match the actual bytes is worse than no record.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path


def _git_commit(src: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(src), "rev-parse", "HEAD"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"git failed: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def _check_clean(src: Path) -> None:
    result = subprocess.run(
        ["git", "-C", str(src), "status", "--porcelain"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"git failed: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    if result.stdout.strip():
        print(f"Refusing to sync — {src} has uncommitted changes", file=sys.stderr)
        print("Commit or stash them first.", file=sys.stderr)
        sys.exit(1)


def sync(source_dir: Path = Path("../quantik-qfen-visualizer")) -> None:
    dest = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "quantik_models"
        / "play"
        / "app"
    )

    if not source_dir.is_dir():
        print(f"Source directory does not exist: {source_dir}", file=sys.stderr)
        sys.exit(1)

    _check_clean(source_dir)
    commit = _git_commit(source_dir)

    # Clean destination — a removed file must not survive.
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    # Copy index.html and src/
    shutil.copy2(source_dir / "index.html", dest / "index.html")
    src_dir = dest / "src"
    src_dir.mkdir(exist_ok=True)
    for child in (source_dir / "src").iterdir():
        if child.is_dir():
            shutil.copytree(child, src_dir / child.name)
        else:
            shutil.copy2(child, src_dir / child.name)

    # Write provenance
    source_json = dest / "SOURCE.json"
    source_json.write_text(
        json.dumps(
            {
                "repository": source_dir.name,
                "commit": commit,
                "synced": date.today().isoformat(),
            },
            indent=2,
        )
        + "\n"
    )

    print(f"Synced {source_dir.name} {commit[:8]} -> {dest}")
    print(f"  index.html  {(dest / 'index.html').stat().st_size} bytes")
    src_count = sum(1 for _ in (dest / "src").iterdir())
    print(f"  src/        {src_count} entries")


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    source = Path(args[0]) if args else Path("../quantik-qfen-visualizer")
    sync(source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
