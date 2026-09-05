"""Assert a built sdist and wheel contain what they should — and nothing else.

Run after `python -m build`, in CI and before any manual upload:

    python scripts/check_dist.py dist/

`twine check` validates the metadata renders. It says nothing about the
payload, and the payload is where the irreversible mistakes are: a wheel
that omits `py.typed` ships a package type checkers silently treat as
untyped, and an sdist that picks up `runs/` uploads tens of gigabytes of
corpora and checkpoints to PyPI, under the wrong licence, permanently.
PyPI does not allow re-uploading a filename, so both are found here or
after the fact.
"""

from __future__ import annotations

import sys
import tarfile
import zipfile
from pathlib import Path

# Directories that must never appear in either distribution. `runs/` is the
# one that matters — it is gitignored, it holds every corpus and checkpoint,
# and the weights in it carry a different licence from this package.
FORBIDDEN_PREFIXES = ("runs/", "staging/", "docker/staging/", ".venv/", ".git/")
FORBIDDEN_SUFFIXES = (".npz", ".parquet", ".jsonl", ".onnx", ".pyc")

# Present in the wheel or the package is not what it claims to be.
REQUIRED_IN_WHEEL = ("quantik_models/py.typed", "quantik_models/hub.py")

# Present in the sdist or its test suite cannot run.
REQUIRED_IN_SDIST = (
    "tests/fixtures/checkpoints/smoke-best/manifest.json",
    "tests/fixtures/checkpoints/smoke-best/weights.safetensors",
    "CHANGELOG.md",
    "LICENSE",
)


def _wheel_names(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        return archive.namelist()


def _sdist_names(path: Path) -> list[str]:
    with tarfile.open(path) as archive:
        # Strip the `quantik_models-X.Y.Z/` prefix so the required-file list
        # does not have to know the version.
        return [n.split("/", 1)[1] for n in archive.getnames() if "/" in n]


def check(dist_dir: Path) -> list[str]:
    problems: list[str] = []
    wheels = sorted(dist_dir.glob("*.whl"))
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        return [
            f"expected exactly one wheel and one sdist in {dist_dir}, found "
            f"{len(wheels)} and {len(sdists)} — a stale artifact from an "
            "earlier version is the usual cause, and it uploads too"
        ]

    for label, names in (
        ("wheel", _wheel_names(wheels[0])),
        ("sdist", _sdist_names(sdists[0])),
    ):
        for name in names:
            if name.startswith(FORBIDDEN_PREFIXES):
                problems.append(f"{label} contains {name!r} (forbidden path)")
            if name.endswith(FORBIDDEN_SUFFIXES):
                problems.append(f"{label} contains {name!r} (forbidden file type)")

        required = REQUIRED_IN_WHEEL if label == "wheel" else REQUIRED_IN_SDIST
        for expected in required:
            if expected not in names:
                problems.append(f"{label} is missing {expected!r}")

    return problems


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    dist_dir = Path(args[0]) if args else Path("dist")
    problems = check(dist_dir)
    for problem in problems:
        print(f"  {problem}", file=sys.stderr)
    if problems:
        print(f"\n{len(problems)} problem(s) in {dist_dir}", file=sys.stderr)
        return 1
    print(f"{dist_dir} looks publishable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
