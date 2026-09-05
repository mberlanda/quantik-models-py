"""The package's own metadata, checked the way its code is.

Every assertion here stands for a way a release goes out wrong while every
other test is green: a version that disagrees with its own changelog, a
console script naming a function somebody renamed, a `py.typed` that was
never added to `package-data` and so ships in the checkout and not in the
wheel. None of these break anything until a user installs the artifact,
which is after it is too late to change it.
"""

from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

import pytest

import quantik_models

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"


@pytest.fixture(scope="module")
def project() -> dict:
    if not PYPROJECT.exists():  # pragma: no cover - installed-package runs
        pytest.skip("pyproject.toml is not shipped in the wheel")
    return tomllib.loads(PYPROJECT.read_text())["project"]


# --- version -------------------------------------------------------------


def test_version_is_declared_once(project) -> None:
    # `version` must stay dynamic: the moment it is also written here, the
    # two can disagree and the wheel wins silently.
    assert "version" in project.get("dynamic", [])
    assert "version" not in project


def test_installed_metadata_matches_the_package(project) -> None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed = version("quantik-models")
    except PackageNotFoundError:  # pragma: no cover - not installed
        pytest.skip("quantik-models is not installed in this environment")
    assert installed == quantik_models.__version__


def test_version_is_semver() -> None:
    import re

    assert re.fullmatch(r"\d+\.\d+\.\d+([-.].+)?", quantik_models.__version__)


def test_changelog_documents_the_current_version() -> None:
    changelog = ROOT / "CHANGELOG.md"
    if not changelog.exists():  # pragma: no cover - installed-package runs
        pytest.skip("CHANGELOG.md is not shipped in the wheel")
    assert f"## {quantik_models.__version__} - " in changelog.read_text(), (
        f"CHANGELOG.md has no released section for "
        f"{quantik_models.__version__}; a release with no entry is a release "
        "nobody can read the diff of"
    )


# --- what the wheel has to carry ----------------------------------------


def test_py_typed_ships_with_the_package(project) -> None:
    marker = Path(quantik_models.__file__).parent / "py.typed"
    assert marker.exists(), "py.typed missing from the installed package"
    config = tomllib.loads(PYPROJECT.read_text())
    package_data = config["tool"]["setuptools"]["package-data"]
    assert "py.typed" in package_data["quantik_models"], (
        "py.typed exists in the tree but is not in package-data, so it is "
        "in the checkout and not in the wheel"
    )


def test_every_console_script_resolves(project) -> None:
    for name, target in project["scripts"].items():
        module_name, _, attribute = target.partition(":")
        module = importlib.import_module(module_name)
        assert callable(getattr(module, attribute, None)), (
            f"console script {name!r} points at {target!r}, which is not a "
            "callable — the script installs fine and fails on first run"
        )


# --- dependency and licence hygiene --------------------------------------


def test_runtime_dependencies_stay_minimal(project) -> None:
    # torch, onnxruntime, pyarrow, matplotlib and huggingface_hub are extras
    # on purpose: the torch-free layer is a tested configuration and the
    # Docker image depends on it. A dependency added here instead of to an
    # extra breaks that without failing anything.
    names = {d.split(">")[0].split("<")[0].split("=")[0].strip() for d in project["dependencies"]}
    assert names == {"numpy", "quantik-core"}


def test_optional_extras_cover_every_heavy_import(project) -> None:
    extras = project["optional-dependencies"]
    assert "torch" in extras and "onnx" in extras and "hub" in extras
    # `all` must actually mean all, or the developer install silently drifts
    # from the extras it claims to aggregate.
    aggregated = extras["all"][0]
    for extra in extras:
        if extra == "all":
            continue
        assert extra in aggregated, f"extra {extra!r} missing from [all]"


def test_licence_is_declared_as_an_spdx_expression(project) -> None:
    # PEP 639: the SPDX expression and the `License ::` classifier are
    # mutually exclusive, and setuptools>=77 refuses a build carrying both.
    assert project["license"] == "MIT"
    assert not any(c.startswith("License ::") for c in project["classifiers"])


def test_build_requires_a_setuptools_that_understands_pep_639() -> None:
    config = tomllib.loads(PYPROJECT.read_text())
    requires = config["build-system"]["requires"]
    setuptools_req = next(r for r in requires if r.startswith("setuptools"))
    floor = int(setuptools_req.split(">=")[1].split(".")[0])
    assert floor >= 77, (
        "pyproject uses PEP 639 `license`/`license-files`, which setuptools "
        "only supports from 77; a lower floor lets a resolver build a wheel "
        "with no licence metadata"
    )
