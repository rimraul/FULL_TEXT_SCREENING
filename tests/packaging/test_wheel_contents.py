"""codex_code_review_afeba09.md finding 1: a real wheel build was
inspected directly and confirmed `docling_artifacts.py` was shipped but
`docling_artifacts_manifest.json` was NOT -- setuptools does not include
non-.py files inside a package by default, and only explicit
`[tool.setuptools.package-data]` (added to pyproject.toml alongside this
test) fixes it. Every prior test ran against the source/editable checkout,
which has the file sitting right next to the module regardless of
packaging config -- masking this entirely.

Deliberately NOT marked `@pytest.mark.slow`: building a wheel offline
(`--no-deps --no-build-isolation`, no network, no Docling import needed)
takes a few seconds, not minutes, and this is exactly the kind of
regression worth catching on every default test run, not an opt-in one.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent


@pytest.fixture(scope="module")
def wheel_path(tmp_path_factory) -> Path:
    """Built ONCE and shared across both tests below -- a wheel build
    takes real time even offline (~7s measured); no need to pay it twice."""
    out_dir = tmp_path_factory.mktemp("dist")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", str(PROJECT_ROOT), "--no-deps", "--no-build-isolation", "-w", str(out_dir)],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"wheel build failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    wheels = list(out_dir.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"
    return wheels[0]


def test_wheel_includes_docling_artifacts_manifest(wheel_path):
    with zipfile.ZipFile(wheel_path) as z:
        names = z.namelist()
    assert any(n.endswith("ft_screening/docling_artifacts_manifest.json") for n in names), (
        f"docling_artifacts_manifest.json missing from wheel contents: {sorted(names)}"
    )
    assert any(n.endswith("ft_screening/docling_artifacts.py") for n in names)


def test_wheel_manifest_loads_and_validates_a_controlled_fixture(wheel_path, tmp_path):
    """Extracts the built wheel (no full `pip install` needed -- the
    functions under test here are pure stdlib, no pydantic/docling
    required) and imports `ft_screening.docling_artifacts` FROM the
    extracted wheel specifically, proving the shipped manifest is both
    present AND loadable/usable, not just present as inert bytes."""
    extract_dir = tmp_path / "extracted"
    with zipfile.ZipFile(wheel_path) as z:
        z.extractall(extract_dir)

    module_path = extract_dir / "ft_screening" / "docling_artifacts.py"
    assert module_path.exists()
    spec = importlib.util.spec_from_file_location("ft_screening_wheel_docling_artifacts", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    manifest = module.load_manifest()
    assert "expected_tree_digest" in manifest
    assert len(manifest["expected_tree_digest"]) == 64  # a real sha256 hex digest, not empty/placeholder

    # Validate against a small, controlled, synthetic tree -- not the real
    # ~1.2 GB bundle -- proving validate() itself works from the wheel's
    # own extracted copy of the module.
    fixture_dir = tmp_path / "fixture_artifacts"
    fixture_dir.mkdir()
    (fixture_dir / "a.bin").write_bytes(b"hello")
    expected = module.compute_tree_digest(fixture_dir)
    validated = module.validate(fixture_dir, expected_tree_digest=expected)
    assert validated == expected

    (fixture_dir / "a.bin").write_bytes(b"tampered")
    try:
        module.validate(fixture_dir, expected_tree_digest=expected)
    except module.ArtifactValidationError:
        pass
    else:
        raise AssertionError("validate() must reject a modified file, even from the wheel's own copy")
