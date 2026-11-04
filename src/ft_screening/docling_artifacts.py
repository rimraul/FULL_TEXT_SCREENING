"""Pinning and integrity verification for Docling's model artifacts
(codex_code_review_af44129.md finding 1; codex_code_review_afeba09.md
finding 1 tightened this further -- see that finding for what changed).

Docling resolves its models from 4 upstream Hugging Face repositories.
Inspecting Docling's own model classes directly (not assumed) showed only
2 of the 4 are left at a mutable `revision="main"` by Docling itself --
`TableStructureModel`/`docling-project/docling-models` is pinned to tag
`v2.3.0`, `DocumentPictureClassifier`/`docling-project/DocumentFigureClassifier`
to `v1.0.1`; only `LayoutModel`/`docling-project/docling-layout-heron` and
`CodeFormulaModel`/`docling-project/CodeFormulaV2` use `revision="main"`.
RapidOCR's models are fetched from versioned ModelScope release URLs
(`.../resolve/v3.4.0/...`), not an HF revision at all -- already
effectively immutable as long as the installed `docling`/`rapidocr`
package versions are pinned (see `pyproject.toml`).

This module closes the remaining gap three ways:
1. `bootstrap()` downloads ALL FOUR Hugging Face repos at an EXPLICIT,
   hardcoded commit via `huggingface_hub.snapshot_download` directly --
   codex_code_review_afeba09.md finding 1 pointed out that delegating the
   already-tag-pinned two to Docling's own download methods meant the
   BOOTSTRAP itself still wasn't fully commit-addressed, even though the
   tree digest would eventually catch a moved tag after the fact.
   Bootstraps `validate()`s the completed bundle at the end, rather than
   waiting for first use to discover a bad download.
2. `compute_tree_digest()`/`validate()` provide a canonical content digest
   of the whole artifacts directory, checked against
   `docling_artifacts_manifest.json` (committed to the repo) before a
   `DocumentConverter` is ever constructed -- catches an empty directory,
   a missing file, a modified/corrupted file, or a mixed bundle from a
   different bootstrap, not just "the directory exists". `validate()`
   accepts an explicit `expected_tree_digest` override so tests can check
   missing-file/modified-file cases against a tiny synthetic directory,
   never needing to touch the real ~1.2 GB bundle.
3. `document.py`'s `parser_options["artifacts_manifest_digest"]` is now
   MANDATORY (codex_code_review_afeba09.md finding 1): a caller MUST
   read `load_manifest()["expected_tree_digest"]` (or use
   `DoclingDocumentParser.recommended_parser_options()`) and declare it,
   so ordinary requests -- not just ones that opt in -- get the model
   identity bound into `execution_fingerprint`/`criterion_cache_key`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

_MANIFEST_PATH = Path(__file__).parent / "docling_artifacts_manifest.json"

# Resolved 2026-08-13 from the Hugging Face cache metadata of a download
# (each repo's `.cache/huggingface/download/*.metadata` first line is the
# commit the download actually resolved to, whether requested via "main"
# or a tag) -- pinning ALL FOUR repos to these exact commits, not
# re-resolving "main" OR a tag (which can be force-moved upstream) on
# every bootstrap. The latter two were already tag-pinned in Docling's
# own code (v2.3.0, v1.0.1) -- pinned here too so the BOOTSTRAP itself is
# fully commit-addressed (codex_code_review_afeba09.md finding 1), not
# just verified after the fact by the tree digest.
PINNED_REVISIONS: dict[str, str] = {
    "docling-project/docling-layout-heron": "8f39ad3c0b4c58e9c2d2c84a38465abf757272d8",
    "docling-project/CodeFormulaV2": "ecedbe111d15c2dc60bfd4a823cbe80127b58af4",
    "docling-project/docling-models": "fc0f2d45e2218ea24bce5045f58a389aed16dc23",  # tag v2.3.0 at bootstrap time
    "docling-project/DocumentFigureClassifier": "12b08ac4e56f280a2a9a1ad920eacc1af60f6a72",  # tag v1.0.1 at bootstrap time
}

# Maps each pinned repo to the local directory name Docling itself expects
# it under (must match Docling's own `*_model_repo_folder`/
# `model_spec.model_repo_folder` conventions exactly, or DocumentConverter
# won't find the files) -- hardcoded rather than importing Docling's
# classes just to read a folder-name constant, since bootstrap() already
# needs those classes for RapidOCR anyway; kept as plain strings here so
# `PINNED_REVISIONS`'s keys stay the single source of truth for "which
# repos are pinned" independent of Docling's own import cost.
_LOCAL_DIR_NAMES: dict[str, str] = {
    "docling-project/docling-layout-heron": "docling-project--docling-layout-heron",
    "docling-project/CodeFormulaV2": "docling-project--CodeFormulaV2",
    "docling-project/docling-models": "docling-project--docling-models",
    "docling-project/DocumentFigureClassifier": "docling-project--DocumentFigureClassifier",
}

_EXCLUDED_PATH_SEGMENTS = (".cache", ".locks")


class ArtifactValidationError(Exception):
    """Raised when a local Docling artifacts directory doesn't match the
    committed manifest -- missing, empty, or content-modified."""


def load_manifest() -> dict:
    return json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))


def compute_tree_digest(artifacts_path: Path) -> str:
    """Canonical content digest of every actual model file under
    `artifacts_path` -- excludes `huggingface_hub`'s own bookkeeping
    (`.cache/huggingface/...`, `.locks/...`), which can legitimately vary
    across otherwise-identical downloads (timestamps, lock files) and
    would make the digest spuriously unstable if included."""
    entries: list[str] = []
    for path in sorted(artifacts_path.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(artifacts_path)
        if any(part in _EXCLUDED_PATH_SEGMENTS for part in relative.parts):
            continue
        file_hash = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(1 << 20):
                file_hash.update(chunk)
        entries.append(f"{relative.as_posix()}:{file_hash.hexdigest()}")
    tree_hash = hashlib.sha256("\n".join(sorted(entries)).encode("utf-8"))
    return tree_hash.hexdigest()


def validate(artifacts_path: Path, *, expected_tree_digest: str | None = None) -> str:
    """Computes the live tree digest of `artifacts_path` and checks it
    against `expected_tree_digest` (defaults to the committed manifest's
    `expected_tree_digest` when not given). Returns the digest on success
    (bind this to parser identity via `parser_options`); raises
    `ArtifactValidationError` on any mismatch -- empty directory, missing
    file, modified file, or a bundle from an entirely different source
    all produce a different digest and are rejected identically, with a
    message naming both the expected and actual digest.

    The `expected_tree_digest` override exists so tests can exercise
    missing-file/modified-file cases against a small synthetic directory
    (codex_code_review_afeba09.md finding 3) without ever touching the
    real ~1.2 GB bundle or the committed manifest."""
    if expected_tree_digest is None:
        expected_tree_digest = load_manifest()["expected_tree_digest"]
    actual = compute_tree_digest(artifacts_path)
    if actual != expected_tree_digest:
        raise ArtifactValidationError(
            f"Docling model artifacts at {artifacts_path!s} do not match the expected tree digest "
            f"{expected_tree_digest!r} (got {actual!r}). This means the directory is empty, missing files, "
            "contains modified/corrupted files, or was populated from a different bootstrap than the one "
            "this project pins. Re-run the bootstrap command in CLAUDE.md's Environment section."
        )
    return actual


def bootstrap(output_dir: Path, *, progress: bool = False, force: bool = False) -> str:
    """One-time model download, fully commit-pinned: ALL FOUR Hugging
    Face repos are fetched via `huggingface_hub.snapshot_download(...,
    revision=<explicit commit>)` directly -- codex_code_review_afeba09.md
    finding 1: bypassing Docling's own tag-based download methods for the
    two it already pins by tag too, not just the two it leaves at "main",
    so the bootstrap itself is commit-addressed rather than only
    caught-after-the-fact by the tree digest. RapidOCR still goes through
    Docling's own downloader (already immutable, versioned release URLs,
    no HF revision concept applies). Validates the completed bundle
    against the committed manifest before returning -- a bad/incomplete
    download is caught immediately, not deferred to first parse. Returns
    the validated digest. Requires network access -- see CLAUDE.md's
    Environment section for the CA-bundle workaround needed to reach
    Hugging Face Hub/ModelScope from this machine.
    """
    from huggingface_hub import snapshot_download

    from docling.utils.model_downloader import RapidOcrModel

    output_dir.mkdir(parents=True, exist_ok=True)

    for repo_id, revision in PINNED_REVISIONS.items():
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=output_dir / _LOCAL_DIR_NAMES[repo_id],
            force_download=force,
        )

    for backend in ("torch", "onnxruntime"):
        RapidOcrModel.download_models(
            backend=backend, local_dir=output_dir / RapidOcrModel._model_repo_folder, force=force, progress=progress
        )

    return validate(output_dir)
