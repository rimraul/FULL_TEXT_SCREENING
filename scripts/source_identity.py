"""Captures a self-documenting source-identity record for a calibration/
evaluation run -- git commit (when available), whether the RELEVANT files
are dirty relative to that commit (a commit hash alone is insufficient if
the files that actually determine behavior have uncommitted changes), a
SHA-256 of the adjudication source specifically, and a combined digest over
a small, explicitly-named set of behavior-relevant source files. Never
raises on a missing git binary/non-repo checkout -- git_commit/git_dirty
become None rather than failing the whole capture, since the file hashes
(the actually load-bearing identity) don't depend on git being present at
all.
"""
from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT = Path(r"C:/projects/FT_SCREENING")

# The exact, explicitly-named file set that determines adjudication
# behavior: the prompt/schema/response-validation logic itself
# (adjudication.py), the usage/telemetry and evidence/document shapes it
# depends on (components.py), and the canonical-JSON serialization the
# prompt construction reuses (hashing.py). Deliberately narrow and named,
# not "the whole src tree" -- an unrelated change elsewhere (e.g. the
# gold-set sealing machinery) must not make this identity look dirty for a
# reason that has nothing to do with what was actually sent to the model.
RELEVANT_SOURCE_FILES: tuple[str, ...] = (
    "src/ft_screening/adjudication.py",
    "src/ft_screening/components.py",
    "src/ft_screening/hashing.py",
)


class SourceIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    git_commit: str | None = None
    # None only when git_commit itself is None (git unavailable/not a
    # repo) -- whenever a commit IS known, dirty state is always
    # determined, never left ambiguous.
    git_dirty_relevant_files: bool | None = None
    prompt_template_version: str = Field(min_length=1)
    output_schema_version: str = Field(min_length=1)
    adjudication_file_sha256: str = Field(min_length=64, max_length=64)
    relevant_source_tree_sha256: str = Field(min_length=64, max_length=64)
    relevant_source_files: tuple[str, ...] = Field(min_length=1)
    captured_at: str = Field(min_length=1)


def _run_git(args: list[str]) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def capture_source_identity() -> SourceIdentity:
    from ft_screening.adjudication import LLMCriterionAdjudicator
    from ft_screening.hashing import canonical_json_bytes

    commit_output = _run_git(["rev-parse", "HEAD"])
    git_commit = commit_output.strip() if commit_output is not None else None

    git_dirty: bool | None = None
    if git_commit is not None:
        status_output = _run_git(["status", "--porcelain", "--", *RELEVANT_SOURCE_FILES])
        # A real git failure here (git worked for rev-parse but not
        # status) is different from "clean" -- surface as unknown (None)
        # rather than silently claiming clean.
        git_dirty = bool(status_output.strip()) if status_output is not None else None

    file_hashes: dict[str, str] = {}
    for rel_path in RELEVANT_SOURCE_FILES:
        path = REPO_ROOT / rel_path
        file_hashes[rel_path] = hashlib.sha256(path.read_bytes()).hexdigest()

    tree_payload = {"files": dict(sorted(file_hashes.items()))}
    tree_digest = hashlib.sha256(canonical_json_bytes(tree_payload)).hexdigest()

    return SourceIdentity(
        git_commit=git_commit,
        git_dirty_relevant_files=git_dirty,
        prompt_template_version=LLMCriterionAdjudicator.PROMPT_TEMPLATE_VERSION,
        output_schema_version=LLMCriterionAdjudicator.OUTPUT_SCHEMA_VERSION,
        adjudication_file_sha256=file_hashes["src/ft_screening/adjudication.py"],
        relevant_source_tree_sha256=tree_digest,
        relevant_source_files=tuple(sorted(file_hashes)),
        captured_at=datetime.now(timezone.utc).isoformat(),
    )
