"""Regression tests for codex_code_review_146f98e.md -- the seventh
hardening round (a single, focused finding: content already hashed
before an interruption was not checkpointed, so replacing that file's
bytes at the same path before resume was silently accepted).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from ft_screening.components import Components
from ft_screening.errors import ResumeConflictError
from ft_screening.fakes import FakeCriterionAdjudicator, FakeDocumentParser, FakeEvidenceRetriever
import ft_screening.orchestrator as orchestrator_module
from ft_screening.orchestrator import _run_ft_screening
from ft_screening.schema import CriteriaSet, CriterionSpec, ScreeningConfig, ScreeningItem, ScreeningRequest
from ft_screening.store import FileScreeningStore


def make_pdf(tmp_path: Path, name: str, content: bytes) -> Path:
    path = tmp_path / "pdfs" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def make_criteria_set() -> CriteriaSet:
    return CriteriaSet(
        criteria_set_id="cs-1", version="1",
        criteria=[
            CriterionSpec(
                criterion_id="population", version="1", eligibility_requirement="adult humans",
                missing_evidence_policy="review", exclusion_reason_code="wrong_population",
                exclusion_reason_label="Wrong population", priority=1,
            )
        ],
    )


def make_config() -> ScreeningConfig:
    return ScreeningConfig(
        parser_name="fake", parser_version="1", chunker_version="1", retriever_name="fake",
        retriever_config_version="1", adjudicator_provider="fake", adjudicator_model="fake-1",
        model_config_id="cfg-1", prompt_template_version="1", output_schema_version="1",
        aggregation_policy_version="1",
    )


def make_components(output_dir: Path, store: FileScreeningStore) -> Components:
    return Components(
        parser=FakeDocumentParser(), retriever=FakeEvidenceRetriever(), adjudicator=FakeCriterionAdjudicator(),
        store=store,
    )


def test_finding_changed_bytes_at_same_path_after_checkpoint_is_rejected(tmp_path, monkeypatch):
    """Codex's exact reproduction: two-PDF run, first file hashes
    successfully, interrupt while hashing the second, replace the FIRST
    file's bytes in place (same path, same document_id/report_id/
    citation -- provisional_request_binding is unchanged), then resume.
    Must be rejected, not silently accepted under the new (different)
    content."""
    pdf1 = make_pdf(tmp_path, "a.pdf", b"content-a-original")
    pdf2 = make_pdf(tmp_path, "b.pdf", b"content-b")
    output_dir = tmp_path / "outputs"
    items = [
        ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1),
        ScreeningItem(document_id="d2", report_id="r2", pdf_path=pdf2),
    ]
    request = ScreeningRequest(
        items=items, criteria_set=make_criteria_set(), config=make_config(),
        output_dir=output_dir, run_id="checkpoint-tamper-run",
    )

    real_hash_fn = orchestrator_module._hash_file_streaming

    def interrupting_hash(path, **kwargs):
        if "b.pdf" in str(path):
            raise KeyboardInterrupt()
        return real_hash_fn(path, **kwargs)

    monkeypatch.setattr(orchestrator_module, "_hash_file_streaming", interrupting_hash)
    store = FileScreeningStore(output_dir=output_dir)
    with pytest.raises(KeyboardInterrupt):
        _run_ft_screening(request, make_components(output_dir, store))
    monkeypatch.setattr(orchestrator_module, "_hash_file_streaming", real_hash_fn)

    manifest = store.read_run_manifest(run_id="checkpoint-tamper-run")
    checkpoints = manifest.get("document_hash_checkpoints")
    assert checkpoints and "d1" in checkpoints  # the first file's hash survived the interrupt
    assert "d2" not in checkpoints  # the second file was never reached

    # Replace the FIRST file's bytes at the SAME path -- the path-based
    # provisional_request_binding alone would not catch this.
    pdf1.write_bytes(b"content-a-REPLACED")

    resume_request = ScreeningRequest(
        items=items, criteria_set=make_criteria_set(), config=make_config(),
        output_dir=output_dir, resume_run_id="checkpoint-tamper-run",
    )
    with pytest.raises(ResumeConflictError):
        _run_ft_screening(resume_request, make_components(output_dir, store))


def test_finding_unchanged_content_still_resumes_cleanly(tmp_path, monkeypatch):
    """The checkpoint check must not create false positives -- resuming
    with the SAME bytes at the SAME path must still work."""
    pdf1 = make_pdf(tmp_path, "a.pdf", b"content-a-original")
    pdf2 = make_pdf(tmp_path, "b.pdf", b"content-b")
    output_dir = tmp_path / "outputs"
    items = [
        ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1),
        ScreeningItem(document_id="d2", report_id="r2", pdf_path=pdf2),
    ]
    request = ScreeningRequest(
        items=items, criteria_set=make_criteria_set(), config=make_config(),
        output_dir=output_dir, run_id="checkpoint-clean-resume-run",
    )

    real_hash_fn = orchestrator_module._hash_file_streaming

    def interrupting_hash(path, **kwargs):
        if "b.pdf" in str(path):
            raise KeyboardInterrupt()
        return real_hash_fn(path, **kwargs)

    monkeypatch.setattr(orchestrator_module, "_hash_file_streaming", interrupting_hash)
    store = FileScreeningStore(output_dir=output_dir)
    with pytest.raises(KeyboardInterrupt):
        _run_ft_screening(request, make_components(output_dir, store))
    monkeypatch.setattr(orchestrator_module, "_hash_file_streaming", real_hash_fn)

    resume_request = ScreeningRequest(
        items=items, criteria_set=make_criteria_set(), config=make_config(),
        output_dir=output_dir, resume_run_id="checkpoint-clean-resume-run",
    )
    result = _run_ft_screening(resume_request, make_components(output_dir, store))
    assert result.run_status == "completed"


def test_finding_subprocess_crash_then_tampered_content_is_rejected(tmp_path):
    """Real cross-process crash equivalent -- the crash-helper's existing
    "mid_hashing" mode already crashes after hashing the FIRST of two
    files and before the second, which is exactly the state this finding
    is about; this test tampers with the first file afterward, in the
    parent process, and asserts resume is rejected."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True)
    helper = Path(__file__).parent / "_crash_helper.py"

    proc = subprocess.run(
        [sys.executable, str(helper), str(output_dir), "checkpoint-subprocess-run", "mid_hashing"],
        capture_output=True, text=True, timeout=30,
    )
    assert "crashing" in proc.stdout

    store = FileScreeningStore(output_dir=output_dir)
    manifest = store.read_run_manifest(run_id="checkpoint-subprocess-run")
    assert manifest is not None
    checkpoints = manifest.get("document_hash_checkpoints")
    assert checkpoints and "d1" in checkpoints

    # Tamper with the first file's bytes at the same path the crashed
    # child process already hashed.
    (output_dir / "pdfs" / "a.pdf").write_bytes(b"tampered-after-crash")

    pdf1 = output_dir / "pdfs" / "a.pdf"
    pdf2 = output_dir / "pdfs" / "b.pdf"
    items = [
        ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1),
        ScreeningItem(document_id="d2", report_id="r2", pdf_path=pdf2),
    ]
    # Must match _crash_helper.py's "mid_hashing" mode's own request
    # construction exactly (execution_fingerprint must match, or resume is
    # rejected for a DIFFERENT reason than the one this test targets).
    subprocess_criteria_set = CriteriaSet(
        criteria_set_id="cs", version="1",
        criteria=[
            CriterionSpec(
                criterion_id="population", version="1", eligibility_requirement="adults",
                missing_evidence_policy="review", exclusion_reason_code="wrong_population",
                exclusion_reason_label="Wrong population", priority=1,
            )
        ],
    )
    resume_request = ScreeningRequest(
        items=items, criteria_set=subprocess_criteria_set, config=make_config(),
        output_dir=output_dir, resume_run_id="checkpoint-subprocess-run",
    )
    with pytest.raises(ResumeConflictError):
        _run_ft_screening(resume_request, make_components(output_dir, store))
