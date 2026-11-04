"""Regression tests for codex_code_review_810ccf0.md -- the sixth
hardening round. Named one group per finding, mirroring the convention
from tests/contract/test_hardening*.py.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

import pytest

from ft_screening.components import Components
from ft_screening.errors import ResumeConflictError
from ft_screening.fakes import FakeCriterionAdjudicator, FakeDocumentParser, FakeEvidenceRetriever
import ft_screening.orchestrator as orchestrator_module
from ft_screening.orchestrator import _run_ft_screening
from ft_screening.schema import (
    CriteriaSet,
    CriterionSpec,
    ScreeningConfig,
    ScreeningItem,
    ScreeningRequest,
)
from ft_screening.store import FileScreeningStore


def make_pdf(tmp_path: Path, name: str, content: bytes = b"%PDF-1.4 fake") -> Path:
    path = tmp_path / "pdfs" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def make_criteria_set(**overrides) -> CriteriaSet:
    criterion = CriterionSpec(
        criterion_id="population",
        version="1",
        eligibility_requirement="adult humans",
        missing_evidence_policy="review",
        exclusion_reason_code="wrong_population",
        exclusion_reason_label="Wrong population",
        priority=1,
    )
    defaults = dict(criteria_set_id="cs-1", version="1", criteria=[criterion])
    defaults.update(overrides)
    return CriteriaSet(**defaults)


def make_config(**overrides) -> ScreeningConfig:
    defaults = dict(
        parser_name="fake",
        parser_version="1",
        chunker_version="1",
        retriever_name="fake",
        retriever_config_version="1",
        adjudicator_provider="fake",
        adjudicator_model="fake-1",
        model_config_id="cfg-1",
        prompt_template_version="1",
        output_schema_version="1",
        aggregation_policy_version="1",
    )
    defaults.update(overrides)
    return ScreeningConfig(**defaults)


def make_components(output_dir: Path, *, parser=None, retriever=None, adjudicator=None, store=None, **adjudicator_kwargs) -> Components:
    return Components(
        parser=parser or FakeDocumentParser(),
        retriever=retriever or FakeEvidenceRetriever(),
        adjudicator=adjudicator or FakeCriterionAdjudicator(**adjudicator_kwargs),
        store=store or FileScreeningStore(output_dir=output_dir),
    )


# ---------------------------------------------------------------------------
# Finding 1: a provisional run must not be resumable as a different request
# ---------------------------------------------------------------------------


def _interrupt_mid_hashing(monkeypatch, *, on_filename: str):
    real_hash_fn = orchestrator_module._hash_file_streaming

    def interrupting_hash(path, **kwargs):
        if on_filename in str(path):
            raise KeyboardInterrupt()
        return real_hash_fn(path, **kwargs)

    monkeypatch.setattr(orchestrator_module, "_hash_file_streaming", interrupting_hash)
    return real_hash_fn


def test_finding1_changed_request_resume_of_a_provisional_run_is_rejected(tmp_path, monkeypatch):
    """codex_code_review_810ccf0.md's exact reproduction: start a run,
    interrupt during Stage A1 hashing (leaving only a PROVISIONAL manifest
    -- request_fingerprint still None), then resume the same run_id with
    an entirely different request (different PDF, document_id, report_id,
    citation, item_error_policy, max_workers). Must be rejected before any
    file hashing happens on the new request's behalf, not silently
    accepted because execution_fingerprint alone matched."""
    pdf1 = make_pdf(tmp_path, "a.pdf", b"content-a")
    pdf2 = make_pdf(tmp_path, "b.pdf", b"content-b")
    output_dir = tmp_path / "outputs"
    original_items = [
        ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1),
        ScreeningItem(document_id="d2", report_id="r2", pdf_path=pdf2),
    ]
    original_request = ScreeningRequest(
        items=original_items, criteria_set=make_criteria_set(), config=make_config(),
        output_dir=output_dir, run_id="provisional-takeover-run",
    )
    real_hash_fn = _interrupt_mid_hashing(monkeypatch, on_filename="b.pdf")
    store = FileScreeningStore(output_dir=output_dir)
    with pytest.raises(KeyboardInterrupt):
        _run_ft_screening(original_request, make_components(output_dir, store=store))
    monkeypatch.setattr(orchestrator_module, "_hash_file_streaming", real_hash_fn)

    manifest = store.read_run_manifest(run_id="provisional-takeover-run")
    assert manifest is not None
    assert manifest.get("request_fingerprint") is None  # still provisional

    other_pdf = make_pdf(tmp_path, "totally-different.pdf", b"unrelated-content")
    changed_request = ScreeningRequest(
        items=[ScreeningItem(document_id="totally-new", report_id="new-report", pdf_path=other_pdf)],
        criteria_set=make_criteria_set(),
        config=make_config(item_error_policy="fail_fast", max_workers=4),
        output_dir=output_dir,
        resume_run_id="provisional-takeover-run",
    )
    with pytest.raises(ResumeConflictError):
        _run_ft_screening(changed_request, make_components(output_dir, store=store))


def test_finding1_identical_request_resume_of_a_provisional_run_still_works(tmp_path, monkeypatch):
    """The flip side of the above -- a legitimate resume with the SAME
    request during the provisional window must still be accepted (this is
    what round 5's fix was for; finding 1 only tightens the check, it must
    not break the legitimate case)."""
    pdf1 = make_pdf(tmp_path, "a.pdf", b"content-a")
    pdf2 = make_pdf(tmp_path, "b.pdf", b"content-b")
    output_dir = tmp_path / "outputs"
    items = [
        ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1),
        ScreeningItem(document_id="d2", report_id="r2", pdf_path=pdf2),
    ]
    request = ScreeningRequest(
        items=items, criteria_set=make_criteria_set(), config=make_config(),
        output_dir=output_dir, run_id="provisional-legit-resume-run",
    )
    real_hash_fn = _interrupt_mid_hashing(monkeypatch, on_filename="b.pdf")
    store = FileScreeningStore(output_dir=output_dir)
    with pytest.raises(KeyboardInterrupt):
        _run_ft_screening(request, make_components(output_dir, store=store))
    monkeypatch.setattr(orchestrator_module, "_hash_file_streaming", real_hash_fn)

    resume_request = ScreeningRequest(
        items=items, criteria_set=make_criteria_set(), config=make_config(),
        output_dir=output_dir, resume_run_id="provisional-legit-resume-run",
    )
    result = _run_ft_screening(resume_request, make_components(output_dir, store=store))
    assert result.run_status == "completed"


# ---------------------------------------------------------------------------
# Finding 2: interruption/failure during the (provisional) manifest write
# itself must not strand the run
# ---------------------------------------------------------------------------


def test_finding2_interrupt_during_manifest_write_leaves_nothing_claimed(tmp_path, monkeypatch):
    pdf1 = make_pdf(tmp_path, "a.pdf")
    output_dir = tmp_path / "outputs"
    request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1)],
        criteria_set=make_criteria_set(), config=make_config(),
        output_dir=output_dir, run_id="manifest-write-interrupt-run",
    )
    store = FileScreeningStore(output_dir=output_dir)

    call_count = {"n": 0}
    real_write = store.write_run_manifest

    def interrupting_write(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise KeyboardInterrupt()
        return real_write(*args, **kwargs)

    monkeypatch.setattr(store, "write_run_manifest", interrupting_write)
    with pytest.raises(KeyboardInterrupt):
        _run_ft_screening(request, make_components(output_dir, store=store))

    run_dir = output_dir / "runs" / "manifest-write-interrupt-run"
    assert not run_dir.exists()  # nothing durable was ever claimed

    monkeypatch.setattr(store, "write_run_manifest", real_write)

    # A resume attempt correctly finds nothing to resume.
    resume_request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1)],
        criteria_set=make_criteria_set(), config=make_config(),
        output_dir=output_dir, resume_run_id="manifest-write-interrupt-run",
    )
    with pytest.raises(ResumeConflictError):
        _run_ft_screening(resume_request, make_components(output_dir, store=store))

    # A FRESH retry with the same run_id proceeds cleanly -- this is the
    # actual bug from the review: previously this raised RunSetupError
    # because begin_run() had already created a directory before the
    # manifest write was attempted.
    result = _run_ft_screening(request, make_components(output_dir, store=store))
    assert result.run_status == "completed"


def test_finding2_subprocess_crash_during_manifest_write_leaves_nothing_claimed(tmp_path):
    """Real cross-process crash (os._exit), not monkeypatching -- proves
    the fix holds under an actual process death, not just a Python
    exception caught mid-call."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True)
    helper = Path(__file__).parent / "_crash_helper.py"

    proc = subprocess.run(
        [sys.executable, str(helper), str(output_dir), "manifest-crash-run", "pre_manifest"],
        capture_output=True, text=True, timeout=30,
    )
    assert "crashing" in proc.stdout

    run_dir = output_dir / "runs" / "manifest-crash-run"
    assert not run_dir.exists()  # nothing durable was claimed before the crash

    store = FileScreeningStore(output_dir=output_dir)
    with pytest.raises(ResumeConflictError):
        store.begin_run(run_id="manifest-crash-run", resume=True)

    # A fresh claim of the same run_id must succeed -- proves both the OS
    # lock was released AND nothing was left stranded.
    store.begin_run(run_id="manifest-crash-run", resume=False)
    store.end_run(run_id="manifest-crash-run")


# ---------------------------------------------------------------------------
# Finding 3: cache-outage warnings must use stage="persist" and never leak
# raw backend exception text into the public result
# ---------------------------------------------------------------------------


class _SecretLeakingCacheStore(FileScreeningStore):
    def __init__(self, *, output_dir: Path, fail_get: bool = False, fail_put: bool = False) -> None:
        super().__init__(output_dir=output_dir)
        self._fail_get = fail_get
        self._fail_put = fail_put

    def get_cached_assessment(self, *, cache_key: str):
        if self._fail_get:
            raise ConnectionError("connection to postgresql://user:SUPERSECRET@cache.internal/db failed")
        return super().get_cached_assessment(cache_key=cache_key)

    def put_cached_assessment(self, *, cache_key: str, assessment, source_run_id: str) -> None:
        if self._fail_put:
            raise ConnectionError("connection to postgresql://user:SUPERSECRET@cache.internal/db failed")
        return super().put_cached_assessment(cache_key=cache_key, assessment=assessment, source_run_id=source_run_id)


def test_finding3_cache_read_failure_uses_persist_stage_and_no_secret_leak(tmp_path, caplog):
    pdf1 = make_pdf(tmp_path, "a.pdf")
    output_dir = tmp_path / "outputs"
    store = _SecretLeakingCacheStore(output_dir=output_dir, fail_get=True)
    request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1)],
        criteria_set=make_criteria_set(), config=make_config(), output_dir=output_dir,
    )
    with caplog.at_level(logging.WARNING):
        result = _run_ft_screening(request, make_components(output_dir, store=store))
    item = result.items[0]
    assert item.execution_status == "completed"
    cache_errors = [e for e in item.errors if e.code == "persistence_failed"]
    assert cache_errors
    for err in cache_errors:
        assert err.stage == "persist"
        assert "SUPERSECRET" not in err.message
        assert "postgresql://" not in err.message
    # The public artifact must round-trip clean too, not just the in-memory object.
    _, item_results_path, _ = store.artifact_paths(run_id=result.run_id)
    assert "SUPERSECRET" not in item_results_path.read_text(encoding="utf-8")
    run_result_path, _, _ = store.artifact_paths(run_id=result.run_id)
    assert "SUPERSECRET" not in run_result_path.read_text(encoding="utf-8")
    # The real exception text IS captured, just not in the public artifact.
    assert "SUPERSECRET" in caplog.text


def test_finding3_cache_write_failure_uses_persist_stage_and_no_secret_leak(tmp_path, caplog):
    pdf1 = make_pdf(tmp_path, "a.pdf")
    output_dir = tmp_path / "outputs"
    store = _SecretLeakingCacheStore(output_dir=output_dir, fail_put=True)
    request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1)],
        criteria_set=make_criteria_set(), config=make_config(), output_dir=output_dir,
    )
    with caplog.at_level(logging.WARNING):
        result = _run_ft_screening(request, make_components(output_dir, store=store))
    item = result.items[0]
    assert item.execution_status == "completed"
    cache_errors = [e for e in item.errors if e.code == "persistence_failed"]
    assert cache_errors
    for err in cache_errors:
        assert err.stage == "persist"
        assert "SUPERSECRET" not in err.message
    assert "SUPERSECRET" in caplog.text
