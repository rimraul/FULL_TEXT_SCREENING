"""Regression tests for codex_code_review_c4f274a.md -- the fifth
hardening round (Codex's fifth review overall of the orchestration
implementation). Named one group per finding, mirroring the convention
from tests/contract/test_hardening*.py.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ft_screening.components import Chunk, Components, ParsedDocument
from ft_screening.errors import ResumeConflictError
from ft_screening.fakes import FakeCriterionAdjudicator, FakeDocumentParser, FakeEvidenceRetriever
import ft_screening.orchestrator as orchestrator_module
from ft_screening.orchestrator import _run_ft_screening
from ft_screening.schema import (
    CriteriaSet,
    CriterionSpec,
    DocumentCoverage,
    ScreeningConfig,
    ScreeningItem,
    ScreeningRequest,
    ScreeningRunResult,
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
# Finding 1: pre-manifest interruption/crash must be genuinely resumable,
# not merely lock-reacquirable.
# ---------------------------------------------------------------------------


def test_finding1_interrupt_mid_hashing_leaves_a_resumable_provisional_manifest(tmp_path, monkeypatch):
    """The review's exact reproduction: interrupt during Stage A1 file
    hashing (after the run directory/lock exist but before every input has
    been hashed). Must leave a manifest.json (provisional -- request_fingerprint
    unset) so a full _run_ft_screening(resume_run_id=...) call succeeds,
    not just begin_run(resume=True)."""
    pdf1 = make_pdf(tmp_path, "a.pdf", b"content-a")
    pdf2 = make_pdf(tmp_path, "b.pdf", b"content-b")
    output_dir = tmp_path / "outputs"
    items = [
        ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1),
        ScreeningItem(document_id="d2", report_id="r2", pdf_path=pdf2),
    ]
    request = ScreeningRequest(
        items=items, criteria_set=make_criteria_set(), config=make_config(),
        output_dir=output_dir, run_id="hash-interrupt-run",
    )

    real_hash_fn = orchestrator_module._hash_file_streaming

    def interrupting_hash(path, **kwargs):
        if "b.pdf" in str(path):
            raise KeyboardInterrupt()
        return real_hash_fn(path, **kwargs)

    monkeypatch.setattr(orchestrator_module, "_hash_file_streaming", interrupting_hash)

    store = FileScreeningStore(output_dir=output_dir)
    with pytest.raises(KeyboardInterrupt):
        _run_ft_screening(request, make_components(output_dir, store=store))

    manifest = store.read_run_manifest(run_id="hash-interrupt-run")
    assert manifest is not None  # provisional manifest survived the interrupt
    assert manifest.get("request_fingerprint") is None

    monkeypatch.setattr(orchestrator_module, "_hash_file_streaming", real_hash_fn)

    resume_request = ScreeningRequest(
        items=items, criteria_set=make_criteria_set(), config=make_config(),
        output_dir=output_dir, resume_run_id="hash-interrupt-run",
    )
    result = _run_ft_screening(resume_request, make_components(output_dir, store=store))
    assert result.run_status == "completed"
    assert [i.execution_status for i in result.items] == ["completed", "completed"]


def test_finding1_interrupt_before_lock_acquired_leaves_no_manifest(tmp_path, monkeypatch):
    """Interrupting BEFORE begin_run() (still pre-manifest, but also
    pre-lock) must not fabricate a manifest -- there is nothing to make
    provisional yet."""
    pdf1 = make_pdf(tmp_path, "a.pdf")
    output_dir = tmp_path / "outputs"
    request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1)],
        criteria_set=make_criteria_set(), config=make_config(),
        output_dir=output_dir, run_id="pre-lock-interrupt-run",
    )

    def interrupting_compute(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(orchestrator_module, "compute_criteria_set_hash", interrupting_compute)
    with pytest.raises(KeyboardInterrupt):
        _run_ft_screening(request, make_components(output_dir))

    run_dir = output_dir / "runs" / "pre-lock-interrupt-run"
    assert not run_dir.exists()


def test_finding1_subprocess_crash_mid_hashing_is_semantically_resumable(tmp_path):
    """The review's central point: a subprocess test that only proves
    begin_run(resume=True) succeeds after a crash is NOT sufficient --
    it must prove a full _run_ft_screening(resume_run_id=...) call
    actually completes the run. _crash_helper.py's "mid_hashing" mode
    crashes (os._exit) a real _run_ft_screening() call partway through
    Stage A1 hashing of a genuine two-item request; this test then
    performs the real recovery in-process."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True)
    helper = Path(__file__).parent / "_crash_helper.py"

    proc = subprocess.run(
        [sys.executable, str(helper), str(output_dir), "mid-hash-crash-run", "mid_hashing"],
        capture_output=True, text=True, timeout=30,
    )
    assert "crashing" in proc.stdout  # child confirmed it reached the injected crash point

    store = FileScreeningStore(output_dir=output_dir)
    manifest = store.read_run_manifest(run_id="mid-hash-crash-run")
    assert manifest is not None  # the crash happened AFTER the provisional manifest write

    pdf1 = output_dir / "pdfs" / "a.pdf"
    pdf2 = output_dir / "pdfs" / "b.pdf"
    # Must match _crash_helper.py's "mid_hashing" mode's own request
    # construction EXACTLY -- execution_fingerprint is derived from
    # criteria_set/config content, and a resume attempt with a different
    # fingerprint is correctly rejected as a ResumeConflictError (a
    # different check than the one this test is exercising).
    criteria_set = make_criteria_set(
        criteria_set_id="cs",
        criteria=[
            CriterionSpec(
                criterion_id="population", version="1", eligibility_requirement="adults",
                missing_evidence_policy="review", exclusion_reason_code="wrong_population",
                exclusion_reason_label="Wrong population", priority=1,
            )
        ],
    )
    config = make_config()
    items = [
        ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1),
        ScreeningItem(document_id="d2", report_id="r2", pdf_path=pdf2),
    ]
    resume_request = ScreeningRequest(
        items=items, criteria_set=criteria_set, config=config,
        output_dir=output_dir, resume_run_id="mid-hash-crash-run",
    )
    result = _run_ft_screening(resume_request, make_components(output_dir, store=store))
    assert result.run_status == "completed"
    assert [i.execution_status for i in result.items] == ["completed", "completed"]


def test_finding1_resume_preflight_interrupt_before_baseline_does_not_write_snapshot(tmp_path, monkeypatch):
    """A resume attempt that is itself interrupted before terminal_results
    has been hydrated from the store (baseline_established still False)
    must not overwrite the prior durable snapshot with a bogus empty one."""
    pdf1 = make_pdf(tmp_path, "a.pdf")
    output_dir = tmp_path / "outputs"
    request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1)],
        criteria_set=make_criteria_set(), config=make_config(),
        output_dir=output_dir, run_id="resume-preflight-run",
    )
    store = FileScreeningStore(output_dir=output_dir)
    first = _run_ft_screening(request, make_components(output_dir, store=store))
    assert first.run_status == "completed"

    run_result_path, _, _ = store.artifact_paths(run_id="resume-preflight-run")
    before = run_result_path.read_text()

    def interrupting_compute(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(orchestrator_module, "compute_request_fingerprint", interrupting_compute)
    resume_request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1)],
        criteria_set=make_criteria_set(), config=make_config(),
        output_dir=output_dir, resume_run_id="resume-preflight-run",
    )
    with pytest.raises(KeyboardInterrupt):
        _run_ft_screening(resume_request, make_components(output_dir, store=store))

    assert run_result_path.read_text() == before  # untouched -- no bogus snapshot was written


# ---------------------------------------------------------------------------
# Finding 2: interrupted snapshots must not falsely blame fail_fast
# ---------------------------------------------------------------------------


def test_finding2_interrupted_not_processed_item_does_not_mention_fail_fast(tmp_path):
    items = [
        ScreeningItem(document_id=f"d{i}", report_id=f"r{i}", pdf_path=make_pdf(tmp_path, f"ki{i}.pdf", str(i).encode()))
        for i in range(3)
    ]
    output_dir = tmp_path / "outputs"
    request = ScreeningRequest(
        items=items, criteria_set=make_criteria_set(), config=make_config(max_workers=1),
        output_dir=output_dir, run_id="interrupt-cancel-run",
    )

    class InterruptingAdjudicator(FakeCriterionAdjudicator):
        def adjudicate(self, **kwargs):
            if kwargs["document"].document_id == "d1":
                raise KeyboardInterrupt()
            return super().adjudicate(**kwargs)

    components = Components(
        parser=FakeDocumentParser(), retriever=FakeEvidenceRetriever(), adjudicator=InterruptingAdjudicator(),
        store=FileScreeningStore(output_dir=output_dir),
    )
    # item_error_policy is left at its default ("continue") deliberately --
    # cancellation, not fail_fast, is what stopped work here.
    assert request.config.item_error_policy == "continue"
    with pytest.raises(KeyboardInterrupt):
        _run_ft_screening(request, components)

    store = FileScreeningStore(output_dir=output_dir)
    run_result_path, _, _ = store.artifact_paths(run_id="interrupt-cancel-run")
    snapshot = ScreeningRunResult.model_validate_json(run_result_path.read_text())

    not_processed = snapshot.items[2]
    assert not_processed.execution_status == "not_processed"
    message = not_processed.errors[0].message
    assert "fail_fast" not in message
    assert "interrupted" in message or "cancellation" in message


def test_finding2_fail_fast_not_processed_still_blames_fail_fast(tmp_path):
    """The ORIGINAL helper (_not_processed_result) must still be used, and
    still say what it always said, for genuine fail_fast suppression --
    only the interrupt/cancellation path changed."""
    items = [
        ScreeningItem(document_id=f"d{i}", report_id=f"r{i}", pdf_path=make_pdf(tmp_path, f"ff{i}.pdf", str(i).encode()))
        for i in range(3)
    ]
    output_dir = tmp_path / "outputs"
    request = ScreeningRequest(
        items=items, criteria_set=make_criteria_set(), config=make_config(max_workers=1, item_error_policy="fail_fast"),
        output_dir=output_dir, run_id="fail-fast-run",
    )
    parser = FakeDocumentParser(fail_document_ids={"d0"})
    result = _run_ft_screening(request, make_components(output_dir, parser=parser))
    not_processed = [i for i in result.items if i.execution_status == "not_processed"]
    assert not_processed
    assert "fail_fast" in not_processed[0].errors[0].message


# ---------------------------------------------------------------------------
# Finding 3: parsed chunks must be validated against document coverage
# ---------------------------------------------------------------------------


def make_coverage(**overrides) -> DocumentCoverage:
    defaults = dict(
        status="complete", total_pages=1, parsed_pages=[1], failed_pages=[],
        ocr_used=False, extracted_characters=64, warnings=[],
    )
    defaults.update(overrides)
    return DocumentCoverage(**defaults)


def test_finding3_chunk_page_not_in_parsed_pages_is_rejected():
    coverage = make_coverage(
        status="partial", total_pages=2, parsed_pages=[1], failed_pages=[2], extracted_characters=10,
    )
    bad_chunk = Chunk(chunk_id="c1", section=None, page=999, start_char=0, end_char=5, text="hello")
    with pytest.raises(Exception):
        ParsedDocument(document_id="d1", document_sha256="x" * 64, coverage=coverage, chunks=(bad_chunk,))


def test_finding3_chunk_page_in_failed_pages_is_rejected():
    coverage = make_coverage(
        status="partial", total_pages=2, parsed_pages=[1], failed_pages=[2], extracted_characters=10,
    )
    bad_chunk = Chunk(chunk_id="c1", section=None, page=2, start_char=0, end_char=5, text="hello")
    with pytest.raises(Exception):
        ParsedDocument(document_id="d1", document_sha256="x" * 64, coverage=coverage, chunks=(bad_chunk,))


def test_finding3_chunk_end_char_beyond_extracted_characters_is_rejected():
    coverage = make_coverage(extracted_characters=3)
    bad_chunk = Chunk(chunk_id="c1", section=None, page=1, start_char=0, end_char=50, text="hello world")
    with pytest.raises(Exception):
        ParsedDocument(document_id="d1", document_sha256="x" * 64, coverage=coverage, chunks=(bad_chunk,))


def test_finding3_positive_coverage_with_only_empty_chunk_text_is_rejected():
    coverage = make_coverage(extracted_characters=10)
    empty_chunk = Chunk(chunk_id="c1", section=None, page=1, start_char=0, end_char=0, text="   ")
    with pytest.raises(Exception):
        ParsedDocument(document_id="d1", document_sha256="x" * 64, coverage=coverage, chunks=(empty_chunk,))


def test_finding3_consistent_chunks_are_accepted():
    coverage = make_coverage(extracted_characters=10)
    chunk = Chunk(chunk_id="c1", section=None, page=1, start_char=0, end_char=10, text="0123456789")
    doc = ParsedDocument(document_id="d1", document_sha256="x" * 64, coverage=coverage, chunks=(chunk,))
    assert doc.chunks[0].chunk_id == "c1"


def test_finding3_orchestrator_boundary_rejects_a_misbehaving_parser(tmp_path):
    """The review's own reproduction: a parser claims one complete page
    (total_pages=1, parsed_pages=[1]) but attaches a chunk citing
    page=999. Must not reach a completed/include outcome -- must become
    this item's processing_failed(parse_failed) result, not silently
    validated public evidence citing a page that does not exist."""
    pdf1 = make_pdf(tmp_path, "a.pdf")
    output_dir = tmp_path / "outputs"

    class MisbehavingParser(FakeDocumentParser):
        def parse(self, **kwargs):
            coverage = make_coverage(total_pages=1, parsed_pages=[1], extracted_characters=64)
            bad_chunk = Chunk(
                chunk_id="bad-chunk", section=None, page=999, start_char=0, end_char=10, text="0123456789",
            )
            return ParsedDocument(
                document_id=kwargs["document_id"], document_sha256=kwargs["document_sha256"],
                coverage=coverage, chunks=(bad_chunk,),
            )

    request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1)],
        criteria_set=make_criteria_set(), config=make_config(), output_dir=output_dir,
    )
    result = _run_ft_screening(request, make_components(output_dir, parser=MisbehavingParser()))
    item = result.items[0]
    assert item.execution_status == "processing_failed"
    assert item.errors[0].code == "parse_failed"


# ---------------------------------------------------------------------------
# Finding 4: cache-store I/O failure must not be reported as a scientific
# evidence-retrieval failure
# ---------------------------------------------------------------------------


class _RaisingCacheStore(FileScreeningStore):
    def __init__(self, *, output_dir: Path, fail_get: bool = False, fail_put: bool = False) -> None:
        super().__init__(output_dir=output_dir)
        self._fail_get = fail_get
        self._fail_put = fail_put

    def get_cached_assessment(self, *, cache_key: str):
        if self._fail_get:
            raise ConnectionError("simulated cache backend outage")
        return super().get_cached_assessment(cache_key=cache_key)

    def put_cached_assessment(self, *, cache_key: str, assessment, source_run_id: str) -> None:
        if self._fail_put:
            raise ConnectionError("simulated cache backend outage")
        return super().put_cached_assessment(cache_key=cache_key, assessment=assessment, source_run_id=source_run_id)


def test_finding4_cache_read_failure_degrades_to_a_miss_not_retrieval_failed(tmp_path):
    pdf1 = make_pdf(tmp_path, "a.pdf")
    output_dir = tmp_path / "outputs"
    store = _RaisingCacheStore(output_dir=output_dir, fail_get=True)
    request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1)],
        criteria_set=make_criteria_set(), config=make_config(), output_dir=output_dir,
    )
    result = _run_ft_screening(request, make_components(output_dir, store=store))
    item = result.items[0]
    assert item.execution_status == "completed"
    assert item.provenance.cache_status == "miss"
    codes = [e.code for e in item.errors]
    assert "retrieval_failed" not in codes
    assert "persistence_failed" in codes
    assert all(e.severity == "warning" for e in item.errors if e.code == "persistence_failed")

    _, _, ledger_path = store.artifact_paths(run_id=result.run_id)
    ledger_events = [json.loads(line)["type"] for line in ledger_path.read_text().strip().splitlines()]
    assert "cache_read_failed" in ledger_events


def test_finding4_cache_write_failure_does_not_discard_a_valid_assessment(tmp_path):
    pdf1 = make_pdf(tmp_path, "a.pdf")
    output_dir = tmp_path / "outputs"
    store = _RaisingCacheStore(output_dir=output_dir, fail_put=True)
    request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1)],
        criteria_set=make_criteria_set(), config=make_config(), output_dir=output_dir,
    )
    result = _run_ft_screening(request, make_components(output_dir, store=store))
    item = result.items[0]
    assert item.execution_status == "completed"  # the computed assessment is still valid and used
    assert item.eligibility is not None
    codes = [e.code for e in item.errors]
    assert "persistence_failed" in codes

    _, _, ledger_path = store.artifact_paths(run_id=result.run_id)
    ledger_events = [json.loads(line)["type"] for line in ledger_path.read_text().strip().splitlines()]
    assert "cache_write_failed" in ledger_events


def test_finding4_cache_read_failure_does_not_stop_the_run(tmp_path):
    """Reinforces the "cache reads are optional" policy choice at run
    scope, not just item scope -- other items in the same run are
    unaffected by one criterion's cache read failure."""
    pdf1 = make_pdf(tmp_path, "a.pdf")
    pdf2 = make_pdf(tmp_path, "b.pdf", b"other-content")
    output_dir = tmp_path / "outputs"
    store = _RaisingCacheStore(output_dir=output_dir, fail_get=True)
    request = ScreeningRequest(
        items=[
            ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1),
            ScreeningItem(document_id="d2", report_id="r2", pdf_path=pdf2),
        ],
        criteria_set=make_criteria_set(), config=make_config(), output_dir=output_dir,
    )
    result = _run_ft_screening(request, make_components(output_dir, store=store))
    assert result.run_status == "completed"
    assert all(i.execution_status == "completed" for i in result.items)
