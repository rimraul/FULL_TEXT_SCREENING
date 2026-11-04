"""Regression tests for codex_code_review_408846d.md -- the second
hardening round. Named one group per finding, mirroring
tests/contract/test_hardening.py's convention for the first round.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ft_screening.components import Components
from ft_screening.errors import ResumeConflictError
from ft_screening.fakes import FakeCriterionAdjudicator, FakeDocumentParser, FakeEvidenceRetriever
from ft_screening.orchestrator import _run_ft_screening
import ft_screening.orchestrator as orchestrator_module
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


def make_components(output_dir: Path, *, parser=None, retriever=None, adjudicator=None, **adjudicator_kwargs) -> Components:
    return Components(
        parser=parser or FakeDocumentParser(),
        retriever=retriever or FakeEvidenceRetriever(),
        adjudicator=adjudicator or FakeCriterionAdjudicator(**adjudicator_kwargs),
        store=FileScreeningStore(output_dir=output_dir),
    )


# ---------------------------------------------------------------------------
# Finding 1: malformed component return VALUES (not just wrong identity)
# never escape as uncaught exceptions
# ---------------------------------------------------------------------------


class _MalformedParser:
    def parse(self, **kwargs):
        return object()  # not a ParsedDocument -- no .document_id, .coverage, etc.


class _NoneReturningRetriever:
    def retrieve(self, **kwargs):
        return None  # not iterable as a list of EvidenceSpan


class _MalformedAdjudicator:
    def adjudicate(self, **kwargs):
        return object()  # not a CriterionAssessment -- no .criterion_id, .model, etc.


def test_finding1_malformed_parser_return_value_becomes_item_outcome(tmp_path):
    pdf1 = make_pdf(tmp_path, "a.pdf")
    output_dir = tmp_path / "outputs"
    request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1)],
        criteria_set=make_criteria_set(), config=make_config(), output_dir=output_dir,
    )
    # Must not raise -- the malformed return value is caught, not propagated.
    result = _run_ft_screening(request, make_components(output_dir, parser=_MalformedParser()))
    assert result.items[0].execution_status == "processing_failed"
    assert result.items[0].errors[0].code == "parse_failed"


def test_finding1_none_returning_retriever_becomes_item_outcome(tmp_path):
    pdf1 = make_pdf(tmp_path, "a.pdf")
    output_dir = tmp_path / "outputs"
    request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1)],
        criteria_set=make_criteria_set(), config=make_config(), output_dir=output_dir,
    )
    result = _run_ft_screening(request, make_components(output_dir, retriever=_NoneReturningRetriever()))
    assert result.items[0].execution_status == "processing_failed"
    assert result.items[0].errors[0].code == "retrieval_failed"


def test_finding1_malformed_adjudicator_return_value_becomes_item_outcome(tmp_path):
    pdf1 = make_pdf(tmp_path, "a.pdf")
    output_dir = tmp_path / "outputs"
    request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1)],
        criteria_set=make_criteria_set(), config=make_config(), output_dir=output_dir,
    )
    result = _run_ft_screening(request, make_components(output_dir, adjudicator=_MalformedAdjudicator()))
    assert result.items[0].execution_status == "processing_failed"
    assert result.items[0].errors[0].code == "adjudication_failed"


# ---------------------------------------------------------------------------
# Finding 3: interruption guard covers preflight; hashing failures/existing
# directories are classified correctly, never escape
# ---------------------------------------------------------------------------


def test_finding3_interrupt_during_preflight_hashing_writes_durable_snapshot(tmp_path, monkeypatch):
    items = [
        ScreeningItem(document_id=f"d{i}", report_id=f"r{i}", pdf_path=make_pdf(tmp_path, f"h{i}.pdf", str(i).encode()))
        for i in range(3)
    ]
    output_dir = tmp_path / "outputs"
    request = ScreeningRequest(
        items=items, criteria_set=make_criteria_set(), config=make_config(),
        output_dir=output_dir, run_id="preflight-interrupt-run",
    )

    real_hash_fn = orchestrator_module._hash_file_streaming

    def interrupting_hash(path, **kwargs):
        if "h1" in str(path):
            raise KeyboardInterrupt()
        return real_hash_fn(path, **kwargs)

    monkeypatch.setattr(orchestrator_module, "_hash_file_streaming", interrupting_hash)

    with pytest.raises(KeyboardInterrupt):
        _run_ft_screening(request, make_components(output_dir))

    store = FileScreeningStore(output_dir=output_dir)
    run_result_path, _, ledger_path = store.artifact_paths(run_id="preflight-interrupt-run")
    assert run_result_path.exists()  # a run directory + snapshot exist even though the interrupt hit during hashing
    snapshot = ScreeningRunResult.model_validate_json(run_result_path.read_text())
    assert snapshot.run_status == "interrupted"
    ledger_events = [json.loads(line)["type"] for line in ledger_path.read_text().strip().splitlines()]
    assert "run_interrupted" in ledger_events


def test_finding3_unreadable_file_during_hashing_becomes_item_outcome(tmp_path, monkeypatch):
    pdf1 = make_pdf(tmp_path, "a.pdf")
    pdf2 = make_pdf(tmp_path, "b.pdf", b"different")
    output_dir = tmp_path / "outputs"
    request = ScreeningRequest(
        items=[
            ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1),
            ScreeningItem(document_id="d2", report_id="r2", pdf_path=pdf2),
        ],
        criteria_set=make_criteria_set(), config=make_config(), output_dir=output_dir,
    )

    real_hash_fn = orchestrator_module._hash_file_streaming

    def flaky_hash(path, **kwargs):
        if "a.pdf" in str(path):
            raise OSError("permission denied (simulated)")
        return real_hash_fn(path, **kwargs)

    monkeypatch.setattr(orchestrator_module, "_hash_file_streaming", flaky_hash)

    result = _run_ft_screening(request, make_components(output_dir))  # must not raise
    d1 = next(i for i in result.items if i.document_id == "d1")
    d2 = next(i for i in result.items if i.document_id == "d2")
    assert d1.execution_status == "input_unavailable"
    assert d1.errors[0].code == "unexpected_error"
    assert d2.execution_status == "completed"  # unaffected by d1's failure


def test_finding3_existing_directory_classified_as_input_not_file(tmp_path):
    directory_as_pdf_path = tmp_path / "pdfs" / "this_is_a_directory.pdf"
    directory_as_pdf_path.mkdir(parents=True)
    output_dir = tmp_path / "outputs"
    request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=directory_as_pdf_path)],
        criteria_set=make_criteria_set(), config=make_config(), output_dir=output_dir,
    )
    result = _run_ft_screening(request, make_components(output_dir))
    item = result.items[0]
    assert item.execution_status == "input_unavailable"
    assert item.errors[0].code == "input_not_file"  # not "input_missing"


# ---------------------------------------------------------------------------
# Finding 4: chunk-ID membership is unconditional; parser consistency check
# ---------------------------------------------------------------------------


def test_finding4_empty_chunk_set_still_rejects_invented_evidence(tmp_path):
    pdf1 = make_pdf(tmp_path, "a.pdf")
    output_dir = tmp_path / "outputs"
    empty_chunk_coverage = DocumentCoverage(
        status="partial", total_pages=1, parsed_pages=[], failed_pages=[1], ocr_used=False,
        extracted_characters=0, warnings=[],
    )
    parser = FakeDocumentParser(coverage_by_document_id={"d1": empty_chunk_coverage}, chunks_by_document_id={"d1": []})
    from ft_screening.schema import EvidenceSpan

    invented_span = EvidenceSpan(
        document_id="d1", document_sha256="a" * 64, chunk_id="invented-chunk-that-does-not-exist",
        excerpt="x", stance="relevant", retrieval_method="fake", retrieval_query_version="1",
    )
    import hashlib

    real_hash = hashlib.sha256(pdf1.read_bytes()).hexdigest()
    invented_span = invented_span.model_copy(update={"document_sha256": real_hash})
    retriever = FakeEvidenceRetriever(evidence_by_key={("d1", "population"): [invented_span]})
    request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1)],
        criteria_set=make_criteria_set(), config=make_config(), output_dir=output_dir,
    )
    result = _run_ft_screening(request, make_components(output_dir, parser=parser, retriever=retriever))
    assert result.items[0].execution_status == "processing_failed"
    assert result.items[0].errors[0].code == "retrieval_failed"


def test_finding4_extracted_characters_with_no_chunks_rejected(tmp_path):
    pdf1 = make_pdf(tmp_path, "a.pdf")
    output_dir = tmp_path / "outputs"
    inconsistent_coverage = DocumentCoverage(
        status="complete", total_pages=1, parsed_pages=[1], failed_pages=[], ocr_used=False,
        extracted_characters=500, warnings=[],  # claims content...
    )
    parser = FakeDocumentParser(
        coverage_by_document_id={"d1": inconsistent_coverage}, chunks_by_document_id={"d1": []}  # ...but no chunks
    )
    request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1)],
        criteria_set=make_criteria_set(), config=make_config(), output_dir=output_dir,
    )
    result = _run_ft_screening(request, make_components(output_dir, parser=parser))
    assert result.items[0].execution_status == "processing_failed"
    assert result.items[0].errors[0].code == "parse_failed"


# ---------------------------------------------------------------------------
# Finding 5: callback disablement is logged; concurrent emit doesn't crash
# ---------------------------------------------------------------------------


def test_finding5_callback_disablement_logged_to_ledger(tmp_path, caplog):
    """codex_code_review_53f21de.md's security note: truncation is not
    redaction. The PUBLIC ledger artifact must carry only the exception
    type and a fixed, content-free message; the real exception text
    (which could contain paths/secrets, here containing "boom" as a stand-
    in) belongs only in the runtime logger."""
    pdf1 = make_pdf(tmp_path, "a.pdf")
    output_dir = tmp_path / "outputs"
    request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1)],
        criteria_set=make_criteria_set(), config=make_config(),
        output_dir=output_dir, run_id="cb-log-run",
    )

    def bad_callback(event):
        raise RuntimeError("boom")

    with caplog.at_level("WARNING", logger="ft_screening.orchestrator"):
        _run_ft_screening(request, make_components(output_dir), progress_cb=bad_callback)

    store = FileScreeningStore(output_dir=output_dir)
    _, _, ledger_path = store.artifact_paths(run_id="cb-log-run")
    events = [json.loads(line) for line in ledger_path.read_text().strip().splitlines()]
    disabled_events = [e for e in events if e["type"] == "progress_callback_disabled"]
    assert len(disabled_events) == 1
    assert disabled_events[0]["exception_type"] == "RuntimeError"
    assert "boom" not in disabled_events[0]["message"]  # not in the public artifact...
    assert disabled_events[0]["message"]  # ...but still a real, non-empty, neutral message

    # ...the real exception text goes to the runtime logger instead.
    assert any("RuntimeError" in record.message for record in caplog.records)
    assert any(record.exc_info is not None for record in caplog.records)


def test_finding5_concurrent_item_started_emission_does_not_crash_or_double_fire(tmp_path):
    items = [
        ScreeningItem(document_id=f"d{i}", report_id=f"r{i}", pdf_path=make_pdf(tmp_path, f"cb{i}.pdf", str(i).encode()))
        for i in range(6)
    ]
    output_dir = tmp_path / "outputs"
    request = ScreeningRequest(
        items=items, criteria_set=make_criteria_set(), config=make_config(max_workers=4), output_dir=output_dir,
    )
    events = []

    def failing_after_first(event):
        events.append(event)
        raise RuntimeError("fails every time")

    result = _run_ft_screening(request, make_components(output_dir), progress_cb=failing_after_first)
    assert result.run_status == "completed"  # a hostile callback must never affect the scientific result
    assert len(events) == 1  # disabled after exactly one delivery, even under concurrency


# ---------------------------------------------------------------------------
# Finding 6: mixed-source cache-hit provenance
# ---------------------------------------------------------------------------


def test_finding6_mixed_source_cache_hit_does_not_claim_a_false_single_source(tmp_path):
    pdf1 = make_pdf(tmp_path, "a.pdf", b"content-for-criterion-set")
    output_dir = tmp_path / "outputs"
    store = FileScreeningStore(output_dir=output_dir)

    criterion_a = CriterionSpec(
        criterion_id="population", version="1", eligibility_requirement="adults",
        missing_evidence_policy="review", exclusion_reason_code="wrong_population",
        exclusion_reason_label="Wrong population", priority=1,
    )
    criterion_b = CriterionSpec(
        criterion_id="design", version="1", eligibility_requirement="empirical",
        missing_evidence_policy="review", exclusion_reason_code="not_empirical",
        exclusion_reason_label="Not empirical", priority=2,
    )

    def components():
        return Components(
            parser=FakeDocumentParser(), retriever=FakeEvidenceRetriever(), adjudicator=FakeCriterionAdjudicator(), store=store
        )

    # Run A: caches "population" only (single-criterion set).
    req_a = ScreeningRequest(
        items=[ScreeningItem(document_id="doc-A", report_id="report-A", pdf_path=pdf1)],
        criteria_set=CriteriaSet(criteria_set_id="cs-a", version="1", criteria=[criterion_a]),
        config=make_config(), output_dir=output_dir, run_id="run-A",
    )
    _run_ft_screening(req_a, components())

    # Run B: caches "design" only, same content, different criteria set (so
    # it doesn't collide with run A's cache key for "population").
    req_b = ScreeningRequest(
        items=[ScreeningItem(document_id="doc-B", report_id="report-B", pdf_path=pdf1)],
        criteria_set=CriteriaSet(criteria_set_id="cs-b", version="1", criteria=[criterion_b]),
        config=make_config(), output_dir=output_dir, run_id="run-B",
    )
    _run_ft_screening(req_b, components())

    # Run C: uses BOTH criteria against the same content -- "population"
    # should hit run A's cache entry, "design" should hit run B's. Mixed sources.
    req_c = ScreeningRequest(
        items=[ScreeningItem(document_id="doc-C", report_id="report-C", pdf_path=pdf1)],
        criteria_set=CriteriaSet(criteria_set_id="cs-c", version="1", criteria=[criterion_a, criterion_b]),
        config=make_config(), output_dir=output_dir, run_id="run-C",
    )
    result_c = _run_ft_screening(req_c, components())
    item = result_c.items[0]
    assert item.provenance.cache_status == "hit"  # both criteria were cache hits
    assert item.provenance.source_result_run_id is None  # but sources differ -- must not claim a single false source

    # The per-criterion ledger events carry the real, non-lossy lineage.
    _, _, ledger_path = store.artifact_paths(run_id="run-C")
    cache_hit_events = [json.loads(line) for line in ledger_path.read_text().strip().splitlines() if json.loads(line)["type"] == "cache_hit"]
    sources_by_criterion = {e["criterion_id"]: e["source_run_id"] for e in cache_hit_events}
    assert sources_by_criterion == {"population": "run-A", "design": "run-B"}


# ---------------------------------------------------------------------------
# Finding 7: duplicate-content reuse preserves warnings and logs its source
# ---------------------------------------------------------------------------


def test_finding7_reuse_preserves_canonical_warnings(tmp_path):
    pdf1 = make_pdf(tmp_path, "a.pdf", b"identical-bytes")
    pdf2 = make_pdf(tmp_path, "b.pdf", b"identical-bytes")
    output_dir = tmp_path / "outputs"
    partial_coverage = DocumentCoverage(
        status="partial", total_pages=2, parsed_pages=[1], failed_pages=[2], ocr_used=False,
        extracted_characters=10, warnings=[],
    )
    parser = FakeDocumentParser(coverage_by_document_id={"canonical-doc": partial_coverage})
    request = ScreeningRequest(
        items=[
            ScreeningItem(document_id="canonical-doc", report_id="canonical-report", pdf_path=pdf1),
            ScreeningItem(document_id="duplicate-doc", report_id="duplicate-report", pdf_path=pdf2),
        ],
        criteria_set=make_criteria_set(),
        config=make_config(duplicate_content_policy="reuse"),
        output_dir=output_dir,
        run_id="reuse-warnings-run",
    )
    result = _run_ft_screening(request, make_components(output_dir, parser=parser))
    canonical_item, duplicate_item = result.items
    assert canonical_item.errors  # the canonical item carries the partial-coverage warning
    assert duplicate_item.errors  # and now so does the reused duplicate (previously dropped)
    assert duplicate_item.errors[0].code == "insufficient_coverage"


def test_finding7_ledger_records_content_reuse_source(tmp_path):
    pdf1 = make_pdf(tmp_path, "a.pdf", b"identical-bytes")
    pdf2 = make_pdf(tmp_path, "b.pdf", b"identical-bytes")
    output_dir = tmp_path / "outputs"
    request = ScreeningRequest(
        items=[
            ScreeningItem(document_id="canonical-doc", report_id="canonical-report", pdf_path=pdf1),
            ScreeningItem(document_id="duplicate-doc", report_id="duplicate-report", pdf_path=pdf2),
        ],
        criteria_set=make_criteria_set(),
        config=make_config(duplicate_content_policy="reuse"),
        output_dir=output_dir,
        run_id="reuse-ledger-run",
    )
    _run_ft_screening(request, make_components(output_dir))

    store = FileScreeningStore(output_dir=output_dir)
    _, _, ledger_path = store.artifact_paths(run_id="reuse-ledger-run")
    events = [json.loads(line) for line in ledger_path.read_text().strip().splitlines()]
    reuse_events = [e for e in events if e["type"] == "content_reused"]
    assert len(reuse_events) == 1
    assert reuse_events[0]["canonical_input_index"] == 0
    assert reuse_events[0]["canonical_document_id"] == "canonical-doc"
    assert reuse_events[0]["reused_input_index"] == 1
    assert reuse_events[0]["reused_document_id"] == "duplicate-doc"
