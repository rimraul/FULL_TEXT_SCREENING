"""Adversarial/durability tests added per codex_code_review_44d9e90.md --
each test is named after the finding it closes. These exercise externally
observable guarantees the first hardening round's test suite missed.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from ft_screening.components import Chunk, Components, ParsedDocument
from ft_screening.fakes import FakeCriterionAdjudicator, FakeDocumentParser, FakeEvidenceRetriever
from ft_screening.orchestrator import _run_ft_screening
from ft_screening.schema import (
    CriteriaSet,
    CriterionSpec,
    DocumentCoverage,
    EvidenceSpan,
    ScreeningConfig,
    ScreeningItem,
    ScreeningItemResult,
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


def make_components(output_dir: Path, *, parser=None, retriever=None, **adjudicator_kwargs) -> Components:
    return Components(
        parser=parser or FakeDocumentParser(),
        retriever=retriever or FakeEvidenceRetriever(),
        adjudicator=FakeCriterionAdjudicator(**adjudicator_kwargs),
        store=FileScreeningStore(output_dir=output_dir),
    )


# ---------------------------------------------------------------------------
# Finding 1: every terminal result is durably persisted, including preflight
# and not_processed outcomes
# ---------------------------------------------------------------------------


def test_finding1_every_execution_status_persisted_to_item_results_jsonl(tmp_path):
    missing = tmp_path / "pdfs" / "missing.pdf"
    dup_a = make_pdf(tmp_path, "dup_a.pdf", b"same-bytes")
    dup_b = make_pdf(tmp_path, "dup_b.pdf", b"same-bytes")
    fails = make_pdf(tmp_path, "fails.pdf", b"fails-content")
    never_started = make_pdf(tmp_path, "never.pdf", b"never-content")

    items = [
        ScreeningItem(document_id="dup-a", report_id="r-dup-a", pdf_path=dup_a),
        ScreeningItem(document_id="dup-b", report_id="r-dup-b", pdf_path=dup_b),
        ScreeningItem(document_id="missing", report_id="r-missing", pdf_path=missing),
        ScreeningItem(document_id="fails", report_id="r-fails", pdf_path=fails),
        ScreeningItem(document_id="never", report_id="r-never", pdf_path=never_started),
    ]
    output_dir = tmp_path / "outputs"
    request = ScreeningRequest(
        items=items,
        criteria_set=make_criteria_set(),
        config=make_config(duplicate_content_policy="error", item_error_policy="fail_fast", max_workers=1),
        output_dir=output_dir,
        run_id="persist-run",
    )
    components = make_components(output_dir, fail_keys={("fails", "population")})
    result = _run_ft_screening(request, components)

    statuses = [i.execution_status for i in result.items]
    # index 2 ("missing") is a preflight-fatal input_unavailable outcome --
    # under fail_fast that stops index 3 ("fails") from ever being
    # submitted too (finding 8: preflight fatals count toward the rule).
    assert statuses == ["duplicate_input", "duplicate_input", "input_unavailable", "not_processed", "not_processed"]

    # Read item_results.jsonl directly -- must contain ALL five, not just
    # whichever happened to be "completed" (there are none here).
    _, item_results_path, _ = FileScreeningStore(output_dir=output_dir).artifact_paths(run_id="persist-run")
    persisted_indices = set()
    with open(item_results_path) as f:
        for line in f:
            persisted_indices.add(ScreeningItemResult.model_validate_json(line).input_index)
    assert persisted_indices == {0, 1, 2, 3, 4}


def test_finding1_item_results_jsonl_has_no_duplicate_lines_per_index(tmp_path):
    pdf1 = make_pdf(tmp_path, "a.pdf")
    output_dir = tmp_path / "outputs"
    request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1)],
        criteria_set=make_criteria_set(),
        config=make_config(),
        output_dir=output_dir,
        run_id="single-run",
    )
    _run_ft_screening(request, make_components(output_dir))
    _, item_results_path, _ = FileScreeningStore(output_dir=output_dir).artifact_paths(run_id="single-run")
    lines = item_results_path.read_text().strip().splitlines()
    assert len(lines) == 1  # materialized current view -- one line per input_index, not one per write


# ---------------------------------------------------------------------------
# Finding 3: document coverage gates eligibility
# ---------------------------------------------------------------------------


def test_finding3_partial_coverage_caps_decision_at_needs_human_review(tmp_path):
    pdf1 = make_pdf(tmp_path, "a.pdf")
    output_dir = tmp_path / "outputs"
    partial_coverage = DocumentCoverage(
        status="partial", total_pages=3, parsed_pages=[1], failed_pages=[2], ocr_used=False,
        extracted_characters=50, warnings=[],
    )
    parser = FakeDocumentParser(coverage_by_document_id={"d1": partial_coverage})
    request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1)],
        criteria_set=make_criteria_set(),
        config=make_config(),
        output_dir=output_dir,
    )
    # Adjudicator would otherwise say "satisfied" -> "include"; coverage must override this.
    result = _run_ft_screening(request, make_components(output_dir, parser=parser))
    item = result.items[0]
    assert item.execution_status == "completed"
    assert item.eligibility.final_decision == "needs_human_review"
    assert any(e.code == "insufficient_coverage" and e.severity == "warning" for e in item.errors)


def test_finding3_partial_coverage_caps_even_a_would_be_exclude(tmp_path):
    """codex_code_review_408846d.md finding 2: the first hardening round
    only capped 'include' -> needs_human_review under partial coverage,
    and this test originally locked in that an 'exclude' survived partial
    coverage unchanged. That was itself a contract violation -- the frozen
    text caps EVERY decision at needs_human_review under partial coverage,
    with no carve-out for a confident-looking exclude. Corrected here."""
    pdf1 = make_pdf(tmp_path, "a.pdf")
    output_dir = tmp_path / "outputs"
    partial_coverage = DocumentCoverage(
        status="partial", total_pages=3, parsed_pages=[1], failed_pages=[2], ocr_used=False,
        extracted_characters=50, warnings=[],
    )
    parser = FakeDocumentParser(coverage_by_document_id={"d1": partial_coverage})
    request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1)],
        criteria_set=make_criteria_set(),
        config=make_config(),
        output_dir=output_dir,
    )
    result = _run_ft_screening(
        request, make_components(output_dir, parser=parser, status_by_key={("d1", "population"): "violated"})
    )
    eligibility = result.items[0].eligibility
    assert eligibility.final_decision == "needs_human_review"
    assert eligibility.primary_exclusion_reason_code is None  # schema requires this whenever decision != exclude
    assert "wrong_population" in eligibility.exclusion_reason_codes  # but the reviewer context survives


def test_finding3_failed_coverage_never_reaches_retriever(tmp_path):
    pdf1 = make_pdf(tmp_path, "a.pdf")
    output_dir = tmp_path / "outputs"
    failed_coverage = DocumentCoverage(
        status="failed", total_pages=3, parsed_pages=[], failed_pages=[1, 2, 3], ocr_used=False,
        extracted_characters=0, warnings=["parser could not read this file"],
    )
    parser = FakeDocumentParser(coverage_by_document_id={"d1": failed_coverage})
    retriever_calls: list = []

    class TrackingRetriever(FakeEvidenceRetriever):
        def retrieve(self, **kwargs):
            retriever_calls.append(kwargs["criterion"].criterion_id)
            return super().retrieve(**kwargs)

    request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1)],
        criteria_set=make_criteria_set(),
        config=make_config(),
        output_dir=output_dir,
    )
    result = _run_ft_screening(request, make_components(output_dir, parser=parser, retriever=TrackingRetriever()))
    item = result.items[0]
    assert item.execution_status == "processing_failed"
    assert item.errors[0].code == "insufficient_coverage"
    assert retriever_calls == []  # never reached


def test_finding3_complete_coverage_unaffected(tmp_path):
    pdf1 = make_pdf(tmp_path, "a.pdf")
    output_dir = tmp_path / "outputs"
    request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1)],
        criteria_set=make_criteria_set(),
        config=make_config(),
        output_dir=output_dir,
    )
    result = _run_ft_screening(request, make_components(output_dir))
    item = result.items[0]
    assert item.eligibility.final_decision == "include"
    assert item.errors == []


# ---------------------------------------------------------------------------
# Finding 4: private-component outputs are validated at the boundary
# ---------------------------------------------------------------------------


def test_finding4_wrong_parser_identity_rejected(tmp_path):
    pdf1 = make_pdf(tmp_path, "a.pdf")
    output_dir = tmp_path / "outputs"
    parser = FakeDocumentParser(identity_override_by_document_id={"d1": ("WRONG-ID", "f" * 64)})
    request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1)],
        criteria_set=make_criteria_set(),
        config=make_config(),
        output_dir=output_dir,
    )
    result = _run_ft_screening(request, make_components(output_dir, parser=parser))
    item = result.items[0]
    assert item.execution_status == "processing_failed"
    assert item.errors[0].code == "parse_failed"
    assert item.eligibility is None  # the wrong identity never reaches a public result


def test_finding4_wrong_evidence_document_id_rejected(tmp_path):
    pdf1 = make_pdf(tmp_path, "a.pdf")
    output_dir = tmp_path / "outputs"
    wrong_span = EvidenceSpan(
        document_id="SOME-OTHER-DOCUMENT",
        document_sha256="f" * 64,
        chunk_id="whatever",
        excerpt="x",
        stance="relevant",
        retrieval_method="fake",
        retrieval_query_version="1",
    )
    retriever = FakeEvidenceRetriever(evidence_by_key={("d1", "population"): [wrong_span]})
    request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1)],
        criteria_set=make_criteria_set(),
        config=make_config(),
        output_dir=output_dir,
    )
    result = _run_ft_screening(request, make_components(output_dir, retriever=retriever))
    item = result.items[0]
    assert item.execution_status == "processing_failed"
    assert item.errors[0].code == "retrieval_failed"


def test_finding4_wrong_evidence_chunk_id_rejected(tmp_path):
    pdf1 = make_pdf(tmp_path, "a.pdf")
    output_dir = tmp_path / "outputs"
    bad_span = EvidenceSpan(
        document_id="d1",
        document_sha256="f" * 64,  # will be overridden by real hash below via document_sha256 mismatch too
        chunk_id="chunk-that-does-not-exist",
        excerpt="x",
        stance="relevant",
        retrieval_method="fake",
        retrieval_query_version="1",
    )
    # Fix the sha to match what will actually be computed, isolating the test to chunk_id only.
    import hashlib

    real_hash = hashlib.sha256(pdf1.read_bytes()).hexdigest()
    bad_span = bad_span.model_copy(update={"document_sha256": real_hash})
    retriever = FakeEvidenceRetriever(evidence_by_key={("d1", "population"): [bad_span]})
    request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1)],
        criteria_set=make_criteria_set(),
        config=make_config(),
        output_dir=output_dir,
    )
    result = _run_ft_screening(request, make_components(output_dir, retriever=retriever))
    assert result.items[0].execution_status == "processing_failed"
    assert result.items[0].errors[0].code == "retrieval_failed"


def test_finding4_wrong_criterion_id_from_adjudicator_rejected(tmp_path):
    pdf1 = make_pdf(tmp_path, "a.pdf")
    output_dir = tmp_path / "outputs"
    components = make_components(
        output_dir, field_overrides_by_key={("d1", "population"): {"criterion_id": "not-population"}}
    )
    request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1)],
        criteria_set=make_criteria_set(),
        config=make_config(),
        output_dir=output_dir,
    )
    result = _run_ft_screening(request, components)
    assert result.items[0].execution_status == "processing_failed"
    assert result.items[0].errors[0].code == "adjudication_failed"


def test_finding4_wrong_model_from_adjudicator_rejected(tmp_path):
    pdf1 = make_pdf(tmp_path, "a.pdf")
    output_dir = tmp_path / "outputs"
    components = make_components(output_dir, field_overrides_by_key={("d1", "population"): {"model": "wrong-model"}})
    request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1)],
        criteria_set=make_criteria_set(),
        config=make_config(),
        output_dir=output_dir,
    )
    result = _run_ft_screening(request, components)
    assert result.items[0].execution_status == "processing_failed"
    assert result.items[0].errors[0].code == "adjudication_failed"


# ---------------------------------------------------------------------------
# Finding 5: callback disablement, item_started, completed_items, interruption
# ---------------------------------------------------------------------------


def test_finding5_callback_disabled_after_first_exception(tmp_path):
    pdf1 = make_pdf(tmp_path, "a.pdf")
    pdf2 = make_pdf(tmp_path, "b.pdf", b"different")
    output_dir = tmp_path / "outputs"
    request = ScreeningRequest(
        items=[
            ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1),
            ScreeningItem(document_id="d2", report_id="r2", pdf_path=pdf2),
        ],
        criteria_set=make_criteria_set(),
        config=make_config(),
        output_dir=output_dir,
    )
    call_count = {"n": 0}

    def bad_callback(event):
        call_count["n"] += 1
        raise RuntimeError("callback always fails")

    _run_ft_screening(request, make_components(output_dir), progress_cb=bad_callback)
    assert call_count["n"] == 1  # disabled after the first failure, never invoked again


def test_finding5_item_started_event_emitted(tmp_path):
    pdf1 = make_pdf(tmp_path, "a.pdf")
    output_dir = tmp_path / "outputs"
    request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1)],
        criteria_set=make_criteria_set(),
        config=make_config(),
        output_dir=output_dir,
    )
    events = []
    _run_ft_screening(request, make_components(output_dir), progress_cb=events.append)
    event_names = [e.event for e in events]
    assert "item_started" in event_names
    assert "run_started" in event_names
    assert "run_completed" in event_names


def test_finding5_completed_items_tracks_real_progress(tmp_path):
    items = [
        ScreeningItem(document_id=f"d{i}", report_id=f"r{i}", pdf_path=make_pdf(tmp_path, f"{i}.pdf", str(i).encode()))
        for i in range(3)
    ]
    output_dir = tmp_path / "outputs"
    request = ScreeningRequest(items=items, criteria_set=make_criteria_set(), config=make_config(), output_dir=output_dir)
    events = []
    _run_ft_screening(request, make_components(output_dir), progress_cb=events.append)
    completed_values = [e.completed_items for e in events if e.event == "item_completed"]
    assert completed_values == [1, 2, 3]  # monotonically increasing, not stuck at 0


def test_finding5_keyboard_interrupt_writes_durable_snapshot(tmp_path):
    items = [
        ScreeningItem(document_id=f"d{i}", report_id=f"r{i}", pdf_path=make_pdf(tmp_path, f"ki{i}.pdf", str(i).encode()))
        for i in range(3)
    ]
    output_dir = tmp_path / "outputs"
    request = ScreeningRequest(
        items=items, criteria_set=make_criteria_set(), config=make_config(max_workers=1),
        output_dir=output_dir, run_id="interrupt-run",
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
    with pytest.raises(KeyboardInterrupt):
        _run_ft_screening(request, components)

    store = FileScreeningStore(output_dir=output_dir)
    run_result_path, _, ledger_path = store.artifact_paths(run_id="interrupt-run")
    assert run_result_path.exists()
    from ft_screening.schema import ScreeningRunResult

    snapshot = ScreeningRunResult.model_validate_json(run_result_path.read_text())
    assert snapshot.run_status == "interrupted"
    assert snapshot.items[0].execution_status == "completed"  # d0 finished before the interrupt
    assert snapshot.items[2].execution_status == "not_processed"  # d2 never started

    ledger_events = [json.loads(line)["type"] for line in ledger_path.read_text().strip().splitlines()]
    assert "run_interrupted" in ledger_events

    # The lock must still be released despite the interrupt. Since
    # codex_code_review_0f48a3d.md finding 2, locking is a real OS-level
    # advisory lock, not file existence -- the .lock file itself stays on
    # disk (it's the lock's target, not the signal), so "released" is now
    # verified by successfully re-acquiring it, not by the file vanishing.
    recovery_store = FileScreeningStore(output_dir=output_dir)
    recovery_store.begin_run(run_id="interrupt-run", resume=True)  # must not raise
    recovery_store.end_run(run_id="interrupt-run")


# ---------------------------------------------------------------------------
# Finding 6: cross-run cache-hit provenance and poisoning prevention
# ---------------------------------------------------------------------------


def test_finding6_cross_run_cache_hit_rebinds_ids_and_records_source(tmp_path):
    pdf1 = make_pdf(tmp_path, "a.pdf", b"identical-bytes")
    pdf2 = make_pdf(tmp_path, "b.pdf", b"identical-bytes")
    output_dir = tmp_path / "outputs"
    store = FileScreeningStore(output_dir=output_dir)

    def components():
        return Components(
            parser=FakeDocumentParser(), retriever=FakeEvidenceRetriever(), adjudicator=FakeCriterionAdjudicator(), store=store
        )

    req1 = ScreeningRequest(
        items=[ScreeningItem(document_id="doc-A", report_id="report-A", pdf_path=pdf1)],
        criteria_set=make_criteria_set(), config=make_config(), output_dir=output_dir, run_id="run-A",
    )
    result1 = _run_ft_screening(req1, components())
    assert result1.items[0].provenance.cache_status == "miss"

    req2 = ScreeningRequest(
        items=[ScreeningItem(document_id="doc-B", report_id="report-B", pdf_path=pdf2)],  # different IDs, same bytes
        criteria_set=make_criteria_set(), config=make_config(), output_dir=output_dir, run_id="run-B",
    )
    result2 = _run_ft_screening(req2, components())
    item = result2.items[0]
    assert item.provenance.cache_status == "hit"
    assert item.provenance.source_result_run_id == "run-A"  # NOT None, and NOT "run-B"
    assert item.eligibility.document_id == "doc-B"
    for assessment in item.eligibility.assessments:
        for span in assessment.evidence:
            assert span.document_id == "doc-B"  # rebound, never "doc-A"


def test_finding6_invalid_assessment_never_reaches_the_cache(tmp_path):
    pdf1 = make_pdf(tmp_path, "a.pdf")
    output_dir = tmp_path / "outputs"
    components = make_components(
        output_dir, field_overrides_by_key={("d1", "population"): {"criterion_id": "wrong"}}
    )
    request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1)],
        criteria_set=make_criteria_set(), config=make_config(), output_dir=output_dir,
    )
    result = _run_ft_screening(request, components)
    assert result.items[0].execution_status == "processing_failed"

    cache_dir = output_dir / "cache"
    cache_files = list(cache_dir.rglob("*.json")) if cache_dir.exists() else []
    assert cache_files == []  # the boundary violation was caught before put_cached_assessment


# ---------------------------------------------------------------------------
# Finding 8: bounded incremental submission under fail_fast + concurrency
# ---------------------------------------------------------------------------


def test_finding8_no_new_component_call_after_fatal_observed(tmp_path):
    items = [
        ScreeningItem(document_id=f"d{i}", report_id=f"r{i}", pdf_path=make_pdf(tmp_path, f"cc{i}.pdf", str(i).encode()))
        for i in range(6)
    ]
    output_dir = tmp_path / "outputs"
    request = ScreeningRequest(
        items=items,
        criteria_set=make_criteria_set(),
        config=make_config(item_error_policy="fail_fast", max_workers=2),
        output_dir=output_dir,
    )
    call_log: list = []
    adjudicator = FakeCriterionAdjudicator(
        fail_keys={("d0", "population")},
        delay_by_key={("d1", "population"): 0.05},  # keep d1 in flight while d0's failure registers
        call_log=call_log,
    )
    components = Components(
        parser=FakeDocumentParser(), retriever=FakeEvidenceRetriever(), adjudicator=adjudicator,
        store=FileScreeningStore(output_dir=output_dir),
    )
    result = _run_ft_screening(request, components)

    # Only the initial max_workers=2 batch (d0, d1) should ever have been
    # dispatched -- d2..d5 must never have reached the adjudicator.
    called_documents = {key[0] for key, _ts in call_log}
    assert called_documents == {"d0", "d1"}
    statuses = {i.document_id: i.execution_status for i in result.items}
    assert statuses["d0"] == "processing_failed"
    assert statuses["d1"] == "completed"  # already running -- allowed to finish
    for doc_id in ("d2", "d3", "d4", "d5"):
        assert statuses[doc_id] == "not_processed"


# ---------------------------------------------------------------------------
# Finding 9: store lifecycle hardening -- races and corrupted artifacts
# ---------------------------------------------------------------------------


def test_finding9_concurrent_fresh_run_creation_only_one_wins(tmp_path):
    output_dir = tmp_path / "outputs"
    store_a = FileScreeningStore(output_dir=output_dir)
    store_b = FileScreeningStore(output_dir=output_dir)
    results = {}
    barrier = threading.Barrier(2)

    def attempt(name, store):
        barrier.wait()
        try:
            store.begin_run(run_id="race-run", resume=False)
            results[name] = "success"
        except Exception as exc:
            results[name] = type(exc).__name__

    t1 = threading.Thread(target=attempt, args=("a", store_a))
    t2 = threading.Thread(target=attempt, args=("b", store_b))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    outcomes = sorted(results.values())
    assert outcomes == ["RunSetupError", "success"]  # never a raw FileExistsError


def test_finding9_corrupt_item_results_line_skipped_not_crashed(tmp_path):
    output_dir = tmp_path / "outputs"
    store = FileScreeningStore(output_dir=output_dir)
    store.begin_run(run_id="corrupt-run", resume=False)
    try:
        _, item_results_path, _ = store.artifact_paths(run_id="corrupt-run")
        item_results_path.parent.mkdir(parents=True, exist_ok=True)
        item_results_path.write_text("{not valid json\n")
        results = store.read_completed_item_results(run_id="corrupt-run")
        assert results == {}  # corrupt line skipped, not raised
    finally:
        store.end_run(run_id="corrupt-run")


def test_finding9_corrupt_cache_entry_treated_as_miss(tmp_path):
    output_dir = tmp_path / "outputs"
    store = FileScreeningStore(output_dir=output_dir)
    cache_path = store._cache_path(cache_key="deadbeef" * 8)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("not json at all")
    assert store.get_cached_assessment(cache_key="deadbeef" * 8) is None


def test_finding9_corrupt_manifest_treated_as_missing(tmp_path):
    output_dir = tmp_path / "outputs"
    store = FileScreeningStore(output_dir=output_dir)
    store.begin_run(run_id="bad-manifest-run", resume=False)
    try:
        manifest_path = store._manifest_path("bad-manifest-run")
        # codex_code_review_810ccf0.md finding 2: begin_run() no longer
        # pre-creates the run directory (only manifest.json's own atomic
        # write does, as a side effect) -- this test writes directly to the
        # manifest path without going through that write, so it must create
        # the parent directory itself now.
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text("{broken")
        assert store.read_run_manifest(run_id="bad-manifest-run") is None
    finally:
        store.end_run(run_id="bad-manifest-run")


def test_finding9_provenance_timing_reflects_real_processing_duration(tmp_path):
    pdf1 = make_pdf(tmp_path, "a.pdf")
    output_dir = tmp_path / "outputs"
    request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1)],
        criteria_set=make_criteria_set(), config=make_config(), output_dir=output_dir,
    )
    components = make_components(output_dir, delay_by_key={("d1", "population"): 0.15})
    result = _run_ft_screening(request, components)
    provenance = result.items[0].provenance
    elapsed = (provenance.completed_at - provenance.started_at).total_seconds()
    assert elapsed >= 0.1  # started_at captured BEFORE the delayed adjudication, not after
