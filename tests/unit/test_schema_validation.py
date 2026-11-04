"""Local-validator tests for ft_screening.schema.

Covers api_contract_v2.md correction 6's "local Pydantic validators":
unique criterion IDs/reason codes, span offset ordering,
revision/supersession consistency, decision/adjudication consistency,
DocumentCoverage page-list invariants, ScreeningRequest ID uniqueness and
at-most-one-run-identifier, and ScreeningRunResult summary/ordering
consistency.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from ft_screening.schema import (
    CriteriaSet,
    CriterionAssessment,
    CriterionSpec,
    DocumentCoverage,
    EligibilityResult,
    EvidenceSpan,
    ExecutionProvenance,
    HumanAdjudication,
    ProcessingError,
    ProgressEvent,
    RunArtifacts,
    RunSummary,
    ScreeningConfig,
    ScreeningItem,
    ScreeningItemResult,
    ScreeningRequest,
    ScreeningRunResult,
)

NOW = datetime(2026, 8, 12, tzinfo=timezone.utc)


def make_criterion(**overrides) -> CriterionSpec:
    defaults = dict(
        criterion_id="population",
        version="1",
        eligibility_requirement="study population is adult humans",
        missing_evidence_policy="review",
        exclusion_reason_code="wrong_population",
        exclusion_reason_label="Wrong population",
        priority=1,
    )
    defaults.update(overrides)
    return CriterionSpec(**defaults)


def make_criteria_set(criteria=None) -> CriteriaSet:
    return CriteriaSet(
        criteria_set_id="cs-1",
        version="1",
        criteria=criteria or [make_criterion()],
    )


# ---------------------------------------------------------------------------
# CriterionSpec / CriteriaSet
# ---------------------------------------------------------------------------


def test_criterion_spec_valid():
    make_criterion()


def test_criterion_spec_allowed_sources_empty_rejected():
    with pytest.raises(ValidationError):
        make_criterion(allowed_sources=[])


def test_criterion_spec_allowed_sources_duplicate_rejected():
    with pytest.raises(ValidationError):
        make_criterion(allowed_sources=["Methods", "Methods"])


def test_criteria_set_duplicate_criterion_id_rejected():
    c1 = make_criterion(criterion_id="pop", exclusion_reason_code="r1")
    c2 = make_criterion(criterion_id="pop", exclusion_reason_code="r2")
    with pytest.raises(ValidationError):
        make_criteria_set([c1, c2])


def test_criteria_set_duplicate_exclusion_reason_code_rejected():
    c1 = make_criterion(criterion_id="pop", exclusion_reason_code="r1")
    c2 = make_criterion(criterion_id="design", exclusion_reason_code="r1")
    with pytest.raises(ValidationError):
        make_criteria_set([c1, c2])


def test_criteria_set_empty_criteria_rejected():
    with pytest.raises(ValidationError):
        CriteriaSet(criteria_set_id="cs-1", version="1", criteria=[])


# ---------------------------------------------------------------------------
# EvidenceSpan
# ---------------------------------------------------------------------------


def make_span(**overrides) -> EvidenceSpan:
    defaults = dict(
        document_id="doc-1",
        document_sha256="a" * 64,
        chunk_id="chunk-1",
        section="Methods",
        page=3,
        start_char=10,
        end_char=20,
        excerpt="adult participants aged 18+",
        stance="supports",
        retrieval_method="lexical",
        retrieval_query_version="1",
        rank=1,
        score=0.9,
    )
    defaults.update(overrides)
    return EvidenceSpan(**defaults)


def test_evidence_span_valid():
    make_span()


def test_evidence_span_end_before_start_rejected():
    with pytest.raises(ValidationError):
        make_span(start_char=20, end_char=10)


def test_evidence_span_equal_start_end_allowed():
    make_span(start_char=10, end_char=10)


# ---------------------------------------------------------------------------
# EligibilityResult
# ---------------------------------------------------------------------------


def make_assessment(**overrides) -> CriterionAssessment:
    defaults = dict(
        criterion_id="population",
        criterion_version="1",
        status="satisfied",
        evidence=[],
        rationale="clearly adult population",
        requires_human_review=False,
        model="stub-model",
        prompt_template_version="1",
        output_schema_version="1",
    )
    defaults.update(overrides)
    return CriterionAssessment(**defaults)


def make_eligibility_result(**overrides) -> EligibilityResult:
    defaults = dict(
        result_id="res-1",
        revision=1,
        supersedes_result_id=None,
        document_id="doc-1",
        document_sha256="a" * 64,
        report_id="report-1",
        study_id=None,
        criteria_set_hash="b" * 64,
        execution_fingerprint="c" * 64,
        assessments=[make_assessment()],
        proposed_decision="include",
        final_decision="include",
        exclusion_reason_codes=[],
        primary_exclusion_reason_code=None,
        human_adjudication=None,
        created_at=NOW,
    )
    defaults.update(overrides)
    return EligibilityResult(**defaults)


def test_eligibility_result_valid():
    make_eligibility_result()


def test_revision_1_with_supersedes_rejected():
    with pytest.raises(ValidationError):
        make_eligibility_result(revision=1, supersedes_result_id="res-0")


def test_revision_gt_1_without_supersedes_rejected():
    with pytest.raises(ValidationError):
        make_eligibility_result(revision=2, supersedes_result_id=None)


def test_revision_gt_1_with_supersedes_allowed():
    make_eligibility_result(revision=2, supersedes_result_id="res-0")


def test_final_decision_must_equal_proposed_without_adjudication():
    with pytest.raises(ValidationError):
        make_eligibility_result(proposed_decision="include", final_decision="exclude")


def test_final_decision_must_equal_human_adjudication_when_present():
    adjudication = HumanAdjudication(decision="exclude", adjudicator="reviewer-1", adjudicated_at=NOW)
    with pytest.raises(ValidationError):
        make_eligibility_result(
            proposed_decision="include",
            final_decision="include",
            human_adjudication=adjudication,
        )


def test_final_decision_matching_human_adjudication_allowed():
    adjudication = HumanAdjudication(decision="exclude", adjudicator="reviewer-1", adjudicated_at=NOW)
    make_eligibility_result(
        proposed_decision="include",
        final_decision="exclude",
        human_adjudication=adjudication,
    )


def test_primary_exclusion_reason_requires_exclude_decision():
    with pytest.raises(ValidationError):
        make_eligibility_result(
            proposed_decision="include",
            final_decision="include",
            primary_exclusion_reason_code="wrong_population",
        )


def test_primary_exclusion_reason_must_be_in_list():
    with pytest.raises(ValidationError):
        make_eligibility_result(
            proposed_decision="exclude",
            final_decision="exclude",
            exclusion_reason_codes=["other_reason"],
            primary_exclusion_reason_code="wrong_population",
        )


def test_primary_exclusion_reason_valid():
    make_eligibility_result(
        proposed_decision="exclude",
        final_decision="exclude",
        exclusion_reason_codes=["wrong_population"],
        primary_exclusion_reason_code="wrong_population",
    )


# ---------------------------------------------------------------------------
# DocumentCoverage
# ---------------------------------------------------------------------------


def make_coverage(**overrides) -> DocumentCoverage:
    defaults = dict(
        status="complete",
        total_pages=3,
        parsed_pages=[1, 2, 3],
        failed_pages=[],
        ocr_used=False,
        extracted_characters=1000,
        warnings=[],
    )
    defaults.update(overrides)
    return DocumentCoverage(**defaults)


def test_document_coverage_valid():
    make_coverage()


def test_document_coverage_overlapping_pages_rejected():
    with pytest.raises(ValidationError):
        make_coverage(parsed_pages=[1, 2], failed_pages=[2, 3])


def test_document_coverage_duplicate_parsed_pages_rejected():
    with pytest.raises(ValidationError):
        make_coverage(parsed_pages=[1, 1, 2])


def test_document_coverage_page_exceeds_total_rejected():
    with pytest.raises(ValidationError):
        make_coverage(total_pages=2, parsed_pages=[1, 2, 3])


def test_document_coverage_complete_with_failed_pages_rejected():
    with pytest.raises(ValidationError):
        make_coverage(status="complete", parsed_pages=[1, 2], failed_pages=[3])


def test_document_coverage_failed_with_parsed_pages_rejected():
    with pytest.raises(ValidationError):
        make_coverage(status="failed", parsed_pages=[1], failed_pages=[2, 3], total_pages=3)


def test_document_coverage_partial_allowed():
    make_coverage(status="partial", total_pages=3, parsed_pages=[1, 2], failed_pages=[3])


# ---------------------------------------------------------------------------
# ScreeningRequest
# ---------------------------------------------------------------------------


def make_item(**overrides) -> ScreeningItem:
    defaults = dict(
        document_id="doc-1",
        report_id="report-1",
        pdf_path=Path("data/pdfs/report-1.pdf"),
    )
    defaults.update(overrides)
    return ScreeningItem(**defaults)


def make_config(**overrides) -> ScreeningConfig:
    defaults = dict(
        parser_name="docling",
        parser_version="1",
        chunker_version="1",
        retriever_name="lexical",
        retriever_config_version="1",
        adjudicator_provider="stub",
        adjudicator_model="stub-model",
        model_config_id="cfg-1",
        prompt_template_version="1",
        output_schema_version="1",
        aggregation_policy_version="1",
    )
    defaults.update(overrides)
    return ScreeningConfig(**defaults)


def make_request(items=None, **overrides) -> ScreeningRequest:
    defaults = dict(
        items=items or [make_item()],
        criteria_set=make_criteria_set(),
        config=make_config(),
        output_dir=Path("outputs/ft_screening"),
    )
    defaults.update(overrides)
    return ScreeningRequest(**defaults)


def test_screening_request_valid():
    make_request()


def test_screening_request_both_run_identifiers_rejected():
    with pytest.raises(ValidationError):
        make_request(run_id="run-1", resume_run_id="run-0")


def test_screening_request_only_run_id_allowed():
    make_request(run_id="run-1")


def test_screening_request_only_resume_run_id_allowed():
    make_request(resume_run_id="run-0")


def test_screening_request_duplicate_document_id_rejected():
    items = [make_item(document_id="doc-1", report_id="r1"), make_item(document_id="doc-1", report_id="r2")]
    with pytest.raises(ValidationError):
        make_request(items=items)


def test_screening_request_duplicate_report_id_rejected():
    items = [make_item(document_id="doc-1", report_id="r1"), make_item(document_id="doc-2", report_id="r1")]
    with pytest.raises(ValidationError):
        make_request(items=items)


# ---------------------------------------------------------------------------
# ScreeningRunResult
# ---------------------------------------------------------------------------


def make_provenance(**overrides) -> ExecutionProvenance:
    defaults = dict(
        document_sha256="a" * 64,
        criteria_set_hash="b" * 64,
        execution_fingerprint="c" * 64,
        cache_status="miss",
        source_result_run_id=None,
        started_at=NOW,
        completed_at=NOW,
    )
    defaults.update(overrides)
    return ExecutionProvenance(**defaults)


def make_item_result(**overrides) -> ScreeningItemResult:
    defaults = dict(
        input_index=0,
        document_id="doc-1",
        report_id="report-1",
        study_id=None,
        citation={},
        execution_status="completed",
        eligibility=make_eligibility_result(),
        coverage=None,
        provenance=make_provenance(),
        errors=[],
    )
    defaults.update(overrides)
    return ScreeningItemResult(**defaults)


def test_item_result_completed_requires_eligibility():
    with pytest.raises(ValidationError):
        make_item_result(execution_status="completed", eligibility=None)


def test_item_result_completed_with_fatal_error_rejected():
    fatal = ProcessingError(
        code="unexpected_error",
        stage="adjudicate",
        severity="fatal",
        message="boom",
        retryable=False,
    )
    with pytest.raises(ValidationError):
        make_item_result(execution_status="completed", errors=[fatal])


def test_item_result_non_completed_with_eligibility_rejected():
    with pytest.raises(ValidationError):
        make_item_result(execution_status="input_unavailable", eligibility=make_eligibility_result())


def test_item_result_not_processed_without_eligibility_allowed():
    make_item_result(execution_status="not_processed", eligibility=None)


def make_run_result(items, run_status="completed", **overrides) -> ScreeningRunResult:
    completed = sum(1 for i in items if i.execution_status == "completed")
    included = sum(
        1 for i in items if i.execution_status == "completed" and i.eligibility.final_decision == "include"
    )
    excluded = sum(
        1 for i in items if i.execution_status == "completed" and i.eligibility.final_decision == "exclude"
    )
    needs_review = sum(
        1
        for i in items
        if i.execution_status == "completed" and i.eligibility.final_decision == "needs_human_review"
    )
    input_unavailable = sum(1 for i in items if i.execution_status == "input_unavailable")
    duplicate_input = sum(1 for i in items if i.execution_status == "duplicate_input")
    processing_failed = sum(1 for i in items if i.execution_status == "processing_failed")
    not_processed = sum(1 for i in items if i.execution_status == "not_processed")

    defaults = dict(
        api_version="1.0",
        schema_version="1",
        run_id="run-1",
        run_status=run_status,
        criteria_set_id="cs-1",
        criteria_set_version="1",
        criteria_set_hash="b" * 64,
        execution_fingerprint="c" * 64,
        started_at=NOW,
        completed_at=NOW,
        items=items,
        summary=RunSummary(
            total=len(items),
            completed=completed,
            included=included,
            excluded=excluded,
            needs_human_review=needs_review,
            input_unavailable=input_unavailable,
            duplicate_input=duplicate_input,
            processing_failed=processing_failed,
            not_processed=not_processed,
            cache_hits=0,
        ),
        artifacts=RunArtifacts(
            run_result_json=Path("outputs/ft_screening/runs/run-1/run_result.json"),
            item_results_jsonl=Path("outputs/ft_screening/runs/run-1/item_results.jsonl"),
            ledger_jsonl=Path("outputs/ft_screening/runs/run-1/ledger.jsonl"),
        ),
    )
    defaults.update(overrides)
    return ScreeningRunResult(**defaults)


def test_run_result_valid_single_completed_item():
    make_run_result([make_item_result()])


def test_run_result_wrong_input_index_order_rejected():
    second_eligibility = make_eligibility_result(document_id="d2", report_id="r2")
    second_provenance = make_provenance()
    items = [
        make_item_result(input_index=1),
        make_item_result(
            input_index=0,
            report_id="r2",
            document_id="d2",
            eligibility=second_eligibility,
            provenance=second_provenance,
        ),
    ]
    with pytest.raises(ValidationError):
        make_run_result(items)


def test_run_result_summary_mismatch_rejected():
    item = make_item_result()
    run = None
    with pytest.raises(ValidationError):
        ScreeningRunResult(
            api_version="1.0",
            schema_version="1",
            run_id="run-1",
            run_status="completed",
            criteria_set_id="cs-1",
            criteria_set_version="1",
            criteria_set_hash="b" * 64,
            execution_fingerprint="c" * 64,
            started_at=NOW,
            completed_at=NOW,
            items=[item],
            summary=RunSummary(
                total=999,  # deliberately wrong
                completed=1,
                included=1,
                excluded=0,
                needs_human_review=0,
                input_unavailable=0,
                duplicate_input=0,
                processing_failed=0,
                not_processed=0,
                cache_hits=0,
            ),
            artifacts=RunArtifacts(
                run_result_json=Path("run_result.json"),
                item_results_jsonl=Path("item_results.jsonl"),
                ledger_jsonl=Path("ledger.jsonl"),
            ),
        )


def test_run_status_completed_requires_no_non_completed_items():
    not_processed_item = make_item_result(
        input_index=1,
        document_id="doc-2",
        report_id="report-2",
        execution_status="not_processed",
        eligibility=None,
    )
    items = [make_item_result(), not_processed_item]
    with pytest.raises(ValidationError):
        make_run_result(items, run_status="completed")


def test_run_status_completed_with_issues_allowed():
    not_processed_item = make_item_result(
        input_index=1,
        document_id="doc-2",
        report_id="report-2",
        execution_status="not_processed",
        eligibility=None,
    )
    items = [make_item_result(), not_processed_item]
    make_run_result(items, run_status="completed_with_issues")


def test_run_status_failed_requires_no_completed_items():
    completed_item = make_item_result()  # execution_status="completed" by default
    with pytest.raises(ValidationError):
        make_run_result([completed_item], run_status="failed")


def test_run_status_failed_allowed_when_nothing_completed():
    not_processed_item = make_item_result(
        input_index=0,
        document_id="doc-1",
        report_id="report-1",
        execution_status="not_processed",
        eligibility=None,
    )
    make_run_result([not_processed_item], run_status="failed")


# ---------------------------------------------------------------------------
# Finding 3: completed_with_issues must not accept a zero-completed run
# (executable counterexample from codex_code_review_bae78ac.md)
# ---------------------------------------------------------------------------


def test_completed_with_issues_rejects_zero_completed_items():
    all_not_processed = [
        make_item_result(
            input_index=0, document_id="doc-1", report_id="report-1",
            execution_status="not_processed", eligibility=None,
        ),
        make_item_result(
            input_index=1, document_id="doc-2", report_id="report-2",
            execution_status="not_processed", eligibility=None,
        ),
    ]
    with pytest.raises(ValidationError):
        make_run_result(all_not_processed, run_status="completed_with_issues")


def test_all_invalid_item_batch_uses_failed_status():
    all_not_processed = [
        make_item_result(
            input_index=0, document_id="doc-1", report_id="report-1",
            execution_status="not_processed", eligibility=None,
        ),
        make_item_result(
            input_index=1, document_id="doc-2", report_id="report-2",
            execution_status="not_processed", eligibility=None,
        ),
    ]
    make_run_result(all_not_processed, run_status="failed")  # must not raise


# ---------------------------------------------------------------------------
# Finding 2: nested identity/fingerprint consistency
# ---------------------------------------------------------------------------


def test_completed_item_eligibility_document_id_must_match():
    mismatched_eligibility = make_eligibility_result(document_id="OTHER")
    with pytest.raises(ValidationError):
        make_item_result(eligibility=mismatched_eligibility)


def test_completed_item_eligibility_report_id_must_match():
    mismatched_eligibility = make_eligibility_result(report_id="OTHER")
    with pytest.raises(ValidationError):
        make_item_result(eligibility=mismatched_eligibility)


def test_completed_item_eligibility_study_id_must_match():
    mismatched_eligibility = make_eligibility_result(study_id="study-x")
    with pytest.raises(ValidationError):
        make_item_result(eligibility=mismatched_eligibility, study_id=None)


def test_completed_item_requires_provenance():
    with pytest.raises(ValidationError):
        make_item_result(provenance=None)


def test_completed_item_provenance_document_sha256_must_match_eligibility():
    mismatched_provenance = make_provenance(document_sha256="d" * 64)
    with pytest.raises(ValidationError):
        make_item_result(provenance=mismatched_provenance)


def test_completed_item_provenance_criteria_set_hash_must_match_eligibility():
    mismatched_provenance = make_provenance(criteria_set_hash="d" * 64)
    with pytest.raises(ValidationError):
        make_item_result(provenance=mismatched_provenance)


def test_completed_item_provenance_execution_fingerprint_must_match_eligibility():
    mismatched_provenance = make_provenance(execution_fingerprint="d" * 64)
    with pytest.raises(ValidationError):
        make_item_result(provenance=mismatched_provenance)


def test_run_result_rejects_item_with_mismatched_criteria_set_hash():
    mismatched_eligibility = make_eligibility_result(criteria_set_hash="d" * 64)
    mismatched_provenance = make_provenance(criteria_set_hash="d" * 64)
    item = make_item_result(eligibility=mismatched_eligibility, provenance=mismatched_provenance)
    with pytest.raises(ValidationError):
        make_run_result([item])  # make_run_result's default criteria_set_hash is "b"*64


def test_run_result_rejects_item_with_mismatched_execution_fingerprint():
    mismatched_eligibility = make_eligibility_result(execution_fingerprint="d" * 64)
    mismatched_provenance = make_provenance(execution_fingerprint="d" * 64)
    item = make_item_result(eligibility=mismatched_eligibility, provenance=mismatched_provenance)
    with pytest.raises(ValidationError):
        make_run_result([item])  # make_run_result's default execution_fingerprint is "c"*64


def test_run_result_rejects_duplicate_document_report_pairs():
    item_a = make_item_result(input_index=0)
    item_b = make_item_result(input_index=1)  # same document_id/report_id as item_a
    with pytest.raises(ValidationError):
        make_run_result([item_a, item_b])


# ---------------------------------------------------------------------------
# Finding 4: input validation gaps
# ---------------------------------------------------------------------------


def test_run_id_rejects_path_traversal():
    with pytest.raises(ValidationError):
        make_request(run_id="../escape")


def test_run_id_rejects_path_separator():
    with pytest.raises(ValidationError):
        make_request(run_id="a/b")


def test_run_id_accepts_safe_token():
    make_request(run_id="run-2026-08-12_001")


def test_config_rejects_empty_parser_name():
    with pytest.raises(ValidationError):
        make_config(parser_name="")


def test_config_rejects_empty_adjudicator_model():
    with pytest.raises(ValidationError):
        make_config(adjudicator_model="")


def test_citation_rejects_non_json_safe_value():
    with pytest.raises(ValidationError):
        make_item(citation={"tags": {1, 2, 3}})  # a set is not JSON-safe


def test_citation_rejects_non_finite_float():
    with pytest.raises(ValidationError):
        make_item(citation={"score": float("nan")})


def test_citation_defensively_copies_nested_mutable_input():
    original = {"authors": ["A", "B"]}
    item = make_item(citation=original)
    original["authors"].append("C")
    assert item.citation["authors"] == ["A", "B"]  # unaffected by post-construction mutation


def test_config_options_reject_non_json_safe_value():
    with pytest.raises(ValidationError):
        make_config(parser_options={"bad": object()})


def test_evidence_span_document_sha256_rejects_wrong_length():
    with pytest.raises(ValidationError):
        make_span(document_sha256="not-a-sha256")


def test_evidence_span_document_sha256_normalizes_case():
    span = make_span(document_sha256="A" * 64)
    assert span.document_sha256 == "a" * 64


def test_eligibility_result_rejects_malformed_criteria_set_hash():
    with pytest.raises(ValidationError):
        make_eligibility_result(criteria_set_hash="not-a-hash")


def test_eligibility_result_rejects_malformed_execution_fingerprint():
    with pytest.raises(ValidationError):
        make_eligibility_result(execution_fingerprint="not-a-hash")


def test_provenance_rejects_malformed_criteria_set_hash():
    with pytest.raises(ValidationError):
        make_provenance(criteria_set_hash="not-a-hash")


def test_provenance_rejects_malformed_execution_fingerprint():
    with pytest.raises(ValidationError):
        make_provenance(execution_fingerprint="not-a-hash")


def test_run_result_rejects_malformed_criteria_set_hash():
    with pytest.raises(ValidationError):
        make_run_result([make_item_result()], criteria_set_hash="not-a-hash")


def test_run_result_rejects_malformed_execution_fingerprint():
    with pytest.raises(ValidationError):
        make_run_result([make_item_result()], execution_fingerprint="not-a-hash")


def test_evidence_span_score_rejects_infinity():
    with pytest.raises(ValidationError):
        make_span(score=float("inf"))


def test_evidence_span_score_rejects_nan():
    with pytest.raises(ValidationError):
        make_span(score=float("nan"))


def test_allowed_sources_duplicate_after_normalization_rejected():
    with pytest.raises(ValidationError):
        make_criterion(allowed_sources=["Methods", " methods "])  # same after NFC+trim+casefold


def test_allowed_sources_normalized_on_construction():
    criterion = make_criterion(allowed_sources=[" Methods ", "Results"])
    assert criterion.allowed_sources == ["methods", "results"]


# ---------------------------------------------------------------------------
# Finding 5: DocumentCoverage contradictory states
# (executable counterexamples from codex_code_review_bae78ac.md)
# ---------------------------------------------------------------------------


def test_document_coverage_complete_with_zero_extracted_characters_rejected():
    with pytest.raises(ValidationError):
        make_coverage(status="complete", total_pages=3, parsed_pages=[1], failed_pages=[], extracted_characters=0)


def test_document_coverage_complete_with_partial_page_list_rejected():
    with pytest.raises(ValidationError):
        make_coverage(status="complete", total_pages=3, parsed_pages=[1], extracted_characters=100)


def test_document_coverage_failed_with_nonzero_extracted_characters_rejected():
    with pytest.raises(ValidationError):
        make_coverage(status="failed", parsed_pages=[], failed_pages=[1], extracted_characters=500)


def test_document_coverage_failed_zero_everything_allowed():
    make_coverage(status="failed", total_pages=1, parsed_pages=[], failed_pages=[1], extracted_characters=0)


# ---------------------------------------------------------------------------
# Finding 6: timestamps must be tz-aware UTC and correctly ordered
# ---------------------------------------------------------------------------


def test_eligibility_result_rejects_naive_datetime():
    with pytest.raises(ValidationError):
        make_eligibility_result(created_at=datetime(2026, 8, 12))  # no tzinfo


def test_eligibility_result_normalizes_non_utc_offset():
    plus_two = timezone(timedelta(hours=2))
    dt = datetime(2026, 8, 12, 14, 0, tzinfo=plus_two)
    result = make_eligibility_result(created_at=dt)
    assert result.created_at.utcoffset() == timedelta(0)
    assert result.created_at.hour == 12  # 14:00+02:00 -> 12:00 UTC


def test_provenance_rejects_completed_before_started():
    with pytest.raises(ValidationError):
        make_provenance(started_at=NOW, completed_at=datetime(2020, 1, 1, tzinfo=timezone.utc))


def test_provenance_allows_completed_equal_started():
    make_provenance(started_at=NOW, completed_at=NOW)


def test_run_result_requires_completed_at():
    with pytest.raises(ValidationError):
        make_run_result([make_item_result()], completed_at=None)


def test_run_result_rejects_completed_before_started():
    earlier = datetime(2020, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(ValidationError):
        make_run_result([make_item_result()], started_at=NOW, completed_at=earlier)


# ---------------------------------------------------------------------------
# Additional test gaps: ProgressEvent
# ---------------------------------------------------------------------------


def test_progress_event_completed_items_must_not_exceed_total():
    with pytest.raises(ValidationError):
        ProgressEvent(
            run_id="run-1",
            event="run_started",
            occurred_at=NOW,
            completed_items=5,
            total_items=2,
        )


def test_progress_event_item_started_requires_item_identity():
    with pytest.raises(ValidationError):
        ProgressEvent(
            run_id="run-1",
            event="item_started",
            occurred_at=NOW,
            completed_items=0,
            total_items=1,
            # input_index/document_id/report_id all omitted
        )


def test_progress_event_item_started_with_identity_allowed():
    ProgressEvent(
        run_id="run-1",
        event="item_started",
        occurred_at=NOW,
        completed_items=0,
        total_items=1,
        input_index=0,
        document_id="doc-1",
        report_id="report-1",
    )


def test_progress_event_run_started_does_not_require_item_identity():
    ProgressEvent(run_id="run-1", event="run_started", occurred_at=NOW, completed_items=0, total_items=1)


def test_progress_event_rejects_naive_timestamp():
    with pytest.raises(ValidationError):
        ProgressEvent(
            run_id="run-1",
            event="run_started",
            occurred_at=datetime(2026, 8, 12),
            completed_items=0,
            total_items=1,
        )


# ---------------------------------------------------------------------------
# Additional test gaps: JSON-mode round trips
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda: make_criterion(),
        lambda: make_criteria_set(),
        lambda: make_span(),
        lambda: make_assessment(),
        lambda: make_eligibility_result(),
        lambda: make_item(citation={"title": "x", "authors": ["A", "B"], "year": 2026}),
        lambda: make_config(parser_options={"nested": {"a": [1, 2, 3]}}),
        lambda: make_request(),
        lambda: make_coverage(),
        lambda: make_provenance(),
        lambda: make_item_result(),
        lambda: make_run_result([make_item_result()]),
    ],
)
def test_json_mode_round_trip(build):
    model = build()
    dumped = model.model_dump(mode="json")
    restored = type(model).model_validate(dumped)
    assert restored == model
