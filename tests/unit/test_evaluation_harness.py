"""Tests for ft_screening.evaluation_harness (v6): the three-arm
evaluation metrics defined in gold_set_protocol_v7.md §8. Fully
offline/synthetic -- exercises the metric FORMULAS and Arm 2's wiring (via
a small local fake adjudicator, no real LLM call), not the real
20-document gold set (which doesn't exist until annotation is complete).

Written after three rounds of adversarial review of this harness
(codex_code_review_a36e0df.md, codex_code_review_dca0970.md,
codex_code_review_1c40a27.md) -- several tests here directly reproduce
those reviews' exact adversarial repros.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from ft_screening.components import AdjudicationOutcome, AdjudicationUsage, Components
from ft_screening.evaluation_harness import (
    MISSING_OR_FAILED,
    DocumentDecisionResult,
    EvaluationResultArtifact,
    GoldEvidenceCheckpointDriftError,
    _document_sha256_for_decision,
    build_criterion_status_predictions_from_gold_evidence,
    build_criterion_status_predictions_from_run_result,
    build_document_decisions_from_run_result,
    compute_arm_gold_evidence_document_decisions,
    compute_retrieval_oracle_metrics,
    correct_exclusion_reason_rate,
    criterion_status_agreement,
    false_exclusion_rate,
    false_inclusion_rate,
    final_decision_agreement,
    load_frozen_evaluation_result_artifact,
    parser_success_rate,
    processing_completion_report,
    review_rate,
    run_arm_e2e,
    run_arm_gold_evidence,
    token_usage_summary_e2e,
    token_usage_summary_gold_evidence,
    token_usage_summary_gold_evidence_per_document,
    verify_result_artifact_against_gold_set,
    wall_time_summary_e2e,
    wall_time_summary_gold_evidence,
)
from ft_screening.fakes import FakeCriterionAdjudicator, FakeDocumentParser, FakeEvidenceRetriever
from ft_screening.gold import (
    GoldCriterionAnnotation,
    GoldDocumentAnnotation,
    GoldSet,
    RetrievalOracleArtifact,
    RetrievalOracleJudgmentSet,
    RetrievalOracleSpan,
    RetrievalSufficiencyJudgment,
    WorksheetSourceProvenance,
    WorksheetSourceRecord,
    hash_pdf_file,
    make_evidence_key,
)
from ft_screening.hashing import compute_criteria_set_hash
from ft_screening.schema import (
    CriteriaSet,
    CriterionAssessment,
    CriterionSpec,
    DocumentCoverage,
    EligibilityResult,
    EvidenceSpan,
    ExecutionProvenance,
    RunArtifacts,
    RunSummary,
    ScreeningConfig,
    ScreeningItemResult,
    ScreeningRunResult,
)
from ft_screening.store import FileScreeningStore

NOW = datetime(2026, 8, 15, tzinfo=timezone.utc)


def make_criteria_set() -> CriteriaSet:
    criteria = [
        CriterionSpec(
            criterion_id="population", version="1", eligibility_requirement="adult humans",
            missing_evidence_policy="review", exclusion_reason_code="wrong_population",
            exclusion_reason_label="Wrong population", priority=1,
        ),
        CriterionSpec(
            criterion_id="design", version="1", eligibility_requirement="randomized controlled trial",
            missing_evidence_policy="pass", exclusion_reason_code="wrong_design",
            exclusion_reason_label="Wrong design", priority=2,
        ),
    ]
    return CriteriaSet(criteria_set_id="cs-1", version="1", criteria=criteria)


def make_criterion_annotation(**overrides) -> GoldCriterionAnnotation:
    defaults = dict(
        document_id="d1", criterion_id="population", document_page_count=20, gold_status="satisfied",
        supporting_pages=(3,), supporting_passages=("adults were enrolled",),
        contradicting_pages=(), contradicting_passages=(),
        evidence_absent=False, evidence_ambiguous=False, evidence_conflicting=False,
        human_review_required=False, annotator_id="RR", annotation_round="primary", notes="",
    )
    defaults.update(overrides)
    return GoldCriterionAnnotation(**defaults)


def make_document_annotation(**overrides) -> GoldDocumentAnnotation:
    defaults = dict(
        document_id="d1", document_page_count=20, gold_final_decision="include",
        gold_primary_exclusion_reason_code=None, aggregation_override_rationale=None,
        annotator_id="RR", annotation_round="primary", notes="",
    )
    defaults.update(overrides)
    return GoldDocumentAnnotation(**defaults)


def _stub_source_provenance(**overrides) -> WorksheetSourceProvenance:
    defaults = dict(
        protocol_version="gold_set_protocol_v3", criteria_set_hash="a" * 64,
        blind_manifest_hash="b" * 64, sample_manifest_hash="c" * 64,
        primary=WorksheetSourceRecord(
            role="primary", seal_path="stub/primary_seal.json", seal_content_hash="d" * 64,
            criterion_csv_path="stub/primary_criteria.csv", criterion_csv_sha256="e" * 64,
            document_csv_path="stub/primary_documents.csv", document_csv_sha256="f" * 64,
        ),
        merged_criterion_csv_sha256="1" * 64, merged_document_csv_sha256="2" * 64,
        merged_at="2026-08-15T00:00:00+00:00",
    )
    defaults.update(overrides)
    return WorksheetSourceProvenance(**defaults)


def make_gold_set(criterion_annotations, document_annotations, **overrides) -> GoldSet:
    doc_ids = tuple(sorted({a.document_id for a in document_annotations}))
    defaults = dict(
        protocol_version="gold_set_protocol_v3", blind_manifest_hash="d" * 64, sample_manifest_hash="a" * 64,
        document_pdf_sha256={d: "b" * 64 for d in doc_ids}, expected_document_ids=doc_ids,
        expected_dual_review_document_ids=(), criteria_set=make_criteria_set(),
        criterion_annotations=tuple(criterion_annotations), document_annotations=tuple(document_annotations),
        source_provenance=_stub_source_provenance(), frozen_at="2026-08-15T00:00:00+00:00",
    )
    defaults.update(overrides)
    return GoldSet(**defaults)


def _two_document_gold_set() -> GoldSet:
    criterion_annotations = [
        make_criterion_annotation(document_id="d1", criterion_id="population", document_page_count=20, human_review_required=False),
        make_criterion_annotation(
            document_id="d1", criterion_id="design", document_page_count=20,
            supporting_pages=(5,), supporting_passages=("RCT design",), human_review_required=False,
        ),
        make_criterion_annotation(
            document_id="d2", criterion_id="population", document_page_count=15, gold_status="violated",
            supporting_pages=(), supporting_passages=(), contradicting_pages=(2,),
            contradicting_passages=("children only",), human_review_required=True,
        ),
        make_criterion_annotation(
            document_id="d2", criterion_id="design", document_page_count=15, gold_status="not_reported",
            evidence_absent=True, supporting_pages=(), supporting_passages=(), human_review_required=False,
        ),
    ]
    document_annotations = [
        make_document_annotation(document_id="d1", document_page_count=20, gold_final_decision="include"),
        make_document_annotation(
            document_id="d2", document_page_count=15, gold_final_decision="exclude",
            gold_primary_exclusion_reason_code="wrong_population",
        ),
    ]
    return make_gold_set(criterion_annotations, document_annotations)


# ---------------------------------------------------------------------------
# Strict key-coverage scoring
# ---------------------------------------------------------------------------


def test_criterion_status_agreement_rejects_missing_predictions():
    gold_set = _two_document_gold_set()
    with pytest.raises(ValueError):
        criterion_status_agreement(gold_set=gold_set, predicted_status_by_key={("d1", "population"): "satisfied"})


def test_criterion_status_agreement_with_full_coverage_including_missing_or_failed():
    gold_set = _two_document_gold_set()
    predicted = {
        ("d1", "population"): "satisfied", ("d1", "design"): MISSING_OR_FAILED,
        ("d2", "population"): "violated", ("d2", "design"): "not_reported",
    }
    matrix = criterion_status_agreement(gold_set=gold_set, predicted_status_by_key=predicted)
    assert matrix.accuracy() == 0.75


def test_final_decision_agreement_with_full_coverage():
    gold_set = _two_document_gold_set()
    predicted = {"d1": "include", "d2": "needs_human_review"}
    matrix = final_decision_agreement(gold_set=gold_set, predicted_decision_by_document=predicted)
    assert matrix.accuracy() == 0.5


def test_false_exclusion_rate_reports_decided_and_missing_counts():
    """codex_code_review_dca0970.md finding 5: the denominator is the
    full gold-include count, and decided/missing must be explicit."""
    gold_set = _two_document_gold_set()
    predicted = {"d1": MISSING_OR_FAILED, "d2": "exclude"}
    report = false_exclusion_rate(gold_set=gold_set, predicted_decision_by_document=predicted)
    assert report.denominator == 1  # 1 gold-include document (d1)
    assert report.missing_count == 1
    assert report.decided_count == 0
    assert report.rate == 0.0  # missing is not scored as a false exclusion, but is not "correct" either


def test_false_inclusion_rate_with_full_coverage():
    gold_set = _two_document_gold_set()
    predicted = {"d1": "include", "d2": "include"}
    report = false_inclusion_rate(gold_set=gold_set, predicted_decision_by_document=predicted)
    assert report.rate == 1.0
    assert report.document_ids == ("d2",)


def test_review_rate_cross_tab_against_gold_hard_cases():
    gold_set = _two_document_gold_set()
    predicted = {"d1": "include", "d2": "needs_human_review"}
    report = review_rate(gold_set=gold_set, predicted_decision_by_document=predicted)
    assert report.review_rate_among_gold_hard_cases == 1.0
    assert report.review_rate_among_gold_easy_cases == 0.0


def test_correct_exclusion_reason_rate():
    gold_set = _two_document_gold_set()
    decisions = {
        "d1": DocumentDecisionResult(final_decision="include", primary_exclusion_reason_code=None),
        "d2": DocumentDecisionResult(final_decision="exclude", primary_exclusion_reason_code="wrong_population"),
    }
    assert correct_exclusion_reason_rate(gold_set=gold_set, document_decisions=decisions) == 1.0


def test_processing_completion_report_requires_exact_coverage():
    """codex_code_review_dca0970.md finding 5's exact repro: an empty
    prediction dict previously returned empty completed AND empty
    missing lists instead of naming all expected failures."""
    gold_set = _two_document_gold_set()
    with pytest.raises(ValueError):
        processing_completion_report(gold_set=gold_set, predicted_status_by_key={}, predicted_decision_by_document={})


def test_processing_completion_report_names_missing_ids():
    gold_set = _two_document_gold_set()
    predicted_status = {
        ("d1", "population"): "satisfied", ("d1", "design"): MISSING_OR_FAILED,
        ("d2", "population"): "violated", ("d2", "design"): "not_reported",
    }
    predicted_decision = {"d1": "include", "d2": MISSING_OR_FAILED}
    report = processing_completion_report(
        gold_set=gold_set, predicted_status_by_key=predicted_status, predicted_decision_by_document=predicted_decision,
    )
    assert report.missing_or_failed_document_ids == ("d2",)
    assert report.missing_or_failed_criterion_keys == (("d1", "design"),)


# ---------------------------------------------------------------------------
# Token usage / wall time -- unit-labeled, missing/unavailable never lost
# ---------------------------------------------------------------------------


def make_eligibility_result(**overrides) -> EligibilityResult:
    defaults = dict(
        result_id="res-1", revision=1, supersedes_result_id=None, document_id="d1", document_sha256="a" * 64,
        report_id="d1", study_id=None, criteria_set_hash="b" * 64, execution_fingerprint="c" * 64,
        assessments=[
            CriterionAssessment(
                criterion_id="population", criterion_version="1", status="satisfied", evidence=[],
                rationale="ok", requires_human_review=False, model="fake", prompt_template_version="1",
                output_schema_version="1",
            )
        ],
        proposed_decision="include", final_decision="include", exclusion_reason_codes=[],
        primary_exclusion_reason_code=None, human_adjudication=None, created_at=NOW,
    )
    defaults.update(overrides)
    return EligibilityResult(**defaults)


def make_provenance(**overrides) -> ExecutionProvenance:
    defaults = dict(
        document_sha256="a" * 64, criteria_set_hash="b" * 64, execution_fingerprint="c" * 64,
        cache_status="miss", source_result_run_id=None, started_at=NOW, completed_at=NOW + timedelta(seconds=42),
        usage_available=True, total_input_tokens=100, total_output_tokens=20, total_tokens=120,
        total_provider_calls=1, total_retries=0,
    )
    defaults.update(overrides)
    return ExecutionProvenance(**defaults)


def make_coverage(**overrides) -> DocumentCoverage:
    defaults = dict(status="complete", total_pages=10, parsed_pages=list(range(1, 11)), failed_pages=[], extracted_characters=1000, ocr_used=False)
    defaults.update(overrides)
    return DocumentCoverage(**defaults)


def make_item_result(**overrides) -> ScreeningItemResult:
    defaults = dict(
        input_index=0, document_id="d1", report_id="d1", study_id=None, citation={},
        execution_status="completed", eligibility=make_eligibility_result(), coverage=make_coverage(),
        provenance=make_provenance(), errors=[],
    )
    defaults.update(overrides)
    return ScreeningItemResult(**defaults)


def make_run_result(items) -> ScreeningRunResult:
    completed = sum(1 for i in items if i.execution_status == "completed")
    included = sum(1 for i in items if i.execution_status == "completed" and i.eligibility.final_decision == "include")
    excluded = sum(1 for i in items if i.execution_status == "completed" and i.eligibility.final_decision == "exclude")
    needs_review = sum(
        1 for i in items if i.execution_status == "completed" and i.eligibility.final_decision == "needs_human_review"
    )
    input_unavailable = sum(1 for i in items if i.execution_status == "input_unavailable")
    duplicate_input = sum(1 for i in items if i.execution_status == "duplicate_input")
    processing_failed = sum(1 for i in items if i.execution_status == "processing_failed")
    not_processed = sum(1 for i in items if i.execution_status == "not_processed")
    run_status = "completed" if processing_failed == 0 and not_processed == 0 else "completed_with_issues"
    return ScreeningRunResult(
        api_version="1.0", schema_version="1", run_id="run-1", run_status=run_status,
        criteria_set_id="cs-1", criteria_set_version="1", criteria_set_hash="b" * 64,
        execution_fingerprint="c" * 64, started_at=NOW, completed_at=NOW + timedelta(minutes=5), items=items,
        summary=RunSummary(
            total=len(items), completed=completed, included=included, excluded=excluded,
            needs_human_review=needs_review, input_unavailable=input_unavailable, duplicate_input=duplicate_input,
            processing_failed=processing_failed, not_processed=not_processed, cache_hits=0,
        ),
        artifacts=RunArtifacts(run_result_json="run.json", item_results_jsonl="items.jsonl", ledger_jsonl="ledger.jsonl"),
    )


def test_token_usage_summary_e2e_counts_missing_provenance_as_unavailable():
    """codex_code_review_dca0970.md finding 5's exact repro: an item with
    NO provenance at all previously vanished from both counts instead of
    being counted unavailable."""
    gold_set = _two_document_gold_set()
    items = [
        make_item_result(input_index=0, document_id="d1", report_id="d1",
                          eligibility=make_eligibility_result(document_id="d1", report_id="d1"),
                          provenance=make_provenance(total_input_tokens=100, total_output_tokens=20, total_tokens=120)),
        make_item_result(input_index=1, document_id="d2", report_id="d2", execution_status="processing_failed",
                          eligibility=None, coverage=make_coverage(status="failed", parsed_pages=[], extracted_characters=0),
                          provenance=None),
    ]
    summary = token_usage_summary_e2e(gold_set=gold_set, run_result=make_run_result(items))
    assert summary.unit == "document"
    assert summary.n_usage_available == 1
    assert summary.n_usage_unavailable == 1  # d2 counted, not silently dropped
    assert summary.total_input_tokens == 100


def test_wall_time_summary_e2e_uses_provenance_timestamps():
    items = [
        make_item_result(input_index=0, document_id="d1", report_id="d1",
                          eligibility=make_eligibility_result(document_id="d1", report_id="d1"),
                          provenance=make_provenance(started_at=NOW, completed_at=NOW + timedelta(seconds=100))),
        make_item_result(input_index=1, document_id="d2", report_id="d2",
                          eligibility=make_eligibility_result(document_id="d2", report_id="d2"),
                          provenance=make_provenance(started_at=NOW, completed_at=NOW + timedelta(seconds=200))),
    ]
    summary = wall_time_summary_e2e(make_run_result(items))
    assert summary.total_seconds == 300.0


def test_parser_success_rate():
    gold_set = _two_document_gold_set()
    items = [
        make_item_result(input_index=0, document_id="d1", report_id="d1",
                          eligibility=make_eligibility_result(document_id="d1", report_id="d1"),
                          coverage=make_coverage(status="complete")),
        make_item_result(input_index=1, document_id="d2", report_id="d2", execution_status="processing_failed",
                          eligibility=None, coverage=make_coverage(status="failed", parsed_pages=[], extracted_characters=0)),
    ]
    assert parser_success_rate(gold_set=gold_set, run_result=make_run_result(items)) == 0.5


def test_build_document_decisions_from_run_result_marks_failed_items_as_missing():
    gold_set = _two_document_gold_set()
    items = [
        make_item_result(input_index=0, document_id="d1", report_id="d1", eligibility=make_eligibility_result(document_id="d1", report_id="d1")),
        make_item_result(input_index=1, document_id="d2", report_id="d2", execution_status="processing_failed",
                          eligibility=None, coverage=make_coverage(status="failed", parsed_pages=[], extracted_characters=0)),
    ]
    decisions = build_document_decisions_from_run_result(gold_set=gold_set, run_result=make_run_result(items))
    assert decisions["d1"].final_decision == "include"
    assert decisions["d2"].final_decision == MISSING_OR_FAILED


# ---------------------------------------------------------------------------
# Arm 2 wiring
# ---------------------------------------------------------------------------


class _FakeAdjudicator:
    model = "fake-1"
    prompt_template_version = "1"
    output_schema_version = "1"

    def __init__(self, *, status_by_key=None, requires_review_by_key=None, fail_keys=None):
        self._status_by_key = status_by_key or {}
        self._requires_review_by_key = requires_review_by_key or {}
        self._fail_keys = fail_keys or set()
        self.call_log = []

    def adjudicate(self, *, document, criterion, evidence, model_options):
        key = (document.document_id, criterion.criterion_id)
        self.call_log.append(key)
        if key in self._fail_keys:
            raise RuntimeError("simulated adjudication failure")
        status = self._status_by_key.get(key, "satisfied")
        assessment = CriterionAssessment(
            criterion_id=criterion.criterion_id, criterion_version=criterion.version, status=status,
            evidence=evidence, rationale="ok", requires_human_review=self._requires_review_by_key.get(key, False),
            model=self.model, prompt_template_version="1", output_schema_version="1",
        )
        usage = AdjudicationUsage(
            model=self.model, input_tokens=100, output_tokens=20, total_tokens=120,
            provider_call_count=1, retry_count=0, usage_available=True,
        )
        return AdjudicationOutcome(assessment=assessment, usage=usage)


def test_gold_evidence_spans_exposes_only_passage_text_never_status_or_decision_fields():
    """Explicit requirement for the new Arm-2 prompt-6 runner: gold
    evidence spans handed to the adjudicator must carry ONLY the
    supporting/contradicting passage text -- never gold_status, the
    document-level gold_final_decision, or any exclusion-reason field.
    Uses distinctive, greppable sentinel values for every non-evidence
    field so a leak of ANY of them into an excerpt would be caught, not
    just the ones a hand-picked assertion happens to check."""
    from ft_screening.evaluation_harness import _gold_evidence_spans

    annotation = make_criterion_annotation(
        document_id="d1", criterion_id="population",
        gold_status="unclear",  # a real Literal value -- gold_status itself is strictly typed, unlike free-text fields
        supporting_pages=(3,), supporting_passages=("adults were enrolled",),
        contradicting_pages=(4,), contradicting_passages=("children excluded",),
        evidence_conflicting=True, human_review_required=True,
        annotator_id="SENTINEL-ANNOTATOR-ID", notes="SENTINEL-NOTES-VALUE",
    )
    spans = _gold_evidence_spans("d1", annotation)

    assert len(spans) == 2
    excerpts = {s.excerpt for s in spans}
    assert excerpts == {"adults were enrolled", "children excluded"}
    for sentinel in ("unclear", "SENTINEL-ANNOTATOR-ID", "SENTINEL-NOTES-VALUE"):
        for span in spans:
            assert sentinel not in span.excerpt
            assert sentinel not in (span.section or "")
    # GoldCriterionAnnotation carries no document-level decision/exclusion
    # fields at all (those live on GoldDocumentAnnotation, a separate
    # object never passed to _gold_evidence_spans) -- confirmed structurally,
    # not just by absence from the rendered excerpt text.
    assert not hasattr(annotation, "gold_final_decision")
    assert not hasattr(annotation, "gold_primary_exclusion_reason_code")


def test_run_arm_gold_evidence_persists_result_then_manifest_bound_to_it(tmp_path):
    """codex_code_review_772916e.md finding 4: the persisted manifest
    previously recorded configuration only, never the outcomes/failures/
    usage actually scored. The result artifact must exist and the
    manifest must bind its exact hash."""
    gold_set = _two_document_gold_set()
    adjudicator = _FakeAdjudicator()

    arm_result, result_artifact, manifest = run_arm_gold_evidence(
        gold_set=gold_set, adjudicator=adjudicator, model_options={}, provider_identity="test-provider",
        manifest_output_dir=tmp_path, result_version_label="eval_result_gold_evidence",
        manifest_version_label="eval_manifest_gold_evidence",
    )

    assert len(adjudicator.call_log) == 4
    assert manifest.arm == "gold_evidence"
    assert manifest.gold_set_hash == gold_set.content_hash()
    assert manifest.execution_fingerprint  # a real, non-placeholder hash
    assert manifest.result_artifact_hash == result_artifact.content_hash()
    assert (tmp_path / "eval_manifest_gold_evidence.json").exists()
    assert (tmp_path / "eval_result_gold_evidence.json").exists()
    assert result_artifact.document_final_decision["d1"] == "include"
    assert result_artifact.usage_unit == "criterion_call"
    assert result_artifact.n_usage_available == 4  # billable per-call total, not the per-document-complete summary
    assert result_artifact.total_tokens == 4 * 120
    assert result_artifact.n_documents_usage_complete == 2  # per-document completeness kept as a separate figure


def test_run_arm_gold_evidence_survives_a_per_criterion_failure(tmp_path):
    gold_set = _two_document_gold_set()
    adjudicator = _FakeAdjudicator(fail_keys={("d2", "population")})
    arm_result, _, _ = run_arm_gold_evidence(
        gold_set=gold_set, adjudicator=adjudicator, model_options={}, provider_identity="test-provider",
        manifest_output_dir=tmp_path, result_version_label="eval_result", manifest_version_label="eval_manifest",
    )
    assert arm_result.outcomes_by_key[("d2", "population")] is None
    assert arm_result.outcomes_by_key[("d1", "population")] is not None


# ---------------------------------------------------------------------------
# run_arm_gold_evidence: checkpoint_dir (resume durability) and
# extra_config_summary_fields (source-identity recording), prompted by
# building the new prompt-6 Arm-1/Arm-2 runners.
# ---------------------------------------------------------------------------


class _KillAfterNAdjudicator:
    """Simulates an abruptly killed process: after `fail_after` successful
    calls, the NEXT call raises KeyboardInterrupt -- a BaseException, which
    run_arm_gold_evidence's `except Exception` around adjudicate() never
    catches (same discipline as LLMCriterionAdjudicator's own retry loop),
    so this reaches the caller exactly as a real kill would. Identity
    fields match _FakeAdjudicator's defaults so a resumed run's
    execution_fingerprint matches the interrupted run's."""

    model = "fake-1"
    prompt_template_version = "1"
    output_schema_version = "1"

    def __init__(self, *, fail_after: int) -> None:
        self.fail_after = fail_after
        self.call_count = 0

    def adjudicate(self, *, document, criterion, evidence, model_options):
        self.call_count += 1
        if self.call_count > self.fail_after:
            raise KeyboardInterrupt("simulated process kill")
        assessment = CriterionAssessment(
            criterion_id=criterion.criterion_id, criterion_version=criterion.version, status="satisfied",
            evidence=evidence, rationale="ok", requires_human_review=False,
            model=self.model, prompt_template_version=self.prompt_template_version, output_schema_version=self.output_schema_version,
        )
        usage = AdjudicationUsage(
            model=self.model, input_tokens=10, output_tokens=5, total_tokens=15,
            provider_call_count=1, retry_count=0, usage_available=True,
        )
        return AdjudicationOutcome(assessment=assessment, usage=usage)


def test_run_arm_gold_evidence_checkpoint_kill_then_resume_never_repeats_completed_calls(tmp_path):
    gold_set = _two_document_gold_set()  # 4 (document, criterion) calls total
    checkpoint_dir = tmp_path / "checkpoints"

    killer = _KillAfterNAdjudicator(fail_after=2)
    with pytest.raises(KeyboardInterrupt):
        run_arm_gold_evidence(
            gold_set=gold_set, adjudicator=killer, model_options={}, provider_identity="test-provider",
            manifest_output_dir=tmp_path, result_version_label="eval_result", manifest_version_label="eval_manifest",
            checkpoint_dir=checkpoint_dir,
        )
    assert killer.call_count == 3  # 2 succeeded, the 3rd is the one that raised
    assert len(list(checkpoint_dir.glob("checkpoint-*.json"))) == 2
    assert not (tmp_path / "eval_result.json").exists(), "no final artifact must be frozen on an interrupted run"

    resumer = _FakeAdjudicator()
    arm_result, result_artifact, manifest = run_arm_gold_evidence(
        gold_set=gold_set, adjudicator=resumer, model_options={}, provider_identity="test-provider",
        manifest_output_dir=tmp_path, result_version_label="eval_result", manifest_version_label="eval_manifest",
        checkpoint_dir=checkpoint_dir,
    )
    assert len(resumer.call_log) == 2  # only the 2 remaining calls -- the first 2 were never repeated
    assert len(arm_result.outcomes_by_key) == 4
    assert (tmp_path / "eval_result.json").exists()
    assert manifest.result_artifact_hash == result_artifact.content_hash()


def test_run_arm_gold_evidence_checkpoint_from_a_different_config_is_rejected(tmp_path):
    gold_set = _two_document_gold_set()
    checkpoint_dir = tmp_path / "checkpoints"
    run_arm_gold_evidence(
        gold_set=gold_set, adjudicator=_FakeAdjudicator(), model_options={}, provider_identity="test-provider",
        manifest_output_dir=tmp_path, result_version_label="eval_result", manifest_version_label="eval_manifest",
        checkpoint_dir=checkpoint_dir,
    )
    with pytest.raises(GoldEvidenceCheckpointDriftError):
        run_arm_gold_evidence(
            gold_set=gold_set, adjudicator=_FakeAdjudicator(), model_options={}, provider_identity="a-DIFFERENT-provider",
            manifest_output_dir=tmp_path, result_version_label="eval_result_2", manifest_version_label="eval_manifest_2",
            checkpoint_dir=checkpoint_dir,
        )


def test_run_arm_gold_evidence_extra_config_summary_fields_merge_into_manifest(tmp_path):
    gold_set = _two_document_gold_set()
    _, _, manifest = run_arm_gold_evidence(
        gold_set=gold_set, adjudicator=_FakeAdjudicator(), model_options={}, provider_identity="test-provider",
        manifest_output_dir=tmp_path, result_version_label="eval_result", manifest_version_label="eval_manifest",
        extra_config_summary_fields={"source_identity": {"git_commit": "deadbeef"}},
    )
    assert manifest.config_summary["source_identity"] == {"git_commit": "deadbeef"}
    assert manifest.config_summary["provider_identity"] == "test-provider"  # existing fields still present


def test_run_arm_gold_evidence_extra_config_summary_fields_collision_raises(tmp_path):
    gold_set = _two_document_gold_set()
    with pytest.raises(ValueError, match="collides"):
        run_arm_gold_evidence(
            gold_set=gold_set, adjudicator=_FakeAdjudicator(), model_options={}, provider_identity="test-provider",
            manifest_output_dir=tmp_path, result_version_label="eval_result", manifest_version_label="eval_manifest",
            extra_config_summary_fields={"provider_identity": "trying-to-overwrite"},
        )


# ---------------------------------------------------------------------------
# run_arm_e2e: resume=True (threading into ScreeningRequest.resume_run_id,
# reusing _run_ft_screening's own already-hardened resume mechanism) and
# extra_config_summary_fields.
# ---------------------------------------------------------------------------


def _e2e_make_pdf(tmp_path, name: str, content: bytes) -> Path:
    path = tmp_path / "pdfs" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _e2e_gold_set_and_pdfs(tmp_path):
    pdf1 = _e2e_make_pdf(tmp_path, "d1.pdf", b"doc one content")
    pdf2 = _e2e_make_pdf(tmp_path, "d2.pdf", b"doc two content")
    pdf_paths_by_document_id = {"d1": str(pdf1), "d2": str(pdf2)}
    document_pdf_sha256 = {"d1": hash_pdf_file(pdf1), "d2": hash_pdf_file(pdf2)}

    criterion_annotations = [
        make_criterion_annotation(document_id="d1", criterion_id="population"),
        make_criterion_annotation(document_id="d1", criterion_id="design", supporting_pages=(5,), supporting_passages=("RCT design",)),
        make_criterion_annotation(document_id="d2", criterion_id="population", gold_status="violated", supporting_pages=(), supporting_passages=(), contradicting_pages=(2,), contradicting_passages=("children only",), human_review_required=True),
        make_criterion_annotation(document_id="d2", criterion_id="design", gold_status="not_reported", evidence_absent=True, supporting_pages=(), supporting_passages=()),
    ]
    document_annotations = [
        make_document_annotation(document_id="d1", gold_final_decision="include"),
        make_document_annotation(document_id="d2", gold_final_decision="exclude", gold_primary_exclusion_reason_code="wrong_population"),
    ]
    gold_set = make_gold_set(criterion_annotations, document_annotations, document_pdf_sha256=document_pdf_sha256)
    return gold_set, pdf_paths_by_document_id


def _e2e_config(**overrides) -> ScreeningConfig:
    defaults = dict(
        parser_name="fake", parser_version="1", chunker_version="1",
        retriever_name="fake", retriever_config_version="1",
        adjudicator_provider="fake", adjudicator_model="fake-1", model_config_id="cfg-1",
        prompt_template_version="1", output_schema_version="1", aggregation_policy_version="1",
        cache_mode="off", item_error_policy="fail_fast", max_workers=1,
    )
    defaults.update(overrides)
    return ScreeningConfig(**defaults)


def test_run_arm_e2e_resume_false_is_the_exact_prior_default_behavior(tmp_path):
    """resume=False (the default, omitted by every existing caller) must
    behave EXACTLY as run_arm_e2e did before this parameter existed --
    proven by a plain, uninterrupted, complete run."""
    gold_set, pdf_paths = _e2e_gold_set_and_pdfs(tmp_path)
    output_dir = tmp_path / "outputs"
    components = Components(
        parser=FakeDocumentParser(), retriever=FakeEvidenceRetriever(),
        adjudicator=FakeCriterionAdjudicator(), store=FileScreeningStore(output_dir=output_dir),
    )
    result, result_artifact, manifest = run_arm_e2e(
        gold_set=gold_set, pdf_paths_by_document_id=pdf_paths, components=components, config=_e2e_config(),
        output_dir=output_dir, run_id="e2e-resume-false-test", result_version_label="eval_result", manifest_version_label="eval_manifest",
    )
    assert result.run_status == "completed"
    assert result_artifact.missing_or_failed_document_ids == ()


def test_run_arm_e2e_resume_true_completes_only_unfinished_items(tmp_path):
    """Mirrors tests/contract/test_contract.py's own
    test_7_resume_completes_only_unfinished_items, but through
    run_arm_e2e()'s call surface -- proves run_arm_e2e(resume=True)
    correctly threads resume_run_id into the SAME already-hardened
    orchestrator resume mechanism that test already proves, rather than
    bypassing it."""
    gold_set, pdf_paths = _e2e_gold_set_and_pdfs(tmp_path)
    output_dir = tmp_path / "outputs"
    store = FileScreeningStore(output_dir=output_dir)

    failing_components = Components(
        parser=FakeDocumentParser(), retriever=FakeEvidenceRetriever(),
        adjudicator=FakeCriterionAdjudicator(fail_keys={("d1", "population")}), store=store,
    )
    result1, _, _ = run_arm_e2e(
        gold_set=gold_set, pdf_paths_by_document_id=pdf_paths, components=failing_components, config=_e2e_config(),
        output_dir=output_dir, run_id="e2e-resume-true-test", result_version_label="eval_result", manifest_version_label="eval_manifest",
        resume=False,
    )
    assert [i.execution_status for i in result1.items] != ["completed", "completed"]

    fixed_components = Components(
        parser=FakeDocumentParser(), retriever=FakeEvidenceRetriever(),
        adjudicator=FakeCriterionAdjudicator(), store=store,
    )
    result2, result_artifact2, manifest2 = run_arm_e2e(
        gold_set=gold_set, pdf_paths_by_document_id=pdf_paths, components=fixed_components, config=_e2e_config(),
        output_dir=output_dir, run_id="e2e-resume-true-test", result_version_label="eval_result_resumed",
        manifest_version_label="eval_manifest_resumed", resume=True,
    )
    assert [i.execution_status for i in result2.items] == ["completed", "completed"]
    assert result2.run_status == "completed"
    assert result_artifact2.missing_or_failed_document_ids == ()
    assert manifest2.result_artifact_hash == result_artifact2.content_hash()


def test_run_arm_e2e_extra_config_summary_fields_merge_and_collision(tmp_path):
    gold_set, pdf_paths = _e2e_gold_set_and_pdfs(tmp_path)
    output_dir = tmp_path / "outputs"
    components = Components(
        parser=FakeDocumentParser(), retriever=FakeEvidenceRetriever(),
        adjudicator=FakeCriterionAdjudicator(), store=FileScreeningStore(output_dir=output_dir),
    )
    _, _, manifest = run_arm_e2e(
        gold_set=gold_set, pdf_paths_by_document_id=pdf_paths, components=components, config=_e2e_config(),
        output_dir=output_dir, run_id="e2e-extra-fields-test", result_version_label="eval_result", manifest_version_label="eval_manifest",
        extra_config_summary_fields={"source_identity": {"git_commit": "deadbeef"}},
    )
    assert manifest.config_summary["source_identity"] == {"git_commit": "deadbeef"}
    assert manifest.config_summary["adjudicator_model"] == "fake-1"  # existing fields still present

    with pytest.raises(ValueError, match="collides"):
        run_arm_e2e(
            gold_set=gold_set, pdf_paths_by_document_id=pdf_paths, components=components, config=_e2e_config(),
            output_dir=output_dir, run_id="e2e-extra-fields-collision-test", result_version_label="eval_result_2",
            manifest_version_label="eval_manifest_2", extra_config_summary_fields={"adjudicator_model": "trying-to-overwrite"},
        )


def test_compute_arm_gold_evidence_document_decisions_uses_real_aggregation_policy(tmp_path):
    gold_set = _two_document_gold_set()
    adjudicator = _FakeAdjudicator(
        status_by_key={
            ("d1", "population"): "satisfied", ("d1", "design"): "satisfied",
            ("d2", "population"): "violated", ("d2", "design"): "not_reported",
        },
    )
    arm_result, _, _ = run_arm_gold_evidence(
        gold_set=gold_set, adjudicator=adjudicator, model_options={}, provider_identity="test-provider",
        manifest_output_dir=tmp_path, result_version_label="eval_result", manifest_version_label="eval_manifest",
    )
    decisions = compute_arm_gold_evidence_document_decisions(gold_set=gold_set, arm_result=arm_result)
    assert decisions["d1"].final_decision == "include"
    assert decisions["d2"].final_decision == "exclude"


# ---------------------------------------------------------------------------
# _document_sha256_for_decision -- the fix for the real incident where a
# caller (Arm 2b) supplied evidence bound to a REAL PDF hash while this
# function used to hardcode "0"*64 unconditionally, crashing
# aggregation.build_eligibility_result()'s own evidence-identity check.
# ---------------------------------------------------------------------------


def _assessment_with_evidence(*, document_sha256s: list[str], criterion_id: str = "c1") -> CriterionAssessment:
    evidence = [
        EvidenceSpan(
            document_id="d1", document_sha256=sha, chunk_id=f"chunk-{i}", excerpt="text", stance="relevant",
            retrieval_method="test", retrieval_query_version="v1",
        )
        for i, sha in enumerate(document_sha256s)
    ]
    return CriterionAssessment(
        criterion_id=criterion_id, criterion_version="1", status="satisfied", evidence=evidence, rationale="r",
        requires_human_review=False, model="m", prompt_template_version="1", output_schema_version="1",
    )


def test_document_sha256_for_decision_derives_from_evidence_real_hash():
    """The real-incident case: evidence bound to a genuine PDF hash (as
    Arm 2b's redacted evidence spans carry) must be used AS-IS, not
    overridden by a hardcoded placeholder."""
    gold_set = _two_document_gold_set()
    real_hash = "f" * 64
    assessments = [_assessment_with_evidence(document_sha256s=[real_hash])]
    assert _document_sha256_for_decision(doc_id="d1", assessments=assessments, gold_set=gold_set) == real_hash


def test_document_sha256_for_decision_derives_placeholder_when_that_is_what_evidence_carries():
    """Backward compatibility: Arm 2/_gold_evidence_spans's own evidence
    is uniformly bound to "0"*64 -- this must still be derived correctly
    (not just tolerated), since existing gold_evidence arms' regression
    tests depend on it."""
    gold_set = _two_document_gold_set()
    placeholder = "0" * 64
    assessments = [_assessment_with_evidence(document_sha256s=[placeholder, placeholder])]
    assert _document_sha256_for_decision(doc_id="d1", assessments=assessments, gold_set=gold_set) == placeholder


def test_document_sha256_for_decision_falls_back_to_gold_set_pdf_hash_when_no_evidence():
    gold_set = _two_document_gold_set()
    assessments = [_assessment_with_evidence(document_sha256s=[])]  # zero evidence spans
    expected = gold_set.document_pdf_sha256["d1"]
    assert _document_sha256_for_decision(doc_id="d1", assessments=assessments, gold_set=gold_set) == expected


def test_document_sha256_for_decision_rejects_inconsistent_hashes():
    gold_set = _two_document_gold_set()
    assessments = [
        _assessment_with_evidence(document_sha256s=["a" * 64], criterion_id="c1"),
        _assessment_with_evidence(document_sha256s=["b" * 64], criterion_id="c2"),
    ]
    with pytest.raises(ValueError, match="multiple different document_sha256"):
        _document_sha256_for_decision(doc_id="d1", assessments=assessments, gold_set=gold_set)


def test_compute_arm_gold_evidence_document_decisions_accepts_real_document_sha256_evidence():
    """End-to-end reproduction of the actual Arm 2b incident: an
    ArmGoldEvidenceResult whose evidence is bound to the real GoldSet PDF
    hash (not "0"*64) must be scored successfully, not raise
    AggregationError."""
    from ft_screening.components import AdjudicationOutcome, AdjudicationUsage
    from ft_screening.evaluation_harness import ArmGoldEvidenceResult

    gold_set = _two_document_gold_set()
    real_hash_d1 = gold_set.document_pdf_sha256["d1"]
    real_hash_d2 = gold_set.document_pdf_sha256["d2"]

    def _outcome(document_id, document_sha256, criterion_id, status):
        assessment = CriterionAssessment(
            criterion_id=criterion_id, criterion_version="1", status=status,
            evidence=[EvidenceSpan(
                document_id=document_id, document_sha256=document_sha256,
                chunk_id=f"{criterion_id}-0", excerpt="text", stance="relevant", retrieval_method="test", retrieval_query_version="v1",
            )],
            rationale="r", requires_human_review=False, model="m", prompt_template_version="1", output_schema_version="1",
        )
        usage = AdjudicationUsage(model="m", input_tokens=1, output_tokens=1, total_tokens=2, provider_call_count=1, retry_count=0, usage_available=True)
        return AdjudicationOutcome(assessment=assessment, usage=usage)

    arm_result = ArmGoldEvidenceResult(
        outcomes_by_key={
            ("d1", "population"): _outcome("d1", real_hash_d1, "population", "satisfied"),
            ("d1", "design"): _outcome("d1", real_hash_d1, "design", "satisfied"),
            ("d2", "population"): _outcome("d2", real_hash_d2, "population", "violated"),
            ("d2", "design"): _outcome("d2", real_hash_d2, "design", "not_reported"),
        },
        wall_seconds_by_document={},
    )
    decisions = compute_arm_gold_evidence_document_decisions(gold_set=gold_set, arm_result=arm_result)
    assert decisions["d1"].final_decision == "include"
    assert decisions["d2"].final_decision == "exclude"


def test_token_usage_summary_gold_evidence_per_call_vs_per_document(tmp_path):
    """codex_code_review_dca0970.md finding 5: the per-call and
    per-document summaries must report different, correctly-labeled units."""
    gold_set = _two_document_gold_set()
    adjudicator = _FakeAdjudicator()
    arm_result, _, _ = run_arm_gold_evidence(
        gold_set=gold_set, adjudicator=adjudicator, model_options={}, provider_identity="test-provider",
        manifest_output_dir=tmp_path, result_version_label="eval_result", manifest_version_label="eval_manifest",
    )
    per_call = token_usage_summary_gold_evidence(gold_set=gold_set, arm_result=arm_result)
    per_doc = token_usage_summary_gold_evidence_per_document(gold_set=gold_set, arm_result=arm_result)

    assert per_call.unit == "criterion_call"
    assert per_call.n_usage_available == 4  # 4 criterion calls
    assert per_call.total_input_tokens == 400

    assert per_doc.unit == "document"
    assert per_doc.n_usage_available == 2  # 2 documents
    assert per_doc.total_input_tokens == 400  # same grand total, different n/mean
    assert per_doc.mean_input_tokens == 200.0  # 2 criteria x 100 tokens per document


def test_token_usage_summary_gold_evidence_counts_a_failed_criterion_as_unavailable(tmp_path):
    gold_set = _two_document_gold_set()
    adjudicator = _FakeAdjudicator(fail_keys={("d2", "population")})
    arm_result, _, _ = run_arm_gold_evidence(
        gold_set=gold_set, adjudicator=adjudicator, model_options={}, provider_identity="test-provider",
        manifest_output_dir=tmp_path, result_version_label="eval_result", manifest_version_label="eval_manifest",
    )
    per_call = token_usage_summary_gold_evidence(gold_set=gold_set, arm_result=arm_result)
    assert per_call.n_usage_unavailable == 1
    assert per_call.n_usage_available == 3


def test_wall_time_summary_gold_evidence_has_no_parse_or_retrieve_cost(tmp_path):
    gold_set = _two_document_gold_set()
    adjudicator = _FakeAdjudicator()
    arm_result, _, _ = run_arm_gold_evidence(
        gold_set=gold_set, adjudicator=adjudicator, model_options={}, provider_identity="test-provider",
        manifest_output_dir=tmp_path, result_version_label="eval_result", manifest_version_label="eval_manifest",
    )
    summary = wall_time_summary_gold_evidence(arm_result)
    assert summary.n_documents == 2


# ---------------------------------------------------------------------------
# EvaluationResultArtifact's own validators, and the billable-vs-completeness
# usage split (codex_code_review_1c40a27.md finding 4)
# ---------------------------------------------------------------------------


def _valid_result_artifact_kwargs(**overrides) -> dict:
    kwargs = dict(
        arm="e2e", gold_set_hash="a" * 64,
        criterion_status_by_key={"d1::population": "satisfied"},
        document_final_decision={"d1": "include"},
        document_primary_exclusion_reason={"d1": None},
        missing_or_failed_document_ids=(), missing_or_failed_criterion_keys=(),
        usage_unit="document", total_input_tokens=100, total_output_tokens=20, total_tokens=120,
        n_usage_available=1, n_usage_unavailable=0,
        n_documents_usage_complete=1, n_documents_usage_incomplete=0,
        total_wall_seconds=1.0, frozen_at="2026-08-16T00:00:00+00:00",
    )
    kwargs.update(overrides)
    return kwargs


def test_evaluation_result_artifact_baseline_is_valid():
    EvaluationResultArtifact(**_valid_result_artifact_kwargs())


def test_evaluation_result_artifact_rejects_token_arithmetic_mismatch():
    """Exact repro: total_input_tokens=10, total_output_tokens=20,
    total_tokens=999 was previously accepted outright."""
    with pytest.raises(Exception):
        EvaluationResultArtifact(**_valid_result_artifact_kwargs(total_input_tokens=10, total_output_tokens=20, total_tokens=999))


def test_evaluation_result_artifact_rejects_invalid_status_value():
    with pytest.raises(Exception):
        EvaluationResultArtifact(**_valid_result_artifact_kwargs(criterion_status_by_key={"d1::population": "bogus"}))


def test_evaluation_result_artifact_rejects_invalid_decision_value():
    with pytest.raises(Exception):
        EvaluationResultArtifact(**_valid_result_artifact_kwargs(document_final_decision={"d1": "bogus"}))


def test_evaluation_result_artifact_rejects_mismatched_exclusion_reason_keys():
    with pytest.raises(Exception):
        EvaluationResultArtifact(**_valid_result_artifact_kwargs(document_primary_exclusion_reason={}))


def test_evaluation_result_artifact_rejects_a_reason_without_an_exclude_decision():
    with pytest.raises(Exception):
        EvaluationResultArtifact(**_valid_result_artifact_kwargs(document_primary_exclusion_reason={"d1": "wrong_population"}))


def test_evaluation_result_artifact_rejects_a_non_sha_gold_set_hash():
    """Exact repro: a non-SHA gold hash was previously accepted outright."""
    with pytest.raises(Exception):
        EvaluationResultArtifact(**_valid_result_artifact_kwargs(gold_set_hash="not-a-sha"))


def test_evaluation_result_artifact_rejects_an_unparseable_frozen_at():
    with pytest.raises(Exception):
        EvaluationResultArtifact(**_valid_result_artifact_kwargs(frozen_at="not-a-time"))


def test_evaluation_result_artifact_rejects_zero_availability_counts_not_covering_the_unit():
    """Exact repro: zero availability counts were previously accepted
    even though they don't account for the single document actually
    scored."""
    with pytest.raises(Exception):
        EvaluationResultArtifact(**_valid_result_artifact_kwargs(
            n_usage_available=0, n_usage_unavailable=0,
            total_input_tokens=0, total_output_tokens=0, total_tokens=0,
        ))


def test_evaluation_result_artifact_rejects_a_missing_key_partition_mismatch():
    with pytest.raises(Exception):
        EvaluationResultArtifact(**_valid_result_artifact_kwargs(missing_or_failed_document_ids=("d1",)))


def test_run_arm_gold_evidence_billable_total_survives_a_partial_document_failure(tmp_path):
    """Exact repro: three successful calls at 120 tokens each and one
    failed call. The persisted total must be 360 (every successful,
    billed call), not 240 -- the old per-document-COMPLETE summary
    dropped ALL of d2's tokens, including its one successful call,
    because d2's OTHER criterion failed."""
    gold_set = _two_document_gold_set()
    adjudicator = _FakeAdjudicator(fail_keys={("d2", "population")})
    _, result_artifact, _ = run_arm_gold_evidence(
        gold_set=gold_set, adjudicator=adjudicator, model_options={}, provider_identity="test-provider",
        manifest_output_dir=tmp_path, result_version_label="eval_result", manifest_version_label="eval_manifest",
    )
    assert result_artifact.usage_unit == "criterion_call"
    assert result_artifact.n_usage_available == 3
    assert result_artifact.total_tokens == 3 * 120
    assert result_artifact.n_documents_usage_complete == 1  # only d1 has every criterion available
    assert result_artifact.n_documents_usage_incomplete == 1  # d2 has one missing call


def test_run_arm_gold_evidence_rejects_a_blank_provider_identity(tmp_path):
    gold_set = _two_document_gold_set()
    adjudicator = _FakeAdjudicator()
    with pytest.raises(ValueError):
        run_arm_gold_evidence(
            gold_set=gold_set, adjudicator=adjudicator, model_options={}, provider_identity="   ",
            manifest_output_dir=tmp_path, result_version_label="eval_result", manifest_version_label="eval_manifest",
        )


def test_load_frozen_evaluation_result_artifact_verifies_against_gold_set(tmp_path):
    gold_set = _two_document_gold_set()
    adjudicator = _FakeAdjudicator()
    _, result_artifact, _ = run_arm_gold_evidence(
        gold_set=gold_set, adjudicator=adjudicator, model_options={}, provider_identity="test-provider",
        manifest_output_dir=tmp_path, result_version_label="eval_result", manifest_version_label="eval_manifest",
    )
    loaded = load_frozen_evaluation_result_artifact(tmp_path / "eval_result.json", gold_set=gold_set)
    assert loaded.content_hash() == result_artifact.content_hash()

    other_gold_set = make_gold_set(
        [make_criterion_annotation(document_id="dX", criterion_id="population", document_page_count=10),
         make_criterion_annotation(document_id="dX", criterion_id="design", document_page_count=10, supporting_pages=(1,), supporting_passages=("x",))],
        [make_document_annotation(document_id="dX", document_page_count=10)],
    )
    with pytest.raises(ValueError):
        load_frozen_evaluation_result_artifact(tmp_path / "eval_result.json", gold_set=other_gold_set)


def test_verify_result_artifact_against_gold_set_rejects_a_hash_mismatch():
    gold_set = _two_document_gold_set()
    result = EvaluationResultArtifact(**_valid_result_artifact_kwargs(
        gold_set_hash="0" * 64,
        criterion_status_by_key={"d1::population": "satisfied", "d1::design": "satisfied", "d2::population": "violated", "d2::design": "not_reported"},
        document_final_decision={"d1": "include", "d2": "exclude"},
        document_primary_exclusion_reason={"d1": None, "d2": "wrong_population"},
        n_usage_available=2, n_usage_unavailable=0, n_documents_usage_complete=2, n_documents_usage_incomplete=0,
    ))
    with pytest.raises(ValueError):
        verify_result_artifact_against_gold_set(result, gold_set)


# ---------------------------------------------------------------------------
# Arm 3: retrieval-oracle metrics, requiring exact cross-binding
# ---------------------------------------------------------------------------


def _artifact_and_judgments(gold_set, *, sufficient_keys=frozenset(), document_sha256=None, criteria_set_hash=None):
    evidence_by_key = {
        make_evidence_key("d1", "population"): [RetrievalOracleSpan(chunk_id="c1", page=3, excerpt="adults")],
        make_evidence_key("d1", "design"): [RetrievalOracleSpan(chunk_id="c2", page=5, excerpt="RCT")],
        make_evidence_key("d2", "population"): [RetrievalOracleSpan(chunk_id="c3", page=99, excerpt="unrelated")],
        make_evidence_key("d2", "design"): [],
    }
    artifact = RetrievalOracleArtifact(
        parser_name="docling", parser_version="2.61.2", parser_options_hash="e" * 64,
        retriever_name="lexical_bm25", retriever_config_version="1", retriever_options_hash="a" * 64,
        criteria_set_hash=criteria_set_hash or compute_criteria_set_hash(gold_set.criteria_set),
        document_sha256=document_sha256 or dict(gold_set.document_pdf_sha256), evidence_by_key=evidence_by_key,
        frozen_at="2026-08-15T00:00:00+00:00",
    )
    judgments = [
        RetrievalSufficiencyJudgment(
            document_id=doc, criterion_id=crit, retrieval_sufficient=(doc, crit) in sufficient_keys, annotator_id="RR",
        )
        for doc in ("d1", "d2") for crit in ("population", "design")
    ]
    judgment_set = RetrievalOracleJudgmentSet(
        gold_set_hash=gold_set.content_hash(), retrieval_artifact_hash=artifact.content_hash(),
        judgments=tuple(judgments), frozen_at="2026-08-15T00:00:00+00:00",
    )
    return artifact, judgment_set


def test_retrieval_oracle_metrics_mechanical_recall_hit_and_miss():
    gold_set = _two_document_gold_set()
    artifact, judgment_set = _artifact_and_judgments(gold_set, sufficient_keys={("d1", "population"), ("d1", "design")})
    report = compute_retrieval_oracle_metrics(gold_set=gold_set, retrieval_artifact=artifact, judgment_set=judgment_set)
    assert report.n_criteria_with_evidence == 3
    assert report.mechanical_evidence_recall == 2 / 3


def test_retrieval_oracle_metrics_rejects_a_document_sha256_mismatch():
    """codex_code_review_dca0970.md finding 3's exact repro: an artifact
    with deliberately wrong document hashes previously scored successfully."""
    gold_set = _two_document_gold_set()
    artifact, judgment_set = _artifact_and_judgments(gold_set, document_sha256={"d1": "0" * 64, "d2": "1" * 64})
    with pytest.raises(ValueError):
        compute_retrieval_oracle_metrics(gold_set=gold_set, retrieval_artifact=artifact, judgment_set=judgment_set)


def test_retrieval_oracle_metrics_rejects_a_criteria_set_hash_mismatch():
    gold_set = _two_document_gold_set()
    artifact, judgment_set = _artifact_and_judgments(gold_set, criteria_set_hash="f" * 64)
    with pytest.raises(ValueError):
        compute_retrieval_oracle_metrics(gold_set=gold_set, retrieval_artifact=artifact, judgment_set=judgment_set)


def test_retrieval_oracle_metrics_rejects_a_judgment_set_bound_to_a_different_gold_set():
    gold_set = _two_document_gold_set()
    other_gold_set = make_gold_set(
        [make_criterion_annotation(document_id="dX", criterion_id="population", document_page_count=10),
         make_criterion_annotation(document_id="dX", criterion_id="design", document_page_count=10, supporting_pages=(1,), supporting_passages=("x",))],
        [make_document_annotation(document_id="dX", document_page_count=10)],
    )
    artifact, judgment_set = _artifact_and_judgments(other_gold_set)
    with pytest.raises(ValueError):
        compute_retrieval_oracle_metrics(gold_set=gold_set, retrieval_artifact=artifact, judgment_set=judgment_set)
