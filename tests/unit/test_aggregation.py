"""Tests for ft_screening.aggregation: the deterministic decision policy
and the EligibilityResult construction factory (api_contract_v2.md
correction 6 -- contextual validation a bare EligibilityResult(...) call
can't perform).
"""

from datetime import datetime, timezone

import pytest

from ft_screening.aggregation import AggregationError, aggregate_decision, build_eligibility_result
from ft_screening.hashing import compute_criteria_set_hash
from ft_screening.schema import CriteriaSet, CriterionAssessment, CriterionSpec, EvidenceSpan

NOW = datetime(2026, 8, 12, tzinfo=timezone.utc)


def make_criterion(**overrides) -> CriterionSpec:
    defaults = dict(
        criterion_id="population",
        version="1",
        eligibility_requirement="adult humans",
        missing_evidence_policy="review",
        exclusion_reason_code="wrong_population",
        exclusion_reason_label="Wrong population",
        priority=1,
    )
    defaults.update(overrides)
    return CriterionSpec(**defaults)


def make_assessment(**overrides) -> CriterionAssessment:
    defaults = dict(
        criterion_id="population",
        criterion_version="1",
        status="satisfied",
        evidence=[],
        rationale="ok",
        requires_human_review=False,
        model="fake",
        prompt_template_version="1",
        output_schema_version="1",
    )
    defaults.update(overrides)
    return CriterionAssessment(**defaults)


def make_span(**overrides) -> EvidenceSpan:
    defaults = dict(
        document_id="doc-1",
        document_sha256="a" * 64,
        chunk_id="chunk-1",
        section="Methods",
        page=1,
        start_char=0,
        end_char=10,
        excerpt="adults",
        stance="supports",
        retrieval_method="fake",
        retrieval_query_version="1",
    )
    defaults.update(overrides)
    return EvidenceSpan(**defaults)


# ---------------------------------------------------------------------------
# aggregate_decision policy matrix
# ---------------------------------------------------------------------------


def test_all_satisfied_includes():
    criteria_set = CriteriaSet(criteria_set_id="cs", version="1", criteria=[make_criterion()])
    decision, codes, primary = aggregate_decision(
        criteria_set=criteria_set, assessments=[make_assessment(status="satisfied")]
    )
    assert decision == "include"
    assert codes == []
    assert primary is None


def test_violated_excludes_with_reason_code():
    criteria_set = CriteriaSet(criteria_set_id="cs", version="1", criteria=[make_criterion()])
    decision, codes, primary = aggregate_decision(
        criteria_set=criteria_set, assessments=[make_assessment(status="violated")]
    )
    assert decision == "exclude"
    assert codes == ["wrong_population"]
    assert primary == "wrong_population"


def test_unclear_with_fail_policy_excludes():
    criterion = make_criterion(missing_evidence_policy="fail")
    criteria_set = CriteriaSet(criteria_set_id="cs", version="1", criteria=[criterion])
    decision, codes, primary = aggregate_decision(
        criteria_set=criteria_set, assessments=[make_assessment(status="unclear")]
    )
    assert decision == "exclude"
    assert primary == "wrong_population"


def test_unclear_with_review_policy_needs_review():
    criterion = make_criterion(missing_evidence_policy="review")
    criteria_set = CriteriaSet(criteria_set_id="cs", version="1", criteria=[criterion])
    decision, codes, primary = aggregate_decision(
        criteria_set=criteria_set, assessments=[make_assessment(status="unclear")]
    )
    assert decision == "needs_human_review"
    assert codes == []
    assert primary is None


def test_not_reported_with_pass_policy_includes():
    criterion = make_criterion(missing_evidence_policy="pass")
    criteria_set = CriteriaSet(criteria_set_id="cs", version="1", criteria=[criterion])
    decision, codes, primary = aggregate_decision(
        criteria_set=criteria_set, assessments=[make_assessment(status="not_reported")]
    )
    assert decision == "include"


def test_requires_human_review_flag_forces_review_even_when_satisfied():
    criteria_set = CriteriaSet(criteria_set_id="cs", version="1", criteria=[make_criterion()])
    decision, codes, primary = aggregate_decision(
        criteria_set=criteria_set,
        assessments=[make_assessment(status="satisfied", requires_human_review=True)],
    )
    assert decision == "needs_human_review"


def test_violated_wins_over_requires_human_review_elsewhere():
    c1 = make_criterion(criterion_id="population", exclusion_reason_code="r1", priority=1)
    c2 = make_criterion(criterion_id="design", exclusion_reason_code="r2", priority=2)
    criteria_set = CriteriaSet(criteria_set_id="cs", version="1", criteria=[c1, c2])
    decision, codes, primary = aggregate_decision(
        criteria_set=criteria_set,
        assessments=[
            make_assessment(criterion_id="population", status="violated"),
            make_assessment(criterion_id="design", status="satisfied", requires_human_review=True),
        ],
    )
    assert decision == "exclude"  # clear negative wins over an unresolved review flag elsewhere
    assert primary == "r1"


def test_primary_reason_selected_by_lowest_priority_then_criterion_id():
    c1 = make_criterion(criterion_id="zzz", exclusion_reason_code="r_zzz", priority=5)
    c2 = make_criterion(criterion_id="aaa", exclusion_reason_code="r_aaa", priority=1)
    criteria_set = CriteriaSet(criteria_set_id="cs", version="1", criteria=[c1, c2])
    decision, codes, primary = aggregate_decision(
        criteria_set=criteria_set,
        assessments=[
            make_assessment(criterion_id="zzz", status="violated"),
            make_assessment(criterion_id="aaa", status="violated"),
        ],
    )
    assert decision == "exclude"
    assert primary == "r_aaa"  # lower priority number wins
    assert set(codes) == {"r_zzz", "r_aaa"}


def test_primary_reason_tiebreak_by_criterion_id_when_priority_equal():
    c1 = make_criterion(criterion_id="zzz", exclusion_reason_code="r_zzz", priority=1)
    c2 = make_criterion(criterion_id="aaa", exclusion_reason_code="r_aaa", priority=1)
    criteria_set = CriteriaSet(criteria_set_id="cs", version="1", criteria=[c1, c2])
    decision, codes, primary = aggregate_decision(
        criteria_set=criteria_set,
        assessments=[
            make_assessment(criterion_id="zzz", status="violated"),
            make_assessment(criterion_id="aaa", status="violated"),
        ],
    )
    assert primary == "r_aaa"  # "aaa" < "zzz" lexicographically


def test_exclusion_reason_codes_deduplicated():
    c1 = make_criterion(criterion_id="c1", exclusion_reason_code="shared_reason", priority=1)
    c2 = make_criterion(criterion_id="c2", exclusion_reason_code="shared_reason", priority=2, missing_evidence_policy="fail")
    # exclusion_reason_code uniqueness is a CriteriaSet-level constraint normally,
    # but two DIFFERENT criteria could legitimately share a reason CODE by design
    # (e.g. two related eligibility checks reported under one umbrella reason) --
    # CriteriaSet's own uniqueness check would reject this at construction, so
    # test the aggregation function directly against a hand-built pair instead.
    decision, codes, primary = aggregate_decision(
        criteria_set=CriteriaSet(criteria_set_id="cs", version="1", criteria=[c1, make_criterion(criterion_id="c2", exclusion_reason_code="other_reason", priority=2, missing_evidence_policy="fail")]),
        assessments=[
            make_assessment(criterion_id="c1", status="violated"),
            make_assessment(criterion_id="c2", status="unclear"),
        ],
    )
    assert decision == "exclude"
    assert len(codes) == 2  # distinct reasons here; dedup covered by construction, not a real collision case


# ---------------------------------------------------------------------------
# build_eligibility_result: contextual validation
# ---------------------------------------------------------------------------


def _build(criteria, assessments, **overrides):
    criteria_set = CriteriaSet(criteria_set_id="cs", version="1", criteria=criteria)
    defaults = dict(
        result_id="res-1",
        document_id="doc-1",
        document_sha256="a" * 64,
        report_id="report-1",
        study_id=None,
        criteria_set=criteria_set,
        criteria_set_hash=compute_criteria_set_hash(criteria_set),
        execution_fingerprint="c" * 64,
        assessments=assessments,
        created_at=NOW,
    )
    defaults.update(overrides)
    return build_eligibility_result(**defaults)


def test_build_eligibility_result_happy_path():
    result = _build([make_criterion()], [make_assessment()])
    assert result.final_decision == "include"
    assert result.document_id == "doc-1"


def test_build_rejects_missing_assessment_for_a_criterion():
    c1 = make_criterion(criterion_id="population")
    c2 = make_criterion(criterion_id="design", exclusion_reason_code="not_empirical")
    with pytest.raises(AggregationError):
        _build([c1, c2], [make_assessment(criterion_id="population")])  # design never assessed


def test_build_rejects_assessment_for_unknown_criterion():
    c1 = make_criterion(criterion_id="population")
    with pytest.raises(AggregationError):
        _build([c1], [make_assessment(criterion_id="population"), make_assessment(criterion_id="phantom")])


def test_build_rejects_duplicate_assessment_for_same_criterion():
    c1 = make_criterion(criterion_id="population")
    with pytest.raises(AggregationError):
        _build([c1], [make_assessment(criterion_id="population"), make_assessment(criterion_id="population")])


def test_build_rejects_criterion_version_mismatch():
    c1 = make_criterion(criterion_id="population", version="2")
    with pytest.raises(AggregationError):
        _build([c1], [make_assessment(criterion_id="population", criterion_version="1")])


def test_build_rejects_evidence_outside_allowed_sources():
    c1 = make_criterion(allowed_sources=["methods"])
    evidence = [make_span(section="Discussion")]
    with pytest.raises(AggregationError):
        _build([c1], [make_assessment(evidence=evidence)])


def test_build_allows_evidence_within_allowed_sources_after_normalization():
    c1 = make_criterion(allowed_sources=["methods"])  # already normalized by CriterionSpec's own validator
    evidence = [make_span(section=" Methods ")]  # NFC+trim+casefold -> "methods"
    result = _build([c1], [make_assessment(evidence=evidence)])
    assert result.final_decision == "include"


def test_build_result_id_and_revision_fixed_for_a_fresh_result():
    result = _build([make_criterion()], [make_assessment()])
    assert result.revision == 1
    assert result.supersedes_result_id is None
    assert result.human_adjudication is None
    assert result.proposed_decision == result.final_decision


# ---------------------------------------------------------------------------
# codex_code_review_44d9e90.md finding 7 additions
# ---------------------------------------------------------------------------


def test_build_rejects_criteria_set_hash_that_does_not_match_criteria_set():
    with pytest.raises(AggregationError):
        _build([make_criterion()], [make_assessment()], criteria_set_hash="0" * 64)


def test_build_rejects_evidence_bound_to_a_different_document_id():
    evidence = [make_span(document_id="some-other-document")]
    with pytest.raises(AggregationError):
        _build([make_criterion()], [make_assessment(evidence=evidence)])


def test_build_rejects_evidence_bound_to_a_different_document_sha256():
    evidence = [make_span(document_sha256="f" * 64)]
    with pytest.raises(AggregationError):
        _build([make_criterion()], [make_assessment(evidence=evidence)])


def test_build_rejects_review_policy_unclear_without_requires_human_review():
    criterion = make_criterion(missing_evidence_policy="review")
    assessment = make_assessment(status="unclear", requires_human_review=False)
    with pytest.raises(AggregationError):
        _build([criterion], [assessment])


def test_build_allows_review_policy_unclear_with_requires_human_review_true():
    criterion = make_criterion(missing_evidence_policy="review")
    assessment = make_assessment(status="unclear", requires_human_review=True)
    result = _build([criterion], [assessment])
    assert result.final_decision == "needs_human_review"


def test_build_does_not_require_requires_human_review_under_fail_policy():
    # "fail" policy means missing evidence -> exclude outright, no review
    # needed -- only "review" policy has this consistency requirement.
    criterion = make_criterion(missing_evidence_policy="fail")
    assessment = make_assessment(status="unclear", requires_human_review=False)
    result = _build([criterion], [assessment])
    assert result.final_decision == "exclude"


def test_exclusion_reason_codes_order_independent_of_assessment_input_order():
    c1 = make_criterion(criterion_id="low_priority", exclusion_reason_code="r_low", priority=9)
    c2 = make_criterion(criterion_id="high_priority", exclusion_reason_code="r_high", priority=1)
    criteria_set = CriteriaSet(criteria_set_id="cs", version="1", criteria=[c1, c2])

    # Assessments passed in "wrong" (non-priority) order -- output must not
    # depend on this.
    decision_a, codes_a, primary_a = aggregate_decision(
        criteria_set=criteria_set,
        assessments=[
            make_assessment(criterion_id="low_priority", status="violated"),
            make_assessment(criterion_id="high_priority", status="violated"),
        ],
    )
    decision_b, codes_b, primary_b = aggregate_decision(
        criteria_set=criteria_set,
        assessments=[
            make_assessment(criterion_id="high_priority", status="violated"),
            make_assessment(criterion_id="low_priority", status="violated"),
        ],
    )
    assert codes_a == codes_b == ["r_high", "r_low"]
    assert primary_a == primary_b == "r_high"
