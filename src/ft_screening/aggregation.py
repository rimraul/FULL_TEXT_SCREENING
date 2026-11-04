"""Deterministic eligibility aggregation and the EligibilityResult factory.

api_contract_v2.md correction 6 requires a completed EligibilityResult be
constructed through a factory that performs CONTEXTUAL validation (needs a
CriteriaSet the model itself doesn't hold) -- not arbitrary direct
EligibilityResult(...) construction. This module is that factory.

`AGGREGATION_POLICY_VERSION` is this module's own live version identity --
`orchestrator.resolve_components()` validates a caller's declared
`ScreeningConfig.aggregation_policy_version` against it (codex_code_review_2be6cfa.md
finding 1: previously nothing checked that value even though it's hashed
into `execution_fingerprint`, so an arbitrary string could resolve
successfully and hash a policy identity that doesn't exist).

Decision policy (documented here since CriterionSpec.eligibility_requirement
is human-readable prose, not executable -- the state machine lives in code,
per api_contract_v2.md's eligibility_requirement correction):

  a. Any assessment with status == "violated"            -> exclude
  b. Else any unclear/not_reported criterion whose
     missing_evidence_policy == "fail"                    -> exclude
  c. Else any assessment.requires_human_review == True,
     or any unclear/not_reported criterion whose
     missing_evidence_policy == "review"                  -> needs_human_review
  d. Else (all satisfied, or unclear/not_reported with
     missing_evidence_policy == "pass")                    -> include

A clear negative (a) wins over an unresolved (c) -- once a study is
definitively excluded on one criterion, further review of another
criterion's missing evidence doesn't change that. exclusion_reason_codes
collects every criterion contributing to (a)/(b); primary_exclusion_reason_code
is the contributing criterion with the lowest CriterionSpec.priority
(priority is a rank: lower number = decided first), tie-broken by
criterion_id ascending.
"""

from __future__ import annotations

import unicodedata
from datetime import datetime
from typing import Literal

from ft_screening.hashing import compute_criteria_set_hash
from ft_screening.schema import CriteriaSet, CriterionAssessment, EligibilityResult

Decision = Literal["include", "exclude", "needs_human_review"]

AGGREGATION_POLICY_VERSION = "1"


class AggregationError(ValueError):
    """Contextual validation failure -- an orchestration bug, not a caller input error."""


def aggregate_decision(
    *, criteria_set: CriteriaSet, assessments: list[CriterionAssessment]
) -> tuple[Decision, list[str], str | None]:
    spec_by_id = {c.criterion_id: c for c in criteria_set.criteria}

    excluding_criteria: list[tuple[int, str]] = []  # (priority, criterion_id) for primary selection + ordering
    needs_review = False

    for assessment in assessments:
        spec = spec_by_id[assessment.criterion_id]
        if assessment.requires_human_review:
            needs_review = True

        if assessment.status == "violated":
            excluding_criteria.append((spec.priority, spec.criterion_id))
        elif assessment.status in ("unclear", "not_reported"):
            if spec.missing_evidence_policy == "fail":
                excluding_criteria.append((spec.priority, spec.criterion_id))
            elif spec.missing_evidence_policy == "review":
                needs_review = True
            # "pass" -> neutral, no contribution

    if excluding_criteria:
        # Sort once by (priority, criterion_id) -- makes both the primary
        # reason AND the exclusion_reason_codes list order deterministic
        # independent of the input `assessments` list's own order.
        excluding_criteria.sort()
        primary_criterion_id = excluding_criteria[0][1]
        primary_reason_code = spec_by_id[primary_criterion_id].exclusion_reason_code
        exclusion_reason_codes = list(
            dict.fromkeys(spec_by_id[cid].exclusion_reason_code for _, cid in excluding_criteria)
        )
        return "exclude", exclusion_reason_codes, primary_reason_code

    if needs_review:
        return "needs_human_review", [], None

    return "include", [], None


def build_eligibility_result(
    *,
    result_id: str,
    document_id: str,
    document_sha256: str,
    report_id: str,
    study_id: str | None,
    criteria_set: CriteriaSet,
    criteria_set_hash: str,
    execution_fingerprint: str,
    assessments: list[CriterionAssessment],
    created_at: datetime,
) -> EligibilityResult:
    """The only sanctioned way to construct a completed EligibilityResult.

    Performs the contextual validation EligibilityResult itself can't (it
    doesn't hold a CriteriaSet): assessment-to-criteria-set completeness,
    per-criterion version match, allowed_sources enforcement,
    requires_human_review/missing_evidence_policy consistency, evidence-to-
    result document identity, and that criteria_set_hash actually hashes
    the supplied CriteriaSet. Evidence-to-document TEXT fidelity is NOT
    checked here -- that needs the real parsed document text, out of scope
    until real parsing exists (plan_v5 step 6).
    """
    if compute_criteria_set_hash(criteria_set) != criteria_set_hash:
        raise AggregationError(
            "criteria_set_hash does not match the hash of the supplied criteria_set -- "
            "the caller passed a hash that doesn't describe this CriteriaSet"
        )

    spec_by_id = {c.criterion_id: c for c in criteria_set.criteria}
    assessed_ids = {a.criterion_id for a in assessments}
    expected_ids = set(spec_by_id)
    if assessed_ids != expected_ids:
        missing = expected_ids - assessed_ids
        extra = assessed_ids - expected_ids
        raise AggregationError(
            f"assessments do not cover the criteria set exactly: missing={sorted(missing)}, "
            f"unexpected={sorted(extra)}"
        )
    if len(assessments) != len(expected_ids):
        raise AggregationError("duplicate assessment(s) for the same criterion_id")

    for assessment in assessments:
        spec = spec_by_id[assessment.criterion_id]
        if assessment.criterion_version != spec.version:
            raise AggregationError(
                f"assessment for {assessment.criterion_id!r} was made against criterion_version="
                f"{assessment.criterion_version!r}, but the active CriteriaSet has version={spec.version!r}"
            )
        if assessment.status in ("unclear", "not_reported") and spec.missing_evidence_policy == "review":
            if not assessment.requires_human_review:
                raise AggregationError(
                    f"assessment for {assessment.criterion_id!r} has status={assessment.status!r} under a "
                    "'review' missing_evidence_policy but requires_human_review is False -- the adjudicator "
                    "must set requires_human_review=True whenever this policy applies"
                )
        for span in assessment.evidence:
            if span.document_id != document_id or span.document_sha256 != document_sha256:
                raise AggregationError(
                    f"evidence for criterion {assessment.criterion_id!r} is bound to "
                    f"document_id={span.document_id!r}/document_sha256={span.document_sha256!r}, "
                    f"not this result's document_id={document_id!r}/document_sha256={document_sha256!r}"
                )
            if spec.allowed_sources is not None:
                normalized_section = _normalize_source_label(span.section)
                if normalized_section not in spec.allowed_sources:
                    raise AggregationError(
                        f"evidence for criterion {assessment.criterion_id!r} came from section "
                        f"{span.section!r}, not in its allowed_sources {spec.allowed_sources!r}"
                    )

    decision, exclusion_reason_codes, primary_exclusion_reason_code = aggregate_decision(
        criteria_set=criteria_set, assessments=assessments
    )

    return EligibilityResult(
        result_id=result_id,
        revision=1,
        supersedes_result_id=None,
        document_id=document_id,
        document_sha256=document_sha256,
        report_id=report_id,
        study_id=study_id,
        criteria_set_hash=criteria_set_hash,
        execution_fingerprint=execution_fingerprint,
        assessments=assessments,
        proposed_decision=decision,
        final_decision=decision,
        exclusion_reason_codes=exclusion_reason_codes,
        primary_exclusion_reason_code=primary_exclusion_reason_code,
        human_adjudication=None,
        created_at=created_at,
    )


def _normalize_source_label(section: str | None) -> str | None:
    if section is None:
        return None
    return unicodedata.normalize("NFC", section).strip().casefold()
