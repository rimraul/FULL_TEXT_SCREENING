"""Three-arm evaluation harness for `gold_set_protocol_v9.md` §5-8.

Consumes a frozen `GoldSet` (and, for Arm 3, a frozen `RetrievalOracleArtifact`
+ `RetrievalOracleJudgmentSet`) plus the real pipeline components, and
reports exactly the metrics `gold_set_protocol_v9.md` §8 defines -- no
metric definition here may change once a gold set has been frozen and
scored, without a new protocol version.

Fifth revision. `codex_code_review_772916e.md` found the persisted
`EvaluationRunManifest` from the prior round recorded configuration only,
never the outcomes/failures/usage/wall-time actually scored, fixed with a
new `EvaluationResultArtifact` persisted FIRST and exclusively. The
NEXT round (`codex_code_review_1c40a27.md`) found that artifact's own
persisted Arm-2 usage total silently dropped a successful, billed call's
tokens whenever a sibling criterion call for the same document failed
(the per-document-COMPLETE summary was used for a figure that needed to
be the per-call BILLABLE sum instead), and that the artifact itself had
no validators for its own advertised coverage/arithmetic. Both fixed here
-- see `gold_set_protocol_v9.md` §0.

No currency/cost calculation happens here (token counts and wall time
only) and `aggregation.py`'s eligibility-decision logic is used exactly
as-is (via `aggregate_decision`/`build_eligibility_result`), never
reimplemented or special-cased for evaluation.
"""

from __future__ import annotations

import hashlib
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ft_screening.components import AdjudicationOutcome, Components, ParsedDocument
from ft_screening.gold import (
    CriterionKey,
    GoldCriterionAnnotation,
    GoldSet,
    RetrievalOracleArtifact,
    RetrievalOracleJudgmentSet,
    freeze_frozen_model,
    hash_pdf_file,
    load_frozen_model,
    make_evidence_key,
)
from ft_screening.hashing import canonical_json_bytes, compute_criteria_set_hash
from ft_screening.orchestrator import _run_ft_screening
from ft_screening.schema import (
    CriteriaSet,
    CriterionAssessment,
    DocumentCoverage,
    EvidenceSpan,
    ScreeningConfig,
    ScreeningItem,
    ScreeningRequest,
    ScreeningRunResult,
)

CriterionStatus = Literal["satisfied", "violated", "unclear", "not_reported"]
FinalDecision = Literal["include", "exclude", "needs_human_review"]

MISSING_OR_FAILED = "missing_or_failed"
GOLD_EVIDENCE_CONSTRUCTION_VERSION = "gold-evidence-v2"


# ---------------------------------------------------------------------------
# Evaluation-result artifact -- the actual outcomes/failures/usage/wall-time
# that were scored, persisted FIRST and exclusively
# (codex_code_review_772916e.md finding 4: EvaluationRunManifest recorded
# configuration only, never what was actually run and scored -- an Arm-2
# process could exit after the manifest write with no durable outcomes at
# all). The manifest (below) then binds this artifact's own hash.
# ---------------------------------------------------------------------------


_VALID_CRITERION_STATUSES = {"satisfied", "violated", "unclear", "not_reported", MISSING_OR_FAILED}
_VALID_FINAL_DECISIONS = {"include", "exclude", "needs_human_review", MISSING_OR_FAILED}


class EvaluationResultArtifact(BaseModel):
    """codex_code_review_1c40a27.md finding 4: direct construction
    previously accepted a non-SHA gold hash, bogus statuses/decisions,
    mismatched document/reason keys, arithmetically inconsistent token
    counts, and an unparseable `frozen_at`. Every one of those is now a
    hard validation error, not merely a documented convention."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    arm: Literal["e2e", "gold_evidence"]
    gold_set_hash: str = Field(min_length=64, max_length=64)
    criterion_status_by_key: dict[str, str] = Field(min_length=1)  # "doc::crit" -> status or MISSING_OR_FAILED
    document_final_decision: dict[str, str] = Field(min_length=1)
    document_primary_exclusion_reason: dict[str, str | None]
    missing_or_failed_document_ids: tuple[str, ...] = Field(default_factory=tuple)
    missing_or_failed_criterion_keys: tuple[str, ...] = Field(default_factory=tuple)
    # `unit` names what total_*_tokens/n_usage_* are counted over. Arm 1 is
    # always "document". Arm 2's PERSISTED totals are "criterion_call" --
    # the actual billable per-call sum, not the per-document-COMPLETE sum
    # (finding 4: the latter silently drops a paid, successful call's
    # tokens whenever ANY other call for that same document failed). The
    # per-document completeness view is kept too, but as its own separate
    # fields below, never substituted for the billable total.
    usage_unit: Literal["document", "criterion_call"]
    total_input_tokens: int = Field(ge=0)
    total_output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    n_usage_available: int = Field(ge=0)
    n_usage_unavailable: int = Field(ge=0)
    n_documents_usage_complete: int = Field(ge=0)
    n_documents_usage_incomplete: int = Field(ge=0)
    total_wall_seconds: float = Field(ge=0.0)
    frozen_at: str = Field(min_length=1)

    @field_validator("gold_set_hash")
    @classmethod
    def _gold_set_hash_is_sha256_hex(cls, v: str) -> str:
        if len(v) != 64 or any(c not in "0123456789abcdef" for c in v.lower()):
            raise ValueError(f"gold_set_hash must be a 64-character hex sha256 digest, got {v!r}")
        return v

    @field_validator("frozen_at")
    @classmethod
    def _frozen_at_is_iso(cls, v: str) -> str:
        try:
            datetime.fromisoformat(v)
        except ValueError as exc:
            raise ValueError(f"frozen_at must be an ISO-8601 timestamp, got {v!r}") from exc
        return v

    @model_validator(mode="after")
    def _status_and_decision_vocab(self) -> "EvaluationResultArtifact":
        bad_status = {k: v for k, v in self.criterion_status_by_key.items() if v not in _VALID_CRITERION_STATUSES}
        if bad_status:
            raise ValueError(f"criterion_status_by_key has invalid status values: {bad_status}")
        bad_decision = {k: v for k, v in self.document_final_decision.items() if v not in _VALID_FINAL_DECISIONS}
        if bad_decision:
            raise ValueError(f"document_final_decision has invalid decision values: {bad_decision}")
        return self

    @model_validator(mode="after")
    def _exclusion_reason_keys_and_consistency(self) -> "EvaluationResultArtifact":
        if set(self.document_primary_exclusion_reason) != set(self.document_final_decision):
            raise ValueError("document_primary_exclusion_reason must have exactly the same keys as document_final_decision")
        for doc_id, decision in self.document_final_decision.items():
            reason = self.document_primary_exclusion_reason[doc_id]
            if decision == "exclude" and not reason:
                raise ValueError(f"document_id={doc_id!r}: final_decision='exclude' requires a primary_exclusion_reason")
            if decision != "exclude" and reason:
                raise ValueError(f"document_id={doc_id!r}: primary_exclusion_reason must be null unless final_decision=='exclude'")
        return self

    @model_validator(mode="after")
    def _missing_key_partitions_exact(self) -> "EvaluationResultArtifact":
        expected_missing_docs = {d for d, v in self.document_final_decision.items() if v == MISSING_OR_FAILED}
        if set(self.missing_or_failed_document_ids) != expected_missing_docs:
            raise ValueError("missing_or_failed_document_ids does not exactly match the MISSING_OR_FAILED keys in document_final_decision")
        expected_missing_keys = {k for k, v in self.criterion_status_by_key.items() if v == MISSING_OR_FAILED}
        if set(self.missing_or_failed_criterion_keys) != expected_missing_keys:
            raise ValueError("missing_or_failed_criterion_keys does not exactly match the MISSING_OR_FAILED keys in criterion_status_by_key")
        return self

    @model_validator(mode="after")
    def _token_arithmetic(self) -> "EvaluationResultArtifact":
        if self.total_tokens != self.total_input_tokens + self.total_output_tokens:
            raise ValueError(
                f"total_tokens={self.total_tokens} does not equal total_input_tokens+total_output_tokens="
                f"{self.total_input_tokens + self.total_output_tokens}"
            )
        if self.n_usage_available == 0 and self.total_tokens != 0:
            raise ValueError("n_usage_available=0 but nonzero tokens are recorded -- tokens must come from available calls")
        return self

    @model_validator(mode="after")
    def _usage_counts_match_unit_and_coverage(self) -> "EvaluationResultArtifact":
        expected = len(self.document_final_decision) if self.usage_unit == "document" else len(self.criterion_status_by_key)
        if self.n_usage_available + self.n_usage_unavailable != expected:
            raise ValueError(
                f"n_usage_available+n_usage_unavailable={self.n_usage_available + self.n_usage_unavailable} does "
                f"not equal the number of {self.usage_unit!r} entries scored ({expected})"
            )
        n_docs = len(self.document_final_decision)
        if self.n_documents_usage_complete + self.n_documents_usage_incomplete != n_docs:
            raise ValueError(
                "n_documents_usage_complete+n_documents_usage_incomplete does not equal the number of documents "
                f"scored ({n_docs})"
            )
        return self

    def content_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"frozen_at"})
        return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def verify_result_artifact_against_gold_set(result: EvaluationResultArtifact, gold_set: GoldSet) -> None:
    """The load/validation boundary against a supplied frozen `GoldSet`
    that finding 4 pointed out was missing -- a result artifact's own
    validators can check internal consistency, but only this can check
    that it actually names THIS gold set and covers exactly its keys."""
    if result.gold_set_hash != gold_set.content_hash():
        raise ValueError("result.gold_set_hash does not match this gold_set's content_hash()")
    expected_keys = {make_evidence_key(ann.document_id, ann.criterion_id) for ann in gold_set.criterion_annotations}
    if set(result.criterion_status_by_key) != expected_keys:
        raise ValueError("result.criterion_status_by_key does not cover exactly the gold set's criterion keys")
    expected_ids = set(gold_set.expected_document_ids)
    if set(result.document_final_decision) != expected_ids:
        raise ValueError("result.document_final_decision does not cover exactly the gold set's document ids")


class EvaluationRunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    arm: Literal["e2e", "gold_evidence"]
    gold_set_hash: str = Field(min_length=1)
    execution_fingerprint: str = Field(min_length=1)
    result_artifact_hash: str = Field(min_length=1)
    config_summary: dict
    recorded_at: str = Field(min_length=1)

    def content_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"recorded_at"})
        return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_frozen_evaluation_result_artifact(
    json_path: Path, *, expected_hash: str | None = None, gold_set: GoldSet | None = None
) -> EvaluationResultArtifact:
    """`gold_set`, if supplied, additionally binds the loaded artifact to
    it via `verify_result_artifact_against_gold_set` (finding 4)."""
    result = load_frozen_model(EvaluationResultArtifact, json_path, expected_hash=expected_hash, expected_artifact_type="EvaluationResultArtifact")
    if gold_set is not None:
        verify_result_artifact_against_gold_set(result, gold_set)
    return result


def load_frozen_evaluation_run_manifest(json_path: Path, *, expected_hash: str | None = None) -> EvaluationRunManifest:
    return load_frozen_model(EvaluationRunManifest, json_path, expected_hash=expected_hash, expected_artifact_type="EvaluationRunManifest")


# ---------------------------------------------------------------------------
# Arm 1: real retrieval -> LLM adjudication (end to end)
# ---------------------------------------------------------------------------


def run_arm_e2e(
    *,
    gold_set: GoldSet,
    pdf_paths_by_document_id: dict[str, str],
    components: Components,
    config: ScreeningConfig,
    output_dir,
    run_id: str,
    result_version_label: str,
    manifest_version_label: str,
    resume: bool = False,
    extra_config_summary_fields: dict | None = None,
) -> tuple[ScreeningRunResult, EvaluationResultArtifact, EvaluationRunManifest]:
    """The real production path. `config.cache_mode` MUST be "off" or
    "refresh". Every PDF is re-hashed and compared against
    `gold_set.document_pdf_sha256` BEFORE any parsing/adjudication work
    starts -- Arm 1 must score the EXACT PDF bytes the gold set was
    annotated against, not just files that happen to share a
    document_id. The RESULT (outcomes/failures/usage/wall-time) is
    persisted first and exclusively; the manifest then binds its hash --
    never the other way around (codex_code_review_772916e.md finding 4:
    a process could previously exit after the manifest write with no
    durable outcomes at all).

    `resume`: False (default, exact prior behavior for every existing
    caller) submits `run_id` as a FRESH run. True submits it as
    `resume_run_id` instead -- `_run_ft_screening`'s own orchestrator
    already has full manifest/execution_fingerprint-verified resume
    support (see orchestrator.py's `_run_ft_screening`, extensively
    covered by tests/contract/test_hardening_round3/5/6/7.py and
    test_lifecycle_matrix.py); this parameter only exposes that existing
    capability through run_arm_e2e's own call surface, which previously
    always constructed a fresh `run_id`. Per-document/per-criterion
    durability comes from the SAME store-backed mechanism the production
    path already relies on, not a separate, parallel implementation.

    `extra_config_summary_fields`: merged into the manifest's
    `config_summary` dict (e.g. source-identity/commit-tracking fields a
    caller wants durably recorded). Raises if any key collides with one
    `config_summary` already sets, rather than silently overwriting real
    provenance data."""
    if config.cache_mode not in ("off", "refresh"):
        raise ValueError(
            f'run_arm_e2e requires config.cache_mode in ("off", "refresh") for a genuinely fresh '
            f'evaluation call, got {config.cache_mode!r} -- a "use" run can silently score cached '
            "results from a different (or no) provider call as if they were this run's own"
        )

    document_ids = sorted(gold_set.expected_document_ids)
    for doc_id in document_ids:
        pdf_path = Path(pdf_paths_by_document_id[doc_id])
        actual_hash = hash_pdf_file(pdf_path)
        expected_hash = gold_set.document_pdf_sha256[doc_id]
        if actual_hash != expected_hash:
            raise ValueError(
                f"document_id={doc_id!r}: the PDF at {pdf_path} hashes to {actual_hash!r}, but the frozen "
                f"gold set expects document_pdf_sha256={expected_hash!r} -- Arm 1 must score the exact PDF "
                "bytes the gold set was annotated against"
            )

    items = [
        ScreeningItem(document_id=doc_id, report_id=doc_id, pdf_path=pdf_paths_by_document_id[doc_id])
        for doc_id in document_ids
    ]
    request = ScreeningRequest(
        items=items, criteria_set=gold_set.criteria_set, config=config, output_dir=output_dir,
        run_id=None if resume else run_id, resume_run_id=run_id if resume else None,
    )
    result = _run_ft_screening(request, components)

    criterion_predictions = build_criterion_status_predictions_from_run_result(gold_set=gold_set, run_result=result)
    document_decisions = build_document_decisions_from_run_result(gold_set=gold_set, run_result=result)
    completion = processing_completion_report(
        gold_set=gold_set, predicted_status_by_key=criterion_predictions,
        predicted_decision_by_document={d: r.final_decision for d, r in document_decisions.items()},
    )
    usage = token_usage_summary_e2e(gold_set=gold_set, run_result=result)
    wall = wall_time_summary_e2e(result)

    result_artifact = EvaluationResultArtifact(
        arm="e2e", gold_set_hash=gold_set.content_hash(),
        criterion_status_by_key={make_evidence_key(*k): v for k, v in criterion_predictions.items()},
        document_final_decision={d: r.final_decision for d, r in document_decisions.items()},
        document_primary_exclusion_reason={d: r.primary_exclusion_reason_code for d, r in document_decisions.items()},
        missing_or_failed_document_ids=completion.missing_or_failed_document_ids,
        missing_or_failed_criterion_keys=tuple(make_evidence_key(*k) for k in completion.missing_or_failed_criterion_keys),
        usage_unit="document",
        total_input_tokens=usage.total_input_tokens, total_output_tokens=usage.total_output_tokens,
        total_tokens=usage.total_tokens, n_usage_available=usage.n_usage_available,
        n_usage_unavailable=usage.n_usage_unavailable,
        n_documents_usage_complete=usage.n_usage_available, n_documents_usage_incomplete=usage.n_usage_unavailable,
        total_wall_seconds=wall.total_seconds,
        frozen_at=_now_iso(),
    )
    freeze_frozen_model(result_artifact, output_dir=output_dir, version_label=result_version_label, artifact_type="EvaluationResultArtifact")

    config_summary = {
        "adjudicator_model": config.adjudicator_model, "adjudicator_provider": config.adjudicator_provider,
        "prompt_template_version": config.prompt_template_version, "output_schema_version": config.output_schema_version,
        "model_options": config.model_options, "cache_mode": config.cache_mode,
        "parser_name": config.parser_name, "parser_version": config.parser_version,
        "retriever_name": config.retriever_name, "retriever_config_version": config.retriever_config_version,
        "production_run_id": result.run_id, "production_run_result_path": str(result.artifacts.run_result_json),
    }
    if extra_config_summary_fields:
        colliding = set(extra_config_summary_fields) & set(config_summary)
        if colliding:
            raise ValueError(f"extra_config_summary_fields collides with existing config_summary key(s): {sorted(colliding)}")
        config_summary.update(extra_config_summary_fields)
    manifest = EvaluationRunManifest(
        arm="e2e", gold_set_hash=gold_set.content_hash(), execution_fingerprint=result.execution_fingerprint,
        result_artifact_hash=result_artifact.content_hash(), config_summary=config_summary, recorded_at=_now_iso(),
    )
    freeze_frozen_model(manifest, output_dir=output_dir, version_label=manifest_version_label, artifact_type="EvaluationRunManifest")
    return result, result_artifact, manifest


@dataclass(frozen=True)
class DocumentDecisionResult:
    final_decision: str  # FinalDecision or MISSING_OR_FAILED
    primary_exclusion_reason_code: str | None


def build_criterion_status_predictions_from_run_result(
    *, gold_set: GoldSet, run_result: ScreeningRunResult
) -> dict[CriterionKey, str]:
    item_by_doc: dict[str, object] = {item.document_id: item for item in run_result.items}
    predictions: dict[CriterionKey, str] = {}
    for ann in gold_set.criterion_annotations:
        key = (ann.document_id, ann.criterion_id)
        item = item_by_doc.get(ann.document_id)
        if item is None or item.execution_status != "completed" or item.eligibility is None:
            predictions[key] = MISSING_OR_FAILED
            continue
        assessment = next((a for a in item.eligibility.assessments if a.criterion_id == ann.criterion_id), None)
        predictions[key] = assessment.status if assessment is not None else MISSING_OR_FAILED
    return predictions


def build_document_decisions_from_run_result(
    *, gold_set: GoldSet, run_result: ScreeningRunResult
) -> dict[str, DocumentDecisionResult]:
    item_by_doc = {item.document_id: item for item in run_result.items}
    decisions: dict[str, DocumentDecisionResult] = {}
    for doc_id in gold_set.expected_document_ids:
        item = item_by_doc.get(doc_id)
        if item is None or item.execution_status != "completed" or item.eligibility is None:
            decisions[doc_id] = DocumentDecisionResult(final_decision=MISSING_OR_FAILED, primary_exclusion_reason_code=None)
            continue
        decisions[doc_id] = DocumentDecisionResult(
            final_decision=item.eligibility.final_decision,
            primary_exclusion_reason_code=item.eligibility.primary_exclusion_reason_code,
        )
    return decisions


def parser_success_rate(*, gold_set: GoldSet, run_result: ScreeningRunResult) -> float:
    item_by_doc = {item.document_id: item for item in run_result.items}
    total = len(gold_set.expected_document_ids)
    if total == 0:
        return 0.0
    succeeded = sum(
        1 for doc_id in gold_set.expected_document_ids
        if (item := item_by_doc.get(doc_id)) is not None and item.coverage is not None and item.coverage.status != "failed"
    )
    return succeeded / total


# ---------------------------------------------------------------------------
# Arm 2: gold evidence -> LLM adjudication (isolates adjudication error)
# ---------------------------------------------------------------------------


def _placeholder_document(document_id: str) -> ParsedDocument:
    return ParsedDocument(
        document_id=document_id, document_sha256="0" * 64,
        coverage=DocumentCoverage(
            status="complete", total_pages=1, parsed_pages=[1], failed_pages=[],
            extracted_characters=1, ocr_used=False,
        ),
    )


def _gold_evidence_spans(document_id: str, annotation: GoldCriterionAnnotation) -> list[EvidenceSpan]:
    spans: list[EvidenceSpan] = []
    for i, (page, passage) in enumerate(zip(annotation.supporting_pages, annotation.supporting_passages)):
        spans.append(EvidenceSpan(
            document_id=document_id, document_sha256="0" * 64,
            chunk_id=f"gold-{annotation.criterion_id}-supports-{i}", section=None, page=page,
            start_char=0, end_char=max(len(passage), 1), excerpt=passage, stance="supports",
            retrieval_method="gold_annotation", retrieval_query_version=GOLD_EVIDENCE_CONSTRUCTION_VERSION,
        ))
    for i, (page, passage) in enumerate(zip(annotation.contradicting_pages, annotation.contradicting_passages)):
        spans.append(EvidenceSpan(
            document_id=document_id, document_sha256="0" * 64,
            chunk_id=f"gold-{annotation.criterion_id}-contradicts-{i}", section=None, page=page,
            start_char=0, end_char=max(len(passage), 1), excerpt=passage, stance="contradicts",
            retrieval_method="gold_annotation", retrieval_query_version=GOLD_EVIDENCE_CONSTRUCTION_VERSION,
        ))
    return spans


@dataclass(frozen=True)
class ArmGoldEvidenceResult:
    outcomes_by_key: dict[CriterionKey, AdjudicationOutcome | None]  # None -> adjudication call failed
    wall_seconds_by_document: dict[str, float]


class GoldEvidenceCheckpointDriftError(RuntimeError):
    """Raised when an on-disk per-criterion checkpoint's own
    execution_identity_hash doesn't match the CURRENT run's -- e.g. the
    gold set, model, prompt/schema version, or model_options changed since
    that checkpoint was written. Never silently reused across a config
    change; the caller must clear/relocate the stale checkpoint directory
    or fix the drift."""


class GoldEvidenceCriterionCheckpoint(BaseModel):
    """One (document_id, criterion_id) call's durable, individually-frozen
    checkpoint -- written immediately after that call completes (success
    OR failure), under an exclusive-write path keyed by a hash of
    (document_id, criterion_id) (see `_gold_evidence_checkpoint_path`),
    reusing `freeze_frozen_model`'s write-tmp-then-hardlink mechanism,
    which already refuses to overwrite an existing file."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    execution_identity_hash: str = Field(min_length=64, max_length=64)
    document_id: str = Field(min_length=1)
    criterion_id: str = Field(min_length=1)
    outcome: AdjudicationOutcome | None = None  # None means the provider call failed for this criterion
    checkpointed_at: str = Field(min_length=1)

    def content_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"checkpointed_at"})
        return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _gold_evidence_checkpoint_path(checkpoint_dir: Path, *, document_id: str, criterion_id: str) -> Path:
    key_hash = hashlib.sha256(f"{document_id}::{criterion_id}".encode("utf-8")).hexdigest()[:32]
    return checkpoint_dir / f"checkpoint-{key_hash}.json"


def run_arm_gold_evidence(
    *,
    gold_set: GoldSet,
    adjudicator,
    model_options: dict,
    provider_identity: str,
    manifest_output_dir: Path,
    result_version_label: str,
    manifest_version_label: str,
    checkpoint_dir: Path | None = None,
    extra_config_summary_fields: dict | None = None,
) -> tuple[ArmGoldEvidenceResult, EvaluationResultArtifact, EvaluationRunManifest]:
    """Isolates adjudication error. Resilient to a per-criterion
    adjudication failure -- records None rather than aborting the whole
    batch. `provider_identity` (e.g. `"openai"`) is an explicit,
    caller-supplied string bound into the execution fingerprint --
    `adjudicate()` implementations don't expose their transport publicly,
    so this is honest about being caller-asserted rather than
    introspected (codex_code_review_772916e.md finding 4's request for
    provider/transport identity in the fingerprint). The RESULT is
    persisted first and exclusively; the manifest then binds its hash.

    `provider_identity` must be nonblank after stripping -- an empty
    string was previously accepted and fingerprinted, making the
    provider binding optional in practice (codex_code_review_1c40a27.md
    finding 4).

    `checkpoint_dir`: None (default, exact prior behavior for every
    existing caller) means no per-criterion durability -- exactly as
    before this parameter existed. When set, every (document_id,
    criterion_id) call is checkpointed individually and immediately after
    it completes; a call whose checkpoint already exists is loaded and
    reused verbatim (no new provider call), after verifying the
    checkpoint's execution_identity_hash matches this run's own (which
    covers gold_set/model/prompt/schema/model_options/provider_identity --
    see `execution_fingerprint` below) -- a checkpoint from a different
    config raises `GoldEvidenceCheckpointDriftError` rather than being
    silently reused. This is the same discipline
    scripts/calibration_v4.py established, applied here to the real
    gold-evidence arm rather than a synthetic calibration corpus.

    `extra_config_summary_fields`: merged into the manifest's
    `config_summary`/fingerprint dict (e.g. source-identity fields).
    Raises on any key collision with an existing key."""
    provider_identity = provider_identity.strip()
    if not provider_identity:
        raise ValueError("provider_identity must be a nonblank string identifying the adjudicator's transport/provider")

    # Computed BEFORE the loop (not just at manifest-build time) so it can
    # also serve as the checkpoint identity binding -- one hash, two uses,
    # rather than two independently-computed identities that could drift
    # apart from each other.
    fingerprint_payload = {
        "gold_set_hash": gold_set.content_hash(),
        "evidence_construction_version": GOLD_EVIDENCE_CONSTRUCTION_VERSION,
        "provider_identity": provider_identity,
        "adjudicator_model": getattr(adjudicator, "model", None),
        "prompt_template_version": getattr(adjudicator, "prompt_template_version", None),
        "output_schema_version": getattr(adjudicator, "output_schema_version", None),
        "model_options": model_options,
    }
    execution_fingerprint = hashlib.sha256(canonical_json_bytes(fingerprint_payload)).hexdigest()

    criteria_by_id = {c.criterion_id: c for c in gold_set.criteria_set.criteria}
    outcomes: dict[CriterionKey, AdjudicationOutcome | None] = {}
    wall_by_doc: dict[str, float] = defaultdict(float)

    for annotation in gold_set.criterion_annotations:
        key = (annotation.document_id, annotation.criterion_id)

        if checkpoint_dir is not None:
            checkpoint_path = _gold_evidence_checkpoint_path(checkpoint_dir, document_id=annotation.document_id, criterion_id=annotation.criterion_id)
            if checkpoint_path.exists():
                record = load_frozen_model(GoldEvidenceCriterionCheckpoint, checkpoint_path, expected_artifact_type="GoldEvidenceCriterionCheckpoint")
                if record.execution_identity_hash != execution_fingerprint:
                    raise GoldEvidenceCheckpointDriftError(
                        f"checkpoint {checkpoint_path} was frozen under execution_identity_hash="
                        f"{record.execution_identity_hash!r}, which does not match the current run's "
                        f"{execution_fingerprint!r} -- refusing to silently reuse a checkpoint from a "
                        "different gold set/model/prompt/schema/config"
                    )
                outcomes[key] = record.outcome
                continue

        criterion = criteria_by_id[annotation.criterion_id]
        evidence = _gold_evidence_spans(annotation.document_id, annotation)
        document = _placeholder_document(annotation.document_id)
        started = time.monotonic()
        try:
            outcome = adjudicator.adjudicate(
                document=document, criterion=criterion, evidence=evidence, model_options=model_options,
            )
        except Exception:  # noqa: BLE001 -- one failed criterion must not abort the whole arm
            outcome = None
        wall_by_doc[annotation.document_id] += time.monotonic() - started
        outcomes[key] = outcome

        if checkpoint_dir is not None:
            checkpoint_path = _gold_evidence_checkpoint_path(checkpoint_dir, document_id=annotation.document_id, criterion_id=annotation.criterion_id)
            freeze_frozen_model(
                GoldEvidenceCriterionCheckpoint(
                    execution_identity_hash=execution_fingerprint, document_id=annotation.document_id,
                    criterion_id=annotation.criterion_id, outcome=outcome, checkpointed_at=_now_iso(),
                ),
                output_dir=checkpoint_dir, version_label=checkpoint_path.stem, artifact_type="GoldEvidenceCriterionCheckpoint",
            )

    arm_result = ArmGoldEvidenceResult(outcomes_by_key=outcomes, wall_seconds_by_document=dict(wall_by_doc))

    criterion_predictions = build_criterion_status_predictions_from_gold_evidence(gold_set=gold_set, arm_result=arm_result)
    document_decisions = compute_arm_gold_evidence_document_decisions(gold_set=gold_set, arm_result=arm_result)
    completion = processing_completion_report(
        gold_set=gold_set, predicted_status_by_key=criterion_predictions,
        predicted_decision_by_document={d: r.final_decision for d, r in document_decisions.items()},
    )
    # BILLABLE totals: every successful provider call's tokens, regardless of
    # whether some OTHER call for the same document failed
    # (codex_code_review_1c40a27.md finding 4: the previous persisted total
    # used the per-document-COMPLETE summary, which drops an entire
    # document's tokens -- including successful, paid calls -- whenever
    # any ONE criterion in that document lacked telemetry).
    usage_billable = token_usage_summary_gold_evidence(gold_set=gold_set, arm_result=arm_result)
    # Per-document COMPLETENESS: kept as its own separate figure, never
    # substituted for the billable total above.
    usage_per_doc = token_usage_summary_gold_evidence_per_document(gold_set=gold_set, arm_result=arm_result)
    wall = wall_time_summary_gold_evidence(arm_result)

    result_artifact = EvaluationResultArtifact(
        arm="gold_evidence", gold_set_hash=gold_set.content_hash(),
        criterion_status_by_key={make_evidence_key(*k): v for k, v in criterion_predictions.items()},
        document_final_decision={d: r.final_decision for d, r in document_decisions.items()},
        document_primary_exclusion_reason={d: r.primary_exclusion_reason_code for d, r in document_decisions.items()},
        missing_or_failed_document_ids=completion.missing_or_failed_document_ids,
        missing_or_failed_criterion_keys=tuple(make_evidence_key(*k) for k in completion.missing_or_failed_criterion_keys),
        usage_unit="criterion_call",
        total_input_tokens=usage_billable.total_input_tokens, total_output_tokens=usage_billable.total_output_tokens,
        total_tokens=usage_billable.total_tokens, n_usage_available=usage_billable.n_usage_available,
        n_usage_unavailable=usage_billable.n_usage_unavailable,
        n_documents_usage_complete=usage_per_doc.n_usage_available, n_documents_usage_incomplete=usage_per_doc.n_usage_unavailable,
        total_wall_seconds=wall.total_seconds,
        frozen_at=_now_iso(),
    )
    freeze_frozen_model(result_artifact, output_dir=manifest_output_dir, version_label=result_version_label, artifact_type="EvaluationResultArtifact")

    # execution_fingerprint/fingerprint_payload were already computed
    # ABOVE, before the loop -- reused here verbatim (not recomputed) so
    # the checkpoint identity binding and the manifest's own fingerprint
    # can never drift apart from each other.
    config_summary = dict(fingerprint_payload)
    if extra_config_summary_fields:
        colliding = set(extra_config_summary_fields) & set(config_summary)
        if colliding:
            raise ValueError(f"extra_config_summary_fields collides with existing config_summary key(s): {sorted(colliding)}")
        config_summary.update(extra_config_summary_fields)
    manifest = EvaluationRunManifest(
        arm="gold_evidence", gold_set_hash=gold_set.content_hash(), execution_fingerprint=execution_fingerprint,
        result_artifact_hash=result_artifact.content_hash(), config_summary=config_summary, recorded_at=_now_iso(),
    )
    freeze_frozen_model(manifest, output_dir=manifest_output_dir, version_label=manifest_version_label, artifact_type="EvaluationRunManifest")
    return arm_result, result_artifact, manifest


def build_criterion_status_predictions_from_gold_evidence(
    *, gold_set: GoldSet, arm_result: ArmGoldEvidenceResult
) -> dict[CriterionKey, str]:
    predictions: dict[CriterionKey, str] = {}
    for ann in gold_set.criterion_annotations:
        key = (ann.document_id, ann.criterion_id)
        outcome = arm_result.outcomes_by_key.get(key)
        predictions[key] = outcome.assessment.status if outcome is not None else MISSING_OR_FAILED
    return predictions


def _document_sha256_for_decision(*, doc_id: str, assessments: list[CriterionAssessment], gold_set: GoldSet) -> str:
    """Derives the document identity hash to bind an aggregated
    EligibilityResult to, from the ACTUAL evidence the assessments cite --
    never a hardcoded placeholder. Every gold_evidence-family arm's
    evidence spans carry SOME document_sha256 (the historical "0"*64
    placeholder for arms built on `_gold_evidence_spans`, e.g. Arm 2; a
    real PDF hash for arms that know one, e.g. Arm 2b's redacted
    packages) -- reading it off the evidence itself means this function
    adapts correctly to either convention with no arm-specific branching,
    and previously-passing callers (whose evidence is uniformly "0"*64)
    are unaffected.

    Falls back to the GoldSet's own real PDF hash only when a document's
    assessments cite NO evidence at all across every criterion (e.g. an
    entirely empty-package outcome) -- there is nothing to derive from,
    so the one value this function actually KNOWS to be true is used.

    Raises if the SAME document's assessments disagree on
    document_sha256 across different criteria -- never expected, and
    never silently resolved by picking one. (Incident this fixes: Arm 2b's
    runner emitted evidence bound to the real PDF hash while this
    function hardcoded "0"*64, which crashed build_eligibility_result's
    own evidence-identity check on every document. The fix belongs here,
    in the shared function, rather than as a runner-side workaround that
    papers over a hardcoded assumption every gold_evidence-family caller
    should be able to rely on being correct.)"""
    hashes = {span.document_sha256 for a in assessments for span in a.evidence}
    if len(hashes) > 1:
        raise ValueError(
            f"document_id={doc_id!r}: assessments cite evidence bound to multiple different document_sha256 "
            f"values ({sorted(hashes)}) -- refusing to guess which is correct"
        )
    if hashes:
        return next(iter(hashes))
    return gold_set.document_pdf_sha256[doc_id]


def compute_arm_gold_evidence_document_decisions(
    *, gold_set: GoldSet, arm_result: ArmGoldEvidenceResult
) -> dict[str, DocumentDecisionResult]:
    from ft_screening.aggregation import build_eligibility_result

    criteria_set_hash = compute_criteria_set_hash(gold_set.criteria_set)
    decisions: dict[str, DocumentDecisionResult] = {}
    by_doc: dict[str, list[GoldCriterionAnnotation]] = {}
    for ann in gold_set.criterion_annotations:
        by_doc.setdefault(ann.document_id, []).append(ann)

    for doc_id, anns in by_doc.items():
        outcomes = [arm_result.outcomes_by_key.get((doc_id, a.criterion_id)) for a in anns]
        if any(o is None for o in outcomes):
            decisions[doc_id] = DocumentDecisionResult(final_decision=MISSING_OR_FAILED, primary_exclusion_reason_code=None)
            continue
        assessments = [o.assessment for o in outcomes]  # type: ignore[union-attr]
        document_sha256 = _document_sha256_for_decision(doc_id=doc_id, assessments=assessments, gold_set=gold_set)
        result = build_eligibility_result(
            result_id=f"gold-evidence-arm:{doc_id}:1", document_id=doc_id, document_sha256=document_sha256,
            report_id=doc_id, study_id=None, criteria_set=gold_set.criteria_set,
            criteria_set_hash=criteria_set_hash, execution_fingerprint="0" * 64,
            assessments=assessments, created_at=datetime.now(timezone.utc),
        )
        decisions[doc_id] = DocumentDecisionResult(
            final_decision=result.final_decision, primary_exclusion_reason_code=result.primary_exclusion_reason_code,
        )
    return decisions


# ---------------------------------------------------------------------------
# Arm 3: real retrieval -> oracle sufficiency (isolates retrieval loss),
# scored against the FROZEN RetrievalOracleArtifact + RetrievalOracleJudgmentSet
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievalOracleReport:
    mechanical_evidence_recall: float
    human_sufficiency_rate: float
    n_criteria_with_evidence: int
    n_criteria_evaluated: int


def compute_retrieval_oracle_metrics(
    *, gold_set: GoldSet, retrieval_artifact: RetrievalOracleArtifact, judgment_set: RetrievalOracleJudgmentSet
) -> RetrievalOracleReport:
    """Every binding is verified, not trusted (codex_code_review_dca0970.md
    finding 3): the judgment set's own hashes, the retrieval artifact's
    criteria_set_hash, its document_sha256 (EXACT dict equality against
    the gold set's, not just an overlap), and its evidence_by_key
    coverage (EXACTLY the gold set's 60 criterion keys)."""
    if judgment_set.gold_set_hash != gold_set.content_hash():
        raise ValueError("judgment_set.gold_set_hash does not match this gold_set's content_hash()")
    if judgment_set.retrieval_artifact_hash != retrieval_artifact.content_hash():
        raise ValueError("judgment_set.retrieval_artifact_hash does not match this retrieval_artifact's content_hash()")
    if retrieval_artifact.criteria_set_hash != compute_criteria_set_hash(gold_set.criteria_set):
        raise ValueError(
            "retrieval_artifact.criteria_set_hash does not match gold_set.criteria_set -- Arm 3 must be "
            "scored against the SAME criteria set the gold set uses"
        )
    if retrieval_artifact.document_sha256 != gold_set.document_pdf_sha256:
        raise ValueError(
            "retrieval_artifact.document_sha256 does not exactly match gold_set.document_pdf_sha256 -- Arm 3 "
            "must have parsed the EXACT PDF bytes the gold set was annotated against, not merely an "
            "overlapping or partially-matching set"
        )

    expected_keys = {(ann.document_id, ann.criterion_id) for ann in gold_set.criterion_annotations}
    artifact_keys = {tuple(k.split("::", 1)) for k in retrieval_artifact.evidence_by_key}
    if artifact_keys != expected_keys:
        raise ValueError(
            "retrieval_artifact.evidence_by_key does not cover exactly the gold set's criterion keys "
            f"(missing={sorted(expected_keys - artifact_keys)}, extra={sorted(artifact_keys - expected_keys)})"
        )

    judgment_by_key = {(j.document_id, j.criterion_id): j for j in judgment_set.judgments}
    judged_keys = set(judgment_by_key)
    if judged_keys != expected_keys:
        raise ValueError(
            "judgment_set does not cover exactly the gold set's criterion rows "
            f"(missing={sorted(expected_keys - judged_keys)}, extra={sorted(judged_keys - expected_keys)})"
        )

    recall_hits = 0
    recall_total = 0
    sufficient_count = 0
    evaluated_count = 0

    for annotation in gold_set.criterion_annotations:
        key = (annotation.document_id, annotation.criterion_id)
        evaluated_count += 1
        if judgment_by_key[key].retrieval_sufficient:
            sufficient_count += 1

        gold_pages = set(annotation.supporting_pages) | set(annotation.contradicting_pages)
        if not gold_pages:
            continue
        recall_total += 1
        retrieved_spans = retrieval_artifact.evidence_by_key.get(make_evidence_key(*key), ())
        retrieved_pages = {s.page for s in retrieved_spans if s.page is not None}
        if gold_pages & retrieved_pages:
            recall_hits += 1

    return RetrievalOracleReport(
        mechanical_evidence_recall=(recall_hits / recall_total) if recall_total else 0.0,
        human_sufficiency_rate=(sufficient_count / evaluated_count) if evaluated_count else 0.0,
        n_criteria_with_evidence=recall_total, n_criteria_evaluated=evaluated_count,
    )


# ---------------------------------------------------------------------------
# Strict, coverage-checked scoring
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfusionMatrix:
    counts: dict[tuple[str, str], int] = field(default_factory=dict)

    def accuracy(self) -> float:
        total = sum(self.counts.values())
        if total == 0:
            return 0.0
        correct = sum(v for (g, p), v in self.counts.items() if g == p)
        return correct / total


def _require_exact_coverage(expected: set, actual: set, *, label: str) -> None:
    if actual != expected:
        missing = expected - actual
        extra = actual - expected
        raise ValueError(
            f"{label}: predictions do not cover exactly the gold set's keys "
            f"(missing={sorted(missing, key=str)}, extra={sorted(extra, key=str)}) -- fill every gold key "
            f"explicitly (use evaluation_harness.MISSING_OR_FAILED for anything not actually produced), "
            "never omit one"
        )


def criterion_status_agreement(*, gold_set: GoldSet, predicted_status_by_key: dict[CriterionKey, str]) -> ConfusionMatrix:
    expected_keys = {(ann.document_id, ann.criterion_id) for ann in gold_set.criterion_annotations}
    _require_exact_coverage(expected_keys, set(predicted_status_by_key), label="criterion_status_agreement")
    counts: Counter[tuple[str, str]] = Counter()
    for annotation in gold_set.criterion_annotations:
        key = (annotation.document_id, annotation.criterion_id)
        counts[(annotation.gold_status, predicted_status_by_key[key])] += 1
    return ConfusionMatrix(counts=dict(counts))


def final_decision_agreement(*, gold_set: GoldSet, predicted_decision_by_document: dict[str, str]) -> ConfusionMatrix:
    expected_ids = set(gold_set.expected_document_ids)
    _require_exact_coverage(expected_ids, set(predicted_decision_by_document), label="final_decision_agreement")
    counts: Counter[tuple[str, str]] = Counter()
    for annotation in gold_set.document_annotations:
        counts[(annotation.gold_final_decision, predicted_decision_by_document[annotation.document_id])] += 1
    return ConfusionMatrix(counts=dict(counts))


@dataclass(frozen=True)
class ExclusionErrorReport:
    """`rate`'s denominator is the FULL gold include/exclude count,
    including any missing/failed predictions (codex_code_review_dca0970.md
    finding 5: excluding missing predictions from the denominator would
    let a pipeline that processes zero documents report a 0% error rate).
    `decided_count`/`missing_count` make that split explicit rather than
    leaving it implicit in the rate alone."""

    rate: float
    document_ids: tuple[str, ...]
    denominator: int
    decided_count: int
    missing_count: int


def false_exclusion_rate(*, gold_set: GoldSet, predicted_decision_by_document: dict[str, str]) -> ExclusionErrorReport:
    expected_ids = set(gold_set.expected_document_ids)
    _require_exact_coverage(expected_ids, set(predicted_decision_by_document), label="false_exclusion_rate")
    gold_include = [a.document_id for a in gold_set.document_annotations if a.gold_final_decision == "include"]
    false_excluded = [doc_id for doc_id in gold_include if predicted_decision_by_document[doc_id] == "exclude"]
    missing = [doc_id for doc_id in gold_include if predicted_decision_by_document[doc_id] == MISSING_OR_FAILED]
    rate = (len(false_excluded) / len(gold_include)) if gold_include else 0.0
    return ExclusionErrorReport(
        rate=rate, document_ids=tuple(sorted(false_excluded)), denominator=len(gold_include),
        decided_count=len(gold_include) - len(missing), missing_count=len(missing),
    )


def false_inclusion_rate(*, gold_set: GoldSet, predicted_decision_by_document: dict[str, str]) -> ExclusionErrorReport:
    expected_ids = set(gold_set.expected_document_ids)
    _require_exact_coverage(expected_ids, set(predicted_decision_by_document), label="false_inclusion_rate")
    gold_exclude = [a.document_id for a in gold_set.document_annotations if a.gold_final_decision == "exclude"]
    false_included = [doc_id for doc_id in gold_exclude if predicted_decision_by_document[doc_id] == "include"]
    missing = [doc_id for doc_id in gold_exclude if predicted_decision_by_document[doc_id] == MISSING_OR_FAILED]
    rate = (len(false_included) / len(gold_exclude)) if gold_exclude else 0.0
    return ExclusionErrorReport(
        rate=rate, document_ids=tuple(sorted(false_included)), denominator=len(gold_exclude),
        decided_count=len(gold_exclude) - len(missing), missing_count=len(missing),
    )


def correct_exclusion_reason_rate(*, gold_set: GoldSet, document_decisions: dict[str, DocumentDecisionResult]) -> float:
    expected_ids = set(gold_set.expected_document_ids)
    _require_exact_coverage(expected_ids, set(document_decisions), label="correct_exclusion_reason_rate")
    gold_by_doc = {a.document_id: a for a in gold_set.document_annotations}
    agreed_excludes = [
        doc_id for doc_id in gold_set.expected_document_ids
        if gold_by_doc[doc_id].gold_final_decision == "exclude" and document_decisions[doc_id].final_decision == "exclude"
    ]
    if not agreed_excludes:
        return 0.0
    correct = sum(
        1 for doc_id in agreed_excludes
        if document_decisions[doc_id].primary_exclusion_reason_code == gold_by_doc[doc_id].gold_primary_exclusion_reason_code
    )
    return correct / len(agreed_excludes)


@dataclass(frozen=True)
class ReviewRateReport:
    review_rate: float
    review_rate_among_gold_hard_cases: float
    review_rate_among_gold_easy_cases: float


def review_rate(*, gold_set: GoldSet, predicted_decision_by_document: dict[str, str]) -> ReviewRateReport:
    expected_ids = set(gold_set.expected_document_ids)
    _require_exact_coverage(expected_ids, set(predicted_decision_by_document), label="review_rate")
    hard_doc_ids = {ann.document_id for ann in gold_set.criterion_annotations if ann.human_review_required}
    all_doc_ids = expected_ids
    easy_doc_ids = all_doc_ids - hard_doc_ids

    def _rate(doc_ids: set) -> float:
        if not doc_ids:
            return 0.0
        reviewed = sum(1 for d in doc_ids if predicted_decision_by_document[d] == "needs_human_review")
        return reviewed / len(doc_ids)

    return ReviewRateReport(
        review_rate=_rate(all_doc_ids), review_rate_among_gold_hard_cases=_rate(hard_doc_ids),
        review_rate_among_gold_easy_cases=_rate(easy_doc_ids),
    )


# ---------------------------------------------------------------------------
# Processing-completion / failure accounting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProcessingCompletionReport:
    completed_document_ids: tuple[str, ...]
    missing_or_failed_document_ids: tuple[str, ...]
    completed_criterion_keys: tuple[CriterionKey, ...]
    missing_or_failed_criterion_keys: tuple[CriterionKey, ...]


def processing_completion_report(
    *, gold_set: GoldSet, predicted_status_by_key: dict[CriterionKey, str], predicted_decision_by_document: dict[str, str]
) -> ProcessingCompletionReport:
    """Requires exact coverage of both dicts (codex_code_review_dca0970.md
    finding 5: an empty prediction dict previously produced empty
    completed AND empty missing lists, instead of naming every expected
    key as missing)."""
    expected_keys = {(ann.document_id, ann.criterion_id) for ann in gold_set.criterion_annotations}
    _require_exact_coverage(expected_keys, set(predicted_status_by_key), label="processing_completion_report (criteria)")
    expected_ids = set(gold_set.expected_document_ids)
    _require_exact_coverage(expected_ids, set(predicted_decision_by_document), label="processing_completion_report (documents)")

    completed_docs = tuple(sorted(d for d, v in predicted_decision_by_document.items() if v != MISSING_OR_FAILED))
    missing_docs = tuple(sorted(d for d, v in predicted_decision_by_document.items() if v == MISSING_OR_FAILED))
    completed_keys = tuple(sorted((k for k, v in predicted_status_by_key.items() if v != MISSING_OR_FAILED)))
    missing_keys = tuple(sorted((k for k, v in predicted_status_by_key.items() if v == MISSING_OR_FAILED)))
    return ProcessingCompletionReport(
        completed_document_ids=completed_docs, missing_or_failed_document_ids=missing_docs,
        completed_criterion_keys=completed_keys, missing_or_failed_criterion_keys=missing_keys,
    )


# ---------------------------------------------------------------------------
# Token usage / wall time -- unit-labeled, missing/unavailable never
# silently dropped (codex_code_review_dca0970.md finding 5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UsageSummary:
    unit: Literal["document", "criterion_call"]
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    mean_input_tokens: float
    mean_output_tokens: float
    mean_total_tokens: float
    n_usage_available: int
    n_usage_unavailable: int


def token_usage_summary_e2e(*, gold_set: GoldSet, run_result: ScreeningRunResult) -> UsageSummary:
    """Per-document. An item with NO provenance at all (e.g. it never
    reached that stage before failing) counts as usage_unavailable, not
    silently dropped from both counts (finding 5's exact repro)."""
    provenance_by_doc = {item.document_id: item.provenance for item in run_result.items}
    available = []
    unavailable_count = 0
    for doc_id in gold_set.expected_document_ids:
        provenance = provenance_by_doc.get(doc_id)
        if provenance is None or not provenance.usage_available:
            unavailable_count += 1
            continue
        available.append(provenance)

    total_input = sum(p.total_input_tokens for p in available)
    total_output = sum(p.total_output_tokens for p in available)
    total = sum(p.total_tokens for p in available)
    n = len(available)
    return UsageSummary(
        unit="document", total_input_tokens=total_input, total_output_tokens=total_output, total_tokens=total,
        mean_input_tokens=(total_input / n) if n else 0.0, mean_output_tokens=(total_output / n) if n else 0.0,
        mean_total_tokens=(total / n) if n else 0.0, n_usage_available=n, n_usage_unavailable=unavailable_count,
    )


def token_usage_summary_gold_evidence(*, gold_set: GoldSet, arm_result: ArmGoldEvidenceResult) -> UsageSummary:
    """Per-criterion-call (NOT per-document -- see
    token_usage_summary_gold_evidence_per_document for a figure
    comparable to Arm 1's). A failed (`None`) outcome counts as
    usage_unavailable, not silently dropped before that count is taken
    (finding 5's exact repro)."""
    expected_keys = {(ann.document_id, ann.criterion_id) for ann in gold_set.criterion_annotations}
    available = []
    unavailable_count = 0
    for key in expected_keys:
        outcome = arm_result.outcomes_by_key.get(key)
        if outcome is None or not outcome.usage.usage_available:
            unavailable_count += 1
            continue
        available.append(outcome.usage)

    total_input = sum(u.input_tokens for u in available)
    total_output = sum(u.output_tokens for u in available)
    total = sum(u.total_tokens for u in available)
    n = len(available)
    return UsageSummary(
        unit="criterion_call", total_input_tokens=total_input, total_output_tokens=total_output, total_tokens=total,
        mean_input_tokens=(total_input / n) if n else 0.0, mean_output_tokens=(total_output / n) if n else 0.0,
        mean_total_tokens=(total / n) if n else 0.0, n_usage_available=n, n_usage_unavailable=unavailable_count,
    )


def token_usage_summary_gold_evidence_per_document(
    *, gold_set: GoldSet, arm_result: ArmGoldEvidenceResult
) -> UsageSummary:
    """Per-document (sums each document's criterion-call usage, then
    means across documents) -- the figure actually comparable to
    token_usage_summary_e2e's, unlike the raw per-call summary
    (codex_code_review_dca0970.md finding 5: v2's two summaries looked
    comparable while silently using different units)."""
    by_doc: dict[str, list] = defaultdict(list)
    for ann in gold_set.criterion_annotations:
        by_doc[ann.document_id].append((ann.document_id, ann.criterion_id))

    available_totals = []
    unavailable_count = 0
    for doc_id, keys in by_doc.items():
        outcomes = [arm_result.outcomes_by_key.get(k) for k in keys]
        if any(o is None or not o.usage.usage_available for o in outcomes):
            unavailable_count += 1
            continue
        available_totals.append((
            sum(o.usage.input_tokens for o in outcomes),
            sum(o.usage.output_tokens for o in outcomes),
            sum(o.usage.total_tokens for o in outcomes),
        ))

    total_input = sum(t[0] for t in available_totals)
    total_output = sum(t[1] for t in available_totals)
    total = sum(t[2] for t in available_totals)
    n = len(available_totals)
    return UsageSummary(
        unit="document", total_input_tokens=total_input, total_output_tokens=total_output, total_tokens=total,
        mean_input_tokens=(total_input / n) if n else 0.0, mean_output_tokens=(total_output / n) if n else 0.0,
        mean_total_tokens=(total / n) if n else 0.0, n_usage_available=n, n_usage_unavailable=unavailable_count,
    )


@dataclass(frozen=True)
class WallTimeSummary:
    total_seconds: float
    mean_seconds_per_document: float
    n_documents: int


def wall_time_summary_e2e(run_result: ScreeningRunResult) -> WallTimeSummary:
    durations = [
        (item.provenance.completed_at - item.provenance.started_at).total_seconds()
        for item in run_result.items if item.provenance is not None
    ]
    total = sum(durations)
    n = len(durations)
    return WallTimeSummary(total_seconds=total, mean_seconds_per_document=(total / n) if n else 0.0, n_documents=n)


def wall_time_summary_gold_evidence(arm_result: ArmGoldEvidenceResult) -> WallTimeSummary:
    durations = list(arm_result.wall_seconds_by_document.values())
    total = sum(durations)
    n = len(durations)
    return WallTimeSummary(total_seconds=total, mean_seconds_per_document=(total / n) if n else 0.0, n_documents=n)


# ---------------------------------------------------------------------------
# Deliberately out of scope for this pilot: page-span fidelity (would need
# a char-offset gold schema this pilot's scale doesn't justify).
# ---------------------------------------------------------------------------
