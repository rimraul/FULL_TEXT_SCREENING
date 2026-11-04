"""Public domain models for FT_SCREENING.

Implements api_contract_v2.md exactly. Models are Pydantic `frozen=True`
"attribute-frozen validated snapshots" -- NOT deeply immutable or hashable
(Pydantic's frozen=True is shallow: list/dict fields stay mutable, and
these models are not hashable). Never hash a model instance directly; hash
its canonical JSON serialization instead (see hashing.py).

Only LOCAL validators live here -- checks a model can enforce using fields
it holds itself. CONTEXTUAL validation (e.g. an EvidenceSpan's excerpt
against real document text, or an EligibilityResult's assessment coverage
against a CriteriaSet it doesn't contain) belongs in a result/aggregation
factory, not here (api_contract_v2.md correction 6).

Second-round implementation review (codex_code_review_bae78ac.md) added:
JSON-safety + defensive copying for Any-typed dicts (citation/options);
SHA-256 field format validation; finite-score enforcement; allowed_sources
NFC+trim+casefold normalization; safe-token run identifiers; non-empty
config identifiers; tz-aware UTC timestamp normalization + ordering;
completed-item/run identity and fingerprint cross-checks; a corrected
DocumentCoverage status state table; and a corrected completed/
completed_with_issues/failed run_status partition.
"""

from __future__ import annotations

import math
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def _validate_sha256(value: str) -> str:
    normalized = value.lower()
    if not _SHA256_RE.match(normalized):
        raise ValueError(f"expected 64 lowercase hexadecimal characters (sha256), got {value!r}")
    return normalized


def _validate_safe_token(value: str) -> str:
    if not _SAFE_TOKEN_RE.match(value):
        raise ValueError(
            f"expected a filesystem-safe token ([A-Za-z0-9_-], 1-128 chars, no '.'/'..'/path "
            f"separators), got {value!r}"
        )
    return value


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        raise ValueError("datetime must be timezone-aware (UTC)")
    return dt.astimezone(timezone.utc)


def _deep_validate_json(value: Any, *, path: str = "$") -> Any:
    """Recursively validate `value` is JSON-safe and return a fresh (defensively
    copied) structure -- never the caller's original mutable containers.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, type(None))):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path}: non-finite float is not JSON-safe: {value!r}")
        return value
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise ValueError(f"{path}: object keys must be strings, got {type(k).__name__}")
            result[k] = _deep_validate_json(v, path=f"{path}.{k}")
        return result
    if isinstance(value, list):
        return [_deep_validate_json(v, path=f"{path}[{i}]") for i, v in enumerate(value)]
    raise ValueError(f"{path}: value of type {type(value).__name__} is not JSON-safe")


class CriterionSpec(BaseModel):
    """Immutable protocol rule. One CriteriaSet groups all criteria for one review."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    criterion_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    eligibility_requirement: str = Field(min_length=1)
    missing_evidence_policy: Literal["review", "fail", "pass"]
    exclusion_reason_code: str = Field(min_length=1)
    exclusion_reason_label: str = Field(min_length=1)
    priority: int = Field(ge=0)
    allowed_sources: list[str] | None = None

    @field_validator("allowed_sources", mode="after")
    @classmethod
    def _normalize_allowed_sources(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        if len(v) == 0:
            raise ValueError("allowed_sources, if set, must not be empty")
        normalized: list[str] = []
        seen: set[str] = set()
        for label in v:
            norm = unicodedata.normalize("NFC", label).strip().casefold()
            if not norm:
                raise ValueError("allowed_sources entries must not be empty/whitespace-only")
            if norm in seen:
                raise ValueError(f"allowed_sources contains a duplicate after normalization: {norm!r}")
            seen.add(norm)
            normalized.append(norm)
        return normalized


class CriteriaSet(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    criteria_set_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    criteria: list[CriterionSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_criterion_ids_and_reason_codes(self) -> "CriteriaSet":
        ids = [c.criterion_id for c in self.criteria]
        if len(set(ids)) != len(ids):
            raise ValueError("criterion_id values must be unique within a CriteriaSet")
        codes = [c.exclusion_reason_code for c in self.criteria]
        if len(set(codes)) != len(codes):
            raise ValueError("exclusion_reason_code values must be unique within a CriteriaSet")
        return self


class EvidenceSpan(BaseModel):
    # revalidate_instances="always": the orchestrator calls
    # EvidenceSpan.model_validate() on retriever output defensively
    # (codex_code_review_53f21de.md finding 2) -- Pydantic's own default
    # short-circuits model_validate() on an already-instance input via a
    # bare isinstance check, which would make that defense a no-op for
    # exactly a malformed/model_construct-bypassed component's output.
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    document_id: str
    document_sha256: str
    chunk_id: str
    section: str | None = None
    page: int | None = Field(default=None, ge=1)
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)
    excerpt: str
    stance: Literal["supports", "contradicts", "relevant"]
    retrieval_method: str
    retrieval_query_version: str
    rank: int | None = Field(default=None, ge=1)
    score: float | None = None

    @field_validator("document_sha256", mode="after")
    @classmethod
    def _validate_document_sha256(cls, v: str) -> str:
        return _validate_sha256(v)

    @field_validator("score", mode="after")
    @classmethod
    def _score_must_be_finite(cls, v: float | None) -> float | None:
        if v is not None and not math.isfinite(v):
            raise ValueError("score must be finite")
        return v

    @model_validator(mode="after")
    def _span_order(self) -> "EvidenceSpan":
        if self.start_char is not None and self.end_char is not None:
            if self.end_char < self.start_char:
                raise ValueError("end_char must be >= start_char")
        return self


class CriterionAssessment(BaseModel):
    # See EvidenceSpan's model_config comment -- same rationale, the
    # orchestrator revalidates adjudicator output the same way.
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    criterion_id: str
    criterion_version: str
    status: Literal["satisfied", "violated", "unclear", "not_reported"]
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    rationale: str
    requires_human_review: bool
    model: str
    prompt_template_version: str
    output_schema_version: str


class HumanAdjudication(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: Literal["include", "exclude", "needs_human_review"]
    adjudicator: str = Field(min_length=1)
    adjudicated_at: datetime
    note: str | None = None

    @field_validator("adjudicated_at", mode="after")
    @classmethod
    def _adjudicated_at_utc(cls, v: datetime) -> datetime:
        return _ensure_utc(v)


class EligibilityResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    result_id: str
    revision: int = Field(ge=1)
    supersedes_result_id: str | None = None
    document_id: str
    document_sha256: str
    report_id: str
    study_id: str | None = None
    criteria_set_hash: str
    execution_fingerprint: str
    assessments: list[CriterionAssessment]
    proposed_decision: Literal["include", "exclude", "needs_human_review"]
    final_decision: Literal["include", "exclude", "needs_human_review"]
    exclusion_reason_codes: list[str] = Field(default_factory=list)
    primary_exclusion_reason_code: str | None = None
    human_adjudication: HumanAdjudication | None = None
    created_at: datetime

    @field_validator("document_sha256", "criteria_set_hash", "execution_fingerprint", mode="after")
    @classmethod
    def _validate_hash_fields(cls, v: str) -> str:
        return _validate_sha256(v)

    @field_validator("created_at", mode="after")
    @classmethod
    def _created_at_utc(cls, v: datetime) -> datetime:
        return _ensure_utc(v)

    @model_validator(mode="after")
    def _revision_supersession(self) -> "EligibilityResult":
        if self.revision == 1 and self.supersedes_result_id is not None:
            raise ValueError("revision 1 must not have supersedes_result_id")
        if self.revision > 1 and self.supersedes_result_id is None:
            raise ValueError("revision > 1 must have supersedes_result_id")
        return self

    @model_validator(mode="after")
    def _final_decision_matches_adjudication(self) -> "EligibilityResult":
        if self.human_adjudication is None:
            if self.final_decision != self.proposed_decision:
                raise ValueError(
                    "final_decision must equal proposed_decision when there is no human_adjudication"
                )
        else:
            if self.final_decision != self.human_adjudication.decision:
                raise ValueError(
                    "final_decision must equal human_adjudication.decision when present"
                )
        return self

    @model_validator(mode="after")
    def _primary_exclusion_reason_consistency(self) -> "EligibilityResult":
        if self.primary_exclusion_reason_code is not None:
            if self.final_decision != "exclude":
                raise ValueError(
                    "primary_exclusion_reason_code must be null unless final_decision == 'exclude'"
                )
            if self.primary_exclusion_reason_code not in self.exclusion_reason_codes:
                raise ValueError(
                    "primary_exclusion_reason_code must be one of exclusion_reason_codes"
                )
        return self


class ScreeningItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str = Field(min_length=1)
    report_id: str = Field(min_length=1)
    study_id: str | None = None
    pdf_path: Path
    expected_sha256: str | None = None
    citation: dict[str, Any] = Field(default_factory=dict)

    @field_validator("expected_sha256", mode="after")
    @classmethod
    def _validate_expected_sha256(cls, v: str | None) -> str | None:
        return v if v is None else _validate_sha256(v)

    @field_validator("citation", mode="after")
    @classmethod
    def _citation_json_safe_copy(cls, v: dict[str, Any]) -> dict[str, Any]:
        return _deep_validate_json(v, path="citation")


class ScreeningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    parser_name: str = Field(min_length=1)
    parser_version: str = Field(min_length=1)
    chunker_version: str = Field(min_length=1)
    parser_options: dict[str, Any] = Field(default_factory=dict)
    retriever_name: str = Field(min_length=1)
    retriever_config_version: str = Field(min_length=1)
    retriever_options: dict[str, Any] = Field(default_factory=dict)
    adjudicator_provider: str = Field(min_length=1)
    adjudicator_model: str = Field(min_length=1)
    model_config_id: str = Field(min_length=1)
    model_options: dict[str, Any] = Field(default_factory=dict)
    prompt_template_version: str = Field(min_length=1)
    output_schema_version: str = Field(min_length=1)
    aggregation_policy_version: str = Field(min_length=1)

    cache_mode: Literal["use", "refresh", "off"] = "use"
    duplicate_content_policy: Literal["error", "reuse"] = "error"
    item_error_policy: Literal["continue", "fail_fast"] = "continue"
    max_workers: int = Field(default=1, ge=1)

    @field_validator("parser_options", "retriever_options", "model_options", mode="after")
    @classmethod
    def _options_json_safe_copy(cls, v: dict[str, Any]) -> dict[str, Any]:
        return _deep_validate_json(v, path="options")


class ScreeningRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    api_version: Literal["1.0"] = "1.0"
    run_id: str | None = None
    resume_run_id: str | None = None
    items: list[ScreeningItem] = Field(min_length=1)
    criteria_set: CriteriaSet
    config: ScreeningConfig
    output_dir: Path

    @field_validator("run_id", "resume_run_id", mode="after")
    @classmethod
    def _validate_run_identifier(cls, v: str | None) -> str | None:
        return v if v is None else _validate_safe_token(v)

    @model_validator(mode="after")
    def _at_most_one_run_identifier(self) -> "ScreeningRequest":
        if self.run_id is not None and self.resume_run_id is not None:
            raise ValueError("at most one of run_id/resume_run_id may be supplied")
        return self

    @model_validator(mode="after")
    def _unique_item_identifiers(self) -> "ScreeningRequest":
        document_ids = [item.document_id for item in self.items]
        if len(set(document_ids)) != len(document_ids):
            raise ValueError("document_id values must be unique within a request")
        report_ids = [item.report_id for item in self.items]
        if len(set(report_ids)) != len(report_ids):
            raise ValueError("report_id values must be unique within a request (v1: one document per report)")
        return self


class ProcessingError(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: Literal[
        "input_missing",
        "input_not_file",
        "hash_mismatch",
        "duplicate_input",
        "parse_failed",
        "insufficient_coverage",
        "retrieval_failed",
        "adjudication_failed",
        "persistence_failed",
        "unexpected_error",
    ]
    stage: Literal["preflight", "parse", "retrieve", "adjudicate", "aggregate", "persist"]
    severity: Literal["warning", "fatal"]
    message: str
    retryable: bool
    exception_type: str | None = None


class ExecutionProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document_sha256: str
    criteria_set_hash: str
    execution_fingerprint: str
    cache_status: Literal["hit", "miss", "refresh", "disabled", "content_reuse"]
    source_result_run_id: str | None = None
    started_at: datetime
    completed_at: datetime

    # Usage telemetry aggregated across every criterion this item needed
    # (walking-skeleton follow-up -- operational telemetry only, never
    # part of criterion_cache_key/execution_fingerprint, never consulted
    # by aggregation.py's eligibility decision logic; no currency/cost
    # calculation lives here, since prices change -- persist model +
    # token counts so cost can be computed externally). `usage_available`
    # is True only if EVERY criterion's own usage was known -- either a
    # real provider call's reported tokens, or a cache hit's confirmed
    # zero (the orchestrator synthesizes that record itself, since it
    # KNOWS no provider call happened this run) -- False if any criterion's
    # telemetry was genuinely missing (e.g. a fake/legacy adjudicator),
    # so a caller can never mistake "no telemetry" for "confirmed zero cost".
    usage_available: bool = False
    total_input_tokens: int = Field(default=0, ge=0)
    total_output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    total_provider_calls: int = Field(default=0, ge=0)
    total_retries: int = Field(default=0, ge=0)

    @field_validator("document_sha256", "criteria_set_hash", "execution_fingerprint", mode="after")
    @classmethod
    def _validate_hash_fields(cls, v: str) -> str:
        return _validate_sha256(v)

    @field_validator("started_at", "completed_at", mode="after")
    @classmethod
    def _timestamps_utc(cls, v: datetime) -> datetime:
        return _ensure_utc(v)

    @model_validator(mode="after")
    def _completed_not_before_started(self) -> "ExecutionProvenance":
        if self.completed_at < self.started_at:
            raise ValueError("completed_at must be >= started_at")
        return self


class DocumentCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["complete", "partial", "failed"]
    total_pages: int | None = Field(default=None, ge=0)
    parsed_pages: list[int] = Field(default_factory=list)
    failed_pages: list[int] = Field(default_factory=list)
    ocr_used: bool
    extracted_characters: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _page_list_invariants(self) -> "DocumentCoverage":
        for name, pages in (("parsed_pages", self.parsed_pages), ("failed_pages", self.failed_pages)):
            if any(p < 1 for p in pages):
                raise ValueError(f"{name} entries must be >= 1")
            if len(set(pages)) != len(pages):
                raise ValueError(f"{name} must not contain duplicate page numbers")
        overlap = set(self.parsed_pages) & set(self.failed_pages)
        if overlap:
            raise ValueError(f"parsed_pages and failed_pages must be disjoint, overlap: {sorted(overlap)}")
        if self.total_pages is not None:
            max_page = max(self.parsed_pages + self.failed_pages, default=0)
            if max_page > self.total_pages:
                raise ValueError("parsed_pages/failed_pages must not exceed total_pages")
        return self

    @model_validator(mode="after")
    def _status_consistency(self) -> "DocumentCoverage":
        # State table per codex_code_review_bae78ac.md finding 5:
        #   complete: positive extracted content, no failed pages, and (when
        #             total_pages is known) every page accounted for as parsed.
        #   failed:   zero parsed pages and zero extracted characters.
        #   partial:  the residual case -- no extra constraint beyond the
        #             page-list invariants above.
        if self.status == "complete":
            if self.extracted_characters <= 0:
                raise ValueError("status == 'complete' requires extracted_characters > 0")
            if len(self.failed_pages) != 0:
                raise ValueError("status == 'complete' requires an empty failed_pages")
            if self.total_pages is not None and len(self.parsed_pages) != self.total_pages:
                raise ValueError(
                    "status == 'complete' requires every page to be accounted for as parsed "
                    "when total_pages is known"
                )
        if self.status == "failed":
            if len(self.parsed_pages) != 0:
                raise ValueError("status == 'failed' requires an empty parsed_pages")
            if self.extracted_characters != 0:
                raise ValueError("status == 'failed' requires extracted_characters == 0")
        return self


class ProgressEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    event: Literal[
        "run_started",
        "item_started",
        "item_completed",
        "item_failed",
        "run_completed",
        "run_interrupted",
    ]
    occurred_at: datetime
    completed_items: int = Field(ge=0)
    total_items: int = Field(ge=0)
    input_index: int | None = Field(default=None, ge=0)
    document_id: str | None = None
    report_id: str | None = None
    message: str | None = None

    @field_validator("occurred_at", mode="after")
    @classmethod
    def _occurred_at_utc(cls, v: datetime) -> datetime:
        return _ensure_utc(v)

    @model_validator(mode="after")
    def _completed_not_more_than_total(self) -> "ProgressEvent":
        if self.completed_items > self.total_items:
            raise ValueError("completed_items must be <= total_items")
        return self

    @model_validator(mode="after")
    def _item_events_carry_item_identity(self) -> "ProgressEvent":
        if self.event in ("item_started", "item_completed", "item_failed"):
            if self.input_index is None or self.document_id is None or self.report_id is None:
                raise ValueError(
                    f"event {self.event!r} requires input_index, document_id, and report_id to be set"
                )
        return self


class ScreeningItemResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input_index: int = Field(ge=0)
    document_id: str
    report_id: str
    study_id: str | None = None
    citation: dict[str, Any]
    execution_status: Literal[
        "completed",
        "input_unavailable",
        "duplicate_input",
        "processing_failed",
        "not_processed",
    ]
    eligibility: EligibilityResult | None = None
    coverage: DocumentCoverage | None = None
    provenance: ExecutionProvenance | None = None
    errors: list[ProcessingError] = Field(default_factory=list)

    @field_validator("citation", mode="after")
    @classmethod
    def _citation_json_safe_copy(cls, v: dict[str, Any]) -> dict[str, Any]:
        return _deep_validate_json(v, path="citation")

    @model_validator(mode="after")
    def _eligibility_presence_matches_status(self) -> "ScreeningItemResult":
        has_fatal_error = any(e.severity == "fatal" for e in self.errors)
        if self.execution_status == "completed":
            if self.eligibility is None:
                raise ValueError("execution_status == 'completed' requires eligibility to be set")
            if has_fatal_error:
                raise ValueError("execution_status == 'completed' must not carry a fatal error")
        else:
            if self.eligibility is not None:
                raise ValueError(
                    "eligibility must be absent for non-completed execution_status "
                    "(missing/duplicate/failed/not_processed items are never eligibility exclusions)"
                )
        return self

    @model_validator(mode="after")
    def _completed_identity_and_fingerprint_consistency(self) -> "ScreeningItemResult":
        if self.execution_status != "completed" or self.eligibility is None:
            return self
        elig = self.eligibility
        if elig.document_id != self.document_id:
            raise ValueError("eligibility.document_id must equal the item's document_id")
        if elig.report_id != self.report_id:
            raise ValueError("eligibility.report_id must equal the item's report_id")
        if elig.study_id != self.study_id:
            raise ValueError("eligibility.study_id must equal the item's study_id")
        if self.provenance is None:
            raise ValueError("a completed item must carry provenance")
        if self.provenance.document_sha256 != elig.document_sha256:
            raise ValueError("provenance.document_sha256 must equal eligibility.document_sha256")
        if self.provenance.criteria_set_hash != elig.criteria_set_hash:
            raise ValueError("provenance.criteria_set_hash must equal eligibility.criteria_set_hash")
        if self.provenance.execution_fingerprint != elig.execution_fingerprint:
            raise ValueError(
                "provenance.execution_fingerprint must equal eligibility.execution_fingerprint"
            )
        return self


class RunSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    total: int = Field(ge=0)
    completed: int = Field(ge=0)
    included: int = Field(ge=0)
    excluded: int = Field(ge=0)
    needs_human_review: int = Field(ge=0)
    input_unavailable: int = Field(ge=0)
    duplicate_input: int = Field(ge=0)
    processing_failed: int = Field(ge=0)
    not_processed: int = Field(ge=0)
    cache_hits: int = Field(ge=0)


class RunArtifacts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_result_json: Path
    item_results_jsonl: Path
    ledger_jsonl: Path


class ScreeningRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    api_version: Literal["1.0"]
    schema_version: str
    run_id: str
    run_status: Literal["completed", "completed_with_issues", "failed", "interrupted"]
    criteria_set_id: str
    criteria_set_version: str
    criteria_set_hash: str
    execution_fingerprint: str
    started_at: datetime
    completed_at: datetime | None = None
    items: list[ScreeningItemResult]
    summary: RunSummary
    artifacts: RunArtifacts

    @field_validator("criteria_set_hash", "execution_fingerprint", mode="after")
    @classmethod
    def _validate_hash_fields(cls, v: str) -> str:
        return _validate_sha256(v)

    @field_validator("started_at", mode="after")
    @classmethod
    def _started_at_utc(cls, v: datetime) -> datetime:
        return _ensure_utc(v)

    @field_validator("completed_at", mode="after")
    @classmethod
    def _completed_at_utc(cls, v: datetime | None) -> datetime | None:
        return v if v is None else _ensure_utc(v)

    @model_validator(mode="after")
    def _completion_time_present_and_ordered(self) -> "ScreeningRunResult":
        # Every run_status value describes a terminal (or, for "interrupted", a
        # durable pre-cancellation snapshot) state -- completed_at is always
        # expected, even though the field type stays Optional per the contract.
        if self.completed_at is None:
            raise ValueError("completed_at must be set (every ScreeningRunResult describes a terminal state)")
        if self.completed_at < self.started_at:
            raise ValueError("completed_at must be >= started_at")
        return self

    @model_validator(mode="after")
    def _items_ordered_by_input_index(self) -> "ScreeningRunResult":
        expected = list(range(len(self.items)))
        actual = [item.input_index for item in self.items]
        if actual != expected:
            raise ValueError("items must be ordered by input_index, matching request order")
        return self

    @model_validator(mode="after")
    def _unique_document_report_pairs(self) -> "ScreeningRunResult":
        pairs = [(i.document_id, i.report_id) for i in self.items]
        if len(set(pairs)) != len(pairs):
            raise ValueError("document_id/report_id pairs must be unique within a run result")
        return self

    @model_validator(mode="after")
    def _completed_items_match_run_fingerprints(self) -> "ScreeningRunResult":
        for item in self.items:
            if item.execution_status == "completed" and item.eligibility is not None:
                if item.eligibility.criteria_set_hash != self.criteria_set_hash:
                    raise ValueError(
                        f"item {item.document_id!r}'s eligibility.criteria_set_hash does not match "
                        "the run's criteria_set_hash"
                    )
                if item.eligibility.execution_fingerprint != self.execution_fingerprint:
                    raise ValueError(
                        f"item {item.document_id!r}'s eligibility.execution_fingerprint does not match "
                        "the run's execution_fingerprint"
                    )
        return self

    @model_validator(mode="after")
    def _summary_matches_items(self) -> "ScreeningRunResult":
        expected = RunSummary(
            total=len(self.items),
            completed=sum(1 for i in self.items if i.execution_status == "completed"),
            included=sum(
                1
                for i in self.items
                if i.execution_status == "completed" and i.eligibility.final_decision == "include"
            ),
            excluded=sum(
                1
                for i in self.items
                if i.execution_status == "completed" and i.eligibility.final_decision == "exclude"
            ),
            needs_human_review=sum(
                1
                for i in self.items
                if i.execution_status == "completed"
                and i.eligibility.final_decision == "needs_human_review"
            ),
            input_unavailable=sum(1 for i in self.items if i.execution_status == "input_unavailable"),
            duplicate_input=sum(1 for i in self.items if i.execution_status == "duplicate_input"),
            processing_failed=sum(1 for i in self.items if i.execution_status == "processing_failed"),
            not_processed=sum(1 for i in self.items if i.execution_status == "not_processed"),
            cache_hits=sum(
                1 for i in self.items if i.provenance is not None and i.provenance.cache_status == "hit"
            ),
        )
        if self.summary != expected:
            raise ValueError(f"summary does not match items: expected {expected!r}, got {self.summary!r}")
        return self

    @model_validator(mode="after")
    def _run_status_consistency(self) -> "ScreeningRunResult":
        # Partition per codex_code_review_bae78ac.md finding 3:
        #   completed              => completed == total
        #   completed_with_issues  => 0 < completed < total
        #   failed                 => completed == 0
        #   interrupted            => unconstrained (cancellation snapshot)
        completed = self.summary.completed
        total = self.summary.total
        if self.run_status == "completed" and completed != total:
            raise ValueError("run_status == 'completed' requires completed == total")
        if self.run_status == "completed_with_issues" and not (0 < completed < total):
            raise ValueError("run_status == 'completed_with_issues' requires 0 < completed < total")
        if self.run_status == "failed" and completed != 0:
            raise ValueError("run_status == 'failed' requires completed == 0")
        return self
