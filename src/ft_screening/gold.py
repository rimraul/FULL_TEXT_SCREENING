"""Gold-standard annotation schema and freeze mechanism for
`gold_set_protocol_v9.md`. Not part of `api_contract_v2.md`'s public
schema -- this is evaluation infrastructure, consumed by
`evaluation_harness.py`, never by the production screening pipeline.

Ninth revision. Eight prior adversarial reviews, summarized: rounds 1-4
(`codex_code_review_a36e0df.md` through `codex_code_review_1c40a27.md`)
closed the freeze mechanism's overwritability, blind-manifest leaks,
`raw_pass_count` forgeability, hash coverage, and a PROTOCOL-DOCUMENT-
ITSELF disclosure. Round 5 (`codex_code_review_ade9f48.md`) forced two
METHODOLOGICAL calls no code patch alone could make -- the solo
annotator already label-exposed for 7 of 20 documents (resolved by
reclassifying phase 1 as label-masked/non-blind/descriptive,
`gold_set_protocol_v9.md` §1) -- and started the sealed-worksheet
workflow. Round 6 (`codex_code_review_93bf42d.md`) found that FIRST
workflow only checked document-ID coverage (not criteria), left source
provenance optional/unverified, checked seal EXISTENCE not validity,
and let reruns erase human answers. Round 7 (`codex_code_review_73cf5d9.md`)
found the round-6 fixes had recorded seal identity fields
(`protocol_version`/`criteria_set_hash`/`document_ids`) without ever
CHECKING them, compared only resolved outcomes (not the raw audit trail,
so a forged raw row survived as long as the adjudicated result was
untouched), and retained provenance paths without verified digests
(adjudication files never hashed at all). All fixed via
`verify_worksheet_pass_seal()` (the one place every trust boundary
checks every identity field, now also binding `blind_manifest_hash`/
`sample_manifest_hash`), a reconstruction that returns and compares the
full audit trail and becomes the frozen `GoldSet`'s authoritative
source, and a `WorksheetSourceRecord` per role retaining verified hashes
including adjudication's.

Round 8 (`codex_code_review_6f5dcd0.md`) confirmed that trust chain
APPROVED -- "no new methodological or architectural weakness... does not
justify reconsidering the sealed-worksheet approach" -- and found only
two Medium publication-control gaps in the round-7 set-publish
primitive: its rollback caught `Exception` but the primitive's own race-
conflict path deliberately raises `SystemExit` (a `BaseException`), so
that exact branch skipped rollback and stranded an earlier output from
the same attempt; and rollback deleted by pathname alone, so a
concurrently-edited file was deleted even though its content no longer
matched what this attempt wrote. Separately, the seal script's own
snapshot-pair copy step still used two sequential `shutil.copyfile()`
calls outside the shared set-publish mechanism entirely. Fixed here:
rollback now catches `BaseException`; a new `safe_unlink_if_matches()`
verifies content ownership before ANY rollback deletion; the seal script
now publishes its snapshot pair through the same
`publish_output_set_exclusive()` every other workflow script uses -- see
`gold_set_protocol_v9.md` §0. No annotation happened under v1-v8.

Every frozen artifact (`GoldSet`, `RetrievalOracleArtifact`,
`RetrievalOracleJudgmentSet`, `evaluation_harness.EvaluationResultArtifact`,
`evaluation_harness.EvaluationRunManifest`) is a single JSON file: an
envelope with an explicit `artifact_type`/`envelope_schema_version`
(verified against the supported value BEFORE the payload is even
parsed), an `envelope_header_hash` covering that outer metadata itself, a
semantic `content_hash` (excludes audit timestamps, for stable
cross-artifact binding), and a `full_payload_hash` (covers EVERYTHING,
including timestamps) -- all verified on load, so neither the outer
envelope metadata, the content, nor its own recorded freeze time can be
edited undetected.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ft_screening.hashing import canonical_json_bytes, compute_criteria_set_hash
from ft_screening.schema import CriteriaSet

AnnotationRound = Literal["primary", "second_pass", "agreed", "adjudicated"]
_RAW_ANNOTATION_ROUNDS = {"primary", "second_pass", "adjudicated"}  # "agreed" is synthesized only, never raw input
CriterionStatus = Literal["satisfied", "violated", "unclear", "not_reported"]
FinalDecision = Literal["include", "exclude", "needs_human_review"]

CriterionKey = tuple[str, str]  # (document_id, criterion_id)

ENVELOPE_SCHEMA_VERSION = "1"

# codex_code_review_1c40a27.md finding 3: three fields were all named
# "protocol_version" (blind manifest, sample manifest, freeze_gold_set's
# argument) and nothing compared them -- the committed manifests actually
# disagreed (v3 and v2) while the intended frozen gold set was v4. Only
# one protocol version is ever "current"; freeze_gold_set requires all
# three to equal it exactly, not merely be nonempty strings.
SUPPORTED_PROTOCOL_VERSION = "gold_set_protocol_v9"


def _parse_bool(value: str, *, field_name: str) -> bool:
    normalized = value.strip().upper()
    if normalized == "TRUE":
        return True
    if normalized == "FALSE":
        return False
    raise ValueError(f"{field_name} must be TRUE or FALSE, got {value!r}")


def _parse_json_int_list(value: str, *, field_name: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be a JSON array (e.g. \"[3, 4]\"), got {value!r}") from exc
    if not isinstance(parsed, list) or not all(isinstance(x, int) and not isinstance(x, bool) for x in parsed):
        raise ValueError(f"{field_name} must be a JSON array of integers, got {value!r}")
    return tuple(parsed)


def _parse_json_str_list(value: str, *, field_name: str) -> tuple[str, ...]:
    if not value.strip():
        return ()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be a JSON array of strings, got {value!r}") from exc
    if not isinstance(parsed, list) or not all(isinstance(x, str) for x in parsed):
        raise ValueError(f"{field_name} must be a JSON array of strings, got {value!r}")
    return tuple(parsed)


def hash_pdf_file(path: Path) -> str:
    """Streaming sha256 of the ACTUAL file bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def real_pdf_page_count(path: Path) -> int:
    """The REAL page count of the actual PDF file, via PyMuPDF -- used at
    freeze time to verify a manifest's claimed page_count against reality
    (codex_code_review_772916e.md finding 3: the claimed value was
    previously trusted outright, so a document with fabricated byte
    content and a false page-count claim froze successfully). Deferred
    import -- matches this project's established convention for
    heavier/optional-feeling dependencies (see adjudication.py's deferred
    `openai` import) so importing gold.py itself stays cheap for callers
    that never freeze anything."""
    import fitz  # PyMuPDF

    with fitz.open(path) as doc:
        return doc.page_count


# ---------------------------------------------------------------------------
# Strict manifest schemas -- codex_code_review_772916e.md finding 3: the
# blind manifest was hashed but never actually PARSED/validated at freeze
# time, so it could silently diverge from the sealed sample manifest.
# ---------------------------------------------------------------------------


class BlindManifestEntry(BaseModel):
    """Exactly what the annotator's navigation file may contain -- no
    stratum, no reason code, no dual-review membership. `extra="forbid"`
    means an injected judgmental field is a hard validation error at load
    time, not a silent pass-through."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str = Field(min_length=1)
    pdf_path: str = Field(min_length=1)
    document_page_count: int = Field(ge=1)


class BlindManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol_version: str = Field(min_length=1)
    documents: tuple[BlindManifestEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> "BlindManifest":
        ids = [d.document_id for d in self.documents]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate document_id in blind manifest")
        return self


class SampleManifestEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str = Field(min_length=1)
    stratum: str = Field(min_length=1)
    iasia_reason_code: str | None = None
    pdf_path: str = Field(min_length=1)
    pdf_sha256: str = Field(min_length=64, max_length=64)
    page_count: int = Field(ge=1)
    title: str = ""


class SampleManifest(BaseModel):
    """The SEALED sampling manifest -- must never be opened during
    annotation. `extra="forbid"` with explicit optional descriptive
    fields (rather than `extra="allow"`) so an accidental or injected
    extra field is still caught, not silently tolerated."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol_version: str = Field(min_length=1)
    source_corpus: str = ""
    excluded_as_already_used_by_walking_skeleton: tuple[str, ...] = Field(default_factory=tuple)
    dual_review_subset: tuple[str, ...] = Field(default_factory=tuple)
    documents: tuple[SampleManifestEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids_and_dual_review_known(self) -> "SampleManifest":
        ids = {d.document_id for d in self.documents}
        if len(ids) != len(self.documents):
            raise ValueError("duplicate document_id in sample manifest")
        unknown_dual = set(self.dual_review_subset) - ids
        if unknown_dual:
            raise ValueError(f"dual_review_subset references unknown document_ids: {sorted(unknown_dual)}")
        return self


def _validate_manifests_cross_bound(blind: BlindManifest, sample: SampleManifest) -> None:
    """codex_code_review_772916e.md finding 3's exact repro: a blind
    manifest pointing a document_id at a completely unrelated pdf_path,
    or carrying a different page count, previously froze successfully
    because nothing ever compared the two manifests against each other."""
    if blind.protocol_version != sample.protocol_version:
        raise ValueError(
            f"blind_manifest.protocol_version={blind.protocol_version!r} does not match "
            f"sample_manifest.protocol_version={sample.protocol_version!r}"
        )
    blind_by_id = {d.document_id: d for d in blind.documents}
    sample_by_id = {d.document_id: d for d in sample.documents}
    if set(blind_by_id) != set(sample_by_id):
        raise ValueError(
            "blind_manifest and sample_manifest document_id sets differ "
            f"(blind-only={sorted(set(blind_by_id) - set(sample_by_id))}, "
            f"sample-only={sorted(set(sample_by_id) - set(blind_by_id))})"
        )
    for doc_id, blind_entry in blind_by_id.items():
        sample_entry = sample_by_id[doc_id]
        if blind_entry.pdf_path != sample_entry.pdf_path:
            raise ValueError(
                f"document_id={doc_id!r}: blind_manifest pdf_path={blind_entry.pdf_path!r} does not match "
                f"sample_manifest pdf_path={sample_entry.pdf_path!r}"
            )
        if blind_entry.document_page_count != sample_entry.page_count:
            raise ValueError(
                f"document_id={doc_id!r}: blind_manifest document_page_count={blind_entry.document_page_count} "
                f"does not match sample_manifest page_count={sample_entry.page_count}"
            )


# ---------------------------------------------------------------------------
# Shared exclusive-write / mandatory-verify freeze mechanism -- a single
# JSON envelope with BOTH a semantic hash (excludes audit timestamps, for
# stable cross-artifact binding) and a full-payload hash (covers
# everything, including timestamps -- codex_code_review_772916e.md
# finding 5: content_hash alone let frozen_at/recorded_at be edited
# undetected).
# ---------------------------------------------------------------------------


def _envelope_header_hash(*, envelope_schema_version: str, artifact_type: str, content_hash: str, full_payload_hash: str) -> str:
    """Hashes the envelope's own OUTER metadata -- codex_code_review_1c40a27.md
    finding 5: `envelope_schema_version` was written but never verified on
    load, and neither hash covered it, so changing it (e.g. "1" -> "999")
    went undetected by both hashes. This header hash is computed from
    fields that are themselves fixed at freeze time, so tampering with any
    one of them (without also recomputing this hash, which requires
    knowing nothing secret -- it's just a stronger check, not a secret)
    is caught on load."""
    header = {
        "envelope_schema_version": envelope_schema_version,
        "artifact_type": artifact_type,
        "content_hash": content_hash,
        "full_payload_hash": full_payload_hash,
    }
    return hashlib.sha256(canonical_json_bytes(header)).hexdigest()


def _freeze_frozen_model(model: BaseModel, *, output_dir: Path, version_label: str, artifact_type: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{version_label}.json"
    payload = json.loads(model.model_dump_json())  # type: ignore[attr-defined]
    content_hash = model.content_hash()  # type: ignore[attr-defined]
    full_payload_hash = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    envelope = {
        "envelope_schema_version": ENVELOPE_SCHEMA_VERSION,
        "artifact_type": artifact_type,
        "content_hash": content_hash,
        "full_payload_hash": full_payload_hash,
        "envelope_header_hash": _envelope_header_hash(
            envelope_schema_version=ENVELOPE_SCHEMA_VERSION, artifact_type=artifact_type,
            content_hash=content_hash, full_payload_hash=full_payload_hash,
        ),
        "payload": payload,
    }
    tmp_path = output_dir / f".{version_label}.{uuid.uuid4().hex}.tmp"
    tmp_path.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
    try:
        os.link(tmp_path, json_path)
    except FileExistsError:
        raise FileExistsError(
            f"refusing to overwrite an existing frozen artifact: {json_path} already exists -- "
            "freeze under a new version_label instead of overwriting a previously frozen one"
        ) from None
    finally:
        tmp_path.unlink(missing_ok=True)


_ENVELOPE_KEYS = {
    "envelope_schema_version", "artifact_type", "content_hash", "full_payload_hash",
    "envelope_header_hash", "payload",
}


def _load_frozen_model(cls, json_path: Path, *, expected_hash: str | None = None, expected_artifact_type: str | None = None):
    envelope = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(envelope, dict) or set(envelope.keys()) != _ENVELOPE_KEYS:
        raise ValueError(f"{json_path} is not a valid frozen-artifact envelope (expected exactly {sorted(_ENVELOPE_KEYS)} keys)")
    if envelope["envelope_schema_version"] != ENVELOPE_SCHEMA_VERSION:
        raise ValueError(
            f"{json_path} has envelope_schema_version={envelope['envelope_schema_version']!r}, but this code "
            f"only supports {ENVELOPE_SCHEMA_VERSION!r} -- rejected before parsing the payload"
        )
    actual_header_hash = _envelope_header_hash(
        envelope_schema_version=envelope["envelope_schema_version"], artifact_type=envelope["artifact_type"],
        content_hash=envelope["content_hash"], full_payload_hash=envelope["full_payload_hash"],
    )
    if actual_header_hash != envelope["envelope_header_hash"]:
        raise ValueError(
            f"{json_path}'s outer envelope metadata (schema version / artifact type / hashes) does not match "
            "its own envelope_header_hash -- it may have been edited after freezing"
        )
    if expected_artifact_type is not None and envelope["artifact_type"] != expected_artifact_type:
        raise ValueError(f"{json_path} has artifact_type={envelope['artifact_type']!r}, expected {expected_artifact_type!r}")

    obj = cls.model_validate(envelope["payload"])
    actual_content_hash = obj.content_hash()
    if actual_content_hash != envelope["content_hash"]:
        raise ValueError(
            f"{json_path}'s payload does not match its own embedded content_hash "
            f"(embedded={envelope['content_hash']!r}, actual={actual_content_hash!r}) -- it may have been edited after freezing"
        )
    actual_full_hash = hashlib.sha256(canonical_json_bytes(envelope["payload"])).hexdigest()
    if actual_full_hash != envelope["full_payload_hash"]:
        raise ValueError(
            f"{json_path}'s payload (including its audit timestamp) does not match its own full_payload_hash "
            "-- it may have been edited after freezing, even if content_hash alone would not have detected it"
        )
    if expected_hash is not None and actual_content_hash != expected_hash:
        raise ValueError(
            f"{json_path} does not match caller-supplied expected_hash={expected_hash!r} (actual={actual_content_hash!r})"
        )
    return obj


# ---------------------------------------------------------------------------
# Phase 1: gold criterion/document truth
# ---------------------------------------------------------------------------


class GoldCriterionAnnotation(BaseModel):
    """One row of the phase-1 (blind) annotation worksheet
    (gold_set_protocol_v4.md §5), validated against the full coherence
    table in §4."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str = Field(min_length=1)
    criterion_id: str = Field(min_length=1)
    document_page_count: int = Field(ge=1)
    gold_status: CriterionStatus
    supporting_pages: tuple[int, ...] = Field(default_factory=tuple)
    supporting_passages: tuple[str, ...] = Field(default_factory=tuple)
    contradicting_pages: tuple[int, ...] = Field(default_factory=tuple)
    contradicting_passages: tuple[str, ...] = Field(default_factory=tuple)
    evidence_absent: bool
    evidence_ambiguous: bool
    evidence_conflicting: bool
    human_review_required: bool
    annotator_id: str = Field(min_length=1)
    annotation_round: AnnotationRound
    notes: str = ""

    @field_validator("supporting_pages", "contradicting_pages", mode="after")
    @classmethod
    def _pages_positive(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        for page in v:
            if page < 1:
                raise ValueError(f"page numbers must be >= 1, got {page}")
        return v

    @field_validator("supporting_passages", "contradicting_passages", mode="after")
    @classmethod
    def _passages_nonblank(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        for passage in v:
            if not passage.strip():
                raise ValueError("passages must be nonblank verbatim quotations")
        return v

    @model_validator(mode="after")
    def _pages_within_document(self) -> "GoldCriterionAnnotation":
        for page in (*self.supporting_pages, *self.contradicting_pages):
            if page > self.document_page_count:
                raise ValueError(
                    f"page {page} exceeds document_page_count={self.document_page_count} for "
                    f"document_id={self.document_id!r}"
                )
        return self

    @model_validator(mode="after")
    def _passages_match_pages(self) -> "GoldCriterionAnnotation":
        if len(self.supporting_pages) != len(self.supporting_passages):
            raise ValueError("supporting_pages and supporting_passages must have the same length (one passage per page)")
        if len(self.contradicting_pages) != len(self.contradicting_passages):
            raise ValueError("contradicting_pages and contradicting_passages must have the same length")
        return self

    @model_validator(mode="after")
    def _evidence_absent_consistency(self) -> "GoldCriterionAnnotation":
        has_evidence = bool(self.supporting_pages) or bool(self.contradicting_pages)
        if self.evidence_absent and has_evidence:
            raise ValueError("evidence_absent=True but supporting/contradicting pages were also recorded")
        if not self.evidence_absent and not has_evidence:
            raise ValueError("evidence_absent=False requires at least one supporting or contradicting page")
        return self

    @model_validator(mode="after")
    def _conflicting_biconditional(self) -> "GoldCriterionAnnotation":
        has_both_directions = bool(self.supporting_pages) and bool(self.contradicting_pages)
        if has_both_directions and not self.evidence_conflicting:
            raise ValueError("both supporting AND contradicting pages are present -- evidence_conflicting must be True")
        if self.evidence_conflicting and not has_both_directions:
            raise ValueError("evidence_conflicting=True requires both supporting AND contradicting pages")
        return self

    @model_validator(mode="after")
    def _conflicting_forces_unclear_and_review(self) -> "GoldCriterionAnnotation":
        if self.evidence_conflicting:
            if self.gold_status != "unclear":
                raise ValueError('evidence_conflicting=True requires gold_status="unclear"')
            if not self.human_review_required:
                raise ValueError("evidence_conflicting=True requires human_review_required=True")
        return self

    @model_validator(mode="after")
    def _ambiguous_semantics(self) -> "GoldCriterionAnnotation":
        if self.evidence_ambiguous:
            if self.evidence_absent:
                raise ValueError("evidence_ambiguous=True requires evidence_absent=False -- ambiguous evidence is still evidence")
            if self.gold_status != "unclear":
                raise ValueError('evidence_ambiguous=True requires gold_status="unclear"')
            if not self.human_review_required:
                raise ValueError("evidence_ambiguous=True requires human_review_required=True")
        return self

    @model_validator(mode="after")
    def _decisive_status_requires_matching_evidence(self) -> "GoldCriterionAnnotation":
        if self.gold_status == "satisfied" and not self.supporting_pages:
            raise ValueError('gold_status="satisfied" requires at least one supporting page')
        if self.gold_status == "violated" and not self.contradicting_pages:
            raise ValueError('gold_status="violated" requires at least one contradicting page')
        return self

    @classmethod
    def from_csv_row(cls, row: dict[str, str]) -> "GoldCriterionAnnotation":
        round_value = row["annotation_round"]
        if round_value not in _RAW_ANNOTATION_ROUNDS:
            raise ValueError(
                f"annotation_round={round_value!r} is not a valid RAW worksheet value -- only "
                f"{sorted(_RAW_ANNOTATION_ROUNDS)} may appear in source CSV rows; 'agreed' is a synthesized "
                "state produced only by collapsing a genuine primary+second_pass pair"
            )
        return cls(
            document_id=row["document_id"],
            criterion_id=row["criterion_id"],
            document_page_count=int(row["document_page_count"]),
            gold_status=row["gold_status"],  # type: ignore[arg-type]
            supporting_pages=_parse_json_int_list(row.get("supporting_pages", ""), field_name="supporting_pages"),
            supporting_passages=_parse_json_str_list(row.get("supporting_passages", ""), field_name="supporting_passages"),
            contradicting_pages=_parse_json_int_list(row.get("contradicting_pages", ""), field_name="contradicting_pages"),
            contradicting_passages=_parse_json_str_list(row.get("contradicting_passages", ""), field_name="contradicting_passages"),
            evidence_absent=_parse_bool(row["evidence_absent"], field_name="evidence_absent"),
            evidence_ambiguous=_parse_bool(row["evidence_ambiguous"], field_name="evidence_ambiguous"),
            evidence_conflicting=_parse_bool(row["evidence_conflicting"], field_name="evidence_conflicting"),
            human_review_required=_parse_bool(row["human_review_required"], field_name="human_review_required"),
            annotator_id=row["annotator_id"],
            annotation_round=round_value,  # type: ignore[arg-type]
            notes=row.get("notes", ""),
        )


class GoldDocumentAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str = Field(min_length=1)
    document_page_count: int = Field(ge=1)
    gold_final_decision: FinalDecision
    gold_primary_exclusion_reason_code: str | None = None
    aggregation_override_rationale: str | None = None
    annotator_id: str = Field(min_length=1)
    annotation_round: AnnotationRound
    notes: str = ""

    @model_validator(mode="after")
    def _exclusion_reason_consistency(self) -> "GoldDocumentAnnotation":
        if self.gold_final_decision == "exclude" and not self.gold_primary_exclusion_reason_code:
            raise ValueError('gold_final_decision="exclude" requires gold_primary_exclusion_reason_code')
        if self.gold_final_decision != "exclude" and self.gold_primary_exclusion_reason_code:
            raise ValueError('gold_primary_exclusion_reason_code must be null unless gold_final_decision=="exclude"')
        return self

    @classmethod
    def from_csv_row(cls, row: dict[str, str]) -> "GoldDocumentAnnotation":
        round_value = row["annotation_round"]
        if round_value not in _RAW_ANNOTATION_ROUNDS:
            raise ValueError(
                f"annotation_round={round_value!r} is not a valid RAW worksheet value -- only "
                f"{sorted(_RAW_ANNOTATION_ROUNDS)} may appear in source CSV rows"
            )
        reason = row.get("gold_primary_exclusion_reason_code", "").strip()
        rationale = row.get("aggregation_override_rationale", "").strip()
        return cls(
            document_id=row["document_id"],
            document_page_count=int(row["document_page_count"]),
            gold_final_decision=row["gold_final_decision"],  # type: ignore[arg-type]
            gold_primary_exclusion_reason_code=reason or None,
            aggregation_override_rationale=rationale or None,
            annotator_id=row["annotator_id"],
            annotation_round=round_value,  # type: ignore[arg-type]
            notes=row.get("notes", ""),
        )


def _rows_content_equal(a: BaseModel, b: BaseModel, *, fields: list[str]) -> bool:
    return all(getattr(a, f) == getattr(b, f) for f in fields)


_CRITERION_CONTENT_FIELDS = [
    "gold_status", "supporting_pages", "supporting_passages", "contradicting_pages",
    "contradicting_passages", "evidence_absent", "evidence_ambiguous", "evidence_conflicting",
    "human_review_required", "document_page_count",
]
_DOCUMENT_CONTENT_FIELDS = [
    "document_page_count", "gold_final_decision", "gold_primary_exclusion_reason_code", "aggregation_override_rationale",
]


def criterion_rows_agree(a: GoldCriterionAnnotation, b: GoldCriterionAnnotation) -> bool:
    """Public wrapper around the exact agreement check `gold.py` itself
    uses to collapse dual-review rows -- exposed so an adjudication
    worksheet generator can compute the SAME disagreeing-keys set the
    freeze path will later re-derive, rather than reimplementing (and
    risking drifting from) the equality rule."""
    return _rows_content_equal(a, b, fields=_CRITERION_CONTENT_FIELDS)


def document_rows_agree(a: GoldDocumentAnnotation, b: GoldDocumentAnnotation) -> bool:
    return _rows_content_equal(a, b, fields=_DOCUMENT_CONTENT_FIELDS)


# ---------------------------------------------------------------------------
# Dual-review audit trail -- codex_code_review_772916e.md finding 2: the
# frozen artifact must retain the RAW primary/second_pass(/adjudicated)
# rows for every dual-review key, so "a genuine second pass happened" is
# structurally provable from the artifact itself, not merely asserted by
# a caller-settable integer.
# ---------------------------------------------------------------------------


class CriterionDualReviewAudit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str = Field(min_length=1)
    criterion_id: str = Field(min_length=1)
    primary: GoldCriterionAnnotation
    second_pass: GoldCriterionAnnotation
    adjudicated: GoldCriterionAnnotation | None = None

    @model_validator(mode="after")
    def _rounds_and_keys_correct(self) -> "CriterionDualReviewAudit":
        if self.primary.annotation_round != "primary":
            raise ValueError("audit.primary must have annotation_round='primary'")
        if self.second_pass.annotation_round != "second_pass":
            raise ValueError("audit.second_pass must have annotation_round='second_pass'")
        if self.adjudicated is not None and self.adjudicated.annotation_round != "adjudicated":
            raise ValueError("audit.adjudicated must have annotation_round='adjudicated'")
        for row in (self.primary, self.second_pass, *([self.adjudicated] if self.adjudicated else [])):
            if row.document_id != self.document_id or row.criterion_id != self.criterion_id:
                raise ValueError("every audit row must share the audit's own document_id/criterion_id")
        return self

    @model_validator(mode="after")
    def _adjudication_only_resolves_disagreement(self) -> "CriterionDualReviewAudit":
        """codex_code_review_1c40a27.md finding 2: an adjudicated row was
        previously accepted even when primary and second_pass already
        agreed, letting a third row silently override unanimous
        independent review. Enforced here (not just in the collapse
        function) so a directly-constructed audit is independently
        valid, never only valid when built by `_collapse_criterion_rows`."""
        agree = _rows_content_equal(self.primary, self.second_pass, fields=_CRITERION_CONTENT_FIELDS)
        if agree and self.adjudicated is not None:
            raise ValueError(
                "primary and second_pass already agree -- an 'adjudicated' row may only resolve a genuine "
                "disagreement, not override unanimous independent review"
            )
        if not agree and self.adjudicated is None:
            raise ValueError("primary and second_pass disagree -- an 'adjudicated' row is required to resolve them")
        return self

    def resolves_to(self) -> GoldCriterionAnnotation:
        """The authoritative annotation this audit trail must collapse
        to -- an adjudicated row if the two raw passes disagreed, else
        the primary row re-labeled 'agreed'."""
        if self.adjudicated is not None:
            return self.adjudicated
        return self.primary.model_copy(update={"annotation_round": "agreed"})


class DocumentDualReviewAudit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str = Field(min_length=1)
    primary: GoldDocumentAnnotation
    second_pass: GoldDocumentAnnotation
    adjudicated: GoldDocumentAnnotation | None = None

    @model_validator(mode="after")
    def _rounds_and_keys_correct(self) -> "DocumentDualReviewAudit":
        if self.primary.annotation_round != "primary":
            raise ValueError("audit.primary must have annotation_round='primary'")
        if self.second_pass.annotation_round != "second_pass":
            raise ValueError("audit.second_pass must have annotation_round='second_pass'")
        if self.adjudicated is not None and self.adjudicated.annotation_round != "adjudicated":
            raise ValueError("audit.adjudicated must have annotation_round='adjudicated'")
        for row in (self.primary, self.second_pass, *([self.adjudicated] if self.adjudicated else [])):
            if row.document_id != self.document_id:
                raise ValueError("every audit row must share the audit's own document_id")
        return self

    @model_validator(mode="after")
    def _adjudication_only_resolves_disagreement(self) -> "DocumentDualReviewAudit":
        agree = _rows_content_equal(self.primary, self.second_pass, fields=_DOCUMENT_CONTENT_FIELDS)
        if agree and self.adjudicated is not None:
            raise ValueError(
                "primary and second_pass already agree -- an 'adjudicated' row may only resolve a genuine "
                "disagreement, not override unanimous independent review"
            )
        if not agree and self.adjudicated is None:
            raise ValueError("primary and second_pass disagree -- an 'adjudicated' row is required to resolve them")
        return self

    def resolves_to(self) -> GoldDocumentAnnotation:
        if self.adjudicated is not None:
            return self.adjudicated
        return self.primary.model_copy(update={"annotation_round": "agreed"})


def _collapse_criterion_rows(
    rows: list[GoldCriterionAnnotation], *, dual_review_ids: set[str]
) -> tuple[list[GoldCriterionAnnotation], list[CriterionDualReviewAudit]]:
    by_key: dict[CriterionKey, list[GoldCriterionAnnotation]] = {}
    for row in rows:
        by_key.setdefault((row.document_id, row.criterion_id), []).append(row)

    selected: list[GoldCriterionAnnotation] = []
    audits: list[CriterionDualReviewAudit] = []
    for key, group in by_key.items():
        doc_id, criterion_id = key
        is_dual = doc_id in dual_review_ids
        primary = [r for r in group if r.annotation_round == "primary"]
        second = [r for r in group if r.annotation_round == "second_pass"]
        adjudicated = [r for r in group if r.annotation_round == "adjudicated"]

        if is_dual:
            if len(primary) != 1 or len(second) != 1:
                raise ValueError(
                    f"{key} is in the dual-review subset and requires exactly one raw 'primary' row and one "
                    f"raw 'second_pass' row (found primary={len(primary)}, second_pass={len(second)})"
                )
            if len(adjudicated) > 1:
                raise ValueError(f"more than one 'adjudicated' row for {key}")
            agree = _rows_content_equal(primary[0], second[0], fields=_CRITERION_CONTENT_FIELDS)
            if not agree and not adjudicated:
                raise ValueError(f"{key} has disagreeing primary/second_pass rows and no 'adjudicated' row resolving them")
            # Agreement-vs-adjudication ordering is checked here AND re-enforced inside
            # CriterionDualReviewAudit's own validator (codex_code_review_1c40a27.md finding 2:
            # an adjudicated row was previously accepted before checking whether the two raw
            # passes even disagreed, letting it silently override unanimous independent review).
            audit = CriterionDualReviewAudit(
                document_id=doc_id, criterion_id=criterion_id, primary=primary[0], second_pass=second[0],
                adjudicated=adjudicated[0] if adjudicated else None,
            )
            selected.append(audit.resolves_to())
            audits.append(audit)
            continue

        if adjudicated:
            if len(adjudicated) > 1:
                raise ValueError(f"more than one 'adjudicated' row for {key}")
            selected.append(adjudicated[0])
            continue
        if len(group) == 1:
            selected.append(group[0])
            continue
        if len(primary) == 1 and len(second) == 1 and _rows_content_equal(primary[0], second[0], fields=_CRITERION_CONTENT_FIELDS):
            selected.append(primary[0].model_copy(update={"annotation_round": "agreed"}))
            continue
        rounds = [r.annotation_round for r in group]
        raise ValueError(f"{key} has {len(group)} rows ({rounds}) with no 'adjudicated' row resolving a disagreement")
    return selected, audits


def _collapse_document_rows(
    rows: list[GoldDocumentAnnotation], *, dual_review_ids: set[str]
) -> tuple[list[GoldDocumentAnnotation], list[DocumentDualReviewAudit]]:
    by_key: dict[str, list[GoldDocumentAnnotation]] = {}
    for row in rows:
        by_key.setdefault(row.document_id, []).append(row)

    selected: list[GoldDocumentAnnotation] = []
    audits: list[DocumentDualReviewAudit] = []
    for doc_id, group in by_key.items():
        is_dual = doc_id in dual_review_ids
        primary = [r for r in group if r.annotation_round == "primary"]
        second = [r for r in group if r.annotation_round == "second_pass"]
        adjudicated = [r for r in group if r.annotation_round == "adjudicated"]

        if is_dual:
            if len(primary) != 1 or len(second) != 1:
                raise ValueError(
                    f"document_id={doc_id!r} is in the dual-review subset and requires exactly one raw "
                    f"'primary' row and one raw 'second_pass' row (found primary={len(primary)}, second_pass={len(second)})"
                )
            if len(adjudicated) > 1:
                raise ValueError(f"more than one 'adjudicated' row for document_id={doc_id!r}")
            agree = _rows_content_equal(primary[0], second[0], fields=_DOCUMENT_CONTENT_FIELDS)
            if not agree and not adjudicated:
                raise ValueError(f"document_id={doc_id!r} has disagreeing primary/second_pass rows and no 'adjudicated' row resolving them")
            audit = DocumentDualReviewAudit(
                document_id=doc_id, primary=primary[0], second_pass=second[0],
                adjudicated=adjudicated[0] if adjudicated else None,
            )
            selected.append(audit.resolves_to())
            audits.append(audit)
            continue

        if adjudicated:
            if len(adjudicated) > 1:
                raise ValueError(f"more than one 'adjudicated' row for document_id={doc_id!r}")
            selected.append(adjudicated[0])
            continue
        if len(group) == 1:
            selected.append(group[0])
            continue
        if len(primary) == 1 and len(second) == 1 and _rows_content_equal(primary[0], second[0], fields=_DOCUMENT_CONTENT_FIELDS):
            selected.append(primary[0].model_copy(update={"annotation_round": "agreed"}))
            continue
        rounds = [r.annotation_round for r in group]
        raise ValueError(f"document_id={doc_id!r} has {len(group)} rows ({rounds}) with no 'adjudicated' row resolving a disagreement")
    return selected, audits


def load_gold_annotations_from_csv_multi(
    criterion_csv_paths: list[Path], document_csv_paths: list[Path], *, dual_review_document_ids: set[str] = frozenset()
) -> tuple[list[GoldCriterionAnnotation], list[GoldDocumentAnnotation], list[CriterionDualReviewAudit], list[DocumentDualReviewAudit]]:
    """Like `load_gold_annotations_from_csv`, but reads and concatenates
    rows from MULTIPLE criterion/document CSVs before collapsing --
    lets the trusted freeze path reconstruct the merged annotation set
    directly from several separately-sealed source files (primary,
    second_pass, adjudication) without needing them pre-merged into one
    file on disk first (codex_code_review_93bf42d.md finding 2)."""
    criterion_rows: list[GoldCriterionAnnotation] = []
    for path in criterion_csv_paths:
        with open(path, newline="", encoding="utf-8") as f:
            criterion_rows += [GoldCriterionAnnotation.from_csv_row(row) for row in csv.DictReader(f)]
    document_rows: list[GoldDocumentAnnotation] = []
    for path in document_csv_paths:
        with open(path, newline="", encoding="utf-8") as f:
            document_rows += [GoldDocumentAnnotation.from_csv_row(row) for row in csv.DictReader(f)]

    criteria, criterion_audits = _collapse_criterion_rows(criterion_rows, dual_review_ids=dual_review_document_ids)
    documents, document_audits = _collapse_document_rows(document_rows, dual_review_ids=dual_review_document_ids)
    return criteria, documents, criterion_audits, document_audits


def load_gold_annotations_from_csv(
    criterion_csv_path: Path, document_csv_path: Path, *, dual_review_document_ids: set[str] = frozenset()
) -> tuple[list[GoldCriterionAnnotation], list[GoldDocumentAnnotation], list[CriterionDualReviewAudit], list[DocumentDualReviewAudit]]:
    """Reads the two phase-1 worksheets, collapses them into one
    authoritative annotation per key, and returns the dual-review audit
    trail alongside. `freeze_gold_set()` calls this directly from raw CSV
    paths -- there is no supported way to hand it pre-collapsed
    annotations and skip this process (codex_code_review_772916e.md
    finding 2)."""
    return load_gold_annotations_from_csv_multi(
        [criterion_csv_path], [document_csv_path], dual_review_document_ids=dual_review_document_ids
    )


# ---------------------------------------------------------------------------
# Worksheet-pass sealing -- codex_code_review_ade9f48.md finding 2: no
# blank second-pass worksheet, primary-pass seal, or generation/merge step
# existed, so the only way to add a "second_pass" row was to reopen and
# edit the ALREADY-ANSWERED primary CSV, exposing the primary pass's own
# answers to the same annotator during the supposedly-independent second
# pass. `seal_worksheet_pass()` is the enforcement point: a pass's CSVs
# may only be sealed once every row is genuinely that pass's own round
# label (never a mix) and covers EXACTLY its expected document x criterion
# key set (codex_code_review_93bf42d.md finding 1 -- the original version
# only checked the document_id SET, so a pass missing an entire criterion
# for every document, or with a duplicated key, both sealed as
# "complete"). The real human workflow (see gold_set_protocol_v7.md §6)
# is: seal the primary pass -> ONLY THEN reveal the dual-review subset
# and generate blank, independently-shuffled second-pass worksheets ->
# seal those -> generate a disagreement-only adjudication worksheet from
# the two sealed passes -> freeze, which independently reconstructs the
# merged annotation set from the verified sealed sources itself
# (`WorksheetSourceProvenance`, below) rather than trusting a separately
# produced "merged" file.
# ---------------------------------------------------------------------------


class WorksheetPassSeal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pass_name: Literal["primary", "second_pass"]
    protocol_version: str = Field(min_length=1)
    criteria_set_hash: str = Field(min_length=64, max_length=64)
    blind_manifest_hash: str = Field(min_length=1)
    sample_manifest_hash: str = Field(min_length=1)
    document_ids: tuple[str, ...] = Field(min_length=1)
    criterion_csv_sha256: str = Field(min_length=64, max_length=64)
    document_csv_sha256: str = Field(min_length=64, max_length=64)
    sealed_at: str = Field(min_length=1)

    def content_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"sealed_at"})
        return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def seal_worksheet_pass(
    *,
    pass_name: Literal["primary", "second_pass"],
    protocol_version: str,
    criteria_set: CriteriaSet,
    blind_manifest_path: Path,
    sample_manifest_path: Path,
    criterion_csv_path: Path,
    document_csv_path: Path,
    expected_document_ids: set[str],
    output_dir: Path,
    version_label: str,
) -> WorksheetPassSeal:
    """Validates a worksheet pass is genuinely COMPLETE and internally
    homogeneous before hashing and sealing it:

    - every row's `annotation_round` equals `pass_name` (a mixed file is
      exactly the bypass this workflow exists to prevent: appending
      second-pass answers into an already-answered primary file);
    - EXACTLY one document row per `expected_document_ids` (no missing,
      no duplicate document rows);
    - EXACTLY one criterion row per `(document_id, criterion_id)` in the
      full Cartesian product of `expected_document_ids` x
      `criteria_set`'s criterion IDs -- no missing key, no extra key, no
      duplicate key (codex_code_review_93bf42d.md finding 1's exact
      repro: a pass with one row per document but a whole criterion
      MISSING for every document, and a pass with the same key
      duplicated, both previously sealed as "complete").

    `criteria_set_hash`, `protocol_version`, and now `blind_manifest_hash`/
    `sample_manifest_hash` (hashed from the actual manifest bytes, not
    merely recorded from caller-supplied strings) are bound into the
    resulting seal so "complete" has a durable, unambiguous meaning tied
    to a specific criteria identity AND a specific manifest generation --
    codex_code_review_73cf5d9.md finding 1: these identity fields
    previously existed on the seal but nothing ever CHECKED them at a
    trust boundary, so a seal for the wrong protocol, wrong criteria set,
    or a stale manifest (mutated after sealing) verified successfully.
    `protocol_version` is rejected here, at seal time, if unsupported --
    not only later at freeze time.

    The caller decides what `criterion_csv_path`/`document_csv_path`
    point at; the real workflow seals an immutable COPY of the working
    worksheet, not the working file itself, so later edits to the
    original can't retroactively invalidate a seal silently."""
    if protocol_version != SUPPORTED_PROTOCOL_VERSION:
        raise ValueError(
            f"protocol_version={protocol_version!r} is not the currently supported protocol "
            f"({SUPPORTED_PROTOCOL_VERSION!r}) -- seal_worksheet_pass does not accept an arbitrary "
            "nonempty string here"
        )
    with open(criterion_csv_path, newline="", encoding="utf-8") as f:
        criterion_rows = [GoldCriterionAnnotation.from_csv_row(row) for row in csv.DictReader(f)]
    with open(document_csv_path, newline="", encoding="utf-8") as f:
        document_rows = [GoldDocumentAnnotation.from_csv_row(row) for row in csv.DictReader(f)]

    for ann in criterion_rows:
        if ann.annotation_round != pass_name:
            raise ValueError(
                f"{criterion_csv_path} contains a row for document_id={ann.document_id!r}, "
                f"criterion_id={ann.criterion_id!r} with annotation_round={ann.annotation_round!r}, expected "
                f"exactly {pass_name!r} -- this file must contain ONLY {pass_name!r} rows, never a mix from "
                "a different pass appended into the same worksheet"
            )
    for ann in document_rows:
        if ann.annotation_round != pass_name:
            raise ValueError(
                f"{document_csv_path} contains a row for document_id={ann.document_id!r} with "
                f"annotation_round={ann.annotation_round!r}, expected exactly {pass_name!r}"
            )

    expected_document_ids = set(expected_document_ids)

    document_id_counts = Counter(ann.document_id for ann in document_rows)
    duplicate_documents = {doc_id for doc_id, n in document_id_counts.items() if n > 1}
    if duplicate_documents:
        raise ValueError(f"{document_csv_path} has duplicate document rows for document_ids={sorted(duplicate_documents)}")
    actual_ids = set(document_id_counts)
    if actual_ids != expected_document_ids:
        raise ValueError(
            f"{document_csv_path} covers document_ids={sorted(actual_ids)}, expected exactly "
            f"{sorted(expected_document_ids)} -- the {pass_name!r} pass must be COMPLETE before sealing"
        )

    expected_criterion_ids = {c.criterion_id for c in criteria_set.criteria}
    expected_keys = {(doc_id, crit_id) for doc_id in expected_document_ids for crit_id in expected_criterion_ids}
    criterion_key_counts = Counter((ann.document_id, ann.criterion_id) for ann in criterion_rows)
    duplicate_keys = {key for key, n in criterion_key_counts.items() if n > 1}
    if duplicate_keys:
        raise ValueError(f"{criterion_csv_path} has duplicate criterion rows for keys={sorted(duplicate_keys)}")
    actual_keys = set(criterion_key_counts)
    if actual_keys != expected_keys:
        raise ValueError(
            f"{criterion_csv_path} covers keys={sorted(actual_keys)}, expected exactly the full "
            f"document x criterion product {sorted(expected_keys)} (missing={sorted(expected_keys - actual_keys)}, "
            f"extra={sorted(actual_keys - expected_keys)}) -- the {pass_name!r} pass must be COMPLETE before sealing"
        )

    seal = WorksheetPassSeal(
        pass_name=pass_name, protocol_version=protocol_version, criteria_set_hash=compute_criteria_set_hash(criteria_set),
        blind_manifest_hash=hashlib.sha256(Path(blind_manifest_path).read_bytes()).hexdigest(),
        sample_manifest_hash=hashlib.sha256(Path(sample_manifest_path).read_bytes()).hexdigest(),
        document_ids=tuple(sorted(actual_ids)),
        criterion_csv_sha256=hash_pdf_file(Path(criterion_csv_path)),
        document_csv_sha256=hash_pdf_file(Path(document_csv_path)),
        sealed_at=datetime.now(timezone.utc).isoformat(),
    )
    _freeze_frozen_model(seal, output_dir=output_dir, version_label=version_label, artifact_type="WorksheetPassSeal")
    return seal


def load_frozen_worksheet_pass_seal(json_path: Path, *, expected_hash: str | None = None) -> WorksheetPassSeal:
    return _load_frozen_model(WorksheetPassSeal, json_path, expected_hash=expected_hash, expected_artifact_type="WorksheetPassSeal")


def verify_worksheet_pass_seal(
    seal: WorksheetPassSeal,
    *,
    expected_pass_name: Literal["primary", "second_pass"],
    expected_protocol_version: str,
    expected_criteria_set_hash: str,
    expected_document_ids: set[str],
    expected_blind_manifest_hash: str,
    expected_sample_manifest_hash: str,
    criterion_csv_path: Path,
    document_csv_path: Path,
) -> None:
    """The ONE place every trust boundary verifies a sealed pass -- both
    its recorded IDENTITY fields and the ACTUAL current bytes of its
    sealed CSV snapshot. Used both by `freeze_gold_set()`'s
    reconstruction and by every workflow script (via
    `scripts.gold_set_workflow_common.load_and_verify_seal`), so the
    check exists in exactly one place rather than being reimplemented --
    correctly in one call site, incompletely in another -- per script
    (codex_code_review_93bf42d.md finding 3's lesson, extended here).

    codex_code_review_73cf5d9.md finding 1's exact repro: a seal
    recording `protocol_version`/`criteria_set_hash`/`document_ids`
    verified successfully even when those fields NAMED a different
    protocol, a different criteria set, or a second-pass seal covering
    MORE documents than the actual dual-review subset -- because nothing
    ever compared the recorded identity against the identity actually
    being frozen or revealed against. Every field is checked here, not
    just the pass name and the two CSV hashes."""
    if seal.pass_name != expected_pass_name:
        raise ValueError(f"seal has pass_name={seal.pass_name!r}, expected {expected_pass_name!r}")
    if seal.protocol_version != expected_protocol_version:
        raise ValueError(
            f"seal.protocol_version={seal.protocol_version!r} does not match the expected "
            f"protocol_version={expected_protocol_version!r}"
        )
    if seal.criteria_set_hash != expected_criteria_set_hash:
        raise ValueError(
            f"seal.criteria_set_hash={seal.criteria_set_hash!r} does not match the expected "
            f"criteria_set_hash={expected_criteria_set_hash!r}"
        )
    if set(seal.document_ids) != set(expected_document_ids):
        raise ValueError(
            f"seal.document_ids={sorted(seal.document_ids)} does not match the expected document set "
            f"EXACTLY {sorted(expected_document_ids)} -- a {expected_pass_name!r} seal must cover exactly "
            "the right documents, not a superset or subset"
        )
    if seal.blind_manifest_hash != expected_blind_manifest_hash:
        raise ValueError(
            f"seal.blind_manifest_hash does not match the current blind manifest -- it may have changed "
            "since this pass was sealed"
        )
    if seal.sample_manifest_hash != expected_sample_manifest_hash:
        raise ValueError(
            "seal.sample_manifest_hash does not match the current sample manifest -- it may have changed "
            "since this pass was sealed"
        )
    actual_criterion_hash = hash_pdf_file(Path(criterion_csv_path))
    if actual_criterion_hash != seal.criterion_csv_sha256:
        raise ValueError(
            f"{criterion_csv_path} does not match its own {seal.pass_name!r}-pass seal "
            f"(expected sha256={seal.criterion_csv_sha256!r}, actual={actual_criterion_hash!r}) -- "
            "it may have been edited since sealing"
        )
    actual_document_hash = hash_pdf_file(Path(document_csv_path))
    if actual_document_hash != seal.document_csv_sha256:
        raise ValueError(
            f"{document_csv_path} does not match its own {seal.pass_name!r}-pass seal "
            f"(expected sha256={seal.document_csv_sha256!r}, actual={actual_document_hash!r}) -- "
            "it may have been edited since sealing"
        )


class WorksheetSourceRecord(BaseModel):
    """One verified source file pair, retaining not just its path but
    the ACTUAL verified sha256 (and, for sealed passes, the seal's own
    `content_hash()`) -- codex_code_review_73cf5d9.md finding 3: a path
    string alone means the file at that location could be silently
    replaced after freezing (even coherently, seal+CSVs together) with
    no detectable change to the frozen `GoldSet`. `adjudication` has no
    seal (its coverage is a subset of the dual-review keys, not a full
    Cartesian product `seal_worksheet_pass()` could validate), so its
    `seal_path`/`seal_content_hash` are always `None` -- but its CSV
    hashes are still recorded and verified, closing the prior gap where
    adjudication files were never hashed at all."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["primary", "second_pass", "adjudication"]
    seal_path: str | None = None
    seal_content_hash: str | None = None
    criterion_csv_path: str = Field(min_length=1)
    criterion_csv_sha256: str = Field(min_length=64, max_length=64)
    document_csv_path: str = Field(min_length=1)
    document_csv_sha256: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def _seal_fields_match_role(self) -> "WorksheetSourceRecord":
        needs_seal = self.role in ("primary", "second_pass")
        has_seal = self.seal_path is not None or self.seal_content_hash is not None
        if needs_seal and not (self.seal_path is not None and self.seal_content_hash is not None):
            raise ValueError(f"role={self.role!r} requires both seal_path and seal_content_hash")
        if not needs_seal and has_seal:
            raise ValueError(f"role={self.role!r} must not carry seal_path/seal_content_hash (adjudication files aren't sealed)")
        return self


class WorksheetSourceProvenance(BaseModel):
    """A strict, versioned record of exactly which sealed source files
    were combined to produce a v6+ `GoldSet`'s raw annotation input --
    replaces a prior permissive `{name: hash}` dict that proved listed
    files hadn't changed since being named, but nothing tied them to
    being the ACTUAL sealed passes or verified they produced the frozen
    rows (codex_code_review_93bf42d.md finding 2's exact repro: an empty
    manifest, and a manifest naming only an unrelated text file, both
    froze successfully). `freeze_gold_set()` treats this as MANDATORY,
    independently reconstructs the merged annotation set (and its dual-
    review audit trail -- codex_code_review_73cf5d9.md finding 2) from
    the files named here (after verifying each sealed pair against its
    own seal, INCLUDING the seal's recorded protocol/criteria/manifest
    identity -- finding 1), and rejects the freeze if that reconstruction
    disagrees with the caller-supplied merged CSVs -- so this record
    can't just describe unrelated files, it has to actually PRODUCE the
    frozen result. `merged_criterion_csv_sha256`/`merged_document_csv_sha256`
    additionally retain the merged CSVs' own hashes as a secondary
    reproducibility check (finding 3)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol_version: str = Field(min_length=1)
    criteria_set_hash: str = Field(min_length=64, max_length=64)
    blind_manifest_hash: str = Field(min_length=1)
    sample_manifest_hash: str = Field(min_length=1)
    primary: WorksheetSourceRecord
    second_pass: WorksheetSourceRecord | None = None
    adjudication: WorksheetSourceRecord | None = None
    merged_criterion_csv_sha256: str = Field(min_length=64, max_length=64)
    merged_document_csv_sha256: str = Field(min_length=64, max_length=64)
    merged_at: str = Field(min_length=1)

    @model_validator(mode="after")
    def _roles_correct(self) -> "WorksheetSourceProvenance":
        if self.primary.role != "primary":
            raise ValueError("primary.role must be 'primary'")
        if self.second_pass is not None and self.second_pass.role != "second_pass":
            raise ValueError("second_pass.role must be 'second_pass'")
        if self.adjudication is not None and self.adjudication.role != "adjudication":
            raise ValueError("adjudication.role must be 'adjudication'")
        return self

    @model_validator(mode="after")
    def _second_pass_and_adjudication_together(self) -> "WorksheetSourceProvenance":
        if (self.second_pass is None) != (self.adjudication is None):
            raise ValueError("second_pass and adjudication must be both present or both absent")
        return self


def _recompute_merged_annotations_from_provenance(
    provenance: WorksheetSourceProvenance,
    *,
    protocol_version: str,
    criteria_set: CriteriaSet,
    expected_document_ids: set[str],
    dual_review_document_ids: set[str],
    blind_manifest_hash: str,
    sample_manifest_hash: str,
) -> tuple[
    dict[CriterionKey, GoldCriterionAnnotation],
    dict[str, GoldDocumentAnnotation],
    dict[CriterionKey, CriterionDualReviewAudit],
    dict[str, DocumentDualReviewAudit],
]:
    """Independently reconstructs the merged annotation set -- AND its
    full dual-review audit trail (codex_code_review_73cf5d9.md finding
    2: a prior version discarded the reconstructed audits and compared
    only resolved outcomes, which are IDENTICAL whether or not a raw
    primary/second_pass row was forged, as long as the adjudicated
    row -- which is what disagreeing keys actually resolve to -- was
    left unchanged) -- from the VERIFIED sealed sources named in
    `provenance`. Called from inside `freeze_gold_set()`, whose caller-
    supplied `criterion_csv_path`/`document_csv_path` are compared
    against this reconstruction key-for-key, audits included, not just
    resolved values. Returns key-indexed dicts (not ordered lists) since
    row order across independently-read files is not itself meaningful."""
    criteria_set_hash = compute_criteria_set_hash(criteria_set)
    if provenance.protocol_version != protocol_version:
        raise ValueError(
            f"source_provenance.protocol_version={provenance.protocol_version!r} does not match the "
            f"freeze_gold_set(protocol_version={protocol_version!r}) argument"
        )
    if provenance.criteria_set_hash != criteria_set_hash:
        raise ValueError("source_provenance.criteria_set_hash does not match the actual criteria_set being frozen")
    if provenance.blind_manifest_hash != blind_manifest_hash:
        raise ValueError("source_provenance.blind_manifest_hash does not match the current blind manifest")
    if provenance.sample_manifest_hash != sample_manifest_hash:
        raise ValueError("source_provenance.sample_manifest_hash does not match the current sample manifest")

    def _verify_record_bytes(record: WorksheetSourceRecord) -> None:
        actual_criterion = hash_pdf_file(Path(record.criterion_csv_path))
        if actual_criterion != record.criterion_csv_sha256:
            raise ValueError(
                f"{record.criterion_csv_path} does not match its recorded provenance hash for role={record.role!r} "
                "-- it may have been edited since merging"
            )
        actual_document = hash_pdf_file(Path(record.document_csv_path))
        if actual_document != record.document_csv_sha256:
            raise ValueError(
                f"{record.document_csv_path} does not match its recorded provenance hash for role={record.role!r} "
                "-- it may have been edited since merging"
            )

    def _verify_sealed_record(record: WorksheetSourceRecord, *, expected_pass_name: str, expected_ids: set[str]) -> None:
        _verify_record_bytes(record)
        seal = load_frozen_worksheet_pass_seal(Path(record.seal_path))
        if seal.content_hash() != record.seal_content_hash:
            raise ValueError(
                f"source_provenance's recorded seal_content_hash for role={record.role!r} does not match the "
                f"actual seal at {record.seal_path} -- the seal (and possibly its CSV pair) may have been replaced"
            )
        verify_worksheet_pass_seal(
            seal, expected_pass_name=expected_pass_name, expected_protocol_version=protocol_version,
            expected_criteria_set_hash=criteria_set_hash, expected_document_ids=expected_ids,
            expected_blind_manifest_hash=blind_manifest_hash, expected_sample_manifest_hash=sample_manifest_hash,
            criterion_csv_path=Path(record.criterion_csv_path), document_csv_path=Path(record.document_csv_path),
        )

    _verify_sealed_record(provenance.primary, expected_pass_name="primary", expected_ids=expected_document_ids)
    criterion_paths = [Path(provenance.primary.criterion_csv_path)]
    document_paths = [Path(provenance.primary.document_csv_path)]

    if dual_review_document_ids:
        # A sample with a dual-review subset REQUIRES a second pass and an
        # adjudication artifact -- never silently downgraded to optional
        # (codex_code_review_93bf42d.md finding 3).
        if provenance.second_pass is None:
            raise ValueError(
                "the sample manifest has a non-empty dual_review_subset, but source_provenance has no "
                "second_pass record -- a second pass is required, not optional, whenever dual review applies"
            )
        _verify_sealed_record(provenance.second_pass, expected_pass_name="second_pass", expected_ids=dual_review_document_ids)
        criterion_paths.append(Path(provenance.second_pass.criterion_csv_path))
        document_paths.append(Path(provenance.second_pass.document_csv_path))

        if provenance.adjudication is None:
            raise ValueError(
                "the sample manifest has a non-empty dual_review_subset, but source_provenance has no "
                "adjudication record -- an adjudication artifact is required whenever dual review applies, "
                "even a zero-row one recording that no keys disagreed"
            )
        _verify_record_bytes(provenance.adjudication)
        criterion_paths.append(Path(provenance.adjudication.criterion_csv_path))
        document_paths.append(Path(provenance.adjudication.document_csv_path))
    elif provenance.second_pass is not None:
        raise ValueError(
            "the sample manifest has an EMPTY dual_review_subset, but source_provenance has a second_pass record"
        )

    criteria, documents, criterion_audits, document_audits = load_gold_annotations_from_csv_multi(
        criterion_paths, document_paths, dual_review_document_ids=dual_review_document_ids,
    )
    return (
        {(a.document_id, a.criterion_id): a for a in criteria},
        {a.document_id: a for a in documents},
        {(a.document_id, a.criterion_id): a for a in criterion_audits},
        {a.document_id: a for a in document_audits},
    )


class GoldSet(BaseModel):
    """The complete, frozen phase-1 gold standard."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol_version: str = Field(min_length=1)
    blind_manifest_hash: str = Field(min_length=1)
    sample_manifest_hash: str = Field(min_length=1)
    document_pdf_sha256: dict[str, str] = Field(min_length=1)
    expected_document_ids: tuple[str, ...] = Field(min_length=1)
    expected_dual_review_document_ids: tuple[str, ...] = Field(default_factory=tuple)
    criteria_set: CriteriaSet
    criterion_annotations: tuple[GoldCriterionAnnotation, ...] = Field(min_length=1)
    document_annotations: tuple[GoldDocumentAnnotation, ...] = Field(min_length=1)
    criterion_dual_review_audit: tuple[CriterionDualReviewAudit, ...] = Field(default_factory=tuple)
    document_dual_review_audit: tuple[DocumentDualReviewAudit, ...] = Field(default_factory=tuple)
    # The VERIFIED provenance record of exactly which sealed source files
    # were reconstructed into this GoldSet's annotations -- mandatory,
    # not an optional/unstructured hash dict (codex_code_review_93bf42d.md
    # finding 2: an empty manifest, and one naming only an unrelated file,
    # both previously froze successfully).
    source_provenance: WorksheetSourceProvenance
    frozen_at: str = Field(min_length=1)

    @model_validator(mode="after")
    def _coverage_and_uniqueness(self) -> "GoldSet":
        criterion_ids = {c.criterion_id for c in self.criteria_set.criteria}

        seen_criterion_keys: set[CriterionKey] = set()
        doc_ids_from_criteria: set[str] = set()
        for ann in self.criterion_annotations:
            key = (ann.document_id, ann.criterion_id)
            if key in seen_criterion_keys:
                raise ValueError(f"duplicate criterion annotation for {key}")
            seen_criterion_keys.add(key)
            if ann.criterion_id not in criterion_ids:
                raise ValueError(f"criterion_annotation references unknown criterion_id={ann.criterion_id!r}")
            doc_ids_from_criteria.add(ann.document_id)

        doc_ids_from_docs: set[str] = set()
        for ann in self.document_annotations:
            if ann.document_id in doc_ids_from_docs:
                raise ValueError(f"duplicate document annotation for document_id={ann.document_id!r}")
            doc_ids_from_docs.add(ann.document_id)

        if doc_ids_from_criteria != doc_ids_from_docs:
            raise ValueError(
                "criterion_annotations and document_annotations must cover exactly the same set of document_ids"
            )

        expected = set(self.expected_document_ids)
        if doc_ids_from_docs != expected:
            raise ValueError(
                "annotated document_ids do not match the sample manifest's expected_document_ids exactly "
                f"(missing: {sorted(expected - doc_ids_from_docs)}, unexpected: {sorted(doc_ids_from_docs - expected)})"
            )

        for doc_id in doc_ids_from_docs:
            covered = {ann.criterion_id for ann in self.criterion_annotations if ann.document_id == doc_id}
            missing = criterion_ids - covered
            if missing:
                raise ValueError(f"document_id={doc_id!r} is missing criterion annotations for {sorted(missing)}")
        return self

    @model_validator(mode="after")
    def _page_count_consistent_per_document(self) -> "GoldSet":
        page_count_by_doc: dict[str, int] = {}
        for ann in self.criterion_annotations:
            existing = page_count_by_doc.setdefault(ann.document_id, ann.document_page_count)
            if existing != ann.document_page_count:
                raise ValueError(f"document_id={ann.document_id!r} has inconsistent document_page_count across its criterion rows")
        for ann in self.document_annotations:
            criterion_count = page_count_by_doc.get(ann.document_id)
            if criterion_count is not None and criterion_count != ann.document_page_count:
                raise ValueError(f"document_id={ann.document_id!r}: document-level and criterion-level document_page_count disagree")
        return self

    @model_validator(mode="after")
    def _dual_review_enforced_from_retained_audit(self) -> "GoldSet":
        """Checks the RETAINED RAW audit trail, not a caller-settable
        counter (codex_code_review_772916e.md finding 2's exact repro: a
        lone row with `raw_pass_count` hand-set to 2 previously satisfied
        this check with zero raw passes behind it). Every dual-review key
        must have a corresponding audit record whose `resolves_to()`
        exactly matches the authoritative annotation actually frozen."""
        dual_ids = set(self.expected_dual_review_document_ids)
        criterion_audit_by_key = {(a.document_id, a.criterion_id): a for a in self.criterion_dual_review_audit}
        document_audit_by_id = {a.document_id: a for a in self.document_dual_review_audit}

        for ann in self.criterion_annotations:
            if ann.document_id not in dual_ids:
                continue
            key = (ann.document_id, ann.criterion_id)
            audit = criterion_audit_by_key.get(key)
            if audit is None:
                raise ValueError(f"{key} is in the dual-review subset but has no retained raw-pass audit record")
            expected = audit.resolves_to()
            if ann.annotation_round != expected.annotation_round or not _rows_content_equal(ann, expected, fields=_CRITERION_CONTENT_FIELDS):
                raise ValueError(f"{key}'s authoritative annotation does not match what its retained audit record resolves to")

        for ann in self.document_annotations:
            if ann.document_id not in dual_ids:
                continue
            audit = document_audit_by_id.get(ann.document_id)
            if audit is None:
                raise ValueError(f"document_id={ann.document_id!r} is in the dual-review subset but has no retained raw-pass audit record")
            expected = audit.resolves_to()
            if ann.annotation_round != expected.annotation_round or not _rows_content_equal(ann, expected, fields=_DOCUMENT_CONTENT_FIELDS):
                raise ValueError(f"document_id={ann.document_id!r}'s authoritative annotation does not match its retained audit record")
        return self

    @model_validator(mode="after")
    def _aggregation_cross_check(self) -> "GoldSet":
        from ft_screening.aggregation import aggregate_decision
        from ft_screening.schema import CriterionAssessment

        spec_by_id = {c.criterion_id: c for c in self.criteria_set.criteria}
        by_doc: dict[str, list[GoldCriterionAnnotation]] = {}
        for ann in self.criterion_annotations:
            by_doc.setdefault(ann.document_id, []).append(ann)
        doc_ann_by_id = {d.document_id: d for d in self.document_annotations}

        for doc_id, criterion_anns in by_doc.items():
            doc_ann = doc_ann_by_id[doc_id]
            assessments = []
            for ann in criterion_anns:
                spec = spec_by_id[ann.criterion_id]
                requires_review = ann.human_review_required or (
                    ann.gold_status in ("unclear", "not_reported") and spec.missing_evidence_policy == "review"
                )
                assessments.append(CriterionAssessment(
                    criterion_id=ann.criterion_id, criterion_version=spec.version, status=ann.gold_status,
                    evidence=[], rationale="gold annotation aggregation cross-check (gold.py)",
                    requires_human_review=requires_review, model="gold-annotation",
                    prompt_template_version="gold-v1", output_schema_version="gold-v1",
                ))
            decision, _codes, primary_reason = aggregate_decision(criteria_set=self.criteria_set, assessments=assessments)

            mismatches: list[str] = []
            if decision != doc_ann.gold_final_decision:
                mismatches.append(
                    f"aggregation.aggregate_decision() computes final_decision={decision!r}, but the annotator "
                    f"declared gold_final_decision={doc_ann.gold_final_decision!r}"
                )
            if decision == "exclude" and primary_reason != doc_ann.gold_primary_exclusion_reason_code:
                mismatches.append(
                    f"aggregation.aggregate_decision() computes primary_exclusion_reason_code={primary_reason!r}, "
                    f"but the annotator declared {doc_ann.gold_primary_exclusion_reason_code!r}"
                )
            if mismatches and not doc_ann.aggregation_override_rationale:
                raise ValueError(
                    f"document_id={doc_id!r}: " + "; ".join(mismatches) + ". Either fix the annotation or set "
                    "aggregation_override_rationale explaining why this is an intentional, acknowledged override."
                )
        return self

    def content_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"frozen_at"})
        return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def freeze_gold_set(
    *,
    protocol_version: str,
    criteria_set: CriteriaSet,
    criterion_csv_path: Path,
    document_csv_path: Path,
    blind_manifest_path: Path,
    sample_manifest_path: Path,
    output_dir: Path,
    version_label: str,
    source_provenance_path: Path,
) -> GoldSet:
    """Validates, hashes, and durably (exclusive-write) writes the frozen
    gold set. Consumes RAW CSV paths directly and performs collapsing
    internally (codex_code_review_772916e.md finding 2) -- there is no
    supported entry point that accepts pre-collapsed annotations. Both
    manifests are strictly parsed and cross-bound (finding 3), and every
    PDF is re-hashed AND re-page-counted from its actual bytes.

    `source_provenance_path` is MANDATORY (codex_code_review_93bf42d.md
    finding 2: an absent or empty source manifest previously froze
    successfully) and must point at a JSON `WorksheetSourceProvenance`
    document. It is not merely hashed and trusted: every sealed source it
    names is re-verified against its own seal (finding 3), a second pass
    and adjudication artifact are REQUIRED whenever the sample manifest's
    `dual_review_subset` is non-empty (never silently downgraded to
    optional), and the merged annotation set is INDEPENDENTLY
    RECONSTRUCTED from those verified sources and compared -- key for
    key -- against what `criterion_csv_path`/`document_csv_path`
    actually contain. Any mismatch is rejected; only a provenance record
    that PRODUCES the exact frozen annotations is accepted, and the
    validated record itself (not just a hash of it) is retained on the
    frozen `GoldSet` as `source_provenance`."""
    if protocol_version != SUPPORTED_PROTOCOL_VERSION:
        raise ValueError(
            f"protocol_version={protocol_version!r} is not the currently supported protocol "
            f"({SUPPORTED_PROTOCOL_VERSION!r}) -- freeze_gold_set does not accept an arbitrary nonempty "
            "string here (codex_code_review_1c40a27.md finding 3)"
        )
    blind_bytes = blind_manifest_path.read_bytes()
    blind_manifest_hash = hashlib.sha256(blind_bytes).hexdigest()
    blind_manifest = BlindManifest.model_validate_json(blind_bytes)

    sample_bytes = sample_manifest_path.read_bytes()
    sample_manifest_hash = hashlib.sha256(sample_bytes).hexdigest()
    sample_manifest = SampleManifest.model_validate_json(sample_bytes)

    _validate_manifests_cross_bound(blind_manifest, sample_manifest)
    if blind_manifest.protocol_version != protocol_version:
        raise ValueError(
            f"blind_manifest.protocol_version={blind_manifest.protocol_version!r} does not match the "
            f"freeze_gold_set(protocol_version={protocol_version!r}) argument"
        )

    expected_document_ids = tuple(sorted(d.document_id for d in sample_manifest.documents))
    expected_dual_review_document_ids = tuple(sorted(sample_manifest.dual_review_subset))

    document_pdf_sha256: dict[str, str] = {}
    document_page_count_from_manifest: dict[str, int] = {}
    for entry in sample_manifest.documents:
        pdf_path = Path(entry.pdf_path)
        actual_hash = hash_pdf_file(pdf_path)
        if actual_hash != entry.pdf_sha256:
            raise ValueError(
                f"document_id={entry.document_id!r}: the actual PDF bytes at {pdf_path} hash to "
                f"{actual_hash!r}, but the sample manifest claims pdf_sha256={entry.pdf_sha256!r}"
            )
        actual_page_count = real_pdf_page_count(pdf_path)
        if actual_page_count != entry.page_count:
            raise ValueError(
                f"document_id={entry.document_id!r}: the actual PDF at {pdf_path} has {actual_page_count} "
                f"pages, but the sample manifest claims page_count={entry.page_count}"
            )
        document_pdf_sha256[entry.document_id] = actual_hash
        document_page_count_from_manifest[entry.document_id] = entry.page_count

    criterion_annotations, document_annotations, criterion_audits, document_audits = load_gold_annotations_from_csv(
        criterion_csv_path, document_csv_path, dual_review_document_ids=set(expected_dual_review_document_ids),
    )

    for ann in criterion_annotations:
        expected_pages = document_page_count_from_manifest.get(ann.document_id)
        if expected_pages is not None and ann.document_page_count != expected_pages:
            raise ValueError(
                f"document_id={ann.document_id!r}: annotated document_page_count={ann.document_page_count} "
                f"does not match the sample manifest's real page count={expected_pages}"
            )

    source_provenance_bytes = Path(source_provenance_path).read_bytes()
    source_provenance = WorksheetSourceProvenance.model_validate_json(source_provenance_bytes)

    actual_merged_criterion_hash = hash_pdf_file(Path(criterion_csv_path))
    if actual_merged_criterion_hash != source_provenance.merged_criterion_csv_sha256:
        raise ValueError(
            f"criterion_csv_path={criterion_csv_path} does not match source_provenance.merged_criterion_csv_sha256 "
            "-- the merged file may have been edited after the merge step ran (codex_code_review_73cf5d9.md finding 3)"
        )
    actual_merged_document_hash = hash_pdf_file(Path(document_csv_path))
    if actual_merged_document_hash != source_provenance.merged_document_csv_sha256:
        raise ValueError(
            f"document_csv_path={document_csv_path} does not match source_provenance.merged_document_csv_sha256 "
            "-- the merged file may have been edited after the merge step ran (codex_code_review_73cf5d9.md finding 3)"
        )

    recomputed_criteria_by_key, recomputed_documents_by_id, recomputed_criterion_audits_by_key, recomputed_document_audits_by_id = (
        _recompute_merged_annotations_from_provenance(
            source_provenance, protocol_version=protocol_version, criteria_set=criteria_set,
            expected_document_ids=set(expected_document_ids), dual_review_document_ids=set(expected_dual_review_document_ids),
            blind_manifest_hash=blind_manifest_hash, sample_manifest_hash=sample_manifest_hash,
        )
    )
    actual_criteria_by_key = {(a.document_id, a.criterion_id): a for a in criterion_annotations}
    actual_documents_by_id = {a.document_id: a for a in document_annotations}
    actual_criterion_audits_by_key = {(a.document_id, a.criterion_id): a for a in criterion_audits}
    actual_document_audits_by_id = {a.document_id: a for a in document_audits}
    if recomputed_criteria_by_key != actual_criteria_by_key:
        raise ValueError(
            "source_provenance's sealed sources, independently reconstructed, do not match "
            "criterion_csv_path's actual annotations -- the merged file may not have been derived from the "
            "recorded sources (codex_code_review_93bf42d.md finding 2)"
        )
    if recomputed_documents_by_id != actual_documents_by_id:
        raise ValueError(
            "source_provenance's sealed sources, independently reconstructed, do not match "
            "document_csv_path's actual annotations -- the merged file may not have been derived from the "
            "recorded sources (codex_code_review_93bf42d.md finding 2)"
        )
    if recomputed_criterion_audits_by_key != actual_criterion_audits_by_key:
        raise ValueError(
            "source_provenance's sealed sources, independently reconstructed, do not match the merged file's "
            "own criterion dual-review audit trail -- a raw primary/second_pass/adjudicated row may have been "
            "forged in the merged file while its RESOLVED outcome was left unchanged (codex_code_review_73cf5d9.md "
            "finding 2's exact repro)"
        )
    if recomputed_document_audits_by_id != actual_document_audits_by_id:
        raise ValueError(
            "source_provenance's sealed sources, independently reconstructed, do not match the merged file's "
            "own document dual-review audit trail (codex_code_review_73cf5d9.md finding 2)"
        )

    # The VERIFIED reconstruction from sealed sources is authoritative
    # (codex_code_review_73cf5d9.md finding 2's required fix) -- the
    # merged-CSV parse above is retained purely as a cross-check, never
    # as the source of the frozen values, even though the two are now
    # proven byte-for-byte equivalent by the comparisons above.
    gold_set = GoldSet(
        protocol_version=protocol_version,
        blind_manifest_hash=blind_manifest_hash,
        sample_manifest_hash=sample_manifest_hash,
        document_pdf_sha256=document_pdf_sha256,
        expected_document_ids=expected_document_ids,
        expected_dual_review_document_ids=expected_dual_review_document_ids,
        criteria_set=criteria_set,
        criterion_annotations=tuple(recomputed_criteria_by_key.values()),
        document_annotations=tuple(recomputed_documents_by_id.values()),
        criterion_dual_review_audit=tuple(recomputed_criterion_audits_by_key.values()),
        document_dual_review_audit=tuple(recomputed_document_audits_by_id.values()),
        source_provenance=source_provenance,
        frozen_at=datetime.now(timezone.utc).isoformat(),
    )
    _freeze_frozen_model(gold_set, output_dir=output_dir, version_label=version_label, artifact_type="GoldSet")
    return gold_set


def load_frozen_gold_set(json_path: Path, *, expected_hash: str | None = None) -> GoldSet:
    return _load_frozen_model(GoldSet, json_path, expected_hash=expected_hash, expected_artifact_type="GoldSet")


# ---------------------------------------------------------------------------
# Arm 3: retrieval-oracle artifact
# ---------------------------------------------------------------------------


class RetrievalOracleSpan(BaseModel):
    """A page/excerpt/chunk_id PROJECTION of the real retriever's output
    -- NOT the full EvidenceSpan record (see gold_set_protocol_v4.md
    §7's documented limitation)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str = Field(min_length=1)
    page: int | None = Field(default=None, ge=1)
    excerpt: str


class RetrievalOracleArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    parser_name: str = Field(min_length=1)
    parser_version: str = Field(min_length=1)
    parser_options_hash: str = Field(min_length=1)
    retriever_name: str = Field(min_length=1)
    retriever_config_version: str = Field(min_length=1)
    retriever_options_hash: str = Field(min_length=1)
    criteria_set_hash: str = Field(min_length=1)
    document_sha256: dict[str, str] = Field(min_length=1)
    evidence_by_key: dict[str, tuple[RetrievalOracleSpan, ...]] = Field(min_length=1)
    frozen_at: str = Field(min_length=1)

    def content_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"frozen_at"})
        return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def make_evidence_key(document_id: str, criterion_id: str) -> str:
    return f"{document_id}::{criterion_id}"


def freeze_retrieval_oracle_artifact(
    *,
    parser_name: str,
    parser_version: str,
    parser_options: dict,
    retriever_name: str,
    retriever_config_version: str,
    retriever_options: dict,
    criteria_set: CriteriaSet,
    document_sha256: dict[str, str],
    evidence_by_key: dict[str, list[RetrievalOracleSpan]],
    output_dir: Path,
    version_label: str,
) -> RetrievalOracleArtifact:
    artifact = RetrievalOracleArtifact(
        parser_name=parser_name, parser_version=parser_version,
        parser_options_hash=hashlib.sha256(canonical_json_bytes(parser_options)).hexdigest(),
        retriever_name=retriever_name, retriever_config_version=retriever_config_version,
        retriever_options_hash=hashlib.sha256(canonical_json_bytes(retriever_options)).hexdigest(),
        criteria_set_hash=compute_criteria_set_hash(criteria_set),
        document_sha256=document_sha256,
        evidence_by_key={k: tuple(v) for k, v in evidence_by_key.items()},
        frozen_at=datetime.now(timezone.utc).isoformat(),
    )
    _freeze_frozen_model(artifact, output_dir=output_dir, version_label=version_label, artifact_type="RetrievalOracleArtifact")
    return artifact


def load_frozen_retrieval_oracle_artifact(json_path: Path, *, expected_hash: str | None = None) -> RetrievalOracleArtifact:
    return _load_frozen_model(RetrievalOracleArtifact, json_path, expected_hash=expected_hash, expected_artifact_type="RetrievalOracleArtifact")


# ---------------------------------------------------------------------------
# Phase 2: retrieval-sufficiency judgments
# ---------------------------------------------------------------------------


class RetrievalSufficiencyJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str = Field(min_length=1)
    criterion_id: str = Field(min_length=1)
    retrieval_sufficient: bool
    annotator_id: str = Field(min_length=1)
    notes: str = ""

    @classmethod
    def from_csv_row(cls, row: dict[str, str]) -> "RetrievalSufficiencyJudgment":
        return cls(
            document_id=row["document_id"], criterion_id=row["criterion_id"],
            retrieval_sufficient=_parse_bool(row["retrieval_sufficient"], field_name="retrieval_sufficient"),
            annotator_id=row["annotator_id"], notes=row.get("notes", ""),
        )


class RetrievalOracleJudgmentSet(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    gold_set_hash: str = Field(min_length=1)
    retrieval_artifact_hash: str = Field(min_length=1)
    judgments: tuple[RetrievalSufficiencyJudgment, ...] = Field(min_length=1)
    frozen_at: str = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_keys(self) -> "RetrievalOracleJudgmentSet":
        seen: set[CriterionKey] = set()
        for j in self.judgments:
            key = (j.document_id, j.criterion_id)
            if key in seen:
                raise ValueError(f"duplicate retrieval-sufficiency judgment for {key}")
            seen.add(key)
        return self

    def content_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"frozen_at"})
        return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def freeze_retrieval_oracle_judgment_set(
    *,
    gold_set: GoldSet,
    retrieval_artifact: RetrievalOracleArtifact,
    judgments: list[RetrievalSufficiencyJudgment],
    output_dir: Path,
    version_label: str,
) -> RetrievalOracleJudgmentSet:
    judgment_set = RetrievalOracleJudgmentSet(
        gold_set_hash=gold_set.content_hash(), retrieval_artifact_hash=retrieval_artifact.content_hash(),
        judgments=tuple(judgments), frozen_at=datetime.now(timezone.utc).isoformat(),
    )
    _freeze_frozen_model(judgment_set, output_dir=output_dir, version_label=version_label, artifact_type="RetrievalOracleJudgmentSet")
    return judgment_set


def load_frozen_retrieval_oracle_judgment_set(json_path: Path, *, expected_hash: str | None = None) -> RetrievalOracleJudgmentSet:
    return _load_frozen_model(RetrievalOracleJudgmentSet, json_path, expected_hash=expected_hash, expected_artifact_type="RetrievalOracleJudgmentSet")


# ---------------------------------------------------------------------------
# Generic frozen-model helpers, reused by evaluation_harness.py for
# EvaluationResultArtifact/EvaluationRunManifest persistence
# ---------------------------------------------------------------------------


def freeze_frozen_model(model: BaseModel, *, output_dir: Path, version_label: str, artifact_type: str) -> None:
    _freeze_frozen_model(model, output_dir=output_dir, version_label=version_label, artifact_type=artifact_type)


def load_frozen_model(cls, json_path: Path, *, expected_hash: str | None = None, expected_artifact_type: str | None = None):
    return _load_frozen_model(cls, json_path, expected_hash=expected_hash, expected_artifact_type=expected_artifact_type)
