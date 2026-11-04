"""Arm 2b -- LLM adjudication using RETROSPECTIVELY CURATED, human-reviewed
evidence packages (gold_evidence_set_v1), not the raw GoldSet citations
Arm 2/Arm 2 (prompt-6) adjudicate against.

WHAT THIS IS, EXPLICITLY: "LLM adjudication using retrospectively curated,
human-reviewed evidence packages." It answers whether prompt-6 adjudicates
ADEQUATE evidence correctly. It does NOT provide an independent validation
estimate and is NOT an unbiased upper bound -- gold_evidence_set_v1's
packages were built AFTER inspecting Arm 1/Arm 2 disagreements
(outputs/prompt6_disagreement_audit_v1.md), and some expansions were
themselves informed by retrieved evidence from those same disagreement
investigations. Every report using this arm's results must carry this
caveat, verbatim or in substance -- see ARM2B_METHOD_CAVEAT below.

--------------------------------------------------------------------------
LEAKAGE CONTROL (the whole reason this is a SEPARATE script/arm from
run_arm2_gold_evidence_prompt6.py, not a flag on it):

gold_evidence_set_v1's GoldEvidenceEntry/GoldEvidenceExcerpt records carry
several ANSWER-LIKE human judgment fields alongside the actual evidence
text -- stance, supported_inference, review_notes, sufficiency fields,
human_attestation -- plus the underlying GoldCriterionAnnotation's own
gold_status/gold_final_decision/exclusion-reason fields. This script sends
the MODEL only:
  - excerpt text (GoldEvidenceExcerpt.heading_or_quotation -- the only
    field verbatim quotations live in; body_excerpt is unused, see
    promote_gold_evidence_set_from_worksheet.py RULE 6)
  - page / source location (GoldEvidenceExcerpt.page / .source_section)
  - a NEUTRAL chunk_id (f"arm2b-{document_id}-{criterion_id}-{ordinal}" --
    contains no stance/role token; contrast with the EXISTING
    _gold_evidence_spans() in evaluation_harness.py, which embeds
    "-supports-"/"-contradicts-" directly into ITS chunk_id strings --
    those never reach the model either, since _build_prompt() only ever
    renders {chunk_id, section, page, excerpt} into the prompt JSON, but
    Arm 2b's chunk_ids are neutral even at the OBJECT level, not just
    because of what _build_prompt happens to omit)
  - EvidenceSpan.stance is set to "relevant" UNCONDITIONALLY for every
    Arm 2b evidence span, regardless of the underlying excerpt's real
    stance -- deliberately discarding that information rather than
    threading it through, even though _build_prompt() already never
    renders .stance into the prompt (adjudication.py:560-563) and the
    model's own supporting_chunk_ids/contradicting_chunk_ids response
    OVERWRITES it on the bound evidence returned in the outcome anyway
    (adjudication.py:789/793) -- belt-and-braces: this field is neutral
    even as an in-memory object, not merely omitted from a render step.
  - evidence modality is deliberately NOT sent (the instruction allowed
    it optionally) -- every excerpt in gold_evidence_set_v1 is real text
    (the one figure/OCR case was human-transcribed INTO text), so there
    is nothing a modality tag would usefully distinguish here, and adding
    an unused field would only be one more thing to prove doesn't leak.

`build_evidence_projection()` freezes the COMPLETE model-visible
projection (exactly the 4 fields above, per chunk, for all 60 keys) as
its own artifact (Arm2bEvidenceProjection, `arm2b_evidence_projection_v1.json`)
BEFORE any API call -- during preflight(), zero network access required --
specifically so a human/auditor can read that one file and see the exact
text this run sent, rather than trusting this docstring's claim. Preflight
is idempotent: a second run reloads and hash-verifies the existing
projection rather than re-freezing or silently trusting it.

`gold_evidence_set_v1` is LOCKED to content_hash
eb7ac00b2fd1d6456b8bdf2ac58d67f58af01ae65ceaff42bd7c53c2512fc2db and its
promotion manifest is loaded and verified (gold_set_hash/draft binding
consistency) -- see EXPECTED_GOLD_EVIDENCE_SET_HASH below.

--------------------------------------------------------------------------
WHAT'S DELIBERATELY UNCHANGED: model=gpt-5.6-terra, reasoning_effort=
"medium", max_completion_tokens=4096, adjudication-prompt-6/
adjudication-schema-2 -- IDENTICAL to run_arm2_gold_evidence_prompt6.py's
locked config, verified against live adjudicator constants the same way.
Checkpointing reuses evaluation_harness.py's OWN
GoldEvidenceCriterionCheckpoint / _gold_evidence_checkpoint_path /
GoldEvidenceCheckpointDriftError verbatim (imported, not reimplemented).
Scoring reuses evaluation_harness.py's OWN
build_criterion_status_predictions_from_gold_evidence /
compute_arm_gold_evidence_document_decisions / criterion_status_agreement /
false_exclusion_rate / false_inclusion_rate / correct_exclusion_reason_rate /
review_rate / token_usage_summary_gold_evidence(_per_document) /
wall_time_summary_gold_evidence -- ALL evidence-source-agnostic (they only
ever read `gold_set` for the key list and `ArmGoldEvidenceResult` for
outcomes), so every one is reused UNCHANGED, verbatim.

evaluation_harness.py's `run_arm_gold_evidence()` hardcodes
`_gold_evidence_spans()` (the raw-GoldSet-citation evidence source) with
no injection point for a different evidence source, so this script cannot
call it directly -- it reimplements ONLY the orchestration loop (mirroring
run_arm_gold_evidence's structure exactly: same checkpoint-then-call-then-
checkpoint sequence, same EvaluationResultArtifact/EvaluationRunManifest
construction, same freeze_frozen_model persistence order), swapping in
`_redacted_evidence_spans_for_key()` as the thing that differs.

ONE small, SHARED fix did land in evaluation_harness.py itself
(`compute_arm_gold_evidence_document_decisions()` now derives its
`document_sha256` from the assessments' own evidence, falling back to
`gold_set.document_pdf_sha256` only when a document cites no evidence at
all -- see `_document_sha256_for_decision()`). The first real Arm 2b run
crashed here: this script's evidence carries the REAL PDF hash (more
honest provenance than the historical "0"*64 placeholder every other
gold_evidence arm's evidence uses), but the function hardcoded "0"*64
unconditionally, so `aggregation.build_eligibility_result()`'s own
evidence-identity check failed on every document. Fixed in the SHARED
function (correct for every caller, including the pre-existing "0"*64
convention, which is simply what gets derived when that's what the
evidence carries) rather than as a runner-side workaround -- an earlier,
uncommitted version of this fix lived only here, ephemeral-normalizing
the arm result before calling the (then-still-hardcoded) shared function;
that version's result is preserved as
`arm2b_gold_evidence_set_result_v1.json` (a historical incident record,
superseded by `..._v2.json`, produced entirely offline from the same
underlying 60 checkpoints -- zero new provider calls).

Distinct run identity throughout: OUTPUT_DIR, RESULT_VERSION_LABEL,
MANIFEST_VERSION_LABEL, CHECKPOINT_DIR, and EVIDENCE_PROJECTION_VERSION_LABEL
all use an "arm2b_" prefix -- never collides with Arm 2/Arm 2 (prompt-6)'s
own artifacts, which this script never reads or touches.

Usage:
    python run_arm2b_gold_evidence_set.py             # preflight only, zero API calls
    python run_arm2b_gold_evidence_set.py --execute    # runs Arm 2b for real (up to 60 paid
                                                        # OpenAI adjudication calls, resuming
                                                        # from any existing per-criterion checkpoints)
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import sys
import time
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

sys.path.insert(0, r"C:/projects/FT_SCREENING/src")
sys.path.insert(0, r"C:/projects/FT_SCREENING/scripts")

from ft_screening.adjudication import LLMCriterionAdjudicator, OpenAIChatCompletionTransport  # noqa: E402
from ft_screening.evaluation_harness import (  # noqa: E402
    ArmGoldEvidenceResult,
    EvaluationResultArtifact,
    EvaluationRunManifest,
    GoldEvidenceCheckpointDriftError,
    GoldEvidenceCriterionCheckpoint,
    _gold_evidence_checkpoint_path,
    _now_iso,
    _placeholder_document,
    build_criterion_status_predictions_from_gold_evidence,
    compute_arm_gold_evidence_document_decisions,
    correct_exclusion_reason_rate,
    criterion_status_agreement,
    false_exclusion_rate,
    false_inclusion_rate,
    final_decision_agreement,
    make_evidence_key,
    processing_completion_report,
    review_rate,
    token_usage_summary_gold_evidence,
    token_usage_summary_gold_evidence_per_document,
    wall_time_summary_gold_evidence,
)
from ft_screening.gold import GoldSet, freeze_frozen_model, load_frozen_gold_set, load_frozen_model  # noqa: E402
from ft_screening.gold_evidence_set import GoldEvidenceEntry, GoldEvidenceSet  # noqa: E402
from ft_screening.hashing import canonical_json_bytes  # noqa: E402
from ft_screening.schema import EvidenceSpan  # noqa: E402

from promote_gold_evidence_set_from_worksheet import GoldEvidenceSetPromotionManifest  # noqa: E402
from source_identity import capture_source_identity  # noqa: E402

# NOTE: my_api_keys is deliberately NOT imported at module level -- see
# _load_credentials() inside execute(). Module import and preflight() must
# never load or touch credentials.

GOLD_SET_DIR = Path(r"C:/projects/FT_SCREENING/gold_set")
FROZEN_GOLD_SET_PATH = GOLD_SET_DIR / "frozen" / "gold_set_v1.json"
EXPECTED_GOLD_SET_HASH = "6fdfaebfd84e7ed33101d1a9b9ba4960367ab6985f1a52e3a13cabc2e396280a"

GOLD_EVIDENCE_SET_DIR = Path(r"C:/projects/FT_SCREENING/outputs/gold_evidence_set_v1")
FROZEN_GOLD_EVIDENCE_SET_PATH = GOLD_EVIDENCE_SET_DIR / "gold_evidence_set_v1.json"
PROMOTION_MANIFEST_PATH = GOLD_EVIDENCE_SET_DIR / "gold_evidence_set_v1_promotion_manifest.json"
# LOCKED, per explicit instruction -- the exact content_hash gold_evidence_set_v1
# was frozen under (promote_gold_evidence_set_from_worksheet.py --freeze, this session).
EXPECTED_GOLD_EVIDENCE_SET_HASH = "eb7ac00b2fd1d6456b8bdf2ac58d67f58af01ae65ceaff42bd7c53c2512fc2db"
EXPECTED_CRITERION_ANNOTATION_COUNT = 60  # 20 documents x 3 criteria

OUTPUT_DIR = Path(r"C:/projects/FT_SCREENING/outputs/arm2b_gold_evidence_set_v1")
RESULT_VERSION_LABEL = "arm2b_gold_evidence_set_result_v1"
MANIFEST_VERSION_LABEL = "arm2b_gold_evidence_set_manifest_v1"
# v2: produced entirely offline (--finalize-from-checkpoints) from the SAME
# 60 checkpoints v1 used -- v1 is preserved unchanged as the historical
# incident record (see finalize_from_checkpoints()'s own docstring); v2 is
# the canonical, provenance-clean artifact.
RESULT_VERSION_LABEL_V2 = "arm2b_gold_evidence_set_result_v2"
MANIFEST_VERSION_LABEL_V2 = "arm2b_gold_evidence_set_manifest_v2"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
EVIDENCE_PROJECTION_VERSION_LABEL = "arm2b_evidence_projection_v1"

ARM2B_EVIDENCE_CONSTRUCTION_VERSION = "arm2b-human-reviewed-evidence-v1"
ARM2B_RETRIEVAL_METHOD = "gold_evidence_set_human_reviewed"

# Locked candidate configuration -- IDENTICAL to run_arm2_gold_evidence_prompt6.py's,
# deliberately held fixed so this arm is comparable to Arm 1/Arm 2 (prompt-6);
# only the evidence SOURCE (and its redaction) differs.
LOCKED_ADJUDICATOR_MODEL = "gpt-5.6-terra"
LOCKED_REASONING_EFFORT = "medium"
LOCKED_MAX_COMPLETION_TOKENS = 4096
LOCKED_PROMPT_TEMPLATE_VERSION = "adjudication-prompt-6"
LOCKED_OUTPUT_SCHEMA_VERSION = "adjudication-schema-2"
PROVIDER_IDENTITY = "openai"

ARM2B_METHOD_CAVEAT = (
    "ARM 2b METHOD (mandatory on every report using these results): \"LLM adjudication using "
    "retrospectively curated, human-reviewed evidence packages.\" This is NOT an unbiased upper bound -- "
    "gold_evidence_set_v1's evidence packages were built AFTER inspecting Arm 1/Arm 2 disagreements "
    "(outputs/prompt6_disagreement_audit_v1.md), and some expansions were themselves informed by "
    "retrieved evidence surfaced during that same investigation. Arm 2b can answer whether prompt-6 "
    "adjudicates ADEQUATE evidence correctly -- it does not provide an independent validation estimate."
)

LABEL_MASKED_CAVEAT = (
    "CAVEAT (gold_set_protocol_v9.md section 1): the underlying gold set is LABEL-MASKED, NOT BLIND -- "
    "the annotator had prior exposure to 7/20 documents' Stage-1 labels. Results below are "
    "EXPLORATORY/PILOT ONLY. No acceptance thresholds are pre-registered against this run."
)


class LockedConfigDriftError(RuntimeError):
    """Raised when a hardcoded LOCKED_* literal no longer matches the live
    class constant it's supposed to pin. A real exception, not an assert --
    assert is stripped entirely under `python -O` (PYTHONOPTIMIZE)."""


def verify_locked_config() -> None:
    if LOCKED_PROMPT_TEMPLATE_VERSION != LLMCriterionAdjudicator.PROMPT_TEMPLATE_VERSION:
        raise LockedConfigDriftError(
            f"locked prompt_template_version={LOCKED_PROMPT_TEMPLATE_VERSION!r} no longer matches live "
            f"LLMCriterionAdjudicator.PROMPT_TEMPLATE_VERSION={LLMCriterionAdjudicator.PROMPT_TEMPLATE_VERSION!r}"
        )
    if LOCKED_OUTPUT_SCHEMA_VERSION != LLMCriterionAdjudicator.OUTPUT_SCHEMA_VERSION:
        raise LockedConfigDriftError(
            f"locked output_schema_version={LOCKED_OUTPUT_SCHEMA_VERSION!r} no longer matches live "
            f"LLMCriterionAdjudicator.OUTPUT_SCHEMA_VERSION={LLMCriterionAdjudicator.OUTPUT_SCHEMA_VERSION!r}"
        )


# ---------------------------------------------------------------------------
# Model-visible evidence projection -- the auditable "exactly this, nothing
# else" artifact. Field set matches _build_prompt()'s own evidence_block
# shape 1:1 (adjudication.py:560-563: chunk_id/section/page/excerpt) -- not
# "morally equivalent", the SAME four fields, so this really is a record of
# what left the process boundary, not an approximation of it.
# ---------------------------------------------------------------------------


class Arm2bEvidenceChunk(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str = Field(min_length=1)
    page: int | None = Field(default=None, ge=1)
    section: str | None = None
    excerpt: str = Field(min_length=1)


class Arm2bEvidenceProjectionEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str = Field(min_length=1)
    criterion_id: str = Field(min_length=1)
    chunks: tuple[Arm2bEvidenceChunk, ...] = ()


class Arm2bEvidenceProjection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    gold_set_hash: str = Field(min_length=64, max_length=64)
    gold_evidence_set_hash: str = Field(min_length=64, max_length=64)
    evidence_construction_version: str = Field(min_length=1)
    entries: tuple[Arm2bEvidenceProjectionEntry, ...]
    frozen_at: str = Field(min_length=1)

    def content_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"frozen_at"})
        return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _redacted_evidence_spans_for_key(
    *, document_id: str, criterion_id: str, entry: GoldEvidenceEntry, document_sha256: str,
) -> list[EvidenceSpan]:
    """The ONLY function in this script that reads a GoldEvidenceEntry's
    self_contained_excerpts. Reads EXACTLY two fields off each excerpt --
    .heading_or_quotation (the only field verbatim text lives in; see
    RULE 6 in promote_gold_evidence_set_from_worksheet.py) and .page/
    .source_section -- and NEVER reads .stance, .supported_inference,
    .evidence_role, .extraction_method, .source_document_sha256, or
    anything on the ENTRY itself (review_notes, sufficiency fields,
    human_attestation, reviewer_id/reviewed_at, original_*_passages)."""
    spans: list[EvidenceSpan] = []
    for i, excerpt in enumerate(entry.self_contained_excerpts):
        spans.append(EvidenceSpan(
            document_id=document_id, document_sha256=document_sha256,
            chunk_id=f"arm2b-{document_id}-{criterion_id}-{i}",
            section=excerpt.source_section, page=excerpt.page, excerpt=excerpt.heading_or_quotation,
            stance="relevant",  # deliberately neutral for every span -- see module docstring.
            retrieval_method=ARM2B_RETRIEVAL_METHOD, retrieval_query_version=ARM2B_EVIDENCE_CONSTRUCTION_VERSION,
        ))
    return spans


def build_evidence_projection(gold_set: GoldSet, gold_evidence_set: GoldEvidenceSet) -> Arm2bEvidenceProjection:
    """Pure, zero-API-call: constructs the complete model-visible
    projection for all 60 keys. Callable and inspectable entirely offline."""
    entries_by_key = {(e.document_id, e.criterion_id): e for e in gold_evidence_set.entries}
    projection_entries = []
    for ann in gold_set.criterion_annotations:
        key = (ann.document_id, ann.criterion_id)
        entry = entries_by_key[key]
        document_sha256 = gold_set.document_pdf_sha256[ann.document_id]
        spans = _redacted_evidence_spans_for_key(
            document_id=ann.document_id, criterion_id=ann.criterion_id, entry=entry, document_sha256=document_sha256,
        )
        chunks = tuple(Arm2bEvidenceChunk(chunk_id=s.chunk_id, page=s.page, section=s.section, excerpt=s.excerpt) for s in spans)
        projection_entries.append(Arm2bEvidenceProjectionEntry(document_id=ann.document_id, criterion_id=ann.criterion_id, chunks=chunks))
    return Arm2bEvidenceProjection(
        gold_set_hash=gold_set.content_hash(), gold_evidence_set_hash=gold_evidence_set.content_hash(),
        evidence_construction_version=ARM2B_EVIDENCE_CONSTRUCTION_VERSION, entries=tuple(projection_entries), frozen_at=_now_iso(),
    )


def ensure_evidence_projection_frozen(
    gold_set: GoldSet, gold_evidence_set: GoldEvidenceSet, *, output_dir: Path,
) -> tuple[Arm2bEvidenceProjection, Path]:
    """Idempotent: builds the projection (pure, offline), and either
    verifies it against an already-frozen copy (raising on any mismatch)
    or freezes it fresh -- never silently overwrites, never re-derives a
    DIFFERENT projection for the same run without raising."""
    projection = build_evidence_projection(gold_set, gold_evidence_set)
    path = output_dir / f"{EVIDENCE_PROJECTION_VERSION_LABEL}.json"
    if path.exists():
        existing = load_frozen_model(Arm2bEvidenceProjection, path, expected_artifact_type="Arm2bEvidenceProjection")
        if existing.content_hash() != projection.content_hash():
            raise ValueError(
                f"existing evidence projection at {path} has content_hash={existing.content_hash()!r}, which "
                f"does not match this run's freshly-built projection content_hash={projection.content_hash()!r} "
                "-- refusing to overwrite"
            )
        return projection, path
    output_dir.mkdir(parents=True, exist_ok=True)
    freeze_frozen_model(projection, output_dir=output_dir, version_label=EVIDENCE_PROJECTION_VERSION_LABEL, artifact_type="Arm2bEvidenceProjection")
    return projection, path


class PreflightResult:
    def __init__(self, *, gold_set: GoldSet, gold_evidence_set: GoldEvidenceSet, evidence_projection: Arm2bEvidenceProjection, model_options: dict, source_identity) -> None:
        self.gold_set = gold_set
        self.gold_evidence_set = gold_evidence_set
        self.evidence_projection = evidence_projection
        self.model_options = model_options
        self.source_identity = source_identity


def preflight() -> PreflightResult:
    """Loads and verifies the frozen GoldSet, the LOCKED gold_evidence_set_v1
    hash, and its promotion manifest; verifies exact 60-key coverage on
    both; builds and (idempotently) freezes the model-visible evidence
    projection; verifies the locked adjudicator config; captures source
    identity; checks for output collisions. Makes ZERO network calls and
    constructs no adjudicator/transport."""
    print("=== Arm 2b (gold_evidence_set_v1 -> LLM adjudication) preflight ===\n")

    gold_set = load_frozen_gold_set(FROZEN_GOLD_SET_PATH, expected_hash=EXPECTED_GOLD_SET_HASH)
    print(f"GoldSet loaded and verified: content_hash={gold_set.content_hash()}")

    gold_evidence_set = load_frozen_model(GoldEvidenceSet, FROZEN_GOLD_EVIDENCE_SET_PATH, expected_artifact_type="GoldEvidenceSet")
    if gold_evidence_set.content_hash() != EXPECTED_GOLD_EVIDENCE_SET_HASH:
        raise ValueError(
            f"gold_evidence_set_v1.json has content_hash={gold_evidence_set.content_hash()!r}, which does not "
            f"match the LOCKED EXPECTED_GOLD_EVIDENCE_SET_HASH={EXPECTED_GOLD_EVIDENCE_SET_HASH!r} -- refusing "
            "to run Arm 2b against an evidence set other than the one it was explicitly locked to"
        )
    if gold_evidence_set.gold_set_hash != gold_set.content_hash():
        raise ValueError("gold_evidence_set_v1.gold_set_hash does not match the loaded gold_set_v1.json's own content_hash()")
    print(f"GoldEvidenceSet loaded and LOCKED-hash-verified: content_hash={gold_evidence_set.content_hash()}")

    manifest = load_frozen_model(GoldEvidenceSetPromotionManifest, PROMOTION_MANIFEST_PATH, expected_artifact_type="GoldEvidenceSetPromotionManifest")
    if manifest.evidence_set_content_hash != gold_evidence_set.content_hash():
        raise ValueError("promotion manifest's evidence_set_content_hash does not match the loaded gold_evidence_set_v1's content_hash()")
    if manifest.gold_set_hash != gold_set.content_hash():
        raise ValueError("promotion manifest's gold_set_hash does not match the loaded gold_set_v1's content_hash()")
    print(f"Promotion manifest verified: worksheet_sha256={manifest.worksheet_sha256}, draft_content_hash={manifest.draft_content_hash}")

    if len(gold_set.criterion_annotations) != EXPECTED_CRITERION_ANNOTATION_COUNT:
        raise ValueError(f"expected exactly {EXPECTED_CRITERION_ANNOTATION_COUNT} criterion annotations, got {len(gold_set.criterion_annotations)}")
    gold_set_keys = {(a.document_id, a.criterion_id) for a in gold_set.criterion_annotations}
    evidence_set_keys = {(e.document_id, e.criterion_id) for e in gold_evidence_set.entries}
    if gold_set_keys != evidence_set_keys:
        raise ValueError(f"gold_set and gold_evidence_set key sets differ -- missing={sorted(gold_set_keys - evidence_set_keys)}, extra={sorted(evidence_set_keys - gold_set_keys)}")
    print(f"Exact coverage verified: {len(gold_set_keys)} keys on both GoldSet and GoldEvidenceSet.")

    evidence_projection, projection_path = ensure_evidence_projection_frozen(gold_set, gold_evidence_set, output_dir=OUTPUT_DIR)
    n_chunks = sum(len(e.chunks) for e in evidence_projection.entries)
    print(f"\nModel-visible evidence projection: {projection_path}")
    print(f"  content_hash={evidence_projection.content_hash()}")
    print(f"  {len(evidence_projection.entries)} entries, {n_chunks} total evidence chunks (fields per chunk: chunk_id, page, section, excerpt -- nothing else)")

    verify_locked_config()
    model_options = {"max_completion_tokens": LOCKED_MAX_COMPLETION_TOKENS, "reasoning_effort": LOCKED_REASONING_EFFORT}
    print("\nLocked config (IDENTICAL to run_arm2_gold_evidence_prompt6.py's -- only the evidence source/redaction differs):")
    print(f"  adjudicator_model={LOCKED_ADJUDICATOR_MODEL!r}  model_options={model_options!r}")
    print(f"  prompt_template_version={LOCKED_PROMPT_TEMPLATE_VERSION!r}  output_schema_version={LOCKED_OUTPUT_SCHEMA_VERSION!r}")
    print(f"  provider_identity={PROVIDER_IDENTITY!r}")

    source_identity = capture_source_identity()
    print(f"\nSource identity: git_commit={source_identity.git_commit}, git_dirty_relevant_files={source_identity.git_dirty_relevant_files}")

    # Output-collision checking is deliberately NOT done here -- it depends
    # on which ACTION the caller intends (--execute writes the v1-labeled
    # pair; --finalize-from-checkpoints writes the v2-labeled pair), and
    # preflight() itself must stay usable regardless of which pair already
    # exists. See _check_no_output_collision(), called by each action itself.
    existing_checkpoints = sorted(p.stem for p in CHECKPOINT_DIR.glob("*.json")) if CHECKPOINT_DIR.exists() else []
    if existing_checkpoints:
        print(f"{len(existing_checkpoints)} existing per-criterion checkpoint(s) found under {CHECKPOINT_DIR} -- will be verified and reused, not re-executed.")

    n_calls = len(gold_set.criterion_annotations)
    print(f"\n{n_calls} adjudication calls planned, {n_calls - len(existing_checkpoints)} of which require a real call this run.")

    print("\n" + ARM2B_METHOD_CAVEAT)
    print("\n" + LABEL_MASKED_CAVEAT)
    return PreflightResult(gold_set=gold_set, gold_evidence_set=gold_evidence_set, evidence_projection=evidence_projection, model_options=model_options, source_identity=source_identity)


def _check_no_output_collision(result_version_label: str, manifest_version_label: str) -> None:
    result_path = OUTPUT_DIR / f"{result_version_label}.json"
    manifest_path = OUTPUT_DIR / f"{manifest_version_label}.json"
    collisions = [p for p in (result_path, manifest_path) if p.exists()]
    if collisions:
        raise FileExistsError(f"refusing to proceed: {[str(p) for p in collisions]} already exist -- this run was already completed and frozen.")
    print(f"No output collisions: {OUTPUT_DIR} is clear for {result_version_label}/{manifest_version_label}.")


def _load_credentials() -> None:
    sys.path.insert(0, r"C:/projects")
    with contextlib.redirect_stdout(io.StringIO()):
        import my_api_keys  # noqa: F401 -- import-for-side-effect (sets os.environ)

    import os

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in the environment after loading my_api_keys -- cannot execute Arm 2b")
    print("Credentials loaded (OPENAI_API_KEY present; no key names or values printed).")


def run_arm2b(
    *, gold_set: GoldSet, gold_evidence_set: GoldEvidenceSet, adjudicator, model_options: dict,
    provider_identity: str, evidence_projection_hash: str, manifest_output_dir: Path,
    result_version_label: str, manifest_version_label: str, checkpoint_dir: Path,
    extra_config_summary_fields: dict | None = None,
) -> tuple[ArmGoldEvidenceResult, EvaluationResultArtifact, EvaluationRunManifest]:
    """Mirrors evaluation_harness.run_arm_gold_evidence()'s structure
    exactly (same checkpoint-then-call-then-checkpoint sequence, same
    artifact construction/persistence order) -- the ONLY difference is
    the evidence source: _redacted_evidence_spans_for_key() (reading
    gold_evidence_set) instead of _gold_evidence_spans() (reading raw
    gold_set citations, which run_arm_gold_evidence hardcodes with no
    injection point). evaluation_harness.py itself is not modified."""
    provider_identity = provider_identity.strip()
    if not provider_identity:
        raise ValueError("provider_identity must be a nonblank string")

    fingerprint_payload = {
        "gold_set_hash": gold_set.content_hash(),
        "gold_evidence_set_hash": gold_evidence_set.content_hash(),
        "evidence_projection_hash": evidence_projection_hash,
        "evidence_construction_version": ARM2B_EVIDENCE_CONSTRUCTION_VERSION,
        "provider_identity": provider_identity,
        "adjudicator_model": getattr(adjudicator, "model", None),
        "prompt_template_version": getattr(adjudicator, "prompt_template_version", None),
        "output_schema_version": getattr(adjudicator, "output_schema_version", None),
        "model_options": model_options,
    }
    execution_fingerprint = hashlib.sha256(canonical_json_bytes(fingerprint_payload)).hexdigest()

    criteria_by_id = {c.criterion_id: c for c in gold_set.criteria_set.criteria}
    entries_by_key = {(e.document_id, e.criterion_id): e for e in gold_evidence_set.entries}
    outcomes: dict[tuple[str, str], object] = {}
    wall_by_doc: dict[str, float] = defaultdict(float)

    for annotation in gold_set.criterion_annotations:
        key = (annotation.document_id, annotation.criterion_id)

        checkpoint_path = _gold_evidence_checkpoint_path(checkpoint_dir, document_id=annotation.document_id, criterion_id=annotation.criterion_id)
        if checkpoint_path.exists():
            record = load_frozen_model(GoldEvidenceCriterionCheckpoint, checkpoint_path, expected_artifact_type="GoldEvidenceCriterionCheckpoint")
            if record.execution_identity_hash != execution_fingerprint:
                raise GoldEvidenceCheckpointDriftError(
                    f"checkpoint {checkpoint_path} was frozen under execution_identity_hash={record.execution_identity_hash!r}, "
                    f"which does not match the current run's {execution_fingerprint!r}"
                )
            outcomes[key] = record.outcome
            continue

        criterion = criteria_by_id[annotation.criterion_id]
        document_sha256 = gold_set.document_pdf_sha256[annotation.document_id]
        evidence = _redacted_evidence_spans_for_key(
            document_id=annotation.document_id, criterion_id=annotation.criterion_id,
            entry=entries_by_key[key], document_sha256=document_sha256,
        )
        document = _placeholder_document(annotation.document_id)
        started = time.monotonic()
        try:
            outcome = adjudicator.adjudicate(document=document, criterion=criterion, evidence=evidence, model_options=model_options)
        except Exception:  # noqa: BLE001 -- one failed criterion must not abort the whole arm
            outcome = None
        wall_by_doc[annotation.document_id] += time.monotonic() - started
        outcomes[key] = outcome

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
    usage_billable = token_usage_summary_gold_evidence(gold_set=gold_set, arm_result=arm_result)
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


def execute(pf: PreflightResult) -> None:
    _check_no_output_collision(RESULT_VERSION_LABEL, MANIFEST_VERSION_LABEL)
    _load_credentials()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    adjudicator = LLMCriterionAdjudicator(transport=OpenAIChatCompletionTransport(), model=LOCKED_ADJUDICATOR_MODEL)

    n_calls = len(pf.gold_set.criterion_annotations)
    print(f"\n=== Executing Arm 2b: {n_calls} adjudication calls, resuming from checkpoints ===\n")

    arm_result, result_artifact, manifest = run_arm2b(
        gold_set=pf.gold_set, gold_evidence_set=pf.gold_evidence_set, adjudicator=adjudicator, model_options=pf.model_options,
        provider_identity=PROVIDER_IDENTITY, evidence_projection_hash=pf.evidence_projection.content_hash(),
        manifest_output_dir=OUTPUT_DIR, result_version_label=RESULT_VERSION_LABEL, manifest_version_label=MANIFEST_VERSION_LABEL,
        checkpoint_dir=CHECKPOINT_DIR, extra_config_summary_fields={"source_identity": pf.source_identity.model_dump()},
    )
    print(f"EvaluationResultArtifact frozen: content_hash={result_artifact.content_hash()}")
    print(f"EvaluationRunManifest frozen: content_hash={manifest.content_hash()}")

    criterion_predictions = build_criterion_status_predictions_from_gold_evidence(gold_set=pf.gold_set, arm_result=arm_result)
    document_decisions = compute_arm_gold_evidence_document_decisions(gold_set=pf.gold_set, arm_result=arm_result)
    predicted_decision_by_document = {d: r.final_decision for d, r in document_decisions.items()}

    completion = processing_completion_report(
        gold_set=pf.gold_set, predicted_status_by_key=criterion_predictions,
        predicted_decision_by_document=predicted_decision_by_document,
    )
    criterion_matrix = criterion_status_agreement(gold_set=pf.gold_set, predicted_status_by_key=criterion_predictions)
    decision_matrix = final_decision_agreement(gold_set=pf.gold_set, predicted_decision_by_document=predicted_decision_by_document)
    false_excl = false_exclusion_rate(gold_set=pf.gold_set, predicted_decision_by_document=predicted_decision_by_document)
    false_incl = false_inclusion_rate(gold_set=pf.gold_set, predicted_decision_by_document=predicted_decision_by_document)
    reason_rate = correct_exclusion_reason_rate(gold_set=pf.gold_set, document_decisions=document_decisions)
    rr = review_rate(gold_set=pf.gold_set, predicted_decision_by_document=predicted_decision_by_document)
    usage_billable = token_usage_summary_gold_evidence(gold_set=pf.gold_set, arm_result=arm_result)
    usage_per_doc = token_usage_summary_gold_evidence_per_document(gold_set=pf.gold_set, arm_result=arm_result)
    wall = wall_time_summary_gold_evidence(arm_result)

    print("\n--- Metrics (via evaluation_harness, not reimplemented) ---")
    print(f"criterion_status_agreement accuracy = {criterion_matrix.accuracy():.4f}")
    print(f"final_decision_agreement accuracy = {decision_matrix.accuracy():.4f}")
    print(f"false_exclusion_rate = {false_excl.rate:.4f}  (documents: {false_excl.document_ids})")
    print(f"false_inclusion_rate = {false_incl.rate:.4f}  (documents: {false_incl.document_ids})")
    print(f"correct_exclusion_reason_rate = {reason_rate:.4f}")
    print(f"review_rate = {rr.review_rate:.4f}  (hard cases: {rr.review_rate_among_gold_hard_cases:.4f}, easy cases: {rr.review_rate_among_gold_easy_cases:.4f})")
    print(f"completed documents = {len(completion.completed_document_ids)}/{len(pf.gold_set.expected_document_ids)}  missing/failed = {completion.missing_or_failed_document_ids}")
    print(f"completed criteria = {len(completion.completed_criterion_keys)}/{n_calls}  missing/failed = {completion.missing_or_failed_criterion_keys}")
    print(f"token usage (billable): input={usage_billable.total_input_tokens} output={usage_billable.total_output_tokens} total={usage_billable.total_tokens}")
    print(f"wall time: {wall.total_seconds:.1f}s total")

    print("\n" + ARM2B_METHOD_CAVEAT)
    print("\n" + LABEL_MASKED_CAVEAT)


def _load_all_checkpoints(gold_set: GoldSet, *, checkpoint_dir: Path, expected_execution_fingerprint: str) -> ArmGoldEvidenceResult:
    """Loads every (document_id, criterion_id) outcome STRICTLY from
    existing checkpoint files -- makes ZERO provider calls (no adjudicator
    is ever constructed or referenced here), and raises
    GoldEvidenceCheckpointDriftError if any checkpoint was frozen under a
    different execution identity than expected (same identity discipline
    run_arm2b()'s own resume path already applies), or a plain ValueError
    if any checkpoint is simply missing. wall_seconds_by_document is
    intentionally empty -- checkpoint files never recorded per-call wall
    time to begin with, so offline finalization cannot honestly report
    any; see wall_time_unavailable_after_checkpoint_resume in the v2
    manifest's config_summary."""
    outcomes: dict[tuple[str, str], object] = {}
    missing: list[tuple[str, str]] = []
    for annotation in gold_set.criterion_annotations:
        key = (annotation.document_id, annotation.criterion_id)
        checkpoint_path = _gold_evidence_checkpoint_path(checkpoint_dir, document_id=annotation.document_id, criterion_id=annotation.criterion_id)
        if not checkpoint_path.exists():
            missing.append(key)
            continue
        record = load_frozen_model(GoldEvidenceCriterionCheckpoint, checkpoint_path, expected_artifact_type="GoldEvidenceCriterionCheckpoint")
        if record.execution_identity_hash != expected_execution_fingerprint:
            raise GoldEvidenceCheckpointDriftError(
                f"checkpoint {checkpoint_path} was frozen under execution_identity_hash={record.execution_identity_hash!r}, "
                f"which does not match the expected {expected_execution_fingerprint!r} -- refusing to finalize from a "
                "checkpoint set that doesn't match this run's own identity"
            )
        outcomes[key] = record.outcome
    if missing:
        raise ValueError(f"cannot finalize offline: {len(missing)} checkpoint(s) missing: {sorted(missing)} -- run --execute first")
    return ArmGoldEvidenceResult(outcomes_by_key=outcomes, wall_seconds_by_document={})


def finalize_from_checkpoints(pf: PreflightResult) -> None:
    """Re-finalizes Arm 2b's scoring/persistence step ENTIRELY OFFLINE
    from the 60 already-completed checkpoints -- constructs no adjudicator
    or transport, loads no credentials, makes ZERO new provider calls.
    Produces result_v2/manifest_v2 using the NOW-FIXED shared
    evaluation_harness.compute_arm_gold_evidence_document_decisions()
    (see the module docstring's "ONE small, SHARED fix" section) --
    the runner-side normalization the original v1 run needed no longer
    exists or is needed. v1 is loaded (never modified) purely to bind its
    hashes into v2's own provenance."""
    v1_result_path = OUTPUT_DIR / f"{RESULT_VERSION_LABEL}.json"
    v1_manifest_path = OUTPUT_DIR / f"{MANIFEST_VERSION_LABEL}.json"
    if not (v1_result_path.exists() and v1_manifest_path.exists()):
        raise FileNotFoundError(f"expected historical v1 artifacts at {v1_result_path} / {v1_manifest_path} to bind into v2 provenance -- run --execute first")
    v1_result = load_frozen_model(EvaluationResultArtifact, v1_result_path, expected_artifact_type="EvaluationResultArtifact")
    v1_manifest = load_frozen_model(EvaluationRunManifest, v1_manifest_path, expected_artifact_type="EvaluationRunManifest")
    print(f"Loaded historical v1 (preserved unchanged): result_content_hash={v1_result.content_hash()}, manifest_content_hash={v1_manifest.content_hash()}")

    _check_no_output_collision(RESULT_VERSION_LABEL_V2, MANIFEST_VERSION_LABEL_V2)

    fingerprint_payload = {
        "gold_set_hash": pf.gold_set.content_hash(),
        "gold_evidence_set_hash": pf.gold_evidence_set.content_hash(),
        "evidence_projection_hash": pf.evidence_projection.content_hash(),
        "evidence_construction_version": ARM2B_EVIDENCE_CONSTRUCTION_VERSION,
        "provider_identity": PROVIDER_IDENTITY,
        "adjudicator_model": LOCKED_ADJUDICATOR_MODEL,
        "prompt_template_version": LOCKED_PROMPT_TEMPLATE_VERSION,
        "output_schema_version": LOCKED_OUTPUT_SCHEMA_VERSION,
        "model_options": pf.model_options,
    }
    execution_fingerprint = hashlib.sha256(canonical_json_bytes(fingerprint_payload)).hexdigest()

    n_calls = len(pf.gold_set.criterion_annotations)
    print(f"\n=== Finalizing Arm 2b OFFLINE from {n_calls} existing checkpoints -- ZERO new provider calls ===\n")
    arm_result = _load_all_checkpoints(pf.gold_set, checkpoint_dir=CHECKPOINT_DIR, expected_execution_fingerprint=execution_fingerprint)

    criterion_predictions = build_criterion_status_predictions_from_gold_evidence(gold_set=pf.gold_set, arm_result=arm_result)
    document_decisions = compute_arm_gold_evidence_document_decisions(gold_set=pf.gold_set, arm_result=arm_result)
    completion = processing_completion_report(
        gold_set=pf.gold_set, predicted_status_by_key=criterion_predictions,
        predicted_decision_by_document={d: r.final_decision for d, r in document_decisions.items()},
    )
    usage_billable = token_usage_summary_gold_evidence(gold_set=pf.gold_set, arm_result=arm_result)
    usage_per_doc = token_usage_summary_gold_evidence_per_document(gold_set=pf.gold_set, arm_result=arm_result)
    wall = wall_time_summary_gold_evidence(arm_result)  # honestly 0.0 -- see wall_time_unavailable flag below, not silently dropped

    result_artifact = EvaluationResultArtifact(
        arm="gold_evidence", gold_set_hash=pf.gold_set.content_hash(),
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
    freeze_frozen_model(result_artifact, output_dir=OUTPUT_DIR, version_label=RESULT_VERSION_LABEL_V2, artifact_type="EvaluationResultArtifact")

    checkpoint_paths = sorted(CHECKPOINT_DIR.glob("*.json"))
    checkpoint_execution_identity_hashes = {
        load_frozen_model(GoldEvidenceCriterionCheckpoint, p, expected_artifact_type="GoldEvidenceCriterionCheckpoint").execution_identity_hash
        for p in checkpoint_paths
    }
    if len(checkpoint_execution_identity_hashes) != 1:
        raise ValueError(f"checkpoints do not share a single execution_identity_hash: {sorted(checkpoint_execution_identity_hashes)}")

    runner_file_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    evaluation_harness_file_sha256 = hashlib.sha256(Path(r"C:/projects/FT_SCREENING/src/ft_screening/evaluation_harness.py").read_bytes()).hexdigest()
    source_identity = capture_source_identity()

    config_summary = dict(fingerprint_payload)
    config_summary.update({
        "source_identity": source_identity.model_dump(),
        "v1_result_content_hash": v1_result.content_hash(),
        "v1_manifest_content_hash": v1_manifest.content_hash(),
        "checkpoint_execution_identity_hash": next(iter(checkpoint_execution_identity_hashes)),
        "n_checkpoints_reused": len(checkpoint_paths),
        "new_provider_calls": 0,
        "runner_file_sha256": runner_file_sha256,
        "evaluation_harness_file_sha256": evaluation_harness_file_sha256,
        "wall_time_unavailable_after_checkpoint_resume": True,
    })
    manifest = EvaluationRunManifest(
        arm="gold_evidence", gold_set_hash=pf.gold_set.content_hash(), execution_fingerprint=execution_fingerprint,
        result_artifact_hash=result_artifact.content_hash(), config_summary=config_summary, recorded_at=_now_iso(),
    )
    freeze_frozen_model(manifest, output_dir=OUTPUT_DIR, version_label=MANIFEST_VERSION_LABEL_V2, artifact_type="EvaluationRunManifest")

    print(f"EvaluationResultArtifact v2 frozen: content_hash={result_artifact.content_hash()}")
    print(f"EvaluationRunManifest v2 frozen: content_hash={manifest.content_hash()}")
    print(f"new_provider_calls=0, n_checkpoints_reused={len(checkpoint_paths)}, checkpoint_execution_identity_hash={next(iter(checkpoint_execution_identity_hashes))}")
    print(f"runner_file_sha256={runner_file_sha256}")
    print(f"evaluation_harness_file_sha256={evaluation_harness_file_sha256}")
    print("wall_time_unavailable_after_checkpoint_resume=True (checkpoint files never recorded per-call wall time)")

    criterion_matrix = criterion_status_agreement(gold_set=pf.gold_set, predicted_status_by_key=criterion_predictions)
    decision_matrix = final_decision_agreement(gold_set=pf.gold_set, predicted_decision_by_document={d: r.final_decision for d, r in document_decisions.items()})
    false_excl = false_exclusion_rate(gold_set=pf.gold_set, predicted_decision_by_document={d: r.final_decision for d, r in document_decisions.items()})
    false_incl = false_inclusion_rate(gold_set=pf.gold_set, predicted_decision_by_document={d: r.final_decision for d, r in document_decisions.items()})
    reason_rate = correct_exclusion_reason_rate(gold_set=pf.gold_set, document_decisions=document_decisions)
    rr = review_rate(gold_set=pf.gold_set, predicted_decision_by_document={d: r.final_decision for d, r in document_decisions.items()})

    print("\n--- Metrics (via evaluation_harness, not reimplemented) ---")
    print(f"criterion_status_agreement accuracy = {criterion_matrix.accuracy():.4f}")
    print(f"final_decision_agreement accuracy = {decision_matrix.accuracy():.4f}")
    print(f"false_exclusion_rate = {false_excl.rate:.4f}  (documents: {false_excl.document_ids})")
    print(f"false_inclusion_rate = {false_incl.rate:.4f}  (documents: {false_incl.document_ids})")
    print(f"correct_exclusion_reason_rate = {reason_rate:.4f}")
    print(f"review_rate = {rr.review_rate:.4f}  (hard cases: {rr.review_rate_among_gold_hard_cases:.4f}, easy cases: {rr.review_rate_among_gold_easy_cases:.4f})")
    print(f"completed documents = {len(completion.completed_document_ids)}/{len(pf.gold_set.expected_document_ids)}  missing/failed = {completion.missing_or_failed_document_ids}")
    print(f"completed criteria = {len(completion.completed_criterion_keys)}/{n_calls}  missing/failed = {completion.missing_or_failed_criterion_keys}")
    print(f"token usage (billable): input={usage_billable.total_input_tokens} output={usage_billable.total_output_tokens} total={usage_billable.total_tokens}")

    print("\n" + ARM2B_METHOD_CAVEAT)
    print("\n" + LABEL_MASKED_CAVEAT)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Actually run Arm 2b (paid OpenAI calls). Omit for preflight-only.")
    parser.add_argument("--finalize-from-checkpoints", action="store_true", help="Re-finalize offline from existing checkpoints as result_v2/manifest_v2. Zero new API calls.")
    args = parser.parse_args(argv)

    pf = preflight()

    if args.finalize_from_checkpoints:
        finalize_from_checkpoints(pf)
        return

    if not args.execute:
        n_calls = len(pf.gold_set.criterion_annotations)
        print(f"\nDRY RUN complete -- no API calls made. Pass --execute to run Arm 2b for real (up to {n_calls} paid OpenAI adjudication calls), or --finalize-from-checkpoints to re-finalize offline from existing checkpoints.")
        return

    execute(pf)


if __name__ == "__main__":
    main()
