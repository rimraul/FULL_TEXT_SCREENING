"""GoldEvidenceSet: a separately-versioned, read-only-of-GoldSet augmentation
that repairs the evidentiary gap documented in
outputs/prompt6_disagreement_audit_v1.md -- `GoldCriterionAnnotation.
supporting_passages`/`contradicting_passages` overloaded ONE field for
several genuinely different evidentiary roles.

Two rounds of annotator clarification are load-bearing for this design (both
recorded in outputs/prompt6_disagreement_audit_v1.md):

  1. A figure-caption citation can be a LOCATOR for real, non-textual
     evidence (e.g. a search-string image) that simply cannot be copied as
     text -- not a defective or mis-cited annotation.
  2. A bare subsection heading can ITSELF be legitimate STRUCTURAL evidence
     that a topic is an organized discussion focus of the publication (e.g.
     a dedicated section titled "Governance" is direct structural evidence
     that governance is discussed as an explicit topic) -- even though that
     same heading cannot, by itself, support a specific PROPOSITIONAL claim
     about findings, mechanisms, or conclusions. Treating every citation as
     an interchangeable propositional excerpt (as the first version of this
     module did, and as Arm 2's evidence construction still does) is an
     EVIDENCE-SEMANTICS MISMATCH, not necessarily a sign the annotation
     itself is deficient.

Design constraints, all explicit per-instruction:
  - Bound to a specific frozen GoldSet by `gold_set_hash` -- never usable
    against a different or later GoldSet without a fresh binding.
  - RETAINS the original citation verbatim on every entry (never overwrites
    or discards gold_set_v1.json's own supporting/contradicting
    pages/passages) -- this module only ever reads GoldSet, never writes it.
  - Every excerpt carries an explicit `evidence_role` (`verbatim_text`,
    `structural_heading`, `visual_evidence_locator`, `annotator_summary`, or
    `locator_only`) and a required `supported_inference` stating EXACTLY
    what claim that piece of evidence is meant to support -- so "this
    heading proves the topic is an organized discussion focus" and "this
    heading is just a placeholder for missing body text" are never
    ambiguous with each other again.
  - The heading/caption-shape prohibition from the first version of this
    module now applies ONLY to roles that claim to carry propositional
    content (`verbatim_text`, `annotator_summary`) -- `structural_heading`/
    `visual_evidence_locator`/`locator_only` are EXPECTED to be short and
    heading-shaped; that is the correct shape for those roles, not a
    violation needing an override.
  - Carries an explicit, nullable `sufficient_for_independent_adjudication`
    judgment per entry -- None means "not yet assessed", never silently
    defaulted to True. Distinguishes `sufficiency_assessed_by` values so a
    draft (Claude-authored, single-pass) assessment is never confused with a
    human/formal one.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ft_screening.hashing import canonical_json_bytes

# Reused (in spirit) from scripts/screen_gold_evidence_self_containedness.py
# -- see that module for the full disclosure of what this heuristic does and
# does not catch. Duplicated here as plain regex (no import of a scripts/
# module from src/) rather than risking the two silently diverging; if the
# screener's heuristic changes, this must be updated to match. Applied ONLY
# to roles that claim propositional content -- see EvidenceRole below.
_HEADING_NUMBER_RE = re.compile(r"^\d+(\.\d+)*\.?\s+\S")
_CAPTION_PREFIX_RE = re.compile(r"^(fig(ure)?|table)\.?\s*\d", re.IGNORECASE)
_SENTENCE_END_RE = re.compile(r"[.!?]\s*$")
_MIN_SUBSTANTIVE_LENGTH = 60


def _looks_non_substantive(passage: str) -> tuple[bool, str]:
    text = passage.strip()
    if not text:
        return True, "empty"
    if _CAPTION_PREFIX_RE.match(text):
        return True, "figure/table caption prefix"
    if _HEADING_NUMBER_RE.match(text) and len(text) < 120:
        return True, "numbered-heading pattern"
    if len(text) < _MIN_SUBSTANTIVE_LENGTH and not _SENTENCE_END_RE.search(text):
        return True, f"short ({len(text)} chars) with no sentence-ending punctuation"
    return False, ""


ExtractionMethod = Literal["retained_original_citation", "expanded_from_pdf_section", "human_transcribed"]

# - verbatim_text: a direct quotation the criterion can be adjudicated
#   against as a proposition (what the old, single-role schema assumed
#   every excerpt was).
# - structural_heading: a section/subsection title recorded because its
#   EXISTENCE is the evidence -- supports "this topic is an organized
#   discussion focus", never a specific finding/mechanism/conclusion claim.
# - visual_evidence_locator: points at non-textual evidence (a figure,
#   table image, chart) that cannot be copied as text -- the evidence
#   itself lives outside any text-only pipeline's reach.
# - annotator_summary: the annotator's own paraphrase/summary of evidence
#   found in the source, not a verbatim quotation.
# - locator_only: a page/section pointer with no evidentiary content of its
#   own (the historical "just tells you where to look" case) -- distinct
#   from structural_heading, which DOES carry evidentiary weight for a
#   discussion-focus claim; locator_only carries none at all.
EvidenceRole = Literal["verbatim_text", "structural_heading", "visual_evidence_locator", "annotator_summary", "locator_only"]

# Roles whose excerpt text is being offered AS a proposition the criterion
# can be adjudicated against directly -- these are the only roles the
# heading/caption-shape prohibition applies to. A structural_heading is
# EXPECTED to look exactly like a heading; that is correct, not a defect.
_PROPOSITIONAL_ROLES: frozenset[EvidenceRole] = frozenset({"verbatim_text", "annotator_summary"})


class GoldEvidenceExcerpt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stance: Literal["supports", "contradicts"]
    page: int = Field(ge=1)
    evidence_role: EvidenceRole
    # The exact heading, caption, or quotation as found/cited -- always
    # present, always the shortest faithful record of what was actually
    # seen. For evidence_role="structural_heading" this alone IS the
    # evidence; for "verbatim_text" it is a (possibly short) quotation that
    # may be supplemented by body_excerpt below.
    heading_or_quotation: str = Field(min_length=1)
    # What this piece of evidence is specifically claimed to support --
    # required for every excerpt precisely so a structural_heading's
    # narrower claim ("X is an organized discussion focus") is never
    # conflated with a verbatim_text excerpt's broader one ("X is true").
    supported_inference: str = Field(min_length=1)
    # Optional, bounded expansion beyond the heading/quotation -- e.g. the
    # source paragraph a structural_heading sits atop, when available and
    # relevant. None is normal and expected for structural_heading/
    # visual_evidence_locator/locator_only roles.
    body_excerpt: str | None = None
    extraction_method: ExtractionMethod
    # Binds this excerpt to the EXACT PDF bytes it was pulled from, when
    # derived by re-parsing -- None for a retained original citation (the
    # GoldSet's own document_pdf_sha256 already covers that case).
    source_document_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    source_section: str | None = None

    @model_validator(mode="after")
    def _propositional_roles_must_be_substantive(self) -> "GoldEvidenceExcerpt":
        if self.evidence_role in _PROPOSITIONAL_ROLES:
            text_to_check = self.body_excerpt or self.heading_or_quotation
            bad, reason = _looks_non_substantive(text_to_check)
            if bad:
                raise ValueError(
                    f"evidence_role={self.evidence_role!r} claims propositional content, but its text "
                    f"looks heading/caption-like ({reason}) with no body_excerpt to supplement it -- "
                    "use evidence_role='structural_heading' (or another non-propositional role) instead "
                    "if the heading/caption IS the evidence, or supply a substantive body_excerpt"
                )
        return self


class GoldEvidenceEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str = Field(min_length=1)
    criterion_id: str = Field(min_length=1)

    # Retained verbatim from the frozen GoldSet this entry is bound to --
    # never altered, never omitted, regardless of how insufficient it is.
    original_supporting_pages: tuple[int, ...] = ()
    original_supporting_passages: tuple[str, ...] = ()
    original_contradicting_pages: tuple[int, ...] = ()
    original_contradicting_passages: tuple[str, ...] = ()

    self_contained_excerpts: tuple[GoldEvidenceExcerpt, ...] = ()

    # Explicit, nullable judgment -- None means genuinely not yet assessed,
    # never silently treated as either True or False by any consumer of
    # this model.
    sufficient_for_independent_adjudication: bool | None = None
    sufficiency_assessed_by: str | None = None
    sufficiency_rationale: str = ""

    # Human-review provenance, distinct from sufficiency_assessed_by (which
    # already covers non-human passes like "claude_draft_pass") --
    # reviewer_id/reviewed_at/review_notes are the fields
    # verify_ready_for_formal_freeze() requires be populated before a
    # GoldEvidenceSet may be promoted to the formal (non-draft) name.
    reviewer_id: str | None = None
    reviewed_at: str | None = None
    review_notes: str = ""

    # For an entry with an EMPTY self_contained_excerpts (no supporting or
    # contradicting evidence recorded at all): an empty list alone cannot
    # independently demonstrate that no qualifying evidence exists in the
    # source -- it is equally consistent with nobody having looked
    # exhaustively. This attestation is the explicit, human-made claim that
    # a full-document search was performed and genuinely found nothing --
    # required (see verify_ready_for_formal_freeze) before an empty-package
    # entry can count as human-reviewed.
    human_attestation: Literal["no_qualifying_evidence_found_after_full_document_review"] | None = None

    @model_validator(mode="after")
    def _sufficiency_fields_consistent(self) -> "GoldEvidenceEntry":
        if self.sufficient_for_independent_adjudication is not None and self.sufficiency_assessed_by is None:
            raise ValueError(
                f"{self.document_id}::{self.criterion_id}: sufficient_for_independent_adjudication is set "
                "but sufficiency_assessed_by is None -- every judgment must record who/what made it"
            )
        return self

    @model_validator(mode="after")
    def _human_review_fields_consistent(self) -> "GoldEvidenceEntry":
        if (self.reviewed_at is not None or self.review_notes.strip()) and self.reviewer_id is None:
            raise ValueError(
                f"{self.document_id}::{self.criterion_id}: reviewed_at/review_notes is set but reviewer_id is "
                "None -- every human review must record who performed it"
            )
        if self.human_attestation is not None and not self.self_contained_excerpts:
            pass  # the intended case: attesting that an exhaustive search found nothing
        elif self.human_attestation is not None and self.self_contained_excerpts:
            raise ValueError(
                f"{self.document_id}::{self.criterion_id}: human_attestation is set but "
                "self_contained_excerpts is non-empty -- the exhaustive-search-found-nothing attestation "
                "only applies to an entry with NO recorded excerpts"
            )
        return self


class GoldEvidenceSet(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    gold_set_hash: str = Field(min_length=64, max_length=64)
    version_label: str = Field(min_length=1)
    entries: tuple[GoldEvidenceEntry, ...]
    frozen_at: str = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_keys(self) -> "GoldEvidenceSet":
        seen: set[tuple[str, str]] = set()
        for e in self.entries:
            key = (e.document_id, e.criterion_id)
            if key in seen:
                raise ValueError(f"duplicate GoldEvidenceEntry for {key}")
            seen.add(key)
        return self

    def content_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"frozen_at"})
        return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def verify_gold_evidence_set_matches_gold_set(evidence_set: GoldEvidenceSet, gold_set) -> None:
    """Binding + exact-coverage check -- the boundary this module promises:
    a GoldEvidenceSet is only ever meaningful against the ONE GoldSet it was
    built from, and only if it covers exactly that GoldSet's own keys (a
    partial/draft set covering fewer keys is fine to construct and inspect,
    but must not silently pass this check as if it were complete)."""
    if evidence_set.gold_set_hash != gold_set.content_hash():
        raise ValueError(
            f"GoldEvidenceSet.gold_set_hash={evidence_set.gold_set_hash!r} does not match "
            f"gold_set.content_hash()={gold_set.content_hash()!r}"
        )
    expected_keys = {(a.document_id, a.criterion_id) for a in gold_set.criterion_annotations}
    actual_keys = {(e.document_id, e.criterion_id) for e in evidence_set.entries}
    if actual_keys != expected_keys:
        missing = expected_keys - actual_keys
        extra = actual_keys - expected_keys
        raise ValueError(
            f"GoldEvidenceSet does not cover exactly the GoldSet's own keys -- missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )


def verify_ready_for_formal_freeze(evidence_set: GoldEvidenceSet) -> None:
    """The gate a GoldEvidenceSet must pass before it may be promoted from
    a draft name (e.g. "gold_evidence_set_claude_draft_v1") to the formal
    "gold_evidence_set_v1" name. Checks the MECHANICALLY VERIFIABLE subset
    of that promotion's real requirements -- raises ValueError listing
    EVERY violation found (not just the first), so a reviewer sees the
    complete remaining worklist in one pass, never a fix-one-see-the-next
    loop.

    Deliberately does NOT and CANNOT verify genuine semantic/content
    adequacy: whether an expansion actually resolves a case, whether a
    construct question is actually settled, whether a reviewer's judgment
    is CORRECT. Those remain human judgments this function can only
    require be RECORDED (a non-empty reviewer_id and review_notes), never
    validate the substance of."""
    problems: list[str] = []
    for e in evidence_set.entries:
        key = f"{e.document_id}::{e.criterion_id}"
        if e.sufficient_for_independent_adjudication is None:
            problems.append(f"{key}: sufficient_for_independent_adjudication is still None")
        if e.reviewer_id is None:
            problems.append(f"{key}: no reviewer_id recorded -- human review is not attributed")
        elif not e.review_notes.strip():
            problems.append(f"{key}: reviewer_id is recorded but review_notes is empty")
        if not e.self_contained_excerpts and e.human_attestation is None:
            problems.append(
                f"{key}: no excerpts recorded and no human_attestation -- an empty package cannot count as "
                "human-reviewed without an explicit exhaustive-search attestation"
            )
        for x in e.self_contained_excerpts:
            if x.extraction_method in ("expanded_from_pdf_section", "human_transcribed") and x.source_document_sha256 is None:
                problems.append(
                    f"{key}: excerpt (page={x.page}, role={x.evidence_role}) has "
                    f"extraction_method={x.extraction_method!r} but no source_document_sha256 -- not source-bound"
                )
    if problems:
        raise ValueError(
            f"GoldEvidenceSet is not ready for formal freeze ({len(problems)} problem(s)):\n  " + "\n  ".join(problems)
        )


def build_entry_from_gold_annotation(annotation) -> GoldEvidenceEntry:
    """Starting point for every entry: retains the original citation
    verbatim, with no self_contained_excerpts and no sufficiency judgment
    yet -- callers add excerpts/judgments on top of this, never by mutating
    GoldSet itself."""
    return GoldEvidenceEntry(
        document_id=annotation.document_id, criterion_id=annotation.criterion_id,
        original_supporting_pages=tuple(annotation.supporting_pages),
        original_supporting_passages=tuple(annotation.supporting_passages),
        original_contradicting_pages=tuple(annotation.contradicting_pages),
        original_contradicting_passages=tuple(annotation.contradicting_passages),
    )


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
