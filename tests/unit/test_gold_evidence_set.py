"""Offline tests for ft_screening.gold_evidence_set -- the GoldEvidenceSet
schema built to repair the evidentiary gap documented in
outputs/prompt6_disagreement_audit_v1.md, including the evidence-role
taxonomy added after the annotator's clarification that a structural
heading can itself be legitimate evidence for a discussion-focus claim.
No network/API calls.
"""
from __future__ import annotations

import sys

import pytest

sys.path.insert(0, r"C:/projects/FT_SCREENING/src")

from ft_screening.gold_evidence_set import (  # noqa: E402
    GoldEvidenceEntry,
    GoldEvidenceExcerpt,
    GoldEvidenceSet,
    build_entry_from_gold_annotation,
    verify_gold_evidence_set_matches_gold_set,
    verify_ready_for_formal_freeze,
)


class _FakeAnnotation:
    def __init__(self, document_id, criterion_id, supporting_pages=(), supporting_passages=(), contradicting_pages=(), contradicting_passages=()):
        self.document_id = document_id
        self.criterion_id = criterion_id
        self.supporting_pages = supporting_pages
        self.supporting_passages = supporting_passages
        self.contradicting_pages = contradicting_pages
        self.contradicting_passages = contradicting_passages


class _FakeGoldSet:
    def __init__(self, annotations, gold_set_hash="a" * 64):
        self.criterion_annotations = annotations
        self._hash = gold_set_hash

    def content_hash(self):
        return self._hash


def test_build_entry_retains_original_citation_verbatim():
    ann = _FakeAnnotation("d1", "c1", supporting_pages=(3,), supporting_passages=("a real sentence here.",))
    entry = build_entry_from_gold_annotation(ann)
    assert entry.document_id == "d1"
    assert entry.criterion_id == "c1"
    assert entry.original_supporting_pages == (3,)
    assert entry.original_supporting_passages == ("a real sentence here.",)
    assert entry.self_contained_excerpts == ()
    assert entry.sufficient_for_independent_adjudication is None


def test_verbatim_text_excerpt_accepted_when_substantive():
    excerpt = GoldEvidenceExcerpt(
        stance="supports", page=3, evidence_role="verbatim_text",
        heading_or_quotation="This is a genuinely substantive excerpt with real sentence content describing the finding.",
        supported_inference="Directly establishes the criterion's claim.",
        extraction_method="expanded_from_pdf_section",
    )
    assert excerpt.evidence_role == "verbatim_text"


@pytest.mark.parametrize("bad_text", [
    "4 Coordination Mechanisms",
    "Fig. 1. Research protocol adopted for the literature review.",
    "3.4 DRR Strategies",
])
def test_verbatim_text_excerpt_rejected_when_heading_shaped_and_no_body_excerpt(bad_text):
    with pytest.raises(ValueError, match="looks heading/caption-like"):
        GoldEvidenceExcerpt(
            stance="supports", page=3, evidence_role="verbatim_text", heading_or_quotation=bad_text,
            supported_inference="Some claim.", extraction_method="retained_original_citation",
        )


def test_verbatim_text_excerpt_accepted_with_substantive_body_excerpt_even_if_heading_short():
    excerpt = GoldEvidenceExcerpt(
        stance="supports", page=3, evidence_role="verbatim_text", heading_or_quotation="5.1.3 Governance",
        body_excerpt="Trust between stakeholders plays a key role in collaborative decision-making, as shown by multiple case studies discussed in this section.",
        supported_inference="Establishes the substantive governance discussion.",
        extraction_method="expanded_from_pdf_section",
    )
    assert excerpt.body_excerpt is not None


def test_structural_heading_role_accepted_without_any_override_even_though_heading_shaped():
    """The core fix from the annotator's second clarification: a heading is
    the CORRECT shape for evidence_role='structural_heading', not a
    violation needing justification."""
    excerpt = GoldEvidenceExcerpt(
        stance="supports", page=13, evidence_role="structural_heading", heading_or_quotation="5.1.3 Governance",
        supported_inference="Supports that governance is an organized, dedicated discussion focus of the publication -- "
                             "does NOT by itself support any specific substantive governance claim, finding, or mechanism.",
        extraction_method="retained_original_citation",
    )
    assert excerpt.evidence_role == "structural_heading"


def test_visual_evidence_locator_role_accepted_without_override():
    excerpt = GoldEvidenceExcerpt(
        stance="supports", page=1, evidence_role="visual_evidence_locator", heading_or_quotation="Fig. 1. Research protocol adopted for the literature review.",
        supported_inference="Locates a search-string image showing hazard/disaster search terms -- not copyable as text.",
        extraction_method="retained_original_citation",
    )
    assert excerpt.evidence_role == "visual_evidence_locator"


def test_locator_only_role_accepted_without_override():
    excerpt = GoldEvidenceExcerpt(
        stance="supports", page=4, evidence_role="locator_only", heading_or_quotation="4 Coordination Mechanisms",
        supported_inference="Points to where the annotator found the answer; carries no evidentiary content of its own.",
        extraction_method="retained_original_citation",
    )
    assert excerpt.evidence_role == "locator_only"


def test_supported_inference_is_required():
    with pytest.raises(ValueError):
        GoldEvidenceExcerpt(
            stance="supports", page=1, evidence_role="structural_heading", heading_or_quotation="5.1.3 Governance",
            supported_inference="", extraction_method="retained_original_citation",
        )


def test_sufficiency_judgment_requires_assessed_by():
    with pytest.raises(ValueError, match="sufficiency_assessed_by"):
        GoldEvidenceEntry(document_id="d1", criterion_id="c1", sufficient_for_independent_adjudication=True)


def test_sufficiency_judgment_with_assessed_by_is_accepted():
    entry = GoldEvidenceEntry(
        document_id="d1", criterion_id="c1", sufficient_for_independent_adjudication=True,
        sufficiency_assessed_by="claude_draft_pass", sufficiency_rationale="Excerpt directly states the finding.",
    )
    assert entry.sufficient_for_independent_adjudication is True


def test_gold_evidence_set_rejects_duplicate_keys():
    entries = (
        GoldEvidenceEntry(document_id="d1", criterion_id="c1"),
        GoldEvidenceEntry(document_id="d1", criterion_id="c1"),
    )
    with pytest.raises(ValueError, match="duplicate"):
        GoldEvidenceSet(gold_set_hash="a" * 64, version_label="v1", entries=entries, frozen_at="2026-01-01T00:00:00+00:00")


def test_content_hash_changes_when_an_entry_changes():
    base_entries = (GoldEvidenceEntry(document_id="d1", criterion_id="c1"),)
    es1 = GoldEvidenceSet(gold_set_hash="a" * 64, version_label="v1", entries=base_entries, frozen_at="2026-01-01T00:00:00+00:00")
    es2 = GoldEvidenceSet(
        gold_set_hash="a" * 64, version_label="v1",
        entries=(GoldEvidenceEntry(document_id="d1", criterion_id="c1", sufficiency_rationale="different"),),
        frozen_at="2026-01-01T00:00:00+00:00",
    )
    assert es1.content_hash() != es2.content_hash()


def test_content_hash_stable_across_frozen_at_changes():
    entries = (GoldEvidenceEntry(document_id="d1", criterion_id="c1"),)
    es1 = GoldEvidenceSet(gold_set_hash="a" * 64, version_label="v1", entries=entries, frozen_at="2026-01-01T00:00:00+00:00")
    es2 = GoldEvidenceSet(gold_set_hash="a" * 64, version_label="v1", entries=entries, frozen_at="2027-06-06T00:00:00+00:00")
    assert es1.content_hash() == es2.content_hash()


def test_verify_matches_gold_set_passes_on_exact_binding_and_coverage():
    gold_set = _FakeGoldSet([_FakeAnnotation("d1", "c1"), _FakeAnnotation("d1", "c2")])
    evidence_set = GoldEvidenceSet(
        gold_set_hash=gold_set.content_hash(), version_label="v1",
        entries=(GoldEvidenceEntry(document_id="d1", criterion_id="c1"), GoldEvidenceEntry(document_id="d1", criterion_id="c2")),
        frozen_at="2026-01-01T00:00:00+00:00",
    )
    verify_gold_evidence_set_matches_gold_set(evidence_set, gold_set)  # must not raise


def test_verify_matches_gold_set_rejects_hash_mismatch():
    gold_set = _FakeGoldSet([_FakeAnnotation("d1", "c1")], gold_set_hash="a" * 64)
    evidence_set = GoldEvidenceSet(
        gold_set_hash="b" * 64, version_label="v1",
        entries=(GoldEvidenceEntry(document_id="d1", criterion_id="c1"),), frozen_at="2026-01-01T00:00:00+00:00",
    )
    with pytest.raises(ValueError, match="gold_set_hash"):
        verify_gold_evidence_set_matches_gold_set(evidence_set, gold_set)


def test_verify_matches_gold_set_rejects_incomplete_coverage():
    gold_set = _FakeGoldSet([_FakeAnnotation("d1", "c1"), _FakeAnnotation("d1", "c2")])
    evidence_set = GoldEvidenceSet(
        gold_set_hash=gold_set.content_hash(), version_label="v1",
        entries=(GoldEvidenceEntry(document_id="d1", criterion_id="c1"),),  # missing c2
        frozen_at="2026-01-01T00:00:00+00:00",
    )
    with pytest.raises(ValueError, match="missing"):
        verify_gold_evidence_set_matches_gold_set(evidence_set, gold_set)


# ---------------------------------------------------------------------------
# Human-review provenance + the formal-freeze readiness gate
# ---------------------------------------------------------------------------


def _reviewed_entry(**overrides) -> GoldEvidenceEntry:
    defaults = dict(
        document_id="d1", criterion_id="c1", sufficient_for_independent_adjudication=True,
        sufficiency_assessed_by="human_review",
        self_contained_excerpts=(
            GoldEvidenceExcerpt(
                stance="supports", page=1, evidence_role="verbatim_text", heading_or_quotation="A real substantive quotation with sentence content.",
                supported_inference="Directly establishes the criterion.", extraction_method="retained_original_citation",
            ),
        ),
        reviewer_id="RR", reviewed_at="2026-01-01T00:00:00+00:00", review_notes="Confirmed against criterion wording.",
    )
    defaults.update(overrides)
    return GoldEvidenceEntry(**defaults)


def test_review_fields_require_reviewer_id():
    with pytest.raises(ValueError, match="reviewer_id"):
        GoldEvidenceEntry(document_id="d1", criterion_id="c1", review_notes="some notes with no reviewer set")


def test_review_fields_accepted_together():
    entry = _reviewed_entry()
    assert entry.reviewer_id == "RR"


def test_human_attestation_accepted_for_empty_package():
    entry = GoldEvidenceEntry(
        document_id="d1", criterion_id="c1", sufficient_for_independent_adjudication=True,
        sufficiency_assessed_by="human_review", reviewer_id="RR", review_notes="Searched the full text; nothing qualifies.",
        human_attestation="no_qualifying_evidence_found_after_full_document_review",
    )
    assert entry.human_attestation == "no_qualifying_evidence_found_after_full_document_review"


def test_human_attestation_rejected_when_excerpts_are_present():
    with pytest.raises(ValueError, match="human_attestation"):
        _reviewed_entry(human_attestation="no_qualifying_evidence_found_after_full_document_review")


def test_ready_for_formal_freeze_passes_when_fully_reviewed_and_source_bound():
    entries = (
        _reviewed_entry(document_id="d1", criterion_id="c1"),
        GoldEvidenceEntry(
            document_id="d1", criterion_id="c2", sufficient_for_independent_adjudication=True,
            sufficiency_assessed_by="human_review", reviewer_id="RR", review_notes="Full-text search performed, nothing found.",
            human_attestation="no_qualifying_evidence_found_after_full_document_review",
        ),
    )
    gold_set = _FakeGoldSet([_FakeAnnotation("d1", "c1"), _FakeAnnotation("d1", "c2")])
    evidence_set = GoldEvidenceSet(gold_set_hash=gold_set.content_hash(), version_label="v1", entries=entries, frozen_at="2026-01-01T00:00:00+00:00")
    verify_ready_for_formal_freeze(evidence_set)  # must not raise


def test_ready_for_formal_freeze_rejects_remaining_none_judgment():
    entries = (GoldEvidenceEntry(document_id="d1", criterion_id="c1"),)
    evidence_set = GoldEvidenceSet(gold_set_hash="a" * 64, version_label="v1", entries=entries, frozen_at="2026-01-01T00:00:00+00:00")
    with pytest.raises(ValueError, match="still None"):
        verify_ready_for_formal_freeze(evidence_set)


def test_ready_for_formal_freeze_rejects_missing_reviewer():
    entries = (GoldEvidenceEntry(
        document_id="d1", criterion_id="c1", sufficient_for_independent_adjudication=True, sufficiency_assessed_by="claude_draft_pass",
        self_contained_excerpts=(GoldEvidenceExcerpt(stance="supports", page=1, evidence_role="verbatim_text", heading_or_quotation="Real substantive text with a sentence.", supported_inference="X", extraction_method="retained_original_citation"),),
    ),)
    evidence_set = GoldEvidenceSet(gold_set_hash="a" * 64, version_label="v1", entries=entries, frozen_at="2026-01-01T00:00:00+00:00")
    with pytest.raises(ValueError, match="no reviewer_id"):
        verify_ready_for_formal_freeze(evidence_set)


def test_ready_for_formal_freeze_rejects_empty_package_without_attestation():
    entries = (_reviewed_entry(self_contained_excerpts=()),)
    evidence_set = GoldEvidenceSet(gold_set_hash="a" * 64, version_label="v1", entries=entries, frozen_at="2026-01-01T00:00:00+00:00")
    with pytest.raises(ValueError, match="human_attestation"):
        verify_ready_for_formal_freeze(evidence_set)


def test_ready_for_formal_freeze_rejects_unbound_expanded_excerpt():
    entry = _reviewed_entry(self_contained_excerpts=(
        GoldEvidenceExcerpt(
            stance="supports", page=1, evidence_role="verbatim_text", heading_or_quotation="Real substantive text with a sentence.",
            supported_inference="X", extraction_method="expanded_from_pdf_section", source_document_sha256=None,
        ),
    ))
    evidence_set = GoldEvidenceSet(gold_set_hash="a" * 64, version_label="v1", entries=(entry,), frozen_at="2026-01-01T00:00:00+00:00")
    with pytest.raises(ValueError, match="not source-bound"):
        verify_ready_for_formal_freeze(evidence_set)


def test_ready_for_formal_freeze_reports_every_problem_not_just_the_first():
    entries = (GoldEvidenceEntry(document_id="d1", criterion_id="c1"), GoldEvidenceEntry(document_id="d2", criterion_id="c2"))
    evidence_set = GoldEvidenceSet(gold_set_hash="a" * 64, version_label="v1", entries=entries, frozen_at="2026-01-01T00:00:00+00:00")
    with pytest.raises(ValueError) as exc_info:
        verify_ready_for_formal_freeze(evidence_set)
    message = str(exc_info.value)
    assert message.count("still None") == 2
