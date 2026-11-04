"""Offline tests for scripts/promote_gold_evidence_set_from_worksheet.py --
the real worksheet-consumer/promotion script (unlike the old
finalize_gold_evidence_set_v1.py, which rebuilds a draft from hardcoded
data and never reads the worksheet at all).

Two fixture tiers, by design:
  - Low-level tests call `_build_entry`/`_parse_expanded_arrays` directly
    against a hand-built `GoldCriterionAnnotation` and a lightweight
    duck-typed gold-set stand-in (only `.document_pdf_sha256` is ever
    read by `_build_entry`) -- no real GoldSet/PDF/seal pipeline needed,
    since none of that machinery is exercised by these code paths.
  - Higher-level tests (`build_evidence_set_from_worksheet`) need a REAL,
    fully-validated `GoldSet` (its own model validators, including the
    aggregation cross-check, are strict) -- built via the actual
    seal -> merge -> freeze pipeline with synthetic PDFs, same pattern as
    tests/unit/test_run_arm2_gold_evidence_preflight.py's
    `_build_frozen_gold_set`.

Zero network/API calls anywhere in this file.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import fitz
import pytest

sys.path.insert(0, r"C:/projects/FT_SCREENING/src")
sys.path.insert(0, r"C:/projects/FT_SCREENING/scripts")

from ft_screening.gold import GoldCriterionAnnotation, freeze_gold_set  # noqa: E402
from ft_screening.schema import CriteriaSet, CriterionSpec  # noqa: E402

SCRIPTS_DIR = Path(r"C:/projects/FT_SCREENING/scripts")


def _load_promote_module():
    spec = importlib.util.spec_from_file_location("promote_gold_evidence_set_from_worksheet", SCRIPTS_DIR / "promote_gold_evidence_set_from_worksheet.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PROMOTE = _load_promote_module()

# ---------------------------------------------------------------------------
# Low-level tests: _parse_expanded_arrays, _build_entry -- no real GoldSet needed.
# ---------------------------------------------------------------------------

_KEY = ("d1", "c1")


def _fake_gold_set(document_pdf_sha256: dict[str, str]):
    return SimpleNamespace(document_pdf_sha256=document_pdf_sha256)


def _ann(*, supporting_pages=(1,), supporting_passages=("A real substantive supporting passage, long enough.",),
          contradicting_pages=(), contradicting_passages=(), evidence_absent=False, gold_status="satisfied") -> GoldCriterionAnnotation:
    return GoldCriterionAnnotation(
        document_id="d1", criterion_id="c1", document_page_count=10, gold_status=gold_status,
        supporting_pages=tuple(supporting_pages), supporting_passages=tuple(supporting_passages),
        contradicting_pages=tuple(contradicting_pages), contradicting_passages=tuple(contradicting_passages),
        evidence_absent=evidence_absent, evidence_ambiguous=False, evidence_conflicting=False,
        human_review_required=False, annotator_id="RR", annotation_round="agreed",
    )


def _row(**overrides) -> dict[str, str]:
    base = {
        "document_id": "d1", "criterion_id": "c1",
        "original_supporting_passages": "A real substantive supporting passage, long enough.",
        "original_contradicting_passages": "",
        "reviewer_id": "RR", "reviewed_at": "20/08/2026", "human_sufficiency": "TRUE",
        "human_attestation": "", "expanded_excerpt_text": "", "expanded_excerpt_page": "",
        "review_notes": "Sufficient: baseline test row.",
    }
    base.update(overrides)
    return base


def test_parse_expanded_arrays_malformed_json_raises():
    with pytest.raises(ValueError, match="not valid JSON"):
        PROMOTE._parse_expanded_arrays("not json", "[1]", key=_KEY)


def test_parse_expanded_arrays_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="lengths must match"):
        PROMOTE._parse_expanded_arrays('["a", "b"]', "[1]", key=_KEY)


def test_parse_expanded_arrays_both_empty_is_fine():
    assert PROMOTE._parse_expanded_arrays("", "", key=_KEY) == []


def test_parse_expanded_arrays_one_sided_raises():
    with pytest.raises(ValueError, match="both-empty or both-set"):
        PROMOTE._parse_expanded_arrays('["a"]', "", key=_KEY)


def test_build_entry_missing_review_notes_raises():
    row = _row(review_notes="")
    with pytest.raises(ValueError, match="review_notes is empty"):
        PROMOTE._build_entry(row, _ann(), gold_set=_fake_gold_set({}))


def test_build_entry_empty_package_without_attestation_raises():
    ann = _ann(supporting_pages=(), supporting_passages=(), evidence_absent=True, gold_status="not_reported")
    row = _row(original_supporting_passages="", human_attestation="")
    with pytest.raises(ValueError, match="empty package must be explicitly attested"):
        PROMOTE._build_entry(row, ann, gold_set=_fake_gold_set({}))


def test_build_entry_empty_package_with_attestation_succeeds():
    ann = _ann(supporting_pages=(), supporting_passages=(), evidence_absent=True, gold_status="not_reported")
    row = _row(original_supporting_passages="", human_attestation="no_qualifying_evidence_found_after_full_document_review")
    entry = PROMOTE._build_entry(row, ann, gold_set=_fake_gold_set({}))
    assert entry.self_contained_excerpts == ()
    assert entry.human_attestation == "no_qualifying_evidence_found_after_full_document_review"


def test_build_entry_source_binding_error_unknown_document():
    row = _row(expanded_excerpt_text='["An expanded body-text excerpt, long enough to be substantive."]', expanded_excerpt_page="[3]")
    with pytest.raises(ValueError, match="cannot source-bind"):
        PROMOTE._build_entry(row, _ann(), gold_set=_fake_gold_set({}))  # d1 not in the dict


def test_build_entry_source_binding_success_sets_sha256():
    sha = "a" * 64
    row = _row(expanded_excerpt_text='["An expanded body-text excerpt, long enough to be substantive."]', expanded_excerpt_page="[3]")
    entry = PROMOTE._build_entry(row, _ann(), gold_set=_fake_gold_set({"d1": sha}))
    expanded = [x for x in entry.self_contained_excerpts if x.extraction_method == "expanded_from_pdf_section"]
    assert len(expanded) == 1
    assert expanded[0].source_document_sha256 == sha
    assert expanded[0].page == 3


def test_build_entry_contradiction_only_with_expansion_raises():
    ann = _ann(
        supporting_pages=(), supporting_passages=(),
        contradicting_pages=(2,), contradicting_passages=("A real substantive contradicting passage.",),
        gold_status="violated",
    )
    row = _row(
        original_supporting_passages="", original_contradicting_passages="A real substantive contradicting passage.",
        expanded_excerpt_text='["An expanded body-text excerpt, long enough to be substantive."]', expanded_excerpt_page="[3]",
    )
    with pytest.raises(ValueError, match="RULE 1"):
        PROMOTE._build_entry(row, ann, gold_set=_fake_gold_set({"d1": "a" * 64}))


def test_build_entry_mixed_support_and_contradiction_with_expansion_raises():
    """RULE 1 tightened: a row with BOTH supporting and contradicting
    original passages (mixed/unclear) must also reject expanded evidence,
    not just contradiction-only rows."""
    ann = GoldCriterionAnnotation(
        document_id="d1", criterion_id="c1", document_page_count=10, gold_status="unclear",
        supporting_pages=(1,), supporting_passages=("A real substantive supporting passage, long enough.",),
        contradicting_pages=(2,), contradicting_passages=("A real substantive contradicting passage.",),
        evidence_absent=False, evidence_ambiguous=False, evidence_conflicting=True, human_review_required=True,
        annotator_id="RR", annotation_round="agreed",
    )
    row = _row(
        original_contradicting_passages="A real substantive contradicting passage.",
        expanded_excerpt_text='["An expanded body-text excerpt, long enough to be substantive."]', expanded_excerpt_page="[3]",
    )
    with pytest.raises(ValueError, match="RULE 1"):
        PROMOTE._build_entry(row, ann, gold_set=_fake_gold_set({"d1": "a" * 64}))


def test_build_entry_reviewed_at_blank_raises():
    row = _row(reviewed_at="")
    with pytest.raises(ValueError, match="reviewed_at is blank"):
        PROMOTE._build_entry(row, _ann(), gold_set=_fake_gold_set({}))


def test_build_entry_expanded_page_exceeds_document_page_count_raises():
    row = _row(expanded_excerpt_text='["An expanded body-text excerpt, long enough to be substantive."]', expanded_excerpt_page="[999]")
    with pytest.raises(ValueError, match="exceeds ann.document_page_count"):
        PROMOTE._build_entry(row, _ann(), gold_set=_fake_gold_set({"d1": "a" * 64}))  # _ann() default document_page_count=10


def test_retained_citation_role_visual_locator_not_collapsed_to_structural_heading():
    """The exact regression flagged: a figure/table caption citation must
    stay visual_evidence_locator, distinct from an ordinary numbered
    section heading (structural_heading)."""
    assert PROMOTE._retained_citation_evidence_role("Fig. 1. Research protocol adopted for the literature review.") == "visual_evidence_locator"
    assert PROMOTE._retained_citation_evidence_role("Figure 2. Methodology for the review following the PRISMA concept.") == "visual_evidence_locator"
    assert PROMOTE._retained_citation_evidence_role("Table 4. Principles and elements of adaptive governance.") == "visual_evidence_locator"
    assert PROMOTE._retained_citation_evidence_role("4 Coordination Mechanisms") == "structural_heading"
    assert PROMOTE._retained_citation_evidence_role("A real substantive supporting passage, long enough.") == "verbatim_text"


def test_build_entry_real_regression_keys_preserve_visual_evidence_locator():
    """Reproduces the two exact keys flagged: their retained-original
    excerpt must be visual_evidence_locator, not structural_heading."""
    cases = [
        ("10_1007_978-3-031-54118-6_8", "I2_drr_hazard_framing", "Fig. 1. Research protocol adopted for the literature review."),
        ("10_3390_w16111571", "I1_review_type", "Figure 2. Methodology for the review following the PRISMA concept."),
    ]
    for doc_id, crit_id, passage in cases:
        ann = GoldCriterionAnnotation(
            document_id=doc_id, criterion_id=crit_id, document_page_count=20, gold_status="satisfied",
            supporting_pages=(4,), supporting_passages=(passage,), contradicting_pages=(), contradicting_passages=(),
            evidence_absent=False, evidence_ambiguous=False, evidence_conflicting=False, human_review_required=False,
            annotator_id="RR", annotation_round="agreed",
        )
        row = {
            "document_id": doc_id, "criterion_id": crit_id, "original_supporting_passages": passage,
            "original_contradicting_passages": "", "reviewer_id": "RR", "reviewed_at": "20/08/2026",
            "human_sufficiency": "TRUE", "human_attestation": "", "expanded_excerpt_text": "", "expanded_excerpt_page": "",
            "review_notes": "Sufficient: regression test.",
        }
        entry = PROMOTE._build_entry(row, ann, gold_set=_fake_gold_set({}))
        assert entry.self_contained_excerpts[0].evidence_role == "visual_evidence_locator", (doc_id, crit_id)


def test_build_entry_worksheet_citation_drift_raises():
    row = _row(original_supporting_passages="this does not match the real annotation")
    with pytest.raises(ValueError, match="does not match the frozen GoldSet"):
        PROMOTE._build_entry(row, _ann(), gold_set=_fake_gold_set({}))


def test_build_entry_evidence_role_heading_vs_verbatim():
    heading_ann = _ann(supporting_passages=("4 Coordination Mechanisms",))
    row = _row(original_supporting_passages="4 Coordination Mechanisms")
    entry = PROMOTE._build_entry(row, heading_ann, gold_set=_fake_gold_set({}))
    assert entry.self_contained_excerpts[0].evidence_role == "structural_heading"

    verbatim_ann = _ann()  # long substantive passage from the default helper
    row2 = _row()
    entry2 = PROMOTE._build_entry(row2, verbatim_ann, gold_set=_fake_gold_set({}))
    assert entry2.self_contained_excerpts[0].evidence_role == "verbatim_text"


def test_build_entry_figure_transcription_key_gets_human_transcribed():
    key = next(iter(PROMOTE._FIGURE_TRANSCRIPTION_KEYS))
    ann = GoldCriterionAnnotation(
        document_id=key[0], criterion_id=key[1], document_page_count=10, gold_status="satisfied",
        supporting_pages=(4,), supporting_passages=("A real substantive supporting passage, long enough.",),
        contradicting_pages=(), contradicting_passages=(), evidence_absent=False, evidence_ambiguous=False,
        evidence_conflicting=False, human_review_required=False, annotator_id="RR", annotation_round="agreed",
    )
    row = {
        "document_id": key[0], "criterion_id": key[1],
        "original_supporting_passages": "A real substantive supporting passage, long enough.",
        "original_contradicting_passages": "", "reviewer_id": "RR", "reviewed_at": "20/08/2026",
        "human_sufficiency": "TRUE", "human_attestation": "",
        "expanded_excerpt_text": '["A transcribed figure excerpt, long enough to be substantive."]',
        "expanded_excerpt_page": "[4]", "review_notes": "Sufficient: figure transcription test.",
    }
    entry = PROMOTE._build_entry(row, ann, gold_set=_fake_gold_set({key[0]: "b" * 64}))
    expanded = [x for x in entry.self_contained_excerpts if x.page == 4 and x.heading_or_quotation.startswith("A transcribed")]
    assert len(expanded) == 1
    assert expanded[0].extraction_method == "human_transcribed"

    # A DIFFERENT key with the same shape of expansion must NOT get human_transcribed.
    other_row = _row(expanded_excerpt_text='["An ordinary expanded body-text excerpt, long enough."]', expanded_excerpt_page="[3]")
    other_entry = PROMOTE._build_entry(other_row, _ann(), gold_set=_fake_gold_set({"d1": "c" * 64}))
    other_expanded = [x for x in other_entry.self_contained_excerpts if x.extraction_method != "retained_original_citation"]
    assert len(other_expanded) == 1
    assert other_expanded[0].extraction_method == "expanded_from_pdf_section"


def test_build_entry_invalid_human_sufficiency_raises():
    row = _row(human_sufficiency="yes")
    with pytest.raises(ValueError, match="must be exactly TRUE or FALSE"):
        PROMOTE._build_entry(row, _ann(), gold_set=_fake_gold_set({}))


def test_build_entry_invalid_attestation_value_raises():
    row = _row(human_attestation="something else")
    with pytest.raises(ValueError, match="unrecognized value"):
        PROMOTE._build_entry(row, _ann(), gold_set=_fake_gold_set({}))


# ---------------------------------------------------------------------------
# Higher-level tests: build_evidence_set_from_worksheet against a REAL GoldSet.
# ---------------------------------------------------------------------------

CRITERION_FIELDS = [
    "document_id", "document_page_count", "criterion_id", "criterion_text",
    "gold_status", "supporting_pages", "supporting_passages", "contradicting_pages",
    "contradicting_passages", "evidence_absent", "evidence_ambiguous", "evidence_conflicting",
    "human_review_required", "annotator_id", "annotation_round", "notes",
]
DOCUMENT_FIELDS = [
    "document_id", "document_page_count", "gold_final_decision", "gold_primary_exclusion_reason_code",
    "aggregation_override_rationale", "annotator_id", "annotation_round", "notes",
]


def _criteria_set() -> CriteriaSet:
    return CriteriaSet(criteria_set_id="promote-test", version="1", criteria=[
        CriterionSpec(criterion_id="pop", version="1", eligibility_requirement="x", missing_evidence_policy="review", exclusion_reason_code="wp", exclusion_reason_label="wp", priority=1),
        CriterionSpec(criterion_id="design", version="1", eligibility_requirement="y", missing_evidence_policy="pass", exclusion_reason_code="wd", exclusion_reason_label="wd", priority=2),
    ])


def _write_csv(path: Path, fieldnames: list, rows: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _filled_criterion_row(d, c):
    return {k: "" for k in CRITERION_FIELDS} | {
        "document_id": d, "document_page_count": "5", "criterion_id": c, "criterion_text": f"criterion {c}",
        "gold_status": "satisfied", "supporting_pages": "[1]", "supporting_passages": '["A real substantive supporting passage, long enough."]',
        "contradicting_pages": "[]", "contradicting_passages": "[]",
        "evidence_absent": "FALSE", "evidence_ambiguous": "FALSE", "evidence_conflicting": "FALSE",
        "human_review_required": "FALSE", "annotator_id": "RR", "annotation_round": "primary", "notes": "",
    }


def _load_module(name: str, **path_overrides):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for attr, value in path_overrides.items():
        setattr(mod, attr, value)
    return mod


def _build_frozen_gold_set(tmp_path: Path):
    """2 documents x 2 criteria, both satisfied, no dual review -- same
    shape/helper pattern as test_run_arm2_gold_evidence_preflight.py."""
    document_ids = ["d1", "d2"]
    sealed_dir = tmp_path / "_sealed"
    sealed_dir.mkdir()
    seal_dir = tmp_path / "_primary_seal"
    seal_dir.mkdir()

    pdf_paths, pdf_hashes = {}, {}
    for doc_id in document_ids:
        doc = fitz.open()
        for i in range(5):
            doc.new_page().insert_text((72, 72), f"page {i + 1}")
        pdf_path = (tmp_path / f"{doc_id}.pdf").resolve()
        doc.save(str(pdf_path))
        doc.close()
        pdf_paths[doc_id] = str(pdf_path)
        pdf_hashes[doc_id] = hashlib.sha256(pdf_path.read_bytes()).hexdigest()

    blind_manifest_path = tmp_path / "blind_manifest_v1.json"
    blind_manifest_path.write_text(json.dumps({
        "protocol_version": "gold_set_protocol_v9",
        "documents": [{"document_id": d, "pdf_path": pdf_paths[d], "document_page_count": 5} for d in document_ids],
    }), encoding="utf-8")
    (sealed_dir / "sample_v1.json").write_text(json.dumps({
        "protocol_version": "gold_set_protocol_v9", "dual_review_subset": [],
        "documents": [
            {"document_id": d, "stratum": "INCLUDE", "pdf_path": pdf_paths[d], "pdf_sha256": pdf_hashes[d], "page_count": 5}
            for d in document_ids
        ],
    }), encoding="utf-8")

    primary_criterion_rows = [_filled_criterion_row(d, c) for d in document_ids for c in ("pop", "design")]
    primary_document_rows = [
        {k: "" for k in DOCUMENT_FIELDS} | {
            "document_id": d, "document_page_count": "5", "gold_final_decision": "include",
            "annotator_id": "RR", "annotation_round": "primary",
        }
        for d in document_ids
    ]
    _write_csv(tmp_path / "worksheet_phase1_criteria.csv", CRITERION_FIELDS, primary_criterion_rows)
    _write_csv(tmp_path / "worksheet_phase1_documents.csv", DOCUMENT_FIELDS, primary_document_rows)

    seal_mod = _load_module(
        "gold_set_seal_worksheet_pass", GOLD_SET_DIR=tmp_path, SEALED_DIR=sealed_dir, SEAL_DIR=seal_dir,
        CRITERIA_SET=_criteria_set(),
    )
    sys.argv = ["gold_set_seal_worksheet_pass.py", "primary"]
    seal_mod.main()

    merge_mod = _load_module(
        "gold_set_merge_annotation_sources", GOLD_SET_DIR=tmp_path, SEALED_DIR=sealed_dir, SEAL_DIR=seal_dir,
        MERGED_CRITERION_PATH=tmp_path / "worksheet_merged_criteria.csv",
        MERGED_DOCUMENT_PATH=tmp_path / "worksheet_merged_documents.csv",
        PROVENANCE_PATH=seal_dir / "source_provenance_v1.json", CRITERIA_SET=_criteria_set(),
    )
    merge_mod.main()

    frozen_dir = tmp_path / "frozen"
    gold_set = freeze_gold_set(
        protocol_version="gold_set_protocol_v9", criteria_set=_criteria_set(),
        criterion_csv_path=tmp_path / "worksheet_merged_criteria.csv", document_csv_path=tmp_path / "worksheet_merged_documents.csv",
        blind_manifest_path=blind_manifest_path, sample_manifest_path=sealed_dir / "sample_v1.json",
        output_dir=frozen_dir, version_label="gold_set_v1", source_provenance_path=seal_dir / "source_provenance_v1.json",
    )
    return gold_set


def _baseline_worksheet_rows(gold_set) -> list[dict[str, str]]:
    rows = []
    for ann in gold_set.criterion_annotations:
        rows.append({
            "document_id": ann.document_id, "criterion_id": ann.criterion_id,
            "original_supporting_passages": " ||| ".join(ann.supporting_passages),
            "original_contradicting_passages": " ||| ".join(ann.contradicting_passages),
            "reviewer_id": "RR", "reviewed_at": "20/08/2026", "human_sufficiency": "TRUE",
            "human_attestation": "", "expanded_excerpt_text": "", "expanded_excerpt_page": "",
            "review_notes": f"Sufficient: baseline test row for {ann.document_id}/{ann.criterion_id}.",
        })
    return rows


@pytest.fixture
def gold_set(tmp_path):
    return _build_frozen_gold_set(tmp_path)


def test_build_evidence_set_from_worksheet_happy_path(gold_set):
    rows = _baseline_worksheet_rows(gold_set)
    evidence_set = PROMOTE.build_evidence_set_from_worksheet(gold_set, rows)
    assert len(evidence_set.entries) == 4
    PROMOTE.verify_ready_for_formal_freeze(evidence_set)  # must not raise
    PROMOTE.verify_gold_evidence_set_matches_gold_set(evidence_set, gold_set)  # must not raise


def test_build_evidence_set_from_worksheet_omission_raises(gold_set):
    rows = _baseline_worksheet_rows(gold_set)[:-1]  # drop one row
    with pytest.raises(ValueError, match="does not cover exactly"):
        PROMOTE.build_evidence_set_from_worksheet(gold_set, rows)


def test_build_evidence_set_from_worksheet_duplicate_key_raises(gold_set):
    rows = _baseline_worksheet_rows(gold_set)
    rows.append(dict(rows[0]))  # duplicate the first row
    with pytest.raises(ValueError, match="duplicate row"):
        PROMOTE.build_evidence_set_from_worksheet(gold_set, rows)


# ---------------------------------------------------------------------------
# Output collision.
# ---------------------------------------------------------------------------


def _minimal_evidence_set(gold_set):
    rows = _baseline_worksheet_rows(gold_set)
    return PROMOTE.build_evidence_set_from_worksheet(gold_set, rows)


def test_freeze_evidence_set_refuses_collision(gold_set, tmp_path):
    evidence_set = _minimal_evidence_set(gold_set)
    output_dir = tmp_path / "out"
    first_path = PROMOTE.freeze_evidence_set(evidence_set, output_dir=output_dir, version_label="gold_evidence_set_v1")
    assert first_path.exists()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        PROMOTE.freeze_evidence_set(evidence_set, output_dir=output_dir, version_label="gold_evidence_set_v1")


# ---------------------------------------------------------------------------
# Ordering stability + companion provenance manifest.
# ---------------------------------------------------------------------------


def test_build_evidence_set_from_worksheet_is_independent_of_worksheet_row_order(gold_set):
    """Reordering the CSV rows (e.g. by priority_tier, as the worksheet
    builder does) must never change GoldEvidenceSet.content_hash() --
    entries are emitted in the frozen GoldSet's own annotation order."""
    rows = _baseline_worksheet_rows(gold_set)
    forward = PROMOTE.build_evidence_set_from_worksheet(gold_set, rows)
    reversed_rows = list(reversed(rows))
    backward = PROMOTE.build_evidence_set_from_worksheet(gold_set, reversed_rows)
    assert forward.content_hash() == backward.content_hash()
    assert [e.document_id for e in forward.entries] == [e.document_id for e in backward.entries]


class _FakeDraft:
    content_hash = staticmethod(lambda: "d" * 64)
    version_label = "gold_evidence_set_claude_draft_v2"


def _publish_kwargs(evidence_set, gold_set, output_dir):
    return dict(
        evidence_set=evidence_set, output_dir=output_dir, gold_set=gold_set, draft=_FakeDraft(),
        worksheet_path=Path("worksheet.csv"), worksheet_sha256="e" * 64, version_label="gold_evidence_set_v1",
    )


def test_publish_evidence_set_and_manifest_fresh_writes_both(gold_set, tmp_path):
    evidence_set = _minimal_evidence_set(gold_set)
    output_dir = tmp_path / "out"
    evidence_path, manifest_path, status = PROMOTE.publish_evidence_set_and_manifest(**_publish_kwargs(evidence_set, gold_set, output_dir))
    assert status == "wrote_both"
    assert evidence_path.exists() and manifest_path.exists()

    manifest = PROMOTE.load_frozen_model(PROMOTE.GoldEvidenceSetPromotionManifest, manifest_path, expected_artifact_type="GoldEvidenceSetPromotionManifest")
    assert manifest.evidence_set_content_hash == evidence_set.content_hash()
    assert manifest.gold_set_hash == gold_set.content_hash()
    assert manifest.draft_content_hash == "d" * 64
    assert manifest.worksheet_sha256 == "e" * 64


def test_publish_evidence_set_and_manifest_already_complete_is_idempotent(gold_set, tmp_path):
    evidence_set = _minimal_evidence_set(gold_set)
    output_dir = tmp_path / "out"
    kwargs = _publish_kwargs(evidence_set, gold_set, output_dir)
    PROMOTE.publish_evidence_set_and_manifest(**kwargs)
    evidence_mtime = (output_dir / "gold_evidence_set_v1.json").stat().st_mtime_ns

    _, _, status = PROMOTE.publish_evidence_set_and_manifest(**kwargs)
    assert status == "already_complete"
    # nothing was rewritten
    assert (output_dir / "gold_evidence_set_v1.json").stat().st_mtime_ns == evidence_mtime


def test_publish_evidence_set_and_manifest_recovers_after_kill_between_writes(gold_set, tmp_path):
    """Kill-simulation: freeze ONLY the evidence artifact (as if the
    process died right after that write, before the manifest was
    written), then call publish_evidence_set_and_manifest again and
    confirm it completes by writing ONLY the missing manifest -- the
    evidence artifact is never re-frozen (proven by its mtime/bytes being
    untouched, and by the fact a second freeze_frozen_model call at the
    same path would itself raise FileExistsError if attempted)."""
    evidence_set = _minimal_evidence_set(gold_set)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    # Simulate: only the first write of the pair completed.
    evidence_path = PROMOTE.freeze_evidence_set(evidence_set, output_dir=output_dir, version_label="gold_evidence_set_v1")
    evidence_bytes_before = evidence_path.read_bytes()
    manifest_path = output_dir / "gold_evidence_set_v1_promotion_manifest.json"
    assert not manifest_path.exists()

    # Retry the full publish call -- must recover, not re-freeze, not raise.
    result_evidence_path, result_manifest_path, status = PROMOTE.publish_evidence_set_and_manifest(**_publish_kwargs(evidence_set, gold_set, output_dir))

    assert status == "wrote_manifest_only"
    assert result_evidence_path == evidence_path
    assert result_manifest_path == manifest_path
    assert manifest_path.exists()
    assert evidence_path.read_bytes() == evidence_bytes_before  # byte-identical -- proven untouched, not merely unraised

    manifest = PROMOTE.load_frozen_model(PROMOTE.GoldEvidenceSetPromotionManifest, manifest_path, expected_artifact_type="GoldEvidenceSetPromotionManifest")
    assert manifest.evidence_set_content_hash == evidence_set.content_hash()

    # A second retry after full recovery must now report already_complete.
    _, _, status2 = PROMOTE.publish_evidence_set_and_manifest(**_publish_kwargs(evidence_set, gold_set, output_dir))
    assert status2 == "already_complete"


def test_publish_evidence_set_and_manifest_manifest_without_evidence_raises(tmp_path, gold_set):
    """A manifest present without its evidence artifact is an
    inconsistent state this function must never try to repair by
    guessing -- it raises."""
    evidence_set = _minimal_evidence_set(gold_set)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    manifest = PROMOTE._build_manifest(
        version_label="gold_evidence_set_v1", evidence_set=evidence_set, gold_set=gold_set, draft=_FakeDraft(),
        worksheet_path=Path("worksheet.csv"), worksheet_sha256="e" * 64,
    )
    from ft_screening.gold import freeze_frozen_model
    freeze_frozen_model(manifest, output_dir=output_dir, version_label="gold_evidence_set_v1_promotion_manifest", artifact_type="GoldEvidenceSetPromotionManifest")

    with pytest.raises(ValueError, match="inconsistent state"):
        PROMOTE.publish_evidence_set_and_manifest(**_publish_kwargs(evidence_set, gold_set, output_dir))


def test_publish_evidence_set_and_manifest_hash_mismatch_raises(gold_set, tmp_path):
    """A DIFFERENT evidence_set (different content) than the one already
    frozen at this path must be refused, not silently overwritten."""
    evidence_set = _minimal_evidence_set(gold_set)
    output_dir = tmp_path / "out"
    PROMOTE.publish_evidence_set_and_manifest(**_publish_kwargs(evidence_set, gold_set, output_dir))

    other_rows = _baseline_worksheet_rows(gold_set)
    other_rows[0] = dict(other_rows[0], review_notes="A different review_notes value, changing the content hash.")
    other_evidence_set = PROMOTE.build_evidence_set_from_worksheet(gold_set, other_rows)
    assert other_evidence_set.content_hash() != evidence_set.content_hash()

    with pytest.raises(ValueError, match="inconsistent state"):
        PROMOTE.publish_evidence_set_and_manifest(**_publish_kwargs(other_evidence_set, gold_set, output_dir))
