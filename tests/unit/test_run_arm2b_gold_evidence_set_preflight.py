"""Offline tests for scripts/run_arm2b_gold_evidence_set.py -- the
leakage-control redaction logic, the model-visible evidence projection,
and preflight()'s full wiring (hash locking, promotion-manifest
verification, idempotent projection freeze), all isolated from the real
repo's frozen artifacts via the same `_load_module`-with-path-overrides
pattern test_run_arm2_gold_evidence_preflight.py /
test_promote_gold_evidence_set_from_worksheet.py already use. Zero
network/API calls anywhere in this file -- preflight() never constructs
an adjudicator/transport, and none of these tests call execute()/main(
["--execute"]).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import fitz
import pytest

sys.path.insert(0, r"C:/projects/FT_SCREENING/src")
sys.path.insert(0, r"C:/projects/FT_SCREENING/scripts")

from ft_screening.gold import GoldCriterionAnnotation, freeze_frozen_model, load_frozen_model  # noqa: E402
from ft_screening.gold_evidence_set import GoldEvidenceEntry, GoldEvidenceExcerpt, GoldEvidenceSet, now_iso  # noqa: E402
from ft_screening.schema import EvidenceSpan  # noqa: E402

SCRIPTS_DIR = Path(r"C:/projects/FT_SCREENING/scripts")


def _load_module(name: str, **path_overrides):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    # Registered in sys.modules BEFORE exec (matching what a normal `import`
    # does) -- pydantic's forward-ref resolution for locally-defined nested
    # models (Arm2bEvidenceProjectionEntry's `tuple[Arm2bEvidenceChunk, ...]`
    # field, under `from __future__ import annotations`) looks the module up
    # via sys.modules[cls.__module__]; without this, class construction
    # succeeds but the model is left "not fully defined" and later
    # instantiation raises. A real quirk of dynamic-load-by-path, not a
    # defect in the script itself (importing it normally, e.g. `python
    # scripts/run_arm2b_gold_evidence_set.py`, registers it as `__main__`
    # automatically and never hits this).
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    for attr, value in path_overrides.items():
        setattr(mod, attr, value)
    return mod


ARM2B = _load_module("run_arm2b_gold_evidence_set")


# ---------------------------------------------------------------------------
# Low-level: _redacted_evidence_spans_for_key / build_evidence_projection --
# fixture entries carry a deliberately answer-shaped stance/review_notes so
# a leak would be immediately visible in the projection output.
# ---------------------------------------------------------------------------


def _fake_entry(*, stance: str = "supports", review_notes: str = "SECRET_MARKER_REVIEW_NOTES_MUST_NOT_LEAK") -> GoldEvidenceEntry:
    return GoldEvidenceEntry(
        document_id="d1", criterion_id="c1",
        original_supporting_pages=(3,), original_supporting_passages=("A real substantive supporting passage.",),
        self_contained_excerpts=(
            GoldEvidenceExcerpt(
                stance=stance, page=3, evidence_role="verbatim_text",
                heading_or_quotation="A real substantive supporting passage.",
                supported_inference="SECRET_MARKER_SUPPORTED_INFERENCE_MUST_NOT_LEAK",
                extraction_method="retained_original_citation",
            ),
        ),
        sufficient_for_independent_adjudication=True, sufficiency_assessed_by="human_review:RR",
        sufficiency_rationale="SECRET_MARKER_SUFFICIENCY_RATIONALE_MUST_NOT_LEAK",
        reviewer_id="RR", reviewed_at="20/08/2026", review_notes=review_notes,
    )


def test_redacted_evidence_spans_stance_always_relevant_regardless_of_source():
    for source_stance in ("supports", "contradicts"):
        entry = _fake_entry(stance=source_stance)
        spans = ARM2B._redacted_evidence_spans_for_key(document_id="d1", criterion_id="c1", entry=entry, document_sha256="a" * 64)
        assert len(spans) == 1
        assert spans[0].stance == "relevant"


def test_redacted_evidence_spans_chunk_id_is_neutral():
    entry = _fake_entry()
    spans = ARM2B._redacted_evidence_spans_for_key(document_id="d1", criterion_id="c1", entry=entry, document_sha256="a" * 64)
    chunk_id = spans[0].chunk_id
    assert "support" not in chunk_id.lower()
    assert "contradict" not in chunk_id.lower()
    assert chunk_id == "arm2b-d1-c1-0"


def test_redacted_evidence_spans_excerpt_matches_heading_or_quotation_exactly():
    entry = _fake_entry()
    spans = ARM2B._redacted_evidence_spans_for_key(document_id="d1", criterion_id="c1", entry=entry, document_sha256="a" * 64)
    assert spans[0].excerpt == "A real substantive supporting passage."
    assert "SECRET_MARKER" not in spans[0].excerpt


def test_evidence_span_schema_fields_present_and_correct(monkeypatch=None):
    entry = _fake_entry()
    spans = ARM2B._redacted_evidence_spans_for_key(document_id="d1", criterion_id="c1", entry=entry, document_sha256="a" * 64)
    span = spans[0]
    assert isinstance(span, EvidenceSpan)
    assert span.page == 3
    assert span.section is None  # source_section was never set on the fixture excerpt
    assert span.document_sha256 == "a" * 64


def test_arm2b_evidence_chunk_field_set_is_exactly_four_fields():
    """Regression guard: a future accidental field addition to
    Arm2bEvidenceChunk (e.g. re-adding stance) must fail this test."""
    assert set(ARM2B.Arm2bEvidenceChunk.model_fields.keys()) == {"chunk_id", "page", "section", "excerpt"}


def _gold_set_with_entries(entries: dict) -> tuple[object, GoldEvidenceSet]:
    """Minimal duck-typed gold_set stand-in for build_evidence_projection,
    which only reads .criterion_annotations and .document_pdf_sha256 --
    matches the pattern already used in
    test_promote_gold_evidence_set_from_worksheet.py's low-level tests."""
    from types import SimpleNamespace

    anns = [
        GoldCriterionAnnotation(
            document_id=doc_id, criterion_id=crit_id, document_page_count=10, gold_status="satisfied",
            supporting_pages=(3,), supporting_passages=("A real substantive supporting passage.",),
            contradicting_pages=(), contradicting_passages=(), evidence_absent=False, evidence_ambiguous=False,
            evidence_conflicting=False, human_review_required=False, annotator_id="RR", annotation_round="agreed",
        )
        for (doc_id, crit_id) in entries
    ]
    gold_set = SimpleNamespace(
        criterion_annotations=anns, document_pdf_sha256={doc_id: "a" * 64 for (doc_id, _crit_id) in entries},
        content_hash=lambda: "b" * 64,
    )
    evidence_set = GoldEvidenceSet(
        gold_set_hash="b" * 64, version_label="test", frozen_at=now_iso(),
        entries=tuple(entries[(doc_id, crit_id)] for (doc_id, crit_id) in entries),
    )
    return gold_set, evidence_set


def test_build_evidence_projection_contains_no_review_notes_or_supported_inference_leak():
    entry = GoldEvidenceEntry(
        document_id="d1", criterion_id="c1",
        original_supporting_pages=(3,), original_supporting_passages=("A real substantive supporting passage.",),
        self_contained_excerpts=(
            GoldEvidenceExcerpt(
                stance="supports", page=3, evidence_role="verbatim_text",
                heading_or_quotation="A real substantive supporting passage.",
                supported_inference="SECRET_MARKER_SUPPORTED_INFERENCE_MUST_NOT_LEAK",
                extraction_method="retained_original_citation",
            ),
        ),
        sufficient_for_independent_adjudication=True, sufficiency_assessed_by="human_review:RR",
        sufficiency_rationale="SECRET_MARKER_SUFFICIENCY_RATIONALE_MUST_NOT_LEAK",
        reviewer_id="RR", reviewed_at="20/08/2026", review_notes="SECRET_MARKER_REVIEW_NOTES_MUST_NOT_LEAK",
    )
    gold_set, evidence_set = _gold_set_with_entries({("d1", "c1"): entry})
    projection = ARM2B.build_evidence_projection(gold_set, evidence_set)

    serialized = json.dumps(projection.model_dump(mode="json"))
    for marker in ("SECRET_MARKER_SUPPORTED_INFERENCE", "SECRET_MARKER_SUFFICIENCY_RATIONALE", "SECRET_MARKER_REVIEW_NOTES"):
        assert marker not in serialized, f"{marker} leaked into the evidence projection"
    assert "RR" not in serialized  # reviewer_id must not appear either
    assert "human_review" not in serialized  # sufficiency_assessed_by tag must not appear


def test_build_evidence_projection_empty_package_produces_zero_chunks():
    entry = GoldEvidenceEntry(
        document_id="d1", criterion_id="c1", sufficient_for_independent_adjudication=None,
        reviewer_id="RR", reviewed_at="20/08/2026", review_notes="Exhaustive search, nothing found.",
        human_attestation="no_qualifying_evidence_found_after_full_document_review",
    )
    gold_set, evidence_set = _gold_set_with_entries({("d1", "c1"): entry})
    projection = ARM2B.build_evidence_projection(gold_set, evidence_set)
    assert projection.entries[0].chunks == ()


# ---------------------------------------------------------------------------
# ensure_evidence_projection_frozen: idempotency + tamper detection.
# ---------------------------------------------------------------------------


def test_ensure_evidence_projection_frozen_idempotent(tmp_path):
    entry = _fake_entry()
    gold_set, evidence_set = _gold_set_with_entries({("d1", "c1"): entry})
    output_dir = tmp_path / "out"

    projection1, path1 = ARM2B.ensure_evidence_projection_frozen(gold_set, evidence_set, output_dir=output_dir)
    mtime1 = path1.stat().st_mtime_ns
    projection2, path2 = ARM2B.ensure_evidence_projection_frozen(gold_set, evidence_set, output_dir=output_dir)
    assert path1 == path2
    assert path2.stat().st_mtime_ns == mtime1  # not rewritten
    assert projection1.content_hash() == projection2.content_hash()


def test_ensure_evidence_projection_frozen_detects_mismatch(tmp_path):
    entry = _fake_entry()
    gold_set, evidence_set = _gold_set_with_entries({("d1", "c1"): entry})
    output_dir = tmp_path / "out"
    ARM2B.ensure_evidence_projection_frozen(gold_set, evidence_set, output_dir=output_dir)

    other_entry = _fake_entry(review_notes="A different review_notes value.")
    other_gold_set, other_evidence_set = _gold_set_with_entries({("d1", "c1"): other_entry})
    with pytest.raises(ValueError, match="refusing to overwrite"):
        ARM2B.ensure_evidence_projection_frozen(other_gold_set, other_evidence_set, output_dir=output_dir)


# ---------------------------------------------------------------------------
# verify_locked_config drift detection.
# ---------------------------------------------------------------------------


def test_verify_locked_config_passes_against_live_adjudicator():
    ARM2B.verify_locked_config()  # must not raise


def test_verify_locked_config_detects_prompt_version_drift(monkeypatch):
    monkeypatch.setattr(ARM2B, "LOCKED_PROMPT_TEMPLATE_VERSION", "adjudication-prompt-999")
    with pytest.raises(ARM2B.LockedConfigDriftError, match="prompt_template_version"):
        ARM2B.verify_locked_config()


def test_verify_locked_config_detects_schema_version_drift(monkeypatch):
    monkeypatch.setattr(ARM2B, "LOCKED_OUTPUT_SCHEMA_VERSION", "adjudication-schema-999")
    with pytest.raises(ARM2B.LockedConfigDriftError, match="output_schema_version"):
        ARM2B.verify_locked_config()


# ---------------------------------------------------------------------------
# Full isolated preflight(): real GoldSet/GoldEvidenceSet/manifest built via
# the actual freeze mechanisms, in tmp_path, with module path/hash constants
# overridden -- proves the full wiring (hash lock, manifest verification,
# coverage check, idempotent projection freeze, output-collision check)
# without touching the real repo's frozen artifacts and without any
# network/API access.
# ---------------------------------------------------------------------------


def _build_gold_set_and_evidence_set(tmp_path):
    from ft_screening.gold import freeze_gold_set
    from ft_screening.schema import CriteriaSet, CriterionSpec

    document_ids = ["d1", "d2"]
    sealed_dir = tmp_path / "_sealed"
    sealed_dir.mkdir()
    seal_dir = tmp_path / "_primary_seal"
    seal_dir.mkdir()

    criteria_set = CriteriaSet(criteria_set_id="arm2b-preflight-test", version="1", criteria=[
        CriterionSpec(criterion_id="pop", version="1", eligibility_requirement="x", missing_evidence_policy="review", exclusion_reason_code="wp", exclusion_reason_label="wp", priority=1),
        CriterionSpec(criterion_id="design", version="1", eligibility_requirement="y", missing_evidence_policy="pass", exclusion_reason_code="wd", exclusion_reason_label="wd", priority=2),
    ])

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

    (tmp_path / "blind_manifest_v1.json").write_text(json.dumps({
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

    criterion_fields = [
        "document_id", "document_page_count", "criterion_id", "criterion_text", "gold_status", "supporting_pages",
        "supporting_passages", "contradicting_pages", "contradicting_passages", "evidence_absent", "evidence_ambiguous",
        "evidence_conflicting", "human_review_required", "annotator_id", "annotation_round", "notes",
    ]
    document_fields = [
        "document_id", "document_page_count", "gold_final_decision", "gold_primary_exclusion_reason_code",
        "aggregation_override_rationale", "annotator_id", "annotation_round", "notes",
    ]

    def _crit_row(d, c):
        return {k: "" for k in criterion_fields} | {
            "document_id": d, "document_page_count": "5", "criterion_id": c, "criterion_text": f"criterion {c}",
            "gold_status": "satisfied", "supporting_pages": "[1]", "supporting_passages": '["A real substantive supporting passage."]',
            "contradicting_pages": "[]", "contradicting_passages": "[]", "evidence_absent": "FALSE",
            "evidence_ambiguous": "FALSE", "evidence_conflicting": "FALSE", "human_review_required": "FALSE",
            "annotator_id": "RR", "annotation_round": "primary", "notes": "",
        }

    import csv

    def _write_csv(path, fieldnames, rows):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    _write_csv(tmp_path / "worksheet_phase1_criteria.csv", criterion_fields, [_crit_row(d, c) for d in document_ids for c in ("pop", "design")])
    _write_csv(tmp_path / "worksheet_phase1_documents.csv", document_fields, [
        {k: "" for k in document_fields} | {"document_id": d, "document_page_count": "5", "gold_final_decision": "include", "annotator_id": "RR", "annotation_round": "primary"}
        for d in document_ids
    ])

    seal_mod = _load_module("gold_set_seal_worksheet_pass", GOLD_SET_DIR=tmp_path, SEALED_DIR=sealed_dir, SEAL_DIR=seal_dir, CRITERIA_SET=criteria_set)
    sys.argv = ["gold_set_seal_worksheet_pass.py", "primary"]
    seal_mod.main()

    merge_mod = _load_module(
        "gold_set_merge_annotation_sources", GOLD_SET_DIR=tmp_path, SEALED_DIR=sealed_dir, SEAL_DIR=seal_dir,
        MERGED_CRITERION_PATH=tmp_path / "worksheet_merged_criteria.csv", MERGED_DOCUMENT_PATH=tmp_path / "worksheet_merged_documents.csv",
        PROVENANCE_PATH=seal_dir / "source_provenance_v1.json", CRITERIA_SET=criteria_set,
    )
    merge_mod.main()

    gold_set = freeze_gold_set(
        protocol_version="gold_set_protocol_v9", criteria_set=criteria_set,
        criterion_csv_path=tmp_path / "worksheet_merged_criteria.csv", document_csv_path=tmp_path / "worksheet_merged_documents.csv",
        blind_manifest_path=tmp_path / "blind_manifest_v1.json", sample_manifest_path=sealed_dir / "sample_v1.json",
        output_dir=tmp_path / "frozen", version_label="gold_set_v1", source_provenance_path=seal_dir / "source_provenance_v1.json",
    )

    entries = tuple(
        GoldEvidenceEntry(
            document_id=ann.document_id, criterion_id=ann.criterion_id,
            original_supporting_pages=ann.supporting_pages, original_supporting_passages=ann.supporting_passages,
            self_contained_excerpts=(
                GoldEvidenceExcerpt(
                    stance="supports", page=ann.supporting_pages[0], evidence_role="verbatim_text",
                    heading_or_quotation=ann.supporting_passages[0], supported_inference="test inference",
                    extraction_method="retained_original_citation",
                ),
            ),
            sufficient_for_independent_adjudication=True, sufficiency_assessed_by="human_review:RR",
            sufficiency_rationale="test", reviewer_id="RR", reviewed_at="20/08/2026", review_notes="Sufficient: fixture row.",
        )
        for ann in gold_set.criterion_annotations
    )
    evidence_set = GoldEvidenceSet(gold_set_hash=gold_set.content_hash(), version_label="gold_evidence_set_v1", entries=entries, frozen_at=now_iso())
    return gold_set, evidence_set


def test_preflight_isolated_end_to_end(tmp_path):
    gold_set, evidence_set = _build_gold_set_and_evidence_set(tmp_path)

    ge_output_dir = tmp_path / "gold_evidence_set_v1"
    ge_output_dir.mkdir()
    freeze_frozen_model(evidence_set, output_dir=ge_output_dir, version_label="gold_evidence_set_v1", artifact_type="GoldEvidenceSet")

    manifest = ARM2B.GoldEvidenceSetPromotionManifest(
        version_label="gold_evidence_set_v1", evidence_set_content_hash=evidence_set.content_hash(),
        gold_set_hash=gold_set.content_hash(), draft_content_hash="c" * 64, draft_version_label="draft-test",
        worksheet_path="worksheet.csv", worksheet_sha256="d" * 64, promoted_at=now_iso(),
    )
    freeze_frozen_model(manifest, output_dir=ge_output_dir, version_label="gold_evidence_set_v1_promotion_manifest", artifact_type="GoldEvidenceSetPromotionManifest")

    isolated = _load_module(
        "run_arm2b_gold_evidence_set",
        FROZEN_GOLD_SET_PATH=tmp_path / "frozen" / "gold_set_v1.json", EXPECTED_GOLD_SET_HASH=gold_set.content_hash(),
        FROZEN_GOLD_EVIDENCE_SET_PATH=ge_output_dir / "gold_evidence_set_v1.json", EXPECTED_GOLD_EVIDENCE_SET_HASH=evidence_set.content_hash(),
        PROMOTION_MANIFEST_PATH=ge_output_dir / "gold_evidence_set_v1_promotion_manifest.json",
        EXPECTED_CRITERION_ANNOTATION_COUNT=4, OUTPUT_DIR=tmp_path / "arm2b_out", CHECKPOINT_DIR=tmp_path / "arm2b_out" / "checkpoints",
    )

    pf = isolated.preflight()
    assert pf.gold_evidence_set.content_hash() == evidence_set.content_hash()
    # zero API calls: preflight() succeeded with no OPENAI_API_KEY / my_api_keys loaded
    # anywhere in this test -- proof by construction, not merely an unasserted claim.
    assert (tmp_path / "arm2b_out" / "arm2b_evidence_projection_v1.json").exists()
    assert list((tmp_path / "arm2b_out").glob("*"))  # projection file (and only that) was written


def test_preflight_isolated_rejects_wrong_locked_evidence_set_hash(tmp_path):
    gold_set, evidence_set = _build_gold_set_and_evidence_set(tmp_path)
    ge_output_dir = tmp_path / "gold_evidence_set_v1"
    ge_output_dir.mkdir()
    freeze_frozen_model(evidence_set, output_dir=ge_output_dir, version_label="gold_evidence_set_v1", artifact_type="GoldEvidenceSet")
    manifest = ARM2B.GoldEvidenceSetPromotionManifest(
        version_label="gold_evidence_set_v1", evidence_set_content_hash=evidence_set.content_hash(),
        gold_set_hash=gold_set.content_hash(), draft_content_hash="c" * 64, draft_version_label="draft-test",
        worksheet_path="worksheet.csv", worksheet_sha256="d" * 64, promoted_at=now_iso(),
    )
    freeze_frozen_model(manifest, output_dir=ge_output_dir, version_label="gold_evidence_set_v1_promotion_manifest", artifact_type="GoldEvidenceSetPromotionManifest")

    isolated = _load_module(
        "run_arm2b_gold_evidence_set",
        FROZEN_GOLD_SET_PATH=tmp_path / "frozen" / "gold_set_v1.json", EXPECTED_GOLD_SET_HASH=gold_set.content_hash(),
        FROZEN_GOLD_EVIDENCE_SET_PATH=ge_output_dir / "gold_evidence_set_v1.json",
        EXPECTED_GOLD_EVIDENCE_SET_HASH="0" * 64,  # WRONG, deliberately
        PROMOTION_MANIFEST_PATH=ge_output_dir / "gold_evidence_set_v1_promotion_manifest.json",
        EXPECTED_CRITERION_ANNOTATION_COUNT=4, OUTPUT_DIR=tmp_path / "arm2b_out", CHECKPOINT_DIR=tmp_path / "arm2b_out" / "checkpoints",
    )
    with pytest.raises(ValueError, match="LOCKED"):
        isolated.preflight()


def test_preflight_isolated_rejects_manifest_hash_mismatch(tmp_path):
    gold_set, evidence_set = _build_gold_set_and_evidence_set(tmp_path)
    ge_output_dir = tmp_path / "gold_evidence_set_v1"
    ge_output_dir.mkdir()
    freeze_frozen_model(evidence_set, output_dir=ge_output_dir, version_label="gold_evidence_set_v1", artifact_type="GoldEvidenceSet")
    # Manifest points at a DIFFERENT (wrong) evidence_set_content_hash.
    manifest = ARM2B.GoldEvidenceSetPromotionManifest(
        version_label="gold_evidence_set_v1", evidence_set_content_hash="0" * 64,
        gold_set_hash=gold_set.content_hash(), draft_content_hash="c" * 64, draft_version_label="draft-test",
        worksheet_path="worksheet.csv", worksheet_sha256="d" * 64, promoted_at=now_iso(),
    )
    freeze_frozen_model(manifest, output_dir=ge_output_dir, version_label="gold_evidence_set_v1_promotion_manifest", artifact_type="GoldEvidenceSetPromotionManifest")

    isolated = _load_module(
        "run_arm2b_gold_evidence_set",
        FROZEN_GOLD_SET_PATH=tmp_path / "frozen" / "gold_set_v1.json", EXPECTED_GOLD_SET_HASH=gold_set.content_hash(),
        FROZEN_GOLD_EVIDENCE_SET_PATH=ge_output_dir / "gold_evidence_set_v1.json", EXPECTED_GOLD_EVIDENCE_SET_HASH=evidence_set.content_hash(),
        PROMOTION_MANIFEST_PATH=ge_output_dir / "gold_evidence_set_v1_promotion_manifest.json",
        EXPECTED_CRITERION_ANNOTATION_COUNT=4, OUTPUT_DIR=tmp_path / "arm2b_out", CHECKPOINT_DIR=tmp_path / "arm2b_out" / "checkpoints",
    )
    with pytest.raises(ValueError, match="promotion manifest"):
        isolated.preflight()


def test_preflight_isolated_idempotent_second_call(tmp_path):
    gold_set, evidence_set = _build_gold_set_and_evidence_set(tmp_path)
    ge_output_dir = tmp_path / "gold_evidence_set_v1"
    ge_output_dir.mkdir()
    freeze_frozen_model(evidence_set, output_dir=ge_output_dir, version_label="gold_evidence_set_v1", artifact_type="GoldEvidenceSet")
    manifest = ARM2B.GoldEvidenceSetPromotionManifest(
        version_label="gold_evidence_set_v1", evidence_set_content_hash=evidence_set.content_hash(),
        gold_set_hash=gold_set.content_hash(), draft_content_hash="c" * 64, draft_version_label="draft-test",
        worksheet_path="worksheet.csv", worksheet_sha256="d" * 64, promoted_at=now_iso(),
    )
    freeze_frozen_model(manifest, output_dir=ge_output_dir, version_label="gold_evidence_set_v1_promotion_manifest", artifact_type="GoldEvidenceSetPromotionManifest")

    isolated = _load_module(
        "run_arm2b_gold_evidence_set",
        FROZEN_GOLD_SET_PATH=tmp_path / "frozen" / "gold_set_v1.json", EXPECTED_GOLD_SET_HASH=gold_set.content_hash(),
        FROZEN_GOLD_EVIDENCE_SET_PATH=ge_output_dir / "gold_evidence_set_v1.json", EXPECTED_GOLD_EVIDENCE_SET_HASH=evidence_set.content_hash(),
        PROMOTION_MANIFEST_PATH=ge_output_dir / "gold_evidence_set_v1_promotion_manifest.json",
        EXPECTED_CRITERION_ANNOTATION_COUNT=4, OUTPUT_DIR=tmp_path / "arm2b_out", CHECKPOINT_DIR=tmp_path / "arm2b_out" / "checkpoints",
    )
    pf1 = isolated.preflight()
    pf2 = isolated.preflight()  # must not raise (idempotent)
    assert pf1.evidence_projection.content_hash() == pf2.evidence_projection.content_hash()


# ---------------------------------------------------------------------------
# finalize_from_checkpoints(): offline v2 re-finalization -- zero new
# provider calls, requires v1 to exist first, refuses on missing/drifted
# checkpoints, refuses to overwrite an existing v2.
# ---------------------------------------------------------------------------


def _setup_isolated(tmp_path):
    gold_set, evidence_set = _build_gold_set_and_evidence_set(tmp_path)
    ge_output_dir = tmp_path / "gold_evidence_set_v1"
    ge_output_dir.mkdir()
    freeze_frozen_model(evidence_set, output_dir=ge_output_dir, version_label="gold_evidence_set_v1", artifact_type="GoldEvidenceSet")
    manifest = ARM2B.GoldEvidenceSetPromotionManifest(
        version_label="gold_evidence_set_v1", evidence_set_content_hash=evidence_set.content_hash(),
        gold_set_hash=gold_set.content_hash(), draft_content_hash="c" * 64, draft_version_label="draft-test",
        worksheet_path="worksheet.csv", worksheet_sha256="d" * 64, promoted_at=now_iso(),
    )
    freeze_frozen_model(manifest, output_dir=ge_output_dir, version_label="gold_evidence_set_v1_promotion_manifest", artifact_type="GoldEvidenceSetPromotionManifest")

    isolated = _load_module(
        "run_arm2b_gold_evidence_set",
        FROZEN_GOLD_SET_PATH=tmp_path / "frozen" / "gold_set_v1.json", EXPECTED_GOLD_SET_HASH=gold_set.content_hash(),
        FROZEN_GOLD_EVIDENCE_SET_PATH=ge_output_dir / "gold_evidence_set_v1.json", EXPECTED_GOLD_EVIDENCE_SET_HASH=evidence_set.content_hash(),
        PROMOTION_MANIFEST_PATH=ge_output_dir / "gold_evidence_set_v1_promotion_manifest.json",
        EXPECTED_CRITERION_ANNOTATION_COUNT=4, OUTPUT_DIR=tmp_path / "arm2b_out", CHECKPOINT_DIR=tmp_path / "arm2b_out" / "checkpoints",
    )
    return isolated, gold_set, evidence_set


def _write_checkpoints_for(isolated, pf, *, skip_keys=frozenset(), bad_identity_for=frozenset()):
    """Directly freezes GoldEvidenceCriterionCheckpoint files matching
    what a real --execute run would have produced -- entirely offline,
    computing the SAME execution_fingerprint finalize_from_checkpoints()
    itself derives, so a normal run round-trips correctly; `bad_identity_for`
    deliberately writes a WRONG identity hash for specific keys, to test
    drift detection."""
    fingerprint_payload = {
        "gold_set_hash": pf.gold_set.content_hash(), "gold_evidence_set_hash": pf.gold_evidence_set.content_hash(),
        "evidence_projection_hash": pf.evidence_projection.content_hash(),
        "evidence_construction_version": isolated.ARM2B_EVIDENCE_CONSTRUCTION_VERSION,
        "provider_identity": isolated.PROVIDER_IDENTITY, "adjudicator_model": isolated.LOCKED_ADJUDICATOR_MODEL,
        "prompt_template_version": isolated.LOCKED_PROMPT_TEMPLATE_VERSION, "output_schema_version": isolated.LOCKED_OUTPUT_SCHEMA_VERSION,
        "model_options": pf.model_options,
    }
    fingerprint = isolated.hashlib.sha256(isolated.canonical_json_bytes(fingerprint_payload)).hexdigest()

    from ft_screening.components import AdjudicationOutcome, AdjudicationUsage
    from ft_screening.schema import CriterionAssessment

    isolated.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for ann in pf.gold_set.criterion_annotations:
        key = (ann.document_id, ann.criterion_id)
        if key in skip_keys:
            continue
        assessment = CriterionAssessment(
            criterion_id=ann.criterion_id, criterion_version="1", status="satisfied", evidence=[],
            rationale="r", requires_human_review=False, model=isolated.LOCKED_ADJUDICATOR_MODEL,
            prompt_template_version=isolated.LOCKED_PROMPT_TEMPLATE_VERSION, output_schema_version=isolated.LOCKED_OUTPUT_SCHEMA_VERSION,
        )
        usage = AdjudicationUsage(model=isolated.LOCKED_ADJUDICATOR_MODEL, input_tokens=1, output_tokens=1, total_tokens=2, provider_call_count=1, retry_count=0, usage_available=True)
        outcome = AdjudicationOutcome(assessment=assessment, usage=usage)
        used_identity = "0" * 64 if key in bad_identity_for else fingerprint
        checkpoint_path = isolated._gold_evidence_checkpoint_path(isolated.CHECKPOINT_DIR, document_id=ann.document_id, criterion_id=ann.criterion_id)
        freeze_frozen_model(
            isolated.GoldEvidenceCriterionCheckpoint(execution_identity_hash=used_identity, document_id=ann.document_id, criterion_id=ann.criterion_id, outcome=outcome, checkpointed_at=now_iso()),
            output_dir=isolated.CHECKPOINT_DIR, version_label=checkpoint_path.stem, artifact_type="GoldEvidenceCriterionCheckpoint",
        )
    return fingerprint


def _write_fake_v1(isolated, pf):
    """A minimal, valid historical v1 pair -- finalize_from_checkpoints()
    only reads v1 to bind its hashes into v2's provenance, never to
    re-derive scoring from it."""
    result_artifact = isolated.EvaluationResultArtifact(
        arm="gold_evidence", gold_set_hash=pf.gold_set.content_hash(), criterion_status_by_key={"d1::pop": "satisfied"},
        document_final_decision={"d1": "include"}, document_primary_exclusion_reason={"d1": None}, missing_or_failed_document_ids=(),
        missing_or_failed_criterion_keys=(), usage_unit="criterion_call", total_input_tokens=1, total_output_tokens=1,
        total_tokens=2, n_usage_available=1, n_usage_unavailable=0, n_documents_usage_complete=1,
        n_documents_usage_incomplete=0, total_wall_seconds=0.0, frozen_at=now_iso(),
    )
    freeze_frozen_model(result_artifact, output_dir=isolated.OUTPUT_DIR, version_label=isolated.RESULT_VERSION_LABEL, artifact_type="EvaluationResultArtifact")
    manifest = isolated.EvaluationRunManifest(
        arm="gold_evidence", gold_set_hash=pf.gold_set.content_hash(), execution_fingerprint="1" * 64,
        result_artifact_hash=result_artifact.content_hash(), config_summary={}, recorded_at=now_iso(),
    )
    freeze_frozen_model(manifest, output_dir=isolated.OUTPUT_DIR, version_label=isolated.MANIFEST_VERSION_LABEL, artifact_type="EvaluationRunManifest")


def test_finalize_from_checkpoints_requires_v1_to_exist_first(tmp_path):
    isolated, gold_set, evidence_set = _setup_isolated(tmp_path)
    pf = isolated.preflight()
    _write_checkpoints_for(isolated, pf)
    with pytest.raises(FileNotFoundError, match="run --execute first"):
        isolated.finalize_from_checkpoints(pf)


def test_finalize_from_checkpoints_rejects_missing_checkpoint(tmp_path):
    isolated, gold_set, evidence_set = _setup_isolated(tmp_path)
    pf = isolated.preflight()
    _write_fake_v1(isolated, pf)
    _write_checkpoints_for(isolated, pf, skip_keys={("d1", "pop")})
    with pytest.raises(ValueError, match="checkpoint\\(s\\) missing"):
        isolated.finalize_from_checkpoints(pf)


def test_finalize_from_checkpoints_rejects_drifted_checkpoint(tmp_path):
    isolated, gold_set, evidence_set = _setup_isolated(tmp_path)
    pf = isolated.preflight()
    _write_fake_v1(isolated, pf)
    _write_checkpoints_for(isolated, pf, bad_identity_for={("d1", "pop")})
    with pytest.raises(isolated.GoldEvidenceCheckpointDriftError):
        isolated.finalize_from_checkpoints(pf)


def test_finalize_from_checkpoints_zero_new_calls_and_provenance(tmp_path):
    isolated, gold_set, evidence_set = _setup_isolated(tmp_path)
    pf = isolated.preflight()
    _write_fake_v1(isolated, pf)
    _write_checkpoints_for(isolated, pf)

    isolated.finalize_from_checkpoints(pf)

    v2_result_path = isolated.OUTPUT_DIR / f"{isolated.RESULT_VERSION_LABEL_V2}.json"
    v2_manifest_path = isolated.OUTPUT_DIR / f"{isolated.MANIFEST_VERSION_LABEL_V2}.json"
    assert v2_result_path.exists() and v2_manifest_path.exists()

    manifest = isolated.load_frozen_model(isolated.EvaluationRunManifest, v2_manifest_path, expected_artifact_type="EvaluationRunManifest")
    assert manifest.config_summary["new_provider_calls"] == 0
    assert manifest.config_summary["wall_time_unavailable_after_checkpoint_resume"] is True
    assert manifest.config_summary["n_checkpoints_reused"] == 4
    assert "checkpoint_execution_identity_hash" in manifest.config_summary
    assert "runner_file_sha256" in manifest.config_summary
    assert "evaluation_harness_file_sha256" in manifest.config_summary
    assert "v1_result_content_hash" in manifest.config_summary
    assert "v1_manifest_content_hash" in manifest.config_summary

    # v1 untouched
    v1_result_path = isolated.OUTPUT_DIR / f"{isolated.RESULT_VERSION_LABEL}.json"
    assert v1_result_path.exists()


def test_finalize_from_checkpoints_refuses_v2_collision(tmp_path):
    isolated, gold_set, evidence_set = _setup_isolated(tmp_path)
    pf = isolated.preflight()
    _write_fake_v1(isolated, pf)
    _write_checkpoints_for(isolated, pf)
    isolated.finalize_from_checkpoints(pf)
    with pytest.raises(FileExistsError, match="refusing to proceed"):
        isolated.finalize_from_checkpoints(pf)
