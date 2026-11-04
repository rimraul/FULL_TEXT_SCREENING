"""Tests for ft_screening.gold (v8): the gold-standard annotation schema
and freeze mechanism for gold_set_protocol_v9.md. Fully offline/synthetic
-- no real corpus PDFs, no LLM calls; the real 20-document gold set is
annotated separately per the protocol. Real PyMuPDF-generated PDFs ARE
used here (not just placeholder hash strings) to exercise the freeze-time
real-PDF hash AND page-count verification that
codex_code_review_772916e.md finding 3 required.

Seven rounds of adversarial review found the pre-annotation
infrastructure itself not gate-ready before annotation ever began
(codex_code_review_a36e0df.md, codex_code_review_dca0970.md,
codex_code_review_772916e.md, codex_code_review_1c40a27.md,
codex_code_review_ade9f48.md, codex_code_review_93bf42d.md) -- several
tests here directly reproduce those reviews' exact adversarial repros as
named regression tests.
"""

from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path

import fitz
import pytest

from ft_screening.gold import (
    BlindManifest,
    CriterionDualReviewAudit,
    DocumentDualReviewAudit,
    GoldCriterionAnnotation,
    GoldDocumentAnnotation,
    GoldSet,
    RetrievalOracleArtifact,
    RetrievalOracleJudgmentSet,
    RetrievalOracleSpan,
    RetrievalSufficiencyJudgment,
    SampleManifest,
    WorksheetPassSeal,
    WorksheetSourceProvenance,
    WorksheetSourceRecord,
    criterion_rows_agree,
    document_rows_agree,
    freeze_gold_set,
    freeze_retrieval_oracle_artifact,
    freeze_retrieval_oracle_judgment_set,
    hash_pdf_file,
    load_frozen_gold_set,
    load_frozen_retrieval_oracle_artifact,
    load_frozen_retrieval_oracle_judgment_set,
    load_frozen_worksheet_pass_seal,
    load_gold_annotations_from_csv,
    make_evidence_key,
    real_pdf_page_count,
    seal_worksheet_pass,
    verify_worksheet_pass_seal,
)
from ft_screening.hashing import compute_criteria_set_hash
from ft_screening.schema import CriteriaSet, CriterionSpec


def make_criteria_set(**overrides) -> CriteriaSet:
    criteria = [
        CriterionSpec(
            criterion_id="population", version="1", eligibility_requirement="adult humans",
            missing_evidence_policy="review", exclusion_reason_code="wrong_population",
            exclusion_reason_label="Wrong population", priority=1,
        ),
        CriterionSpec(
            criterion_id="design", version="1", eligibility_requirement="randomized controlled trial",
            missing_evidence_policy="pass", exclusion_reason_code="wrong_design",
            exclusion_reason_label="Wrong design", priority=2,
        ),
    ]
    defaults = dict(criteria_set_id="cs-1", version="1", criteria=criteria)
    defaults.update(overrides)
    return CriteriaSet(**defaults)


def make_criterion_annotation(**overrides) -> GoldCriterionAnnotation:
    defaults = dict(
        document_id="d1", criterion_id="population", document_page_count=3, gold_status="satisfied",
        supporting_pages=(1,), supporting_passages=("adults were enrolled",),
        contradicting_pages=(), contradicting_passages=(),
        evidence_absent=False, evidence_ambiguous=False, evidence_conflicting=False,
        human_review_required=False, annotator_id="RR", annotation_round="primary", notes="",
    )
    defaults.update(overrides)
    return GoldCriterionAnnotation(**defaults)


def make_document_annotation(**overrides) -> GoldDocumentAnnotation:
    defaults = dict(
        document_id="d1", document_page_count=3, gold_final_decision="include",
        gold_primary_exclusion_reason_code=None, aggregation_override_rationale=None,
        annotator_id="RR", annotation_round="primary", notes="",
    )
    defaults.update(overrides)
    return GoldDocumentAnnotation(**defaults)


def make_pdf(tmp_path, name: str, *, pages: int = 3) -> tuple:
    """A real, PyMuPDF-generated PDF with a genuine page count -- freeze_gold_set()
    now opens this file for real (hash AND page count), so a placeholder
    byte string is no longer sufficient for these tests."""
    path = tmp_path / name
    doc = fitz.open()
    for i in range(pages):
        doc.new_page().insert_text((72, 72), f"page {i + 1} of {name}")
    doc.save(str(path))
    doc.close()
    return path, hash_pdf_file(path), real_pdf_page_count(path)


def make_manifests(tmp_path, *, document_ids, dual_review_ids=(), pages_per_doc=None):
    pages_per_doc = pages_per_doc or {doc_id: 3 for doc_id in document_ids}
    entries = []
    for doc_id in document_ids:
        pdf_path, pdf_sha256, page_count = make_pdf(tmp_path, f"{doc_id}.pdf", pages=pages_per_doc[doc_id])
        entries.append({
            "document_id": doc_id, "stratum": "INCLUDE", "iasia_reason_code": None,
            "pdf_path": str(pdf_path), "pdf_sha256": pdf_sha256, "page_count": page_count, "title": "",
        })
    sample_manifest = {
        "protocol_version": "gold_set_protocol_v9", "source_corpus": "", "excluded_as_already_used_by_walking_skeleton": [],
        "dual_review_subset": list(dual_review_ids), "documents": entries,
    }
    sample_path = tmp_path / "sample_v1.json"
    sample_path.write_text(json.dumps(sample_manifest), encoding="utf-8")

    blind_manifest = {
        "protocol_version": "gold_set_protocol_v9",
        "documents": [
            {"document_id": e["document_id"], "pdf_path": e["pdf_path"], "document_page_count": e["page_count"]}
            for e in entries
        ],
    }
    blind_path = tmp_path / "blind_manifest_v1.json"
    blind_path.write_text(json.dumps(blind_manifest), encoding="utf-8")
    return blind_path, sample_path, {e["document_id"]: e["page_count"] for e in entries}


CRITERION_FIELDS = [
    "document_id", "criterion_id", "document_page_count", "gold_status", "supporting_pages",
    "supporting_passages", "contradicting_pages", "contradicting_passages", "evidence_absent",
    "evidence_ambiguous", "evidence_conflicting", "human_review_required", "annotator_id",
    "annotation_round", "notes",
]
DOCUMENT_FIELDS = [
    "document_id", "document_page_count", "gold_final_decision", "gold_primary_exclusion_reason_code",
    "aggregation_override_rationale", "annotator_id", "annotation_round", "notes",
]


def _write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _base_criterion_row(**overrides):
    row = dict(
        document_id="d1", criterion_id="population", document_page_count="3", gold_status="satisfied",
        supporting_pages="[1]", supporting_passages='["adults enrolled"]',
        contradicting_pages="[]", contradicting_passages="[]",
        evidence_absent="FALSE", evidence_ambiguous="FALSE", evidence_conflicting="FALSE",
        human_review_required="FALSE", annotator_id="RR", annotation_round="primary", notes="",
    )
    row.update(overrides)
    return row


def _base_document_row(**overrides):
    row = dict(
        document_id="d1", document_page_count="3", gold_final_decision="include",
        gold_primary_exclusion_reason_code="", aggregation_override_rationale="",
        annotator_id="RR", annotation_round="primary", notes="",
    )
    row.update(overrides)
    return row


def _write_two_criterion_two_document_csvs(tmp_path):
    criterion_csv = tmp_path / "criteria.csv"
    document_csv = tmp_path / "documents.csv"
    _write_csv(criterion_csv, CRITERION_FIELDS, [
        _base_criterion_row(document_id="d1", criterion_id="population"),
        _base_criterion_row(document_id="d1", criterion_id="design", supporting_pages="[2]", supporting_passages='["RCT design"]'),
        _base_criterion_row(document_id="d2", criterion_id="population"),
        _base_criterion_row(document_id="d2", criterion_id="design", supporting_pages="[2]", supporting_passages='["RCT design"]'),
    ])
    _write_csv(document_csv, DOCUMENT_FIELDS, [
        _base_document_row(document_id="d1"), _base_document_row(document_id="d2"),
    ])
    return criterion_csv, document_csv


def _stub_source_provenance(**overrides) -> WorksheetSourceProvenance:
    """For tests that construct `GoldSet` DIRECTLY (not via
    `freeze_gold_set`, which is what actually verifies provenance) --
    `source_provenance` is a required field, so direct constructions need
    SOME syntactically-valid value even when the test isn't about
    provenance at all."""
    defaults = dict(
        protocol_version="gold_set_protocol_v9", criteria_set_hash="a" * 64,
        blind_manifest_hash="b" * 64, sample_manifest_hash="c" * 64,
        primary=WorksheetSourceRecord(
            role="primary", seal_path="stub/primary_seal.json", seal_content_hash="d" * 64,
            criterion_csv_path="stub/primary_criteria.csv", criterion_csv_sha256="e" * 64,
            document_csv_path="stub/primary_documents.csv", document_csv_sha256="f" * 64,
        ),
        merged_criterion_csv_sha256="1" * 64, merged_document_csv_sha256="2" * 64,
        merged_at="2026-08-16T00:00:00+00:00",
    )
    defaults.update(overrides)
    return WorksheetSourceProvenance(**defaults)


def _split_rows_by_round(rows: list[dict]) -> dict:
    by_round: dict = {"primary": [], "second_pass": [], "adjudicated": []}
    for row in rows:
        by_round[row["annotation_round"]].append(row)
    return by_round


def _build_source_provenance(
    scratch_dir, *, protocol_version, criteria_set, criterion_csv_path, document_csv_path,
    blind_manifest_path, sample_manifest_path,
):
    """Test-only plumbing standing in for the real
    gold_set_generate_second_pass_worksheets.py /
    gold_set_merge_annotation_sources.py scripts: splits an
    already-merged criterion/document CSV pair by `annotation_round` back
    into genuinely separate per-pass files, seals the primary (and
    second_pass, if the sample requires dual review) pass, and writes a
    real, verifiable `WorksheetSourceProvenance` pointing at them --
    reproducing exactly what `freeze_gold_set()`'s recompute-and-compare
    check expects, from the SAME test fixtures every existing test
    already builds as one merged file."""
    seal_dir = scratch_dir / f"_seal_{uuid.uuid4().hex}"
    seal_dir.mkdir(parents=True, exist_ok=True)

    sample_manifest = json.loads(Path(sample_manifest_path).read_text(encoding="utf-8"))
    all_document_ids = {d["document_id"] for d in sample_manifest["documents"]}
    dual_review_ids = set(sample_manifest["dual_review_subset"])
    criteria_set_hash = compute_criteria_set_hash(criteria_set)
    blind_manifest_hash = hash_pdf_file(Path(blind_manifest_path))
    sample_manifest_hash = hash_pdf_file(Path(sample_manifest_path))

    with open(criterion_csv_path, newline="", encoding="utf-8") as f:
        criterion_rows = list(csv.DictReader(f))
    with open(document_csv_path, newline="", encoding="utf-8") as f:
        document_rows = list(csv.DictReader(f))
    criterion_by_round = _split_rows_by_round(criterion_rows)
    document_by_round = _split_rows_by_round(document_rows)

    primary_criterion_path = seal_dir / "primary_criteria.csv"
    primary_document_path = seal_dir / "primary_documents.csv"
    _write_csv(primary_criterion_path, CRITERION_FIELDS, criterion_by_round["primary"])
    _write_csv(primary_document_path, DOCUMENT_FIELDS, document_by_round["primary"])
    primary_seal = seal_worksheet_pass(
        pass_name="primary", protocol_version=protocol_version, criteria_set=criteria_set,
        blind_manifest_path=blind_manifest_path, sample_manifest_path=sample_manifest_path,
        criterion_csv_path=primary_criterion_path, document_csv_path=primary_document_path,
        expected_document_ids=all_document_ids, output_dir=seal_dir, version_label="primary_seal",
    )

    provenance = {
        "protocol_version": protocol_version, "criteria_set_hash": criteria_set_hash,
        "blind_manifest_hash": blind_manifest_hash, "sample_manifest_hash": sample_manifest_hash,
        "primary": {
            "role": "primary", "seal_path": str(seal_dir / "primary_seal.json"), "seal_content_hash": primary_seal.content_hash(),
            "criterion_csv_path": str(primary_criterion_path), "criterion_csv_sha256": hash_pdf_file(primary_criterion_path),
            "document_csv_path": str(primary_document_path), "document_csv_sha256": hash_pdf_file(primary_document_path),
        },
        "merged_at": "2026-08-16T00:00:00+00:00",
    }

    merged_criterion_rows = list(criterion_rows)
    merged_document_rows = list(document_rows)

    if dual_review_ids:
        second_pass_criterion_path = seal_dir / "second_pass_criteria.csv"
        second_pass_document_path = seal_dir / "second_pass_documents.csv"
        _write_csv(second_pass_criterion_path, CRITERION_FIELDS, criterion_by_round["second_pass"])
        _write_csv(second_pass_document_path, DOCUMENT_FIELDS, document_by_round["second_pass"])
        second_pass_seal = seal_worksheet_pass(
            pass_name="second_pass", protocol_version=protocol_version, criteria_set=criteria_set,
            blind_manifest_path=blind_manifest_path, sample_manifest_path=sample_manifest_path,
            criterion_csv_path=second_pass_criterion_path, document_csv_path=second_pass_document_path,
            expected_document_ids=dual_review_ids, output_dir=seal_dir, version_label="second_pass_seal",
        )
        adjudication_criterion_path = seal_dir / "adjudication_criteria.csv"
        adjudication_document_path = seal_dir / "adjudication_documents.csv"
        _write_csv(adjudication_criterion_path, CRITERION_FIELDS, criterion_by_round["adjudicated"])
        _write_csv(adjudication_document_path, DOCUMENT_FIELDS, document_by_round["adjudicated"])
        provenance["second_pass"] = {
            "role": "second_pass", "seal_path": str(seal_dir / "second_pass_seal.json"), "seal_content_hash": second_pass_seal.content_hash(),
            "criterion_csv_path": str(second_pass_criterion_path), "criterion_csv_sha256": hash_pdf_file(second_pass_criterion_path),
            "document_csv_path": str(second_pass_document_path), "document_csv_sha256": hash_pdf_file(second_pass_document_path),
        }
        provenance["adjudication"] = {
            "role": "adjudication", "seal_path": None, "seal_content_hash": None,
            "criterion_csv_path": str(adjudication_criterion_path), "criterion_csv_sha256": hash_pdf_file(adjudication_criterion_path),
            "document_csv_path": str(adjudication_document_path), "document_csv_sha256": hash_pdf_file(adjudication_document_path),
        }

    provenance["merged_criterion_csv_sha256"] = hash_pdf_file(Path(criterion_csv_path))
    provenance["merged_document_csv_sha256"] = hash_pdf_file(Path(document_csv_path))

    provenance_path = seal_dir / "source_provenance.json"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    return provenance_path


def _freeze(**kwargs):
    """Drop-in wrapper around `freeze_gold_set()` that also builds and
    attaches a real `source_provenance_path`, so the ~20 existing test
    call sites (each already building a valid merged CSV pair + sample
    manifest) don't each need to hand-roll the sealed-source ceremony
    just to test something unrelated to provenance."""
    kwargs = dict(kwargs)
    output_dir = Path(kwargs["output_dir"])
    provenance_path = _build_source_provenance(
        output_dir.parent, protocol_version=kwargs["protocol_version"], criteria_set=kwargs["criteria_set"],
        criterion_csv_path=kwargs["criterion_csv_path"], document_csv_path=kwargs["document_csv_path"],
        blind_manifest_path=kwargs["blind_manifest_path"], sample_manifest_path=kwargs["sample_manifest_path"],
    )
    kwargs["source_provenance_path"] = provenance_path
    return freeze_gold_set(**kwargs)


# ---------------------------------------------------------------------------
# GoldCriterionAnnotation coherence rules (unchanged from v3, still enforced)
# ---------------------------------------------------------------------------


def test_satisfied_status_requires_a_supporting_page():
    with pytest.raises(Exception):
        make_criterion_annotation(gold_status="satisfied", supporting_pages=(), supporting_passages=())


def test_blank_passage_is_rejected():
    with pytest.raises(Exception):
        make_criterion_annotation(supporting_pages=(1,), supporting_passages=("   ",))


def test_ambiguous_true_with_satisfied_status_is_rejected():
    with pytest.raises(Exception):
        make_criterion_annotation(gold_status="satisfied", evidence_ambiguous=True, human_review_required=False)


def test_page_exceeding_document_page_count_is_rejected():
    with pytest.raises(Exception):
        make_criterion_annotation(document_page_count=2, supporting_pages=(99,), supporting_passages=("x",))


# ---------------------------------------------------------------------------
# Strict blind/sample manifest schemas (codex_code_review_772916e.md finding 3)
# ---------------------------------------------------------------------------


def test_blind_manifest_rejects_an_injected_stratum_field():
    """codex_code_review_772916e.md finding 3's exact repro: a blind
    entry with an injected stratum/reason field previously froze
    successfully because the blind manifest was hashed but never parsed."""
    with pytest.raises(Exception):
        BlindManifest.model_validate({
            "protocol_version": "v4",
            "documents": [{"document_id": "d1", "pdf_path": "x.pdf", "document_page_count": 3, "stratum": "EXCLUDE"}],
        })


def test_sample_manifest_rejects_unknown_dual_review_id():
    with pytest.raises(Exception):
        SampleManifest.model_validate({
            "protocol_version": "v4", "dual_review_subset": ["d-does-not-exist"],
            "documents": [{"document_id": "d1", "stratum": "INCLUDE", "pdf_path": "x.pdf", "pdf_sha256": "a" * 64, "page_count": 3}],
        })


def test_freeze_rejects_a_blind_manifest_pointing_at_an_unrelated_pdf(tmp_path):
    """codex_code_review_772916e.md finding 3's exact repro: a blind
    manifest entry whose pdf_path pointed at a completely different file
    than the sealed manifest previously froze successfully."""
    blind_path, sample_path, _ = make_manifests(tmp_path, document_ids=["d1", "d2"])
    blind_manifest = json.loads(blind_path.read_text(encoding="utf-8"))
    blind_manifest["documents"][0]["pdf_path"] = "C:/definitely/not/the/sample.pdf"
    blind_path.write_text(json.dumps(blind_manifest), encoding="utf-8")

    criterion_csv, document_csv = _write_two_criterion_two_document_csvs(tmp_path)
    with pytest.raises(ValueError):
        _freeze(
            protocol_version="gold_set_protocol_v9", criteria_set=make_criteria_set(),
            criterion_csv_path=criterion_csv, document_csv_path=document_csv,
            blind_manifest_path=blind_path, sample_manifest_path=sample_path,
            output_dir=tmp_path / "out", version_label="gold_set_v1",
        )


def test_freeze_rejects_a_real_page_count_mismatch(tmp_path):
    """codex_code_review_772916e.md finding 3: the claimed page_count was
    trusted outright -- the real PDF's actual page count is now opened
    and compared."""
    blind_path, sample_path, _ = make_manifests(tmp_path, document_ids=["d1", "d2"])
    sample_manifest = json.loads(sample_path.read_text(encoding="utf-8"))
    sample_manifest["documents"][0]["page_count"] = 999  # real PDF only has 3 pages
    sample_path.write_text(json.dumps(sample_manifest), encoding="utf-8")

    criterion_csv, document_csv = _write_two_criterion_two_document_csvs(tmp_path)
    with pytest.raises(ValueError):
        _freeze(
            protocol_version="gold_set_protocol_v9", criteria_set=make_criteria_set(),
            criterion_csv_path=criterion_csv, document_csv_path=document_csv,
            blind_manifest_path=blind_path, sample_manifest_path=sample_path,
            output_dir=tmp_path / "out", version_label="gold_set_v1",
        )


# ---------------------------------------------------------------------------
# Dual-review audit trail (codex_code_review_772916e.md finding 2)
# ---------------------------------------------------------------------------


def test_freeze_gold_set_requires_raw_csv_paths_not_precollapsed_lists():
    """freeze_gold_set()'s signature no longer accepts pre-collapsed
    annotation lists at all -- calling it with the old v3 keyword
    arguments is a TypeError, not a silent bypass."""
    with pytest.raises(TypeError):
        freeze_gold_set(
            protocol_version="gold_set_protocol_v9", criteria_set=make_criteria_set(),
            criterion_annotations=[], document_annotations=[],  # v3's signature, no longer valid
            blind_manifest_path="x", sample_manifest_path="y", output_dir="z", version_label="w",
        )


def test_forged_dual_review_audit_with_mismatched_primary_round_is_rejected():
    """A hand-built audit record whose 'primary' row doesn't actually
    have annotation_round='primary' is rejected at construction --
    closes the shape of forgery codex_code_review_772916e.md finding 2
    demonstrated (a single relabeled row with no real second pass)."""
    fake_primary = make_criterion_annotation(annotation_round="agreed")  # not really "primary"
    with pytest.raises(Exception):
        CriterionDualReviewAudit(
            document_id="d1", criterion_id="population", primary=fake_primary,
            second_pass=make_criterion_annotation(annotation_round="second_pass"),
        )


def test_dual_review_document_missing_from_criterion_annotations_audit_is_rejected():
    """A dual-review document with no retained audit record at all (the
    exact v3 bypass: freeze a lone row and call it done) is rejected."""
    # Constructed directly (not via CSV) to isolate the GoldSet-level check.
    criteria_set = make_criteria_set()
    criterion_anns = (
        make_criterion_annotation(document_id="d1", criterion_id="population", annotation_round="agreed"),
        make_criterion_annotation(document_id="d1", criterion_id="design", annotation_round="agreed", supporting_pages=(2,), supporting_passages=("RCT design",)),
        make_criterion_annotation(document_id="d2", criterion_id="population"),
        make_criterion_annotation(document_id="d2", criterion_id="design", supporting_pages=(2,), supporting_passages=("RCT design",)),
    )
    document_anns = (
        make_document_annotation(document_id="d1", annotation_round="agreed"),
        make_document_annotation(document_id="d2"),
    )
    with pytest.raises(Exception):
        GoldSet(
            protocol_version="gold_set_protocol_v9", blind_manifest_hash="d" * 64, sample_manifest_hash="a" * 64,
            document_pdf_sha256={"d1": "b" * 64, "d2": "c" * 64}, expected_document_ids=("d1", "d2"),
            expected_dual_review_document_ids=("d1",),  # d1 has NO criterion_dual_review_audit entries at all
            criteria_set=criteria_set, criterion_annotations=criterion_anns, document_annotations=document_anns,
            source_provenance=_stub_source_provenance(), frozen_at="2026-08-16T00:00:00+00:00",
        )


def test_dual_review_document_with_a_genuine_audit_trail_is_accepted(tmp_path):
    criterion_csv, document_csv = _write_two_criterion_two_document_csvs(tmp_path)
    # Add a genuine second_pass row (agreeing) for d1's rows.
    with open(criterion_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.append(_base_criterion_row(document_id="d1", criterion_id="population", annotation_round="second_pass", annotator_id="XY"))
    rows.append(_base_criterion_row(document_id="d1", criterion_id="design", annotation_round="second_pass", annotator_id="XY", supporting_pages="[2]", supporting_passages='["RCT design"]'))
    _write_csv(criterion_csv, CRITERION_FIELDS, rows)
    with open(document_csv, newline="", encoding="utf-8") as f:
        doc_rows = list(csv.DictReader(f))
    doc_rows.append(_base_document_row(document_id="d1", annotation_round="second_pass", annotator_id="XY"))
    _write_csv(document_csv, DOCUMENT_FIELDS, doc_rows)

    blind_path, sample_path, _ = make_manifests(tmp_path, document_ids=["d1", "d2"], dual_review_ids=["d1"])
    gold_set = _freeze(
        protocol_version="gold_set_protocol_v9", criteria_set=make_criteria_set(),
        criterion_csv_path=criterion_csv, document_csv_path=document_csv,
        blind_manifest_path=blind_path, sample_manifest_path=sample_path,
        output_dir=tmp_path / "out", version_label="gold_set_v1",
    )
    assert len(gold_set.criterion_dual_review_audit) == 2
    assert len(gold_set.document_dual_review_audit) == 1
    for audit in gold_set.criterion_dual_review_audit:
        assert audit.primary.annotation_round == "primary"
        assert audit.second_pass.annotation_round == "second_pass"


def test_csv_dual_review_requires_exactly_one_primary_and_one_second_pass(tmp_path):
    criterion_csv, document_csv = _write_two_criterion_two_document_csvs(tmp_path)
    with pytest.raises(ValueError):
        load_gold_annotations_from_csv(criterion_csv, document_csv, dual_review_document_ids={"d1"})


def test_csv_raw_agreed_round_is_rejected():
    with pytest.raises(ValueError):
        GoldCriterionAnnotation.from_csv_row(_base_criterion_row(annotation_round="agreed"))


# ---------------------------------------------------------------------------
# Adjudication may only resolve a genuine disagreement, never override
# unanimous independent review (codex_code_review_1c40a27.md finding 2)
# ---------------------------------------------------------------------------


def test_audit_model_rejects_an_unnecessary_adjudicated_row_when_passes_already_agree():
    """Exact repro from the review: primary and second_pass both say
    'satisfied', but a third 'adjudicated' row says 'violated' -- must be
    rejected, not silently accepted as the authoritative answer."""
    primary = make_criterion_annotation(annotation_round="primary", gold_status="satisfied")
    second_pass = make_criterion_annotation(annotation_round="second_pass", gold_status="satisfied")
    adjudicated = make_criterion_annotation(annotation_round="adjudicated", gold_status="violated", supporting_pages=(), supporting_passages=(), contradicting_pages=(1,), contradicting_passages=("x",))
    with pytest.raises(Exception):
        CriterionDualReviewAudit(document_id="d1", criterion_id="population", primary=primary, second_pass=second_pass, adjudicated=adjudicated)


def test_audit_model_requires_adjudicated_row_when_passes_disagree():
    primary = make_criterion_annotation(annotation_round="primary", gold_status="satisfied")
    second_pass = make_criterion_annotation(annotation_round="second_pass", gold_status="violated", supporting_pages=(), supporting_passages=(), contradicting_pages=(1,), contradicting_passages=("x",))
    with pytest.raises(Exception):
        CriterionDualReviewAudit(document_id="d1", criterion_id="population", primary=primary, second_pass=second_pass)


def test_document_audit_model_rejects_an_unnecessary_adjudicated_row_when_passes_already_agree():
    primary = make_document_annotation(annotation_round="primary", gold_final_decision="include")
    second_pass = make_document_annotation(annotation_round="second_pass", gold_final_decision="include")
    adjudicated = make_document_annotation(annotation_round="adjudicated", gold_final_decision="exclude", gold_primary_exclusion_reason_code="wrong_population")
    with pytest.raises(Exception):
        DocumentDualReviewAudit(document_id="d1", primary=primary, second_pass=second_pass, adjudicated=adjudicated)


def test_csv_freeze_rejects_an_adjudicated_row_overriding_agreeing_passes(tmp_path):
    """End-to-end repro through the real CSV-consuming freeze path, not
    just the model constructor directly."""
    criterion_csv, document_csv = _write_two_criterion_two_document_csvs(tmp_path)
    with open(criterion_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # d1/population: primary and second_pass agree (both default "satisfied"/[1]),
    # but an unnecessary adjudicated row is also supplied.
    rows.append(_base_criterion_row(document_id="d1", criterion_id="population", annotation_round="second_pass", annotator_id="XY"))
    rows.append(_base_criterion_row(
        document_id="d1", criterion_id="population", annotation_round="adjudicated", annotator_id="ZZ",
        gold_status="violated", supporting_pages="[]", supporting_passages="[]",
        contradicting_pages="[1]", contradicting_passages='["actually excluded"]',
    ))
    rows.append(_base_criterion_row(document_id="d1", criterion_id="design", annotation_round="second_pass", annotator_id="XY", supporting_pages="[2]", supporting_passages='["RCT design"]'))
    _write_csv(criterion_csv, CRITERION_FIELDS, rows)
    with open(document_csv, newline="", encoding="utf-8") as f:
        doc_rows = list(csv.DictReader(f))
    doc_rows.append(_base_document_row(document_id="d1", annotation_round="second_pass", annotator_id="XY"))
    _write_csv(document_csv, DOCUMENT_FIELDS, doc_rows)

    with pytest.raises(ValueError):
        load_gold_annotations_from_csv(criterion_csv, document_csv, dual_review_document_ids={"d1"})


# ---------------------------------------------------------------------------
# protocol_version cross-validation (codex_code_review_1c40a27.md finding 3)
# ---------------------------------------------------------------------------


def test_freeze_rejects_a_blind_manifest_protocol_version_disagreeing_with_sample(tmp_path):
    """Exact repro: the committed blind and sample manifests actually
    declared different protocol_version strings (v3 vs v2) and nothing
    compared them."""
    blind_path, sample_path, _ = make_manifests(tmp_path, document_ids=["d1", "d2"])
    blind_manifest = json.loads(blind_path.read_text(encoding="utf-8"))
    blind_manifest["protocol_version"] = "gold_set_protocol_v3"
    blind_path.write_text(json.dumps(blind_manifest), encoding="utf-8")

    criterion_csv, document_csv = _write_two_criterion_two_document_csvs(tmp_path)
    with pytest.raises(ValueError):
        _freeze(
            protocol_version="gold_set_protocol_v9", criteria_set=make_criteria_set(),
            criterion_csv_path=criterion_csv, document_csv_path=document_csv,
            blind_manifest_path=blind_path, sample_manifest_path=sample_path,
            output_dir=tmp_path / "out", version_label="gold_set_v1",
        )


def test_freeze_rejects_an_unsupported_protocol_version_argument(tmp_path):
    """Exact repro: my replay changed both manifest values to unrelated
    strings and successfully froze a gold set -- freeze_gold_set must not
    accept an arbitrary nonempty string here."""
    blind_path, sample_path, _ = make_manifests(tmp_path, document_ids=["d1", "d2"])
    blind_manifest = json.loads(blind_path.read_text(encoding="utf-8"))
    blind_manifest["protocol_version"] = "totally-unrelated-string"
    blind_path.write_text(json.dumps(blind_manifest), encoding="utf-8")
    sample_manifest = json.loads(sample_path.read_text(encoding="utf-8"))
    sample_manifest["protocol_version"] = "totally-unrelated-string"
    sample_path.write_text(json.dumps(sample_manifest), encoding="utf-8")

    criterion_csv, document_csv = _write_two_criterion_two_document_csvs(tmp_path)
    with pytest.raises(ValueError):
        _freeze(
            protocol_version="totally-unrelated-string", criteria_set=make_criteria_set(),
            criterion_csv_path=criterion_csv, document_csv_path=document_csv,
            blind_manifest_path=blind_path, sample_manifest_path=sample_path,
            output_dir=tmp_path / "out", version_label="gold_set_v1",
        )


# ---------------------------------------------------------------------------
# Envelope schema version verified, outer metadata hash-covered
# (codex_code_review_1c40a27.md finding 5)
# ---------------------------------------------------------------------------


def test_load_rejects_an_unsupported_envelope_schema_version(tmp_path):
    """Exact repro: changing a valid envelope's envelope_schema_version
    from "1" to "999" without touching either hash previously loaded
    successfully."""
    blind_path, sample_path, _ = make_manifests(tmp_path, document_ids=["d1", "d2"])
    criterion_csv, document_csv = _write_two_criterion_two_document_csvs(tmp_path)
    _freeze(
        protocol_version="gold_set_protocol_v9", criteria_set=make_criteria_set(),
        criterion_csv_path=criterion_csv, document_csv_path=document_csv,
        blind_manifest_path=blind_path, sample_manifest_path=sample_path,
        output_dir=tmp_path / "out", version_label="gold_set_v1",
    )
    json_path = tmp_path / "out" / "gold_set_v1.json"
    envelope = json.loads(json_path.read_text(encoding="utf-8"))
    envelope["envelope_schema_version"] = "999"
    json_path.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(ValueError):
        load_frozen_gold_set(json_path)


def test_load_rejects_tampered_envelope_header_hash(tmp_path):
    """Tampering with content_hash/full_payload_hash together, in a way
    that would otherwise slip past the payload-level checks, is still
    caught by the outer envelope_header_hash."""
    blind_path, sample_path, _ = make_manifests(tmp_path, document_ids=["d1", "d2"])
    criterion_csv, document_csv = _write_two_criterion_two_document_csvs(tmp_path)
    _freeze(
        protocol_version="gold_set_protocol_v9", criteria_set=make_criteria_set(),
        criterion_csv_path=criterion_csv, document_csv_path=document_csv,
        blind_manifest_path=blind_path, sample_manifest_path=sample_path,
        output_dir=tmp_path / "out", version_label="gold_set_v1",
    )
    json_path = tmp_path / "out" / "gold_set_v1.json"
    envelope = json.loads(json_path.read_text(encoding="utf-8"))
    envelope["envelope_header_hash"] = "0" * 64
    json_path.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(ValueError):
        load_frozen_gold_set(json_path)


# ---------------------------------------------------------------------------
# Freeze envelope: dual-hash (semantic + full-payload), artifact_type
# (codex_code_review_772916e.md finding 5)
# ---------------------------------------------------------------------------


def test_freeze_writes_a_dual_hash_envelope_with_artifact_type(tmp_path):
    blind_path, sample_path, _ = make_manifests(tmp_path, document_ids=["d1", "d2"])
    criterion_csv, document_csv = _write_two_criterion_two_document_csvs(tmp_path)
    gold_set = _freeze(
        protocol_version="gold_set_protocol_v9", criteria_set=make_criteria_set(),
        criterion_csv_path=criterion_csv, document_csv_path=document_csv,
        blind_manifest_path=blind_path, sample_manifest_path=sample_path,
        output_dir=tmp_path / "out", version_label="gold_set_v1",
    )
    envelope = json.loads((tmp_path / "out" / "gold_set_v1.json").read_text(encoding="utf-8"))
    assert envelope["artifact_type"] == "GoldSet"
    assert envelope["content_hash"] == gold_set.content_hash()
    assert "full_payload_hash" in envelope
    assert envelope["content_hash"] != envelope["full_payload_hash"]  # frozen_at makes them genuinely different


def test_tampering_with_frozen_at_is_detected_via_full_payload_hash(tmp_path):
    """codex_code_review_772916e.md finding 5's exact repro: changing
    payload.frozen_at without changing content_hash previously passed
    load_frozen_gold_set(expected_hash=original_hash) -- content_hash
    alone doesn't cover the timestamp, but full_payload_hash does."""
    blind_path, sample_path, _ = make_manifests(tmp_path, document_ids=["d1", "d2"])
    criterion_csv, document_csv = _write_two_criterion_two_document_csvs(tmp_path)
    frozen = _freeze(
        protocol_version="gold_set_protocol_v9", criteria_set=make_criteria_set(),
        criterion_csv_path=criterion_csv, document_csv_path=document_csv,
        blind_manifest_path=blind_path, sample_manifest_path=sample_path,
        output_dir=tmp_path / "out", version_label="gold_set_v1",
    )
    json_path = tmp_path / "out" / "gold_set_v1.json"
    envelope = json.loads(json_path.read_text(encoding="utf-8"))
    envelope["payload"]["frozen_at"] = "1900-01-01T00:00:00+00:00"
    json_path.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(ValueError):
        load_frozen_gold_set(json_path, expected_hash=frozen.content_hash())


def test_load_rejects_a_mismatched_artifact_type(tmp_path):
    blind_path, sample_path, _ = make_manifests(tmp_path, document_ids=["d1", "d2"])
    criterion_csv, document_csv = _write_two_criterion_two_document_csvs(tmp_path)
    _freeze(
        protocol_version="gold_set_protocol_v9", criteria_set=make_criteria_set(),
        criterion_csv_path=criterion_csv, document_csv_path=document_csv,
        blind_manifest_path=blind_path, sample_manifest_path=sample_path,
        output_dir=tmp_path / "out", version_label="gold_set_v1",
    )
    json_path = tmp_path / "out" / "gold_set_v1.json"
    envelope = json.loads(json_path.read_text(encoding="utf-8"))
    envelope["artifact_type"] = "RetrievalOracleArtifact"
    json_path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(ValueError):
        load_frozen_gold_set(json_path)


def test_freeze_refuses_to_overwrite_an_existing_version_label(tmp_path):
    blind_path, sample_path, _ = make_manifests(tmp_path, document_ids=["d1", "d2"])
    criterion_csv, document_csv = _write_two_criterion_two_document_csvs(tmp_path)
    _freeze(
        protocol_version="gold_set_protocol_v9", criteria_set=make_criteria_set(),
        criterion_csv_path=criterion_csv, document_csv_path=document_csv,
        blind_manifest_path=blind_path, sample_manifest_path=sample_path,
        output_dir=tmp_path / "out", version_label="gold_set_v1",
    )
    with pytest.raises(FileExistsError):
        _freeze(
            protocol_version="gold_set_protocol_v9", criteria_set=make_criteria_set(),
            criterion_csv_path=criterion_csv, document_csv_path=document_csv,
            blind_manifest_path=blind_path, sample_manifest_path=sample_path,
            output_dir=tmp_path / "out", version_label="gold_set_v1",
        )


def test_load_frozen_gold_set_round_trips(tmp_path):
    blind_path, sample_path, _ = make_manifests(tmp_path, document_ids=["d1", "d2"])
    criterion_csv, document_csv = _write_two_criterion_two_document_csvs(tmp_path)
    frozen = _freeze(
        protocol_version="gold_set_protocol_v9", criteria_set=make_criteria_set(),
        criterion_csv_path=criterion_csv, document_csv_path=document_csv,
        blind_manifest_path=blind_path, sample_manifest_path=sample_path,
        output_dir=tmp_path / "out", version_label="gold_set_v1",
    )
    loaded = load_frozen_gold_set(tmp_path / "out" / "gold_set_v1.json", expected_hash=frozen.content_hash())
    assert loaded.content_hash() == frozen.content_hash()


# ---------------------------------------------------------------------------
# Worksheet-pass sealing and source-manifest provenance
# (codex_code_review_ade9f48.md finding 2)
# ---------------------------------------------------------------------------


def test_seal_worksheet_pass_accepts_a_complete_primary_pass(tmp_path):
    blind_path, sample_path, _ = make_manifests(tmp_path, document_ids=["d1", "d2"])
    criterion_csv, document_csv = _write_two_criterion_two_document_csvs(tmp_path)
    seal = seal_worksheet_pass(
        pass_name="primary", protocol_version="gold_set_protocol_v9", criteria_set=make_criteria_set(),
        blind_manifest_path=blind_path, sample_manifest_path=sample_path,
        criterion_csv_path=criterion_csv, document_csv_path=document_csv,
        expected_document_ids={"d1", "d2"}, output_dir=tmp_path / "seal", version_label="primary_seal_v1",
    )
    assert seal.pass_name == "primary"
    assert seal.document_ids == ("d1", "d2")
    assert seal.criterion_csv_sha256 == hash_pdf_file(criterion_csv)
    assert seal.document_csv_sha256 == hash_pdf_file(document_csv)


def test_seal_worksheet_pass_rejects_a_mixed_round_file(tmp_path):
    """A worksheet with even one row from a different pass is exactly the
    bypass this workflow exists to prevent: appending second-pass answers
    into the already-answered primary CSV."""
    blind_path, sample_path, _ = make_manifests(tmp_path, document_ids=["d1", "d2"])
    criterion_csv, document_csv = _write_two_criterion_two_document_csvs(tmp_path)
    with open(criterion_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.append(_base_criterion_row(document_id="d1", criterion_id="population", annotation_round="second_pass", annotator_id="XY"))
    _write_csv(criterion_csv, CRITERION_FIELDS, rows)
    with pytest.raises(ValueError):
        seal_worksheet_pass(
            pass_name="primary", protocol_version="gold_set_protocol_v9", criteria_set=make_criteria_set(),
            blind_manifest_path=blind_path, sample_manifest_path=sample_path,
            criterion_csv_path=criterion_csv, document_csv_path=document_csv,
            expected_document_ids={"d1", "d2"}, output_dir=tmp_path / "seal", version_label="primary_seal_v1",
        )


def test_seal_worksheet_pass_rejects_incomplete_document_coverage(tmp_path):
    blind_path, sample_path, _ = make_manifests(tmp_path, document_ids=["d1", "d2"])
    criterion_csv, document_csv = _write_two_criterion_two_document_csvs(tmp_path)
    with pytest.raises(ValueError):
        seal_worksheet_pass(
            pass_name="primary", protocol_version="gold_set_protocol_v9", criteria_set=make_criteria_set(),
            blind_manifest_path=blind_path, sample_manifest_path=sample_path,
            criterion_csv_path=criterion_csv, document_csv_path=document_csv,
            expected_document_ids={"d1", "d2", "d3"},  # d3 never annotated
            output_dir=tmp_path / "seal", version_label="primary_seal_v1",
        )


def test_seal_worksheet_pass_round_trips_via_load(tmp_path):
    blind_path, sample_path, _ = make_manifests(tmp_path, document_ids=["d1", "d2"])
    criterion_csv, document_csv = _write_two_criterion_two_document_csvs(tmp_path)
    seal = seal_worksheet_pass(
        pass_name="primary", protocol_version="gold_set_protocol_v9", criteria_set=make_criteria_set(),
        blind_manifest_path=blind_path, sample_manifest_path=sample_path,
        criterion_csv_path=criterion_csv, document_csv_path=document_csv,
        expected_document_ids={"d1", "d2"}, output_dir=tmp_path / "seal", version_label="primary_seal_v1",
    )
    loaded = load_frozen_worksheet_pass_seal(tmp_path / "seal" / "primary_seal_v1.json", expected_hash=seal.content_hash())
    assert loaded.content_hash() == seal.content_hash()


def test_seal_worksheet_pass_refuses_to_reseal(tmp_path):
    blind_path, sample_path, _ = make_manifests(tmp_path, document_ids=["d1", "d2"])
    criterion_csv, document_csv = _write_two_criterion_two_document_csvs(tmp_path)
    seal_worksheet_pass(
        pass_name="primary", protocol_version="gold_set_protocol_v9", criteria_set=make_criteria_set(),
        blind_manifest_path=blind_path, sample_manifest_path=sample_path,
        criterion_csv_path=criterion_csv, document_csv_path=document_csv,
        expected_document_ids={"d1", "d2"}, output_dir=tmp_path / "seal", version_label="primary_seal_v1",
    )
    with pytest.raises(FileExistsError):
        seal_worksheet_pass(
            pass_name="primary", protocol_version="gold_set_protocol_v9", criteria_set=make_criteria_set(),
            blind_manifest_path=blind_path, sample_manifest_path=sample_path,
            criterion_csv_path=criterion_csv, document_csv_path=document_csv,
            expected_document_ids={"d1", "d2"}, output_dir=tmp_path / "seal", version_label="primary_seal_v1",
        )


def test_seal_worksheet_pass_rejects_an_unsupported_protocol_version(tmp_path):
    """codex_code_review_73cf5d9.md finding 1: protocol_version is now
    rejected at SEAL time, not only later at freeze time."""
    blind_path, sample_path, _ = make_manifests(tmp_path, document_ids=["d1", "d2"])
    criterion_csv, document_csv = _write_two_criterion_two_document_csvs(tmp_path)
    with pytest.raises(ValueError):
        seal_worksheet_pass(
            pass_name="primary", protocol_version="wrong-protocol", criteria_set=make_criteria_set(),
            blind_manifest_path=blind_path, sample_manifest_path=sample_path,
            criterion_csv_path=criterion_csv, document_csv_path=document_csv,
            expected_document_ids={"d1", "d2"}, output_dir=tmp_path / "seal", version_label="primary_seal_v1",
        )


# ---------------------------------------------------------------------------
# verify_worksheet_pass_seal: every identity field enforced, not just pass
# name and CSV hashes (codex_code_review_73cf5d9.md finding 1)
# ---------------------------------------------------------------------------


def _sealed_pass(tmp_path, *, document_ids=("d1", "d2")):
    blind_path, sample_path, _ = make_manifests(tmp_path, document_ids=list(document_ids))
    criterion_csv, document_csv = _write_two_criterion_two_document_csvs(tmp_path)
    seal = seal_worksheet_pass(
        pass_name="primary", protocol_version="gold_set_protocol_v9", criteria_set=make_criteria_set(),
        blind_manifest_path=blind_path, sample_manifest_path=sample_path,
        criterion_csv_path=criterion_csv, document_csv_path=document_csv,
        expected_document_ids=set(document_ids), output_dir=tmp_path / "seal", version_label="primary_seal_v1",
    )
    return seal, criterion_csv, document_csv, blind_path, sample_path


def _valid_verify_kwargs(tmp_path):
    seal, criterion_csv, document_csv, blind_path, sample_path = _sealed_pass(tmp_path)
    return dict(
        seal=seal, expected_pass_name="primary", expected_protocol_version="gold_set_protocol_v9",
        expected_criteria_set_hash=compute_criteria_set_hash(make_criteria_set()),
        expected_document_ids={"d1", "d2"}, expected_blind_manifest_hash=hash_pdf_file(blind_path),
        expected_sample_manifest_hash=hash_pdf_file(sample_path),
        criterion_csv_path=criterion_csv, document_csv_path=document_csv,
    )


def test_verify_worksheet_pass_seal_accepts_a_fully_matching_seal(tmp_path):
    verify_worksheet_pass_seal(**_valid_verify_kwargs(tmp_path))  # must not raise


def test_verify_worksheet_pass_seal_rejects_wrong_protocol_version(tmp_path):
    kwargs = _valid_verify_kwargs(tmp_path)
    kwargs["expected_protocol_version"] = "some-other-protocol"
    with pytest.raises(ValueError):
        verify_worksheet_pass_seal(**kwargs)


def test_verify_worksheet_pass_seal_rejects_wrong_criteria_set_hash(tmp_path):
    kwargs = _valid_verify_kwargs(tmp_path)
    kwargs["expected_criteria_set_hash"] = "f" * 64
    with pytest.raises(ValueError):
        verify_worksheet_pass_seal(**kwargs)


def test_verify_worksheet_pass_seal_rejects_an_overbroad_document_id_set(tmp_path):
    """Exact repro: a second-pass seal covering MORE documents than the
    actual dual-review subset (a superset, not an exact match) must be
    rejected, not accepted because it merely CONTAINS the right IDs."""
    kwargs = _valid_verify_kwargs(tmp_path)
    kwargs["expected_document_ids"] = {"d1"}  # seal actually covers {d1, d2} -- a superset
    with pytest.raises(ValueError):
        verify_worksheet_pass_seal(**kwargs)


def test_verify_worksheet_pass_seal_rejects_a_stale_blind_manifest_hash(tmp_path):
    kwargs = _valid_verify_kwargs(tmp_path)
    kwargs["expected_blind_manifest_hash"] = "0" * 64
    with pytest.raises(ValueError):
        verify_worksheet_pass_seal(**kwargs)


def test_verify_worksheet_pass_seal_rejects_a_stale_sample_manifest_hash(tmp_path):
    kwargs = _valid_verify_kwargs(tmp_path)
    kwargs["expected_sample_manifest_hash"] = "0" * 64
    with pytest.raises(ValueError):
        verify_worksheet_pass_seal(**kwargs)


def test_freeze_gold_set_rejects_a_second_pass_seal_covering_more_than_the_dual_review_subset(tmp_path):
    """End-to-end repro through freeze_gold_set: a second-pass seal
    genuinely sealed against {d1, d2} is used where the sample's
    dual_review_subset is only {d1}."""
    blind_path, sample_path, _ = make_manifests(tmp_path, document_ids=["d1", "d2"], dual_review_ids=["d1"])
    criterion_csv, document_csv = _write_two_criterion_two_document_csvs(tmp_path)
    with open(criterion_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # Give BOTH d1 and d2 a second_pass row, then seal the second pass
    # against {d1, d2} -- broader than the real dual_review_subset={d1}.
    rows.append(_base_criterion_row(document_id="d1", criterion_id="population", annotation_round="second_pass", annotator_id="XY"))
    rows.append(_base_criterion_row(document_id="d1", criterion_id="design", annotation_round="second_pass", annotator_id="XY", supporting_pages="[2]", supporting_passages='["RCT design"]'))
    rows.append(_base_criterion_row(document_id="d2", criterion_id="population", annotation_round="second_pass", annotator_id="XY"))
    rows.append(_base_criterion_row(document_id="d2", criterion_id="design", annotation_round="second_pass", annotator_id="XY", supporting_pages="[2]", supporting_passages='["RCT design"]'))
    _write_csv(criterion_csv, CRITERION_FIELDS, rows)
    with open(document_csv, newline="", encoding="utf-8") as f:
        doc_rows = list(csv.DictReader(f))
    doc_rows.append(_base_document_row(document_id="d1", annotation_round="second_pass", annotator_id="XY"))
    doc_rows.append(_base_document_row(document_id="d2", annotation_round="second_pass", annotator_id="XY"))
    _write_csv(document_csv, DOCUMENT_FIELDS, doc_rows)

    seal_dir = tmp_path / "_seal"
    seal_dir.mkdir()
    with open(criterion_csv, newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    by_round = _split_rows_by_round(all_rows)
    with open(document_csv, newline="", encoding="utf-8") as f:
        all_doc_rows = list(csv.DictReader(f))
    doc_by_round = _split_rows_by_round(all_doc_rows)

    primary_criterion_path = seal_dir / "primary_criteria.csv"
    primary_document_path = seal_dir / "primary_documents.csv"
    _write_csv(primary_criterion_path, CRITERION_FIELDS, by_round["primary"])
    _write_csv(primary_document_path, DOCUMENT_FIELDS, doc_by_round["primary"])
    primary_seal = seal_worksheet_pass(
        pass_name="primary", protocol_version="gold_set_protocol_v9", criteria_set=make_criteria_set(),
        blind_manifest_path=blind_path, sample_manifest_path=sample_path,
        criterion_csv_path=primary_criterion_path, document_csv_path=primary_document_path,
        expected_document_ids={"d1", "d2"}, output_dir=seal_dir, version_label="primary_seal",
    )

    second_pass_criterion_path = seal_dir / "second_pass_criteria.csv"
    second_pass_document_path = seal_dir / "second_pass_documents.csv"
    _write_csv(second_pass_criterion_path, CRITERION_FIELDS, by_round["second_pass"])
    _write_csv(second_pass_document_path, DOCUMENT_FIELDS, doc_by_round["second_pass"])
    second_pass_seal = seal_worksheet_pass(
        pass_name="second_pass", protocol_version="gold_set_protocol_v9", criteria_set=make_criteria_set(),
        blind_manifest_path=blind_path, sample_manifest_path=sample_path,
        criterion_csv_path=second_pass_criterion_path, document_csv_path=second_pass_document_path,
        expected_document_ids={"d1", "d2"},  # OVERBROAD: real dual_review_subset is only {d1}
        output_dir=seal_dir, version_label="second_pass_seal",
    )

    adjudication_criterion_path = seal_dir / "adjudication_criteria.csv"
    adjudication_document_path = seal_dir / "adjudication_documents.csv"
    _write_csv(adjudication_criterion_path, CRITERION_FIELDS, by_round["adjudicated"])
    _write_csv(adjudication_document_path, DOCUMENT_FIELDS, doc_by_round["adjudicated"])

    provenance_path = seal_dir / "source_provenance.json"
    provenance_path.write_text(json.dumps({
        "protocol_version": "gold_set_protocol_v9", "criteria_set_hash": compute_criteria_set_hash(make_criteria_set()),
        "blind_manifest_hash": hash_pdf_file(blind_path), "sample_manifest_hash": hash_pdf_file(sample_path),
        "primary": {
            "role": "primary", "seal_path": str(seal_dir / "primary_seal.json"), "seal_content_hash": primary_seal.content_hash(),
            "criterion_csv_path": str(primary_criterion_path), "criterion_csv_sha256": hash_pdf_file(primary_criterion_path),
            "document_csv_path": str(primary_document_path), "document_csv_sha256": hash_pdf_file(primary_document_path),
        },
        "second_pass": {
            "role": "second_pass", "seal_path": str(seal_dir / "second_pass_seal.json"), "seal_content_hash": second_pass_seal.content_hash(),
            "criterion_csv_path": str(second_pass_criterion_path), "criterion_csv_sha256": hash_pdf_file(second_pass_criterion_path),
            "document_csv_path": str(second_pass_document_path), "document_csv_sha256": hash_pdf_file(second_pass_document_path),
        },
        "adjudication": {
            "role": "adjudication", "seal_path": None, "seal_content_hash": None,
            "criterion_csv_path": str(adjudication_criterion_path), "criterion_csv_sha256": hash_pdf_file(adjudication_criterion_path),
            "document_csv_path": str(adjudication_document_path), "document_csv_sha256": hash_pdf_file(adjudication_document_path),
        },
        "merged_criterion_csv_sha256": hash_pdf_file(criterion_csv), "merged_document_csv_sha256": hash_pdf_file(document_csv),
        "merged_at": "2026-08-16T00:00:00+00:00",
    }), encoding="utf-8")

    with pytest.raises(ValueError):
        freeze_gold_set(
            protocol_version="gold_set_protocol_v9", criteria_set=make_criteria_set(),
            criterion_csv_path=criterion_csv, document_csv_path=document_csv,
            blind_manifest_path=blind_path, sample_manifest_path=sample_path,
            output_dir=tmp_path / "out", version_label="gold_set_v1", source_provenance_path=provenance_path,
        )


# ---------------------------------------------------------------------------
# freeze_gold_set must reconstruct and compare the RAW dual-review audit,
# not just resolved outcomes (codex_code_review_73cf5d9.md finding 2)
# ---------------------------------------------------------------------------


def test_freeze_gold_set_rejects_forged_primary_content_preserving_the_resolved_outcome(tmp_path):
    """Exact repro: sealed primary said 'satisfied', sealed second pass
    said 'violated', sealed-source adjudication resolved to 'satisfied'.
    The merged CSV's raw primary passage is then forged while its
    adjudicated row is left unchanged -- both paths still resolve to
    'satisfied' (adjudication wins over disagreeing raw passes either
    way), but the raw audit trail itself now disagrees and must be
    rejected."""
    blind_path, sample_path, _ = make_manifests(tmp_path, document_ids=["d1", "d2"], dual_review_ids=["d1"])
    criterion_csv, document_csv = _write_two_criterion_two_document_csvs(tmp_path)
    with open(criterion_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # d1/population: primary=satisfied, second_pass=violated (disagreement),
    # adjudicated=satisfied (resolves the disagreement).
    rows.append(_base_criterion_row(
        document_id="d1", criterion_id="population", annotation_round="second_pass", annotator_id="XY",
        gold_status="violated", supporting_pages="[]", supporting_passages="[]",
        contradicting_pages="[1]", contradicting_passages='["actually excluded"]',
    ))
    rows.append(_base_criterion_row(document_id="d1", criterion_id="population", annotation_round="adjudicated", annotator_id="ZZ"))
    rows.append(_base_criterion_row(document_id="d1", criterion_id="design", annotation_round="second_pass", annotator_id="XY", supporting_pages="[2]", supporting_passages='["RCT design"]'))
    _write_csv(criterion_csv, CRITERION_FIELDS, rows)
    with open(document_csv, newline="", encoding="utf-8") as f:
        doc_rows = list(csv.DictReader(f))
    doc_rows.append(_base_document_row(document_id="d1", annotation_round="second_pass", annotator_id="XY"))
    _write_csv(document_csv, DOCUMENT_FIELDS, doc_rows)

    provenance_path = _build_source_provenance(
        tmp_path, protocol_version="gold_set_protocol_v9", criteria_set=make_criteria_set(),
        criterion_csv_path=criterion_csv, document_csv_path=document_csv,
        blind_manifest_path=blind_path, sample_manifest_path=sample_path,
    )

    # Now forge the MERGED file's raw primary passage -- leave the
    # adjudicated row (and hence the resolved outcome) unchanged.
    with open(criterion_csv, newline="", encoding="utf-8") as f:
        merged_rows = list(csv.DictReader(f))
    for row in merged_rows:
        if row["document_id"] == "d1" and row["criterion_id"] == "population" and row["annotation_round"] == "primary":
            row["supporting_passages"] = '["FORGED RAW AUDIT"]'
    _write_csv(criterion_csv, CRITERION_FIELDS, merged_rows)

    with pytest.raises(ValueError):
        freeze_gold_set(
            protocol_version="gold_set_protocol_v9", criteria_set=make_criteria_set(),
            criterion_csv_path=criterion_csv, document_csv_path=document_csv,
            blind_manifest_path=blind_path, sample_manifest_path=sample_path,
            output_dir=tmp_path / "out", version_label="gold_set_v1", source_provenance_path=provenance_path,
        )


def test_freeze_gold_set_rejects_forged_second_pass_content_preserving_the_resolved_outcome(tmp_path):
    """Same as the primary-forgery repro, but forging the SECOND_PASS raw
    row instead."""
    blind_path, sample_path, _ = make_manifests(tmp_path, document_ids=["d1", "d2"], dual_review_ids=["d1"])
    criterion_csv, document_csv = _write_two_criterion_two_document_csvs(tmp_path)
    with open(criterion_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.append(_base_criterion_row(
        document_id="d1", criterion_id="population", annotation_round="second_pass", annotator_id="XY",
        gold_status="violated", supporting_pages="[]", supporting_passages="[]",
        contradicting_pages="[1]", contradicting_passages='["actually excluded"]',
    ))
    rows.append(_base_criterion_row(document_id="d1", criterion_id="population", annotation_round="adjudicated", annotator_id="ZZ"))
    rows.append(_base_criterion_row(document_id="d1", criterion_id="design", annotation_round="second_pass", annotator_id="XY", supporting_pages="[2]", supporting_passages='["RCT design"]'))
    _write_csv(criterion_csv, CRITERION_FIELDS, rows)
    with open(document_csv, newline="", encoding="utf-8") as f:
        doc_rows = list(csv.DictReader(f))
    doc_rows.append(_base_document_row(document_id="d1", annotation_round="second_pass", annotator_id="XY"))
    _write_csv(document_csv, DOCUMENT_FIELDS, doc_rows)

    provenance_path = _build_source_provenance(
        tmp_path, protocol_version="gold_set_protocol_v9", criteria_set=make_criteria_set(),
        criterion_csv_path=criterion_csv, document_csv_path=document_csv,
        blind_manifest_path=blind_path, sample_manifest_path=sample_path,
    )

    with open(criterion_csv, newline="", encoding="utf-8") as f:
        merged_rows = list(csv.DictReader(f))
    for row in merged_rows:
        if row["document_id"] == "d1" and row["criterion_id"] == "population" and row["annotation_round"] == "second_pass":
            row["contradicting_passages"] = '["FORGED SECOND PASS"]'
    _write_csv(criterion_csv, CRITERION_FIELDS, merged_rows)

    with pytest.raises(ValueError):
        freeze_gold_set(
            protocol_version="gold_set_protocol_v9", criteria_set=make_criteria_set(),
            criterion_csv_path=criterion_csv, document_csv_path=document_csv,
            blind_manifest_path=blind_path, sample_manifest_path=sample_path,
            output_dir=tmp_path / "out", version_label="gold_set_v1", source_provenance_path=provenance_path,
        )


# ---------------------------------------------------------------------------
# WorksheetSourceRecord: seal fields required for sealed roles, forbidden
# for adjudication (codex_code_review_73cf5d9.md finding 3)
# ---------------------------------------------------------------------------


def test_worksheet_source_record_requires_seal_fields_for_primary_role():
    with pytest.raises(Exception):
        WorksheetSourceRecord(
            role="primary", seal_path=None, seal_content_hash=None,
            criterion_csv_path="x.csv", criterion_csv_sha256="a" * 64,
            document_csv_path="y.csv", document_csv_sha256="b" * 64,
        )


def test_worksheet_source_record_forbids_seal_fields_for_adjudication_role():
    with pytest.raises(Exception):
        WorksheetSourceRecord(
            role="adjudication", seal_path="some_seal.json", seal_content_hash="c" * 64,
            criterion_csv_path="x.csv", criterion_csv_sha256="a" * 64,
            document_csv_path="y.csv", document_csv_sha256="b" * 64,
        )


def test_criterion_rows_agree_detects_agreement_and_disagreement():
    a = make_criterion_annotation(gold_status="satisfied")
    b = make_criterion_annotation(gold_status="satisfied")
    c = make_criterion_annotation(gold_status="violated", supporting_pages=(), supporting_passages=(), contradicting_pages=(1,), contradicting_passages=("x",))
    assert criterion_rows_agree(a, b) is True
    assert criterion_rows_agree(a, c) is False


def test_document_rows_agree_detects_agreement_and_disagreement():
    a = make_document_annotation(gold_final_decision="include")
    b = make_document_annotation(gold_final_decision="include")
    c = make_document_annotation(gold_final_decision="exclude", gold_primary_exclusion_reason_code="wrong_population")
    assert document_rows_agree(a, b) is True
    assert document_rows_agree(a, c) is False


def test_freeze_gold_set_embeds_verified_source_provenance(tmp_path):
    blind_path, sample_path, _ = make_manifests(tmp_path, document_ids=["d1", "d2"])
    criterion_csv, document_csv = _write_two_criterion_two_document_csvs(tmp_path)
    gold_set = _freeze(
        protocol_version="gold_set_protocol_v9", criteria_set=make_criteria_set(),
        criterion_csv_path=criterion_csv, document_csv_path=document_csv,
        blind_manifest_path=blind_path, sample_manifest_path=sample_path,
        output_dir=tmp_path / "out", version_label="gold_set_v1",
    )
    assert gold_set.source_provenance.protocol_version == "gold_set_protocol_v9"
    assert gold_set.source_provenance.second_pass is None  # no dual review in this sample
    assert gold_set.source_provenance.primary.seal_content_hash is not None


def test_freeze_gold_set_requires_source_provenance_path(tmp_path):
    """codex_code_review_93bf42d.md finding 2: an absent or empty source
    manifest previously froze successfully. It is now a required
    parameter, not an optional one."""
    blind_path, sample_path, _ = make_manifests(tmp_path, document_ids=["d1", "d2"])
    criterion_csv, document_csv = _write_two_criterion_two_document_csvs(tmp_path)
    with pytest.raises(TypeError):
        freeze_gold_set(
            protocol_version="gold_set_protocol_v9", criteria_set=make_criteria_set(),
            criterion_csv_path=criterion_csv, document_csv_path=document_csv,
            blind_manifest_path=blind_path, sample_manifest_path=sample_path,
            output_dir=tmp_path / "out", version_label="gold_set_v1",
        )


def test_freeze_gold_set_rejects_an_empty_source_provenance_file(tmp_path):
    """Exact repro: an empty `{}` manifest previously froze successfully."""
    blind_path, sample_path, _ = make_manifests(tmp_path, document_ids=["d1", "d2"])
    criterion_csv, document_csv = _write_two_criterion_two_document_csvs(tmp_path)
    empty_provenance_path = tmp_path / "empty_provenance.json"
    empty_provenance_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        freeze_gold_set(
            protocol_version="gold_set_protocol_v9", criteria_set=make_criteria_set(),
            criterion_csv_path=criterion_csv, document_csv_path=document_csv,
            blind_manifest_path=blind_path, sample_manifest_path=sample_path,
            output_dir=tmp_path / "out", version_label="gold_set_v1", source_provenance_path=empty_provenance_path,
        )


def test_freeze_gold_set_rejects_a_provenance_naming_an_unrelated_file(tmp_path):
    """Exact repro: a manifest whose only listed 'source' was an
    arbitrary unrelated file, unconnected to the actual merged CSVs,
    previously froze successfully -- provenance must PRODUCE the frozen
    rows, not merely name some file that happens to still match a hash."""
    blind_path, sample_path, _ = make_manifests(tmp_path, document_ids=["d1", "d2"])
    criterion_csv, document_csv = _write_two_criterion_two_document_csvs(tmp_path)

    # A validly-SEALED but completely UNRELATED primary pass (different tmp_path, different content).
    other_tmp = tmp_path / "unrelated"
    other_tmp.mkdir()
    other_blind_path, other_sample_path, _ = make_manifests(other_tmp, document_ids=["dX", "dY"])
    other_criterion_csv, other_document_csv = _write_two_criterion_two_document_csvs(other_tmp)
    with open(other_criterion_csv, newline="", encoding="utf-8") as f:
        other_rows = list(csv.DictReader(f))
    for row in other_rows:
        row["document_id"] = {"d1": "dX", "d2": "dY"}[row["document_id"]]
    _write_csv(other_criterion_csv, CRITERION_FIELDS, other_rows)
    with open(other_document_csv, newline="", encoding="utf-8") as f:
        other_doc_rows = list(csv.DictReader(f))
    for row in other_doc_rows:
        row["document_id"] = {"d1": "dX", "d2": "dY"}[row["document_id"]]
    _write_csv(other_document_csv, DOCUMENT_FIELDS, other_doc_rows)
    unrelated_seal = seal_worksheet_pass(
        pass_name="primary", protocol_version="gold_set_protocol_v9", criteria_set=make_criteria_set(),
        blind_manifest_path=other_blind_path, sample_manifest_path=other_sample_path,
        criterion_csv_path=other_criterion_csv, document_csv_path=other_document_csv,
        expected_document_ids={"dX", "dY"}, output_dir=other_tmp, version_label="primary_seal",
    )
    assert unrelated_seal.document_ids == ("dX", "dY")  # sanity: genuinely unrelated document identities

    provenance_path = tmp_path / "unrelated_provenance.json"
    provenance_path.write_text(json.dumps({
        "protocol_version": "gold_set_protocol_v9", "criteria_set_hash": compute_criteria_set_hash(make_criteria_set()),
        "blind_manifest_hash": hash_pdf_file(blind_path), "sample_manifest_hash": hash_pdf_file(sample_path),
        "primary": {
            "role": "primary", "seal_path": str(other_tmp / "primary_seal.json"), "seal_content_hash": unrelated_seal.content_hash(),
            "criterion_csv_path": str(other_criterion_csv), "criterion_csv_sha256": hash_pdf_file(other_criterion_csv),
            "document_csv_path": str(other_document_csv), "document_csv_sha256": hash_pdf_file(other_document_csv),
        },
        "merged_criterion_csv_sha256": hash_pdf_file(criterion_csv), "merged_document_csv_sha256": hash_pdf_file(document_csv),
        "merged_at": "2026-08-16T00:00:00+00:00",
    }), encoding="utf-8")

    with pytest.raises(ValueError):
        freeze_gold_set(
            protocol_version="gold_set_protocol_v9", criteria_set=make_criteria_set(),
            criterion_csv_path=criterion_csv, document_csv_path=document_csv,
            blind_manifest_path=blind_path, sample_manifest_path=sample_path,
            output_dir=tmp_path / "out", version_label="gold_set_v1", source_provenance_path=provenance_path,
        )


def test_freeze_gold_set_rejects_a_post_seal_edit_to_the_sealed_csv(tmp_path):
    """codex_code_review_93bf42d.md finding 3: a sealed CSV edited after
    sealing must be caught at freeze time even if an upstream step forgot
    to reverify it."""
    blind_path, sample_path, _ = make_manifests(tmp_path, document_ids=["d1", "d2"])
    criterion_csv, document_csv = _write_two_criterion_two_document_csvs(tmp_path)
    provenance_path = _build_source_provenance(
        tmp_path, protocol_version="gold_set_protocol_v9", criteria_set=make_criteria_set(),
        criterion_csv_path=criterion_csv, document_csv_path=document_csv,
        blind_manifest_path=blind_path, sample_manifest_path=sample_path,
    )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    # Edit the SEALED primary CSV snapshot after it was sealed.
    sealed_path = Path(provenance["primary"]["criterion_csv_path"])
    with open(sealed_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows[0]["gold_status"] = "violated"
    _write_csv(sealed_path, CRITERION_FIELDS, rows)

    with pytest.raises(ValueError):
        freeze_gold_set(
            protocol_version="gold_set_protocol_v9", criteria_set=make_criteria_set(),
            criterion_csv_path=criterion_csv, document_csv_path=document_csv,
            blind_manifest_path=blind_path, sample_manifest_path=sample_path,
            output_dir=tmp_path / "out", version_label="gold_set_v1", source_provenance_path=provenance_path,
        )


def test_freeze_gold_set_requires_second_pass_when_dual_review_applies(tmp_path):
    """codex_code_review_93bf42d.md finding 3: a missing second pass must
    never be silently treated as optional when the sample requires dual
    review -- even when the merged CSV itself is genuinely well-formed
    (a real second_pass row is present there), a PROVENANCE record that
    omits second_pass_* must still be rejected."""
    blind_path, sample_path, _ = make_manifests(tmp_path, document_ids=["d1", "d2"], dual_review_ids=["d1"])
    criterion_csv, document_csv = _write_two_criterion_two_document_csvs(tmp_path)
    with open(criterion_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.append(_base_criterion_row(document_id="d1", criterion_id="population", annotation_round="second_pass", annotator_id="XY"))
    rows.append(_base_criterion_row(document_id="d1", criterion_id="design", annotation_round="second_pass", annotator_id="XY", supporting_pages="[2]", supporting_passages='["RCT design"]'))
    _write_csv(criterion_csv, CRITERION_FIELDS, rows)
    with open(document_csv, newline="", encoding="utf-8") as f:
        doc_rows = list(csv.DictReader(f))
    doc_rows.append(_base_document_row(document_id="d1", annotation_round="second_pass", annotator_id="XY"))
    _write_csv(document_csv, DOCUMENT_FIELDS, doc_rows)

    provenance_path = _build_source_provenance(
        tmp_path, protocol_version="gold_set_protocol_v9", criteria_set=make_criteria_set(),
        criterion_csv_path=criterion_csv, document_csv_path=document_csv,
        blind_manifest_path=blind_path, sample_manifest_path=sample_path,
    )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    for key in ("second_pass", "adjudication"):
        provenance.pop(key, None)
    stripped_provenance_path = tmp_path / "stripped_provenance.json"
    stripped_provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError):
        freeze_gold_set(
            protocol_version="gold_set_protocol_v9", criteria_set=make_criteria_set(),
            criterion_csv_path=criterion_csv, document_csv_path=document_csv,
            blind_manifest_path=blind_path, sample_manifest_path=sample_path,
            output_dir=tmp_path / "out", version_label="gold_set_v1", source_provenance_path=stripped_provenance_path,
        )


# ---------------------------------------------------------------------------
# RetrievalOracleArtifact / RetrievalOracleJudgmentSet
# ---------------------------------------------------------------------------


def test_freeze_and_load_retrieval_oracle_artifact(tmp_path):
    artifact = freeze_retrieval_oracle_artifact(
        parser_name="docling", parser_version="2.61.2", parser_options={"do_ocr": True},
        retriever_name="lexical_bm25", retriever_config_version="1", retriever_options={},
        criteria_set=make_criteria_set(), document_sha256={"d1": "a" * 64},
        evidence_by_key={make_evidence_key("d1", "population"): [RetrievalOracleSpan(chunk_id="c-1", page=1, excerpt="adults")]},
        output_dir=tmp_path, version_label="retrieval_artifact_v1",
    )
    loaded = load_frozen_retrieval_oracle_artifact(tmp_path / "retrieval_artifact_v1.json", expected_hash=artifact.content_hash())
    assert loaded.content_hash() == artifact.content_hash()


def test_retrieval_oracle_judgment_set_binds_to_gold_set_and_artifact_hashes(tmp_path):
    blind_path, sample_path, _ = make_manifests(tmp_path, document_ids=["d1", "d2"])
    criterion_csv, document_csv = _write_two_criterion_two_document_csvs(tmp_path)
    gold_set = _freeze(
        protocol_version="gold_set_protocol_v9", criteria_set=make_criteria_set(),
        criterion_csv_path=criterion_csv, document_csv_path=document_csv,
        blind_manifest_path=blind_path, sample_manifest_path=sample_path,
        output_dir=tmp_path / "out", version_label="gold_set_v1",
    )
    artifact = freeze_retrieval_oracle_artifact(
        parser_name="docling", parser_version="2.61.2", parser_options={},
        retriever_name="lexical_bm25", retriever_config_version="1", retriever_options={},
        criteria_set=make_criteria_set(), document_sha256=dict(gold_set.document_pdf_sha256),
        evidence_by_key={
            make_evidence_key("d1", "population"): [], make_evidence_key("d1", "design"): [],
            make_evidence_key("d2", "population"): [], make_evidence_key("d2", "design"): [],
        },
        output_dir=tmp_path / "out", version_label="retrieval_artifact_v1",
    )
    judgments = [
        RetrievalSufficiencyJudgment(document_id=doc_id, criterion_id=crit_id, retrieval_sufficient=True, annotator_id="RR")
        for doc_id in ("d1", "d2") for crit_id in ("population", "design")
    ]
    judgment_set = freeze_retrieval_oracle_judgment_set(
        gold_set=gold_set, retrieval_artifact=artifact, judgments=judgments,
        output_dir=tmp_path / "out", version_label="judgment_set_v1",
    )
    loaded = load_frozen_retrieval_oracle_judgment_set(tmp_path / "out" / "judgment_set_v1.json")
    assert loaded.content_hash() == judgment_set.content_hash()


def test_retrieval_oracle_judgment_set_rejects_duplicate_keys():
    with pytest.raises(Exception):
        RetrievalOracleJudgmentSet(
            gold_set_hash="a" * 64, retrieval_artifact_hash="b" * 64,
            judgments=(
                RetrievalSufficiencyJudgment(document_id="d1", criterion_id="population", retrieval_sufficient=True, annotator_id="RR"),
                RetrievalSufficiencyJudgment(document_id="d1", criterion_id="population", retrieval_sufficient=False, annotator_id="RR"),
            ),
            frozen_at="2026-08-16T00:00:00+00:00",
        )
