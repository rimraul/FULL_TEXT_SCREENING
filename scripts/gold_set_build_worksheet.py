"""Builds the gold-set annotation worksheets (gold_set_protocol_v1.md §4):
runs the REAL DoclingDocumentParser + LexicalEvidenceRetriever (never the
adjudicator -- see the protocol's blinding note) over the 20 sampled
documents in gold_set/sample_v1.json, and emits two CSVs for the human
annotator to fill in:

  gold_set/worksheet_criteria.csv    -- one row per document x criterion (60 rows)
  gold_set/worksheet_documents.csv   -- one row per document (20 rows)

Context columns are pre-filled (document_title, page_count, ocr_used,
criterion_text, retrieved_evidence, dual_review_subset); annotation
columns are left blank for the human to fill in Excel/Sheets.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:/projects/FT_SCREENING/src")

from ft_screening.document import DoclingDocumentParser
from ft_screening.retrieval import LexicalEvidenceRetriever
from ft_screening.schema import CriteriaSet, CriterionSpec

MANIFEST_PATH = Path(r"C:/projects/FT_SCREENING/gold_set/sample_v1.json")
OUTPUT_DIR = Path(r"C:/projects/FT_SCREENING/gold_set")

CRITERIA_SET = CriteriaSet(
    criteria_set_id="walking-skeleton-drr-governance",
    version="1",
    criteria=[
        CriterionSpec(
            criterion_id="I1_review_type", version="1",
            eligibility_requirement=(
                "The document is a review-type study (systematic review, scoping review, or "
                "other literature review) that synthesizes existing published literature, rather "
                "than reporting a single primary empirical study or a purely conceptual framework "
                "paper with no literature synthesis."
            ),
            missing_evidence_policy="review", exclusion_reason_code="not_a_review",
            exclusion_reason_label="Not a review-type synthesis of existing literature", priority=1,
        ),
        CriterionSpec(
            criterion_id="I2_drr_hazard_framing", version="1",
            eligibility_requirement=(
                "The document substantively addresses disaster risk reduction (DRR) or resilience "
                "to hazards, with an explicit hazard/disaster framing -- general risk, climate "
                "adaptation, or sustainability content without an explicit hazard/disaster context "
                "does not satisfy this criterion."
            ),
            missing_evidence_policy="review", exclusion_reason_code="off_topic_hazard",
            exclusion_reason_label="No explicit disaster/hazard framing", priority=2,
        ),
        CriterionSpec(
            criterion_id="I3_governance_dimension", version="1",
            eligibility_requirement=(
                "The document substantively analyses at least one of the following governance "
                "dimensions in a DRR/resilience context: institutional capacity, decentralization, "
                "multilevel governance, partnerships, civil society, capacity building, community "
                "participation, education/training, financing, legal/regulatory frameworks, "
                "information systems, or urban planning. Passing mention of government actors "
                "without substantive analysis of one of these dimensions does not satisfy this "
                "criterion."
            ),
            missing_evidence_policy="review", exclusion_reason_code="no_governance",
            exclusion_reason_label="No substantive governance-dimension analysis", priority=3,
        ),
    ],
)

CRITERION_WORKSHEET_FIELDS = [
    "document_id", "document_title", "stratum", "dual_review_subset", "page_count", "ocr_used",
    "criterion_id", "criterion_text", "retrieved_evidence",
    "gold_status", "supporting_pages", "supporting_passages", "contradicting_pages",
    "contradicting_passages", "evidence_absent", "evidence_ambiguous", "evidence_conflicting",
    "human_review_required", "retrieval_sufficient", "annotator_id", "annotation_round", "notes",
]

DOCUMENT_WORKSHEET_FIELDS = [
    "document_id", "document_title", "stratum", "iasia_reason_code_context_only", "dual_review_subset",
    "gold_final_decision", "gold_primary_exclusion_reason_code", "annotator_id", "annotation_round", "notes",
]


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _format_evidence(spans) -> str:
    parts = []
    for span in spans:
        parts.append(f"[page={span.page} chunk={span.chunk_id}] {span.excerpt}")
    return " || ".join(parts) if parts else "(no evidence retrieved)"


def _read_existing_rows(path: Path, fieldnames: list[str]) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    dual_review = set(manifest["dual_review_subset"])
    parser = DoclingDocumentParser()
    retriever = LexicalEvidenceRetriever()

    criteria_path = OUTPUT_DIR / "worksheet_criteria.csv"
    documents_path = OUTPUT_DIR / "worksheet_documents.csv"
    criterion_rows = _read_existing_rows(criteria_path, CRITERION_WORKSHEET_FIELDS)
    document_rows = _read_existing_rows(documents_path, DOCUMENT_WORKSHEET_FIELDS)
    already_done = {row["document_id"] for row in document_rows}
    if already_done:
        print(f"Resuming: {len(already_done)} documents already processed: {sorted(already_done)}")

    for entry in manifest["documents"]:
        doc_id = entry["document_id"]
        if doc_id in already_done:
            continue
        pdf_path = Path(entry["pdf_path"])
        started = time.monotonic()
        document_sha256 = _hash_file(pdf_path)
        document = parser.parse(
            document_id=doc_id, document_sha256=document_sha256, pdf_path=pdf_path,
            parser_options=DoclingDocumentParser.recommended_parser_options(),
        )
        elapsed = time.monotonic() - started
        print(f"{doc_id}: parsed in {elapsed:.1f}s, status={document.coverage.status}, "
              f"pages={document.coverage.total_pages}, ocr_used={document.coverage.ocr_used}")

        for criterion in CRITERIA_SET.criteria:
            evidence = retriever.retrieve(document=document, criterion=criterion, retriever_options={})
            criterion_rows.append({
                "document_id": doc_id, "document_title": entry["title"], "stratum": entry["stratum"],
                "dual_review_subset": "TRUE" if doc_id in dual_review else "FALSE",
                "page_count": document.coverage.total_pages, "ocr_used": document.coverage.ocr_used,
                "criterion_id": criterion.criterion_id, "criterion_text": criterion.eligibility_requirement,
                "retrieved_evidence": _format_evidence(evidence),
                "gold_status": "", "supporting_pages": "", "supporting_passages": "",
                "contradicting_pages": "", "contradicting_passages": "",
                "evidence_absent": "", "evidence_ambiguous": "", "evidence_conflicting": "",
                "human_review_required": "", "retrieval_sufficient": "",
                "annotator_id": "", "annotation_round": "primary", "notes": "",
            })

        document_rows.append({
            "document_id": doc_id, "document_title": entry["title"], "stratum": entry["stratum"],
            "iasia_reason_code_context_only": entry["iasia_reason_code"] or "",
            "dual_review_subset": "TRUE" if doc_id in dual_review else "FALSE",
            "gold_final_decision": "", "gold_primary_exclusion_reason_code": "",
            "annotator_id": "", "annotation_round": "primary", "notes": "",
        })

        # Write after EVERY document, not just at the end -- Docling parsing
        # is slow enough (100-200s/doc observed) that a run interrupted
        # partway through must not lose already-completed work; resuming
        # (via already_done above) picks up exactly where this left off.
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(criteria_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CRITERION_WORKSHEET_FIELDS)
            writer.writeheader()
            writer.writerows(criterion_rows)
        with open(documents_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=DOCUMENT_WORKSHEET_FIELDS)
            writer.writeheader()
            writer.writerows(document_rows)

    print(f"\nWrote {len(criterion_rows)} criterion rows and {len(document_rows)} document rows.")


if __name__ == "__main__":
    main()
