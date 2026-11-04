"""codex_code_review_ade9f48.md finding 2: generates an adjudication
worksheet for ONLY the dual-review keys where the sealed primary and
sealed second-pass rows actually disagree -- both raw answers may
intentionally be shown side by side here, since by this point both
passes are already complete and sealed (showing them earlier would defeat
the second pass's independence). Reads the SEALED copies (verifying each
against its own seal hash, not the possibly-since-edited working files)
so this step cannot run against tampered or stale worksheets.

Refuses to run until both `primary_seal_v1.json` and
`second_pass_seal_v1.json` exist AND verify -- both the pass name/CSV
bytes AND the seal's IDENTITY fields (protocol version, criteria-set
hash, exact document-ID set, blind/sample manifest hashes --
codex_code_review_73cf5d9.md finding 1, via the shared
`load_and_verify_seal` helper), not just an existence check.

Reruns are safe and set-transactional (finding 4): both output files are
preflighted and published together; identical output is a no-op, and a
rerun that would change the content of an already-answered adjudication
worksheet is refused before either file is touched.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, r"C:/projects/FT_SCREENING/src")
sys.path.insert(0, r"C:/projects/FT_SCREENING/scripts")

from ft_screening.gold import (  # noqa: E402
    GoldCriterionAnnotation,
    GoldDocumentAnnotation,
    SUPPORTED_PROTOCOL_VERSION,
    criterion_rows_agree,
    document_rows_agree,
)
from ft_screening.hashing import compute_criteria_set_hash  # noqa: E402
from gold_set_build_worksheet import CRITERIA_SET  # noqa: E402
from gold_set_workflow_common import hash_file, load_and_verify_seal, write_generated_csv_set_idempotent  # noqa: E402

GOLD_SET_DIR = Path(r"C:/projects/FT_SCREENING/gold_set")
SEALED_DIR = GOLD_SET_DIR / "_sealed"
SEAL_DIR = GOLD_SET_DIR / "_primary_seal"

CRITERION_FIELDS = [
    "document_id", "document_page_count", "criterion_id", "criterion_text",
    "gold_status", "supporting_pages", "supporting_passages", "contradicting_pages",
    "contradicting_passages", "evidence_absent", "evidence_ambiguous", "evidence_conflicting",
    "human_review_required", "annotator_id", "annotation_round", "notes",
    "primary_answer_summary", "second_pass_answer_summary",
]
DOCUMENT_FIELDS = [
    "document_id", "document_page_count", "gold_final_decision", "gold_primary_exclusion_reason_code",
    "aggregation_override_rationale", "annotator_id", "annotation_round", "notes",
    "primary_answer_summary", "second_pass_answer_summary",
]


def _summarize_criterion(ann: GoldCriterionAnnotation) -> str:
    return (
        f"status={ann.gold_status} supporting_pages={list(ann.supporting_pages)} "
        f"contradicting_pages={list(ann.contradicting_pages)} evidence_absent={ann.evidence_absent} "
        f"ambiguous={ann.evidence_ambiguous} conflicting={ann.evidence_conflicting} "
        f"review_required={ann.human_review_required}"
    )


def _summarize_document(ann: GoldDocumentAnnotation) -> str:
    return f"decision={ann.gold_final_decision} reason={ann.gold_primary_exclusion_reason_code}"


def main() -> None:
    primary_criterion_path = SEAL_DIR / "primary_criteria_sealed.csv"
    primary_document_path = SEAL_DIR / "primary_documents_sealed.csv"
    second_pass_criterion_path = SEAL_DIR / "second_pass_criteria_sealed.csv"
    second_pass_document_path = SEAL_DIR / "second_pass_documents_sealed.csv"

    blind_manifest_path = GOLD_SET_DIR / "blind_manifest_v1.json"
    sample_manifest_path = SEALED_DIR / "sample_v1.json"
    blind_manifest = json.loads(blind_manifest_path.read_text(encoding="utf-8"))
    all_document_ids = {d["document_id"] for d in blind_manifest["documents"]}
    sample_manifest = json.loads(sample_manifest_path.read_text(encoding="utf-8"))
    dual_review_ids = set(sample_manifest["dual_review_subset"])
    criteria_set_hash = compute_criteria_set_hash(CRITERIA_SET)
    blind_manifest_hash = hash_file(blind_manifest_path)
    sample_manifest_hash = hash_file(sample_manifest_path)

    load_and_verify_seal(
        SEAL_DIR / "primary_seal_v1.json", expected_pass_name="primary",
        criterion_csv_path=primary_criterion_path, document_csv_path=primary_document_path,
        expected_protocol_version=SUPPORTED_PROTOCOL_VERSION, expected_criteria_set_hash=criteria_set_hash,
        expected_document_ids=all_document_ids, expected_blind_manifest_hash=blind_manifest_hash,
        expected_sample_manifest_hash=sample_manifest_hash,
    )
    load_and_verify_seal(
        SEAL_DIR / "second_pass_seal_v1.json", expected_pass_name="second_pass",
        criterion_csv_path=second_pass_criterion_path, document_csv_path=second_pass_document_path,
        expected_protocol_version=SUPPORTED_PROTOCOL_VERSION, expected_criteria_set_hash=criteria_set_hash,
        expected_document_ids=dual_review_ids, expected_blind_manifest_hash=blind_manifest_hash,
        expected_sample_manifest_hash=sample_manifest_hash,
    )

    with open(primary_criterion_path, newline="", encoding="utf-8") as f:
        primary_criterion = {(r["document_id"], r["criterion_id"]): GoldCriterionAnnotation.from_csv_row(r) for r in csv.DictReader(f)}
    with open(second_pass_criterion_path, newline="", encoding="utf-8") as f:
        second_pass_criterion = {(r["document_id"], r["criterion_id"]): GoldCriterionAnnotation.from_csv_row(r) for r in csv.DictReader(f)}
    with open(primary_document_path, newline="", encoding="utf-8") as f:
        primary_document = {r["document_id"]: GoldDocumentAnnotation.from_csv_row(r) for r in csv.DictReader(f)}
    with open(second_pass_document_path, newline="", encoding="utf-8") as f:
        second_pass_document = {r["document_id"]: GoldDocumentAnnotation.from_csv_row(r) for r in csv.DictReader(f)}

    missing_criterion = set(second_pass_criterion) - set(primary_criterion)
    if missing_criterion:
        raise SystemExit(f"second-pass criterion keys not present in the primary pass: {sorted(missing_criterion)}")
    missing_document = set(second_pass_document) - set(primary_document)
    if missing_document:
        raise SystemExit(f"second-pass document_ids not present in the primary pass: {sorted(missing_document)}")

    criterion_out = []
    for key, second in second_pass_criterion.items():
        primary = primary_criterion[key]
        if criterion_rows_agree(primary, second):
            continue
        doc_id, criterion_id = key
        row = {k: "" for k in CRITERION_FIELDS}
        row.update(
            document_id=doc_id, document_page_count=primary.document_page_count, criterion_id=criterion_id,
            annotation_round="adjudicated",
            primary_answer_summary=_summarize_criterion(primary), second_pass_answer_summary=_summarize_criterion(second),
        )
        criterion_out.append(row)

    document_out = []
    for doc_id, second in second_pass_document.items():
        primary = primary_document[doc_id]
        if document_rows_agree(primary, second):
            continue
        row = {k: "" for k in DOCUMENT_FIELDS}
        row.update(
            document_id=doc_id, document_page_count=primary.document_page_count, annotation_round="adjudicated",
            primary_answer_summary=_summarize_document(primary), second_pass_answer_summary=_summarize_document(second),
        )
        document_out.append(row)

    write_generated_csv_set_idempotent([
        (GOLD_SET_DIR / "worksheet_adjudication_criteria.csv", CRITERION_FIELDS, criterion_out),
        (GOLD_SET_DIR / "worksheet_adjudication_documents.csv", DOCUMENT_FIELDS, document_out),
    ])

    print(
        f"{len(criterion_out)} disagreeing criterion keys (of {len(second_pass_criterion)}), "
        f"{len(document_out)} disagreeing documents (of {len(second_pass_document)}) need adjudication."
    )


if __name__ == "__main__":
    main()
