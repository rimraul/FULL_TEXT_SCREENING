"""codex_code_review_ade9f48.md finding 2: generates BLANK,
independently-shuffled second-pass worksheets covering ONLY the sealed
dual-review subset. This is the single point where that subset is first
revealed to the annotator -- it refuses to run until the primary pass has
been VERIFIED sealed (codex_code_review_93bf42d.md finding 3, extended by
codex_code_review_73cf5d9.md finding 1: verification now checks the
seal's IDENTITY fields -- protocol version, criteria-set hash, exact
document-ID set, blind/sample manifest hashes -- not just the pass name
and CSV hashes), so the reveal cannot happen before the full 20-document
primary pass, sealed against THIS exact protocol/criteria/manifest
identity, is genuinely complete and unmodified since.

The generated worksheets contain NO primary-pass answers: they are built
from the still-blank template rows in worksheet_phase1_criteria.csv/
_documents.csv (criterion_id/criterion_text/document_page_count only),
never from the sealed (answered) primary CSVs.

Reruns are safe and set-transactional (codex_code_review_93bf42d.md
finding 4, hardened by codex_code_review_73cf5d9.md finding 4): both
output files are preflighted and published together, so a partial pair
can never be created before the SECOND file is found to already hold
different (possibly human-entered) content.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, r"C:/projects/FT_SCREENING/src")
sys.path.insert(0, r"C:/projects/FT_SCREENING/scripts")

from ft_screening.gold import SUPPORTED_PROTOCOL_VERSION  # noqa: E402
from ft_screening.hashing import compute_criteria_set_hash  # noqa: E402
from gold_set_build_worksheet import CRITERIA_SET  # noqa: E402
from gold_set_workflow_common import hash_file, load_and_verify_seal, write_generated_csv_set_idempotent  # noqa: E402

GOLD_SET_DIR = Path(r"C:/projects/FT_SCREENING/gold_set")
SEALED_DIR = GOLD_SET_DIR / "_sealed"
PRIMARY_SEAL_DIR = GOLD_SET_DIR / "_primary_seal"

# Deliberately different from BLIND_ORDER_SEED (gold_set_randomize_blind_order_v4.py)
# -- an independent shuffle, not a sub-sequence of the phase-1 order.
SECOND_PASS_ORDER_SEED = 20260817

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


def main() -> None:
    blind_manifest_path = GOLD_SET_DIR / "blind_manifest_v1.json"
    sample_manifest_path = SEALED_DIR / "sample_v1.json"
    # The full document-ID list is what the annotator already sees via
    # blind_manifest_v1.json (not secret) -- reading it does not reveal
    # the dual-review subset, which is read only AFTER the seal verifies.
    blind_manifest = json.loads(blind_manifest_path.read_text(encoding="utf-8"))
    all_document_ids = {d["document_id"] for d in blind_manifest["documents"]}

    load_and_verify_seal(
        PRIMARY_SEAL_DIR / "primary_seal_v1.json", expected_pass_name="primary",
        criterion_csv_path=PRIMARY_SEAL_DIR / "primary_criteria_sealed.csv",
        document_csv_path=PRIMARY_SEAL_DIR / "primary_documents_sealed.csv",
        expected_protocol_version=SUPPORTED_PROTOCOL_VERSION, expected_criteria_set_hash=compute_criteria_set_hash(CRITERIA_SET),
        expected_document_ids=all_document_ids, expected_blind_manifest_hash=hash_file(blind_manifest_path),
        expected_sample_manifest_hash=hash_file(sample_manifest_path),
    )

    sample_manifest = json.loads(sample_manifest_path.read_text(encoding="utf-8"))
    dual_review_ids = set(sample_manifest["dual_review_subset"])

    with open(GOLD_SET_DIR / "worksheet_phase1_criteria.csv", newline="", encoding="utf-8") as f:
        template_criterion_rows = [r for r in csv.DictReader(f) if r["document_id"] in dual_review_ids]
    with open(GOLD_SET_DIR / "worksheet_phase1_documents.csv", newline="", encoding="utf-8") as f:
        template_document_rows = [r for r in csv.DictReader(f) if r["document_id"] in dual_review_ids]

    rng = random.Random(SECOND_PASS_ORDER_SEED)
    shuffled_ids = sorted(dual_review_ids)
    rng.shuffle(shuffled_ids)
    order_index = {doc_id: i for i, doc_id in enumerate(shuffled_ids)}

    criterion_rows = []
    for row in template_criterion_rows:
        blank = {k: "" for k in CRITERION_FIELDS}
        blank.update(
            document_id=row["document_id"], document_page_count=row["document_page_count"],
            criterion_id=row["criterion_id"], criterion_text=row["criterion_text"],
            annotation_round="second_pass",
        )
        criterion_rows.append(blank)
    criterion_rows.sort(key=lambda r: (order_index[r["document_id"]], r["criterion_id"]))

    document_rows = []
    for row in template_document_rows:
        blank = {k: "" for k in DOCUMENT_FIELDS}
        blank.update(document_id=row["document_id"], document_page_count=row["document_page_count"], annotation_round="second_pass")
        document_rows.append(blank)
    document_rows.sort(key=lambda r: order_index[r["document_id"]])

    wrote_any = write_generated_csv_set_idempotent([
        (GOLD_SET_DIR / "worksheet_second_pass_criteria.csv", CRITERION_FIELDS, criterion_rows),
        (GOLD_SET_DIR / "worksheet_second_pass_documents.csv", DOCUMENT_FIELDS, document_rows),
    ])

    if not wrote_any:
        print("Second-pass worksheets already exist with identical content -- no-op.")
        return
    print(
        f"Revealed dual-review subset ({len(dual_review_ids)} documents) and wrote "
        f"{len(criterion_rows)} second-pass criterion rows, {len(document_rows)} second-pass document rows. "
        "These worksheets contain NO primary-pass answers."
    )


if __name__ == "__main__":
    main()
