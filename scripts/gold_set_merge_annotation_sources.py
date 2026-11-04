"""codex_code_review_ade9f48.md finding 2, hardened per
codex_code_review_93bf42d.md findings 2-4 and codex_code_review_73cf5d9.md
findings 1/3/4: mechanically merges the sealed primary pass, the sealed
second pass (mandatory whenever the sample's `dual_review_subset` is
non-empty -- never silently optional), and the completed adjudication
worksheet (also mandatory whenever dual review applies, even if it has
zero disagreeing rows) into the single combined CSV shape
`gold.freeze_gold_set()` consumes.

Writes a STRICT `WorksheetSourceProvenance`-shaped JSON: a per-role
`WorksheetSourceRecord` for primary/second_pass/adjudication, each
retaining the VERIFIED sha256 of its criterion/document CSVs (and, for
sealed passes, the seal's own `content_hash()`) -- not just a mutable
path string (finding 3). The adjudication pair is now hashed here for
the first time (it was never sealed OR hashed before). The provenance
also retains the merged CSVs' own hashes as a secondary reproducibility
check, and the protocol/criteria/manifest identity every source was
verified against.

Every sealed source is re-verified against its own seal -- including the
seal's IDENTITY fields (protocol version, criteria-set hash, exact
document-ID set, blind/sample manifest hashes -- finding 1), not just
pass name and CSV bytes -- BEFORE any output is written. All three
outputs (both merged CSVs and the provenance file) are precomputed,
preflighted, and published together as ONE set (finding 4): a rerun
producing byte-identical content is a safe no-op; anything else is
refused before any file is touched, using the project's established
exclusive-write pattern rather than a per-file exists()-then-replace
race.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, r"C:/projects/FT_SCREENING/src")
sys.path.insert(0, r"C:/projects/FT_SCREENING/scripts")

from ft_screening.gold import SUPPORTED_PROTOCOL_VERSION  # noqa: E402
from ft_screening.hashing import compute_criteria_set_hash  # noqa: E402
from gold_set_build_worksheet import CRITERIA_SET  # noqa: E402
from gold_set_workflow_common import hash_file, load_and_verify_seal, publish_output_set_exclusive  # noqa: E402

GOLD_SET_DIR = Path(r"C:/projects/FT_SCREENING/gold_set")
SEALED_DIR = GOLD_SET_DIR / "_sealed"
SEAL_DIR = GOLD_SET_DIR / "_primary_seal"

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

MERGED_CRITERION_PATH = GOLD_SET_DIR / "worksheet_merged_criteria.csv"
MERGED_DOCUMENT_PATH = GOLD_SET_DIR / "worksheet_merged_documents.csv"
PROVENANCE_PATH = SEAL_DIR / "source_provenance_v1.json"


def _read_rows(path: Path, fields: list) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return [{k: row.get(k, "") for k in fields} for row in csv.DictReader(f)]


def _render_csv(fields: list, rows: list) -> bytes:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


def main() -> None:
    blind_manifest_path = GOLD_SET_DIR / "blind_manifest_v1.json"
    sample_manifest_path = SEALED_DIR / "sample_v1.json"
    blind_manifest = json.loads(blind_manifest_path.read_text(encoding="utf-8"))
    all_document_ids = {d["document_id"] for d in blind_manifest["documents"]}
    sample_manifest = json.loads(sample_manifest_path.read_text(encoding="utf-8"))
    dual_review_ids = set(sample_manifest["dual_review_subset"])
    dual_review_required = bool(dual_review_ids)

    criteria_set_hash = compute_criteria_set_hash(CRITERIA_SET)
    blind_manifest_hash = hash_file(blind_manifest_path)
    sample_manifest_hash = hash_file(sample_manifest_path)

    primary_seal_path = SEAL_DIR / "primary_seal_v1.json"
    primary_criterion_path = SEAL_DIR / "primary_criteria_sealed.csv"
    primary_document_path = SEAL_DIR / "primary_documents_sealed.csv"
    primary_seal = load_and_verify_seal(
        primary_seal_path, expected_pass_name="primary",
        criterion_csv_path=primary_criterion_path, document_csv_path=primary_document_path,
        expected_protocol_version=SUPPORTED_PROTOCOL_VERSION, expected_criteria_set_hash=criteria_set_hash,
        expected_document_ids=all_document_ids, expected_blind_manifest_hash=blind_manifest_hash,
        expected_sample_manifest_hash=sample_manifest_hash,
    )
    merged_criterion = _read_rows(primary_criterion_path, CRITERION_FIELDS)
    merged_document = _read_rows(primary_document_path, DOCUMENT_FIELDS)

    provenance = {
        "protocol_version": SUPPORTED_PROTOCOL_VERSION, "criteria_set_hash": criteria_set_hash,
        "blind_manifest_hash": blind_manifest_hash, "sample_manifest_hash": sample_manifest_hash,
        "primary": {
            "role": "primary", "seal_path": str(primary_seal_path), "seal_content_hash": primary_seal.content_hash(),
            "criterion_csv_path": str(primary_criterion_path), "criterion_csv_sha256": hash_file(primary_criterion_path),
            "document_csv_path": str(primary_document_path), "document_csv_sha256": hash_file(primary_document_path),
        },
    }

    if dual_review_required:
        second_pass_seal_path = SEAL_DIR / "second_pass_seal_v1.json"
        second_pass_criterion_path = SEAL_DIR / "second_pass_criteria_sealed.csv"
        second_pass_document_path = SEAL_DIR / "second_pass_documents_sealed.csv"
        second_pass_seal = load_and_verify_seal(
            second_pass_seal_path, expected_pass_name="second_pass",
            criterion_csv_path=second_pass_criterion_path, document_csv_path=second_pass_document_path,
            expected_protocol_version=SUPPORTED_PROTOCOL_VERSION, expected_criteria_set_hash=criteria_set_hash,
            expected_document_ids=dual_review_ids, expected_blind_manifest_hash=blind_manifest_hash,
            expected_sample_manifest_hash=sample_manifest_hash,
        )
        merged_criterion += _read_rows(second_pass_criterion_path, CRITERION_FIELDS)
        merged_document += _read_rows(second_pass_document_path, DOCUMENT_FIELDS)

        adjudication_criterion_path = GOLD_SET_DIR / "worksheet_adjudication_criteria.csv"
        adjudication_document_path = GOLD_SET_DIR / "worksheet_adjudication_documents.csv"
        if not adjudication_criterion_path.exists() or not adjudication_document_path.exists():
            raise SystemExit(
                "the sample requires dual review, so an adjudication artifact is REQUIRED (even a zero-row "
                "one recording that no keys disagreed) -- run gold_set_generate_adjudication_worksheet.py first"
            )
        merged_criterion += _read_rows(adjudication_criterion_path, CRITERION_FIELDS)
        merged_document += _read_rows(adjudication_document_path, DOCUMENT_FIELDS)

        provenance["second_pass"] = {
            "role": "second_pass", "seal_path": str(second_pass_seal_path), "seal_content_hash": second_pass_seal.content_hash(),
            "criterion_csv_path": str(second_pass_criterion_path), "criterion_csv_sha256": hash_file(second_pass_criterion_path),
            "document_csv_path": str(second_pass_document_path), "document_csv_sha256": hash_file(second_pass_document_path),
        }
        # Adjudication files were never sealed OR hashed at all before
        # this fix (codex_code_review_73cf5d9.md finding 3) -- hashed
        # here for the first time, including the valid zero-row case.
        provenance["adjudication"] = {
            "role": "adjudication", "seal_path": None, "seal_content_hash": None,
            "criterion_csv_path": str(adjudication_criterion_path), "criterion_csv_sha256": hash_file(adjudication_criterion_path),
            "document_csv_path": str(adjudication_document_path), "document_csv_sha256": hash_file(adjudication_document_path),
        }

    merged_criterion_bytes = _render_csv(CRITERION_FIELDS, merged_criterion)
    merged_document_bytes = _render_csv(DOCUMENT_FIELDS, merged_document)
    provenance["merged_criterion_csv_sha256"] = hashlib.sha256(merged_criterion_bytes).hexdigest()
    provenance["merged_document_csv_sha256"] = hashlib.sha256(merged_document_bytes).hexdigest()
    provenance["merged_at"] = datetime.now(timezone.utc).isoformat()
    provenance_bytes = json.dumps(provenance, indent=2).encode("utf-8")

    wrote_any = publish_output_set_exclusive([
        (MERGED_CRITERION_PATH, merged_criterion_bytes),
        (MERGED_DOCUMENT_PATH, merged_document_bytes),
        (PROVENANCE_PATH, provenance_bytes),
    ])

    if not wrote_any:
        print("Merged outputs already exist with identical content -- no-op.")
        return
    print(
        f"Merged {len(merged_criterion)} criterion rows / {len(merged_document)} document rows "
        f"-> {MERGED_CRITERION_PATH}, {MERGED_DOCUMENT_PATH}. Wrote source provenance to {PROVENANCE_PATH}."
    )
    print(
        f"Next: gold.freeze_gold_set(protocol_version={SUPPORTED_PROTOCOL_VERSION!r}, ..., "
        f"criterion_csv_path={str(MERGED_CRITERION_PATH)!r}, document_csv_path={str(MERGED_DOCUMENT_PATH)!r}, "
        f"source_provenance_path={str(PROVENANCE_PATH)!r})"
    )


if __name__ == "__main__":
    main()
