"""Builds the phase-1 BLIND annotation worksheets (gold_set_protocol_v2.md
§3/§5) and freezes the retrieval-oracle artifact (§7) for later phase-2
use -- per codex_code_review_a36e0df.md finding 1, gold truth must be
created WITHOUT seeing retrieval output or the Stage-1 (iasia_test)
decision, to avoid incorporation/anchoring bias.

Reuses the ALREADY-COMPUTED retrieval output from the old (retired)
worksheet_criteria.csv (built by gold_set_build_worksheet.py) -- no
Docling rerun needed (Codex's own suggested remediation path). That CSV's
`retrieved_evidence` column is reverse-parsed back into structured spans;
each parse is verified by re-formatting it and checking it reproduces the
original string EXACTLY, so any format-collision (an excerpt containing
the literal " || " or "] " delimiter sequence) is caught, not silently
mis-parsed.

Outputs:
  gold_set/worksheet_phase1_criteria.csv   -- BLIND, for gold annotation
  gold_set/worksheet_phase1_documents.csv  -- BLIND, for gold annotation
  gold_set/_sealed_stratification_reference.json -- stratum/reason-code
      reference, NOT to be opened until after phase-1 freeze
  gold_set/retrieval_artifact_v1.json + .sha256 -- frozen Arm-3 artifact
  gold_set/worksheet_phase2_retrieval_sufficiency.csv -- for phase 2,
      filled in only AFTER phase-1 gold is frozen
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, r"C:/projects/FT_SCREENING/src")

from ft_screening.gold import RetrievalOracleSpan, freeze_retrieval_oracle_artifact, make_evidence_key

MANIFEST_PATH = Path(r"C:/projects/FT_SCREENING/gold_set/sample_v1.json")
OLD_CRITERIA_CSV = Path(r"C:/projects/FT_SCREENING/gold_set/worksheet_criteria.csv")
OUTPUT_DIR = Path(r"C:/projects/FT_SCREENING/gold_set")

PARSER_NAME, PARSER_VERSION = "docling", "2.61.2"
RETRIEVER_NAME, RETRIEVER_CONFIG_VERSION = "lexical_bm25", "1"
RETRIEVER_OPTIONS: dict = {}

_EVIDENCE_ITEM_RE = re.compile(r"^\[page=(None|\d+) chunk=([^\]]+)\] (.*)$", re.DOTALL)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _format_evidence(spans: list[RetrievalOracleSpan]) -> str:
    """MUST match gold_set_build_worksheet.py's original `_format_evidence`
    exactly -- used here only to VERIFY the reverse-parse round-trips."""
    if not spans:
        return "(no evidence retrieved)"
    return " || ".join(f"[page={s.page} chunk={s.chunk_id}] {s.excerpt}" for s in spans)


def _parse_evidence(formatted: str, *, document_id: str, criterion_id: str) -> list[RetrievalOracleSpan]:
    if formatted == "(no evidence retrieved)":
        return []
    parts = formatted.split(" || ")
    spans = []
    for part in parts:
        m = _EVIDENCE_ITEM_RE.match(part)
        if not m:
            raise ValueError(
                f"could not reverse-parse retrieved_evidence for document_id={document_id!r} "
                f"criterion_id={criterion_id!r}: part={part!r}"
            )
        page_str, chunk_id, excerpt = m.groups()
        spans.append(RetrievalOracleSpan(chunk_id=chunk_id, page=None if page_str == "None" else int(page_str), excerpt=excerpt))
    # Round-trip verification: re-formatting must reproduce the ORIGINAL
    # string exactly, or a delimiter collision inside an excerpt silently
    # corrupted the parse.
    if _format_evidence(spans) != formatted:
        raise ValueError(
            f"round-trip verification failed for document_id={document_id!r} criterion_id={criterion_id!r} -- "
            "an excerpt likely contains the literal delimiter sequence used by the old formatter; "
            "this document/criterion's retrieval artifact needs a real Docling rerun instead of reuse"
        )
    return spans


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    dual_review = set(manifest["dual_review_subset"])

    with open(OLD_CRITERIA_CSV, newline="", encoding="utf-8") as f:
        old_rows = list(csv.DictReader(f))

    # --- Sealed stratification reference (NOT part of the blind worksheet) ---
    sealed = {
        entry["document_id"]: {
            "stratum": entry["stratum"], "iasia_reason_code": entry["iasia_reason_code"],
            "dual_review_subset": entry["document_id"] in dual_review,
        }
        for entry in manifest["documents"]
    }
    (OUTPUT_DIR / "_sealed_stratification_reference.json").write_text(
        json.dumps({"_do_not_open_before_phase1_freeze": True, "documents": sealed}, indent=2), encoding="utf-8"
    )

    # --- Phase-1 BLIND worksheets ---
    phase1_criteria_fields = [
        "document_id", "document_page_count", "criterion_id", "criterion_text",
        "gold_status", "supporting_pages", "supporting_passages", "contradicting_pages",
        "contradicting_passages", "evidence_absent", "evidence_ambiguous", "evidence_conflicting",
        "human_review_required", "annotator_id", "annotation_round", "notes",
    ]
    phase1_document_fields = [
        "document_id", "document_page_count", "gold_final_decision", "gold_primary_exclusion_reason_code",
        "aggregation_override_rationale", "annotator_id", "annotation_round", "notes",
    ]

    phase1_criteria_rows = []
    seen_docs: dict[str, str] = {}
    for row in old_rows:
        phase1_criteria_rows.append({
            "document_id": row["document_id"], "document_page_count": row["page_count"],
            "criterion_id": row["criterion_id"], "criterion_text": row["criterion_text"],
            "gold_status": "", "supporting_pages": "", "supporting_passages": "",
            "contradicting_pages": "", "contradicting_passages": "",
            "evidence_absent": "", "evidence_ambiguous": "", "evidence_conflicting": "",
            "human_review_required": "", "annotator_id": "", "annotation_round": "primary", "notes": "",
        })
        seen_docs[row["document_id"]] = row["page_count"]

    phase1_document_rows = [
        {
            "document_id": doc_id, "document_page_count": page_count,
            "gold_final_decision": "", "gold_primary_exclusion_reason_code": "",
            "aggregation_override_rationale": "", "annotator_id": "", "annotation_round": "primary", "notes": "",
        }
        for doc_id, page_count in seen_docs.items()
    ]

    with open(OUTPUT_DIR / "worksheet_phase1_criteria.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=phase1_criteria_fields)
        writer.writeheader()
        writer.writerows(phase1_criteria_rows)
    with open(OUTPUT_DIR / "worksheet_phase1_documents.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=phase1_document_fields)
        writer.writeheader()
        writer.writerows(phase1_document_rows)

    print(f"Wrote {len(phase1_criteria_rows)} phase-1 criterion rows, {len(phase1_document_rows)} phase-1 document rows.")

    # --- Retrieval-oracle artifact (Arm 3), reverse-parsed from the old CSV ---
    document_sha256 = {}
    for entry in manifest["documents"]:
        document_sha256[entry["document_id"]] = _hash_file(Path(entry["pdf_path"]))

    evidence_by_key: dict[str, list[RetrievalOracleSpan]] = {}
    for row in old_rows:
        spans = _parse_evidence(row["retrieved_evidence"], document_id=row["document_id"], criterion_id=row["criterion_id"])
        evidence_by_key[make_evidence_key(row["document_id"], row["criterion_id"])] = spans

    artifact = freeze_retrieval_oracle_artifact(
        parser_name=PARSER_NAME, parser_version=PARSER_VERSION, retriever_name=RETRIEVER_NAME,
        retriever_config_version=RETRIEVER_CONFIG_VERSION, retriever_options=RETRIEVER_OPTIONS,
        document_sha256=document_sha256, evidence_by_key=evidence_by_key,
        output_dir=OUTPUT_DIR, version_label="retrieval_artifact_v1",
    )
    print(f"Froze retrieval_artifact_v1 (hash={artifact.content_hash()[:12]}...), "
          f"{sum(len(v) for v in evidence_by_key.values())} total evidence spans across {len(evidence_by_key)} keys.")

    # --- Phase-2 worksheet (for AFTER phase-1 freeze -- generated now for
    # convenience, since the artifact is already frozen, but must not be
    # opened/filled in until phase 1 is complete). ---
    phase2_fields = [
        "document_id", "criterion_id", "retrieved_evidence", "retrieval_sufficient", "annotator_id", "notes",
    ]
    phase2_rows = []
    for row in old_rows:
        phase2_rows.append({
            "document_id": row["document_id"], "criterion_id": row["criterion_id"],
            "retrieved_evidence": row["retrieved_evidence"], "retrieval_sufficient": "", "annotator_id": "", "notes": "",
        })
    with open(OUTPUT_DIR / "worksheet_phase2_retrieval_sufficiency.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=phase2_fields)
        writer.writeheader()
        writer.writerows(phase2_rows)
    print(f"Wrote {len(phase2_rows)} phase-2 rows (do not fill in until phase-1 gold is frozen).")


if __name__ == "__main__":
    main()
