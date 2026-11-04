"""codex_code_review_dca0970.md finding 1 fix: builds a genuinely BLIND
manifest (document_id, pdf_path, document_page_count ONLY -- no stratum,
no Stage-1 reason code, no dual-review membership) for the annotator to
actually use, since `sample_v1.json` itself still carries the judgmental
fields right next to pdf_path. Also:
- adds `page_count` to `sample_v1.json` (recovered from the already-built
  worksheet_phase1_criteria.csv's `document_page_count` column, no
  Docling rerun needed).
- rebuilds `retrieval_artifact_v1.json` under the NEW single-envelope,
  self-hashing schema (finding 6) with the new `parser_options_hash`/
  `criteria_set_hash` bindings (finding 3) -- reusing the ALREADY
  reverse-parsed evidence spans from the old (now-stale-format)
  retrieval_artifact_v1.json rather than re-parsing text a third time.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, r"C:/projects/FT_SCREENING/src")

from ft_screening.document import DoclingDocumentParser
from ft_screening.gold import RetrievalOracleSpan, freeze_retrieval_oracle_artifact
from ft_screening.schema import CriteriaSet, CriterionSpec

GOLD_SET_DIR = Path(r"C:/projects/FT_SCREENING/gold_set")
SAMPLE_MANIFEST_PATH = GOLD_SET_DIR / "sample_v1.json"
PHASE1_CRITERIA_CSV = GOLD_SET_DIR / "worksheet_phase1_criteria.csv"
OLD_RETRIEVAL_ARTIFACT_PATH = GOLD_SET_DIR / "retrieval_artifact_v1.json"

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


def main() -> None:
    manifest = json.loads(SAMPLE_MANIFEST_PATH.read_text(encoding="utf-8"))

    # --- Recover page_count per document from the already-built phase-1 CSV ---
    page_count_by_doc: dict[str, int] = {}
    with open(PHASE1_CRITERIA_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            page_count_by_doc[row["document_id"]] = int(row["document_page_count"])

    for entry in manifest["documents"]:
        entry["page_count"] = page_count_by_doc[entry["document_id"]]
    SAMPLE_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Added page_count to {len(manifest['documents'])} entries in sample_v1.json")

    # --- Build the genuinely blind manifest (no stratum/reason_code/dual_review) ---
    blind_manifest = {
        "protocol_version": "gold_set_protocol_v3",
        "documents": [
            {
                "document_id": entry["document_id"],
                "pdf_path": entry["pdf_path"],
                "document_page_count": entry["page_count"],
            }
            for entry in manifest["documents"]
        ],
    }
    blind_path = GOLD_SET_DIR / "blind_manifest_v1.json"
    blind_path.write_text(json.dumps(blind_manifest, indent=2), encoding="utf-8")
    print(f"Wrote {blind_path} with {len(blind_manifest['documents'])} documents (no stratum/reason_code/dual_review fields)")

    # --- Rebuild retrieval_artifact_v1.json under the new self-hashing schema ---
    old_artifact = json.loads(OLD_RETRIEVAL_ARTIFACT_PATH.read_text(encoding="utf-8"))
    document_sha256 = old_artifact["document_sha256"]
    evidence_by_key = {
        key: [RetrievalOracleSpan(**span) for span in spans]
        for key, spans in old_artifact["evidence_by_key"].items()
    }

    OLD_RETRIEVAL_ARTIFACT_PATH.unlink()
    old_sidecar = GOLD_SET_DIR / "retrieval_artifact_v1.sha256"
    if old_sidecar.exists():
        old_sidecar.unlink()

    artifact = freeze_retrieval_oracle_artifact(
        parser_name="docling", parser_version="2.61.2",
        parser_options=DoclingDocumentParser.recommended_parser_options(),
        retriever_name="lexical_bm25", retriever_config_version="1", retriever_options={},
        criteria_set=CRITERIA_SET, document_sha256=document_sha256, evidence_by_key=evidence_by_key,
        output_dir=GOLD_SET_DIR, version_label="retrieval_artifact_v1",
    )
    print(f"Rebuilt retrieval_artifact_v1.json (new schema, hash={artifact.content_hash()[:12]}...)")


if __name__ == "__main__":
    main()
