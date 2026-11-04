"""Builds a CSV human-review worksheet from the frozen
gold_evidence_set_claude_draft_v1.json, ordered per the reviewer's stated
priority:
  1. The 4 disputed entries needing expanded evidence (sufficient=None,
     among the originally-flagged keys).
  2. The 1 provisional False (a confident but unconfirmed draft judgment).
  3. The 1 empty-evidence entry (needs a human exhaustive-search attestation).
  4. The remaining 44 heuristic-pass entries (sufficient=None, never
     individually content-reviewed).

Every row carries the actual criterion wording (so the reviewer never has
to cross-reference gold_set_v1.json separately), the retained original
citation, and the current Claude draft (role/excerpt/inference/rationale)
for context -- plus blank columns for reviewer_id, reviewed_at,
human_sufficiency, review_notes, and human_attestation (only meaningful for
the priority-3 row) to be filled in.

Read-only: never modifies gold_evidence_set_claude_draft_v1.json or
gold_set_v1.json.

Usage:
    python build_gold_evidence_review_worksheet.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, r"C:/projects/FT_SCREENING/src")

from ft_screening.gold import load_frozen_gold_set, load_frozen_model  # noqa: E402
from ft_screening.gold_evidence_set import GoldEvidenceSet  # noqa: E402

FROZEN_GOLD_SET_PATH = Path(r"C:/projects/FT_SCREENING/gold_set/frozen/gold_set_v1.json")
EXPECTED_GOLD_SET_HASH = "6fdfaebfd84e7ed33101d1a9b9ba4960367ab6985f1a52e3a13cabc2e396280a"
DRAFT_PATH = Path(r"C:/projects/FT_SCREENING/outputs/gold_evidence_set_v1/gold_evidence_set_claude_draft_v2.json")
OUTPUT_PATH = Path(r"C:/projects/FT_SCREENING/outputs/gold_evidence_set_v1/gold_evidence_review_worksheet_v1.csv")

# The 4 keys explicitly needing evidence EXPANSION (not just re-judgment) --
# tier 1, per the reviewer's stated priority.
TIER1_EXPANSION_NEEDED = {
    ("10_1007_s11269-016-1491-7", "I3_governance_dimension"),
    ("10_5194_nhess-21-1383-2021", "I3_governance_dimension"),
    ("10_1080_09640568_2019_1641474", "I3_governance_dimension"),
    ("10_1007_978-3-031-54118-6_8", "I2_drr_hazard_framing"),
}
TIER2_PROVISIONAL_FALSE = {("10_1080_13669877_2017_1351466", "I2_drr_hazard_framing")}
TIER3_EMPTY_PACKAGE = {("10_1007_978-3-031-54118-6_8", "I3_governance_dimension")}
# Not named in the reviewer's stated 4 tiers (which summed to 50, not all
# 60) -- these 10 already received an actual content-level read and a
# confident True judgment, so they belong ahead of the 44 truly-unreviewed
# heuristic-pass entries (quicker to confirm) but are still real work, not
# tier-1/2/3 urgent. Kept as their own tier rather than silently folded
# into either neighbor.


def _tier(key: tuple[str, str], sufficiency_label: str) -> int:
    if key in TIER1_EXPANSION_NEEDED:
        return 1
    if key in TIER2_PROVISIONAL_FALSE:
        return 2
    if key in TIER3_EMPTY_PACKAGE:
        return 3
    if sufficiency_label == "True":
        return 4
    return 5


def main() -> None:
    gold_set = load_frozen_gold_set(FROZEN_GOLD_SET_PATH, expected_hash=EXPECTED_GOLD_SET_HASH)
    draft = load_frozen_model(GoldEvidenceSet, DRAFT_PATH, expected_artifact_type="GoldEvidenceSet")
    if draft.gold_set_hash != gold_set.content_hash():
        raise ValueError("draft.gold_set_hash does not match the current gold_set_v1.json -- refusing to build a worksheet against a stale binding")

    criteria_by_id = {c.criterion_id: c for c in gold_set.criteria_set.criteria}
    entries_by_key = {(e.document_id, e.criterion_id): e for e in draft.entries}

    rows = []
    for e in draft.entries:
        key = (e.document_id, e.criterion_id)
        criterion = criteria_by_id[e.criterion_id]
        excerpt_summaries = []
        for x in e.self_contained_excerpts:
            body = f" | body_excerpt: {x.body_excerpt}" if x.body_excerpt else ""
            excerpt_summaries.append(
                f"[{x.stance}/{x.evidence_role}] p{x.page} '{x.heading_or_quotation}'{body} "
                f"-- supports: {x.supported_inference}"
            )
        draft_sufficiency_label = {True: "True", False: "False", None: "None (undecided)"}[e.sufficient_for_independent_adjudication]
        rows.append({
            "priority_tier": _tier(key, draft_sufficiency_label),
            "document_id": e.document_id,
            "criterion_id": e.criterion_id,
            "criterion_requirement": criterion.eligibility_requirement,
            "original_gold_status": next((a.gold_status for a in gold_set.criterion_annotations if a.document_id == e.document_id and a.criterion_id == e.criterion_id), ""),
            "original_supporting_passages": " ||| ".join(e.original_supporting_passages),
            "original_contradicting_passages": " ||| ".join(e.original_contradicting_passages),
            "draft_excerpts": "\n".join(excerpt_summaries) if excerpt_summaries else "(none -- empty package)",
            "draft_sufficiency": draft_sufficiency_label,
            "draft_rationale": e.sufficiency_rationale,
            # --- blank columns for the reviewer to fill in ---
            "reviewer_id": "",
            "reviewed_at": "",
            "human_sufficiency": "",  # True / False
            "human_attestation": "no_qualifying_evidence_found_after_full_document_review" if key in TIER3_EMPTY_PACKAGE else "",
            "expanded_excerpt_text": "",  # if the reviewer adds a body/visual excerpt, paste it here
            "expanded_excerpt_page": "",
            "review_notes": "",
        })

    rows.sort(key=lambda r: (r["priority_tier"], r["document_id"], r["criterion_id"]))

    fieldnames = list(rows[0].keys())
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    by_tier: dict[int, int] = {}
    for r in rows:
        by_tier[r["priority_tier"]] = by_tier.get(r["priority_tier"], 0) + 1
    print(f"Wrote {len(rows)} rows to {OUTPUT_PATH}")
    print(f"Tier counts: {by_tier}  (1=expansion needed, 2=provisional False, 3=empty package, "
          "4=already content-reviewed True -- quick confirm, 5=heuristic-pass-only, needs a real first read)")


if __name__ == "__main__":
    main()
