"""Screens every GoldCriterionAnnotation's supporting_passages/contradicting_passages
in the frozen gold_set_v1.json for whether each cited passage LOOKS like a bare
section heading, figure/table caption, or otherwise non-substantive citation
locator, rather than a self-contained excerpt an independent adjudicator could
rule on.

Prompted directly by outputs/prompt6_disagreement_audit_v1.md: 8 of 16 audited
Arm-1/Arm-2 disagreement keys had exactly this problem. This script extends the
check to ALL 60 gold-set entries (the audit only ever looked at the 16
disagreement keys), to scope how far the issue extends.

Deliberately conservative (biased toward flagging, not clearing) and disclosed
in full below -- this is a triage heuristic, not a substitute for a human
"is this sufficient to adjudicate without opening the PDF" judgment. Read-only:
never modifies gold_set_v1.json.

Usage:
    python screen_gold_evidence_self_containedness.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, r"C:/projects/FT_SCREENING/src")

from ft_screening.gold import load_frozen_gold_set  # noqa: E402

FROZEN_GOLD_SET_PATH = Path(r"C:/projects/FT_SCREENING/gold_set/frozen/gold_set_v1.json")
EXPECTED_GOLD_SET_HASH = "6fdfaebfd84e7ed33101d1a9b9ba4960367ab6985f1a52e3a13cabc2e396280a"

_HEADING_NUMBER_RE = re.compile(r"^\d+(\.\d+)*\.?\s+\S")
_CAPTION_PREFIX_RE = re.compile(r"^(fig(ure)?|table)\.?\s*\d", re.IGNORECASE)
_SENTENCE_END_RE = re.compile(r"[.!?]\s*$")
_MIN_SUBSTANTIVE_LENGTH = 60


def looks_non_substantive(passage: str) -> tuple[bool, str]:
    text = passage.strip()
    if not text:
        return True, "empty"
    if _CAPTION_PREFIX_RE.match(text):
        return True, "figure/table caption prefix"
    if _HEADING_NUMBER_RE.match(text) and len(text) < 120:
        return True, "numbered-heading pattern"
    if len(text) < _MIN_SUBSTANTIVE_LENGTH and not _SENTENCE_END_RE.search(text):
        return True, f"short ({len(text)} chars) with no sentence-ending punctuation"
    return False, ""


def main() -> None:
    gold_set = load_frozen_gold_set(FROZEN_GOLD_SET_PATH, expected_hash=EXPECTED_GOLD_SET_HASH)

    flagged: list[tuple[str, str, str, str]] = []  # document_id, criterion_id, side, reason
    total_passages = 0
    for ann in gold_set.criterion_annotations:
        for passage in ann.supporting_passages:
            total_passages += 1
            bad, reason = looks_non_substantive(passage)
            if bad:
                flagged.append((ann.document_id, ann.criterion_id, "supporting", reason))
        for passage in ann.contradicting_passages:
            total_passages += 1
            bad, reason = looks_non_substantive(passage)
            if bad:
                flagged.append((ann.document_id, ann.criterion_id, "contradicting", reason))

    flagged_keys = sorted({(doc, crit) for doc, crit, _, _ in flagged})
    n_annotations_with_evidence = sum(
        1 for ann in gold_set.criterion_annotations if ann.supporting_passages or ann.contradicting_passages
    )

    print(f"GoldSet: {len(gold_set.criterion_annotations)} criterion annotations, "
          f"{n_annotations_with_evidence} with at least one cited passage, {total_passages} total passages.")
    print(f"\n{len(flagged)}/{total_passages} passages flagged as non-substantive.")
    print(f"{len(flagged_keys)}/{n_annotations_with_evidence} (document_id, criterion_id) keys have AT LEAST ONE flagged passage:\n")
    for doc_id, crit_id in flagged_keys:
        reasons = [f"{side}:{reason}" for d, c, side, reason in flagged if d == doc_id and c == crit_id]
        print(f"  {doc_id} :: {crit_id}  ({'; '.join(reasons)})")


if __name__ == "__main__":
    main()
