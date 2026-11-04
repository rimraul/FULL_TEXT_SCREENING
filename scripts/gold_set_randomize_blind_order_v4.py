"""codex_code_review_772916e.md finding 1 fix: `blind_manifest_v1.json`
and both phase-1 worksheets previously preserved the sealed sample's
stratum-block row order (6 INCLUDE, 10 EXCLUDE, 4 UNCERTAIN in sequence),
which combined with the protocol's own stated block counts to let an
annotator infer each document's Stage-1 stratum from its position alone.

Reorders all annotator-facing phase-1 files using a FIXED, documented
seed (reproducible, not re-randomized on every run) that is independent
of stratum. Only touches ROW ORDER -- no content changes.

codex_code_review_1c40a27.md finding 1 fix: the previous version shuffled
whichever order was CURRENTLY on disk, so a second run permuted an
already-permuted order -- not idempotent, and "fixed seed" did not
actually mean "fixed output". The shuffle now always starts from a
canonical, order-independent input (the sorted document_id list), so
running this script any number of times produces byte-identical files.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

GOLD_SET_DIR = Path(r"C:/projects/FT_SCREENING/gold_set")
SEALED_DIR = GOLD_SET_DIR / "_sealed"

BLIND_ORDER_SEED = 20260816  # fixed and documented -- gold_set_protocol_v5.md references this exact value


def _canonical_shuffled_order(document_ids: list[str]) -> list[str]:
    """Always shuffles the SORTED id list, never whatever order happens to
    be on disk -- this is what makes the output idempotent across reruns."""
    canonical = sorted(set(document_ids))
    rng = random.Random(BLIND_ORDER_SEED)
    shuffled = canonical[:]
    rng.shuffle(shuffled)
    return shuffled


def main() -> None:
    blind_path = GOLD_SET_DIR / "blind_manifest_v1.json"
    blind_manifest = json.loads(blind_path.read_text(encoding="utf-8"))
    document_ids = [d["document_id"] for d in blind_manifest["documents"]]

    shuffled_ids = _canonical_shuffled_order(document_ids)
    order_index = {doc_id: i for i, doc_id in enumerate(shuffled_ids)}

    blind_manifest["documents"].sort(key=lambda d: order_index[d["document_id"]])
    blind_path.write_text(json.dumps(blind_manifest, indent=2), encoding="utf-8")
    print(f"Reordered blind_manifest_v1.json ({len(shuffled_ids)} documents)")

    for csv_name in ("worksheet_phase1_criteria.csv", "worksheet_phase1_documents.csv"):
        csv_path = GOLD_SET_DIR / csv_name
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)
        rows.sort(key=lambda r: order_index[r["document_id"]])
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Reordered {csv_name}")

    print(
        "Note: the resulting document order is intentionally not printed or logged anywhere "
        "annotator-facing -- pair it with Stage-1 stratum labels only in a maintainer-only, "
        "sealed check (see scripts/gold_set_verify_blind_order_not_blocked.py), never in "
        "gold_set_protocol_v5.md or any other file the annotator reads."
    )


if __name__ == "__main__":
    main()
