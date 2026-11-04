"""Maintainer-only, sealed-file check: confirms the shuffled document
order in `blind_manifest_v1.json` does not reproduce the sealed sample's
stratum-block ordering.

codex_code_review_1c40a27.md finding 1: the PREVIOUS version of this
check lived as narrative text in `gold_set_protocol_v4.md` itself,
printing the actual shuffled real stratum labels for the first several
documents -- since the blind manifest's row order is public/
annotator-facing, that sentence let an annotator directly recover those
documents' Stage-1 labels before ever opening a PDF.
codex_code_review_ade9f48.md finding 1 then found that even ILLUSTRATING
the pattern with the real label values (as this docstring itself
previously did) is a live leak for as long as the blind order stays
unchanged -- not just a historical description of a fixed bug. This
docstring, like every other annotator-facing or repo-visible file, must
never quote the actual per-document label sequence, only describe that
this script checks for one. This script performs the verification but
reads the SEALED sample manifest (`_sealed/sample_v1.json`, never opened
during annotation) and prints only a pass/fail summary -- never the
actual per-document stratum sequence. Run this after
`gold_set_randomize_blind_order_v4.py`, never publish its full output.
"""

from __future__ import annotations

import json
from pathlib import Path

GOLD_SET_DIR = Path(r"C:/projects/FT_SCREENING/gold_set")


def main() -> None:
    blind_manifest = json.loads((GOLD_SET_DIR / "blind_manifest_v1.json").read_text(encoding="utf-8"))
    sample_manifest = json.loads((GOLD_SET_DIR / "_sealed" / "sample_v1.json").read_text(encoding="utf-8"))

    stratum_by_id = {d["document_id"]: d["stratum"] for d in sample_manifest["documents"]}
    shuffled_strata = [stratum_by_id[d["document_id"]] for d in blind_manifest["documents"]]
    sealed_block_strata = [d["stratum"] for d in sample_manifest["documents"]]

    is_blocked = shuffled_strata == sealed_block_strata
    # A run-length check: the sealed manifest is one contiguous block per
    # stratum, so its longest run equals that stratum's count. If the
    # shuffled order's longest same-stratum run is anywhere near that,
    # position still leaks a lot of information even if not IDENTICAL.
    def _longest_run(seq: list[str]) -> int:
        best = cur = 1
        for i in range(1, len(seq)):
            cur = cur + 1 if seq[i] == seq[i - 1] else 1
            best = max(best, cur)
        return best

    shuffled_longest_run = _longest_run(shuffled_strata)
    sealed_longest_run = _longest_run(sealed_block_strata)

    print(f"documents: {len(shuffled_strata)}")
    print(f"order identical to sealed stratum-block order: {is_blocked}")
    print(f"longest same-stratum run: shuffled={shuffled_longest_run}, sealed-block={sealed_longest_run}")

    if is_blocked:
        raise SystemExit("FAIL: blind_manifest_v1.json's row order reproduces the sealed stratum-block order")
    if shuffled_longest_run >= sealed_longest_run:
        raise SystemExit("FAIL: shuffled order's longest same-stratum run is not shorter than the sealed block's")
    print("PASS: shuffled order is not stratum-blocked")


if __name__ == "__main__":
    main()
