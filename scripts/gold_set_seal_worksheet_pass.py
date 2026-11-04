"""codex_code_review_ade9f48.md finding 2: seals a COMPLETE, single-round
worksheet pass (primary or second_pass) by copying it into an immutable
snapshot and hashing that snapshot -- before this, the only documented
way to add second-pass rows was to reopen the already-answered primary
CSV and append to it, exposing the primary pass's own answers during the
supposedly-independent second pass.

Usage:
    python gold_set_seal_worksheet_pass.py primary
    python gold_set_seal_worksheet_pass.py second_pass

`primary` seals gold_set/worksheet_phase1_{criteria,documents}.csv
against blind_manifest_v1.json's full 20-document set.

`second_pass` seals gold_set/worksheet_second_pass_{criteria,documents}.csv
(produced by gold_set_generate_second_pass_worksheets.py) against the
sealed sample manifest's 5-document dual_review_subset.

All real validation (every row's annotation_round matches the pass being
sealed, and the pass covers EXACTLY the full document x criterion key
set -- codex_code_review_93bf42d.md finding 1: a pass missing a whole
criterion for every document, or with a duplicated key, previously
sealed as "complete") lives in `ft_screening.gold.seal_worksheet_pass` --
this script is only file plumbing around it, passing the one real
`CRITERIA_SET` (from `gold_set_build_worksheet.py`) and
`SUPPORTED_PROTOCOL_VERSION` so the seal binds both identities.

The two sealed snapshots are published as ONE set via the same
`publish_output_set_exclusive()` every other workflow script uses
(codex_code_review_6f5dcd0.md finding 2: this script previously used two
sequential `shutil.copyfile()` calls outside any rollback -- if the
second copy failed, the first remained and blocked a normal retry, and
`copyfile()` can silently overwrite a concurrently-created destination).
If `seal_worksheet_pass()` then rejects the pass as invalid, the
snapshots are removed via the same ownership-aware
`safe_unlink_if_matches()` rollback uses, never a blind unlink.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, r"C:/projects/FT_SCREENING/src")
sys.path.insert(0, r"C:/projects/FT_SCREENING/scripts")

from ft_screening.gold import SUPPORTED_PROTOCOL_VERSION, seal_worksheet_pass  # noqa: E402
from gold_set_build_worksheet import CRITERIA_SET  # noqa: E402 -- the one real, canonical criteria set
from gold_set_workflow_common import publish_output_set_exclusive, safe_unlink_if_matches  # noqa: E402

GOLD_SET_DIR = Path(r"C:/projects/FT_SCREENING/gold_set")
SEALED_DIR = GOLD_SET_DIR / "_sealed"
SEAL_DIR = GOLD_SET_DIR / "_primary_seal"

_WORKING_FILES = {
    "primary": ("worksheet_phase1_criteria.csv", "worksheet_phase1_documents.csv"),
    "second_pass": ("worksheet_second_pass_criteria.csv", "worksheet_second_pass_documents.csv"),
}


def _expected_document_ids(pass_name: str) -> set[str]:
    if pass_name == "primary":
        blind = json.loads((GOLD_SET_DIR / "blind_manifest_v1.json").read_text(encoding="utf-8"))
        return {d["document_id"] for d in blind["documents"]}
    sample = json.loads((SEALED_DIR / "sample_v1.json").read_text(encoding="utf-8"))
    return set(sample["dual_review_subset"])


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in _WORKING_FILES:
        raise SystemExit("usage: gold_set_seal_worksheet_pass.py {primary|second_pass}")
    pass_name = sys.argv[1]
    criterion_name, document_name = _WORKING_FILES[pass_name]
    working_criterion_path = GOLD_SET_DIR / criterion_name
    working_document_path = GOLD_SET_DIR / document_name

    SEAL_DIR.mkdir(parents=True, exist_ok=True)
    sealed_criterion_path = SEAL_DIR / f"{pass_name}_criteria_sealed.csv"
    sealed_document_path = SEAL_DIR / f"{pass_name}_documents_sealed.csv"

    # Read BEFORE validating/hashing, so the seal is always over an
    # immutable snapshot, never the working file someone could keep
    # editing. Published as one exclusive set -- no partial pair can
    # survive a mid-publication failure or a concurrent destination.
    criterion_bytes = working_criterion_path.read_bytes()
    document_bytes = working_document_path.read_bytes()
    publish_output_set_exclusive([
        (sealed_criterion_path, criterion_bytes), (sealed_document_path, document_bytes),
    ])

    try:
        seal = seal_worksheet_pass(
            pass_name=pass_name, protocol_version=SUPPORTED_PROTOCOL_VERSION, criteria_set=CRITERIA_SET,
            blind_manifest_path=GOLD_SET_DIR / "blind_manifest_v1.json", sample_manifest_path=SEALED_DIR / "sample_v1.json",
            criterion_csv_path=sealed_criterion_path, document_csv_path=sealed_document_path,
            expected_document_ids=_expected_document_ids(pass_name), output_dir=SEAL_DIR,
            version_label=f"{pass_name}_seal_v1",
        )
    except BaseException:
        # Validation failed (e.g. the pass isn't actually complete yet) --
        # remove ONLY the snapshots this run just published, and only if
        # they still hold exactly the bytes just written (never a
        # concurrent edit), so a fixed retry isn't blocked by
        # "already exists".
        safe_unlink_if_matches(sealed_criterion_path, criterion_bytes)
        safe_unlink_if_matches(sealed_document_path, document_bytes)
        raise
    print(
        f"Sealed {pass_name!r} pass: {len(seal.document_ids)} documents, "
        f"criterion_csv_sha256={seal.criterion_csv_sha256[:12]}..., document_csv_sha256={seal.document_csv_sha256[:12]}..."
    )


if __name__ == "__main__":
    main()
