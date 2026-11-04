"""Script-level end-to-end tests for the sealed-worksheet workflow
scripts (gold_set_seal_worksheet_pass.py,
gold_set_generate_second_pass_worksheets.py,
gold_set_generate_adjudication_worksheet.py,
gold_set_merge_annotation_sources.py).

codex_code_review_93bf42d.md explicitly asked for script-level tests
(not just tests of the underlying ft_screening.gold functions) covering
invalid seal envelopes, post-seal edits, missing half-pairs, a
required-but-absent second pass, and rerun/partial-failure safety.
codex_code_review_73cf5d9.md added: adjudication files being hashed for
the first time, and a partial output pair never being created before a
refusal. This file reproduces each of those exact adversarial repros
against the REAL scripts (loaded via importlib, with path constants
redirected into a scratch tmp_path, never against the real gold_set/
files).
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import fitz
import pytest

sys.path.insert(0, r"C:/projects/FT_SCREENING/src")
sys.path.insert(0, r"C:/projects/FT_SCREENING/scripts")

from ft_screening.gold import freeze_gold_set  # noqa: E402
from ft_screening.schema import CriteriaSet, CriterionSpec  # noqa: E402

SCRIPTS_DIR = Path(r"C:/projects/FT_SCREENING/scripts")

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


def _criteria_set() -> CriteriaSet:
    return CriteriaSet(criteria_set_id="workflow-test", version="1", criteria=[
        CriterionSpec(criterion_id="pop", version="1", eligibility_requirement="x", missing_evidence_policy="review", exclusion_reason_code="wp", exclusion_reason_label="wp", priority=1),
        CriterionSpec(criterion_id="design", version="1", eligibility_requirement="y", missing_evidence_policy="pass", exclusion_reason_code="wd", exclusion_reason_label="wd", priority=2),
    ])


def _write_csv(path: Path, fieldnames: list, rows: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _filled_criterion_row(d, c, *, status="satisfied", round_="primary", ann_id="RR"):
    return {k: "" for k in CRITERION_FIELDS} | {
        "document_id": d, "document_page_count": "5", "criterion_id": c, "criterion_text": f"criterion {c}",
        "gold_status": status, "supporting_pages": "[1]" if status == "satisfied" else "[]",
        "supporting_passages": '["evidence text"]' if status == "satisfied" else "[]",
        "contradicting_pages": "[1]" if status == "violated" else "[]",
        "contradicting_passages": '["contra text"]' if status == "violated" else "[]",
        "evidence_absent": "FALSE", "evidence_ambiguous": "FALSE", "evidence_conflicting": "FALSE",
        "human_review_required": "FALSE", "annotator_id": ann_id, "annotation_round": round_, "notes": "",
    }


def _setup_workspace(tmp_path: Path, *, document_ids: list, dual_review_ids: list) -> tuple:
    sealed_dir = tmp_path / "_sealed"
    sealed_dir.mkdir()
    seal_dir = tmp_path / "_primary_seal"

    pdf_paths, pdf_hashes = {}, {}
    for doc_id in document_ids:
        doc = fitz.open()
        for i in range(5):
            doc.new_page().insert_text((72, 72), f"page {i + 1}")
        pdf_path = (tmp_path / f"{doc_id}.pdf").resolve()
        doc.save(str(pdf_path))
        doc.close()
        pdf_paths[doc_id] = str(pdf_path)
        import hashlib
        digest = hashlib.sha256()
        with open(pdf_path, "rb") as f:
            digest.update(f.read())
        pdf_hashes[doc_id] = digest.hexdigest()

    (tmp_path / "blind_manifest_v1.json").write_text(json.dumps({
        "protocol_version": "gold_set_protocol_v9",
        "documents": [{"document_id": d, "pdf_path": pdf_paths[d], "document_page_count": 5} for d in document_ids],
    }), encoding="utf-8")
    (sealed_dir / "sample_v1.json").write_text(json.dumps({
        "protocol_version": "gold_set_protocol_v9", "dual_review_subset": dual_review_ids,
        "documents": [
            {"document_id": d, "stratum": "INCLUDE", "pdf_path": pdf_paths[d], "pdf_sha256": pdf_hashes[d], "page_count": 5}
            for d in document_ids
        ],
    }), encoding="utf-8")

    primary_criterion_rows = [_filled_criterion_row(d, c) for d in document_ids for c in ("pop", "design")]
    primary_document_rows = [
        {k: "" for k in DOCUMENT_FIELDS} | {
            "document_id": d, "document_page_count": "5", "gold_final_decision": "include",
            "annotator_id": "RR", "annotation_round": "primary",
        }
        for d in document_ids
    ]
    _write_csv(tmp_path / "worksheet_phase1_criteria.csv", CRITERION_FIELDS, primary_criterion_rows)
    _write_csv(tmp_path / "worksheet_phase1_documents.csv", DOCUMENT_FIELDS, primary_document_rows)

    return sealed_dir, seal_dir, primary_criterion_rows, primary_document_rows


def _load_module(name: str, **path_overrides):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for attr, value in path_overrides.items():
        setattr(mod, attr, value)
    return mod


def _seal_mod(tmp_path, sealed_dir, seal_dir):
    return _load_module(
        "gold_set_seal_worksheet_pass", GOLD_SET_DIR=tmp_path, SEALED_DIR=sealed_dir, SEAL_DIR=seal_dir,
        CRITERIA_SET=_criteria_set(),
    )


# ---------------------------------------------------------------------------
# Finding 1: exact coverage enforced at seal time (script level)
# ---------------------------------------------------------------------------


def test_script_seal_rejects_an_incomplete_pass(tmp_path):
    sealed_dir, seal_dir, primary_criterion_rows, primary_document_rows = _setup_workspace(
        tmp_path, document_ids=["d1", "d2"], dual_review_ids=[],
    )
    incomplete_rows = [_filled_criterion_row("d1", "pop"), _filled_criterion_row("d2", "pop")]  # design missing
    _write_csv(tmp_path / "worksheet_phase1_criteria.csv", CRITERION_FIELDS, incomplete_rows)
    seal_mod = _seal_mod(tmp_path, sealed_dir, seal_dir)
    sys.argv = ["gold_set_seal_worksheet_pass.py", "primary"]
    with pytest.raises((SystemExit, ValueError)):
        seal_mod.main()
    assert not (seal_dir / "primary_seal_v1.json").exists()


def test_script_seal_does_not_leave_a_partial_snapshot_pair_on_second_copy_failure(tmp_path, monkeypatch):
    """codex_code_review_6f5dcd0.md finding 2's exact repro: the seal
    script previously copied its two sealed snapshots via sequential
    shutil.copyfile() calls outside any rollback -- if the second copy
    failed, the first remained on disk and blocked a normal retry. Now
    both are published as one exclusive set."""
    sealed_dir, seal_dir, *_ = _setup_workspace(tmp_path, document_ids=["d1", "d2"], dual_review_ids=[])
    seal_mod = _seal_mod(tmp_path, sealed_dir, seal_dir)

    import gold_set_workflow_common as common
    real_try = common._try_exclusive_write
    call_count = {"n": 0}

    def faulty_try(path, data):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise OSError("simulated disk failure on second copy")
        return real_try(path, data)

    monkeypatch.setattr(common, "_try_exclusive_write", faulty_try)
    sys.argv = ["gold_set_seal_worksheet_pass.py", "primary"]
    with pytest.raises(OSError):
        seal_mod.main()

    assert not (seal_dir / "primary_criteria_sealed.csv").exists()
    assert not (seal_dir / "primary_documents_sealed.csv").exists()
    assert not (seal_dir / "primary_seal_v1.json").exists()

    # A retry with the same (unmodified) working files must now succeed.
    monkeypatch.undo()
    seal_mod2 = _seal_mod(tmp_path, sealed_dir, seal_dir)
    sys.argv = ["gold_set_seal_worksheet_pass.py", "primary"]
    seal_mod2.main()
    assert (seal_dir / "primary_seal_v1.json").exists()


# ---------------------------------------------------------------------------
# Finding 3: verified transitions, not just file existence
# ---------------------------------------------------------------------------


def test_script_second_pass_generator_rejects_an_invalid_seal_file(tmp_path):
    """Exact repro: a file containing the literal text 'not a valid seal'
    must not unlock the dual-review subset."""
    sealed_dir, seal_dir, *_ = _setup_workspace(tmp_path, document_ids=["d1", "d2"], dual_review_ids=["d1"])
    seal_dir.mkdir(parents=True, exist_ok=True)
    (seal_dir / "primary_seal_v1.json").write_bytes(b"not a valid seal")
    (seal_dir / "primary_criteria_sealed.csv").write_text("irrelevant", encoding="utf-8")
    (seal_dir / "primary_documents_sealed.csv").write_text("irrelevant", encoding="utf-8")

    gen_mod = _load_module(
        "gold_set_generate_second_pass_worksheets", GOLD_SET_DIR=tmp_path, SEALED_DIR=sealed_dir, PRIMARY_SEAL_DIR=seal_dir,
    )
    with pytest.raises(SystemExit):
        gen_mod.main()
    assert not (tmp_path / "worksheet_second_pass_criteria.csv").exists()


def test_script_second_pass_generator_rejects_a_post_seal_edit(tmp_path):
    sealed_dir, seal_dir, *_ = _setup_workspace(tmp_path, document_ids=["d1", "d2"], dual_review_ids=["d1"])
    seal_mod = _seal_mod(tmp_path, sealed_dir, seal_dir)
    sys.argv = ["gold_set_seal_worksheet_pass.py", "primary"]
    seal_mod.main()

    # Edit the sealed snapshot after sealing.
    with open(seal_dir / "primary_criteria_sealed.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows[0]["gold_status"] = "violated"
    _write_csv(seal_dir / "primary_criteria_sealed.csv", CRITERION_FIELDS, rows)

    gen_mod = _load_module(
        "gold_set_generate_second_pass_worksheets", GOLD_SET_DIR=tmp_path, SEALED_DIR=sealed_dir, PRIMARY_SEAL_DIR=seal_dir,
        CRITERIA_SET=_criteria_set(),
    )
    with pytest.raises(SystemExit):
        gen_mod.main()


def test_script_merge_requires_second_pass_when_dual_review_applies(tmp_path):
    sealed_dir, seal_dir, *_ = _setup_workspace(tmp_path, document_ids=["d1", "d2"], dual_review_ids=["d1"])
    seal_mod = _seal_mod(tmp_path, sealed_dir, seal_dir)
    sys.argv = ["gold_set_seal_worksheet_pass.py", "primary"]
    seal_mod.main()
    # No second pass ever sealed, despite d1 being in dual_review_subset.
    merge_mod = _load_module(
        "gold_set_merge_annotation_sources", GOLD_SET_DIR=tmp_path, SEALED_DIR=sealed_dir, SEAL_DIR=seal_dir,
        MERGED_CRITERION_PATH=tmp_path / "worksheet_merged_criteria.csv",
        MERGED_DOCUMENT_PATH=tmp_path / "worksheet_merged_documents.csv",
        PROVENANCE_PATH=seal_dir / "source_provenance_v1.json", CRITERIA_SET=_criteria_set(),
    )
    with pytest.raises(SystemExit):
        merge_mod.main()
    assert not (tmp_path / "worksheet_merged_criteria.csv").exists()


# ---------------------------------------------------------------------------
# Finding 4: rerun safety
# ---------------------------------------------------------------------------


def _seal_primary(tmp_path, sealed_dir, seal_dir):
    seal_mod = _seal_mod(tmp_path, sealed_dir, seal_dir)
    sys.argv = ["gold_set_seal_worksheet_pass.py", "primary"]
    seal_mod.main()


def test_script_second_pass_generator_rerun_is_idempotent_when_unchanged(tmp_path):
    sealed_dir, seal_dir, *_ = _setup_workspace(tmp_path, document_ids=["d1", "d2"], dual_review_ids=["d1"])
    _seal_primary(tmp_path, sealed_dir, seal_dir)
    gen_mod = _load_module(
        "gold_set_generate_second_pass_worksheets", GOLD_SET_DIR=tmp_path, SEALED_DIR=sealed_dir, PRIMARY_SEAL_DIR=seal_dir,
        CRITERIA_SET=_criteria_set(),
    )
    gen_mod.main()
    first_bytes = (tmp_path / "worksheet_second_pass_criteria.csv").read_bytes()
    gen_mod.main()  # rerun -- must not raise
    assert (tmp_path / "worksheet_second_pass_criteria.csv").read_bytes() == first_bytes


def test_script_second_pass_generator_rerun_refuses_to_erase_a_human_answer(tmp_path):
    """Exact repro: rerunning the generator silently erased a
    human-entered answer."""
    sealed_dir, seal_dir, *_ = _setup_workspace(tmp_path, document_ids=["d1", "d2"], dual_review_ids=["d1"])
    _seal_primary(tmp_path, sealed_dir, seal_dir)
    gen_mod = _load_module(
        "gold_set_generate_second_pass_worksheets", GOLD_SET_DIR=tmp_path, SEALED_DIR=sealed_dir, PRIMARY_SEAL_DIR=seal_dir,
        CRITERIA_SET=_criteria_set(),
    )
    gen_mod.main()
    with open(tmp_path / "worksheet_second_pass_criteria.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows[0]["gold_status"] = "satisfied"
    _write_csv(tmp_path / "worksheet_second_pass_criteria.csv", CRITERION_FIELDS, rows)

    with pytest.raises(SystemExit):
        gen_mod.main()
    with open(tmp_path / "worksheet_second_pass_criteria.csv", newline="", encoding="utf-8") as f:
        surviving_rows = list(csv.DictReader(f))
    assert surviving_rows[0]["gold_status"] == "satisfied"  # the human answer is NOT erased


def test_script_merge_rerun_is_refused_not_silently_reapplied(tmp_path):
    sealed_dir, seal_dir, *_ = _setup_workspace(tmp_path, document_ids=["d1", "d2"], dual_review_ids=[])
    _seal_primary(tmp_path, sealed_dir, seal_dir)
    merge_mod = _load_module(
        "gold_set_merge_annotation_sources", GOLD_SET_DIR=tmp_path, SEALED_DIR=sealed_dir, SEAL_DIR=seal_dir,
        MERGED_CRITERION_PATH=tmp_path / "worksheet_merged_criteria.csv",
        MERGED_DOCUMENT_PATH=tmp_path / "worksheet_merged_documents.csv",
        PROVENANCE_PATH=seal_dir / "source_provenance_v1.json", CRITERIA_SET=_criteria_set(),
    )
    merge_mod.main()
    merged_bytes_before = (tmp_path / "worksheet_merged_criteria.csv").read_bytes()
    manifest_bytes_before = (seal_dir / "source_provenance_v1.json").read_bytes()

    with pytest.raises(SystemExit):
        merge_mod.main()
    # A refused rerun must not have touched the already-written outputs.
    assert (tmp_path / "worksheet_merged_criteria.csv").read_bytes() == merged_bytes_before
    assert (seal_dir / "source_provenance_v1.json").read_bytes() == manifest_bytes_before


def test_script_generator_does_not_create_a_partial_pair_before_refusing(tmp_path):
    """codex_code_review_73cf5d9.md finding 4's exact repro: the FIRST
    output file was created before the SECOND was found to already hold
    conflicting content, leaving a half-written pair on disk. Both
    outputs must now be preflighted together, so a conflict on EITHER
    one refuses the whole set before anything is written."""
    sealed_dir, seal_dir, *_ = _setup_workspace(tmp_path, document_ids=["d1", "d2"], dual_review_ids=["d1"])
    _seal_primary(tmp_path, sealed_dir, seal_dir)
    gen_mod = _load_module(
        "gold_set_generate_second_pass_worksheets", GOLD_SET_DIR=tmp_path, SEALED_DIR=sealed_dir, PRIMARY_SEAL_DIR=seal_dir,
        CRITERIA_SET=_criteria_set(),
    )
    # Pre-create the SECOND of the two outputs with conflicting content --
    # the first output (criteria.csv) does not exist yet at all.
    (tmp_path / "worksheet_second_pass_documents.csv").write_text("not,the,right,content\n", encoding="utf-8")
    assert not (tmp_path / "worksheet_second_pass_criteria.csv").exists()

    with pytest.raises(SystemExit):
        gen_mod.main()

    # The refusal must have happened BEFORE writing the first output too.
    assert not (tmp_path / "worksheet_second_pass_criteria.csv").exists()


def test_script_merge_hashes_the_adjudication_pair(tmp_path):
    """codex_code_review_73cf5d9.md finding 3: adjudication files were
    never sealed OR hashed at all before this fix. The merge script must
    now record real, verifiable hashes for them (and no seal, since
    adjudication has no coverage-validated seal)."""
    sealed_dir, seal_dir, *_ = _setup_workspace(tmp_path, document_ids=["d1", "d2"], dual_review_ids=["d1"])
    _seal_primary(tmp_path, sealed_dir, seal_dir)
    gen_mod = _load_module(
        "gold_set_generate_second_pass_worksheets", GOLD_SET_DIR=tmp_path, SEALED_DIR=sealed_dir, PRIMARY_SEAL_DIR=seal_dir,
        CRITERIA_SET=_criteria_set(),
    )
    gen_mod.main()
    with open(tmp_path / "worksheet_second_pass_criteria.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row.update(_filled_criterion_row(row["document_id"], row["criterion_id"], round_="second_pass", ann_id="XY"))
    _write_csv(tmp_path / "worksheet_second_pass_criteria.csv", CRITERION_FIELDS, rows)
    with open(tmp_path / "worksheet_second_pass_documents.csv", newline="", encoding="utf-8") as f:
        doc_rows = list(csv.DictReader(f))
    for row in doc_rows:
        row.update(document_id=row["document_id"], document_page_count="5", gold_final_decision="include", annotator_id="XY", annotation_round="second_pass")
    _write_csv(tmp_path / "worksheet_second_pass_documents.csv", DOCUMENT_FIELDS, doc_rows)

    sys.argv = ["gold_set_seal_worksheet_pass.py", "second_pass"]
    seal_mod = _seal_mod(tmp_path, sealed_dir, seal_dir)
    seal_mod.main()

    adj_mod = _load_module(
        "gold_set_generate_adjudication_worksheet", GOLD_SET_DIR=tmp_path, SEALED_DIR=sealed_dir, SEAL_DIR=seal_dir,
        CRITERIA_SET=_criteria_set(),
    )
    adj_mod.main()  # primary and second_pass agree (both defaults) -> zero-row adjudication, still hashed

    merge_mod = _load_module(
        "gold_set_merge_annotation_sources", GOLD_SET_DIR=tmp_path, SEALED_DIR=sealed_dir, SEAL_DIR=seal_dir,
        MERGED_CRITERION_PATH=tmp_path / "worksheet_merged_criteria.csv",
        MERGED_DOCUMENT_PATH=tmp_path / "worksheet_merged_documents.csv",
        PROVENANCE_PATH=seal_dir / "source_provenance_v1.json", CRITERIA_SET=_criteria_set(),
    )
    merge_mod.main()

    provenance = json.loads((seal_dir / "source_provenance_v1.json").read_text(encoding="utf-8"))
    adjudication = provenance["adjudication"]
    assert adjudication["seal_path"] is None
    assert adjudication["seal_content_hash"] is None
    import hashlib
    actual_criterion_hash = hashlib.sha256((tmp_path / "worksheet_adjudication_criteria.csv").read_bytes()).hexdigest()
    actual_document_hash = hashlib.sha256((tmp_path / "worksheet_adjudication_documents.csv").read_bytes()).hexdigest()
    assert adjudication["criterion_csv_sha256"] == actual_criterion_hash
    assert adjudication["document_csv_sha256"] == actual_document_hash


# ---------------------------------------------------------------------------
# End-to-end: the full real workflow, no dual review, actually freezes
# ---------------------------------------------------------------------------


def test_full_workflow_scripts_produce_a_freezable_gold_set_no_dual_review(tmp_path):
    sealed_dir, seal_dir, *_ = _setup_workspace(tmp_path, document_ids=["d1", "d2"], dual_review_ids=[])
    _seal_primary(tmp_path, sealed_dir, seal_dir)
    merge_mod = _load_module(
        "gold_set_merge_annotation_sources", GOLD_SET_DIR=tmp_path, SEALED_DIR=sealed_dir, SEAL_DIR=seal_dir,
        MERGED_CRITERION_PATH=tmp_path / "worksheet_merged_criteria.csv",
        MERGED_DOCUMENT_PATH=tmp_path / "worksheet_merged_documents.csv",
        PROVENANCE_PATH=seal_dir / "source_provenance_v1.json", CRITERIA_SET=_criteria_set(),
    )
    merge_mod.main()

    gold_set = freeze_gold_set(
        protocol_version="gold_set_protocol_v9", criteria_set=_criteria_set(),
        criterion_csv_path=tmp_path / "worksheet_merged_criteria.csv", document_csv_path=tmp_path / "worksheet_merged_documents.csv",
        blind_manifest_path=tmp_path / "blind_manifest_v1.json", sample_manifest_path=sealed_dir / "sample_v1.json",
        output_dir=tmp_path / "out", version_label="gold_set_v1", source_provenance_path=seal_dir / "source_provenance_v1.json",
    )
    assert gold_set.source_provenance.second_pass is None
    assert len(gold_set.criterion_annotations) == 4
