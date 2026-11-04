"""The 14 contract tests api_contract_v2.md requires before scaffolding is
considered done -- run against `_run_ft_screening` with deterministic fakes
and the real `FileScreeningStore` (per plan_v5.md step 4's corrected build
order: these exercise a real, fake-backed orchestration shell, not just
schema shapes).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from ft_screening.components import Components
from ft_screening.errors import ResumeConflictError, RunSetupError
from ft_screening.fakes import FakeCriterionAdjudicator, FakeDocumentParser, FakeEvidenceRetriever
from ft_screening.orchestrator import _run_ft_screening
from ft_screening.schema import (
    CriteriaSet,
    CriterionSpec,
    ScreeningConfig,
    ScreeningItem,
    ScreeningItemResult,
    ScreeningRequest,
    ScreeningRunResult,
)
from ft_screening.store import FileScreeningStore


def make_pdf(tmp_path: Path, name: str, content: bytes = b"%PDF-1.4 fake") -> Path:
    path = tmp_path / "pdfs" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def make_criteria_set(**overrides) -> CriteriaSet:
    criterion = CriterionSpec(
        criterion_id="population",
        version="1",
        eligibility_requirement="adult humans",
        missing_evidence_policy="review",
        exclusion_reason_code="wrong_population",
        exclusion_reason_label="Wrong population",
        priority=1,
    )
    defaults = dict(criteria_set_id="cs-1", version="1", criteria=[criterion])
    defaults.update(overrides)
    return CriteriaSet(**defaults)


def make_config(**overrides) -> ScreeningConfig:
    defaults = dict(
        parser_name="fake",
        parser_version="1",
        chunker_version="1",
        retriever_name="fake",
        retriever_config_version="1",
        adjudicator_provider="fake",
        adjudicator_model="fake-1",
        model_config_id="cfg-1",
        prompt_template_version="1",
        output_schema_version="1",
        aggregation_policy_version="1",
    )
    defaults.update(overrides)
    return ScreeningConfig(**defaults)


def make_components(tmp_path: Path, *, out_name: str = "outputs", **adjudicator_kwargs) -> tuple[Components, Path]:
    output_dir = tmp_path / out_name
    components = Components(
        parser=FakeDocumentParser(),
        retriever=FakeEvidenceRetriever(),
        adjudicator=FakeCriterionAdjudicator(**adjudicator_kwargs),
        store=FileScreeningStore(output_dir=output_dir),
    )
    return components, output_dir


# ---------------------------------------------------------------------------
# 1. Malformed/duplicate IDs fail preflight without creating partial output
# ---------------------------------------------------------------------------


def test_1_duplicate_document_id_fails_before_any_output(tmp_path):
    pdf1 = make_pdf(tmp_path, "a.pdf")
    pdf2 = make_pdf(tmp_path, "b.pdf")
    output_dir = tmp_path / "outputs"
    with pytest.raises(ValidationError):
        ScreeningRequest(
            items=[
                ScreeningItem(document_id="dup", report_id="r1", pdf_path=pdf1),
                ScreeningItem(document_id="dup", report_id="r2", pdf_path=pdf2),
            ],
            criteria_set=make_criteria_set(),
            config=make_config(),
            output_dir=output_dir,
        )
    assert not output_dir.exists()


# ---------------------------------------------------------------------------
# 2. Missing PDFs and hash mismatches are item outcomes, never exclusions
# ---------------------------------------------------------------------------


def test_2_missing_pdf_is_input_unavailable_not_an_exclusion(tmp_path):
    missing = tmp_path / "pdfs" / "does-not-exist.pdf"
    request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=missing)],
        criteria_set=make_criteria_set(),
        config=make_config(),
        output_dir=tmp_path / "outputs",
    )
    components, _ = make_components(tmp_path)
    result = _run_ft_screening(request, components)
    item = result.items[0]
    assert item.execution_status == "input_unavailable"
    assert item.eligibility is None


def test_2_hash_mismatch_is_processing_failed_not_an_exclusion(tmp_path):
    pdf1 = make_pdf(tmp_path, "a.pdf", b"real content")
    request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1, expected_sha256="f" * 64)],
        criteria_set=make_criteria_set(),
        config=make_config(),
        output_dir=tmp_path / "outputs",
    )
    components, _ = make_components(tmp_path)
    result = _run_ft_screening(request, components)
    item = result.items[0]
    assert item.execution_status == "processing_failed"
    assert item.eligibility is None
    assert item.errors[0].code == "hash_mismatch"


# ---------------------------------------------------------------------------
# 3. Output order matches input order under out-of-order parallel completion
# ---------------------------------------------------------------------------


def test_3_output_order_matches_input_order_despite_out_of_order_completion(tmp_path):
    items = [
        ScreeningItem(document_id=f"d{i}", report_id=f"r{i}", pdf_path=make_pdf(tmp_path, f"{i}.pdf", str(i).encode()))
        for i in range(4)
    ]
    request = ScreeningRequest(
        items=items,
        criteria_set=make_criteria_set(),
        config=make_config(max_workers=4),
        output_dir=tmp_path / "outputs",
    )
    # item 0 is the slowest -- naive completion order would put it last.
    components, _ = make_components(tmp_path, delay_by_key={("d0", "population"): 0.2})
    result = _run_ft_screening(request, components)
    assert [i.input_index for i in result.items] == [0, 1, 2, 3]
    assert [i.document_id for i in result.items] == ["d0", "d1", "d2", "d3"]
    assert all(i.execution_status == "completed" for i in result.items)


# ---------------------------------------------------------------------------
# 4. continue and fail_fast preserve the documented completed work
# ---------------------------------------------------------------------------


def test_4_continue_policy_processes_every_item_despite_a_failure(tmp_path):
    items = [
        ScreeningItem(document_id=f"d{i}", report_id=f"r{i}", pdf_path=make_pdf(tmp_path, f"{i}.pdf", str(i).encode()))
        for i in range(3)
    ]
    request = ScreeningRequest(
        items=items,
        criteria_set=make_criteria_set(),
        config=make_config(item_error_policy="continue"),
        output_dir=tmp_path / "outputs",
    )
    components, _ = make_components(tmp_path, fail_keys={("d1", "population")})
    result = _run_ft_screening(request, components)
    statuses = [i.execution_status for i in result.items]
    assert statuses == ["completed", "processing_failed", "completed"]
    assert result.run_status == "completed_with_issues"


def test_4_fail_fast_stops_new_work_but_keeps_completed_items(tmp_path):
    items = [
        ScreeningItem(document_id=f"d{i}", report_id=f"r{i}", pdf_path=make_pdf(tmp_path, f"ff{i}.pdf", str(i).encode()))
        for i in range(3)
    ]
    request = ScreeningRequest(
        items=items,
        criteria_set=make_criteria_set(),
        config=make_config(item_error_policy="fail_fast", max_workers=1),
        output_dir=tmp_path / "outputs",
    )
    components, _ = make_components(tmp_path, fail_keys={("d1", "population")})
    result = _run_ft_screening(request, components)
    statuses = [i.execution_status for i in result.items]
    assert statuses == ["completed", "processing_failed", "not_processed"]


# ---------------------------------------------------------------------------
# 5. Identical content follows the selected duplicate policy deterministically
# ---------------------------------------------------------------------------


def test_5_duplicate_content_error_policy(tmp_path):
    """codex_code_review_44d9e90.md finding 2: the frozen contract says
    EVERY member of a same-content group becomes duplicate_input under
    'error' -- not just the non-first copies. The first version of this
    test codified the wrong (screen-the-first-copy) behavior; corrected
    here."""
    pdf1 = make_pdf(tmp_path, "a.pdf", b"same-bytes")
    pdf2 = make_pdf(tmp_path, "b.pdf", b"same-bytes")
    pdf3 = make_pdf(tmp_path, "c.pdf", b"same-bytes")
    request = ScreeningRequest(
        items=[
            ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1),
            ScreeningItem(document_id="d2", report_id="r2", pdf_path=pdf2),
            ScreeningItem(document_id="d3", report_id="r3", pdf_path=pdf3),
        ],
        criteria_set=make_criteria_set(),
        config=make_config(duplicate_content_policy="error"),
        output_dir=tmp_path / "outputs",
    )
    components, _ = make_components(tmp_path)
    result = _run_ft_screening(request, components)
    assert [i.execution_status for i in result.items] == ["duplicate_input", "duplicate_input", "duplicate_input"]
    assert all(i.eligibility is None for i in result.items)
    assert result.run_status == "failed"  # zero completed -- none of the group was screened


def test_5_duplicate_content_error_policy_does_not_affect_distinct_content(tmp_path):
    pdf1 = make_pdf(tmp_path, "a.pdf", b"same-bytes")
    pdf2 = make_pdf(tmp_path, "b.pdf", b"same-bytes")
    pdf3 = make_pdf(tmp_path, "c.pdf", b"different-bytes")
    request = ScreeningRequest(
        items=[
            ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1),
            ScreeningItem(document_id="d2", report_id="r2", pdf_path=pdf2),
            ScreeningItem(document_id="d3", report_id="r3", pdf_path=pdf3),
        ],
        criteria_set=make_criteria_set(),
        config=make_config(duplicate_content_policy="error"),
        output_dir=tmp_path / "outputs",
    )
    components, _ = make_components(tmp_path)
    result = _run_ft_screening(request, components)
    assert result.items[0].execution_status == "duplicate_input"
    assert result.items[1].execution_status == "duplicate_input"
    assert result.items[2].execution_status == "completed"  # not part of the duplicate group


def test_5_duplicate_content_reuse_policy(tmp_path):
    pdf1 = make_pdf(tmp_path, "a.pdf", b"same-bytes")
    pdf2 = make_pdf(tmp_path, "b.pdf", b"same-bytes")
    request = ScreeningRequest(
        items=[
            ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1),
            ScreeningItem(document_id="d2", report_id="r2", pdf_path=pdf2),
        ],
        criteria_set=make_criteria_set(),
        config=make_config(duplicate_content_policy="reuse"),
        output_dir=tmp_path / "outputs",
    )
    components, _ = make_components(tmp_path)
    result = _run_ft_screening(request, components)
    assert result.items[0].execution_status == "completed"
    assert result.items[1].execution_status == "completed"
    assert result.items[1].provenance.cache_status == "content_reuse"


# ---------------------------------------------------------------------------
# 6. Cache modes produce identical domain results, distinct provenance
# ---------------------------------------------------------------------------


def test_6_cache_use_mode_hits_on_second_run_same_domain_result(tmp_path):
    pdf1 = make_pdf(tmp_path, "a.pdf")
    output_dir = tmp_path / "outputs"
    store = FileScreeningStore(output_dir=output_dir)

    def components():
        return Components(
            parser=FakeDocumentParser(), retriever=FakeEvidenceRetriever(), adjudicator=FakeCriterionAdjudicator(), store=store
        )

    req1 = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1)],
        criteria_set=make_criteria_set(),
        config=make_config(cache_mode="use"),
        output_dir=output_dir,
        run_id="run-1",
    )
    result1 = _run_ft_screening(req1, components())
    assert result1.items[0].provenance.cache_status == "miss"

    req2 = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1)],
        criteria_set=make_criteria_set(),
        config=make_config(cache_mode="use"),
        output_dir=output_dir,
        run_id="run-2",
    )
    result2 = _run_ft_screening(req2, components())
    assert result2.items[0].provenance.cache_status == "hit"
    assert result2.items[0].eligibility.final_decision == result1.items[0].eligibility.final_decision


def test_6_cache_refresh_mode_recomputes_and_overwrites(tmp_path):
    pdf1 = make_pdf(tmp_path, "a.pdf")
    output_dir = tmp_path / "outputs"
    store = FileScreeningStore(output_dir=output_dir)
    req1 = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1)],
        criteria_set=make_criteria_set(),
        config=make_config(cache_mode="use"),
        output_dir=output_dir,
        run_id="run-1",
    )
    components1 = Components(parser=FakeDocumentParser(), retriever=FakeEvidenceRetriever(), adjudicator=FakeCriterionAdjudicator(), store=store)
    _run_ft_screening(req1, components1)

    req2 = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1)],
        criteria_set=make_criteria_set(),
        config=make_config(cache_mode="refresh"),
        output_dir=output_dir,
        run_id="run-2",
    )
    components2 = Components(parser=FakeDocumentParser(), retriever=FakeEvidenceRetriever(), adjudicator=FakeCriterionAdjudicator(), store=store)
    result2 = _run_ft_screening(req2, components2)
    assert result2.items[0].provenance.cache_status == "refresh"


def test_6_cache_off_mode_never_reads_or_writes_cache(tmp_path):
    pdf1 = make_pdf(tmp_path, "a.pdf")
    output_dir = tmp_path / "outputs"
    request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1)],
        criteria_set=make_criteria_set(),
        config=make_config(cache_mode="off"),
        output_dir=output_dir,
    )
    components, _ = make_components(tmp_path)
    result = _run_ft_screening(request, components)
    assert result.items[0].provenance.cache_status == "disabled"
    assert not (output_dir / "cache").exists()


# ---------------------------------------------------------------------------
# 7. Resume rejects changed criteria/config, completes only unfinished items
# ---------------------------------------------------------------------------


def test_7_resume_rejects_changed_criteria_set(tmp_path):
    pdf1 = make_pdf(tmp_path, "a.pdf")
    output_dir = tmp_path / "outputs"
    req1 = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1)],
        criteria_set=make_criteria_set(),
        config=make_config(),
        output_dir=output_dir,
        run_id="run-1",
    )
    components, _ = make_components(tmp_path)
    _run_ft_screening(req1, components)

    changed_criteria_set = make_criteria_set(version="2")
    req2 = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1)],
        criteria_set=changed_criteria_set,
        config=make_config(),
        output_dir=output_dir,
        resume_run_id="run-1",
    )
    with pytest.raises(ResumeConflictError):
        _run_ft_screening(req2, components)


def test_7_resume_completes_only_unfinished_items(tmp_path):
    items = [
        ScreeningItem(document_id=f"d{i}", report_id=f"r{i}", pdf_path=make_pdf(tmp_path, f"r{i}.pdf", str(i).encode()))
        for i in range(3)
    ]
    output_dir = tmp_path / "outputs"
    config = make_config(item_error_policy="fail_fast", max_workers=1)
    criteria_set = make_criteria_set()

    req1 = ScreeningRequest(items=items, criteria_set=criteria_set, config=config, output_dir=output_dir, run_id="run-1")
    store = FileScreeningStore(output_dir=output_dir)
    failing_components = Components(
        parser=FakeDocumentParser(), retriever=FakeEvidenceRetriever(),
        adjudicator=FakeCriterionAdjudicator(fail_keys={("d1", "population")}), store=store,
    )
    result1 = _run_ft_screening(req1, failing_components)
    assert [i.execution_status for i in result1.items] == ["completed", "processing_failed", "not_processed"]

    req2 = ScreeningRequest(items=items, criteria_set=criteria_set, config=config, output_dir=output_dir, resume_run_id="run-1")
    fixed_components = Components(
        parser=FakeDocumentParser(), retriever=FakeEvidenceRetriever(), adjudicator=FakeCriterionAdjudicator(), store=store
    )
    result2 = _run_ft_screening(req2, fixed_components)
    assert [i.execution_status for i in result2.items] == ["completed", "completed", "completed"]
    assert result2.run_status == "completed"


# ---------------------------------------------------------------------------
# 8. Persisted JSON/JSONL round-trips through the declared models
# ---------------------------------------------------------------------------


def test_8_persisted_artifacts_round_trip(tmp_path):
    pdf1 = make_pdf(tmp_path, "a.pdf")
    output_dir = tmp_path / "outputs"
    request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1)],
        criteria_set=make_criteria_set(),
        config=make_config(),
        output_dir=output_dir,
    )
    components, _ = make_components(tmp_path)
    result = _run_ft_screening(request, components)

    restored_run = ScreeningRunResult.model_validate_json(result.artifacts.run_result_json.read_text())
    assert restored_run == result

    lines = result.artifacts.item_results_jsonl.read_text().strip().splitlines()
    restored_items = [ScreeningItemResult.model_validate_json(line) for line in lines]
    assert restored_items == result.items

    ledger_lines = result.artifacts.ledger_jsonl.read_text().strip().splitlines()
    assert len(ledger_lines) >= 1
    for line in ledger_lines:
        json.loads(line)  # must be valid JSON


# ---------------------------------------------------------------------------
# 9. Summaries are rejected when inconsistent with item results
# ---------------------------------------------------------------------------


def test_9_real_orchestrator_output_is_self_consistent(tmp_path):
    pdf1 = make_pdf(tmp_path, "a.pdf")
    request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1)],
        criteria_set=make_criteria_set(),
        config=make_config(),
        output_dir=tmp_path / "outputs",
    )
    components, _ = make_components(tmp_path)
    result = _run_ft_screening(request, components)
    # Re-validating the orchestrator's own output must never fail its own
    # summary-consistency check.
    ScreeningRunResult.model_validate(result.model_dump(mode="json"))

    corrupted = result.model_dump(mode="json")
    corrupted["summary"]["total"] = 999
    with pytest.raises(ValidationError):
        ScreeningRunResult.model_validate(corrupted)


# ---------------------------------------------------------------------------
# 10. slr_pipeline-style records join losslessly by report_id, no downstream import
# ---------------------------------------------------------------------------


def test_10_no_downstream_package_imported():
    import re

    import ft_screening

    # Checks actual import statements only -- documentation mentioning a
    # downstream consumer by name (e.g. api.py's docstring) is fine; an
    # actual `import slr_pipeline`/`from slr_pipeline import ...` would
    # violate the one-way dependency direction this checks for.
    import_pattern = re.compile(r"^\s*(?:import|from)\s+slr_pipeline\b", re.MULTILINE)
    package_dir = Path(ft_screening.__file__).parent
    for py_file in package_dir.glob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        assert not import_pattern.search(text), f"{py_file} imports slr_pipeline -- dependency direction violated"


def test_10_join_by_report_id_lossless(tmp_path):
    items = [
        ScreeningItem(document_id=f"d{i}", report_id=f"record-{i}", pdf_path=make_pdf(tmp_path, f"j{i}.pdf", str(i).encode()))
        for i in range(2)
    ]
    request = ScreeningRequest(
        items=items, criteria_set=make_criteria_set(), config=make_config(), output_dir=tmp_path / "outputs"
    )
    components, _ = make_components(tmp_path)
    result = _run_ft_screening(request, components)

    downstream_records = {"record-0": {"title": "Paper 0"}, "record-1": {"title": "Paper 1"}}
    joined = {item.report_id: downstream_records[item.report_id] for item in result.items}
    assert joined == downstream_records


# ---------------------------------------------------------------------------
# 11. fail_fast: ordered not_processed for never-started items
# ---------------------------------------------------------------------------


def test_11_fail_fast_ordered_not_processed_for_never_started(tmp_path):
    items = [
        ScreeningItem(document_id=f"d{i}", report_id=f"r{i}", pdf_path=make_pdf(tmp_path, f"nf{i}.pdf", str(i).encode()))
        for i in range(4)
    ]
    request = ScreeningRequest(
        items=items,
        criteria_set=make_criteria_set(),
        config=make_config(item_error_policy="fail_fast", max_workers=1),
        output_dir=tmp_path / "outputs",
    )
    components, _ = make_components(tmp_path, fail_keys={("d0", "population")})
    result = _run_ft_screening(request, components)
    statuses = [i.execution_status for i in result.items]
    assert statuses == ["processing_failed", "not_processed", "not_processed", "not_processed"]
    assert [i.input_index for i in result.items] == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# 12. run_id collision fails; concurrent resume of the same run is rejected
# ---------------------------------------------------------------------------


def test_12_run_id_collision_rejected(tmp_path):
    pdf1 = make_pdf(tmp_path, "a.pdf")
    output_dir = tmp_path / "outputs"
    request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1)],
        criteria_set=make_criteria_set(),
        config=make_config(),
        output_dir=output_dir,
        run_id="fixed-run",
    )
    components, _ = make_components(tmp_path)
    _run_ft_screening(request, components)
    with pytest.raises(RunSetupError):
        _run_ft_screening(request, components)


def test_12_concurrent_resume_of_same_run_rejected(tmp_path):
    output_dir = tmp_path / "outputs"
    store = FileScreeningStore(output_dir=output_dir)
    store.begin_run(run_id="shared-run", resume=False)
    try:
        with pytest.raises(ResumeConflictError):
            store.begin_run(run_id="shared-run", resume=True)  # lock still held
    finally:
        store.end_run(run_id="shared-run")


# ---------------------------------------------------------------------------
# 13. All four hashes match golden vectors -- covered in tests/golden/test_golden_hashes.py
# ---------------------------------------------------------------------------


def test_13_golden_hash_coverage_exists_elsewhere():
    import tests.golden.test_golden_hashes as golden_module  # noqa: F401

    assert hasattr(golden_module, "test_criteria_set_hash_golden_vector")


# ---------------------------------------------------------------------------
# 14. Cache hit from a different item's content returns rebound public IDs
# ---------------------------------------------------------------------------


def test_14_reused_duplicate_content_rebinds_public_identifiers(tmp_path):
    pdf1 = make_pdf(tmp_path, "a.pdf", b"identical-bytes")
    pdf2 = make_pdf(tmp_path, "b.pdf", b"identical-bytes")
    request = ScreeningRequest(
        items=[
            ScreeningItem(document_id="canonical-doc", report_id="canonical-report", pdf_path=pdf1),
            ScreeningItem(document_id="duplicate-doc", report_id="duplicate-report", pdf_path=pdf2),
        ],
        criteria_set=make_criteria_set(),
        config=make_config(duplicate_content_policy="reuse"),
        output_dir=tmp_path / "outputs",
        run_id="reuse-run",
    )
    components, _ = make_components(tmp_path)
    result = _run_ft_screening(request, components)

    canonical_item, duplicate_item = result.items
    assert duplicate_item.document_id == "duplicate-doc"
    assert duplicate_item.eligibility.document_id == "duplicate-doc"
    assert duplicate_item.eligibility.report_id == "duplicate-report"
    for assessment in duplicate_item.eligibility.assessments:
        for span in assessment.evidence:
            assert span.document_id == "duplicate-doc"  # never the canonical item's ID
    assert duplicate_item.provenance.cache_status == "content_reuse"
    assert duplicate_item.provenance.source_result_run_id == "reuse-run"
