"""Regression tests for codex_code_review_53f21de.md -- the third
hardening round. Named one group per finding, mirroring the convention
from tests/contract/test_hardening.py and test_hardening_round2.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ft_screening.components import Components, ParsedDocument
from ft_screening.errors import ResumeConflictError
from ft_screening.fakes import FakeCriterionAdjudicator, FakeDocumentParser, FakeEvidenceRetriever
import ft_screening.orchestrator as orchestrator_module
from ft_screening.orchestrator import _run_ft_screening
from ft_screening.schema import (
    CriteriaSet,
    CriterionSpec,
    DocumentCoverage,
    EvidenceSpan,
    ScreeningConfig,
    ScreeningItem,
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


def make_components(output_dir: Path, *, parser=None, retriever=None, adjudicator=None, store=None, **adjudicator_kwargs) -> Components:
    return Components(
        parser=parser or FakeDocumentParser(),
        retriever=retriever or FakeEvidenceRetriever(),
        adjudicator=adjudicator or FakeCriterionAdjudicator(**adjudicator_kwargs),
        store=store or FileScreeningStore(output_dir=output_dir),
    )


# ---------------------------------------------------------------------------
# Finding 1: concurrent ledger writes lose no records -- exact count
# ---------------------------------------------------------------------------


def test_finding1_concurrent_ledger_writes_exact_count_no_loss(tmp_path):
    """Replicates the review's own stress probe: 150 distinct cache
    entries, then an all-hit run at high concurrency. Before the fix this
    lost 1-5 of the 150 expected cache_hit events in 9 of 10 runs."""
    n = 150
    output_dir = tmp_path / "outputs"
    store = FileScreeningStore(output_dir=output_dir)
    criteria_set = make_criteria_set()
    config = make_config(max_workers=32)
    items = [
        ScreeningItem(document_id=f"d{i}", report_id=f"r{i}", pdf_path=make_pdf(tmp_path, f"p{i}.pdf", str(i).encode()))
        for i in range(n)
    ]

    def fresh_components():
        return Components(
            parser=FakeDocumentParser(), retriever=FakeEvidenceRetriever(), adjudicator=FakeCriterionAdjudicator(), store=store
        )

    # Priming run: populates the cache for all n items (n cache misses).
    prime_request = ScreeningRequest(
        items=items, criteria_set=criteria_set, config=config, output_dir=output_dir, run_id="prime-run"
    )
    prime_result = _run_ft_screening(prime_request, fresh_components())
    assert prime_result.run_status == "completed"

    # Real run: every item should be a cache hit -- exactly n cache_hit
    # ledger events expected, written concurrently from Stage B worker threads.
    hit_request = ScreeningRequest(
        items=items, criteria_set=criteria_set, config=config, output_dir=output_dir, run_id="hit-run"
    )
    hit_result = _run_ft_screening(hit_request, fresh_components())
    assert all(i.provenance.cache_status == "hit" for i in hit_result.items)

    _, _, ledger_path = store.artifact_paths(run_id="hit-run")
    events = [json.loads(line) for line in ledger_path.read_text().strip().splitlines()]
    cache_hit_events = [e for e in events if e["type"] == "cache_hit"]
    assert len(cache_hit_events) == n
    assert len({e["document_id"] for e in cache_hit_events}) == n  # all distinct, none duplicated, none lost


# ---------------------------------------------------------------------------
# Finding 2: nested malformed shapes and one-shot iterables
# ---------------------------------------------------------------------------


class _NestedMalformedParser:
    """Simulates a component that bypasses normal field-by-field
    construction (e.g. via model_construct or a duck-typed non-Pydantic
    return) -- a well-typed ParsedDocument whose chunks tuple contains a
    garbage object."""

    def parse(self, *, document_id, document_sha256, pdf_path, parser_options):
        coverage = DocumentCoverage(
            status="complete", total_pages=1, parsed_pages=[1], failed_pages=[], ocr_used=False,
            extracted_characters=10, warnings=[],
        )
        return ParsedDocument.model_construct(
            document_id=document_id, document_sha256=document_sha256, coverage=coverage, chunks=(object(),)
        )


def test_finding2_nested_malformed_chunk_becomes_item_outcome(tmp_path):
    pdf1 = make_pdf(tmp_path, "a.pdf")
    output_dir = tmp_path / "outputs"
    request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1)],
        criteria_set=make_criteria_set(), config=make_config(), output_dir=output_dir,
    )
    result = _run_ft_screening(request, make_components(output_dir, parser=_NestedMalformedParser()))
    assert result.items[0].execution_status == "processing_failed"
    assert result.items[0].errors[0].code == "parse_failed"


class _GeneratorEvidenceRetriever:
    """A retriever returning a one-shot generator instead of a materialized
    list -- must not be silently exhausted by boundary validation before
    the adjudicator gets to use it."""

    def retrieve(self, *, document, criterion, retriever_options):
        def gen():
            yield EvidenceSpan(
                document_id=document.document_id,
                document_sha256=document.document_sha256,
                chunk_id=document.chunks[0].chunk_id,
                excerpt="adult participants",
                stance="supports",
                retrieval_method="fake",
                retrieval_query_version="1",
            )

        return gen()


def test_finding2_generator_evidence_is_not_silently_exhausted(tmp_path):
    pdf1 = make_pdf(tmp_path, "a.pdf")
    output_dir = tmp_path / "outputs"
    request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1)],
        criteria_set=make_criteria_set(), config=make_config(), output_dir=output_dir,
    )
    result = _run_ft_screening(request, make_components(output_dir, retriever=_GeneratorEvidenceRetriever()))
    item = result.items[0]
    assert item.execution_status == "completed"
    total_evidence = sum(len(a.evidence) for a in item.eligibility.assessments)
    assert total_evidence == 1  # not silently 0 -- the generator's one item survived to the adjudicator


# ---------------------------------------------------------------------------
# Finding 3: interruption states -- early (pre-baseline) and resume-preflight
# ---------------------------------------------------------------------------


def test_finding3_interrupt_during_hash_computation_propagates_cleanly(tmp_path, monkeypatch):
    """The review's exact reproduction: interrupting compute_criteria_set_hash()
    itself previously produced an UnboundLocalError that MASKED the
    requested cancellation. Must now propagate KeyboardInterrupt itself,
    and the run lock must still be released."""
    pdf1 = make_pdf(tmp_path, "a.pdf")
    output_dir = tmp_path / "outputs"
    request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1)],
        criteria_set=make_criteria_set(), config=make_config(),
        output_dir=output_dir, run_id="early-interrupt-run",
    )

    def interrupting_compute(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(orchestrator_module, "compute_criteria_set_hash", interrupting_compute)

    with pytest.raises(KeyboardInterrupt):
        _run_ft_screening(request, make_components(output_dir))

    store = FileScreeningStore(output_dir=output_dir)
    assert not store._lock_path("early-interrupt-run").exists()  # lock released, not leaked


def test_finding3_interrupted_resume_preflight_preserves_prior_snapshot(tmp_path, monkeypatch):
    """An interrupt during a RESUME's preflight (before compatibility is
    confirmed / already-completed items are hydrated) must not overwrite
    the prior run's valid run_result.json with a false all-not_processed
    picture."""
    items = [
        ScreeningItem(document_id=f"d{i}", report_id=f"r{i}", pdf_path=make_pdf(tmp_path, f"rp{i}.pdf", str(i).encode()))
        for i in range(3)
    ]
    output_dir = tmp_path / "outputs"
    config = make_config(item_error_policy="fail_fast", max_workers=1)
    criteria_set = make_criteria_set()
    store = FileScreeningStore(output_dir=output_dir)

    # First run: item 1 fails, item 2 never starts (fail_fast).
    req1 = ScreeningRequest(items=items, criteria_set=criteria_set, config=config, output_dir=output_dir, run_id="resume-interrupt-run")
    failing_components = Components(
        parser=FakeDocumentParser(), retriever=FakeEvidenceRetriever(),
        adjudicator=FakeCriterionAdjudicator(fail_keys={("d1", "population")}), store=store,
    )
    result1 = _run_ft_screening(req1, failing_components)
    before_resume = [i.execution_status for i in result1.items]
    assert before_resume == ["completed", "processing_failed", "not_processed"]

    # Second call: resume, but interrupted during preflight hashing --
    # before compatibility is confirmed or already_done is hydrated.
    real_hash_fn = orchestrator_module._hash_file_streaming

    def interrupting_hash(path, **kwargs):
        if "rp1" in str(path):
            raise KeyboardInterrupt()
        return real_hash_fn(path, **kwargs)

    monkeypatch.setattr(orchestrator_module, "_hash_file_streaming", interrupting_hash)

    req2 = ScreeningRequest(items=items, criteria_set=criteria_set, config=config, output_dir=output_dir, resume_run_id="resume-interrupt-run")
    fixed_components = Components(
        parser=FakeDocumentParser(), retriever=FakeEvidenceRetriever(), adjudicator=FakeCriterionAdjudicator(), store=store
    )
    with pytest.raises(KeyboardInterrupt):
        _run_ft_screening(req2, fixed_components)

    # The prior run's snapshot must be untouched -- NOT overwritten with
    # an all-not_processed picture that would lose the real "completed" result.
    run_result_path, _, ledger_path = store.artifact_paths(run_id="resume-interrupt-run")
    after_interrupted_resume = ScreeningRunResult.model_validate_json(run_result_path.read_text())
    assert [i.execution_status for i in after_interrupted_resume.items] == before_resume

    ledger_events = [json.loads(line) for line in ledger_path.read_text().strip().splitlines()]
    assert any(e["type"] == "run_interrupted" and e.get("phase") == "resume_preflight" for e in ledger_events)


def test_finding3_resume_after_interrupted_resume_still_completes(tmp_path, monkeypatch):
    """Once the transient hashing failure is gone, a subsequent (real)
    resume must still succeed and finish the run."""
    items = [
        ScreeningItem(document_id=f"d{i}", report_id=f"r{i}", pdf_path=make_pdf(tmp_path, f"rr{i}.pdf", str(i).encode()))
        for i in range(3)
    ]
    output_dir = tmp_path / "outputs"
    config = make_config(item_error_policy="fail_fast", max_workers=1)
    criteria_set = make_criteria_set()
    store = FileScreeningStore(output_dir=output_dir)

    req1 = ScreeningRequest(items=items, criteria_set=criteria_set, config=config, output_dir=output_dir, run_id="resume-recover-run")
    failing_components = Components(
        parser=FakeDocumentParser(), retriever=FakeEvidenceRetriever(),
        adjudicator=FakeCriterionAdjudicator(fail_keys={("d1", "population")}), store=store,
    )
    _run_ft_screening(req1, failing_components)

    fixed_components = Components(
        parser=FakeDocumentParser(), retriever=FakeEvidenceRetriever(), adjudicator=FakeCriterionAdjudicator(), store=store
    )
    req2 = ScreeningRequest(items=items, criteria_set=criteria_set, config=config, output_dir=output_dir, resume_run_id="resume-recover-run")
    result2 = _run_ft_screening(req2, fixed_components)
    assert result2.run_status == "completed"
    assert [i.execution_status for i in result2.items] == ["completed", "completed", "completed"]


# ---------------------------------------------------------------------------
# Finding 4: malformed cache provenance is a clean miss, never aborts the run
# ---------------------------------------------------------------------------


def test_finding4_cache_entry_with_non_string_source_run_id_is_a_miss(tmp_path):
    pdf1 = make_pdf(tmp_path, "a.pdf")
    output_dir = tmp_path / "outputs"
    store = FileScreeningStore(output_dir=output_dir)
    criteria_set = make_criteria_set()
    config = make_config()

    # Prime the cache normally, then corrupt the envelope's source_run_id
    # to a JSON list -- syntactically valid JSON, wrong shape.
    req1 = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1)],
        criteria_set=criteria_set, config=config, output_dir=output_dir, run_id="prime-run",
    )
    _run_ft_screening(req1, Components(parser=FakeDocumentParser(), retriever=FakeEvidenceRetriever(), adjudicator=FakeCriterionAdjudicator(), store=store))

    cache_dir = output_dir / "cache" / "v1"
    cache_files = list(cache_dir.glob("*.json"))
    assert len(cache_files) == 1
    envelope = json.loads(cache_files[0].read_text())
    envelope["source_run_id"] = ["not", "a", "string"]  # malformed provenance
    cache_files[0].write_text(json.dumps(envelope))

    # A second run hitting this cache entry must not crash -- must treat
    # the malformed envelope as a miss and recompute cleanly.
    pdf2 = make_pdf(tmp_path, "a2.pdf", pdf1.read_bytes())
    req2 = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1)],
        criteria_set=criteria_set, config=config, output_dir=output_dir, run_id="second-run",
    )
    result2 = _run_ft_screening(
        req2, Components(parser=FakeDocumentParser(), retriever=FakeEvidenceRetriever(), adjudicator=FakeCriterionAdjudicator(), store=store)
    )
    assert result2.run_status == "completed"  # did not abort with an unhashable-type TypeError
    assert result2.items[0].provenance.cache_status == "miss"  # malformed envelope correctly treated as a miss


def test_finding4_empty_source_run_id_rejected_by_envelope_validation(tmp_path):
    output_dir = tmp_path / "outputs"
    store = FileScreeningStore(output_dir=output_dir)
    from ft_screening.schema import CriterionAssessment

    assessment = CriterionAssessment(
        criterion_id="population", criterion_version="1", status="satisfied", evidence=[],
        rationale="ok", requires_human_review=False, model="fake-1",
        prompt_template_version="1", output_schema_version="1",
    )
    with pytest.raises(Exception):
        store.put_cached_assessment(cache_key="a" * 64, assessment=assessment, source_run_id="")
