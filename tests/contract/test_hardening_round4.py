"""Regression tests for codex_code_review_0f48a3d.md -- the fourth
hardening round. Named one group per finding, mirroring the convention
from tests/contract/test_hardening*.py.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from ft_screening.components import CachedAssessment, Components
from ft_screening.errors import ResumeConflictError
from ft_screening.fakes import FakeCriterionAdjudicator, FakeDocumentParser, FakeEvidenceRetriever
import ft_screening.orchestrator as orchestrator_module
from ft_screening.orchestrator import _run_ft_screening
from ft_screening.schema import (
    CriteriaSet,
    CriterionAssessment,
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


def make_components(output_dir: Path, *, parser=None, retriever=None, adjudicator=None, store=None, **adjudicator_kwargs) -> Components:
    return Components(
        parser=parser or FakeDocumentParser(),
        retriever=retriever or FakeEvidenceRetriever(),
        adjudicator=adjudicator or FakeCriterionAdjudicator(**adjudicator_kwargs),
        store=store or FileScreeningStore(output_dir=output_dir),
    )


# ---------------------------------------------------------------------------
# Finding 1: early interrupt must not leave a permanently unretryable run
# ---------------------------------------------------------------------------


def test_finding1_early_interrupt_leaves_no_run_directory(tmp_path, monkeypatch):
    """criteria_set_hash/execution_fingerprint are now computed BEFORE
    begin_run() -- interrupting that computation must leave no run
    directory at all, not an empty unretryable one."""
    pdf1 = make_pdf(tmp_path, "a.pdf")
    output_dir = tmp_path / "outputs"
    request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1)],
        criteria_set=make_criteria_set(), config=make_config(),
        output_dir=output_dir, run_id="early-artifacts-run",
    )

    def interrupting_compute(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(orchestrator_module, "compute_criteria_set_hash", interrupting_compute)
    with pytest.raises(KeyboardInterrupt):
        _run_ft_screening(request, make_components(output_dir))

    run_dir = output_dir / "runs" / "early-artifacts-run"
    assert not run_dir.exists()  # nothing was ever created -- no artifacts to strand


def test_finding1_early_interrupt_allows_clean_retry(tmp_path, monkeypatch):
    pdf1 = make_pdf(tmp_path, "a.pdf")
    output_dir = tmp_path / "outputs"
    request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1)],
        criteria_set=make_criteria_set(), config=make_config(),
        output_dir=output_dir, run_id="early-retry-run",
    )

    real_compute = orchestrator_module.compute_criteria_set_hash
    call_count = {"n": 0}

    def flaky_compute(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise KeyboardInterrupt()
        return real_compute(*args, **kwargs)

    monkeypatch.setattr(orchestrator_module, "compute_criteria_set_hash", flaky_compute)

    with pytest.raises(KeyboardInterrupt):
        _run_ft_screening(request, make_components(output_dir))

    # Retry with the SAME run_id and the SAME (now-shared) store instance --
    # must succeed cleanly, not RunSetupError from a stranded directory.
    result = _run_ft_screening(request, make_components(output_dir))
    assert result.run_status == "completed"


# ---------------------------------------------------------------------------
# Finding 2: crash-safe locking -- a real subprocess crash releases the lock
# ---------------------------------------------------------------------------


def test_finding2_subprocess_crash_releases_lock_for_later_resume(tmp_path):
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True)
    helper = Path(__file__).parent / "_crash_helper.py"

    proc = subprocess.run(
        [sys.executable, str(helper), str(output_dir), "crash-run"],
        capture_output=True, text=True, timeout=30,
    )
    assert "locked" in proc.stdout  # child confirmed it acquired the lock before crashing

    # A brand new store instance (simulating a fresh process/retry attempt)
    # must be able to acquire the lock the crashed process never released --
    # the OS released it automatically when the crashed process's file
    # descriptors closed. codex_code_review_810ccf0.md finding 2: the child
    # (via "lock_only" mode) only ever calls begin_run() and crashes --
    # never writes a manifest -- so nothing was ever durably claimed, and
    # the correct recovery here is a FRESH claim (resume=False), not a
    # resume; a resume attempt now correctly raises ResumeConflictError
    # ("nothing durable was ever recorded"), which is exactly finding 2's
    # point: a bare directory/lock with no manifest must never be
    # resumable. See test_hardening_round5.py's
    # test_finding1_subprocess_crash_mid_hashing_is_semantically_resumable
    # for the stronger case where a real manifest DOES exist before the
    # crash, and resume_run_id genuinely recovers the run.
    recovery_store = FileScreeningStore(output_dir=output_dir)
    recovery_store.begin_run(run_id="crash-run", resume=False)  # must not raise
    recovery_store.end_run(run_id="crash-run")


def test_finding2_live_process_still_denies_concurrent_access(tmp_path):
    """The crash-safe lock must still correctly DENY access while the
    holder is genuinely alive -- only a dead holder's lock is recoverable."""
    output_dir = tmp_path / "outputs"
    store_a = FileScreeningStore(output_dir=output_dir)
    store_a.begin_run(run_id="live-run", resume=False)
    try:
        store_b = FileScreeningStore(output_dir=output_dir)
        with pytest.raises(ResumeConflictError):
            store_b.begin_run(run_id="live-run", resume=True)
    finally:
        store_a.end_run(run_id="live-run")


def test_finding2_lock_file_no_longer_the_signal_itself(tmp_path):
    """The .lock file existing on disk must not by itself mean 'locked' --
    only the OS-level advisory lock does. Simulates the exact prior bug: a
    .lock file left behind with no process actually holding the OS lock."""
    output_dir = tmp_path / "outputs"
    store = FileScreeningStore(output_dir=output_dir)
    store.begin_run(run_id="stale-file-run", resume=False)
    store.end_run(run_id="stale-file-run")  # releases the OS lock; .lock file itself remains on disk

    lock_path = store._lock_path("stale-file-run")
    assert lock_path.exists()  # the file is still there...

    # ...but re-acquiring must succeed anyway, since nothing holds the OS
    # lock. Fresh (not resume): this test's begin_run()/end_run() pair
    # never wrote a manifest, so under codex_code_review_810ccf0.md finding
    # 2's stricter "claimed means a manifest exists" rule, nothing was ever
    # claimed -- a second FRESH begin_run() for the same run_id is the
    # correct probe for "the lock FILE's mere existence doesn't block
    # re-acquisition", which is what this test exists to prove.
    store2 = FileScreeningStore(output_dir=output_dir)
    store2.begin_run(run_id="stale-file-run", resume=False)
    store2.end_run(run_id="stale-file-run")


# ---------------------------------------------------------------------------
# Finding 3: cache validation at the protocol boundary, not just FileScreeningStore
# ---------------------------------------------------------------------------


class _InMemoryStore:
    """A minimal, deliberately non-FileScreeningStore ScreeningStore
    implementation -- proves the orchestrator's cache-boundary validation
    doesn't rely on FileScreeningStore's own _CacheEnvelope, since a
    different, protocol-conforming implementation won't have that."""

    def __init__(self, *, output_dir: Path, bad_cache_key: str | None = None, raise_on_get: bool = False) -> None:
        self._output_dir = Path(output_dir)
        self._manifests: dict[str, dict[str, str]] = {}
        self._items: dict[str, dict[int, ScreeningItemResult]] = {}
        self._run_results: dict[str, ScreeningRunResult] = {}
        self.ledger: list[dict[str, Any]] = []
        self._bad_cache_key = bad_cache_key
        self._raise_on_get = raise_on_get

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    def begin_run(self, *, run_id: str, resume: bool) -> None:
        pass

    def end_run(self, *, run_id: str) -> None:
        pass

    def write_run_manifest(
        self,
        *,
        run_id,
        execution_fingerprint,
        request_fingerprint,
        provisional_request_binding=None,
        document_hash_checkpoints=None,
    ) -> None:
        self._manifests[run_id] = {
            "execution_fingerprint": execution_fingerprint,
            "request_fingerprint": request_fingerprint,
            "provisional_request_binding": provisional_request_binding,
            "document_hash_checkpoints": document_hash_checkpoints or {},
        }

    def read_run_manifest(self, *, run_id: str):
        return self._manifests.get(run_id)

    def append_ledger_event(self, *, run_id: str, event: dict) -> None:
        self.ledger.append(event)

    def read_completed_item_results(self, *, run_id: str) -> dict[int, ScreeningItemResult]:
        return dict(self._items.get(run_id, {}))

    def write_item_result(self, *, run_id: str, item_result: ScreeningItemResult) -> None:
        self._items.setdefault(run_id, {})[item_result.input_index] = item_result

    def write_run_result(self, *, run_id: str, run_result: ScreeningRunResult) -> None:
        self._run_results[run_id] = run_result

    def artifact_paths(self, *, run_id: str):
        run_dir = self._output_dir / "runs" / run_id
        return (run_dir / "run_result.json", run_dir / "item_results.jsonl", run_dir / "ledger.jsonl")

    def get_cached_assessment(self, *, cache_key: str) -> CachedAssessment | None:
        if self._raise_on_get:
            raise ConnectionError("simulated store I/O failure")
        if cache_key == self._bad_cache_key:
            assessment = CriterionAssessment(
                criterion_id="population", criterion_version="1", status="satisfied", evidence=[],
                rationale="ok", requires_human_review=False, model="fake-1",
                prompt_template_version="1", output_schema_version="1",
            )
            # Protocol-conforming (returns a CachedAssessment) but
            # malformed (bypasses CachedAssessment's own field validation).
            return CachedAssessment.model_construct(assessment=assessment, source_run_id=["bad", "shape"])
        return None

    def put_cached_assessment(self, *, cache_key: str, assessment, source_run_id: str) -> None:
        pass


def test_finding3_alternate_store_malformed_cache_hit_is_a_miss_not_a_crash(tmp_path):
    pdf1 = make_pdf(tmp_path, "a.pdf")
    output_dir = tmp_path / "outputs"
    from ft_screening.hashing import compute_criterion_cache_key

    criteria_set = make_criteria_set()
    config = make_config()
    # The cache key's document_sha256 must match the REAL hash of pdf1's
    # bytes so the orchestrator actually looks up this exact key.
    import hashlib

    real_hash = hashlib.sha256(pdf1.read_bytes()).hexdigest()
    bad_key = compute_criterion_cache_key(document_sha256=real_hash, criterion=criteria_set.criteria[0], config=config)

    store = _InMemoryStore(output_dir=output_dir, bad_cache_key=bad_key)
    request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1)],
        criteria_set=criteria_set, config=config, output_dir=output_dir,
    )
    result = _run_ft_screening(request, make_components(output_dir, store=store))
    item = result.items[0]
    assert item.execution_status == "completed"  # did not crash on the malformed cache hit
    assert item.provenance.cache_status == "miss"  # treated as a miss, recomputed fresh


def test_finding3_alternate_store_get_cache_failure_becomes_item_outcome(tmp_path):
    """codex_code_review_c4f274a.md finding 4 superseded this test's
    original policy: a cache-READ failure is a store/persistence problem,
    not a scientific evidence-retrieval problem, and no retriever ever ran
    -- reporting it as this item's processing_failed/retrieval_failed
    outcome was itself the bug. Cache reads are optional: a store failure
    here must degrade to a recomputed miss (item still completes) plus a
    sanitized persistence_failed warning and a cache_read_failed ledger
    event, never the item's terminal outcome."""
    pdf1 = make_pdf(tmp_path, "a.pdf")
    output_dir = tmp_path / "outputs"
    store = _InMemoryStore(output_dir=output_dir, raise_on_get=True)
    request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1)],
        criteria_set=make_criteria_set(), config=make_config(), output_dir=output_dir,
    )
    result = _run_ft_screening(request, make_components(output_dir, store=store))
    item = result.items[0]
    assert item.execution_status == "completed"
    assert item.provenance.cache_status == "miss"
    assert any(e.code == "persistence_failed" and e.severity == "warning" for e in item.errors)
    assert any(event["type"] == "cache_read_failed" for event in store.ledger)


# ---------------------------------------------------------------------------
# Finding 4: chunk identity -- uniqueness and span ordering
# ---------------------------------------------------------------------------


def test_finding4_chunk_rejects_reversed_span():
    from ft_screening.components import Chunk

    with pytest.raises(Exception):
        Chunk(chunk_id="c1", section=None, page=1, start_char=10, end_char=5, text="x")


def test_finding4_parsed_document_rejects_duplicate_chunk_ids():
    from ft_screening.components import Chunk, ParsedDocument
    from ft_screening.schema import DocumentCoverage

    coverage = DocumentCoverage(
        status="complete", total_pages=1, parsed_pages=[1], failed_pages=[], ocr_used=False,
        extracted_characters=20, warnings=[],
    )
    chunk_a = Chunk(chunk_id="dup", section="Methods", page=1, start_char=0, end_char=5, text="first")
    chunk_b = Chunk(chunk_id="dup", section="Results", page=1, start_char=10, end_char=15, text="second")
    with pytest.raises(Exception):
        ParsedDocument(document_id="d1", document_sha256="a" * 64, coverage=coverage, chunks=(chunk_a, chunk_b))


def test_finding4_duplicate_chunk_ids_rejected_through_full_orchestration(tmp_path):
    from ft_screening.components import Chunk, ParsedDocument
    from ft_screening.schema import DocumentCoverage

    pdf1 = make_pdf(tmp_path, "a.pdf")
    output_dir = tmp_path / "outputs"

    class DuplicateChunkParser:
        def parse(self, *, document_id, document_sha256, pdf_path, parser_options):
            coverage = DocumentCoverage(
                status="complete", total_pages=1, parsed_pages=[1], failed_pages=[], ocr_used=False,
                extracted_characters=20, warnings=[],
            )
            # model_construct bypasses ParsedDocument's own _unique_chunk_ids
            # validator (simulating a parser that doesn't go through normal
            # construction) -- the orchestrator's own revalidation must still
            # catch this, per revalidate_instances="always".
            return ParsedDocument.model_construct(
                document_id=document_id,
                document_sha256=document_sha256,
                coverage=coverage,
                chunks=(
                    Chunk(chunk_id="same", section="Methods", page=1, start_char=0, end_char=5, text="a"),
                    Chunk(chunk_id="same", section="Results", page=1, start_char=10, end_char=15, text="b"),
                ),
            )

    request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1)],
        criteria_set=make_criteria_set(), config=make_config(), output_dir=output_dir,
    )
    result = _run_ft_screening(request, make_components(output_dir, parser=DuplicateChunkParser()))
    assert result.items[0].execution_status == "processing_failed"
    assert result.items[0].errors[0].code == "parse_failed"
