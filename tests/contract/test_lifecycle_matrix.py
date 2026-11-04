"""Table-driven fresh/resume lifecycle matrix (codex_code_review_810ccf0.md
finding 4: "a lifecycle state table covering fresh/resume x pre-lock,
locked/uninitialized, provisional, finalized, items-partial, and terminal
states" was explicitly required and had not been implemented).

Every prior hardening round (test_hardening*.py) added ad hoc,
finding-specific interrupt tests. This file instead drives ONE shared
mechanism (`_fault_injection.Interruptor`) across every state transition a
run passes through, so a future durable-write point added anywhere in the
lifecycle can be covered by adding one row here rather than inventing a
new bespoke monkeypatch.

State table (each row: where a KeyboardInterrupt lands, and what a
subsequent fresh retry / resume_run_id attempt must do):

| # | Hook point                        | Fresh retry (same run_id) | resume_run_id            |
|---|------------------------------------|----------------------------|---------------------------|
| 1 | pre_lock (before begin_run)        | succeeds cleanly           | ResumeConflictError       |
| 2 | post_lock_pre_manifest              | succeeds cleanly           | ResumeConflictError       |
| 3 | mid_hashing (provisional exists)    | RunSetupError (collision)  | succeeds, run completes   |
| 4 | post_hash_pre_finalize              | RunSetupError (collision)  | succeeds, run completes   |
| 5 | first_item_write (finalized, Stage B)| RunSetupError (collision) | succeeds, run completes   |
| 6 | final_result_write (all items done) | RunSetupError (collision)  | succeeds, run completes   |

Real cross-process crash injection at rows 2-6's analogues (not just an
in-process Python exception) lives in `_crash_helper.py` /
test_hardening_round5.py / test_hardening_round6.py ("mid_hashing",
"pre_manifest", "post_item_write", "post_final_result" subprocess modes)
-- this file complements those with the FULL state table using the
cheaper, faster in-process mechanism, and a subprocess test is not
re-added per row here to avoid redundant slow process-spawning.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from _fault_injection import Interruptor
from ft_screening.components import Components
import ft_screening.orchestrator as orchestrator_module
from ft_screening.errors import ResumeConflictError, RunSetupError
from ft_screening.orchestrator import _run_ft_screening
from ft_screening.schema import (
    CriteriaSet,
    CriterionSpec,
    ScreeningConfig,
    ScreeningItem,
    ScreeningRequest,
    ScreeningRunResult,
)
from ft_screening.store import FileScreeningStore

from ft_screening.fakes import FakeCriterionAdjudicator, FakeDocumentParser, FakeEvidenceRetriever


def make_pdf(tmp_path: Path, name: str, content: bytes) -> Path:
    path = tmp_path / "pdfs" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def make_request(tmp_path: Path, output_dir: Path, run_id: str) -> ScreeningRequest:
    pdf1 = make_pdf(tmp_path, "a.pdf", b"content-a")
    pdf2 = make_pdf(tmp_path, "b.pdf", b"content-b")
    criteria_set = CriteriaSet(
        criteria_set_id="cs-1", version="1",
        criteria=[
            CriterionSpec(
                criterion_id="population", version="1", eligibility_requirement="adult humans",
                missing_evidence_policy="review", exclusion_reason_code="wrong_population",
                exclusion_reason_label="Wrong population", priority=1,
            )
        ],
    )
    config = ScreeningConfig(
        parser_name="fake", parser_version="1", chunker_version="1", retriever_name="fake",
        retriever_config_version="1", adjudicator_provider="fake", adjudicator_model="fake-1",
        model_config_id="cfg-1", prompt_template_version="1", output_schema_version="1",
        aggregation_policy_version="1",
    )
    items = [
        ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1),
        ScreeningItem(document_id="d2", report_id="r2", pdf_path=pdf2),
    ]
    return ScreeningRequest(items=items, criteria_set=criteria_set, config=config, output_dir=output_dir, run_id=run_id)


def make_components(output_dir: Path, store: FileScreeningStore) -> Components:
    return Components(
        parser=FakeDocumentParser(), retriever=FakeEvidenceRetriever(), adjudicator=FakeCriterionAdjudicator(),
        store=store,
    )


def _resume_request(fresh: ScreeningRequest, run_id: str) -> ScreeningRequest:
    return ScreeningRequest(
        items=fresh.items, criteria_set=fresh.criteria_set, config=fresh.config,
        output_dir=fresh.output_dir, resume_run_id=run_id,
    )


def _fresh_retry(fresh: ScreeningRequest) -> ScreeningRequest:
    return ScreeningRequest(
        items=fresh.items, criteria_set=fresh.criteria_set, config=fresh.config,
        output_dir=fresh.output_dir, run_id=fresh.run_id,
    )


# --- Row 1: pre_lock ---------------------------------------------------


def test_lifecycle_pre_lock_fresh_retry_succeeds(tmp_path, monkeypatch):
    output_dir = tmp_path / "outputs"
    run_id = "row1-run"
    request = make_request(tmp_path, output_dir, run_id)
    store = FileScreeningStore(output_dir=output_dir)
    Interruptor(orchestrator_module, "compute_criteria_set_hash", call_number=1).install(monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        _run_ft_screening(request, make_components(output_dir, store))
    monkeypatch.undo()

    assert not (output_dir / "runs" / run_id).exists()
    result = _run_ft_screening(_fresh_retry(request), make_components(output_dir, store))
    assert result.run_status == "completed"


def test_lifecycle_pre_lock_resume_rejected(tmp_path, monkeypatch):
    output_dir = tmp_path / "outputs"
    run_id = "row1b-run"
    request = make_request(tmp_path, output_dir, run_id)
    store = FileScreeningStore(output_dir=output_dir)
    Interruptor(orchestrator_module, "compute_criteria_set_hash", call_number=1).install(monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        _run_ft_screening(request, make_components(output_dir, store))
    monkeypatch.undo()

    with pytest.raises(ResumeConflictError):
        _run_ft_screening(_resume_request(request, run_id), make_components(output_dir, store))


# --- Row 2: post_lock_pre_manifest --------------------------------------


def test_lifecycle_post_lock_pre_manifest_fresh_retry_succeeds(tmp_path, monkeypatch):
    output_dir = tmp_path / "outputs"
    run_id = "row2-run"
    request = make_request(tmp_path, output_dir, run_id)
    store = FileScreeningStore(output_dir=output_dir)
    Interruptor(store, "write_run_manifest", call_number=1).install(monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        _run_ft_screening(request, make_components(output_dir, store))
    monkeypatch.undo()

    assert not (output_dir / "runs" / run_id).exists()
    result = _run_ft_screening(_fresh_retry(request), make_components(output_dir, store))
    assert result.run_status == "completed"


def test_lifecycle_post_lock_pre_manifest_resume_rejected(tmp_path, monkeypatch):
    output_dir = tmp_path / "outputs"
    run_id = "row2b-run"
    request = make_request(tmp_path, output_dir, run_id)
    store = FileScreeningStore(output_dir=output_dir)
    Interruptor(store, "write_run_manifest", call_number=1).install(monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        _run_ft_screening(request, make_components(output_dir, store))
    monkeypatch.undo()

    with pytest.raises(ResumeConflictError):
        _run_ft_screening(_resume_request(request, run_id), make_components(output_dir, store))


# --- Row 3: mid_hashing (provisional manifest already committed) -------


def test_lifecycle_mid_hashing_fresh_retry_collides(tmp_path, monkeypatch):
    output_dir = tmp_path / "outputs"
    run_id = "row3-run"
    request = make_request(tmp_path, output_dir, run_id)
    store = FileScreeningStore(output_dir=output_dir)

    real_hash_fn = orchestrator_module._hash_file_streaming

    def interrupting_hash(path, **kwargs):
        if "b.pdf" in str(path):
            raise KeyboardInterrupt()
        return real_hash_fn(path, **kwargs)

    monkeypatch.setattr(orchestrator_module, "_hash_file_streaming", interrupting_hash)
    with pytest.raises(KeyboardInterrupt):
        _run_ft_screening(request, make_components(output_dir, store))
    monkeypatch.setattr(orchestrator_module, "_hash_file_streaming", real_hash_fn)

    assert (output_dir / "runs" / run_id / "manifest.json").exists()
    with pytest.raises(RunSetupError):
        _run_ft_screening(_fresh_retry(request), make_components(output_dir, store))


def test_lifecycle_mid_hashing_resume_completes(tmp_path, monkeypatch):
    output_dir = tmp_path / "outputs"
    run_id = "row3b-run"
    request = make_request(tmp_path, output_dir, run_id)
    store = FileScreeningStore(output_dir=output_dir)

    real_hash_fn = orchestrator_module._hash_file_streaming

    def interrupting_hash(path, **kwargs):
        if "b.pdf" in str(path):
            raise KeyboardInterrupt()
        return real_hash_fn(path, **kwargs)

    monkeypatch.setattr(orchestrator_module, "_hash_file_streaming", interrupting_hash)
    with pytest.raises(KeyboardInterrupt):
        _run_ft_screening(request, make_components(output_dir, store))
    monkeypatch.setattr(orchestrator_module, "_hash_file_streaming", real_hash_fn)

    result = _run_ft_screening(_resume_request(request, run_id), make_components(output_dir, store))
    assert result.run_status == "completed"


# --- Row 4: post_hash_pre_finalize (2nd write_run_manifest call) -------


def test_lifecycle_post_hash_pre_finalize_fresh_retry_collides(tmp_path, monkeypatch):
    output_dir = tmp_path / "outputs"
    run_id = "row4-run"
    request = make_request(tmp_path, output_dir, run_id)
    store = FileScreeningStore(output_dir=output_dir)
    Interruptor(store, "write_run_manifest", call_number=2).install(monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        _run_ft_screening(request, make_components(output_dir, store))
    monkeypatch.undo()

    with pytest.raises(RunSetupError):
        _run_ft_screening(_fresh_retry(request), make_components(output_dir, store))


def test_lifecycle_post_hash_pre_finalize_resume_completes(tmp_path, monkeypatch):
    output_dir = tmp_path / "outputs"
    run_id = "row4b-run"
    request = make_request(tmp_path, output_dir, run_id)
    store = FileScreeningStore(output_dir=output_dir)
    Interruptor(store, "write_run_manifest", call_number=2).install(monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        _run_ft_screening(request, make_components(output_dir, store))
    monkeypatch.undo()

    result = _run_ft_screening(_resume_request(request, run_id), make_components(output_dir, store))
    assert result.run_status == "completed"


# --- Row 5: first_item_write (manifest finalized, Stage B in progress) --


def test_lifecycle_first_item_write_fresh_retry_collides(tmp_path, monkeypatch):
    output_dir = tmp_path / "outputs"
    run_id = "row5-run"
    request = make_request(tmp_path, output_dir, run_id)
    store = FileScreeningStore(output_dir=output_dir)
    Interruptor(store, "write_item_result", call_number=1).install(monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        _run_ft_screening(request, make_components(output_dir, store))
    monkeypatch.undo()

    with pytest.raises(RunSetupError):
        _run_ft_screening(_fresh_retry(request), make_components(output_dir, store))


def test_lifecycle_first_item_write_resume_completes(tmp_path, monkeypatch):
    output_dir = tmp_path / "outputs"
    run_id = "row5b-run"
    request = make_request(tmp_path, output_dir, run_id)
    store = FileScreeningStore(output_dir=output_dir)
    Interruptor(store, "write_item_result", call_number=1).install(monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        _run_ft_screening(request, make_components(output_dir, store))
    monkeypatch.undo()

    result = _run_ft_screening(_resume_request(request, run_id), make_components(output_dir, store))
    assert result.run_status == "completed"
    assert all(i.execution_status == "completed" for i in result.items)


# --- Row 6: final_result_write (every item already done) ---------------


def test_lifecycle_final_result_write_fresh_retry_collides(tmp_path, monkeypatch):
    output_dir = tmp_path / "outputs"
    run_id = "row6-run"
    request = make_request(tmp_path, output_dir, run_id)
    store = FileScreeningStore(output_dir=output_dir)
    Interruptor(store, "write_run_result", call_number=1).install(monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        _run_ft_screening(request, make_components(output_dir, store))
    monkeypatch.undo()

    with pytest.raises(RunSetupError):
        _run_ft_screening(_fresh_retry(request), make_components(output_dir, store))


def test_lifecycle_final_result_write_resume_completes(tmp_path, monkeypatch):
    output_dir = tmp_path / "outputs"
    run_id = "row6b-run"
    request = make_request(tmp_path, output_dir, run_id)
    store = FileScreeningStore(output_dir=output_dir)
    Interruptor(store, "write_run_result", call_number=1).install(monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        _run_ft_screening(request, make_components(output_dir, store))
    monkeypatch.undo()

    # Every item had already completed in memory before the interrupted
    # write; resuming re-hydrates all of them as already-done and produces
    # a genuinely completed result.
    result = _run_ft_screening(_resume_request(request, run_id), make_components(output_dir, store))
    assert result.run_status == "completed"
    assert all(i.execution_status == "completed" for i in result.items)


# --- Rows 5/6, real subprocess crash (not just an in-process exception) ---
#
# The in-process rows above prove the ORCHESTRATOR's own logic handles an
# interrupt correctly; these two prove the same transitions survive an
# actual process death (os._exit -- no Python exception handling runs at
# all), which is what a real crash looks like. Uses the same
# _crash_helper.py subprocess used by test_hardening_round5.py/round6.py.


def test_lifecycle_subprocess_crash_after_first_item_write_is_resumable(tmp_path):
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True)
    helper = Path(__file__).parent / "_crash_helper.py"

    proc = subprocess.run(
        [sys.executable, str(helper), str(output_dir), "row5-subprocess-run", "post_item_write"],
        capture_output=True, text=True, timeout=30,
    )
    assert "crashing" in proc.stdout

    pdf1 = output_dir / "pdfs" / "a.pdf"
    pdf2 = output_dir / "pdfs" / "b.pdf"
    criteria_set = CriteriaSet(
        criteria_set_id="cs", version="1",
        criteria=[
            CriterionSpec(
                criterion_id="population", version="1", eligibility_requirement="adults",
                missing_evidence_policy="review", exclusion_reason_code="wrong_population",
                exclusion_reason_label="Wrong population", priority=1,
            )
        ],
    )
    config = ScreeningConfig(
        parser_name="fake", parser_version="1", chunker_version="1", retriever_name="fake",
        retriever_config_version="1", adjudicator_provider="fake", adjudicator_model="fake-1",
        model_config_id="cfg-1", prompt_template_version="1", output_schema_version="1",
        aggregation_policy_version="1",
    )
    items = [
        ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf1),
        ScreeningItem(document_id="d2", report_id="r2", pdf_path=pdf2),
    ]
    store = FileScreeningStore(output_dir=output_dir)
    resume_request = ScreeningRequest(
        items=items, criteria_set=criteria_set, config=config, output_dir=output_dir,
        resume_run_id="row5-subprocess-run",
    )
    result = _run_ft_screening(resume_request, make_components(output_dir, store))
    assert result.run_status == "completed"
    assert all(i.execution_status == "completed" for i in result.items)


def test_lifecycle_subprocess_crash_after_final_result_write_leaves_readable_result(tmp_path):
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True)
    helper = Path(__file__).parent / "_crash_helper.py"

    proc = subprocess.run(
        [sys.executable, str(helper), str(output_dir), "row6-subprocess-run", "post_final_result"],
        capture_output=True, text=True, timeout=30,
    )
    assert "crashing" in proc.stdout

    store = FileScreeningStore(output_dir=output_dir)
    run_result_path, _, _ = store.artifact_paths(run_id="row6-subprocess-run")
    assert run_result_path.exists()

    snapshot = ScreeningRunResult.model_validate_json(run_result_path.read_text())
    assert all(i.execution_status == "completed" for i in snapshot.items)

    # The lock must still be cleanly reacquirable -- the crash happened
    # after the OS already owned releasing it on process exit.
    store.begin_run(run_id="row6-subprocess-run", resume=True)
    store.end_run(run_id="row6-subprocess-run")
