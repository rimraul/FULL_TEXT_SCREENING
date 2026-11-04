"""Standalone script run as a SEPARATE PROCESS by
tests/contract/test_hardening_round4.py, test_hardening_round5.py,
test_hardening_round6.py, and test_lifecycle_matrix.py. A real subprocess
is required: threads within one process share OS-level lock ownership, so
a same-process simulation would not actually exercise cross-process crash
recovery.

Modes, selected by argv[3]:

  "lock_only" -- acquire the run lock, print "locked", then os._exit()
    immediately. Proves lock reacquisition after a crash (round 4).

  "mid_hashing" -- run a REAL _run_ft_screening() call against a
  multi-item request, crashing (os._exit()) partway through Stage A1 file
  hashing -- after the manifest write finding 1 (round 5) makes
  provisional, but before the run itself completes. Proves the run is
  actually RESUMABLE afterward (not just that the lock can be
  reacquired), which is the distinction round 5's finding turned on.

  "pre_manifest" -- crash (os._exit()) inside the FIRST
  store.write_run_manifest() call, before it returns -- i.e. before
  anything durable is claimed at all (codex_code_review_810ccf0.md
  finding 2). Proves a real process death at this exact point leaves
  nothing stranded, the same way an in-process KeyboardInterrupt does.

  "post_item_write" -- crash right after the FIRST store.write_item_result()
  call returns, before the run completes. Proves an item persisted just
  before a crash survives and is correctly hydrated on resume (does not
  get reprocessed, does not go missing).

  "post_final_result" -- crash right after store.write_run_result() (the
  terminal run_result.json) returns, before the "run_completed" ledger
  event / end_run(). Proves a crash in this narrow tail window still
  leaves a durable, readable final result and a cleanly reacquirable lock.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def _two_item_request(output_dir_path: Path, run_id: str):
    from ft_screening.schema import CriteriaSet, CriterionSpec, ScreeningConfig, ScreeningItem, ScreeningRequest

    pdf1 = output_dir_path / "pdfs" / "a.pdf"
    pdf2 = output_dir_path / "pdfs" / "b.pdf"
    pdf1.parent.mkdir(parents=True, exist_ok=True)
    pdf1.write_bytes(b"content-a")
    pdf2.write_bytes(b"content-b")

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
    return ScreeningRequest(items=items, criteria_set=criteria_set, config=config, output_dir=output_dir_path, run_id=run_id)


def _components(output_dir_path: Path):
    from ft_screening.components import Components
    from ft_screening.fakes import FakeCriterionAdjudicator, FakeDocumentParser, FakeEvidenceRetriever
    from ft_screening.store import FileScreeningStore

    return Components(
        parser=FakeDocumentParser(), retriever=FakeEvidenceRetriever(), adjudicator=FakeCriterionAdjudicator(),
        store=FileScreeningStore(output_dir=output_dir_path),
    )


def main() -> None:
    output_dir = sys.argv[1]
    run_id = sys.argv[2]
    mode = sys.argv[3] if len(sys.argv) > 3 else "lock_only"

    from ft_screening.store import FileScreeningStore

    if mode == "lock_only":
        store = FileScreeningStore(output_dir=output_dir)
        store.begin_run(run_id=run_id, resume=False)
        sys.stdout.write("locked\n")
        sys.stdout.flush()
        os._exit(1)  # crash -- no end_run(), no cleanup, no atexit
        return

    if mode == "mid_hashing":
        import ft_screening.orchestrator as orchestrator_module

        real_hash_fn = orchestrator_module._hash_file_streaming
        state = {"count": 0}

        def crashing_hash(path, **kwargs):
            state["count"] += 1
            if state["count"] == 2:  # crash while hashing the SECOND file
                sys.stdout.write("crashing\n")
                sys.stdout.flush()
                os._exit(1)
            return real_hash_fn(path, **kwargs)

        orchestrator_module._hash_file_streaming = crashing_hash
        output_dir_path = Path(output_dir)
        request = _two_item_request(output_dir_path, run_id)
        orchestrator_module._run_ft_screening(request, _components(output_dir_path))
        sys.stdout.write("finished-without-crashing\n")  # should never reach here
        sys.stdout.flush()
        return

    if mode == "pre_manifest":
        import ft_screening.orchestrator as orchestrator_module

        output_dir_path = Path(output_dir)
        request = _two_item_request(output_dir_path, run_id)
        components = _components(output_dir_path)

        def crashing_write(*args, **kwargs):
            sys.stdout.write("crashing\n")
            sys.stdout.flush()
            os._exit(1)

        components.store.write_run_manifest = crashing_write
        orchestrator_module._run_ft_screening(request, components)
        sys.stdout.write("finished-without-crashing\n")
        sys.stdout.flush()
        return

    if mode == "post_item_write":
        import ft_screening.orchestrator as orchestrator_module

        output_dir_path = Path(output_dir)
        request = _two_item_request(output_dir_path, run_id)
        components = _components(output_dir_path)
        real_write = components.store.write_item_result
        state = {"count": 0}

        def crashing_write(*args, **kwargs):
            state["count"] += 1
            result = real_write(*args, **kwargs)
            if state["count"] == 1:
                sys.stdout.write("crashing\n")
                sys.stdout.flush()
                os._exit(1)
            return result

        components.store.write_item_result = crashing_write
        orchestrator_module._run_ft_screening(request, components)
        sys.stdout.write("finished-without-crashing\n")
        sys.stdout.flush()
        return

    if mode == "post_final_result":
        import ft_screening.orchestrator as orchestrator_module

        output_dir_path = Path(output_dir)
        request = _two_item_request(output_dir_path, run_id)
        components = _components(output_dir_path)
        real_write = components.store.write_run_result

        def crashing_write(*args, **kwargs):
            result = real_write(*args, **kwargs)
            sys.stdout.write("crashing\n")
            sys.stdout.flush()
            os._exit(1)
            return result

        components.store.write_run_result = crashing_write
        orchestrator_module._run_ft_screening(request, components)
        sys.stdout.write("finished-without-crashing\n")
        sys.stdout.flush()
        return

    raise ValueError(f"unknown mode: {mode!r}")


if __name__ == "__main__":
    main()
