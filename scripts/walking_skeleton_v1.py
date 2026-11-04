"""n=1 walking-skeleton run (plan_v3.md validation-plan step 1 / plan_v5.md
step 8): a plumbing check, NOT a methodology validation. Runs the real
Docling parser, real lexical retriever, and real OpenAI-backed adjudicator
end-to-end over one clearly-INCLUDE and one clearly-EXCLUDE fixture paper,
against a frozen 3-criterion set, and writes a full run directory
(manifest, ledger, item results, run result) to outputs/.

Criteria and fixture papers are borrowed from iasia_test's already-completed
title/abstract screening (screening_context_mvp3.md's I1/I2/I3 + its
screening_decisions.jsonl), chosen specifically because that project has an
unambiguous DOI-keyed decisions file -- see the walking-skeleton criteria
selection discussion in this session. Reusing an existing, already-labeled
pair means the "clearly eligible" / "clearly ineligible" framing is not
invented for this test; it is inherited from a real prior human/LLM-reviewed
screening decision.

Written before `orchestrator.resolve_components()` existed, so this script
still wires `Components` explicitly and calls the private
`_run_ft_screening` directly rather than the public `run_ft_screening()`
path -- both are equally real (resolve_components() would construct the
SAME DoclingDocumentParser/LexicalEvidenceRetriever/LLMCriterionAdjudicator
classes this script constructs by hand); left as-is rather than refactored,
since this script's own already-validated real-money run isn't the place
to introduce an unrelated change.

Requires OPENAI_API_KEY in the environment (loaded here via the user's
shared c:/projects/my_api_keys.py, never printed or logged) and Docling
model artifacts already bootstrapped locally (confirmed present under
~/.cache/docling/models for this run).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:/projects")
import my_api_keys  # noqa: F401 -- sets OPENAI_API_KEY (and others) into os.environ on import

sys.path.insert(0, r"C:/projects/FT_SCREENING/src")

from ft_screening.adjudication import LLMCriterionAdjudicator, OpenAIChatCompletionTransport
from ft_screening.components import Components
from ft_screening.document import DoclingDocumentParser
from ft_screening.orchestrator import _run_ft_screening
from ft_screening.retrieval import LexicalEvidenceRetriever
from ft_screening.schema import CriteriaSet, CriterionSpec, ScreeningConfig, ScreeningItem, ScreeningRequest
from ft_screening.store import FileScreeningStore

IASIA_PDFS = Path(r"C:/projects/iasia_test/outputs/pdfs")

CRITERIA_SET = CriteriaSet(
    criteria_set_id="walking-skeleton-drr-governance",
    version="1",
    criteria=[
        CriterionSpec(
            criterion_id="I1_review_type",
            version="1",
            eligibility_requirement=(
                "The document is a review-type study (systematic review, scoping review, or "
                "other literature review) that synthesizes existing published literature, rather "
                "than reporting a single primary empirical study or a purely conceptual framework "
                "paper with no literature synthesis."
            ),
            missing_evidence_policy="review",
            exclusion_reason_code="not_a_review",
            exclusion_reason_label="Not a review-type synthesis of existing literature",
            priority=1,
        ),
        CriterionSpec(
            criterion_id="I2_drr_hazard_framing",
            version="1",
            eligibility_requirement=(
                "The document substantively addresses disaster risk reduction (DRR) or resilience "
                "to hazards, with an explicit hazard/disaster framing -- general risk, climate "
                "adaptation, or sustainability content without an explicit hazard/disaster context "
                "does not satisfy this criterion."
            ),
            missing_evidence_policy="review",
            exclusion_reason_code="off_topic_hazard",
            exclusion_reason_label="No explicit disaster/hazard framing",
            priority=2,
        ),
        CriterionSpec(
            criterion_id="I3_governance_dimension",
            version="1",
            eligibility_requirement=(
                "The document substantively analyses at least one of the following governance "
                "dimensions in a DRR/resilience context: institutional capacity, decentralization, "
                "multilevel governance, partnerships, civil society, capacity building, community "
                "participation, education/training, financing, legal/regulatory frameworks, "
                "information systems, or urban planning. Passing mention of government actors "
                "without substantive analysis of one of these dimensions does not satisfy this "
                "criterion."
            ),
            missing_evidence_policy="review",
            exclusion_reason_code="no_governance",
            exclusion_reason_label="No substantive governance-dimension analysis",
            priority=3,
        ),
    ],
)

ITEMS = [
    ScreeningItem(
        document_id="d-include-wildfire",
        report_id="r-include-wildfire",
        pdf_path=IASIA_PDFS / "10_1007_s11069-025-07955-0.pdf",
        citation={
            "doi": "10.1007/s11069-025-07955-0",
            "title": "Comparative analysis of Community-Based wildfire management",
            "iasia_stage1_decision": "INCLUDE",
        },
    ),
    ScreeningItem(
        document_id="d-exclude-heat",
        report_id="r-exclude-heat",
        pdf_path=IASIA_PDFS / "10_1007_s13412-025-01018-3.pdf",
        citation={
            "doi": "10.1007/s13412-025-01018-3",
            "title": "Integrated policy and AI-assisted technologies to address extreme heat",
            "iasia_stage1_decision": "EXCLUDE",
            "iasia_stage1_reason_code": "off_topic_hazard",
        },
    ),
]


def main() -> None:
    output_dir = Path(r"C:/projects/FT_SCREENING/outputs/walking_skeleton_v1")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = ScreeningConfig(
        parser_name="docling",
        parser_version="2.61.2",
        chunker_version="1",
        parser_options=DoclingDocumentParser.recommended_parser_options(),
        retriever_name="lexical_bm25",
        retriever_config_version="1",
        retriever_options={},
        adjudicator_provider="openai",
        adjudicator_model="gpt-4o-mini",
        model_config_id="walking-skeleton-v1",
        model_options={"max_completion_tokens": 600},
        prompt_template_version=LLMCriterionAdjudicator.PROMPT_TEMPLATE_VERSION,
        output_schema_version=LLMCriterionAdjudicator.OUTPUT_SCHEMA_VERSION,
        aggregation_policy_version="1",
        cache_mode="use",
        duplicate_content_policy="error",
        item_error_policy="continue",
        max_workers=1,
    )

    request = ScreeningRequest(
        run_id="walking-skeleton-v1",
        items=ITEMS,
        criteria_set=CRITERIA_SET,
        config=config,
        output_dir=output_dir,
    )

    components = Components(
        parser=DoclingDocumentParser(),
        retriever=LexicalEvidenceRetriever(),
        adjudicator=LLMCriterionAdjudicator(transport=OpenAIChatCompletionTransport(), model="gpt-4o-mini"),
        store=FileScreeningStore(output_dir=output_dir),
    )

    events: list[str] = []

    def progress_cb(event) -> None:  # noqa: ANN001
        events.append(f"{time.strftime('%H:%M:%S')} {event.event} input_index={event.input_index}")
        print(events[-1])

    started = time.monotonic()
    result = _run_ft_screening(request, components, progress_cb=progress_cb)
    elapsed = time.monotonic() - started

    print(f"\nrun_status={result.run_status} elapsed={elapsed:.1f}s")
    print(f"completed={result.summary.completed} total={result.summary.total}")
    for item in result.items:
        decision = item.eligibility.final_decision if item.eligibility else None
        print(f"  {item.document_id}: execution_status={item.execution_status} final_decision={decision}")

    run_result_path, item_results_path, ledger_path = components.store.artifact_paths(run_id="walking-skeleton-v1")
    print(f"\nrun_result: {run_result_path}")
    print(f"item_results: {item_results_path}")
    print(f"ledger: {ledger_path}")


if __name__ == "__main__":
    main()
