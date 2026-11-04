"""Second pass over the walking-skeleton request (different run_id, same
output_dir/config/items) to verify criterion_cache_key-based caching:
expect cache_hits == total assessments, zero new Docling conversions, and
zero new OpenAI HTTP requests."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:/projects")
import my_api_keys  # noqa: F401

sys.path.insert(0, r"C:/projects/FT_SCREENING/src")

from ft_screening.adjudication import LLMCriterionAdjudicator, OpenAIChatCompletionTransport
from ft_screening.components import Components
from ft_screening.document import DoclingDocumentParser
from ft_screening.orchestrator import _run_ft_screening
from ft_screening.retrieval import LexicalEvidenceRetriever
from ft_screening.store import FileScreeningStore
from walking_skeleton_v1 import CRITERIA_SET, ITEMS

from ft_screening.schema import ScreeningConfig, ScreeningRequest

RUN_ID = "walking-skeleton-v1-cache-check"


def main() -> None:
    output_dir = Path(r"C:/projects/FT_SCREENING/outputs/walking_skeleton_v1")

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
        run_id=RUN_ID,
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

    started = time.monotonic()
    result = _run_ft_screening(request, components, progress_cb=lambda e: print(e.event, e.input_index))
    elapsed = time.monotonic() - started

    print(f"\nrun_status={result.run_status} elapsed={elapsed:.1f}s")
    print(f"cache_hits={result.summary.cache_hits} completed={result.summary.completed} total={result.summary.total}")
    for item in result.items:
        print(f"  {item.document_id}: cache_status={item.provenance.cache_status if item.provenance else None}")


if __name__ == "__main__":
    main()
