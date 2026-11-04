"""Minimal real-provider check that usage tracking actually works end to
end (not just against fakes): one real OpenAI call through
LLMCriterionAdjudicator, asserting the returned AdjudicationOutcome.usage
carries real, non-zero token counts. Deliberately bypasses Docling parsing
entirely (a synthetic ParsedDocument/EvidenceSpan) -- this is checking the
usage-telemetry wiring, not re-validating parsing."""

from __future__ import annotations

import sys

sys.path.insert(0, r"C:/projects")
import my_api_keys  # noqa: F401

sys.path.insert(0, r"C:/projects/FT_SCREENING/src")

from ft_screening.adjudication import LLMCriterionAdjudicator, OpenAIChatCompletionTransport
from ft_screening.components import Chunk, ParsedDocument
from ft_screening.schema import CriterionSpec, DocumentCoverage, EvidenceSpan

document = ParsedDocument(
    document_id="d1",
    document_sha256="a" * 64,
    coverage=DocumentCoverage(status="complete", total_pages=1, parsed_pages=[1], failed_pages=[], extracted_characters=100, ocr_used=False),
    chunks=(Chunk(chunk_id="c-1", section="Methods", page=1, start_char=0, end_char=50, text="Adults aged 40-75 were enrolled."),),
)
criterion = CriterionSpec(
    criterion_id="population", version="1", eligibility_requirement="Participants are adult humans.",
    missing_evidence_policy="review", exclusion_reason_code="wrong_population",
    exclusion_reason_label="Wrong population", priority=1,
)
evidence = [
    EvidenceSpan(
        document_id="d1", document_sha256="a" * 64, chunk_id="c-1", section="Methods", page=1,
        start_char=0, end_char=50, excerpt="Adults aged 40-75 were enrolled.", stance="relevant",
        retrieval_method="fake", retrieval_query_version="1",
    )
]

adjudicator = LLMCriterionAdjudicator(transport=OpenAIChatCompletionTransport(), model="gpt-4o-mini")
outcome = adjudicator.adjudicate(
    document=document, criterion=criterion, evidence=evidence, model_options={"max_completion_tokens": 300}
)

print("status:", outcome.assessment.status)
print("usage_available:", outcome.usage.usage_available)
print("input_tokens:", outcome.usage.input_tokens)
print("output_tokens:", outcome.usage.output_tokens)
print("total_tokens:", outcome.usage.total_tokens)
print("provider_call_count:", outcome.usage.provider_call_count)
print("retry_count:", outcome.usage.retry_count)

assert outcome.usage.usage_available is True, "real OpenAI call must report usage_available=True"
assert outcome.usage.total_tokens > 0, "a real call must report non-zero tokens"
assert outcome.usage.total_tokens == outcome.usage.input_tokens + outcome.usage.output_tokens
assert outcome.usage.provider_call_count == 1
assert outcome.usage.retry_count == 0
print("\nOK: real usage telemetry verified end to end.")
