"""Runs the reusable conformance suite (_protocol_conformance.py) against
the CURRENT fakes and FileScreeningStore, plus the real DoclingDocumentParser,
LexicalEvidenceRetriever (plan_v5 step 6), and LLMCriterionAdjudicator
(plan_v5 step 7) -- the assertion functions themselves needed no changes
for any of the three real components enrolled so far.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from _protocol_conformance import (
    assert_criterion_adjudicator_conformance,
    assert_document_parser_conformance,
    assert_evidence_retriever_conformance,
    assert_screening_store_conformance,
)
from ft_screening.fakes import FakeCriterionAdjudicator, FakeDocumentParser, FakeEvidenceRetriever
from ft_screening.store import FileScreeningStore


def test_fake_document_parser_conforms(tmp_path):
    assert_document_parser_conformance(FakeDocumentParser, tmp_path=tmp_path)


@pytest.mark.slow
def test_docling_document_parser_conforms(tmp_path):
    """The "good" sanity check at the end of assert_document_parser_conformance
    performs one real Docling conversion (against _make_pdf()'s
    PyMuPDF-generated, structurally valid fixture) -- slow (pays the
    ~120-150s model-warmup cost), excluded from the default suite."""
    from ft_screening.document import DoclingDocumentParser

    assert_document_parser_conformance(
        DoclingDocumentParser, tmp_path=tmp_path,
        good_parser_options=DoclingDocumentParser.recommended_parser_options(),
    )


def test_fake_evidence_retriever_conforms(tmp_path):
    assert_evidence_retriever_conformance(FakeEvidenceRetriever, tmp_path=tmp_path)


def test_lexical_evidence_retriever_conforms(tmp_path):
    """Fast and offline (no model warmup, unlike the Docling parser) --
    no `slow` marker needed."""
    from ft_screening.retrieval import LexicalEvidenceRetriever

    assert_evidence_retriever_conformance(LexicalEvidenceRetriever, tmp_path=tmp_path)


def test_fake_criterion_adjudicator_conforms(tmp_path):
    assert_criterion_adjudicator_conformance(FakeCriterionAdjudicator, tmp_path=tmp_path)


def test_llm_criterion_adjudicator_conforms(tmp_path):
    """Fast and offline -- a fake, always-succeeding transport stands in
    for the real OpenAI adapter; conformance testing must never touch a
    real provider or require an API key."""
    from ft_screening.adjudication import LLMCriterionAdjudicator, TransportResult

    class _AlwaysSucceedsTransport:
        def complete(self, *, response_schema, **_):
            return TransportResult(
                response=response_schema.model_validate(
                    {"supporting_chunk_ids": [], "contradicting_chunk_ids": [], "proposed_status": "satisfied",
                     "rationale": "conformance probe", "requires_human_review": False}
                )
            )

    assert_criterion_adjudicator_conformance(
        lambda: LLMCriterionAdjudicator(transport=_AlwaysSucceedsTransport(), model="fake-1"),
        tmp_path=tmp_path,
        good_config_overrides={
            "prompt_template_version": LLMCriterionAdjudicator.PROMPT_TEMPLATE_VERSION,
            "output_schema_version": LLMCriterionAdjudicator.OUTPUT_SCHEMA_VERSION,
        },
    )


def test_file_screening_store_conforms(tmp_path):
    assert_screening_store_conformance(lambda output_dir: FileScreeningStore(output_dir=output_dir), tmp_path=tmp_path)
