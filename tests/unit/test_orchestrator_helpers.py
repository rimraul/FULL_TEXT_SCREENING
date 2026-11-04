"""Unit tests for orchestrator pieces not covered by the contract suite:
resolve_components()'s rejection of the test-only "fake" identity (see
tests/contract/test_resolve_components.py for the real docling/
lexical_bm25/openai resolution paths), and a couple of pure helper
functions.
"""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from ft_screening.errors import ComponentResolutionError
from ft_screening.orchestrator import (
    _classify_run_status,
    _rebind_assessment_document_id,
    resolve_components,
    run_ft_screening,
)
from ft_screening.schema import (
    CriteriaSet,
    CriterionAssessment,
    CriterionSpec,
    EvidenceSpan,
    RunSummary,
    ScreeningConfig,
    ScreeningItem,
    ScreeningRequest,
)


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


def test_resolve_components_rejects_fake_identity(tmp_path):
    """"fake" is a test-only identity (FakeDocumentParser/etc., see
    fakes.py) -- resolve_components only resolves docling/lexical_bm25/
    openai and must reject "fake" explicitly, not silently succeed or
    raise something unrelated."""
    with pytest.raises(ComponentResolutionError, match="unsupported parser_name"):
        resolve_components(make_config(), output_dir=tmp_path)


def test_run_ft_screening_public_wrapper_rejects_fake_identity_via_resolve_components(tmp_path):
    criteria_set = CriteriaSet(
        criteria_set_id="cs",
        version="1",
        criteria=[
            CriterionSpec(
                criterion_id="population",
                version="1",
                eligibility_requirement="adults",
                missing_evidence_policy="review",
                exclusion_reason_code="wrong_population",
                exclusion_reason_label="Wrong population",
                priority=1,
            )
        ],
    )
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"content")
    request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf)],
        criteria_set=criteria_set,
        config=make_config(),
        output_dir=tmp_path / "outputs",
    )
    with pytest.raises(ComponentResolutionError, match="unsupported parser_name"):
        run_ft_screening(request)


def test_classify_run_status_completed():
    summary = RunSummary(
        total=2, completed=2, included=2, excluded=0, needs_human_review=0,
        input_unavailable=0, duplicate_input=0, processing_failed=0, not_processed=0, cache_hits=0,
    )
    assert _classify_run_status(summary) == "completed"


def test_classify_run_status_failed():
    summary = RunSummary(
        total=2, completed=0, included=0, excluded=0, needs_human_review=0,
        input_unavailable=2, duplicate_input=0, processing_failed=0, not_processed=0, cache_hits=0,
    )
    assert _classify_run_status(summary) == "failed"


def test_classify_run_status_completed_with_issues():
    summary = RunSummary(
        total=2, completed=1, included=1, excluded=0, needs_human_review=0,
        input_unavailable=1, duplicate_input=0, processing_failed=0, not_processed=0, cache_hits=0,
    )
    assert _classify_run_status(summary) == "completed_with_issues"


def test_rebind_assessment_document_id_rebinds_every_evidence_span():
    span = EvidenceSpan(
        document_id="original-doc",
        document_sha256="a" * 64,
        chunk_id="chunk-1",
        section="Methods",
        excerpt="text",
        stance="supports",
        retrieval_method="fake",
        retrieval_query_version="1",
    )
    assessment = CriterionAssessment(
        criterion_id="population",
        criterion_version="1",
        status="satisfied",
        evidence=[span],
        rationale="ok",
        requires_human_review=False,
        model="fake",
        prompt_template_version="1",
        output_schema_version="1",
    )
    rebound = _rebind_assessment_document_id(assessment, document_id="new-doc")
    assert rebound.evidence[0].document_id == "new-doc"
    assert span.document_id == "original-doc"  # original untouched (frozen + cloned)
