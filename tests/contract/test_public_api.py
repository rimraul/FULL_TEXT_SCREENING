"""codex_code_review_44d9e90.md finding 10: a thin api.py is the documented
consumer entry point, and the store's output_dir must match the request's.
"""

from pathlib import Path

import pytest

from ft_screening.components import Components
from ft_screening.errors import RunSetupError
from ft_screening.fakes import FakeCriterionAdjudicator, FakeDocumentParser, FakeEvidenceRetriever
from ft_screening.orchestrator import _run_ft_screening
from ft_screening.schema import CriteriaSet, CriterionSpec, ScreeningConfig, ScreeningItem, ScreeningRequest
from ft_screening.store import FileScreeningStore


def test_documented_consumer_import_path_works():
    from ft_screening.api import (  # noqa: F401
        CriteriaSet,
        CriterionSpec,
        EligibilityResult,
        ScreeningConfig,
        ScreeningItem,
        ScreeningRequest,
        ScreeningRunResult,
        run_ft_screening,
    )


def test_api_module_does_not_export_private_orchestration_function():
    import ft_screening.api as api_module

    assert not hasattr(api_module, "_run_ft_screening")
    assert "_run_ft_screening" not in api_module.__all__


def test_top_level_package_reexports_public_surface():
    import ft_screening

    assert ft_screening.run_ft_screening is not None
    assert ft_screening.ScreeningRequest is not None


def _make_criteria_set() -> CriteriaSet:
    return CriteriaSet(
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


def _make_config() -> ScreeningConfig:
    return ScreeningConfig(
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


def test_store_output_dir_must_match_request_output_dir(tmp_path: Path):
    pdf = tmp_path / "pdfs" / "a.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"content")

    request_output_dir = tmp_path / "outputs"
    store_output_dir = tmp_path / "a_different_directory"  # deliberately mismatched

    request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf)],
        criteria_set=_make_criteria_set(),
        config=_make_config(),
        output_dir=request_output_dir,
    )
    components = Components(
        parser=FakeDocumentParser(),
        retriever=FakeEvidenceRetriever(),
        adjudicator=FakeCriterionAdjudicator(),
        store=FileScreeningStore(output_dir=store_output_dir),
    )
    with pytest.raises(RunSetupError):
        _run_ft_screening(request, components)


def test_store_output_dir_matching_request_succeeds(tmp_path: Path):
    pdf = tmp_path / "pdfs" / "a.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"content")

    output_dir = tmp_path / "outputs"
    request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf)],
        criteria_set=_make_criteria_set(),
        config=_make_config(),
        output_dir=output_dir,
    )
    components = Components(
        parser=FakeDocumentParser(),
        retriever=FakeEvidenceRetriever(),
        adjudicator=FakeCriterionAdjudicator(),
        store=FileScreeningStore(output_dir=output_dir),
    )
    result = _run_ft_screening(request, components)
    assert result.run_status == "completed"
