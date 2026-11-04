"""Regression tests for token-usage telemetry, added as a follow-up to the
n=1 walking-skeleton run (which surfaced that ExecutionProvenance had no
usage fields at all). Operational telemetry only -- these tests
deliberately never touch aggregation.py's eligibility decision logic, and
never compute a currency cost (model + token counts only; prices change).

Covers the orchestrator-level guarantees: per-criterion usage threaded
through CriterionAdjudicator.adjudicate()'s AdjudicationOutcome, aggregated
into ExecutionProvenance per item, a cache hit reporting a CONFIRMED zero
(not the source run's historical cost), a genuinely unavailable usage
source (a fake/legacy adjudicator) never being confused with a confirmed
zero, and a "criterion_usage" ledger event per criterion for auditability.

Adjudicator/transport-level usage tests (TransportUsage extraction from a
real/fake OpenAI completion, provider_call_count/retry_count aggregation
across retries) live in tests/acceptance/test_criterion_adjudicator.py
alongside the rest of adjudication.py's own acceptance suite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ft_screening.components import AdjudicationOutcome, AdjudicationUsage, Components, ParsedDocument
from ft_screening.fakes import FakeCriterionAdjudicator, FakeDocumentParser, FakeEvidenceRetriever
from ft_screening.orchestrator import _run_ft_screening
from ft_screening.schema import (
    CriteriaSet,
    CriterionAssessment,
    CriterionSpec,
    EvidenceSpan,
    ScreeningConfig,
    ScreeningItem,
    ScreeningRequest,
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
        parser_name="fake", parser_version="1", chunker_version="1",
        retriever_name="fake", retriever_config_version="1",
        adjudicator_provider="fake", adjudicator_model="fake-1",
        model_config_id="cfg-1", prompt_template_version="1", output_schema_version="1",
        aggregation_policy_version="1",
    )
    defaults.update(overrides)
    return ScreeningConfig(**defaults)


class _UsageReportingAdjudicator:
    """A CriterionAdjudicator double that reports REAL (usage_available=True)
    usage, unlike FakeCriterionAdjudicator (which always reports
    usage_available=False, since a fake has no real provider call to
    report on). Used specifically to test the orchestrator's aggregation
    of genuinely-available usage without a real network call."""

    model = "fake-1"
    prompt_template_version = "1"
    output_schema_version = "1"

    def __init__(self, *, input_tokens: int = 100, output_tokens: int = 20, provider_call_count: int = 1) -> None:
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._provider_call_count = provider_call_count
        self.call_count = 0

    def adjudicate(
        self, *, document: ParsedDocument, criterion: CriterionSpec, evidence: list[EvidenceSpan], model_options: dict[str, Any]
    ) -> AdjudicationOutcome:
        self.call_count += 1
        assessment = CriterionAssessment(
            criterion_id=criterion.criterion_id, criterion_version=criterion.version, status="satisfied",
            evidence=evidence, rationale="ok", requires_human_review=False, model=self.model,
            prompt_template_version=self.prompt_template_version, output_schema_version=self.output_schema_version,
        )
        usage = AdjudicationUsage(
            model=self.model, input_tokens=self._input_tokens, output_tokens=self._output_tokens,
            total_tokens=self._input_tokens + self._output_tokens, provider_call_count=self._provider_call_count,
            retry_count=self._provider_call_count - 1, usage_available=True,
        )
        return AdjudicationOutcome(assessment=assessment, usage=usage)


def _run_one(tmp_path: Path, *, adjudicator, run_id: str, criteria_set=None, output_dir: Path | None = None):
    pdf = make_pdf(tmp_path, f"{run_id}.pdf")
    output_dir = output_dir or (tmp_path / "outputs")
    request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf)],
        criteria_set=criteria_set or make_criteria_set(), config=make_config(), output_dir=output_dir, run_id=run_id,
    )
    components = Components(
        parser=FakeDocumentParser(), retriever=FakeEvidenceRetriever(),
        adjudicator=adjudicator, store=FileScreeningStore(output_dir=output_dir),
    )
    return _run_ft_screening(request, components), output_dir


def test_cache_miss_aggregates_real_usage_into_execution_provenance(tmp_path):
    adjudicator = _UsageReportingAdjudicator(input_tokens=150, output_tokens=30, provider_call_count=2)
    result, _ = _run_one(tmp_path, adjudicator=adjudicator, run_id="run-1")

    provenance = result.items[0].provenance
    assert provenance.usage_available is True
    assert provenance.total_input_tokens == 150
    assert provenance.total_output_tokens == 30
    assert provenance.total_tokens == 180
    assert provenance.total_provider_calls == 2
    assert provenance.total_retries == 1


def test_cache_hit_reports_a_confirmed_zero_not_the_source_runs_historical_cost(tmp_path):
    """The second (warm) run must NOT attribute the first run's non-zero
    token cost to itself -- it made no provider call this run at all."""
    output_dir = tmp_path / "outputs"
    criteria_set = make_criteria_set()

    first_adjudicator = _UsageReportingAdjudicator(input_tokens=500, output_tokens=100, provider_call_count=1)
    result1, _ = _run_one(tmp_path, adjudicator=first_adjudicator, run_id="run-cold", criteria_set=criteria_set, output_dir=output_dir)
    assert result1.items[0].provenance.total_tokens == 600

    second_adjudicator = _UsageReportingAdjudicator()  # would be called if the cache were bypassed
    result2, _ = _run_one(tmp_path, adjudicator=second_adjudicator, run_id="run-warm", criteria_set=criteria_set, output_dir=output_dir)

    assert second_adjudicator.call_count == 0, "the second run must be a cache hit -- the adjudicator must not be called"
    provenance2 = result2.items[0].provenance
    assert provenance2.cache_status == "hit"
    # Confirmed zero, not the first run's 600 tokens, and NOT "unavailable".
    assert provenance2.usage_available is True
    assert provenance2.total_input_tokens == 0
    assert provenance2.total_output_tokens == 0
    assert provenance2.total_tokens == 0
    assert provenance2.total_provider_calls == 0
    assert provenance2.total_retries == 0


def test_usage_unavailable_from_a_fake_adjudicator_is_not_confused_with_a_confirmed_zero(tmp_path):
    """FakeCriterionAdjudicator has no real provider call to report on --
    usage_available must be False, distinguishing 'no telemetry' from a
    cache hit's genuine, confirmed zero."""
    result, _ = _run_one(tmp_path, adjudicator=FakeCriterionAdjudicator(), run_id="run-fake")

    provenance = result.items[0].provenance
    assert provenance.usage_available is False
    assert provenance.total_tokens == 0  # counts are still 0, but NOT confirmed -- usage_available says so


def test_execution_provenance_usage_available_is_false_if_any_criterion_is_unavailable(tmp_path):
    """Partial telemetry (some criteria known, some not) must never be
    reported as if the whole item's usage picture were complete."""
    criteria_set = make_criteria_set(
        criteria=[
            CriterionSpec(
                criterion_id="population", version="1", eligibility_requirement="adult humans",
                missing_evidence_policy="review", exclusion_reason_code="wrong_population",
                exclusion_reason_label="Wrong population", priority=1,
            ),
            CriterionSpec(
                criterion_id="design", version="1", eligibility_requirement="randomized controlled trial",
                missing_evidence_policy="review", exclusion_reason_code="wrong_design",
                exclusion_reason_label="Wrong design", priority=2,
            ),
        ]
    )

    class _MixedAdjudicator:
        model = "fake-1"
        prompt_template_version = "1"
        output_schema_version = "1"

        def adjudicate(self, *, criterion, evidence, **_):
            assessment = CriterionAssessment(
                criterion_id=criterion.criterion_id, criterion_version=criterion.version, status="satisfied",
                evidence=evidence, rationale="ok", requires_human_review=False, model=self.model,
                prompt_template_version=self.prompt_template_version, output_schema_version=self.output_schema_version,
            )
            available = criterion.criterion_id == "population"
            usage = AdjudicationUsage(
                model=self.model, input_tokens=100 if available else 0, output_tokens=10 if available else 0,
                total_tokens=110 if available else 0, provider_call_count=1, retry_count=0, usage_available=available,
            )
            return AdjudicationOutcome(assessment=assessment, usage=usage)

    result, _ = _run_one(tmp_path, adjudicator=_MixedAdjudicator(), run_id="run-mixed", criteria_set=criteria_set)

    provenance = result.items[0].provenance
    assert provenance.usage_available is False  # one of the two criteria had no real usage
    assert provenance.total_tokens == 110  # the known criterion's tokens are still summed


def test_ledger_contains_a_criterion_usage_event_for_a_cache_miss(tmp_path):
    adjudicator = _UsageReportingAdjudicator(input_tokens=42, output_tokens=8, provider_call_count=1)
    result, output_dir = _run_one(tmp_path, adjudicator=adjudicator, run_id="run-ledger")

    store = FileScreeningStore(output_dir=output_dir)
    _, _, ledger_path = store.artifact_paths(run_id="run-ledger")
    import json

    events = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    usage_events = [e for e in events if e.get("type") == "criterion_usage"]
    assert len(usage_events) == 1
    assert usage_events[0]["cache_status"] == "miss"
    assert usage_events[0]["input_tokens"] == 42
    assert usage_events[0]["output_tokens"] == 8
    assert usage_events[0]["usage_available"] is True


def test_ledger_contains_a_criterion_usage_event_for_a_cache_hit(tmp_path):
    output_dir = tmp_path / "outputs"
    criteria_set = make_criteria_set()
    _run_one(tmp_path, adjudicator=_UsageReportingAdjudicator(), run_id="run-a", criteria_set=criteria_set, output_dir=output_dir)
    _run_one(tmp_path, adjudicator=_UsageReportingAdjudicator(), run_id="run-b", criteria_set=criteria_set, output_dir=output_dir)

    store = FileScreeningStore(output_dir=output_dir)
    _, _, ledger_path = store.artifact_paths(run_id="run-b")
    import json

    events = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    usage_events = [e for e in events if e.get("type") == "criterion_usage"]
    assert len(usage_events) == 1
    assert usage_events[0]["cache_status"] == "hit"
    assert usage_events[0]["input_tokens"] == 0
    assert usage_events[0]["usage_available"] is True


def test_adjudication_usage_rejects_inconsistent_retry_count():
    import pytest

    with pytest.raises(Exception):
        AdjudicationUsage(
            model="fake-1", input_tokens=0, output_tokens=0, total_tokens=0,
            provider_call_count=3, retry_count=0,  # should be 2
            usage_available=False,
        )
