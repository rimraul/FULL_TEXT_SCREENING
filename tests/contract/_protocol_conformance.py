"""Reusable protocol-conformance assertions (codex_code_review_810ccf0.md
finding 4: "reusable parser/retriever/adjudicator/store conformance tests
for wrong top-level type, malformed nested instance, iterator
materialization, identity mismatch, and raised I/O/cancellation" was
explicitly required and had not been implemented).

Each `assert_*_conformance` function takes a FACTORY for a working
component (a zero-arg callable returning a fresh, correctly-behaving
instance) and drives it through a fixed set of adversarial wrappers,
asserting the orchestrator degrades every one of them safely -- never an
uncaught exception, never silent wrong output. The suite is deliberately
component-agnostic: `test_protocol_conformance.py` runs it against the
current fakes today, and the SAME functions can be pointed at real
parser/retriever/adjudicator/store implementations later (plan_v5 steps
6/7) with no changes here.

Five adversarial categories per component (per the review's own list),
applied wherever the component's return shape makes them meaningful (a
docstring on each function notes any category that's not applicable to
that component and why):

  1. wrong top-level type    -- the call returns something that isn't
                                 even the right kind of object at all.
  2. malformed nested instance -- returns a well-typed CONTAINER whose
                                 nested field was set via `model_construct`
                                 (bypassing validation), so revalidation
                                 at the boundary is what has to catch it.
  3. iterator materialization -- returns a one-shot generator instead of a
                                 list; the boundary validator's own
                                 iteration must not exhaust it before the
                                 value is used for real.
  4. identity mismatch        -- returns a value bound to the WRONG
                                 document/criterion/model identity.
  5. raised I/O/cancellation  -- the call raises a plain exception (must
                                 become a graceful item/run outcome per
                                 the component's documented failure
                                 policy) and, separately, a
                                 KeyboardInterrupt (must propagate as a
                                 real interrupt, never be swallowed as a
                                 component failure).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ft_screening.components import AdjudicationOutcome, AdjudicationUsage, Chunk, Components, ParsedDocument
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


def _make_pdf(tmp_path: Path, name: str = "conformance.pdf") -> Path:
    """A STRUCTURALLY VALID single-page PDF with real extractable text --
    not just a "%PDF-1.4" byte stub. The fakes never cared (they don't
    read file bytes at all), but a real parser factory (Docling and
    beyond) enrolled in this suite via its "good" sanity check at the end
    of each assert_*_conformance() function needs something it can
    actually convert; a stub raises there instead of completing."""
    path = tmp_path / "pdfs" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    import fitz  # PyMuPDF -- already a project dependency, lightweight (no ML models)

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Conformance suite fixture document. Sample body text for parsing.")
    doc.save(str(path))
    doc.close()
    return path


def _make_criteria_set() -> CriteriaSet:
    return CriteriaSet(
        criteria_set_id="conformance-cs", version="1",
        criteria=[
            CriterionSpec(
                criterion_id="population", version="1", eligibility_requirement="adult humans",
                missing_evidence_policy="review", exclusion_reason_code="wrong_population",
                exclusion_reason_label="Wrong population", priority=1,
            )
        ],
    )


def _make_config(*, parser_options: dict | None = None, **overrides) -> ScreeningConfig:
    fields: dict = dict(
        parser_name="conformance", parser_version="1", chunker_version="1",
        parser_options=parser_options or {},
        retriever_name="conformance",
        retriever_config_version="1", adjudicator_provider="conformance", adjudicator_model="fake-1",
        model_config_id="cfg-1", prompt_template_version="1", output_schema_version="1",
        aggregation_policy_version="1",
    )
    fields.update(overrides)
    return ScreeningConfig(**fields)


def _run_one_item(
    tmp_path: Path, *, parser=None, retriever=None, adjudicator=None, store=None, run_id: str,
    parser_options: dict | None = None, config_overrides: dict | None = None,
):
    # request.output_dir must match components.store.output_dir exactly
    # (the orchestrator itself enforces this) -- when a caller supplies a
    # store, derive output_dir FROM it rather than generating an
    # independent path that would silently mismatch.
    store = store or FileScreeningStore(output_dir=tmp_path / f"out-{run_id}")
    output_dir = store.output_dir
    pdf = _make_pdf(tmp_path, f"{run_id}.pdf")
    request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf)],
        criteria_set=_make_criteria_set(),
        config=_make_config(parser_options=parser_options, **(config_overrides or {})),
        output_dir=output_dir, run_id=run_id,
    )
    components = Components(
        parser=parser or FakeDocumentParser(),
        retriever=retriever or FakeEvidenceRetriever(),
        adjudicator=adjudicator or FakeCriterionAdjudicator(),
        store=store,
    )
    return _run_ft_screening(request, components)


def _valid_document(document_id: str = "d1") -> ParsedDocument:
    return ParsedDocument(
        document_id=document_id,
        document_sha256="a" * 64,
        coverage=dict(status="complete", total_pages=1, parsed_pages=[1], failed_pages=[], ocr_used=False, extracted_characters=10, warnings=[]),
        chunks=(Chunk(chunk_id="chunk-0", section=None, page=1, start_char=0, end_char=10, text="0123456789"),),
    )


def _valid_evidence(document: ParsedDocument) -> EvidenceSpan:
    chunk = document.chunks[0]
    return EvidenceSpan(
        document_id=document.document_id, document_sha256=document.document_sha256, chunk_id=chunk.chunk_id,
        section=chunk.section, page=chunk.page, start_char=chunk.start_char, end_char=chunk.end_char,
        excerpt=chunk.text, stance="relevant", retrieval_method="fake", retrieval_query_version="1", rank=1, score=1.0,
    )


# ---------------------------------------------------------------------------
# DocumentParser conformance
# ---------------------------------------------------------------------------


class _WrappedParser:
    def __init__(self, fn: Callable[..., Any]) -> None:
        self._fn = fn

    def parse(self, **kwargs):
        return self._fn(**kwargs)


def assert_document_parser_conformance(
    parser_factory: Callable[[], Any], *, tmp_path: Path, good_parser_options: dict | None = None
) -> None:
    """Category 3 (iterator materialization) does not apply: `parse()`
    returns exactly one `ParsedDocument`, not a list/iterable the
    orchestrator consumes -- there is nothing here that a one-shot
    generator could silently exhaust.

    `good_parser_options`: passed only to the final "good" sanity check's
    `ScreeningConfig.parser_options` -- some real parsers require options
    the fakes don't (e.g. DoclingDocumentParser's mandatory
    `artifacts_manifest_digest`, see `DoclingDocumentParser.recommended_parser_options()`)."""
    good = parser_factory()

    # 1. wrong top-level type.
    result = _run_one_item(tmp_path, parser=_WrappedParser(lambda **_: object()), run_id="parser-wrong-type")
    assert result.items[0].execution_status == "processing_failed"
    assert result.items[0].errors[0].code == "parse_failed"

    # 2. malformed nested instance (bypasses field validation via model_construct).
    bad_chunk = Chunk.model_construct(chunk_id="c1", section=None, page=1, start_char=0, end_char=999999, text=object())

    def malformed(**kwargs):
        return ParsedDocument.model_construct(
            document_id=kwargs["document_id"], document_sha256=kwargs["document_sha256"],
            coverage=_valid_document().coverage, chunks=(bad_chunk,),
        )

    result = _run_one_item(tmp_path, parser=_WrappedParser(malformed), run_id="parser-malformed-nested")
    assert result.items[0].execution_status == "processing_failed"
    assert result.items[0].errors[0].code == "parse_failed"

    # 4. identity mismatch.
    result = _run_one_item(
        tmp_path, parser=_WrappedParser(lambda **_: _valid_document(document_id="someone-elses-document")),
        run_id="parser-identity-mismatch",
    )
    assert result.items[0].execution_status == "processing_failed"
    assert result.items[0].errors[0].code == "parse_failed"

    # 5a. raised I/O failure -> graceful item outcome.
    def raising(**_):
        raise OSError("simulated disk failure reading the PDF")

    result = _run_one_item(tmp_path, parser=_WrappedParser(raising), run_id="parser-io-failure")
    assert result.items[0].execution_status == "processing_failed"
    assert result.items[0].errors[0].code == "parse_failed"

    # 5b. raised KeyboardInterrupt -> must propagate as a real interrupt,
    # never be caught and converted into a component failure.
    def interrupting(**_):
        raise KeyboardInterrupt()

    try:
        _run_one_item(tmp_path, parser=_WrappedParser(interrupting), run_id="parser-cancellation")
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("a parser-raised KeyboardInterrupt must propagate, not be swallowed")

    # Sanity: the GOOD factory-produced parser must still work end to end.
    result = _run_one_item(tmp_path, parser=good, run_id="parser-good", parser_options=good_parser_options)
    assert result.items[0].execution_status == "completed"


# ---------------------------------------------------------------------------
# EvidenceRetriever conformance
# ---------------------------------------------------------------------------


class _WrappedRetriever:
    def __init__(self, fn: Callable[..., Any]) -> None:
        self._fn = fn

    def retrieve(self, **kwargs):
        return self._fn(**kwargs)


def assert_evidence_retriever_conformance(retriever_factory: Callable[[], Any], *, tmp_path: Path) -> None:
    good = retriever_factory()

    # 1. wrong top-level type (not even iterable).
    result = _run_one_item(tmp_path, retriever=_WrappedRetriever(lambda **_: object()), run_id="retriever-wrong-type")
    assert result.items[0].execution_status == "processing_failed"
    assert result.items[0].errors[0].code == "retrieval_failed"

    # 2. malformed nested instance.
    def malformed(**kwargs):
        document = kwargs["document"]
        chunk = document.chunks[0]
        return [
            EvidenceSpan.model_construct(
                document_id=document.document_id, document_sha256=document.document_sha256, chunk_id=chunk.chunk_id,
                section=None, page=1, start_char=0, end_char=5, excerpt=object(), stance="relevant",
                retrieval_method="fake", retrieval_query_version="1", rank=1, score=1.0,
            )
        ]

    result = _run_one_item(tmp_path, retriever=_WrappedRetriever(malformed), run_id="retriever-malformed-nested")
    assert result.items[0].execution_status == "processing_failed"
    assert result.items[0].errors[0].code == "retrieval_failed"

    # 3. iterator materialization -- a one-shot generator must still work,
    # not be silently exhausted by boundary validation before use.
    def generator_evidence(**kwargs):
        document = kwargs["document"]
        yield _valid_evidence(document)

    result = _run_one_item(tmp_path, retriever=_WrappedRetriever(generator_evidence), run_id="retriever-generator")
    assert result.items[0].execution_status == "completed"
    assert len(result.items[0].eligibility.assessments[0].evidence) == 1

    # 4. identity mismatch (evidence bound to the wrong document).
    def mismatched(**kwargs):
        document = kwargs["document"]
        span = _valid_evidence(document)
        return [span.model_copy(update={"document_id": "someone-elses-document"})]

    result = _run_one_item(tmp_path, retriever=_WrappedRetriever(mismatched), run_id="retriever-identity-mismatch")
    assert result.items[0].execution_status == "processing_failed"
    assert result.items[0].errors[0].code == "retrieval_failed"

    # 5a. raised I/O failure.
    def raising(**_):
        raise ConnectionError("simulated retrieval backend failure")

    result = _run_one_item(tmp_path, retriever=_WrappedRetriever(raising), run_id="retriever-io-failure")
    assert result.items[0].execution_status == "processing_failed"
    assert result.items[0].errors[0].code == "retrieval_failed"

    # 5b. raised KeyboardInterrupt -> must propagate.
    def interrupting(**_):
        raise KeyboardInterrupt()

    try:
        _run_one_item(tmp_path, retriever=_WrappedRetriever(interrupting), run_id="retriever-cancellation")
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("a retriever-raised KeyboardInterrupt must propagate, not be swallowed")

    result = _run_one_item(tmp_path, retriever=good, run_id="retriever-good")
    assert result.items[0].execution_status == "completed"


# ---------------------------------------------------------------------------
# CriterionAdjudicator conformance
# ---------------------------------------------------------------------------


class _WrappedAdjudicator:
    def __init__(self, fn: Callable[..., Any]) -> None:
        self._fn = fn

    def adjudicate(self, **kwargs):
        return self._fn(**kwargs)


def assert_criterion_adjudicator_conformance(
    adjudicator_factory: Callable[[], Any], *, tmp_path: Path, good_config_overrides: dict | None = None
) -> None:
    """Category 3 (iterator materialization) does not apply: `adjudicate()`
    returns exactly one `CriterionAssessment`, not a list/iterable.

    `good_config_overrides`: passed only to the final "good" sanity
    check's `ScreeningConfig` -- some real adjudicators require identity
    fields (e.g. LLMCriterionAdjudicator's fixed `prompt_template_version`/
    `output_schema_version` class constants) the fakes don't."""
    good = adjudicator_factory()

    def base_fields(criterion, evidence):
        return dict(
            criterion_id=criterion.criterion_id, criterion_version=criterion.version, status="satisfied",
            evidence=evidence, rationale="conformance probe", requires_human_review=False, model="fake-1",
            prompt_template_version="1", output_schema_version="1",
        )

    def usage_for(model: str) -> AdjudicationUsage:
        return AdjudicationUsage(
            model=model, input_tokens=0, output_tokens=0, total_tokens=0,
            provider_call_count=1, retry_count=0, usage_available=False,
        )

    # 1. wrong top-level type.
    result = _run_one_item(tmp_path, adjudicator=_WrappedAdjudicator(lambda **_: object()), run_id="adjudicator-wrong-type")
    assert result.items[0].execution_status == "processing_failed"
    assert result.items[0].errors[0].code == "adjudication_failed"

    # 2. malformed nested instance.
    def malformed(**kwargs):
        fields = base_fields(kwargs["criterion"], kwargs["evidence"])
        assessment = CriterionAssessment.model_construct(**{**fields, "evidence": (object(),)})
        return AdjudicationOutcome.model_construct(assessment=assessment, usage=usage_for("fake-1"))

    result = _run_one_item(tmp_path, adjudicator=_WrappedAdjudicator(malformed), run_id="adjudicator-malformed-nested")
    assert result.items[0].execution_status == "processing_failed"
    assert result.items[0].errors[0].code == "adjudication_failed"

    # 4. identity mismatch (wrong model identity).
    def wrong_model(**kwargs):
        fields = base_fields(kwargs["criterion"], kwargs["evidence"])
        fields["model"] = "not-the-configured-model"
        return AdjudicationOutcome(assessment=CriterionAssessment(**fields), usage=usage_for("not-the-configured-model"))

    result = _run_one_item(tmp_path, adjudicator=_WrappedAdjudicator(wrong_model), run_id="adjudicator-identity-mismatch")
    assert result.items[0].execution_status == "processing_failed"
    assert result.items[0].errors[0].code == "adjudication_failed"

    # 5a. raised I/O failure.
    def raising(**_):
        raise TimeoutError("simulated LLM provider timeout")

    result = _run_one_item(tmp_path, adjudicator=_WrappedAdjudicator(raising), run_id="adjudicator-io-failure")
    assert result.items[0].execution_status == "processing_failed"
    assert result.items[0].errors[0].code == "adjudication_failed"

    # 5b. raised KeyboardInterrupt -> must propagate.
    def interrupting(**_):
        raise KeyboardInterrupt()

    try:
        _run_one_item(tmp_path, adjudicator=_WrappedAdjudicator(interrupting), run_id="adjudicator-cancellation")
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("an adjudicator-raised KeyboardInterrupt must propagate, not be swallowed")

    result = _run_one_item(tmp_path, adjudicator=good, run_id="adjudicator-good", config_overrides=good_config_overrides)
    assert result.items[0].execution_status == "completed"


# ---------------------------------------------------------------------------
# ScreeningStore conformance
# ---------------------------------------------------------------------------


def assert_screening_store_conformance(store_factory: Callable[[Path], Any], *, tmp_path: Path) -> None:
    """`store_factory(output_dir)` builds a fresh, working store rooted at
    `output_dir`. Categories 1-3 don't apply the same way here (the store's
    return SHAPES -- CachedAssessment -- are already covered by
    test_hardening_round4.py's malformed-cache-payload tests via a
    from-scratch Protocol implementation); this focuses on category 5,
    which is where the review's own findings 3/4 (rounds 5 and 6) landed:
    cache-read/write failures must be RECOVERABLE (the run continues), but
    item/result persistence failures must be FATAL (the run stops rather
    than proceed unrecorded -- api_contract_v2.md's own stated policy)."""

    # Cache-read failure: recoverable, item still completes.
    class RaisingGetStore(FileScreeningStore):
        def get_cached_assessment(self, *, cache_key: str):
            raise ConnectionError("simulated cache backend outage")

    out1 = tmp_path / "store-cache-get"
    result = _run_one_item(tmp_path, store=RaisingGetStore(output_dir=out1), run_id="store-cache-get-failure")
    assert result.items[0].execution_status == "completed"
    assert any(e.code == "persistence_failed" for e in result.items[0].errors)

    # Cache-write failure: recoverable, item still completes with its
    # already-computed valid assessment.
    class RaisingPutStore(FileScreeningStore):
        def put_cached_assessment(self, *, cache_key: str, assessment, source_run_id: str) -> None:
            raise ConnectionError("simulated cache backend outage")

    out2 = tmp_path / "store-cache-put"
    result = _run_one_item(tmp_path, store=RaisingPutStore(output_dir=out2), run_id="store-cache-put-failure")
    assert result.items[0].execution_status == "completed"
    assert any(e.code == "persistence_failed" for e in result.items[0].errors)

    # Item-result persistence failure: FATAL -- must propagate, not be
    # downgraded to a soft per-item outcome (an unrecorded "success" would
    # violate the frozen "don't proceed unrecorded" policy).
    class RaisingWriteItemStore(FileScreeningStore):
        def write_item_result(self, *, run_id: str, item_result) -> None:
            raise OSError("simulated disk-full writing item_results.jsonl")

    out3 = tmp_path / "store-item-write"
    try:
        _run_one_item(tmp_path, store=RaisingWriteItemStore(output_dir=out3), run_id="store-item-write-failure")
    except OSError:
        pass
    else:
        raise AssertionError("a write_item_result failure must propagate and stop the run, not be swallowed")

    # Confirm the factory-produced store still works end to end.
    out4 = tmp_path / "store-good"
    good_store = store_factory(out4)
    result = _run_one_item(tmp_path, store=good_store, run_id="store-good")
    assert result.items[0].execution_status == "completed"
