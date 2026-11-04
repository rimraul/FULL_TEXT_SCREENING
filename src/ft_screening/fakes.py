"""Deterministic, test-controllable fake implementations of the private
component protocols (components.py). No real PDF parsing, retrieval, or
LLM calls -- real implementations now exist behind the same Protocol
(document.DoclingDocumentParser, retrieval.LexicalEvidenceRetriever,
adjudication.LLMCriterionAdjudicator, orchestrator.resolve_components()
wires them for the public run_ft_screening() path). These fakes remain
the correct choice for tests: fast, deterministic, no network/credentials/
Docling models needed, and still the only supported way to construct
Components directly and call `_run_ft_screening` for a test (identity
`"fake"` is intentionally NOT one of resolve_components()'s supported
identities).

Includes deliberate misbehavior knobs (`identity_override_*`,
`field_overrides_by_key`) so tests can simulate a buggy real component --
these exist specifically to exercise the orchestrator's/factory's boundary
validation (codex_code_review_44d9e90.md finding 4), not because a fake
should misbehave by default.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Literal

from ft_screening.components import AdjudicationOutcome, AdjudicationUsage, Chunk, ParsedDocument
from ft_screening.schema import CriterionAssessment, CriterionSpec, DocumentCoverage, EvidenceSpan

Key = tuple[str, str]  # (document_id, criterion_id)


class FakeDocumentParser:
    def __init__(
        self,
        *,
        coverage_by_document_id: dict[str, DocumentCoverage] | None = None,
        chunks_by_document_id: dict[str, list[Chunk]] | None = None,
        fail_document_ids: set[str] | None = None,
        identity_override_by_document_id: dict[str, tuple[str, str]] | None = None,
    ) -> None:
        self._coverage_by_document_id = coverage_by_document_id or {}
        self._chunks_by_document_id = chunks_by_document_id or {}
        self._fail_document_ids = fail_document_ids or set()
        self._identity_override_by_document_id = identity_override_by_document_id or {}

    def parse(
        self,
        *,
        document_id: str,
        document_sha256: str,
        pdf_path: Path,
        parser_options: dict[str, Any],
    ) -> ParsedDocument:
        if document_id in self._fail_document_ids:
            raise RuntimeError(f"FakeDocumentParser: configured to fail for document_id={document_id!r}")

        coverage = self._coverage_by_document_id.get(document_id) or DocumentCoverage(
            status="complete",
            total_pages=1,
            parsed_pages=[1],
            failed_pages=[],
            ocr_used=False,
            extracted_characters=64,
            warnings=[],
        )
        # chunk_id derived from CONTENT (document_sha256), not the caller's
        # document_id label -- a real content-addressed chunker would
        # produce the same chunk_id for identical bytes regardless of what
        # ID the caller attaches, which cross-run cache validation
        # (orchestrator._validate_evidence_boundary) depends on.
        # `document_id in ...` (not a falsy-value check) -- an explicitly
        # empty override ([]) must mean zero chunks, not "no override was
        # given, fall back to the default one-chunk list". `[] or default`
        # would silently discard a deliberately empty override since an
        # empty list is falsy in Python.
        if document_id in self._chunks_by_document_id:
            chunks = self._chunks_by_document_id[document_id]
        else:
            # Derived from the CALLER-SUPPLIED coverage (codex_code_review_c4f274a.md
            # finding 3 exposed this): a hardcoded page=1/end_char=63 chunk was
            # only ever coincidentally consistent with whatever coverage a test
            # fixture happened to pass. Once ParsedDocument started validating
            # chunks against coverage, tests that set a smaller
            # extracted_characters (to simulate partial extraction) or an empty
            # parsed_pages (to simulate total failure) started tripping that
            # invariant on a chunk they never meant to be self-consistent with.
            # A default chunk must always be a *valid* ParsedDocument for
            # whatever coverage the caller asked for, the same way a real
            # parser's own chunker output would be.
            if coverage.extracted_characters > 0 and coverage.parsed_pages:
                default_page = coverage.parsed_pages[0]
                full_text = f"Fake parsed content for document {document_id} used for deterministic testing."
                end_char = min(len(full_text), coverage.extracted_characters)
                chunks = [
                    Chunk(
                        chunk_id=f"{document_sha256[:12]}-chunk-0",
                        section="Methods",
                        page=default_page,
                        start_char=0,
                        end_char=end_char,
                        text=full_text[:end_char],
                    )
                ]
            else:
                # No successfully parsed pages / no extracted text -- a real
                # parser would not fabricate a chunk here either.
                chunks = []
        returned_document_id, returned_document_sha256 = self._identity_override_by_document_id.get(
            document_id, (document_id, document_sha256)
        )
        return ParsedDocument(
            document_id=returned_document_id,
            document_sha256=returned_document_sha256,
            coverage=coverage,
            chunks=tuple(chunks),
        )


class FakeEvidenceRetriever:
    def __init__(
        self,
        *,
        evidence_by_key: dict[Key, list[EvidenceSpan]] | None = None,
        fail_keys: set[Key] | None = None,
    ) -> None:
        self._evidence_by_key = evidence_by_key or {}
        self._fail_keys = fail_keys or set()

    def retrieve(
        self,
        *,
        document: ParsedDocument,
        criterion: CriterionSpec,
        retriever_options: dict[str, Any],
    ) -> list[EvidenceSpan]:
        key = (document.document_id, criterion.criterion_id)
        if key in self._fail_keys:
            raise RuntimeError(f"FakeEvidenceRetriever: configured to fail for {key!r}")
        if key in self._evidence_by_key:
            return list(self._evidence_by_key[key])
        if not document.chunks:
            return []
        chunk = document.chunks[0]
        return [
            EvidenceSpan(
                document_id=document.document_id,
                document_sha256=document.document_sha256,
                chunk_id=chunk.chunk_id,
                section=chunk.section,
                page=chunk.page,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                excerpt=chunk.text,
                stance="relevant",
                retrieval_method="fake",
                retrieval_query_version="1",
                rank=1,
                score=1.0,
            )
        ]


class FakeCriterionAdjudicator:
    """`model`/`prompt_template_version`/`output_schema_version` must match
    the ScreeningConfig the fake will be used under -- the orchestrator
    validates a real (or fake) adjudicator's output against those
    configured identities (codex_code_review_44d9e90.md finding 4), so a
    fake that doesn't echo them back would fail that check itself. Defaults
    here match this project's own test-helper `make_config()` convention
    (adjudicator_model="fake-1", *_version="1"); override if a test's
    config differs.
    """

    def __init__(
        self,
        *,
        status_by_key: dict[Key, Literal["satisfied", "violated", "unclear", "not_reported"]] | None = None,
        requires_review_by_key: dict[Key, bool] | None = None,
        fail_keys: set[Key] | None = None,
        delay_by_key: dict[Key, float] | None = None,
        field_overrides_by_key: dict[Key, dict[str, Any]] | None = None,
        call_log: list[tuple[Key, float]] | None = None,
        model: str = "fake-1",
        prompt_template_version: str = "1",
        output_schema_version: str = "1",
    ) -> None:
        self._status_by_key = status_by_key or {}
        self._requires_review_by_key = requires_review_by_key or {}
        self._fail_keys = fail_keys or set()
        self._delay_by_key = delay_by_key or {}
        self._field_overrides_by_key = field_overrides_by_key or {}
        self._call_log = call_log
        self._call_log_lock = threading.Lock()
        self._model = model
        self._prompt_template_version = prompt_template_version
        self._output_schema_version = output_schema_version

    def adjudicate(
        self,
        *,
        document: ParsedDocument,
        criterion: CriterionSpec,
        evidence: list[EvidenceSpan],
        model_options: dict[str, Any],
    ) -> AdjudicationOutcome:
        key = (document.document_id, criterion.criterion_id)
        if self._call_log is not None:
            with self._call_log_lock:
                self._call_log.append((key, time.monotonic()))
        if key in self._fail_keys:
            raise RuntimeError(f"FakeCriterionAdjudicator: configured to fail for {key!r}")
        delay = self._delay_by_key.get(key)
        if delay:
            time.sleep(delay)
        status = self._status_by_key.get(key, "satisfied")
        requires_review = self._requires_review_by_key.get(key, False)
        fields: dict[str, Any] = dict(
            criterion_id=criterion.criterion_id,
            criterion_version=criterion.version,
            status=status,
            evidence=evidence,
            rationale=f"fake adjudication for {key!r}: status={status!r}",
            requires_human_review=requires_review,
            model=self._model,
            prompt_template_version=self._prompt_template_version,
            output_schema_version=self._output_schema_version,
        )
        fields.update(self._field_overrides_by_key.get(key, {}))
        assessment = CriterionAssessment(**fields)
        # A fake has no real provider call to report on -- usage_available=False
        # (never a confirmed zero) makes this distinguishable from a real
        # adjudicator's genuine zero-token result.
        usage = AdjudicationUsage(
            model=self._model, input_tokens=0, output_tokens=0, total_tokens=0,
            provider_call_count=1, retry_count=0, usage_available=False,
        )
        return AdjudicationOutcome(assessment=assessment, usage=usage)
