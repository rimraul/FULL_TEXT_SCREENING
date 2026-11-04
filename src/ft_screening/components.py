"""Private orchestration interfaces (api_contract_v2.md correction 3).

Not part of the public contract -- `document.py`/`retrieval.py`/
`adjudication.py`/`persistence.py` (plan_v5.md steps 6/7) will provide real
implementations of these Protocols later. For now only `fakes.py`
(deterministic, test-controllable) and `store.py` (a real, minimal,
filesystem-backed ScreeningStore) implement them.

`ParsedDocument`/`Chunk` are internal intermediate types -- NOT part of
api_contract_v2.md's public schema. They exist so DocumentParser and
EvidenceRetriever have a concrete return/input type before real parsing
exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ft_screening.schema import (
    CriterionAssessment,
    CriterionSpec,
    DocumentCoverage,
    EvidenceSpan,
    ScreeningItemResult,
    ScreeningRunResult,
    _validate_safe_token,
)


class CachedAssessment(BaseModel):
    """An assessment plus which run originally computed it -- a bare
    CriterionAssessment can't carry cross-run cache-hit provenance.

    Pydantic + revalidate_instances="always" (codex_code_review_0f48a3d.md
    finding 3): this is returned by ScreeningStore.get_cached_assessment(),
    and the orchestrator must not assume every ScreeningStore
    implementation validates its own return value the way
    FileScreeningStore's internal envelope does. A protocol-conforming but
    buggy (or adversarial) alternate store can construct and return a
    syntactically-valid CachedAssessment with source_run_id set to
    something that isn't a plain safe-token string (a list, e.g.) -- that
    previously escaped uncaught all the way to `set(cache_source_run_ids)`
    in the orchestrator. Was a bare dataclass; only FileScreeningStore's
    own envelope was ever validated, not this protocol-level type every
    store implementation returns."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    assessment: CriterionAssessment
    source_run_id: str = Field(min_length=1)

    @field_validator("source_run_id", mode="after")
    @classmethod
    def _validate_source_run_id(cls, v: str) -> str:
        return _validate_safe_token(v)


class Chunk(BaseModel):
    """Pydantic, not a bare dataclass (codex_code_review_53f21de.md finding
    2): a plain dataclass does not enforce field types at construction, so
    a malformed parser could build a ParsedDocument whose `chunks` tuple
    contains non-Chunk objects and it would go entirely unnoticed until
    some later attribute access crashed uncaught.

    `revalidate_instances="always"` matters here specifically: by
    Pydantic's own default (`"never"`), calling `ParsedDocument.model_validate()`
    on an object that is ALREADY an instance of the class short-circuits
    via `isinstance` and skips re-checking nested fields entirely --
    verified empirically, `model_validate()` on a `model_construct()`-built
    instance with `chunks=(object(),)` passed straight through unless this
    is set. The orchestrator calls `model_validate()` on every parser
    return value specifically to defend against a malformed component that
    doesn't go through normal field-by-field construction; without this
    setting that defense would be a no-op for exactly the case it exists
    to catch."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    chunk_id: str = Field(min_length=1)
    section: str | None = None
    page: int | None = Field(default=None, ge=1)
    # DOCUMENT-GLOBAL offsets (codex_code_review_c4f274a.md finding 3's
    # 4th bullet: "document that coordinate convention first") -- position
    # within the full document's extracted text, NOT position within this
    # chunk's own `text` string. Chosen over chunk-local because citing
    # "document character offset 1523-1601" is directly useful downstream
    # (e.g. for EvidenceSpan citations spanning chunk boundaries); a
    # chunk-local convention would require also knowing chunk boundaries
    # to reconstruct a document-wide position.
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    text: str

    @model_validator(mode="after")
    def _span_order(self) -> "Chunk":
        # finding 4: Chunk lacked this basic local invariant, unlike the
        # public EvidenceSpan which already has the equivalent check.
        if self.end_char < self.start_char:
            raise ValueError("end_char must be >= start_char")
        return self


class ParsedDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    document_id: str
    document_sha256: str
    coverage: DocumentCoverage
    chunks: tuple[Chunk, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def _unique_chunk_ids(self) -> "ParsedDocument":
        # finding 4 (round 3): without this, two different chunks
        # (different text/section/offsets) could share one chunk_id -- any
        # EvidenceSpan referencing that ID is then ambiguous between two
        # distinct pieces of text, undermining the provenance chunk_id
        # exists to provide in the first place.
        ids = [c.chunk_id for c in self.chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("chunk_id values must be unique within a ParsedDocument")
        return self

    @model_validator(mode="after")
    def _chunks_consistent_with_coverage(self) -> "ParsedDocument":
        # finding 3 (round 5): chunks and coverage were validated
        # independently, with nothing relating them -- a parser could
        # report coverage.total_pages=1/parsed_pages=[1] while a chunk
        # claimed page=999, and revalidation accepted it: public evidence
        # could then cite a page the SAME result says doesn't exist.
        for chunk in self.chunks:
            if chunk.page is not None and chunk.page not in self.coverage.parsed_pages:
                # Subsumes both an out-of-range page (DocumentCoverage's own
                # validator already bounds parsed_pages by total_pages) and
                # a failed-page reference (parsed_pages/failed_pages are
                # already required to be disjoint) -- one check catches both.
                raise ValueError(
                    f"chunk {chunk.chunk_id!r} references page {chunk.page}, which is not in "
                    f"coverage.parsed_pages={self.coverage.parsed_pages!r}"
                )
            if chunk.end_char > self.coverage.extracted_characters:
                raise ValueError(
                    f"chunk {chunk.chunk_id!r} end_char={chunk.end_char} exceeds "
                    f"coverage.extracted_characters={self.coverage.extracted_characters!r} "
                    "(chunk offsets are document-global -- see Chunk.start_char/end_char)"
                )
        if (
            self.coverage.extracted_characters > 0
            and self.chunks
            and all(len(c.text.strip()) == 0 for c in self.chunks)
        ):
            raise ValueError(
                "coverage reports extracted_characters > 0 but every chunk has empty text -- "
                "positive extracted coverage cannot be represented solely by empty chunks"
            )
        return self


class AdjudicationUsage(BaseModel):
    """Provider-neutral usage telemetry for ONE `CriterionAdjudicator.adjudicate()`
    call, aggregated across every provider attempt that call made (a retry
    contributes to `provider_call_count`/`retry_count` but not to the
    reported token counts -- a failed/timed-out attempt has no completion
    to bill). Operational telemetry only -- never participates in
    `criterion_cache_key`/`execution_fingerprint` and never feeds
    `aggregation.py`'s eligibility decision logic; no currency/cost
    calculation happens here either, since prices change -- persist model
    + token counts so cost can be computed externally.

    `usage_available=False` means real token counts were not obtained for
    this call (e.g. a fake/legacy adjudicator, or a provider that omitted
    usage) -- distinct from a confirmed zero (e.g. the orchestrator
    synthesizes a `usage_available=True`, all-zero record for a cache hit,
    since it KNOWS no provider call happened this run; see orchestrator.py)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    provider_call_count: int = Field(ge=0)
    retry_count: int = Field(ge=0)
    usage_available: bool
    # Reasoning-model telemetry (e.g. gpt-5.6-terra): None when the
    # provider/transport doesn't expose it (every non-reasoning model, and
    # any fake/test transport) -- same "None vs. confirmed 0" discipline as
    # usage_available.
    reasoning_tokens: int | None = Field(default=None, ge=0)
    # The exact model string the PROVIDER echoed back (distinct from
    # `model` above, which is the identity the caller REQUESTED and the
    # only field that participates in fingerprinting/identity). Purely
    # observational -- see TransportUsage.provider_returned_model.
    provider_returned_model: str | None = Field(default=None)

    @model_validator(mode="after")
    def _retry_count_consistent(self) -> "AdjudicationUsage":
        expected_retries = max(self.provider_call_count - 1, 0)
        if self.retry_count != expected_retries:
            raise ValueError(
                f"retry_count={self.retry_count} inconsistent with provider_call_count={self.provider_call_count} "
                f"(expected retry_count={expected_retries})"
            )
        return self


class AdjudicationOutcome(BaseModel):
    """Return type of `CriterionAdjudicator.adjudicate()` -- the assessment
    plus the usage telemetry for producing it, threaded through the SAME
    boundary the orchestrator already revalidates (`revalidate_instances`
    below), rather than a parallel side channel a malformed/adversarial
    adjudicator could bypass."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    assessment: CriterionAssessment
    usage: AdjudicationUsage


class DocumentParser(Protocol):
    def parse(
        self,
        *,
        document_id: str,
        document_sha256: str,
        pdf_path: Path,
        parser_options: dict[str, Any],
    ) -> ParsedDocument:
        """`document_sha256` is supplied by the orchestrator (computed once,
        for real, from the actual file bytes, before any parser runs) --
        the parser must echo it back on the returned ParsedDocument rather
        than recomputing it, so there is exactly one source of truth for
        content identity.
        """
        ...


class EvidenceRetriever(Protocol):
    def retrieve(
        self,
        *,
        document: ParsedDocument,
        criterion: CriterionSpec,
        retriever_options: dict[str, Any],
    ) -> list[EvidenceSpan]: ...


class CriterionAdjudicator(Protocol):
    def adjudicate(
        self,
        *,
        document: ParsedDocument,
        criterion: CriterionSpec,
        evidence: list[EvidenceSpan],
        model_options: dict[str, Any],
    ) -> AdjudicationOutcome: ...


class ScreeningStore(Protocol):
    """Run-scoped artifact persistence, an append-only ledger, per-run
    exclusive locking, resume support, and a content-addressed
    per-criterion assessment cache. See api_contract_v2.md's "Artifacts
    namespaced by run" and "Ordering, caching, resume, idempotency"
    sections for the guarantees an implementation must uphold.
    """

    @property
    def output_dir(self) -> Path:
        """Must equal (be rooted at) the ScreeningRequest.output_dir of any
        request this store is used with -- the orchestrator asserts this."""
        ...

    def begin_run(self, *, run_id: str, resume: bool) -> None:
        """Create (resume=False) or reopen (resume=True) the run directory
        and acquire its exclusive lock. Raises RunSetupError on a fresh-run
        collision, ResumeConflictError if a resume target doesn't exist or
        is already locked by another process.
        """
        ...

    def end_run(self, *, run_id: str) -> None:
        """Release the run's exclusive lock."""
        ...

    def write_run_manifest(
        self,
        *,
        run_id: str,
        execution_fingerprint: str,
        request_fingerprint: str | None,
        provisional_request_binding: str | None = None,
        document_hash_checkpoints: dict[str, str] | None = None,
    ) -> None:
        """`request_fingerprint` may be None: a PROVISIONAL manifest,
        written immediately after the run lock is acquired but before file
        hashing has produced a real request_fingerprint, so the run stays
        resumable even if interrupted before that computation finishes
        (codex_code_review_c4f274a.md finding 1). The final manifest,
        written once request_fingerprint is known, always has a real
        value.

        `provisional_request_binding` (codex_code_review_810ccf0.md finding
        1) is the cheap pre-hash request identity check
        (`hashing.compute_provisional_request_binding`) that protects the
        provisional window: while `request_fingerprint` is still None, a
        resume attempt has nothing else to verify "same request" against
        before doing any file I/O, and `execution_fingerprint` alone
        excludes item identities/citation/policies entirely. A store
        implementation just needs to persist and return this value
        alongside the other two fields; only the orchestrator interprets
        it.

        Implementations must persist this write ATOMICALLY: the moment a
        manifest becomes durably readable via `read_run_manifest` is the
        one and only "this run_id is claimed" signal a fresh run
        (`resume=False`) collision-checks against and a resume
        (`resume=True`) existence-checks against. A run directory (or
        other on-disk artifact) existing with no manifest yet must never,
        by itself, be treated as "claimed" by either check
        (codex_code_review_810ccf0.md finding 2 -- FileScreeningStore no
        longer pre-creates a run directory in `begin_run()` for exactly
        this reason).

        `document_hash_checkpoints` (codex_code_review_146f98e.md's
        finding) maps document_id -> the real content SHA-256 already
        observed for it during THIS run's Stage A1, checkpointed
        incrementally (one extra manifest write per newly-hashed item) so
        that content already hashed before an interruption is durably
        remembered -- without this, `provisional_request_binding` alone
        (a PATH-based proxy, since no real content hash exists yet for a
        not-yet-reached item) could not detect the file at that path
        being silently replaced between the interrupted attempt and a
        resume. `None`/omitted means "no checkpoints yet" (a fresh run's
        first write, before Stage A1 starts)."""
        ...

    def read_run_manifest(self, *, run_id: str) -> dict[str, Any] | None: ...

    def append_ledger_event(self, *, run_id: str, event: dict[str, Any]) -> None: ...

    def read_completed_item_results(self, *, run_id: str) -> dict[int, ScreeningItemResult]:
        """input_index -> already-persisted ScreeningItemResult, for resume."""
        ...

    def write_item_result(self, *, run_id: str, item_result: ScreeningItemResult) -> None:
        """Atomically rewrites item_results.jsonl as the materialized
        current view (one line per input_index, latest write wins) --
        must happen before the item is reported complete."""
        ...

    def write_run_result(self, *, run_id: str, run_result: ScreeningRunResult) -> None:
        """Atomic write of the final run_result.json."""
        ...

    def artifact_paths(self, *, run_id: str) -> tuple[Path, Path, Path]:
        """(run_result_json, item_results_jsonl, ledger_jsonl) for this run."""
        ...

    def get_cached_assessment(self, *, cache_key: str) -> CachedAssessment | None:
        """Returns the assessment plus which run originally computed it
        (cross-run cache-hit provenance) -- never a bare assessment."""
        ...

    def put_cached_assessment(self, *, cache_key: str, assessment: CriterionAssessment, source_run_id: str) -> None:
        """Caller must have already validated `assessment` against the
        current document/criterion/execution config -- an unvalidated
        assessment must never reach the cache (poisons future cache hits)."""
        ...


@dataclass(frozen=True)
class Components:
    parser: DocumentParser
    retriever: EvidenceRetriever
    adjudicator: CriterionAdjudicator
    store: ScreeningStore
