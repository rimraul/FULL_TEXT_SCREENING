"""The orchestration engine (`_run_ft_screening`) and public wrapper
(`run_ft_screening`), per api_contract_v2.md correction 3 and plan_v5.md
step 3. `_run_ft_screening` itself does no PDF parsing, retrieval, or LLM
integration directly -- it wires together whatever `Components` it's
given, real or fake. `resolve_components()` (the v0.1 explicit built-in
factory -- see its own docstring) resolves REAL implementations
(`docling`/`lexical_bm25`/`openai`) for the public `run_ft_screening()`
path; tests still construct `Components` directly with fakes and call
`_run_ft_screening` for fast, deterministic, network-free runs.

Second review round (codex_code_review_44d9e90.md) hardened this module
substantially; see inline comments tagged "finding N" for what changed and
why. Processing stays staged to keep concurrency race-free, now with an
ORDER-SENSITIVE fail-fast pass that incorporates preflight-stage fatal
outcomes (finding 8), not just Stage B failures:

  Stage A1 (always sequential): hash every available file for real (a
    streamed checksum, not "parsing"), verify expected_sha256.
  Stage A2 (always sequential): group items by content hash; under
    duplicate_content_policy="error" EVERY member of a group with more
    than one item becomes duplicate_input (finding 2 -- was previously
    only the non-first copies); under "reuse", the lowest input_index in
    each group is canonical.
  Ordering pass (always sequential): walk items 0..N-1; the first fatal
    outcome encountered (preflight OR, once Stage B has run for earlier
    items, a Stage B failure) marks every later not-yet-terminal item
    `not_processed` under fail_fast -- preflight fatals count too.
  Stage B (sequential, or a bounded thread pool that submits at most
    `max_workers` items at a time and never submits a new one once a fatal
    result has been observed -- finding 8): parse -> validate parser
    identity -> gate on document coverage (finding 3) -> per criterion:
    check+validate cache -> retrieve+validate -> adjudicate+validate
    (finding 4) -> cache only after validation (finding 6) -> aggregate.
  Stage C (sequential): resolve "reuse" duplicates from their now-complete
    canonical result.

Every terminal result (preflight, Stage B, not_processed) is persisted the
moment it's known (finding 1). A progress reporter disables itself after
its first callback exception and tracks real completion counts (finding
5). KeyboardInterrupt is caught, an interrupted snapshot is written and a
`run_interrupted` ledger event recorded, then the exception is re-raised
(finding 5).
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import logging
import threading
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

_logger = logging.getLogger(__name__)

from ft_screening.aggregation import AGGREGATION_POLICY_VERSION, AggregationError, build_eligibility_result
from ft_screening.components import AdjudicationOutcome, AdjudicationUsage, CachedAssessment, Components, ParsedDocument
from ft_screening.errors import ComponentResolutionError, ResumeConflictError, RunSetupError
from ft_screening.hashing import (
    compute_criteria_set_hash,
    compute_criterion_cache_key,
    compute_execution_fingerprint,
    compute_provisional_request_binding,
    compute_request_fingerprint,
)
from ft_screening.schema import (
    CriterionAssessment,
    CriterionSpec,
    DocumentCoverage,
    EvidenceSpan,
    ExecutionProvenance,
    ProcessingError,
    ProgressEvent,
    RunArtifacts,
    RunSummary,
    ScreeningConfig,
    ScreeningItem,
    ScreeningItemResult,
    ScreeningRequest,
    ScreeningRunResult,
)
from ft_screening.store import FileScreeningStore

API_VERSION = "1.0"
SCHEMA_VERSION = "1"

_FATAL_EXECUTION_STATUSES = {"processing_failed", "input_unavailable"}
_HASH_CHUNK_SIZE = 1 << 20  # 1 MiB -- finding 9: stream file hashing, don't read_bytes() whole files


class _BoundaryViolation(ValueError):
    """A private component's output violated its documented identity
    contract (finding 4) -- treated as a processing failure, never
    silently trusted or cached."""


class _ProgressReporter:
    """Wraps the caller's progress_cb: disables itself after its first
    exception (recorded AND logged to the ledger -- finding 5's second
    half, "must be logged" -- not just remembered in a field nobody reads)
    and tracks a real running completion count (finding 5's first half --
    was always hardcoded to 0).

    Bound to exactly one (store, run_id) for its whole life. `emit()` is
    lock-protected: Stage B's worker threads call it concurrently
    (`item_started`), so both the disabled-check and the callback
    invocation itself must be serialized -- otherwise two threads can pass
    the check before the first exception sets it, and the caller's
    callback could be invoked concurrently despite never being documented
    as thread-safe.
    """

    def __init__(
        self,
        callback: Callable[[ProgressEvent], None] | None,
        *,
        total_items: int,
        store,
        run_id: str,
    ) -> None:
        self._callback = callback
        self._disabled = False
        self._completed = 0
        self._total_items = total_items
        self._store = store
        self._run_id = run_id
        self._lock = threading.Lock()
        self.last_exception: Exception | None = None

    def item_done(self) -> None:
        with self._lock:
            self._completed += 1

    def emit(
        self,
        *,
        event: str,
        occurred_at: datetime,
        input_index: int | None = None,
        document_id: str | None = None,
        report_id: str | None = None,
        message: str | None = None,
    ) -> None:
        if self._callback is None:
            return
        with self._lock:
            if self._disabled:
                return
            progress_event = ProgressEvent(
                run_id=self._run_id,
                event=event,
                occurred_at=occurred_at,
                completed_items=self._completed,
                total_items=self._total_items,
                input_index=input_index,
                document_id=document_id,
                report_id=report_id,
                message=message,
            )
            try:
                self._callback(progress_event)
            except Exception as exc:  # noqa: BLE001 -- must never propagate
                self._disabled = True
                self.last_exception = exc
                # Security note (codex_code_review_53f21de.md): truncation
                # is not redaction -- a caller's exception message can
                # contain paths, document text, or secrets. The runtime
                # logger gets the real exception (exc_info); the PUBLIC
                # ledger artifact gets only the exception type and a fixed,
                # content-free message.
                _logger.warning(
                    "progress callback raised %s for run_id=%s; disabling further callbacks for this run",
                    type(exc).__name__,
                    self._run_id,
                    exc_info=exc,
                )
                self._store.append_ledger_event(
                    run_id=self._run_id,
                    event={
                        "type": "progress_callback_disabled",
                        "exception_type": type(exc).__name__,
                        "message": "progress callback raised an exception; further callbacks disabled for this run",
                    },
                )


def _hash_file_streaming(path: Path, *, chunk_size: int = _HASH_CHUNK_SIZE) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


# v0.1: an explicit BUILT-IN factory, not a general plugin registry --
# exactly the three identities below, nothing else. Extend this set (and
# the corresponding _resolve_* branch) when a new real implementation
# actually exists; do not generalize into a registry/entry-point mechanism
# ahead of a second real need for one.
_SUPPORTED_PARSER_NAMES = frozenset({"docling"})
_SUPPORTED_RETRIEVER_NAMES = frozenset({"lexical_bm25"})
_SUPPORTED_ADJUDICATOR_PROVIDERS = frozenset({"openai"})


def _the_one_supported_identity(names: frozenset[str], *, label: str) -> str:
    """`build_supported_screening_config()`'s single source of truth for
    each locked identity name -- reads the SAME `_SUPPORTED_*` registry
    `_resolve_parser`/`_resolve_retriever`/`_resolve_adjudicator` check
    against, rather than duplicating "docling"/"lexical_bm25"/"openai" as
    separate literals that could silently drift from the resolver's own
    registry (codex_code_review_build_supported_screening_config.md
    finding 3). This builder intentionally supports exactly ONE stack, so
    a registry that isn't a singleton is itself a configuration bug --
    fail loudly rather than arbitrarily picking one of several, or
    silently returning nothing for zero."""
    if len(names) != 1:
        raise ComponentResolutionError(
            f"build_supported_screening_config() requires exactly one supported {label} identity, "
            f"found {sorted(names)}"
        )
    return next(iter(names))


def _validate_aggregation_policy_version(config: ScreeningConfig) -> None:
    """The live aggregation implementation (`aggregation.py`) is one
    hard-coded, unversioned state machine -- nothing previously checked
    that a caller's declared `aggregation_policy_version` matched it, even
    though that value is hashed into `execution_fingerprint`
    (codex_code_review_2be6cfa.md finding 1). An arbitrary/typo'd/future
    version must fail resolution, not silently hash a policy identity that
    doesn't exist. Checked first, before any parser/retriever/adjudicator
    resolution -- cheap and import-free, so a bad aggregation_policy_version
    never triggers Docling/OpenAI work first."""
    if config.aggregation_policy_version != AGGREGATION_POLICY_VERSION:
        raise ComponentResolutionError(
            f"config.aggregation_policy_version={config.aggregation_policy_version!r} does not match "
            f"the live aggregation implementation's AGGREGATION_POLICY_VERSION={AGGREGATION_POLICY_VERSION!r}"
        )


def _resolve_parser(config: ScreeningConfig):
    if config.parser_name not in _SUPPORTED_PARSER_NAMES:
        raise ComponentResolutionError(
            f"unsupported parser_name={config.parser_name!r} -- resolve_components only supports {sorted(_SUPPORTED_PARSER_NAMES)}"
        )
    # Lazy import: importing ft_screening.orchestrator (or calling
    # resolve_components for a DIFFERENT identity) must never pull in
    # Docling's own import chain (docling.datamodel.settings/docling_core
    # etc. -- ~20s just for docling_core.types.doc, confirmed empirically
    # in document.py's own module docstring).
    from ft_screening.document import DoclingDocumentParser, _parser_options_model

    # Wrapped -- a missing/broken docling distribution raises a raw
    # importlib.metadata.PackageNotFoundError otherwise, escaping the one
    # consistent ComponentResolutionError boundary this factory otherwise
    # promises (codex_code_review_2be6cfa.md finding 3).
    try:
        installed_version = DoclingDocumentParser.installed_parser_version()
    except Exception as exc:
        raise ComponentResolutionError(
            f"could not determine the installed docling package version -- docling may not be installed or its "
            f"package metadata is unavailable: {type(exc).__name__}"
        ) from exc
    if config.parser_version != installed_version:
        raise ComponentResolutionError(
            f"config.parser_version={config.parser_version!r} does not match the installed docling package "
            f"version {installed_version!r}"
        )
    if config.chunker_version != DoclingDocumentParser.CHUNKER_VERSION:
        raise ComponentResolutionError(
            f"config.chunker_version={config.chunker_version!r} does not match "
            f"DoclingDocumentParser.CHUNKER_VERSION={DoclingDocumentParser.CHUNKER_VERSION!r}"
        )
    try:
        _parser_options_model().model_validate(config.parser_options)
    except Exception as exc:
        raise ComponentResolutionError(f"config.parser_options is incompatible with DoclingDocumentParser: {type(exc).__name__}") from None

    # Default artifacts_path (None -> DoclingDocumentParser's own resolved
    # default, e.g. Docling's settings.cache_dir/"models") -- v0.1 exposes
    # no config surface to override this; construction itself is cheap
    # (artifacts are resolved/validated lazily on first real parse, not here).
    return DoclingDocumentParser()


def _resolve_retriever(config: ScreeningConfig):
    if config.retriever_name not in _SUPPORTED_RETRIEVER_NAMES:
        raise ComponentResolutionError(
            f"unsupported retriever_name={config.retriever_name!r} -- resolve_components only supports {sorted(_SUPPORTED_RETRIEVER_NAMES)}"
        )
    from ft_screening.retrieval import LexicalEvidenceRetriever, _RetrieverOptionsModel

    if config.retriever_config_version != LexicalEvidenceRetriever.RETRIEVER_CONFIG_VERSION:
        raise ComponentResolutionError(
            f"config.retriever_config_version={config.retriever_config_version!r} does not match "
            f"LexicalEvidenceRetriever.RETRIEVER_CONFIG_VERSION={LexicalEvidenceRetriever.RETRIEVER_CONFIG_VERSION!r}"
        )
    try:
        _RetrieverOptionsModel.model_validate(config.retriever_options)
    except Exception as exc:
        raise ComponentResolutionError(f"config.retriever_options is incompatible with LexicalEvidenceRetriever: {type(exc).__name__}") from None

    return LexicalEvidenceRetriever()


def _resolve_adjudicator(config: ScreeningConfig):
    if config.adjudicator_provider not in _SUPPORTED_ADJUDICATOR_PROVIDERS:
        raise ComponentResolutionError(
            f"unsupported adjudicator_provider={config.adjudicator_provider!r} -- resolve_components only supports "
            f"{sorted(_SUPPORTED_ADJUDICATOR_PROVIDERS)}"
        )
    # Lazy import: must never construct an OpenAI client (or even import
    # the `openai` package) for a request that resolves a DIFFERENT
    # adjudicator_provider, and must never happen merely from importing
    # this module.
    from ft_screening.adjudication import LLMCriterionAdjudicator, OpenAIChatCompletionTransport, _ModelOptionsModel

    if config.prompt_template_version != LLMCriterionAdjudicator.PROMPT_TEMPLATE_VERSION:
        raise ComponentResolutionError(
            f"config.prompt_template_version={config.prompt_template_version!r} does not match "
            f"LLMCriterionAdjudicator.PROMPT_TEMPLATE_VERSION={LLMCriterionAdjudicator.PROMPT_TEMPLATE_VERSION!r}"
        )
    if config.output_schema_version != LLMCriterionAdjudicator.OUTPUT_SCHEMA_VERSION:
        raise ComponentResolutionError(
            f"config.output_schema_version={config.output_schema_version!r} does not match "
            f"LLMCriterionAdjudicator.OUTPUT_SCHEMA_VERSION={LLMCriterionAdjudicator.OUTPUT_SCHEMA_VERSION!r}"
        )
    try:
        _ModelOptionsModel.model_validate(config.model_options)
    except Exception as exc:
        raise ComponentResolutionError(f"config.model_options is incompatible with LLMCriterionAdjudicator: {type(exc).__name__}") from None

    # resolve_components() NEVER loads credentials itself (no my_api_keys
    # import, no .env loading) -- downstream projects provide OPENAI_API_KEY
    # (or an equivalent) through their own environment before calling this.
    # Missing/invalid credentials surface HERE, at OpenAI client
    # construction -- the earliest point this factory can actually detect
    # them -- wrapped in ComponentResolutionError for a consistent,
    # library-wide error type rather than a raw openai.OpenAIError leaking
    # through unannotated.
    try:
        transport = OpenAIChatCompletionTransport()
    except Exception as exc:
        raise ComponentResolutionError(
            "failed to construct the OpenAI transport -- likely missing/invalid credentials in the environment "
            "(resolve_components never loads credentials itself; the caller's environment must already provide "
            f"OPENAI_API_KEY before calling it): {type(exc).__name__}"
        ) from exc

    return LLMCriterionAdjudicator(transport=transport, model=config.adjudicator_model)


def resolve_components(config: ScreeningConfig, *, output_dir: Path) -> Components:
    """The v0.1 explicit built-in factory: resolves REAL implementations
    for exactly three identities (`parser_name="docling"`,
    `retriever_name="lexical_bm25"`, `adjudicator_provider="openai"`) --
    not a general plugin registry; extend the `_SUPPORTED_*` sets above
    plus a new `_resolve_*` branch when a genuinely new real
    implementation exists, rather than generalizing ahead of that need.

    `output_dir` is a required, explicit keyword argument (not read off
    `config`, which has no such field) -- `FileScreeningStore` must be
    rooted at the SAME directory the calling `ScreeningRequest` itself
    declares (`_run_ft_screening` already asserts this equality; passing
    it in here rather than letting each caller construct its own store
    keeps that binding centralized).

    Every declared identity/version is validated (parser/chunker version
    against the live `DoclingDocumentParser`, retriever config version
    against the live `LexicalEvidenceRetriever`, prompt/schema version
    against the live `LLMCriterionAdjudicator`) BEFORE any PDF parsing or
    provider call happens -- construction alone (this function returning)
    never touches the network and never parses a PDF. All four
    sub-resolvers use LAZY imports, so calling `resolve_components` for
    one identity set never pulls in Docling or the `openai` package for
    identities it didn't ask for, and merely importing
    `ft_screening.orchestrator` never does either.

    `config.aggregation_policy_version` is validated first, against the
    live aggregation implementation's own version identity -- before any
    parser/retriever/adjudicator resolution, PDF parsing, or provider work
    (api_contract_v3.md correction 9)."""
    _validate_aggregation_policy_version(config)
    parser = _resolve_parser(config)
    retriever = _resolve_retriever(config)
    adjudicator = _resolve_adjudicator(config)
    store = FileScreeningStore(output_dir=output_dir)
    return Components(parser=parser, retriever=retriever, adjudicator=adjudicator, store=store)


def build_supported_screening_config(
    *,
    adjudicator_model: str,
    model_config_id: str,
    model_options: dict[str, Any],
    parser_do_ocr: bool = True,
    parser_do_table_structure: bool = True,
    retriever_options: dict[str, Any] | None = None,
    cache_mode: Literal["use", "refresh", "off"] = "use",
    duplicate_content_policy: Literal["error", "reuse"] = "error",
    item_error_policy: Literal["continue", "fail_fast"] = "continue",
    max_workers: int = 1,
) -> ScreeningConfig:
    """Builds a `ScreeningConfig` for the ONE built-in supported stack
    (`resolve_components()` resolves: docling/lexical_bm25/openai) --
    "supported" because it binds exactly that stack's identities, not a
    general config constructor (api_contract_v4.md correction 10). A
    caller wanting a different stack constructs `ScreeningConfig` directly;
    this helper deliberately cannot express that.

    LOCKED, never caller-overridable (no parameter accepts them; this is
    structural, not a runtime check): `parser_name`, `retriever_name`,
    `adjudicator_provider` (read from `resolve_components()`'s own
    `_SUPPORTED_*` registries, never duplicated as separate literals --
    see `_the_one_supported_identity()`), `parser_version` (read from the
    installed docling package), `chunker_version`, `retriever_config_version`,
    `prompt_template_version`, `output_schema_version`,
    `aggregation_policy_version`, and the Docling `artifacts_manifest_digest`
    (read from the committed artifact manifest). All ten are read from the
    LIVE implementation's own version constants/registries/manifest at
    call time, never hardcoded literals here -- if a component's identity
    or version constant changes, this function's output changes with it
    automatically.

    REQUIRED, no defaults (never silently choose a model/reasoning
    effort/token budget for the caller): `adjudicator_model`,
    `model_config_id`, `model_options` -- and `model_options` must itself
    be a dict that explicitly sets BOTH `max_completion_tokens` AND
    `reasoning_effort` (use `None` for a non-reasoning model -- the
    transport conditionally omits the provider parameter in that case,
    see `_ModelOptionsModel`); this function does NOT fall back to
    `_ModelOptionsModel`'s own defaults for either field, on purpose
    (SLR_chapter will define its own transparent teaching preset on top
    of this, separately).

    CUSTOMIZABLE, keyword-only: `parser_do_ocr`/`parser_do_table_structure`
    (Docling behavior -- deliberately NOT a raw `parser_options` dict
    passthrough, since that dict is also where `artifacts_manifest_digest`
    lives; exposing only these two named booleans makes it structurally
    impossible to pass artifact identity through this helper),
    `retriever_options` (no identity-sensitive field lives in this one, so
    a validated raw dict is safe), `model_options` beyond the two mandatory
    fields above, and the four runtime-only policy fields (`cache_mode`,
    `duplicate_content_policy`, `item_error_policy`, `max_workers`).

    Every option dict is validated through the SAME real per-component
    schema `resolve_components()` itself validates against
    (`_parser_options_model`/`_RetrieverOptionsModel`/`_ModelOptionsModel`),
    then the VALIDATED model is dumped back out -- so the returned
    `ScreeningConfig` is fully self-describing: every effective default
    (`do_table_structure=True`, `max_results=5`, `timeout_seconds=60.0`,
    ...) is materialized into the dict, not left hidden. Same inputs
    against the same live implementation state always produce identical
    output (no randomness, no clock, no environment read beyond package
    metadata and the small committed Docling artifact manifest).

    No side effects: no credentials required, no OpenAI client
    constructed, no network call, no Docling models loaded, no
    directory/file created, `my_api_keys` never imported -- this function
    only reads package metadata and the committed artifact manifest, and
    validates option dicts in memory.

    Errors are sanitized `ComponentResolutionError`s (never a raw
    `pydantic.ValidationError`/`TypeError` payload or a local filesystem
    path) for: a non-dict `model_options`, missing `max_completion_tokens`
    or `reasoning_effort`, an unavailable/unreadable docling installation
    or artifact manifest, any incompatible option dict, or a
    misconfigured (empty or non-singleton) `_SUPPORTED_*` identity
    registry.

    For RESUMING an existing run, use that run's own previously persisted
    `ScreeningConfig` (from its `run_result.json`/manifest), never
    regenerate one from this function -- live constants can advance
    between runs, and a resumed run must stay bound to the exact
    configuration it was originally executed under."""
    if not isinstance(model_options, dict):
        raise ComponentResolutionError(f"model_options must be a dict, got {type(model_options).__name__}")
    if "max_completion_tokens" not in model_options:
        raise ComponentResolutionError(
            "model_options must explicitly set max_completion_tokens -- build_supported_screening_config() "
            "never silently selects a token budget"
        )
    if "reasoning_effort" not in model_options:
        raise ComponentResolutionError(
            "model_options must explicitly set reasoning_effort (use None for a non-reasoning model) -- "
            "build_supported_screening_config() never silently selects it"
        )

    from ft_screening.adjudication import LLMCriterionAdjudicator, _ModelOptionsModel
    from ft_screening.document import DoclingDocumentParser, _parser_options_model
    from ft_screening.retrieval import LexicalEvidenceRetriever, _RetrieverOptionsModel

    parser_name = _the_one_supported_identity(_SUPPORTED_PARSER_NAMES, label="parser_name")
    retriever_name = _the_one_supported_identity(_SUPPORTED_RETRIEVER_NAMES, label="retriever_name")
    adjudicator_provider = _the_one_supported_identity(_SUPPORTED_ADJUDICATOR_PROVIDERS, label="adjudicator_provider")

    try:
        installed_parser_version = DoclingDocumentParser.installed_parser_version()
    except Exception as exc:
        raise ComponentResolutionError(
            f"could not determine the installed docling package version: {type(exc).__name__}"
        ) from exc

    try:
        parser_options_dict = DoclingDocumentParser.recommended_parser_options(
            do_ocr=parser_do_ocr, do_table_structure=parser_do_table_structure
        )
        parser_options = _parser_options_model().model_validate(parser_options_dict)
    except ComponentResolutionError:
        raise
    except Exception as exc:
        raise ComponentResolutionError(f"could not build parser_options: {type(exc).__name__}") from None

    try:
        retriever_options_model = _RetrieverOptionsModel.model_validate(
            {} if retriever_options is None else retriever_options
        )
    except Exception as exc:
        raise ComponentResolutionError(f"retriever_options is incompatible with LexicalEvidenceRetriever: {type(exc).__name__}") from None

    try:
        model_options_model = _ModelOptionsModel.model_validate(model_options)
    except Exception as exc:
        raise ComponentResolutionError(f"model_options is incompatible with LLMCriterionAdjudicator: {type(exc).__name__}") from None

    try:
        return ScreeningConfig(
            parser_name=parser_name,
            parser_version=installed_parser_version,
            chunker_version=DoclingDocumentParser.CHUNKER_VERSION,
            parser_options=parser_options.model_dump(mode="json"),
            retriever_name=retriever_name,
            retriever_config_version=LexicalEvidenceRetriever.RETRIEVER_CONFIG_VERSION,
            retriever_options=retriever_options_model.model_dump(mode="json"),
            adjudicator_provider=adjudicator_provider,
            adjudicator_model=adjudicator_model,
            model_config_id=model_config_id,
            model_options=model_options_model.model_dump(mode="json"),
            prompt_template_version=LLMCriterionAdjudicator.PROMPT_TEMPLATE_VERSION,
            output_schema_version=LLMCriterionAdjudicator.OUTPUT_SCHEMA_VERSION,
            aggregation_policy_version=AGGREGATION_POLICY_VERSION,
            cache_mode=cache_mode,
            duplicate_content_policy=duplicate_content_policy,
            item_error_policy=item_error_policy,
            max_workers=max_workers,
        )
    except ComponentResolutionError:
        raise
    except Exception as exc:
        raise ComponentResolutionError(f"could not construct a valid ScreeningConfig: {type(exc).__name__}") from None


def run_ft_screening(
    request: ScreeningRequest,
    *,
    progress_cb: Callable[[ProgressEvent], None] | None = None,
) -> ScreeningRunResult:
    components = resolve_components(request.config, output_dir=request.output_dir)
    return _run_ft_screening(request, components, progress_cb=progress_cb)


def _run_ft_screening(
    request: ScreeningRequest,
    components: Components,
    *,
    progress_cb: Callable[[ProgressEvent], None] | None = None,
) -> ScreeningRunResult:
    store = components.store

    # finding 10: the injected store must be rooted at this request's own
    # output_dir, or returned artifact paths would silently contradict it.
    if store.output_dir.resolve() != Path(request.output_dir).resolve():
        raise RunSetupError(
            f"components.store is rooted at {store.output_dir!r}, which does not match "
            f"request.output_dir={request.output_dir!r}"
        )

    # codex_code_review_5105135.md finding 3: if the adjudicator exposes
    # its own immutable identity (model/prompt_template_version/
    # output_schema_version -- LLMCriterionAdjudicator does; a component
    # that doesn't is simply skipped here, e.g. FakeCriterionAdjudicator),
    # validate it against request.config ONCE, before parsing/retrieval/
    # any provider work happens for ANY item -- a configuration typo
    # (wrong adjudicator_model, prompt_template_version, or
    # output_schema_version) would otherwise only be caught by the
    # existing post-call _validate_assessment_boundary(), which still
    # runs unchanged as defense in depth, but only after a real,
    # potentially costly provider call already happened -- repeated once
    # per uncached criterion/item.
    #
    # codex_code_review_5a0b49f.md finding 2: `getattr(..., None)` cannot
    # distinguish "this attribute does not exist" from "this attribute
    # exists and is literally None" -- a misconfigured/buggy adjudicator
    # whose `.model` (etc.) property returned None would have silently
    # SKIPPED this preflight entirely (treated as absent) and been let
    # through to real component work, exactly the outcome this check
    # exists to prevent. A dedicated sentinel (never a value any real
    # identity property would return) makes "absent" and "present but
    # None" distinguishable again.
    _identity_not_exposed = object()
    for attr_name, config_attr_name in (
        ("model", "adjudicator_model"),
        ("prompt_template_version", "prompt_template_version"),
        ("output_schema_version", "output_schema_version"),
    ):
        declared = getattr(components.adjudicator, attr_name, _identity_not_exposed)
        expected = getattr(request.config, config_attr_name)
        if declared is not _identity_not_exposed and declared != expected:
            raise RunSetupError(
                f"components.adjudicator.{attr_name}={declared!r} does not match "
                f"request.config.{config_attr_name}={expected!r}"
            )

    try:
        request.output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RunSetupError(f"cannot create/write output_dir={request.output_dir!r}: {exc}") from exc
    pdf_paths_resolved = {item.pdf_path.resolve() for item in request.items}
    if request.output_dir.resolve() in pdf_paths_resolved:
        raise RunSetupError("output_dir must not be one of the source PDF paths")

    # codex_code_review_0f48a3d.md finding 1: computed BEFORE begin_run()
    # entirely, not just before the main try -- an interrupt here now
    # leaves NO run directory at all (nothing was created yet), instead of
    # an empty, permanently-unretryable one (RunSetupError on a fresh
    # retry with the same run_id; no manifest to support resume_run_id
    # either). Round 3's fix only moved this before the guarded try, still
    # after begin_run() had already created the directory and lock.
    criteria_set_hash = compute_criteria_set_hash(request.criteria_set)
    execution_fingerprint = compute_execution_fingerprint(
        criteria_set_hash=criteria_set_hash,
        config=request.config,
        api_version=request.api_version,
        schema_version=SCHEMA_VERSION,
    )
    # codex_code_review_810ccf0.md finding 1: cheap, pre-hash "which
    # request is this" binding -- computed here (no file I/O, so safe to
    # compute before begin_run() like the two hashes above), used below to
    # protect the window where a resumed run's manifest is still
    # provisional (request_fingerprint not yet knowable) and
    # execution_fingerprint alone can't tell two different requests apart.
    provisional_request_binding = compute_provisional_request_binding(
        execution_fingerprint=execution_fingerprint, items=request.items, config=request.config
    )

    # finding 3 (round 3): begin_run() happens before file hashing (not
    # after), and everything from here on is wrapped in the
    # KeyboardInterrupt guard below -- an interrupt during hashing must
    # still produce a durable run directory, snapshot, and ledger event,
    # not vanish with nothing ever written.
    resume = request.resume_run_id is not None
    run_id = request.resume_run_id or request.run_id or str(uuid.uuid4())
    store.begin_run(run_id=run_id, resume=resume)
    reporter = _ProgressReporter(progress_cb, total_items=len(request.items), store=store, run_id=run_id)
    started_at = datetime.now(timezone.utc)
    terminal_results: dict[int, ScreeningItemResult] = {}

    # codex_code_review_c4f274a.md finding 1: two SEPARATE flags for two
    # separate protections -- conflating them (as the prior round did with
    # one `baseline_established`) either overwrites good resume data too
    # early or leaves a crash mid-hashing permanently unresumable.
    #   manifest_established -- a manifest durably exists (provisional, for
    #     a fresh run, or confirmed execution-compatible, for a resume) --
    #     the run is resumable AT ALL if interrupted from here on. Becomes
    #     True almost immediately after begin_run() succeeds, for both
    #     fresh and resume -- BEFORE Stage A1 file hashing, not after.
    #     Round 4's fix only stopped an early interrupt from creating an
    #     EMPTY unretryable directory; it never made a mid-hashing crash
    #     resumable, since the manifest still wasn't written until Stage A1
    #     fully completed.
    #   baseline_established -- terminal_results is trustworthy enough to
    #     build an interrupted snapshot from without losing information.
    #     True immediately for a fresh run (empty is itself correct); only
    #     True for a resume once already-completed items are actually
    #     hydrated from the prior attempt.
    manifest_established = False
    baseline_established = False
    manifest: dict[str, Any] | None = None
    # codex_code_review_146f98e.md: content already hashed for THIS run_id
    # (in this attempt, or a prior interrupted one) is checkpointed here as
    # it's confirmed, and durably persisted in the manifest incrementally.
    # provisional_request_binding alone is a PATH-based proxy for
    # not-yet-hashed items -- it cannot detect the file AT that path being
    # replaced after its real content hash was already observed. This dict
    # closes that gap: every item's real content hash, once known, must
    # never silently change across attempts for the same run_id.
    checkpoints: dict[str, str] = {}

    try:
        try:
            if resume:
                manifest = store.read_run_manifest(run_id=run_id)
                if manifest is None:
                    raise ResumeConflictError(f"run_id={run_id!r}: no manifest found, cannot verify compatibility")
                if manifest.get("execution_fingerprint") != execution_fingerprint:
                    raise ResumeConflictError(
                        f"run_id={run_id!r}: criteria/config changed since the original run -- "
                        "resume requires an identical execution_fingerprint"
                    )
                if manifest.get("request_fingerprint") is None:
                    # codex_code_review_810ccf0.md finding 1: the prior
                    # attempt never got past Stage A1 (manifest is still
                    # provisional), so request_fingerprint can't be checked
                    # yet -- without this check, execution_fingerprint alone
                    # let a resume attach an ENTIRELY DIFFERENT request
                    # (different items/citation/policies) to this run_id,
                    # before any file hashing or further validation ever
                    # ran. Checked BEFORE Stage A1 starts, deliberately --
                    # the whole point is to reject a mismatched resume
                    # before doing any file I/O on its behalf.
                    if manifest.get("provisional_request_binding") != provisional_request_binding:
                        raise ResumeConflictError(
                            f"run_id={run_id!r}: request changed since the original (still-provisional) run -- "
                            "resume requires an identical request binding before file hashing can proceed"
                        )
                # codex_code_review_146f98e.md: hydrate whatever content
                # hashes the prior (possibly interrupted) attempt already
                # confirmed, so Stage A1 below can detect a checkpointed
                # item's bytes having silently changed since then.
                checkpoints = dict(manifest.get("document_hash_checkpoints") or {})
                manifest_established = True
            else:
                # finding 1 (round 5): a PROVISIONAL manifest, written
                # immediately -- request_fingerprint isn't computable yet
                # (needs Stage A1's file hashes), but execution_fingerprint
                # + provisional_request_binding together are enough to make
                # this run_id durably claimed AND request-identified from
                # this point forward, even if the process dies before Stage
                # A1 ever finishes.
                store.write_run_manifest(
                    run_id=run_id,
                    execution_fingerprint=execution_fingerprint,
                    request_fingerprint=None,
                    provisional_request_binding=provisional_request_binding,
                    document_hash_checkpoints=checkpoints,
                )
                manifest_established = True
                baseline_established = True  # fresh run: empty terminal_results IS the correct baseline
            # Stage A1: hash every available file up front (sequential,
            # race-free). Wrapped so a permission error or similar reading
            # one file becomes an item outcome, not an escaping exception
            # (finding 3's second half) -- and an existing non-file path
            # (e.g. a directory) is its own distinct code, not lumped into
            # "missing".
            document_hashes: dict[str, str] = {}
            unavailable_reasons: dict[str, ProcessingError] = {}
            for item in request.items:
                try:
                    if not item.pdf_path.exists():
                        unavailable_reasons[item.document_id] = ProcessingError(
                            code="input_missing",
                            stage="preflight",
                            severity="fatal",
                            message=f"pdf_path does not exist: {item.pdf_path}",
                            retryable=True,
                        )
                        continue
                    if not item.pdf_path.is_file():
                        unavailable_reasons[item.document_id] = ProcessingError(
                            code="input_not_file",
                            stage="preflight",
                            severity="fatal",
                            message=f"pdf_path exists but is not a file: {item.pdf_path}",
                            retryable=False,
                        )
                        continue
                    actual_hash = _hash_file_streaming(item.pdf_path)
                except OSError as exc:
                    unavailable_reasons[item.document_id] = ProcessingError(
                        code="unexpected_error",
                        stage="preflight",
                        severity="fatal",
                        message=f"failed to read {item.pdf_path}: {exc}",
                        retryable=True,
                        exception_type=type(exc).__name__,
                    )
                    continue

                # codex_code_review_146f98e.md: content already checkpointed
                # (this attempt or a prior interrupted one) must never
                # silently change -- direct reproduction was replacing a
                # PDF's bytes in place, at the same path, after it had
                # already been hashed once for this run_id. This is a
                # run-level integrity violation, not a per-item outcome:
                # raises ResumeConflictError immediately, aborting the whole
                # attempt, the same way a full request_fingerprint mismatch
                # already does below.
                checkpointed_hash = checkpoints.get(item.document_id)
                if checkpointed_hash is not None and checkpointed_hash != actual_hash:
                    raise ResumeConflictError(
                        f"run_id={run_id!r}: content at {item.pdf_path} (document_id={item.document_id!r}) "
                        f"changed since it was already observed for this run (checkpointed sha256="
                        f"{checkpointed_hash!r}, now {actual_hash!r}) -- resume requires content already "
                        "hashed for this run to remain unchanged"
                    )
                if checkpointed_hash is None:
                    checkpoints[item.document_id] = actual_hash
                    # Persist immediately -- a further interruption right
                    # after this item's hash but before Stage A1 finishes
                    # must not lose it: it's the input to exactly this fix.
                    store.write_run_manifest(
                        run_id=run_id,
                        execution_fingerprint=execution_fingerprint,
                        request_fingerprint=None,
                        provisional_request_binding=provisional_request_binding,
                        document_hash_checkpoints=checkpoints,
                    )

                if item.expected_sha256 is not None and item.expected_sha256 != actual_hash:
                    unavailable_reasons[item.document_id] = ProcessingError(
                        code="hash_mismatch",
                        stage="preflight",
                        severity="fatal",
                        message=(
                            f"expected_sha256={item.expected_sha256!r} does not match computed hash "
                            f"{actual_hash!r} for {item.pdf_path}"
                        ),
                        retryable=False,
                    )
                    continue
                document_hashes[item.document_id] = actual_hash

            fingerprint_bindings = dict(document_hashes)
            for item in request.items:
                if item.document_id not in fingerprint_bindings and item.expected_sha256 is None:
                    fingerprint_bindings[item.document_id] = hashlib.sha256(
                        f"unavailable:{item.pdf_path.resolve()}".encode("utf-8")
                    ).hexdigest()

            request_fingerprint = compute_request_fingerprint(
                execution_fingerprint=execution_fingerprint,
                items=request.items,
                document_hashes=fingerprint_bindings,
                config=request.config,
            )

            if resume:
                # `manifest` was already read/execution-verified above,
                # before Stage A1 ran. request_fingerprint may still be
                # None in it if the PRIOR attempt itself crashed before
                # finishing Stage A1 -- nothing was hydrated in that case
                # either, so there's nothing to protect; skip that specific
                # comparison and just proceed to redo Stage A1 fresh.
                stored_request_fingerprint = manifest.get("request_fingerprint") if manifest else None
                if stored_request_fingerprint is not None and stored_request_fingerprint != request_fingerprint:
                    raise ResumeConflictError(
                        f"run_id={run_id!r}: criteria/config/request changed since the original run -- "
                        "resume requires an identical request_fingerprint"
                    )
                already_done = {
                    idx: result
                    for idx, result in store.read_completed_item_results(run_id=run_id).items()
                    if result.execution_status == "completed"
                }
                terminal_results.update(already_done)
                for _ in already_done:
                    reporter.item_done()

            # Finalize the manifest now that request_fingerprint is known --
            # overwrites the provisional one (fresh run) or simply
            # reconfirms it with the same value (resume). Carries the same
            # provisional_request_binding forward purely as an audit trail
            # -- request_fingerprint (the real, content-hash-backed check)
            # is authoritative from this point on.
            store.write_run_manifest(
                run_id=run_id,
                execution_fingerprint=execution_fingerprint,
                request_fingerprint=request_fingerprint,
                provisional_request_binding=provisional_request_binding,
                document_hash_checkpoints=checkpoints,
            )

            # Past this point terminal_results is a confirmed-valid
            # baseline (resume: compatibility confirmed + already_done
            # hydrated; fresh run: was already true) -- safe for an
            # interrupt from here on to materialize a snapshot from it.
            baseline_established = True

            store.append_ledger_event(run_id=run_id, event={"type": "run_started"})
            reporter.emit(event="run_started", occurred_at=started_at)

            # Stage A2: duplicate-content classification (finding 2 -- ALL
            # members of an "error"-mode group become duplicate_input, not
            # just the non-first copies).
            groups: dict[str, list[int]] = {}
            for index, item in enumerate(request.items):
                if index in terminal_results or item.document_id in unavailable_reasons:
                    continue
                groups.setdefault(document_hashes[item.document_id], []).append(index)

            duplicate_error_indices: set[int] = set()
            reuse_of: dict[int, int] = {}  # duplicate index -> canonical index ("reuse" mode only)
            canonical_indices: set[int] = set()
            if request.config.duplicate_content_policy == "error":
                for indices in groups.values():
                    if len(indices) > 1:
                        duplicate_error_indices.update(indices)
                    else:
                        canonical_indices.add(indices[0])
            else:  # "reuse"
                for indices in groups.values():
                    canonical = min(indices)
                    canonical_indices.add(canonical)
                    for idx in indices:
                        if idx != canonical:
                            reuse_of[idx] = canonical

            # Persist preflight-terminal results immediately, in order
            # (finding 1).
            for index, item in enumerate(request.items):
                if index in terminal_results:
                    continue
                if item.document_id in unavailable_reasons:
                    reason = unavailable_reasons[item.document_id]
                    status = "processing_failed" if reason.code == "hash_mismatch" else "input_unavailable"
                    result = _terminal_result(item, index, status=status, errors=[reason])
                elif index in duplicate_error_indices:
                    result = _terminal_result(
                        item,
                        index,
                        status="duplicate_input",
                        errors=[
                            ProcessingError(
                                code="duplicate_input",
                                stage="preflight",
                                severity="warning",
                                message=(
                                    f"identical content to {len(groups[document_hashes[item.document_id]]) - 1} "
                                    f"other item(s); duplicate_content_policy='error'"
                                ),
                                retryable=False,
                            )
                        ],
                    )
                else:
                    continue  # canonical or reuse-pending -- resolved below
                terminal_results[index] = result
                _persist_and_emit(store, reporter, run_id, result)

            # Ordered, fail-fast-aware eligibility pass: preflight fatals
            # count toward "stop submitting new work" too (finding 8).
            fail_fast = request.config.item_error_policy == "fail_fast"
            fatal_seen_so_far = False
            stage_b_queue: list[int] = []
            reuse_queue: list[int] = []
            for index in range(len(request.items)):
                if index in terminal_results:
                    if fail_fast and terminal_results[index].execution_status in _FATAL_EXECUTION_STATUSES:
                        fatal_seen_so_far = True
                    continue
                if fail_fast and fatal_seen_so_far:
                    result = _not_processed_result(request.items[index], index)
                    terminal_results[index] = result
                    _persist_and_emit(store, reporter, run_id, result)
                    continue
                if index in reuse_of:
                    reuse_queue.append(index)
                else:
                    stage_b_queue.append(index)

            # Stage B: bounded, order-preserving, fail-fast-aware processing.
            _run_stage_b(
                stage_b_queue,
                request=request,
                components=components,
                document_hashes=document_hashes,
                criteria_set_hash=criteria_set_hash,
                execution_fingerprint=execution_fingerprint,
                run_id=run_id,
                fail_fast=fail_fast,
                store=store,
                reporter=reporter,
                terminal_results=terminal_results,
            )

            # Stage C: resolve "reuse" duplicates from their canonical result.
            for index in reuse_queue:
                canonical_index = reuse_of[index]
                canonical_result = terminal_results[canonical_index]
                item = request.items[index]
                if canonical_result.execution_status == "completed":
                    result = _clone_for_reuse(canonical_result, item=item, index=index, run_id=run_id)
                    # finding 7: the ledger must identify which canonical
                    # item supplied the reused content -- source_run_id
                    # alone only says "this run", not "this item".
                    store.append_ledger_event(
                        run_id=run_id,
                        event={
                            "type": "content_reused",
                            "canonical_input_index": canonical_index,
                            "canonical_document_id": canonical_result.document_id,
                            "reused_input_index": index,
                            "reused_document_id": item.document_id,
                        },
                    )
                elif canonical_result.execution_status == "not_processed":
                    result = _not_processed_result(item, index)
                else:
                    result = _terminal_result(
                        item,
                        index,
                        status="processing_failed",
                        errors=[
                            ProcessingError(
                                code="unexpected_error",
                                stage="preflight",
                                severity="fatal",
                                message=(
                                    f"canonical item at input_index={reuse_of[index]} did not complete "
                                    f"(execution_status={canonical_result.execution_status!r}); nothing to reuse"
                                ),
                                retryable=True,
                            )
                        ],
                    )
                terminal_results[index] = result
                _persist_and_emit(store, reporter, run_id, result)

            ordered_items = [terminal_results[i] for i in range(len(request.items))]
            summary = _build_summary(ordered_items)
            run_status = _classify_run_status(summary)
            completed_at = datetime.now(timezone.utc)

            run_result = _build_run_result(
                store=store,
                run_id=run_id,
                run_status=run_status,
                criteria_set_id=request.criteria_set.criteria_set_id,
                criteria_set_version=request.criteria_set.version,
                criteria_set_hash=criteria_set_hash,
                execution_fingerprint=execution_fingerprint,
                started_at=started_at,
                completed_at=completed_at,
                items=ordered_items,
                summary=summary,
            )
            store.write_run_result(run_id=run_id, run_result=run_result)
            store.append_ledger_event(run_id=run_id, event={"type": "run_completed", "run_status": run_status})
            reporter.emit(event="run_completed", occurred_at=completed_at)
            return run_result

        except KeyboardInterrupt:
            interrupted_at = datetime.now(timezone.utc)
            if not manifest_established:
                # codex_code_review_810ccf0.md finding 2: nothing durable
                # has been claimed yet at this point -- and this branch must
                # not itself create anything durable either. Previously this
                # called store.append_ledger_event(), which auto-creates
                # runs/<run_id>/ as a side effect (its parent.mkdir) -- that
                # directory then existing with no manifest inside it is
                # EXACTLY the stranded, unclaimed state finding 2 requires
                # be impossible. A fresh retry with the same run_id must see
                # nothing at all here, so this stays in-process-log-only,
                # never touching the store.
                _logger.warning(
                    "run_id=%s interrupted before any manifest was ever written -- nothing durable was claimed, "
                    "a fresh retry with the same run_id may proceed",
                    run_id,
                )
                raise
            if resume and not baseline_established:
                # Manifest confirmed execution-compatible, but the prior
                # run's already-completed items haven't been hydrated into
                # terminal_results yet (Stage A1/request_fingerprint
                # verification still in progress). Writing a snapshot now
                # would overwrite a perfectly good existing run_result.json
                # with a false all-not_processed picture. Record the
                # attempt only; leave the last durable snapshot (and
                # item_results.jsonl) exactly as they were.
                try:
                    store.append_ledger_event(
                        run_id=run_id, event={"type": "run_interrupted", "phase": "resume_preflight"}
                    )
                except Exception:  # noqa: BLE001 -- never let ledger-write failure mask the interrupt
                    pass
                raise
            # finding 2: a NEUTRAL cancellation reason -- _not_processed_result
            # is specifically worded for fail_fast suppression and would
            # falsely blame "an earlier fatal item" for every item an
            # interrupt (not fail_fast) left untouched, even under the
            # default item_error_policy="continue".
            ordered_items = [
                terminal_results.get(i) or _interrupted_not_processed_result(request.items[i], i)
                for i in range(len(request.items))
            ]
            summary = _build_summary(ordered_items)
            interrupted_result = _build_run_result(
                store=store,
                run_id=run_id,
                run_status="interrupted",
                criteria_set_id=request.criteria_set.criteria_set_id,
                criteria_set_version=request.criteria_set.version,
                criteria_set_hash=criteria_set_hash,
                execution_fingerprint=execution_fingerprint,
                started_at=started_at,
                completed_at=interrupted_at,
                items=ordered_items,
                summary=summary,
            )
            store.write_run_result(run_id=run_id, run_result=interrupted_result)
            store.append_ledger_event(run_id=run_id, event={"type": "run_interrupted"})
            reporter.emit(event="run_interrupted", occurred_at=interrupted_at)
            raise
    finally:
        store.end_run(run_id=run_id)


def _run_stage_b(
    stage_b_queue: list[int],
    *,
    request: ScreeningRequest,
    components: Components,
    document_hashes: dict[str, str],
    criteria_set_hash: str,
    execution_fingerprint: str,
    run_id: str,
    fail_fast: bool,
    store,
    reporter: _ProgressReporter,
    terminal_results: dict[int, ScreeningItemResult],
) -> None:
    def process_one(index: int) -> ScreeningItemResult:
        item = request.items[index]
        item_started_at = datetime.now(timezone.utc)
        reporter.emit(
            event="item_started",
            occurred_at=item_started_at,
            input_index=index,
            document_id=item.document_id,
            report_id=item.report_id,
        )
        return _process_canonical_item(
            item=item,
            index=index,
            document_sha256=document_hashes[item.document_id],
            request=request,
            components=components,
            criteria_set_hash=criteria_set_hash,
            execution_fingerprint=execution_fingerprint,
            run_id=run_id,
            cache_mode=request.config.cache_mode,
            started_at=item_started_at,
        )

    max_workers = max(1, request.config.max_workers)

    if max_workers == 1:
        fatal_hit = False
        for index in stage_b_queue:
            if fatal_hit:
                result = _not_processed_result(request.items[index], index)
            else:
                result = process_one(index)
                if fail_fast and result.execution_status in _FATAL_EXECUTION_STATUSES:
                    fatal_hit = True
            terminal_results[index] = result
            _persist_and_emit(store, reporter, run_id, result)
        return

    # Bounded incremental submission (finding 8): never more than
    # max_workers futures in flight, and no NEW submission once a fatal
    # result has been observed -- already-running futures still finish.
    pending_iter = iter(stage_b_queue)
    in_flight: dict[concurrent.futures.Future, int] = {}
    fatal_hit = False

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        def top_up() -> None:
            nonlocal fatal_hit
            while not fatal_hit and len(in_flight) < max_workers:
                index = next(pending_iter, None)
                if index is None:
                    return
                in_flight[executor.submit(process_one, index)] = index

        top_up()
        while in_flight:
            done, _ = concurrent.futures.wait(in_flight.keys(), return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                index = in_flight.pop(future)
                result = future.result()
                terminal_results[index] = result
                _persist_and_emit(store, reporter, run_id, result)
                if fail_fast and result.execution_status in _FATAL_EXECUTION_STATUSES:
                    fatal_hit = True
            if not fatal_hit:
                top_up()

    for index in pending_iter:  # never submitted
        result = _not_processed_result(request.items[index], index)
        terminal_results[index] = result
        _persist_and_emit(store, reporter, run_id, result)


def _process_canonical_item(
    *,
    item: ScreeningItem,
    index: int,
    document_sha256: str,
    request: ScreeningRequest,
    components: Components,
    criteria_set_hash: str,
    execution_fingerprint: str,
    run_id: str,
    cache_mode: str,
    started_at: datetime,
) -> ScreeningItemResult:
    """Never raises -- any failure is captured as a processing_failed result."""
    try:
        document: ParsedDocument = components.parser.parse(
            document_id=item.document_id,
            document_sha256=document_sha256,
            pdf_path=item.pdf_path,
            parser_options=request.config.parser_options,
        )
        # codex_code_review_53f21de.md finding 2: defensively re-validate
        # the FULL nested shape, not just the top-level return value -- a
        # parser could return a well-typed ParsedDocument whose `chunks`
        # tuple contains non-Chunk objects, which previously escaped
        # uncaught at the `known_chunk_ids` comprehension below. Chunk/
        # ParsedDocument are now Pydantic models (components.py), so this
        # revalidation actually enforces nested field types.
        document = ParsedDocument.model_validate(document)
        # finding 4/1: identity AND shape validated in the SAME try as the
        # call itself -- a parser returning a malformed value entirely
        # (wrong type, missing attributes), not just wrong identity
        # fields, must still be caught here rather than escape as an
        # uncaught AttributeError/TypeError.
        if document.document_id != item.document_id or document.document_sha256 != document_sha256:
            raise _BoundaryViolation(
                f"parser returned document_id={document.document_id!r}/document_sha256="
                f"{document.document_sha256!r}, expected document_id={item.document_id!r}/"
                f"document_sha256={document_sha256!r}"
            )
        if document.coverage.extracted_characters > 0 and len(document.chunks) == 0:
            raise _BoundaryViolation(
                "parser reported extracted_characters > 0 but returned zero chunks -- "
                "a document with content but no chunks cannot be a usable parse result"
            )
    except Exception as exc:  # noqa: BLE001 -- any malformed component output becomes an item outcome
        return _failed_result(item, index, stage="parse", code="parse_failed", exc=exc)

    # finding 3: document coverage gates eligibility, not just retrieval quality.
    coverage = document.coverage
    if coverage.status == "failed":
        # codex_code_review_1578af2.md finding 2: the parser's ACTUAL
        # DocumentCoverage (which distinguishes "not found" from "not
        # searched successfully", the whole reason this record exists per
        # plan_v3) and its own warnings were previously discarded here --
        # _failed_result() -> _terminal_result() always wrote coverage=None
        # and fabricated a generic ValueError message, throwing away real
        # parser diagnostics a caller of run_ft_screening() could never see.
        # The parser is trusted to have already sanitized its own warning
        # strings (no raw paths/secrets) before they reach here -- this
        # just surfaces what it already decided was public-safe; full
        # exception detail belongs only in the parser's own logs.
        _logger.warning(
            "document_id=%s: parser reported DocumentCoverage.status == 'failed': %s",
            item.document_id,
            "; ".join(coverage.warnings) if coverage.warnings else "(no warnings reported)",
        )
        message = "document parsing failed"
        if coverage.warnings:
            message = f"document parsing failed: {coverage.warnings[0]}"
        return _terminal_result(
            item,
            index,
            status="processing_failed",
            coverage=coverage,
            errors=[
                ProcessingError(
                    code="insufficient_coverage",
                    stage="parse",
                    severity="fatal",
                    message=message,
                    retryable=False,
                )
            ],
        )
    warnings: list[ProcessingError] = []
    if coverage.status == "partial":
        warnings.append(
            ProcessingError(
                code="insufficient_coverage",
                stage="parse",
                severity="warning",
                message="document coverage is partial -- final decision is capped at needs_human_review",
                retryable=False,
            )
        )

    known_chunk_ids = {c.chunk_id for c in document.chunks}
    plain_assessments: list[CriterionAssessment] = []
    cache_source_run_ids: list[str] = []
    usage_records: list[AdjudicationUsage] = []
    item_cache_status = "disabled" if cache_mode == "off" else "hit"

    for criterion in request.criteria_set.criteria:
        cache_key = compute_criterion_cache_key(document_sha256=document_sha256, criterion=criterion, config=request.config)
        cached: CachedAssessment | None = None
        if cache_mode == "use":
            # finding 4 (round 5): a cache READ failure is a store/
            # persistence problem, not a scientific evidence-retrieval
            # problem -- round 4's fix correctly stopped swallowing it
            # silently, but mislabeled it as retrieval_failed (stage
            # "retrieve", as if the RETRIEVER had failed; no retriever ever
            # ran). Policy chosen here: cache reads are OPTIONAL -- treat a
            # failure as a miss and recompute (the assessment can always be
            # recomputed from scratch; losing the cache is a performance
            # cost, not a correctness one), but record it honestly as a
            # persistence_failed WARNING (not fatal) plus a ledger event,
            # so a degraded cache backend stays visible/auditable rather
            # than silently invisible.
            try:
                candidate = components.store.get_cached_assessment(cache_key=cache_key)
            except Exception as exc:
                candidate = None
                # codex_code_review_810ccf0.md finding 3: this is a STORE
                # failure, not a retriever failure -- no retriever ever ran
                # -- so stage must be "persist" (already a valid
                # ProcessingError.stage value, just unused until now), not
                # "retrieve". And the PUBLIC, persisted message must never
                # embed str(exc): a real store backend's exception can
                # contain connection strings/credentials/paths (Codex's own
                # repro: a message containing a live postgresql:// URI with
                # a password). Same rule already applied to progress-
                # callback exceptions (codex_code_review_53f21de.md) --
                # neutral message + exception_type in the public artifact,
                # full exception only to the runtime logger.
                _logger.warning(
                    "cache read failed for run_id=%s document_id=%s criterion_id=%s; treating as a miss "
                    "and recomputing",
                    run_id,
                    item.document_id,
                    criterion.criterion_id,
                    exc_info=exc,
                )
                warnings.append(
                    ProcessingError(
                        code="persistence_failed",
                        stage="persist",
                        severity="warning",
                        message=(
                            f"cache read failed for criterion {criterion.criterion_id!r}; treating as a miss "
                            "and recomputing (see server logs for the underlying error)"
                        ),
                        retryable=True,
                        exception_type=type(exc).__name__,
                    )
                )
                try:
                    components.store.append_ledger_event(
                        run_id=run_id,
                        event={
                            "type": "cache_read_failed",
                            "document_id": item.document_id,
                            "criterion_id": criterion.criterion_id,
                            "cache_key": cache_key,
                            "exception_type": type(exc).__name__,
                        },
                    )
                except Exception:  # noqa: BLE001 -- never let ledger-write failure mask the recovery path
                    pass
            if candidate is not None:
                try:
                    # CachedAssessment.model_validate() protects the
                    # ORCHESTRATOR regardless of which ScreeningStore
                    # implementation returned `candidate` -- FileScreeningStore's
                    # own envelope already validates on disk, but a
                    # different, protocol-conforming store implementation
                    # has no such guarantee, and could return a
                    # syntactically-valid-but-wrong-shaped CachedAssessment
                    # (e.g. source_run_id as a list). Catches broadly
                    # (ValidationError/AttributeError/TypeError/etc, not
                    # just _BoundaryViolation) since a malformed candidate
                    # object can fail in any of those ways before ever
                    # reaching _validate_assessment_boundary.
                    candidate = CachedAssessment.model_validate(candidate)
                    _validate_assessment_boundary(
                        candidate.assessment,
                        criterion=criterion,
                        document=document,
                        config=request.config,
                        known_chunk_ids=known_chunk_ids,
                        check_document_id=False,  # cached content may carry a different original document_id
                    )
                except Exception:  # noqa: BLE001 -- any malformed cache payload is a miss, not a crash
                    cached = None  # corrupt/poisoned entry -- treat as a miss, recompute below
                else:
                    cached = candidate

        if cached is not None:
            plain_assessments.append(_rebind_assessment_document_id(cached.assessment, document_id=item.document_id))
            cache_source_run_ids.append(cached.source_run_id)
            # finding 6: per-criterion cache lineage recorded in the
            # ledger -- the item-level provenance field can only carry one
            # source_result_run_id, which is lossy when different criteria
            # were cached by different runs; the ledger keeps the full record.
            components.store.append_ledger_event(
                run_id=run_id,
                event={
                    "type": "cache_hit",
                    "document_id": item.document_id,
                    "criterion_id": criterion.criterion_id,
                    "cache_key": cache_key,
                    "source_run_id": cached.source_run_id,
                },
            )
            # A cache hit made NO provider call this run -- this is a
            # CONFIRMED zero (usage_available=True), never the source
            # run's own historical cost attributed to this run.
            cache_usage = AdjudicationUsage(
                model=cached.assessment.model, input_tokens=0, output_tokens=0, total_tokens=0,
                provider_call_count=0, retry_count=0, usage_available=True,
            )
            usage_records.append(cache_usage)
            components.store.append_ledger_event(
                run_id=run_id,
                event={
                    "type": "criterion_usage",
                    "document_id": item.document_id,
                    "criterion_id": criterion.criterion_id,
                    "cache_status": "hit",
                    **cache_usage.model_dump(),
                },
            )
            continue

        item_cache_status = "refresh" if cache_mode == "refresh" else ("disabled" if cache_mode == "off" else "miss")
        try:
            evidence = components.retriever.retrieve(
                document=document, criterion=criterion, retriever_options=request.config.retriever_options
            )
            # finding 2: materialize AND revalidate every span BEFORE
            # anything else touches `evidence`. A one-shot iterable (e.g. a
            # generator) would otherwise be exhausted by the boundary
            # validator's own iteration, leaving the adjudicator a dead
            # iterable and silently zero evidence -- a "completed" result
            # with no evidence, not an exception, so nothing would have
            # flagged it. The list comprehension both materializes and
            # validates each span's shape in one pass.
            evidence = [EvidenceSpan.model_validate(span) for span in evidence]
            # finding 4/1: call + shape/identity validation share one try,
            # same rationale as the parser above.
            _validate_evidence_boundary(evidence, document=document, known_chunk_ids=known_chunk_ids, check_document_id=True)
        except Exception as exc:  # noqa: BLE001
            return _failed_result(item, index, stage="retrieve", code="retrieval_failed", exc=exc)
        try:
            outcome = components.adjudicator.adjudicate(
                document=document, criterion=criterion, evidence=evidence, model_options=request.config.model_options
            )
            # finding 2: revalidate the adjudicator's return value too, same
            # rationale as the parser/evidence above. AdjudicationOutcome's
            # own revalidate_instances="always" (components.py) forces this
            # to also deeply revalidate the nested `assessment`/`usage`
            # fields, not just the outer shape.
            outcome = AdjudicationOutcome.model_validate(outcome)
            assessment = outcome.assessment
            _validate_assessment_boundary(
                assessment,
                criterion=criterion,
                document=document,
                config=request.config,
                known_chunk_ids=known_chunk_ids,
                check_document_id=True,
            )
        except Exception as exc:  # noqa: BLE001
            return _failed_result(item, index, stage="adjudicate", code="adjudication_failed", exc=exc)
        # finding 6: only a validated assessment may enter the cache.
        # finding 4 (round 5): a cache WRITE failure must not discard an
        # already-computed, already-validated assessment -- the item's
        # result is correct regardless of whether it gets cached for next
        # time, so a failed write degrades future performance, not this
        # item's correctness. Recorded the same way as a read failure: a
        # persistence_failed warning plus a ledger event, never fatal.
        if cache_mode != "off":
            try:
                components.store.put_cached_assessment(cache_key=cache_key, assessment=assessment, source_run_id=run_id)
            except Exception as exc:  # noqa: BLE001
                # finding 3: same stage/message-sanitization fix as the
                # cache-read failure above.
                _logger.warning(
                    "cache write failed for run_id=%s document_id=%s criterion_id=%s; continuing without "
                    "caching this result",
                    run_id,
                    item.document_id,
                    criterion.criterion_id,
                    exc_info=exc,
                )
                warnings.append(
                    ProcessingError(
                        code="persistence_failed",
                        stage="persist",
                        severity="warning",
                        message=(
                            f"cache write failed for criterion {criterion.criterion_id!r}; continuing without "
                            "caching this result (see server logs for the underlying error)"
                        ),
                        retryable=True,
                        exception_type=type(exc).__name__,
                    )
                )
                try:
                    components.store.append_ledger_event(
                        run_id=run_id,
                        event={
                            "type": "cache_write_failed",
                            "document_id": item.document_id,
                            "criterion_id": criterion.criterion_id,
                            "cache_key": cache_key,
                            "exception_type": type(exc).__name__,
                        },
                    )
                except Exception:  # noqa: BLE001
                    pass
        plain_assessments.append(assessment)
        cache_source_run_ids.append(run_id)
        usage_records.append(outcome.usage)
        components.store.append_ledger_event(
            run_id=run_id,
            event={
                "type": "criterion_usage",
                "document_id": item.document_id,
                "criterion_id": criterion.criterion_id,
                "cache_status": "miss",
                **outcome.usage.model_dump(),
            },
        )

    try:
        eligibility = build_eligibility_result(
            result_id=f"{run_id}:{item.document_id}:1",
            document_id=item.document_id,
            document_sha256=document_sha256,
            report_id=item.report_id,
            study_id=item.study_id,
            criteria_set=request.criteria_set,
            criteria_set_hash=criteria_set_hash,
            execution_fingerprint=execution_fingerprint,
            assessments=plain_assessments,
            created_at=datetime.now(timezone.utc),
        )
    except AggregationError as exc:
        return _failed_result(item, index, stage="aggregate", code="unexpected_error", exc=exc)

    # finding 2: partial coverage caps EVERY decision at needs_human_review
    # -- including what would otherwise be "exclude". The frozen contract
    # (api_contract_v1.md, carried by v2) says partial coverage produces a
    # completed needs_human_review result when assessments remain
    # representable at all, full stop; it does not carve out an exception
    # for a confident-looking exclude. `exclusion_reason_codes` is left
    # untouched as reviewer context (which criteria looked violated);
    # `primary_exclusion_reason_code` is cleared because the schema's own
    # local validator requires it be null unless final_decision=="exclude"
    # -- superseded-but-not-actual decisions don't get to assert a single
    # authoritative reason.
    if coverage.status == "partial" and eligibility.final_decision != "needs_human_review":
        eligibility = eligibility.model_copy(
            update={
                "proposed_decision": "needs_human_review",
                "final_decision": "needs_human_review",
                "primary_exclusion_reason_code": None,
            }
        )

    completed_at = datetime.now(timezone.utc)
    # finding 6: cache_status is item-level (the enum has no "partial"
    # value): "hit" only if EVERY criterion this item needed was actually a
    # cache hit. source_result_run_id is only set when ALL of those hits
    # trace to the SAME source run -- a mixed-source item (criteria cached
    # by different runs) can't be represented by one singular field
    # truthfully, so it's left None; the per-criterion "cache_hit" ledger
    # events above carry the complete, non-lossy lineage.
    source_result_run_id = None
    if item_cache_status == "hit" and cache_source_run_ids:
        unique_sources = set(cache_source_run_ids)
        if len(unique_sources) == 1:
            source_result_run_id = cache_source_run_ids[0]

    # usage_available is True only if EVERY criterion's own usage was known
    # (a real call's reported tokens, or a cache hit's confirmed zero) --
    # False if ANY criterion's telemetry was genuinely missing, so partial
    # telemetry never gets reported as if it were the complete picture.
    provenance = ExecutionProvenance(
        document_sha256=document_sha256,
        criteria_set_hash=criteria_set_hash,
        execution_fingerprint=execution_fingerprint,
        cache_status=item_cache_status,
        source_result_run_id=source_result_run_id,
        started_at=started_at,
        completed_at=completed_at,
        usage_available=all(u.usage_available for u in usage_records) if usage_records else False,
        total_input_tokens=sum(u.input_tokens for u in usage_records),
        total_output_tokens=sum(u.output_tokens for u in usage_records),
        total_tokens=sum(u.total_tokens for u in usage_records),
        total_provider_calls=sum(u.provider_call_count for u in usage_records),
        total_retries=sum(u.retry_count for u in usage_records),
    )
    return ScreeningItemResult(
        input_index=index,
        document_id=item.document_id,
        report_id=item.report_id,
        study_id=item.study_id,
        citation=item.citation,
        execution_status="completed",
        eligibility=eligibility,
        coverage=coverage,
        provenance=provenance,
        errors=warnings,
    )


def _validate_evidence_boundary(
    evidence: list, *, document: ParsedDocument, known_chunk_ids: set[str], check_document_id: bool
) -> None:
    for span in evidence:
        if span.document_sha256 != document.document_sha256:
            raise _BoundaryViolation(
                f"evidence bound to document_sha256={span.document_sha256!r}, expected "
                f"{document.document_sha256!r}"
            )
        if check_document_id and span.document_id != document.document_id:
            raise _BoundaryViolation(
                f"evidence bound to document_id={span.document_id!r}, expected {document.document_id!r}"
            )
        # finding 4: membership is unconditional -- a parser returning zero
        # chunks must not silently disable this check (an empty
        # known_chunk_ids set previously made `if known_chunk_ids and ...`
        # a no-op, letting an invented chunk_id through when the document
        # legitimately has no chunks at all).
        if span.chunk_id not in known_chunk_ids:
            raise _BoundaryViolation(f"evidence references unknown chunk_id={span.chunk_id!r}")


def _validate_assessment_boundary(
    assessment: CriterionAssessment,
    *,
    criterion: CriterionSpec,
    document: ParsedDocument,
    config: ScreeningConfig,
    known_chunk_ids: set[str],
    check_document_id: bool,
) -> None:
    if assessment.criterion_id != criterion.criterion_id or assessment.criterion_version != criterion.version:
        raise _BoundaryViolation(
            f"adjudicator returned criterion_id={assessment.criterion_id!r}/version="
            f"{assessment.criterion_version!r}, expected criterion_id={criterion.criterion_id!r}/"
            f"version={criterion.version!r}"
        )
    if assessment.model != config.adjudicator_model:
        raise _BoundaryViolation(
            f"adjudicator returned model={assessment.model!r}, expected {config.adjudicator_model!r}"
        )
    if assessment.prompt_template_version != config.prompt_template_version:
        raise _BoundaryViolation(
            f"adjudicator returned prompt_template_version={assessment.prompt_template_version!r}, "
            f"expected {config.prompt_template_version!r}"
        )
    if assessment.output_schema_version != config.output_schema_version:
        raise _BoundaryViolation(
            f"adjudicator returned output_schema_version={assessment.output_schema_version!r}, "
            f"expected {config.output_schema_version!r}"
        )
    _validate_evidence_boundary(
        assessment.evidence, document=document, known_chunk_ids=known_chunk_ids, check_document_id=check_document_id
    )


def _rebind_assessment_document_id(assessment: CriterionAssessment, *, document_id: str) -> CriterionAssessment:
    """Cached CriterionAssessments are content-addressed and identity-
    neutral; materializing one for THIS item must rebind every nested
    EvidenceSpan.document_id, never return a stale caller identifier
    (api_contract_v2.md correction 7)."""
    rebound_evidence = [span.model_copy(update={"document_id": document_id}) for span in assessment.evidence]
    return assessment.model_copy(update={"evidence": rebound_evidence})


def _clone_for_reuse(
    canonical: ScreeningItemResult, *, item: ScreeningItem, index: int, run_id: str
) -> ScreeningItemResult:
    rebound_assessments = [
        _rebind_assessment_document_id(a, document_id=item.document_id) for a in canonical.eligibility.assessments
    ]
    eligibility = canonical.eligibility.model_copy(
        update={
            "result_id": f"{run_id}:{item.document_id}:1",
            "document_id": item.document_id,
            "report_id": item.report_id,
            "study_id": item.study_id,
            "assessments": rebound_assessments,
        }
    )
    provenance = canonical.provenance.model_copy(
        update={"cache_status": "content_reuse", "source_result_run_id": run_id}
    )
    return ScreeningItemResult(
        input_index=index,
        document_id=item.document_id,
        report_id=item.report_id,
        study_id=item.study_id,
        citation=item.citation,
        execution_status="completed",
        eligibility=eligibility,
        coverage=canonical.coverage,
        provenance=provenance,
        # finding 7: the canonical result's warnings (e.g. partial-coverage)
        # must survive into the reused item -- coverage/eligibility are
        # cloned, so the caveats that produced them must be too.
        errors=list(canonical.errors),
    )


def _terminal_result(
    item: ScreeningItem, index: int, *, status: str, errors: list[ProcessingError], coverage: DocumentCoverage | None = None
) -> ScreeningItemResult:
    return ScreeningItemResult(
        input_index=index,
        document_id=item.document_id,
        report_id=item.report_id,
        study_id=item.study_id,
        citation=item.citation,
        execution_status=status,
        eligibility=None,
        coverage=coverage,
        provenance=None,
        errors=errors,
    )


def _failed_result(item: ScreeningItem, index: int, *, stage: str, code: str, exc: Exception) -> ScreeningItemResult:
    return _terminal_result(
        item,
        index,
        status="processing_failed",
        errors=[
            ProcessingError(
                code=code,
                stage=stage,
                severity="fatal",
                message=str(exc),
                retryable=False,
                exception_type=type(exc).__name__,
            )
        ],
    )


def _not_processed_result(item: ScreeningItem, index: int) -> ScreeningItemResult:
    """fail_fast suppression ONLY -- see _interrupted_not_processed_result
    for the cancellation/interrupt case (codex_code_review_c4f274a.md
    finding 2: these are different causes and must not share one message)."""
    return _terminal_result(
        item,
        index,
        status="not_processed",
        errors=[
            ProcessingError(
                code="unexpected_error",
                stage="preflight",
                severity="warning",
                message="not processed: item_error_policy='fail_fast' stopped new work after an earlier fatal item",
                retryable=True,
            )
        ],
    )


def _interrupted_not_processed_result(item: ScreeningItem, index: int) -> ScreeningItemResult:
    """Cancellation (KeyboardInterrupt) ONLY -- see _not_processed_result
    for the fail_fast case. Used exclusively when materializing an
    interrupted-run snapshot, regardless of item_error_policy."""
    return _terminal_result(
        item,
        index,
        status="not_processed",
        errors=[
            ProcessingError(
                code="unexpected_error",
                stage="preflight",
                severity="warning",
                message="not processed: run was interrupted (cancellation) before this item was reached",
                retryable=True,
            )
        ],
    )


def _persist_and_emit(store, reporter: _ProgressReporter, run_id: str, result: ScreeningItemResult) -> None:
    store.write_item_result(run_id=run_id, item_result=result)
    store.append_ledger_event(
        run_id=run_id,
        event={"type": "item_result", "input_index": result.input_index, "execution_status": result.execution_status},
    )
    reporter.item_done()
    event_name = "item_completed" if result.execution_status == "completed" else "item_failed"
    reporter.emit(
        event=event_name,
        occurred_at=datetime.now(timezone.utc),
        input_index=result.input_index,
        document_id=result.document_id,
        report_id=result.report_id,
    )


def _build_summary(items: list[ScreeningItemResult]) -> RunSummary:
    return RunSummary(
        total=len(items),
        completed=sum(1 for i in items if i.execution_status == "completed"),
        included=sum(1 for i in items if i.execution_status == "completed" and i.eligibility.final_decision == "include"),
        excluded=sum(1 for i in items if i.execution_status == "completed" and i.eligibility.final_decision == "exclude"),
        needs_human_review=sum(
            1 for i in items if i.execution_status == "completed" and i.eligibility.final_decision == "needs_human_review"
        ),
        input_unavailable=sum(1 for i in items if i.execution_status == "input_unavailable"),
        duplicate_input=sum(1 for i in items if i.execution_status == "duplicate_input"),
        processing_failed=sum(1 for i in items if i.execution_status == "processing_failed"),
        not_processed=sum(1 for i in items if i.execution_status == "not_processed"),
        cache_hits=sum(1 for i in items if i.provenance is not None and i.provenance.cache_status == "hit"),
    )


def _classify_run_status(summary: RunSummary) -> str:
    if summary.completed == summary.total:
        return "completed"
    if summary.completed == 0:
        return "failed"
    return "completed_with_issues"


def _build_run_result(
    *,
    store,
    run_id: str,
    run_status: str,
    criteria_set_id: str,
    criteria_set_version: str,
    criteria_set_hash: str,
    execution_fingerprint: str,
    started_at: datetime,
    completed_at: datetime,
    items: list[ScreeningItemResult],
    summary: RunSummary,
) -> ScreeningRunResult:
    run_result_json, item_results_jsonl, ledger_jsonl = store.artifact_paths(run_id=run_id)
    return ScreeningRunResult(
        api_version=API_VERSION,
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        run_status=run_status,
        criteria_set_id=criteria_set_id,
        criteria_set_version=criteria_set_version,
        criteria_set_hash=criteria_set_hash,
        execution_fingerprint=execution_fingerprint,
        started_at=started_at,
        completed_at=completed_at,
        items=items,
        summary=summary,
        artifacts=RunArtifacts(
            run_result_json=run_result_json, item_results_jsonl=item_results_jsonl, ledger_jsonl=ledger_jsonl
        ),
    )
