"""Real `DocumentParser` implementation (plan_v5 step 6), backed by
Docling. Replaces `fakes.FakeDocumentParser` for production use -- the
orchestrator, `components.py`'s `ParsedDocument`/`Chunk` types, and every
existing test against the fakes are unchanged; this is a drop-in behind
the same Protocol.

Design notes (from hands-on exploration against real PDFs in
`fulltext/inputs/` before writing this file -- see FT_SCREENING session
notes; Docling's API shape isn't fully obvious from its docs alone):

- **Warmup cost dominates.** Constructing a `DocumentConverter` lazily
  loads OCR/layout models; measured ~120-165s cold on this machine,
  almost entirely one-time (subsequent `convert()` calls on the SAME
  instance are much cheaper -- seconds, not minutes, per typical paper,
  though table-structure recognition adds real cost per table-heavy
  document). `DoclingDocumentParser` therefore owns its converter(s) for
  its whole lifetime; callers MUST construct one instance and reuse it
  across every item in a run, never one per document.
- **Corrupt/unreadable/empty PDFs raise, they don't return a
  PARTIAL_SUCCESS/FAILURE result.** Confirmed directly: a truncated PDF,
  a 0-byte file, and a non-PDF file with a `.pdf` extension all raised
  `docling.exceptions.ConversionError` (or a lower-level
  `pypdfium2.PdfiumError`) out of `convert()` itself, never returned a
  `ConversionResult` with a failure status. `parse()` therefore wraps the
  `convert()` call itself in a broad `except Exception`, not just the
  result-status check.
- **Page numbering is 1-indexed in item provenance, 0-indexed in the raw
  page list.** `item.prov[0].page_no` (used for `Chunk.page`, matching
  this module's convention of one page per chunk) is 1-indexed, matching
  natural page numbers a human/citation would use. `total_pages` is
  computed as `len(result.pages)` (a count, so the indexing convention of
  the underlying list doesn't matter for that value alone) -- do not
  read `result.pages[i].page_no` and assume it matches `item.prov[].page_no`
  without the +1 offset; they use different bases.
- **Tables need special handling.** `TableItem` doesn't carry a plain
  `.text` attribute the way `TextItem`/`SectionHeaderItem` do --
  `TableItem.export_to_markdown(doc=doc)` produces a readable markdown
  table, used as that chunk's text (verified directly: produces
  `"TABLE 1 ...\n\n| col | col |\n|---|---|\n..."`-shaped output for a
  real multi-table academic PDF).
"""

from __future__ import annotations

import hashlib
import logging
import threading
from pathlib import Path
from typing import Any

from ft_screening.components import Chunk, ParsedDocument
from ft_screening.schema import DocumentCoverage

_logger = logging.getLogger(__name__)

_CHUNK_SEPARATOR = "\n\n"


def _label_sets() -> tuple[frozenset, frozenset]:
    """Docling's own import chain has real, measurable cost (plugin
    discovery etc. -- confirmed empirically: ~20s just to import
    `docling_core.types.doc`, before any model loading or conversion even
    starts) -- deferred to first actual use (inside DoclingDocumentParser
    methods) rather than paid by every module that merely imports
    `ft_screening.document` for the class/type, e.g. a test file that
    only needs it for a factory reference.
    """
    from docling_core.types.doc import DocItemLabel

    # Item labels treated as section/heading boundaries -- they set the
    # `section` value attached to SUBSEQUENT body chunks AND are
    # themselves emitted as a (short) chunk attributed to the section they
    # introduce (see the loop in _build_parsed_document: a document whose
    # only content Docling classifies as a heading -- observed for real
    # with an OCR'd single-line scanned page -- must not silently lose
    # 100% of its content just because nothing else follows it).
    section_boundary_labels = frozenset({DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE})

    # Item labels whose plain `.text` is included as body content.
    # Deliberately excludes PICTURE/CHART (no extractable text body
    # relevant to eligibility evidence), PAGE_HEADER/PAGE_FOOTER (running
    # headers/footers are layout noise, not content), DOCUMENT_INDEX
    # (table-of-contents entries, not substantive content). TABLE is
    # handled separately (see `_item_text_and_page`) since it has no
    # plain `.text`.
    body_text_labels = frozenset(
        {
            DocItemLabel.TEXT,
            DocItemLabel.PARAGRAPH,
            DocItemLabel.LIST_ITEM,
            DocItemLabel.CAPTION,
            DocItemLabel.FOOTNOTE,
            DocItemLabel.FORMULA,
            DocItemLabel.CODE,
            DocItemLabel.REFERENCE,
        }
    )
    return section_boundary_labels, body_text_labels


def _item_page_no(item: Any) -> int | None:
    prov = getattr(item, "prov", None)
    if not prov:
        return None
    page_no = getattr(prov[0], "page_no", None)
    return int(page_no) if page_no is not None else None


def _item_text_and_page(
    item: Any, *, doc: Any, body_text_labels: frozenset
) -> tuple[str | None, int | None, str | None]:
    """Returns (text, page, extraction_warning) for one Docling item.
    `text` is None if this item contributes no body text (skip it).
    TableItem is special-cased since it carries no plain `.text`
    attribute. `extraction_warning`, when set, must be surfaced into
    `DocumentCoverage.warnings` by the caller -- codex_code_review_1578af2.md
    finding 3: a failed table export was previously only logged, never
    surfaced publicly, and could leave `status="complete"` with an empty
    `warnings` list even though real content was silently lost."""
    label = getattr(item, "label", None)
    page = _item_page_no(item)
    if type(item).__name__ == "TableItem":
        try:
            text = item.export_to_markdown(doc=doc)
        except Exception as exc:  # noqa: BLE001 -- a broken table export must not abort the whole document
            _logger.warning("table markdown export failed, skipping this table: %s", exc, exc_info=exc)
            page_note = f" on page {page}" if page is not None else ""
            return None, page, f"a table{page_note} could not be exported ({type(exc).__name__}) and was skipped"
        return (text or None), page, None
    if label in body_text_labels:
        text = getattr(item, "text", None)
        return (text or None), page, None
    return None, page, None


def _deterministic_chunk_id(document_sha256: str, ordinal: int, text: str) -> str:
    """Content-addressed, deterministic across runs for identical input
    (same document bytes, same ordinal position, same extracted text) --
    the plan_v3 acceptance contract's explicit requirement. Not derived
    from the caller-supplied document_id, matching the convention already
    established by fakes.FakeDocumentParser: a real content-addressed
    chunker should produce the same chunk_id for identical bytes
    regardless of what label the caller attaches to the document."""
    payload = f"{document_sha256}:{ordinal}:{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def _failed_parsed_document(document_id: str, document_sha256: str, *, warning: str) -> ParsedDocument:
    return ParsedDocument(
        document_id=document_id,
        document_sha256=document_sha256,
        coverage=DocumentCoverage(
            status="failed",
            total_pages=None,
            parsed_pages=[],
            failed_pages=[],
            ocr_used=False,
            extracted_characters=0,
            warnings=[warning],
        ),
        chunks=(),
    )


def _any_cell_from_ocr(result: Any) -> bool:
    """`DocumentCoverage.ocr_used` should mean "OCR was actually invoked
    for this document", not merely "OCR was enabled as an option" -- a
    text-native PDF converted with do_ocr=True never actually triggers
    OCR on any page, and reporting ocr_used=True for it would be
    misleading. Each extracted text cell carries its own `from_ocr` flag
    (confirmed via hands-on inspection of a real conversion result); this
    checks every cell of every layout cluster on every page for one that
    was actually OCR-derived."""
    for page in result.pages:
        layout = getattr(getattr(page, "predictions", None), "layout", None)
        for cluster in getattr(layout, "clusters", None) or []:
            for cell in getattr(cluster, "cells", None) or []:
                if getattr(cell, "from_ocr", False):
                    return True
    return False


def _build_parsed_document(
    document_id: str, document_sha256: str, result: Any, *, do_ocr: bool, is_success: bool
) -> ParsedDocument:
    section_boundary_labels, body_text_labels = _label_sets()
    doc = result.document
    total_pages = len(result.pages)
    ocr_used = do_ocr and _any_cell_from_ocr(result)

    chunks: list[Chunk] = []
    full_text_len = 0
    current_section: str | None = None
    pages_with_content: set[int] = set()
    extraction_warnings: list[str] = []
    ordinal = 0

    def emit_chunk(text: str, page: int | None) -> None:
        nonlocal full_text_len, ordinal
        if chunks:
            full_text_len += len(_CHUNK_SEPARATOR)
        start_char = full_text_len
        full_text_len += len(text)
        end_char = full_text_len
        if page is not None:
            pages_with_content.add(page)
        chunks.append(
            Chunk(
                chunk_id=_deterministic_chunk_id(document_sha256, ordinal, text),
                section=current_section,
                page=page,
                start_char=start_char,
                end_char=end_char,
                text=text,
            )
        )
        ordinal += 1

    for item, _level in doc.iterate_items():
        label = getattr(item, "label", None)
        if label in section_boundary_labels:
            heading_text = getattr(item, "text", None)
            if heading_text and heading_text.strip():
                current_section = heading_text.strip()
                # A heading is still real, citable text -- emit it as its
                # own chunk (attributed to the section it introduces), not
                # just a label update. Found via hands-on acceptance
                # testing: a synthetic scanned page whose ONLY content was
                # a single OCR'd line came back from Docling labeled
                # section_header, not text -- without this, that document
                # reported ocr_used=True and a SUCCESS conversion status,
                # yet zero chunks and extracted_characters=0.
                emit_chunk(current_section, _item_page_no(item))
            continue

        text, page, extraction_warning = _item_text_and_page(item, doc=doc, body_text_labels=body_text_labels)
        if extraction_warning:
            extraction_warnings.append(extraction_warning)
        if not text or not text.strip():
            continue
        emit_chunk(text, page)

    extracted_characters = full_text_len

    if is_success:
        has_content = extracted_characters > 0
        # finding 3: a failed item-level extraction (e.g. a table) downgrades
        # an otherwise-"complete" document to "partial" -- the pages
        # themselves were successfully processed (so parsed_pages is still
        # every page), but some content on them was NOT captured, which is
        # exactly what "partial" is for. A document with zero content at
        # all is "partial" regardless (unchanged from before).
        status = "complete" if has_content and not extraction_warnings else "partial"
        parsed_pages = list(range(1, total_pages + 1)) if has_content else sorted(pages_with_content)
        failed_pages: list[int] = []
        warnings: list[str] = list(extraction_warnings)
    else:
        # PARTIAL_SUCCESS (or any other non-SUCCESS status that didn't
        # raise) -- Docling's per-page error granularity wasn't
        # observable in hands-on testing (every corrupt/empty/invalid
        # input tried raised out of convert() instead of returning this
        # status), so this branch is a documented best-effort fallback:
        # pages that contributed at least one extracted item are
        # "parsed", every other page is conservatively "failed" rather
        # than silently assumed fine.
        status = "partial"
        parsed_pages = sorted(pages_with_content)
        failed_pages = sorted(set(range(1, total_pages + 1)) - pages_with_content)
        warnings = [f"Docling conversion status was {result.status!r}, not SUCCESS"]
        result_errors = getattr(result, "errors", None) or []
        for err in result_errors:
            # codex_code_review_af44129.md finding 2: str(err) previously
            # went straight into a PUBLIC coverage record -- confirmed
            # exploitable with a constructed error object whose __str__
            # embedded a secret/path. The type name is safe and still
            # useful; anything err itself carries is logger-only.
            _logger.warning("Docling reported a conversion error for document_id=%s: %s", document_id, err)
            warnings.append(f"a conversion error occurred ({type(err).__name__})")
        warnings.extend(extraction_warnings)

    coverage = DocumentCoverage(
        status=status,
        total_pages=total_pages,
        parsed_pages=parsed_pages,
        failed_pages=failed_pages,
        ocr_used=ocr_used,
        extracted_characters=extracted_characters,
        warnings=warnings,
    )
    return ParsedDocument(
        document_id=document_id, document_sha256=document_sha256, coverage=coverage, chunks=tuple(chunks)
    )


def _parser_options_model():
    """codex_code_review_1578af2.md finding 5: parser_options was applying
    `bool(...)` to whatever JSON-typed value the caller supplied, so
    `{"do_ocr": "false"}` (a non-empty string) silently ENABLED OCR
    instead of disabling it, and a misspelled/unknown option key was
    silently ignored rather than rejected -- both let a bad configuration
    silently alter scientific processing instead of failing loudly.
    Deferred import (pydantic is already a hard dependency, but this
    avoids constructing the model class before it's actually needed)."""
    from pydantic import BaseModel, ConfigDict

    class DoclingParserOptions(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)

        do_ocr: bool = True
        do_table_structure: bool = True
        # codex_code_review_af44129.md finding 1 introduced this as
        # optional; codex_code_review_afeba09.md finding 1 pointed out
        # that "optional" meant ordinary requests never actually bound
        # model identity into execution_fingerprint/criterion_cache_key
        # at all -- a later deliberately-updated bundle could change
        # chunks while an unchanged request kept the same cache key. Now
        # MANDATORY (no default): every caller must read
        # `ft_screening.docling_artifacts.load_manifest()["expected_tree_digest"]`
        # (or use `DoclingDocumentParser.recommended_parser_options()`)
        # and declare it, checked against what's ACTUALLY loaded
        # (self._validated_digest) before any conversion proceeds.
        artifacts_manifest_digest: str

    return DoclingParserOptions


_DOCLING_MODEL_BOOTSTRAP_HINT = (
    "run the one-time Docling model bootstrap first: "
    "ft_screening.docling_artifacts.bootstrap(Path.home() / '.cache' / 'docling' / 'models', progress=True) "
    "(downloads ~1.2 GB of layout/OCR/table models, pinned to fixed revisions, to a local, offline-usable "
    "directory -- see CLAUDE.md's Environment section for the exact command, the CA-bundle workaround "
    "needed for this one network call, and why this must be explicit rather than resolved from the network "
    "on first conversion)"
)


class DoclingDocumentParser:
    """Real `DocumentParser` Protocol implementation. Construct ONCE per
    process/run and reuse across every item -- see the module docstring's
    warmup-cost note.

    `parser_options` (per-call, per the Protocol) is validated against
    `_parser_options_model()` -- strict `bool` typing (no loose string/int
    coercion) and `extra="forbid"` (an unknown key raises, not silently
    ignored):
      - `do_ocr: bool` (default True) -- Docling's OCR fallback for
        scanned/image-only pages.
      - `do_table_structure: bool` (default True) -- table structure
        recognition (slower; produces markdown-table chunk text via
        `TableItem.export_to_markdown` when enabled).
      - `artifacts_manifest_digest: str` (MANDATORY, no default -- see
        below).

    `artifacts_path` (constructor-only, not per-call -- it determines
    which physical model files get loaded, a process-lifetime decision,
    not a per-document one): a local directory containing Docling's
    pre-downloaded layout/OCR/table models (see the bootstrap hint in
    `_DOCLING_MODEL_BOOTSTRAP_HINT` / CLAUDE.md). REQUIRED to exist AND
    content-match `docling_artifacts_manifest.json` before the first
    conversion -- codex_code_review_1578af2.md finding 1: with no
    `artifacts_path`, Docling silently resolves a MUTABLE Hugging Face
    `main` revision over the network on first use. codex_code_review_af44129.md
    finding 1 went further: even a local `artifacts_path` was only ever
    checked with `path.is_dir()` -- an EMPTY directory passed. Every file
    under `artifacts_path` is now hashed and checked against a committed
    tree digest (`ft_screening.docling_artifacts.validate()`) before a
    `DocumentConverter` is ever built.

    That verified digest MUST be declared back by the caller via
    `parser_options["artifacts_manifest_digest"]` -- codex_code_review_afeba09.md
    finding 1: making this optional meant ordinary requests never
    actually bound model identity into `execution_fingerprint`/
    `criterion_cache_key` at all, since nothing forced a caller to set
    it. Use `DoclingDocumentParser.recommended_parser_options()` (a
    convenience that reads the committed manifest directly, no parser
    instance needed) when building `ScreeningConfig.parser_options`. A
    declared value that doesn't match what's actually loaded is
    rejected. There is no code path in this class that reaches the
    network.

    Docling bakes OCR/table-structure settings into the (expensive to
    build) `DocumentConverter` itself, not into a per-`convert()` call --
    so a distinct converter is lazily built and cached per
    (do_ocr, do_table_structure) combination actually requested. In
    practice one run uses one `ScreeningConfig.parser_options` throughout,
    so this still only pays the warmup cost once per run. Construction is
    guarded by a lock (finding 4): the orchestrator calls this component
    concurrently under `max_workers > 1`, and unsynchronized
    check-then-create lazy init let two first-use worker threads each
    build the same expensive converter, silently discarding one.
    """

    # This project's own Docling-result -> Chunk conversion logic version
    # (chunking/section-labeling/table-markdown rules in this file) -- NOT
    # a Docling package concept, so unlike PARSER_VERSION below there is
    # nothing to read dynamically; bump this literal if that conversion
    # logic itself ever changes in a way that would produce different
    # chunks from the same Docling output.
    CHUNKER_VERSION = "1"

    def __init__(self, *, artifacts_path: Path | None = None) -> None:
        self._converters: dict[tuple[bool, bool], Any] = {}
        self._converters_lock = threading.Lock()
        self._artifacts_path_input = artifacts_path
        self._resolved_artifacts_path: Path | None = None
        self._validated_digest: str | None = None

    @staticmethod
    def installed_parser_version() -> str:
        """The ACTUAL installed `docling` package version, read via
        package metadata -- no `docling` import needed for this alone
        (cheap, no ~20s Docling import cost). `orchestrator.resolve_components()`
        validates a caller's declared `ScreeningConfig.parser_version`
        against this live value rather than trusting a hardcoded literal
        that could silently drift from what's actually installed."""
        import importlib.metadata

        return importlib.metadata.version("docling")

    @staticmethod
    def recommended_parser_options(**overrides: Any) -> dict[str, Any]:
        """Convenience for building `ScreeningConfig.parser_options` --
        reads the committed manifest's `expected_tree_digest` directly (no
        parser instance needed, no artifacts directory needed either:
        this is what SHOULD be loaded, not a check of what currently is).
        `**overrides` lets a caller still set `do_ocr`/`do_table_structure`
        without hand-copying the digest separately:
            config = ScreeningConfig(..., parser_options=DoclingDocumentParser.recommended_parser_options(do_ocr=False))
        """
        from ft_screening import docling_artifacts

        return {"artifacts_manifest_digest": docling_artifacts.load_manifest()["expected_tree_digest"], **overrides}

    def _ensure_artifacts_resolved(self) -> tuple[Path, str]:
        """Resolves AND content-validates the artifacts directory exactly
        once per instance (memoized) -- callers must hold `_converters_lock`.
        codex_code_review_af44129.md finding 1: `path.is_dir()` alone
        accepted an EMPTY directory; this now checks every file's content
        against the committed manifest (`docling_artifacts.validate()`)
        before any `DocumentConverter` is built, and returns the verified
        digest so `parse()` can bind it to a caller-declared identity."""
        if self._resolved_artifacts_path is not None:
            return self._resolved_artifacts_path, self._validated_digest

        if self._artifacts_path_input is not None:
            path = self._artifacts_path_input
        else:
            from docling.datamodel.settings import settings

            path = settings.cache_dir / "models"
        if not path.is_dir():
            raise FileNotFoundError(
                f"Docling model artifacts directory not found at {path!s} -- {_DOCLING_MODEL_BOOTSTRAP_HINT}."
            )

        from ft_screening import docling_artifacts

        digest = docling_artifacts.validate(path)  # raises ArtifactValidationError on any mismatch
        self._resolved_artifacts_path = path
        self._validated_digest = digest
        return path, digest

    def _converter_for(self, *, do_ocr: bool, do_table_structure: bool) -> Any:
        key = (do_ocr, do_table_structure)
        with self._converters_lock:
            converter = self._converters.get(key)
            if converter is None:
                # Deferred to first actual use -- see _label_sets()'s
                # docstring for why: importing docling itself costs real
                # time, and a caller that only needs this class's type/
                # factory reference (e.g. a test collecting fixtures)
                # shouldn't pay it.
                from docling.datamodel.accelerator_options import AcceleratorDevice
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import PdfPipelineOptions
                from docling.document_converter import DocumentConverter, PdfFormatOption

                artifacts_path, _digest = self._ensure_artifacts_resolved()
                pipeline_options = PdfPipelineOptions(
                    do_ocr=do_ocr,
                    do_table_structure=do_table_structure,
                    table_structure_options={"do_cell_matching": True},
                    accelerator_options={"device": AcceleratorDevice.CPU},
                    artifacts_path=artifacts_path,
                )
                converter = DocumentConverter(
                    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
                )
                self._converters[key] = converter
            return converter

    def parse(
        self,
        *,
        document_id: str,
        document_sha256: str,
        pdf_path: Path,
        parser_options: dict[str, Any],
    ) -> ParsedDocument:
        from docling.datamodel.base_models import ConversionStatus

        # codex_code_review_af44129.md finding 2 (leak 1): options
        # validation and converter/artifact initialization used to happen
        # OUTSIDE any try/except in this method, so their exceptions
        # (a Pydantic ValidationError embedding the caller's raw,
        # possibly-secret option VALUE; a FileNotFoundError/
        # ArtifactValidationError embedding an absolute local path)
        # reached the orchestrator's generic component boundary, which
        # persists `message=str(exc)` verbatim into the public result.
        # Every failure mode in this method now returns
        # _failed_parsed_document() itself -- parse() never lets an
        # unsanitized exception escape, for any reason.
        try:
            options = _parser_options_model().model_validate(parser_options)
        except Exception as exc:  # noqa: BLE001
            _logger.warning("invalid parser_options for document_id=%s: %s", document_id, exc, exc_info=exc)
            return _failed_parsed_document(
                document_id, document_sha256, warning=f"invalid parser_options: {type(exc).__name__}"
            )

        try:
            converter = self._converter_for(do_ocr=options.do_ocr, do_table_structure=options.do_table_structure)
        except Exception as exc:  # noqa: BLE001 -- covers FileNotFoundError, ArtifactValidationError, etc.
            _logger.warning(
                "Docling converter/artifacts initialization failed for document_id=%s: %s",
                document_id, exc, exc_info=exc,
            )
            return _failed_parsed_document(
                document_id, document_sha256, warning=f"parser initialization failed: {type(exc).__name__}"
            )

        if options.artifacts_manifest_digest != self._validated_digest:
            _logger.warning(
                "declared artifacts_manifest_digest mismatch for document_id=%s (declared=%r, actual=%r)",
                document_id, options.artifacts_manifest_digest, self._validated_digest,
            )
            return _failed_parsed_document(
                document_id, document_sha256,
                warning="declared artifacts_manifest_digest does not match the loaded model artifacts",
            )

        try:
            result = converter.convert(str(pdf_path))
        except Exception as exc:  # noqa: BLE001 -- confirmed empirically: corrupt/empty/invalid PDFs raise here
            # finding 2 (orchestrator-side fix also required, see
            # orchestrator.py): full exception detail (which may contain
            # paths or provider text) goes ONLY to the logger; the public
            # DocumentCoverage.warnings entry is deliberately generic.
            _logger.warning("Docling conversion failed for document_id=%s: %s", document_id, exc, exc_info=exc)
            return _failed_parsed_document(
                document_id, document_sha256, warning=f"Docling conversion failed: {type(exc).__name__}"
            )

        # codex_code_review_afeba09.md finding 2: the status check and
        # _build_parsed_document() call used to happen OUTSIDE any
        # try/except -- an unexpected Docling result shape, a
        # label/provenance conversion issue, or a Pydantic failure while
        # constructing a Chunk/ParsedDocument could still raise past
        # parse() with raw internal text (confirmed directly: a
        # constructed document that raised during iterate_items()
        # propagated a message containing an injected secret path).
        # BaseException (KeyboardInterrupt in particular) is deliberately
        # NOT caught here -- only Exception, matching this project's
        # standing convention that cancellation must never be swallowed
        # as a component failure.
        try:
            if result.status in (ConversionStatus.FAILURE, ConversionStatus.SKIPPED):
                _logger.warning("Docling conversion status was %r for document_id=%s", result.status, document_id)
                return _failed_parsed_document(
                    document_id, document_sha256, warning=f"Docling conversion status was {result.status!r}"
                )

            return _build_parsed_document(
                document_id, document_sha256, result, do_ocr=options.do_ocr,
                is_success=result.status == ConversionStatus.SUCCESS,
            )
        except Exception as exc:  # noqa: BLE001
            _logger.warning(
                "building the parsed document failed for document_id=%s: %s", document_id, exc, exc_info=exc
            )
            return _failed_parsed_document(
                document_id, document_sha256, warning=f"parsed-document construction failed: {type(exc).__name__}"
            )
