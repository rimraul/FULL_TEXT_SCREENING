"""plan_v3.md's acceptance contract for retrieval.py: deterministic,
section-aware lexical retrieval (BM25 / keyword expansion per criterion),
bounded fallbacks (section expansion, then full-document if it fits) when
the baseline under-retrieves, retrieval and adjudication kept strictly
separate, no persistent vector index in v1, no LLM/network calls -- purely
offline and fast, so (unlike test_document_parser.py) nothing here needs
the `slow` marker.

Written BEFORE `src/ft_screening/retrieval.py` exists (frozen first, per
explicit instruction) -- every test below defines the CONTRACT
`LexicalEvidenceRetriever` must satisfy:

  - evidence spans validated against chunk identity, offsets, excerpt,
    document hash, ordering, ranks, and scores
  - missing evidence (no lexical overlap, fallback unavailable/disabled)
  - empty and partial documents
  - duplicate candidates (distinct chunks sharing identical text)
  - section fallback (baseline under-retrieves -> whole section pulled in)
  - full-document fallback (zero overlap anywhere, document small enough
    to fit; and the char-budget-exceeded case, which must NOT fall back)
  - deterministic results across repeated calls
  - malformed retriever_options (unknown key / wrong type), sanitized
  - concurrency (shared stateless instance, many threads)
  - cache-identity changes (retriever_options flow into criterion_cache_key
    end-to-end through the real orchestrator, not just fakes)

Cheap-model chunk triage (plan_v3's "unioned with batched cheap-model
chunk triage") is explicitly OUT of scope here -- deferred to a later,
separately configurable/reviewable layer. This module's retriever must
stay independently testable and fully offline on its own.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from ft_screening.components import Chunk, ParsedDocument
from ft_screening.fakes import FakeDocumentParser
from ft_screening.schema import CriterionSpec, DocumentCoverage, EvidenceSpan


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _chunk(chunk_id: str, *, section: str | None, page: int, start_char: int, end_char: int, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, section=section, page=page, start_char=start_char, end_char=end_char, text=text)


def _document(chunks: list[Chunk], *, document_id: str = "doc-1", sha: str = "a" * 64, status: str = "complete") -> ParsedDocument:
    total_chars = max((c.end_char for c in chunks), default=0)
    pages = sorted({c.page for c in chunks}) or [1]
    coverage = DocumentCoverage(
        status=status,
        total_pages=max(pages),
        parsed_pages=pages,
        failed_pages=[],
        ocr_used=False,
        extracted_characters=total_chars,
        warnings=[],
    )
    return ParsedDocument(document_id=document_id, document_sha256=sha, coverage=coverage, chunks=tuple(chunks))


def _criterion(
    eligibility_requirement: str, *, criterion_id: str = "c1", allowed_sources: list[str] | None = None
) -> CriterionSpec:
    return CriterionSpec(
        criterion_id=criterion_id, version="1", eligibility_requirement=eligibility_requirement,
        missing_evidence_policy="review", exclusion_reason_code="excl", exclusion_reason_label="Excluded", priority=1,
        allowed_sources=allowed_sources,
    )


def _multi_section_document() -> ParsedDocument:
    """A small, realistic-shaped document spanning 4 sections, 2 pages."""
    chunks = [
        _chunk(
            "c-abstract", section="Abstract", page=1, start_char=0, end_char=95,
            text="This study examines physical activity outcomes in older adults with cardiovascular disease.",
        ),
        _chunk(
            "c-methods-1", section="Methods", page=1, start_char=96, end_char=192,
            text="Participants were randomized to a supervised exercise programme or a usual care control group.",
        ),
        _chunk(
            "c-methods-2", section="Methods", page=2, start_char=193, end_char=270,
            text="Randomization used a computer generated allocation sequence with concealed allocation.",
        ),
        _chunk(
            "c-results", section="Results", page=2, start_char=271, end_char=372,
            text="The exercise group showed significant improvement in six minute walk distance versus control.",
        ),
        _chunk(
            "c-discussion", section="Discussion", page=2, start_char=373, end_char=470,
            text="These findings support supervised exercise as an intervention for cardiovascular rehabilitation.",
        ),
    ]
    return _document(chunks)


@pytest.fixture()
def retriever():
    from ft_screening.retrieval import LexicalEvidenceRetriever

    return LexicalEvidenceRetriever()


# ---------------------------------------------------------------------------
# Evidence span validation: chunk identity, offsets, excerpt, document hash,
# ordering, ranks, scores
# ---------------------------------------------------------------------------


def test_evidence_spans_echo_chunk_identity_and_offsets_exactly(retriever):
    document = _multi_section_document()
    criterion = _criterion("randomized allocation concealed sequence generation")
    spans = retriever.retrieve(document=document, criterion=criterion, retriever_options={})

    assert spans, "expected at least one matching span for this criterion"
    for span in spans:
        matching = next(c for c in document.chunks if c.chunk_id == span.chunk_id)
        assert span.document_id == document.document_id
        assert span.document_sha256 == document.document_sha256
        assert span.section == matching.section
        assert span.page == matching.page
        assert span.start_char == matching.start_char
        assert span.end_char == matching.end_char
        assert span.excerpt == matching.text
        assert span.stance in ("supports", "contradicts", "relevant")
        assert span.retrieval_method
        assert span.retrieval_query_version


def test_ranks_are_contiguous_and_scores_non_increasing(retriever):
    document = _multi_section_document()
    criterion = _criterion("randomized allocation concealed sequence generation")
    spans = retriever.retrieve(document=document, criterion=criterion, retriever_options={"max_results": 10})

    ranks = [s.rank for s in spans]
    assert ranks == list(range(1, len(spans) + 1))
    scores = [s.score for s in spans]
    for a, b in zip(scores, scores[1:]):
        assert a >= b, f"scores must be non-increasing by rank, got {scores!r}"
    assert all(s.score is not None and s.score == s.score and s.score not in (float("inf"), float("-inf")) for s in spans)


def test_higher_lexical_overlap_ranks_first(retriever):
    """c-methods-2 shares more query terms than c-methods-1 for this
    criterion -- verifies scoring genuinely discriminates, not just that
    *some* order is produced."""
    document = _multi_section_document()
    criterion = _criterion("randomization allocation sequence concealed allocation generated")
    spans = retriever.retrieve(document=document, criterion=criterion, retriever_options={"max_results": 10, "min_results": 0})

    assert spans[0].chunk_id == "c-methods-2"


# ---------------------------------------------------------------------------
# allowed_sources (codex_code_review_5087749.md finding 1 -- previously
# ignored entirely, letting prohibited-section evidence reach aggregation)
# ---------------------------------------------------------------------------


def test_allowed_sources_excludes_prohibited_section_chunks(retriever):
    document = _multi_section_document()
    # "exercise" appears in both c-methods-1 (Methods) and c-discussion
    # (Discussion) -- without the restriction both would be candidates.
    criterion = _criterion("supervised exercise intervention", allowed_sources=["Methods"])
    spans = retriever.retrieve(document=document, criterion=criterion, retriever_options={"max_results": 10, "min_results": 0})

    assert spans, "expected at least one Methods-section match"
    assert all(s.section == "Methods" for s in spans)


def test_allowed_sources_normalizes_case_and_whitespace(retriever):
    """CriterionSpec.allowed_sources is already NFC/strip/casefold-normalized
    by the schema's own validator; the chunk side must be normalized the
    same way or an exact match would never occur."""
    document = _multi_section_document()
    criterion = _criterion("supervised exercise intervention", allowed_sources=["  METHODS  "])
    spans = retriever.retrieve(document=document, criterion=criterion, retriever_options={"max_results": 10, "min_results": 0})

    assert spans
    assert all(s.section == "Methods" for s in spans)


def test_allowed_sources_excludes_unsectioned_chunks(retriever):
    chunks = [
        _chunk("c-body", section="Methods", page=1, start_char=0, end_char=50, text="Participants were randomized to treatment groups."),
        _chunk("c-footer", section=None, page=1, start_char=51, end_char=100, text="Randomized allocation footer boilerplate text."),
    ]
    document = _document(chunks)
    criterion = _criterion("randomized allocation", allowed_sources=["Methods"])
    spans = retriever.retrieve(document=document, criterion=criterion, retriever_options={"max_results": 10, "min_results": 0})

    chunk_ids = {s.chunk_id for s in spans}
    assert "c-footer" not in chunk_ids


def test_allowed_sources_with_no_matching_section_returns_no_evidence(retriever):
    document = _multi_section_document()
    criterion = _criterion("randomized allocation concealed sequence generation", allowed_sources=["Appendix"])
    spans = retriever.retrieve(document=document, criterion=criterion, retriever_options={"max_results": 10, "min_results": 1})
    assert spans == []


def test_allowed_sources_respected_through_section_fallback(retriever):
    """Section-fallback expansion must never reach outside the
    allowed_sources-restricted pool -- it can only ever pull in MORE
    chunks from within an already-allowed section."""
    document = _multi_section_document()
    criterion = _criterion("concealed allocation sequence generation", allowed_sources=["Methods"])
    spans = retriever.retrieve(
        document=document, criterion=criterion, retriever_options={"max_results": 1, "min_results": 2}
    )
    assert spans
    assert all(s.section == "Methods" for s in spans)


def test_allowed_sources_respected_through_full_document_fallback(retriever):
    document = _multi_section_document()
    criterion = _criterion("unrelated wombat taxonomy classification zoology", allowed_sources=["Methods"])
    spans = retriever.retrieve(
        document=document, criterion=criterion,
        retriever_options={"min_results": 1, "full_document_fallback_max_chars": 20000},
    )
    assert spans
    assert all(s.section == "Methods" for s in spans)


def test_allowed_sources_end_to_end_does_not_fail_aggregation(tmp_path):
    """Codex's exact adversarial reproduction: a criterion restricted to
    'Methods' with an exact lexical match sitting in 'Results' previously
    reached the aggregation factory, which correctly rejected
    out-of-allowed-sources evidence and turned a valid run into an
    aggregate-stage error. The fix belongs in retrieval (never generate
    the prohibited candidate in the first place), verified here through
    the real orchestrator + real aggregation factory, not just the
    retriever in isolation."""
    from ft_screening.orchestrator import _run_ft_screening
    from ft_screening.retrieval import LexicalEvidenceRetriever
    from ft_screening.schema import CriteriaSet, ScreeningConfig, ScreeningItem, ScreeningRequest

    chunks = [
        _chunk("c-methods", section="Methods", page=1, start_char=0, end_char=60, text="Participants were randomized to treatment or control."),
        _chunk("c-results", section="Results", page=1, start_char=61, end_char=130, text="Randomized participants in the treatment group improved significantly."),
    ]
    parser = FakeDocumentParser(
        chunks_by_document_id={"d1": chunks},
        coverage_by_document_id={
            "d1": DocumentCoverage(status="complete", total_pages=1, parsed_pages=[1], failed_pages=[], ocr_used=False, extracted_characters=130, warnings=[]),
        },
    )
    criteria_set = CriteriaSet(
        criteria_set_id="cs-1", version="1",
        criteria=[_criterion("randomized treatment", criterion_id="randomization", allowed_sources=["Methods"])],
    )
    config = ScreeningConfig(
        parser_name="fake", parser_version="1", chunker_version="1",
        retriever_name="lexical", retriever_config_version="1",
        adjudicator_provider="fake", adjudicator_model="fake-1",
        model_config_id="cfg-1", prompt_template_version="1", output_schema_version="1",
        aggregation_policy_version="1",
    )
    output_dir = tmp_path / "outputs"
    pdf = _make_pdf(tmp_path, "allowed-sources.pdf")
    request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf)],
        criteria_set=criteria_set, config=config, output_dir=output_dir,
    )
    components = _make_e2e_components(output_dir, LexicalEvidenceRetriever(), parser=parser)
    result = _run_ft_screening(request, components)

    item = result.items[0]
    assert item.execution_status == "completed"
    for span in item.eligibility.assessments[0].evidence:
        assert span.section == "Methods"


# ---------------------------------------------------------------------------
# Morphological variants and section-heading bonus (codex_code_review_5087749.md
# finding 2)
# ---------------------------------------------------------------------------


def test_morphological_variant_is_retrieved_without_exact_token_match(retriever):
    """Neither "randomized" nor "allocation" appears verbatim in the
    chunk -- only their inflectional variants "randomly"/"allocated"."""
    chunks = [
        _chunk("c-1", section="Methods", page=1, start_char=0, end_char=70, text="Participants were randomly allocated to treatment or control groups."),
    ]
    document = _document(chunks)
    criterion = _criterion("randomized allocation")
    spans = retriever.retrieve(document=document, criterion=criterion, retriever_options={"full_document_fallback_max_chars": 0})
    assert len(spans) == 1
    assert spans[0].chunk_id == "c-1"
    assert spans[0].score > 0


def test_stemming_conflates_the_randomization_derivational_family():
    from ft_screening.retrieval import _normalize_token

    forms = ["randomized", "randomizing", "randomization", "randomizations", "randomly", "randomize"]
    stems = {_normalize_token(form) for form in forms}
    assert stems == {"random"}, f"expected all forms to conflate to one stem, got {stems!r}"


def test_stemming_conflates_the_allocation_derivational_family():
    from ft_screening.retrieval import _normalize_token

    forms = ["allocation", "allocations", "allocated", "allocating", "allocates", "allocate"]
    stems = {_normalize_token(form) for form in forms}
    assert stems == {"alloc"}, f"expected all forms to conflate to one stem, got {stems!r}"


# codex_code_review_da65c76.md finding 1: "freeze a broader screening-
# vocabulary matrix containing both positive pairs and negative
# non-pairs" -- named families are control/controlled, bias/biases,
# include/included/inclusion, exclude/excluded/exclusion,
# analyze/analysis/analyses, blind/blinded/blinding,
# assign/assigned/assignment, diagnose/diagnosed/diagnosis. Two of these
# (blind/blinded/blinding, assign/assigned/assignment) already worked
# under the existing general suffix rules with no code change -- included
# here anyway as a locked-in regression, not assumed to keep working.
_STEMMING_POSITIVE_FAMILIES: dict[str, list[str]] = {
    "control": ["control", "controls", "controlled", "controlling"],
    "bias": ["bias", "biases", "biased"],
    "include": ["include", "includes", "included", "including", "inclusion", "inclusive"],
    "exclude": ["exclude", "excludes", "excluded", "excluding", "exclusion", "exclusive"],
    "analyze": ["analyze", "analyse", "analyzes", "analyses", "analyzed", "analyzing", "analysis"],
    "blind": ["blind", "blinded", "blinding"],
    "assign": ["assign", "assigned", "assignment"],
    "diagnose": ["diagnose", "diagnoses", "diagnosed", "diagnosing", "diagnosis"],
}


def test_stemming_matrix_positive_families_conflate():
    from ft_screening.retrieval import _normalize_token

    for family, forms in _STEMMING_POSITIVE_FAMILIES.items():
        stems = {_normalize_token(form) for form in forms}
        assert len(stems) == 1, f"family {family!r}: expected all forms to share one stem, got {forms!r} -> {stems!r}"


def test_stemming_matrix_negative_pairs_do_not_conflate():
    """Words that must NOT collide -- either because they are unrelated,
    or (for include/exclude) because they are eligibility-criterion
    ANTONYMS and conflating them would be actively dangerous, not just
    imprecise."""
    from ft_screening.retrieval import _normalize_token

    negative_pairs = [
        ("include", "exclude"),
        ("bias", "basis"),
        ("control", "contrast"),
        ("control", "contribute"),
        ("diagnose", "diagram"),
        ("analyze", "analog"),
        ("blind", "bland"),
        ("assign", "assist"),
    ]
    for word_a, word_b in negative_pairs:
        stem_a, stem_b = _normalize_token(word_a), _normalize_token(word_b)
        assert stem_a != stem_b, f"{word_a!r} and {word_b!r} must not conflate to the same stem, both got {stem_a!r}"


def test_bias_and_biases_conflate_to_the_same_stem_not_different_ones():
    """The review's own most concrete example: the general plural-'s'
    rule previously reduced the SINGULAR and PLURAL of the same word to
    DIFFERENT stems ('bias' -> 'bia', 'biases' -> 'bias')."""
    from ft_screening.retrieval import _normalize_token

    assert _normalize_token("bias") == _normalize_token("biases")


def _heading_and_body_document() -> ParsedDocument:
    """Mirrors document.py's real Docling-backed shape exactly: a heading
    is emitted as its OWN chunk with `section == text == the heading
    label` (see document.py's `emit_chunk(current_section, ...)` call
    right after `current_section = heading_text.strip()`), and the body
    chunk that follows it carries the SAME `section` but its own,
    differently-worded text."""
    heading = "Randomization Methods"
    body = "Participants were assigned to groups using a computer generated sequence with concealed envelopes."
    chunks = [
        _chunk("c-heading", section=heading, page=1, start_char=0, end_char=len(heading), text=heading),
        _chunk("c-body", section=heading, page=1, start_char=len(heading) + 2, end_char=len(heading) + 2 + len(body), text=body),
    ]
    return _document(chunks)


def test_section_heading_bonus_applies_without_body_term_overlap(retriever):
    document = _heading_and_body_document()
    criterion = _criterion("randomization methods")
    spans = retriever.retrieve(document=document, criterion=criterion, retriever_options={"max_results": 10, "min_results": 0})

    chunk_ids_by_method = {s.chunk_id: s.retrieval_method for s in spans}
    assert chunk_ids_by_method.get("c-heading") == "lexical_bm25"
    assert chunk_ids_by_method.get("c-body") == "lexical_bm25", (
        "the body chunk must be retrieved via ordinary BM25 tier-1 scoring (its section heading "
        "matches the criterion) even though its own body text never repeats the heading's words -- "
        f"got {chunk_ids_by_method!r}"
    )


# ---------------------------------------------------------------------------
# Missing evidence / empty / partial documents
# ---------------------------------------------------------------------------


def test_missing_evidence_returns_empty_list_when_fallback_disabled(retriever):
    document = _multi_section_document()
    criterion = _criterion("unrelated wombat taxonomy classification zoology")
    spans = retriever.retrieve(
        document=document, criterion=criterion,
        retriever_options={"min_results": 1, "full_document_fallback_max_chars": 0},
    )
    assert spans == []


def test_empty_document_returns_empty_list(retriever):
    document = _document([], status="partial")  # "complete" requires extracted_characters > 0
    criterion = _criterion("adult participants with cardiovascular disease")
    spans = retriever.retrieve(document=document, criterion=criterion, retriever_options={})
    assert spans == []


def test_partial_document_still_retrieves_from_available_chunks(retriever):
    chunks = [
        _chunk(
            "c-methods-1", section="Methods", page=1, start_char=0, end_char=97,
            text="Participants were randomized to a supervised exercise programme or a usual care control group.",
        ),
    ]
    document = _document(chunks, status="partial")
    criterion = _criterion("randomized supervised exercise programme")
    spans = retriever.retrieve(document=document, criterion=criterion, retriever_options={})
    assert len(spans) == 1
    assert spans[0].chunk_id == "c-methods-1"


# ---------------------------------------------------------------------------
# Duplicate candidates
# ---------------------------------------------------------------------------


def test_duplicate_text_across_distinct_chunks_returns_both_separately(retriever):
    boilerplate = "Funding was provided by grant XYZ. Conflicts of interest: none declared."
    chunks = [
        _chunk("c-footer-p1", section=None, page=1, start_char=0, end_char=len(boilerplate), text=boilerplate),
        _chunk(
            "c-footer-p2", section=None, page=2, start_char=len(boilerplate) + 1,
            end_char=2 * len(boilerplate) + 1, text=boilerplate,
        ),
    ]
    document = _document(chunks)
    criterion = _criterion("funding grant conflicts of interest declared")
    spans = retriever.retrieve(document=document, criterion=criterion, retriever_options={"max_results": 10})

    chunk_ids = [s.chunk_id for s in spans]
    assert sorted(chunk_ids) == ["c-footer-p1", "c-footer-p2"]
    assert len(chunk_ids) == len(set(chunk_ids)), "each distinct chunk_id must appear at most once"
    # deterministic tie-break: identical scores -> lower start_char first
    assert chunk_ids == ["c-footer-p1", "c-footer-p2"]


# ---------------------------------------------------------------------------
# Section fallback
# ---------------------------------------------------------------------------


def test_section_fallback_pulls_in_whole_section_when_under_retrieved(retriever):
    """Only c-methods-2 has real lexical overlap; min_results=2 forces
    expansion to the rest of its section (c-methods-1), even though that
    chunk alone would have scored 0 for this criterion."""
    document = _multi_section_document()
    criterion = _criterion("concealed allocation sequence generation")
    spans = retriever.retrieve(
        document=document, criterion=criterion,
        retriever_options={"max_results": 1, "min_results": 2},
    )

    chunk_ids = {s.chunk_id for s in spans}
    assert chunk_ids == {"c-methods-1", "c-methods-2"}
    methods_1 = next(s for s in spans if s.chunk_id == "c-methods-1")
    methods_2 = next(s for s in spans if s.chunk_id == "c-methods-2")
    assert methods_2.retrieval_method == "lexical_bm25"
    assert methods_1.retrieval_method == "lexical_bm25_section_fallback"


def test_section_fallback_not_triggered_when_min_results_already_met(retriever):
    document = _multi_section_document()
    criterion = _criterion("randomization allocation sequence concealed allocation generated")
    spans = retriever.retrieve(
        document=document, criterion=criterion,
        retriever_options={"max_results": 1, "min_results": 1},
    )
    assert len(spans) == 1
    assert spans[0].retrieval_method == "lexical_bm25"


# ---------------------------------------------------------------------------
# Full-document fallback
# ---------------------------------------------------------------------------


def test_full_document_fallback_when_zero_overlap_and_document_fits(retriever):
    document = _multi_section_document()
    criterion = _criterion("unrelated wombat taxonomy classification zoology")
    spans = retriever.retrieve(
        document=document, criterion=criterion,
        retriever_options={"min_results": 1, "full_document_fallback_max_chars": 20000},
    )
    assert {s.chunk_id for s in spans} == {c.chunk_id for c in document.chunks}
    assert all(s.retrieval_method == "lexical_bm25_full_document_fallback" for s in spans)


def test_full_document_fallback_skipped_when_document_exceeds_char_budget(retriever):
    document = _multi_section_document()
    criterion = _criterion("unrelated wombat taxonomy classification zoology")
    spans = retriever.retrieve(
        document=document, criterion=criterion,
        retriever_options={"min_results": 1, "full_document_fallback_max_chars": 10},
    )
    assert spans == []


def test_full_document_fallback_boundary_uses_canonical_separator_joined_length(retriever):
    """codex_code_review_5087749.md finding 3: two 5-char chunks have a
    raw sum of 10, but a canonical "\\n\\n"-joined length of 12 (matching
    document.py's own DocumentCoverage.extracted_characters convention).
    A budget of 10 must REJECT this document (12 > 10) even though a
    naive sum(len(text)) would have wrongly admitted it."""
    chunks = [
        _chunk("c-1", section=None, page=1, start_char=0, end_char=5, text="alpha"),
        _chunk("c-2", section=None, page=1, start_char=7, end_char=12, text="omega"),
    ]
    document = _document(chunks)
    criterion = _criterion("unrelated wombat taxonomy classification zoology")

    spans = retriever.retrieve(
        document=document, criterion=criterion,
        retriever_options={"min_results": 1, "full_document_fallback_max_chars": 10},
    )
    assert spans == [], "canonical length is 12 (10 chars + 2-char separator) -- a budget of 10 must reject it"

    spans = retriever.retrieve(
        document=document, criterion=criterion,
        retriever_options={"min_results": 1, "full_document_fallback_max_chars": 12},
    )
    assert {s.chunk_id for s in spans} == {"c-1", "c-2"}, "a budget of exactly 12 must admit it"


def test_section_expansion_is_bounded_by_max_candidates(retriever):
    """codex_code_review_5087749.md finding 3's exact adversarial replay:
    one real lexical hit plus 99 same-section chunks that don't otherwise
    match, max_results=1, min_results=2 -- section-fallback expansion
    must not silently return all 100."""
    chunks = [
        _chunk(
            "c-hit", section="Methods", page=1, start_char=0, end_char=60,
            text="Randomized allocation concealed sequence generation methods.",
        ),
    ]
    for i in range(99):
        text = f"Unrelated filler sentence number {i} about something else entirely."
        start = 61 + i * 80
        chunks.append(_chunk(f"c-filler-{i}", section="Methods", page=1, start_char=start, end_char=start + len(text), text=text))
    document = _document(chunks)
    criterion = _criterion("randomized allocation concealed sequence generation")

    spans = retriever.retrieve(
        document=document, criterion=criterion,
        retriever_options={"max_results": 1, "min_results": 2, "max_candidates": 20},
    )
    assert len(spans) <= 20
    assert spans[0].chunk_id == "c-hit", "truncation must keep the highest-scoring chunk, not an arbitrary one"


def test_max_candidates_truncation_keeps_the_highest_scoring_chunks(retriever):
    document = _multi_section_document()
    criterion = _criterion("randomized allocation concealed sequence generation randomization exercise")
    full = retriever.retrieve(document=document, criterion=criterion, retriever_options={"max_results": 10, "min_results": 0, "max_candidates": 200})
    truncated = retriever.retrieve(document=document, criterion=criterion, retriever_options={"max_results": 10, "min_results": 0, "max_candidates": 2})

    assert len(truncated) == min(2, len(full))
    assert [s.chunk_id for s in truncated] == [s.chunk_id for s in full[:2]]


def test_full_document_fallback_with_more_than_old_default_chunk_count_still_returns_everything_that_fits(retriever):
    """codex_code_review_da65c76.md finding 2's first adversarial replay:
    25 tiny chunks (well under the char budget, well over the OLD
    max_candidates default of 20) with zero lexical overlap -- "chunk
    count is not a context-size bound." Every chunk that fits the budget
    must come back, not an arbitrary count-based prefix."""
    chunks = [
        _chunk(f"c-{i}", section="Methods", page=1, start_char=i * 10, end_char=i * 10 + 6, text=f"tiny{i}")
        for i in range(25)
    ]
    document = _document(chunks)
    criterion = _criterion("unrelated wombat taxonomy classification zoology")

    spans = retriever.retrieve(
        document=document, criterion=criterion,
        retriever_options={"min_results": 1, "full_document_fallback_max_chars": 20000, "max_evidence_chars": 20000},
    )
    assert {s.chunk_id for s in spans} == {c.chunk_id for c in chunks}, (
        f"expected all 25 chunks (they fit comfortably within budget), got {len(spans)}"
    )


def test_oversized_single_candidate_is_truncated_not_returned_unbounded(retriever):
    """codex_code_review_da65c76.md finding 2's second adversarial
    replay: a single tier-1 match far larger than the evidence budget
    must not pass through unbounded just because max_candidates=1 was
    already as restrictive as possible on COUNT -- count and size are
    different bounds."""
    huge_text = "randomized allocation concealed sequence generation " + ("x" * 5000)
    document = _document([_chunk("c-huge", section="Methods", page=1, start_char=0, end_char=len(huge_text), text=huge_text)])
    criterion = _criterion("randomized allocation concealed sequence generation")

    spans = retriever.retrieve(
        document=document, criterion=criterion,
        retriever_options={"max_candidates": 1, "full_document_fallback_max_chars": 100, "max_evidence_chars": 100},
    )
    assert len(spans) == 1
    span = spans[0]
    assert len(span.excerpt) <= 100
    assert span.retrieval_method.endswith("_truncated")
    assert span.end_char - span.start_char == len(span.excerpt)


def test_oversized_candidate_truncation_is_deterministic(retriever):
    huge_text = "randomized allocation concealed sequence generation " + ("x" * 5000)
    document = _document([_chunk("c-huge", section="Methods", page=1, start_char=0, end_char=len(huge_text), text=huge_text)])
    criterion = _criterion("randomized allocation concealed sequence generation")
    options = {"max_evidence_chars": 100}

    first = retriever.retrieve(document=document, criterion=criterion, retriever_options=options)
    second = retriever.retrieve(document=document, criterion=criterion, retriever_options=options)
    assert [s.model_dump() for s in first] == [s.model_dump() for s in second]


def test_full_document_fallback_with_more_than_100_chunks_still_returns_everything_that_fits(retriever):
    """codex_code_review_2aa9cf0.md finding 1's exact adversarial replay:
    101 tiny chunks, canonical length only 494 chars (well under both
    default 20,000-char budgets), zero lexical overlap. Raising
    max_candidates from 20 to 100 only moved the failure boundary --
    max_candidates must not cap the full-document tier at all once it's
    been admitted."""
    chunks = [
        _chunk(f"c-{i}", section="Methods", page=1, start_char=i * 5, end_char=i * 5 + 4, text=f"w{i}")
        for i in range(101)
    ]
    document = _document(chunks)
    criterion = _criterion("unrelated wombat taxonomy classification zoology")

    spans = retriever.retrieve(
        document=document, criterion=criterion,
        retriever_options={"min_results": 1, "full_document_fallback_max_chars": 20000, "max_evidence_chars": 20000},
    )
    assert {s.chunk_id for s in spans} == {c.chunk_id for c in chunks}, (
        f"expected all 101 chunks (they fit comfortably within budget), got {len(spans)}"
    )
    assert all(s.retrieval_method == "lexical_bm25_full_document_fallback" for s in spans)


def test_full_document_fallback_not_admitted_when_max_evidence_chars_is_tighter(retriever):
    """codex_code_review_2aa9cf0.md finding 1's "cross-option
    inconsistency": if max_evidence_chars is LOWER than
    full_document_fallback_max_chars, the full-document tier must not be
    admitted at all (and then silently truncated into a fragment
    mislabeled as a complete "full document") -- the admission decision
    must use the tighter, EFFECTIVE budget."""
    chunks = [
        _chunk(f"c-{i}", section="Methods", page=1, start_char=i * 20, end_char=i * 20 + 15, text=f"filler text {i}")
        for i in range(10)
    ]
    document = _document(chunks)
    criterion = _criterion("unrelated wombat taxonomy classification zoology")

    spans = retriever.retrieve(
        document=document, criterion=criterion,
        retriever_options={"min_results": 1, "full_document_fallback_max_chars": 20000, "max_evidence_chars": 50},
    )
    assert not any(s.retrieval_method.startswith("lexical_bm25_full_document_fallback") for s in spans), (
        "the full-document tier must not be admitted when the corpus does not fit max_evidence_chars, "
        f"got methods {[s.retrieval_method for s in spans]!r}"
    )


def test_small_max_candidates_does_not_spuriously_trigger_fallback_tiers(retriever):
    """A small max_candidates is an OUTPUT ceiling, not a signal that
    tier-1 under-retrieved -- 5 chunks all score positively (well over
    min_results=3), so no fallback tier should ever fire just because
    max_candidates=2 will truncate the final output down further."""
    chunks = [
        _chunk(
            f"c-{i}", section="Methods", page=1, start_char=i * 60, end_char=i * 60 + 55,
            text=f"Randomized allocation concealed sequence generation variant {i}.",
        )
        for i in range(5)
    ]
    document = _document(chunks)
    criterion = _criterion("randomized allocation concealed sequence generation")

    spans = retriever.retrieve(
        document=document, criterion=criterion,
        retriever_options={"max_results": 10, "min_results": 3, "max_candidates": 2},
    )
    assert len(spans) == 2
    assert all(s.retrieval_method == "lexical_bm25" for s in spans), (
        f"expected only tier-1 results (tier-1 alone already met min_results, uncapped), "
        f"got methods {[s.retrieval_method for s in spans]!r}"
    )


def test_mixed_lexical_and_full_document_fallback_is_not_count_truncated(retriever):
    """codex_code_review_eab22bb.md finding 1's exact adversarial replay:
    2 chunks score lexically (different section from the 3rd), min_results=3
    admits full-document fallback for the 3rd -- max_candidates=1 must
    not silently drop one of the two PRE-EXISTING lexical chunks just
    because they weren't the ones newly added by the full-document tier."""
    chunks = [
        _chunk("m0", section="Methods", page=1, start_char=0, end_char=50, text="Randomized allocation used for group assignment."),
        _chunk("m1", section="Methods", page=1, start_char=51, end_char=110, text="Randomized allocation determined the final groups."),
        _chunk("m2", section="Discussion", page=1, start_char=111, end_char=160, text="These results support the study conclusions overall."),
    ]
    document = _document(chunks)
    criterion = _criterion("randomized allocation")

    spans = retriever.retrieve(
        document=document, criterion=criterion,
        retriever_options={"min_results": 3, "max_candidates": 1, "full_document_fallback_max_chars": 20000, "max_evidence_chars": 20000},
    )
    assert {s.chunk_id for s in spans} == {"m0", "m1", "m2"}, (
        f"the full-document tier was admitted, so every chunk (including the pre-existing lexical hits) "
        f"must survive regardless of max_candidates, got {[s.chunk_id for s in spans]!r}"
    )
    methods_by_id = {s.chunk_id: s.retrieval_method for s in spans}
    assert methods_by_id["m0"] == "lexical_bm25"
    assert methods_by_id["m1"] == "lexical_bm25"
    assert methods_by_id["m2"] == "lexical_bm25_full_document_fallback"


def _huge_chunk_with_match_at(position: str) -> tuple[ParsedDocument, CriterionSpec]:
    """A ~10,000-char Methods chunk with the phrase "randomized
    allocation" placed at the requested position -- filler text
    elsewhere shares no query vocabulary at all."""
    phrase = "randomized allocation"
    filler = "the quick brown fox jumps over lazy dogs repeatedly here "
    padding = (filler * 200)[:9900]
    if position == "start":
        text = phrase + " " + padding
    elif position == "end":
        text = padding + " " + phrase
    else:
        half = len(padding) // 2
        text = padding[:half] + " " + phrase + " " + padding[half:]
    document = _document([_chunk("c-huge", section="Methods", page=1, start_char=0, end_char=len(text), text=text)])
    criterion = _criterion("randomized allocation")
    return document, criterion


def test_truncated_excerpt_preserves_matching_evidence_at_end(retriever):
    """codex_code_review_2aa9cf0.md finding 2's exact adversarial replay:
    a positively-scored chunk whose matching phrase sits near the END --
    a naive chunk.text[:budget] prefix would return only filler."""
    document, criterion = _huge_chunk_with_match_at("end")
    spans = retriever.retrieve(document=document, criterion=criterion, retriever_options={"max_evidence_chars": 100})

    assert len(spans) == 1
    span = spans[0]
    assert span.score > 0, "sanity check: this chunk must have actually scored positively"
    assert span.retrieval_method.endswith("_truncated")
    assert len(span.excerpt) <= 100
    assert "randomized" in span.excerpt.lower() or "allocation" in span.excerpt.lower(), (
        f"a positively-scored span must retain at least one matching term, got excerpt={span.excerpt!r}"
    )


def test_truncated_excerpt_preserves_matching_evidence_in_middle(retriever):
    document, criterion = _huge_chunk_with_match_at("middle")
    spans = retriever.retrieve(document=document, criterion=criterion, retriever_options={"max_evidence_chars": 100})

    assert len(spans) == 1
    span = spans[0]
    assert span.score > 0
    assert "randomized" in span.excerpt.lower() or "allocation" in span.excerpt.lower(), (
        f"a positively-scored span must retain at least one matching term, got excerpt={span.excerpt!r}"
    )


def test_truncated_excerpt_preserves_matching_evidence_at_start(retriever):
    document, criterion = _huge_chunk_with_match_at("start")
    spans = retriever.retrieve(document=document, criterion=criterion, retriever_options={"max_evidence_chars": 100})

    assert len(spans) == 1
    span = spans[0]
    assert span.score > 0
    assert "randomized" in span.excerpt.lower() or "allocation" in span.excerpt.lower(), (
        f"a positively-scored span must retain at least one matching term, got excerpt={span.excerpt!r}"
    )


def test_truncated_excerpt_offsets_are_internally_consistent():
    from ft_screening.retrieval import LexicalEvidenceRetriever

    document, criterion = _huge_chunk_with_match_at("end")
    chunk_text = document.chunks[0].text
    chunk_start = document.chunks[0].start_char
    spans = LexicalEvidenceRetriever().retrieve(document=document, criterion=criterion, retriever_options={"max_evidence_chars": 100})

    span = spans[0]
    assert span.end_char - span.start_char == len(span.excerpt)
    # the window must be a real sub-range of the underlying chunk
    assert 0 <= span.start_char
    assert span.end_char <= 10000 + 200  # generous upper bound on the fixture's own total length
    # codex_code_review_eab22bb.md finding 2's explicitly required
    # assertion: the reported offsets must describe EXACTLY the returned
    # excerpt when sliced from the chunk's own original text.
    assert chunk_text[span.start_char - chunk_start : span.end_char - chunk_start] == span.excerpt


def test_truncated_excerpt_preserves_evidence_with_decomposed_unicode_before_the_hit(retriever):
    """codex_code_review_eab22bb.md finding 2's exact adversarial replay:
    a long prefix of DECOMPOSED Unicode (combining-mark sequences) before
    the real match -- NFC normalization shortens this prefix (composing
    each base+combining-mark pair into one codepoint), so any offset
    computed in a normalized copy no longer identifies the same position
    in the ORIGINAL string. The window must still land on the real match,
    not on the (now-misaligned) filler region."""
    decomposed_e_acute = "é"  # "e" + COMBINING ACUTE ACCENT -- 2 codepoints, NFC-composes to 1 ("é")
    prefix = decomposed_e_acute * 100  # 200 codepoints, NFC-normalizes down to 100
    text = prefix + " randomized allocation " + ("filler " * 50)
    document = _document([_chunk("c-huge", section="Methods", page=1, start_char=0, end_char=len(text), text=text)])
    criterion = _criterion("randomized allocation")

    spans = retriever.retrieve(document=document, criterion=criterion, retriever_options={"max_evidence_chars": 40})

    assert len(spans) == 1
    span = spans[0]
    assert span.score > 0
    assert "randomized" in span.excerpt.lower() or "allocation" in span.excerpt.lower(), (
        f"expected the window to land on the real match despite the decomposed-Unicode prefix, "
        f"got excerpt={span.excerpt!r}"
    )
    assert text[span.start_char : span.end_char] == span.excerpt


def test_budget_shorter_than_matched_token_omits_the_candidate_instead_of_returning_filler(retriever):
    """codex_code_review_eab22bb.md finding 2's "too-small-budget"
    policy: max_evidence_chars smaller than the matched token itself
    ("randomized" is 10 characters) cannot possibly preserve any matching
    evidence -- the candidate must be OMITTED, never returned as a
    positive-scored span with unrelated filler in its excerpt."""
    huge_text = ("filler " * 100) + "randomized allocation" + (" more filler" * 100)
    document = _document([_chunk("c-huge", section="Methods", page=1, start_char=0, end_char=len(huge_text), text=huge_text)])
    criterion = _criterion("randomized allocation")

    spans = retriever.retrieve(document=document, criterion=criterion, retriever_options={"max_evidence_chars": 3})
    assert spans == [], f"expected the oversized, unpreservable candidate to be omitted entirely, got {spans!r}"


def test_windowing_skips_an_oversized_earliest_match_for_a_later_fitting_one(retriever):
    """codex_code_review_5025df3.md's exact adversarial replay: the
    EARLIEST matched token ("randomization", 13 chars) does not fit a
    5-character budget, but a LATER matched token ("trial", 5 chars)
    fits exactly -- the policy is "omit only when NO matched token fits,"
    not "omit unless the very first one happens to fit"."""
    text = "randomization " + ("filler " * 50) + "trial " + ("more filler " * 50)
    document = _document([_chunk("c-huge", section="Methods", page=1, start_char=0, end_char=len(text), text=text)])
    criterion = _criterion("randomization trial")

    spans = retriever.retrieve(document=document, criterion=criterion, retriever_options={"max_evidence_chars": 5})

    assert len(spans) == 1, f"expected the later-fitting 'trial' match to anchor a window, got {spans!r}"
    span = spans[0]
    assert span.score > 0
    assert span.excerpt == "trial"
    assert text[span.start_char : span.end_char] == span.excerpt


def test_windowed_span_carries_the_current_query_version(retriever):
    """codex_code_review_eab22bb.md finding 3: a windowed span must be
    audit-distinguishable from one produced by a prior (pre-windowing)
    implementation -- _QUERY_VERSION must be bumped whenever candidate-
    generation/selection semantics change, not just when scoring changes."""
    from ft_screening.retrieval import _QUERY_VERSION

    document, criterion = _huge_chunk_with_match_at("end")
    spans = retriever.retrieve(document=document, criterion=criterion, retriever_options={"max_evidence_chars": 100})

    assert len(spans) == 1
    assert spans[0].retrieval_query_version == _QUERY_VERSION == "lexical-bm25-5"


def test_truncated_evidence_with_matched_term_survives_through_orchestrator(tmp_path):
    """codex_code_review_2aa9cf0.md finding 2: "run them through the
    orchestrator boundary" -- a windowed excerpt reports start_char/
    end_char that describe the WINDOW, not the underlying chunk's own
    recorded offsets; confirms the orchestrator's boundary validation
    (which checks document identity and chunk_id membership, not
    excerpt/offset consistency against the chunk) accepts this without
    treating it as a malformed component."""
    from ft_screening.orchestrator import _run_ft_screening
    from ft_screening.retrieval import LexicalEvidenceRetriever
    from ft_screening.schema import DocumentCoverage

    document, criterion = _huge_chunk_with_match_at("end")
    huge_chunk = document.chunks[0]
    parser = FakeDocumentParser(
        chunks_by_document_id={"d1": list(document.chunks)},
        coverage_by_document_id={
            "d1": DocumentCoverage(
                status="complete", total_pages=1, parsed_pages=[1], failed_pages=[], ocr_used=False,
                extracted_characters=huge_chunk.end_char, warnings=[],
            )
        },
    )
    output_dir = tmp_path / "outputs"
    pdf = _make_pdf(tmp_path, "truncated-evidence.pdf")
    from ft_screening.schema import CriteriaSet, ScreeningConfig, ScreeningItem, ScreeningRequest

    criteria_set = CriteriaSet(criteria_set_id="cs-1", version="1", criteria=[criterion])
    config = ScreeningConfig(
        parser_name="fake", parser_version="1", chunker_version="1",
        retriever_name="lexical", retriever_config_version="1",
        retriever_options={"max_evidence_chars": 100},
        adjudicator_provider="fake", adjudicator_model="fake-1",
        model_config_id="cfg-1", prompt_template_version="1", output_schema_version="1",
        aggregation_policy_version="1",
    )
    request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf)],
        criteria_set=criteria_set, config=config, output_dir=output_dir,
    )
    components = _make_e2e_components(output_dir, LexicalEvidenceRetriever(), parser=parser)
    result = _run_ft_screening(request, components)

    item = result.items[0]
    assert item.execution_status == "completed"
    evidence = item.eligibility.assessments[0].evidence
    assert len(evidence) == 1
    assert "randomized" in evidence[0].excerpt.lower() or "allocation" in evidence[0].excerpt.lower()


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_results_are_deterministic_across_repeated_calls(retriever):
    document = _multi_section_document()
    criterion = _criterion("randomized allocation concealed sequence generation")
    first = retriever.retrieve(document=document, criterion=criterion, retriever_options={"max_results": 10})
    second = retriever.retrieve(document=document, criterion=criterion, retriever_options={"max_results": 10})
    assert [s.model_dump() for s in first] == [s.model_dump() for s in second]


# ---------------------------------------------------------------------------
# Malformed retriever_options
# ---------------------------------------------------------------------------


def test_unknown_retriever_option_is_rejected_without_leaking_the_value(retriever):
    document = _multi_section_document()
    criterion = _criterion("randomized allocation")
    with pytest.raises(Exception) as excinfo:
        retriever.retrieve(
            document=document, criterion=criterion,
            retriever_options={"not_a_real_option": "SECRET-sentinel-value"},
        )
    assert "SECRET-sentinel-value" not in str(excinfo.value)


def test_wrong_type_retriever_option_is_rejected_without_leaking_the_value(retriever):
    document = _multi_section_document()
    criterion = _criterion("randomized allocation")
    with pytest.raises(Exception) as excinfo:
        retriever.retrieve(
            document=document, criterion=criterion,
            retriever_options={"max_results": "SECRET-sentinel-value"},
        )
    assert "SECRET-sentinel-value" not in str(excinfo.value)


def test_min_results_greater_than_max_results_is_a_valid_configuration(retriever):
    """max_results caps only the first-tier BM25 results; min_results is
    the threshold that triggers fallback expansion, which is deliberately
    NOT capped by max_results (see test_section_fallback_pulls_in_whole_section_when_under_retrieved).
    min_results > max_results is therefore meaningful, not malformed."""
    document = _multi_section_document()
    criterion = _criterion("randomized allocation concealed sequence generation")
    spans = retriever.retrieve(
        document=document, criterion=criterion,
        retriever_options={"max_results": 1, "min_results": 5},
    )
    assert len(spans) >= 1


def test_infinite_section_match_bonus_is_rejected_unconditionally(retriever):
    """codex_code_review_5087749.md finding 4: section_match_bonus=inf
    previously validated successfully and only failed LATER, and only
    for a document/criterion pair whose section actually matched --
    i.e. validity depended on document content. Rejected here at
    validation time regardless of whether this particular document would
    ever have exercised the section-bonus branch."""
    document = _document([_chunk("c-1", section=None, page=1, start_char=0, end_char=10, text="irrelevant")])
    criterion = _criterion("irrelevant query terms here")
    with pytest.raises(Exception):
        retriever.retrieve(document=document, criterion=criterion, retriever_options={"section_match_bonus": float("inf")})


def test_infinite_bm25_k1_is_rejected_unconditionally(retriever):
    document = _document([_chunk("c-1", section=None, page=1, start_char=0, end_char=10, text="irrelevant")])
    criterion = _criterion("irrelevant query terms here")
    with pytest.raises(Exception):
        retriever.retrieve(document=document, criterion=criterion, retriever_options={"bm25_k1": float("inf")})


def test_nan_min_score_is_rejected(retriever):
    document = _document([_chunk("c-1", section=None, page=1, start_char=0, end_char=10, text="irrelevant")])
    criterion = _criterion("irrelevant query terms here")
    with pytest.raises(Exception):
        retriever.retrieve(document=document, criterion=criterion, retriever_options={"min_score": float("nan")})


# No orchestrator end-to-end leak test for the non-finite-float case
# (unlike test_malformed_option_end_to_end_does_not_leak_via_orchestrator
# below, which uses a wrong-TYPE value): ScreeningConfig.retriever_options
# already goes through schema.py's own JSON-safety validator
# (_deep_validate_json), which rejects non-finite floats at REQUEST
# CONSTRUCTION time -- confirmed directly: building a ScreeningRequest
# with retriever_options={"section_match_bonus": float("inf")} raises
# there, before an orchestrator run (and therefore _failed_result) is
# ever reached. The direct-call tests above are the correct and
# sufficient coverage for finding 4's own concern (validity must not
# depend on document content) -- this is a second, independent layer of
# protection, not a gap.


def test_malformed_option_end_to_end_does_not_leak_via_orchestrator(tmp_path):
    """The generic orchestrator failure path (_failed_result) uses
    str(exc) directly -- an unsanitized pydantic ValidationError escaping
    retrieve() would embed the raw option value in the PUBLIC
    ProcessingError.message, the same class of leak already fixed for
    document.py's parser_options (codex_code_review_afeba09.md finding 2)."""
    from ft_screening.retrieval import LexicalEvidenceRetriever

    result = _run_one_item(
        tmp_path, retriever=LexicalEvidenceRetriever(), run_id="retriever-malformed-options",
        retriever_options={"max_results": "SECRET-orchestrator-sentinel"},
    )
    item = result.items[0]
    assert item.execution_status == "processing_failed"
    for error in item.errors:
        assert "SECRET-orchestrator-sentinel" not in error.message


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


def test_concurrent_calls_on_shared_instance_do_not_cross_contaminate(retriever):
    documents = [
        _document(
            [_chunk(f"doc{i}-c0", section="Methods", page=1, start_char=0, end_char=40, text=f"keyword{i} appears in this document body text.")],
            document_id=f"doc-{i}", sha=(f"{i}" * 64)[:64],
        )
        for i in range(8)
    ]
    criteria = [_criterion(f"keyword{i}", criterion_id=f"crit-{i}") for i in range(8)]

    def call(i: int):
        return i, retriever.retrieve(document=documents[i], criterion=criteria[i], retriever_options={})

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(call, range(8)))

    for i, spans in results:
        assert len(spans) == 1
        assert spans[0].chunk_id == f"doc{i}-c0"
        assert spans[0].document_id == f"doc-{i}"


def test_concurrent_end_to_end_through_orchestrator(tmp_path):
    from ft_screening.orchestrator import _run_ft_screening
    from ft_screening.retrieval import LexicalEvidenceRetriever
    from ft_screening.schema import ScreeningItem

    items = [ScreeningItem(document_id=f"d{i}", report_id=f"r{i}", pdf_path=_make_pdf(tmp_path, f"concurrent-{i}.pdf")) for i in range(3)]
    output_dir = tmp_path / "outputs"
    request = _make_e2e_request(output_dir, items, max_workers=3)
    components = _make_e2e_components(output_dir, LexicalEvidenceRetriever())

    result = _run_ft_screening(request, components)
    assert all(item.execution_status == "completed" for item in result.items)


# ---------------------------------------------------------------------------
# Cache-identity changes (end-to-end, real orchestrator + real hashing)
# ---------------------------------------------------------------------------


def _multi_chunk_fake_parser() -> "object":
    """A FakeDocumentParser whose one document has 3 chunks that all
    genuinely overlap the standard e2e criterion ("randomized allocation
    concealed sequence generation") to varying degrees -- needed because
    the DEFAULT FakeDocumentParser only ever produces one chunk, which
    would make evidence-count comparisons across retriever_options
    meaningless (a single chunk is either retrieved or not, regardless of
    max_results)."""
    from ft_screening.fakes import FakeDocumentParser
    from ft_screening.schema import DocumentCoverage

    texts = [
        "Randomized allocation used a concealed sequence generation method for group assignment.",
        "The concealed sequence generation process followed a randomized protocol design.",
        "Randomized participants entered the allocation phase after concealed sequence generation review.",
    ]
    chunks = []
    offset = 0
    for i, text in enumerate(texts):
        end = offset + len(text)
        chunks.append(_chunk(f"mc-{i}", section="Methods", page=1, start_char=offset, end_char=end, text=text))
        offset = end + 1
    total_chars = offset
    return FakeDocumentParser(
        chunks_by_document_id={"d1": chunks},
        coverage_by_document_id={
            "d1": DocumentCoverage(
                status="complete", total_pages=1, parsed_pages=[1], failed_pages=[],
                ocr_used=False, extracted_characters=total_chars, warnings=[],
            )
        },
    )


def test_changing_retriever_options_invalidates_the_cache_end_to_end(tmp_path):
    from ft_screening.orchestrator import _run_ft_screening
    from ft_screening.retrieval import LexicalEvidenceRetriever

    output_dir = tmp_path / "outputs"
    pdf = _make_pdf(tmp_path, "cache-identity.pdf")

    request_a = _make_e2e_request(output_dir, pdf, retriever_options={"max_results": 1})
    components_a = _make_e2e_components(output_dir, LexicalEvidenceRetriever(), parser=_multi_chunk_fake_parser())
    result_a = _run_ft_screening(request_a, components_a)
    assert result_a.items[0].execution_status == "completed"
    evidence_a = result_a.items[0].eligibility.assessments[0].evidence

    request_b = _make_e2e_request(output_dir, pdf, run_id="run-b", retriever_options={"max_results": 10, "min_results": 0})
    components_b = _make_e2e_components(output_dir, LexicalEvidenceRetriever(), parser=_multi_chunk_fake_parser())
    result_b = _run_ft_screening(request_b, components_b)
    assert result_b.items[0].execution_status == "completed"
    evidence_b = result_b.items[0].eligibility.assessments[0].evidence

    assert len(evidence_a) != len(evidence_b), (
        "different retriever_options must not silently share a cached assessment -- "
        f"got {len(evidence_a)} vs {len(evidence_b)} evidence spans"
    )


# ---------------------------------------------------------------------------
# Protocol conformance enrollment sanity (full suite lives in
# tests/contract/test_protocol_conformance.py -- this just confirms the
# factory used there constructs cleanly)
# ---------------------------------------------------------------------------


def test_retriever_constructs_with_no_arguments():
    from ft_screening.retrieval import LexicalEvidenceRetriever

    LexicalEvidenceRetriever()


# ---------------------------------------------------------------------------
# e2e helpers (mirrors tests/acceptance/test_document_parser.py's own
# _make_e2e_request/_make_e2e_components, adapted for a real retriever
# instead of a real parser -- uses FakeDocumentParser, fast/offline)
# ---------------------------------------------------------------------------


def _make_pdf(tmp_path, name: str):
    import fitz  # PyMuPDF

    path = tmp_path / "pdfs" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Randomized allocation concealed sequence generation methods results.")
    doc.save(str(path))
    doc.close()
    return path


def _make_e2e_request(output_dir, pdf_path_or_items, *, run_id: str = "run-a", **config_overrides):
    from ft_screening.schema import CriteriaSet, ScreeningConfig, ScreeningItem, ScreeningRequest

    criteria_set = CriteriaSet(
        criteria_set_id="cs-1", version="1",
        criteria=[_criterion("randomized allocation concealed sequence generation", criterion_id="randomization")],
    )
    defaults = dict(
        parser_name="fake", parser_version="1", chunker_version="1",
        retriever_name="lexical", retriever_config_version="1",
        adjudicator_provider="fake", adjudicator_model="fake-1",
        model_config_id="cfg-1", prompt_template_version="1", output_schema_version="1",
        aggregation_policy_version="1", cache_mode="use",
    )
    defaults.update(config_overrides)
    config = ScreeningConfig(**defaults)
    if isinstance(pdf_path_or_items, list):
        items = pdf_path_or_items
    else:
        items = [ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf_path_or_items)]
    return ScreeningRequest(items=items, criteria_set=criteria_set, config=config, output_dir=output_dir, run_id=run_id)


def _make_e2e_components(output_dir, retriever, *, parser=None):
    from ft_screening.components import Components
    from ft_screening.fakes import FakeCriterionAdjudicator, FakeDocumentParser
    from ft_screening.store import FileScreeningStore

    return Components(
        parser=parser or FakeDocumentParser(), retriever=retriever, adjudicator=FakeCriterionAdjudicator(),
        store=FileScreeningStore(output_dir=output_dir),
    )


def _run_one_item(tmp_path, *, retriever, run_id: str, retriever_options: dict):
    from ft_screening.orchestrator import _run_ft_screening

    output_dir = tmp_path / f"out-{run_id}"
    pdf = _make_pdf(tmp_path, f"{run_id}.pdf")
    request = _make_e2e_request(output_dir, pdf, run_id=run_id, retriever_options=retriever_options)
    components = _make_e2e_components(output_dir, retriever)
    return _run_ft_screening(request, components)
