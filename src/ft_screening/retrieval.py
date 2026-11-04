"""Real `EvidenceRetriever` implementation (plan_v5 step 6, retriever
half), replacing `fakes.FakeEvidenceRetriever` for production use -- same
Protocol, zero orchestrator changes.

Per plan_v3.md's `retrieval.py` section: "deterministic, section-aware
lexical retrieval (BM25 / keyword expansion per criterion)... bounded
fallbacks when the baseline under-retrieves: expand to the complete
relevant section, or send the full normalized document text if it fits."
Retrieval and adjudication stay strictly separate -- this module never
judges stance (supports/contradicts); every span it returns is
`stance="relevant"`, leaving that judgment to `adjudication.py`.

**Cheap-model chunk triage** (plan_v3's "unioned with batched cheap-model
chunk triage") is deliberately NOT implemented here. It requires an LLM
call, is a separate cost/latency/reliability surface, and per plan_v3's
own decomposition should be reviewable on its own -- this module (the
lexical baseline) must stay independently testable and fully offline.
Adding cheap-model triage later means unioning its candidate chunk_ids
with this module's own selection (see `_merge_tiers`) and deduplicating
by chunk_id exactly the way the section/full-document fallback tiers
already do internally -- no redesign needed.

**No persistent vector index in v1** (plan_v3, unchanged): a
`VectorRetriever` variant is a possible later addition behind the same
`EvidenceRetriever` Protocol, added only if the plan's own oracle
experiment (adjudicate once against human-selected evidence, once against
retrieved evidence) shows a measured recall gap this lexical baseline
can't close.

Identity note: unlike `document.py`'s Docling backend, this module has no
downloaded model artifacts and therefore no analogous
`artifacts_manifest_digest` concern -- `LexicalEvidenceRetriever` is pure
Python over already-validated inputs (`ParsedDocument`/`CriterionSpec`),
so `ScreeningConfig.retriever_name`/`retriever_config_version` already
identify it uniquely, and `retriever_options` already flows into
`execution_fingerprint`/`criterion_cache_key` (hashing.py) the same way
`parser_options` does -- changing any tuning knob here automatically
invalidates stale cache entries, no extra binding required. **Changing
this module's own CODE (e.g. the `_SUFFIX_RULES` stemming table or the
BM25 formula) does NOT by itself change any `retriever_options` value,
so it does NOT automatically invalidate a cache computed under the old
code.** That is exactly what `ScreeningConfig.retriever_config_version`
exists for (same convention already established for `parser_version`/
`chunker_version`) -- a deployment that changes this module's candidate-
generation semantics MUST bump `retriever_config_version`, or stale
cached assessments computed under different retrieval semantics can be
silently served. `_QUERY_VERSION` below is bumped alongside any such
change for on-span audit/provenance purposes, but it is NOT itself part
of the cache key (only `ScreeningConfig` fields are).

codex_code_review_5087749.md remediation (2026-08-14), all four findings
closed:
1. **High -- `allowed_sources` was ignored.** `CriterionSpec.allowed_sources`
   ("restrict evidence to these sections, if set" -- plan_v3.md) was never
   read; BM25 and both fallback tiers could return evidence from a
   prohibited section, which the aggregation factory then correctly
   rejected, turning a valid run into an aggregate-stage error. Fixed:
   the candidate chunk pool is filtered to allowed sections (normalized
   the same NFC/strip/casefold way `CriterionSpec` itself normalizes the
   list) BEFORE scoring and before every fallback tier -- an unsectioned
   chunk (`section=None`) can never satisfy a restriction, and an empty
   post-filter pool returns no evidence rather than falling through to
   an unrestricted one.
2. **High -- exact-token matching missed ordinary morphological variants,
   and the section-heading bonus required body-term overlap to apply at
   all.** Added `_normalize_token`, a small deterministic suffix-
   stripping normalizer (see its own docstring) applied uniformly to
   both query terms and chunk/section tokens. Decoupled the section-
   match bonus from body-term frequency: a chunk earns `section_bonus`
   for each query term found in ITS OWN section label regardless of
   whether that term also appears in the chunk's body -- previously a
   body chunk under a matching heading (real Docling shape: the heading
   is its own chunk, `chunk.section == chunk.text` for that chunk) could
   score zero and be excluded even though its enclosing section was an
   exact match.
3. **Medium -- fallback bounds didn't reflect actual context size, and
   section expansion was unbounded.** The full-document-fit check now
   uses the same chunk-join convention `document.py` uses to compute
   `DocumentCoverage.extracted_characters` (`"\n\n"`-joined, not a naive
   `sum(len(text))`) -- see `_canonical_length`. Added `max_candidates`,
   a hard, deterministic ceiling applied as the final step across ALL
   tiers combined (bounds a pathologically large section-fallback
   expansion the same way it bounds everything else) -- truncation keeps
   the highest-scoring, deterministically-ordered subset.
4. **Low -- non-finite tuning values passed validation and failed
   conditionally later, depending on document content.**
   `_RetrieverOptionsModel` now sets `allow_inf_nan=False`, rejecting
   `inf`/`nan` for every float option at validation time regardless of
   whether a later code path would have actually used the offending
   value.

codex_code_review_da65c76.md remediation (2026-08-14) -- a fresh
adversarial pass on the round-1 fix confirmed findings 1 and 4 fully
closed, but found findings 2 and 3's remediation still incomplete:

1. **High -- the hand-written stemmer's general suffix rules still
   missed ordinary screening vocabulary** (bias/biases, control/
   controlled, include/included/inclusion, analysis/analyses,
   diagnose/diagnosis) -- confirmed by direct adversarial replay, not
   just theoretical gaps. Two DIFFERENT problems, not one: consonant
   doubling before -ed/-ing (a general fix risks mangling words that
   legitimately end in a doubled consonant, e.g. "assess"), and
   irregular Latin/Greek-derived singular/plural pairs no general suffix
   rule can safely derive without a dictionary. Fixed with
   `_IRREGULAR_STEMS`, an explicit, curated, per-word exception table
   checked before any suffix rule -- exactly the review's own suggested
   fallback ("make criterion keyword/synonym expansions explicit
   versioned data"). Verified directly against the review's named
   families AND negative non-pairs (include/exclude must never
   conflate -- they are eligibility-criterion antonyms).
2. **High -- `max_candidates` bounded chunk COUNT, which is not a
   context-size bound.** Two failure modes from one design flaw: (a) a
   legitimate 25-tiny-chunk full-document fallback that fit comfortably
   within its character budget was silently cut to 20 chunks just
   because of the old default count cap -- "chunk count is not a
   reliable proxy for whether the relevant section content was
   retrieved" (a variant of the SAME lesson from finding 2's round-1
   fix, now applied to fallback SIZE instead of fallback COVERAGE); (b)
   a single 50,011-character tier-1 chunk passed through completely
   unbounded even with `max_candidates=1`, since capping ITEM COUNT does
   nothing to cap one huge item's SIZE. Fixed by separating the two
   concerns explicitly: `max_candidates` (raised to a generous,
   defensive-only default) still bounds pathological item counts, and a
   NEW `max_evidence_chars` -- a canonical-character budget, applied as
   the true final step via `_apply_evidence_budget` -- bounds the actual
   payload size. A candidate set that fits the budget is returned WHOLE
   regardless of chunk count; a single oversized leading candidate is
   truncated (never silently dropped, never returned unbounded) and its
   `retrieval_method` gets an explicit `_truncated` suffix so the
   truncation is visible on the span itself, without a public schema
   change.

codex_code_review_2aa9cf0.md remediation (2026-08-14) -- accepted the
morphology matrix and non-finite-option fixes outright, but found round
2's OWN fix for findings 2/3 (max_candidates/max_evidence_chars) still
incomplete in two further concrete paths:

1. **High -- `max_candidates` still silently truncated a fitting
   full-document fallback**, just at a higher boundary (101 tiny chunks,
   canonical length 494, dropped the 101st). The bug was structural, not
   a threshold: `_apply_evidence_budget` unconditionally re-sliced
   `ordered[:max_candidates]` before the char-budget walk even started,
   capping the full-document tier by COUNT regardless of how well it fit
   `max_evidence_chars`. Separately, a genuine cross-option bug: the
   full-document tier's ADMISSION check compared against
   `full_document_fallback_max_chars` alone, so a corpus that fit that
   number but not the (possibly tighter) `max_evidence_chars` still got
   admitted, then truncated into a fragment mislabeled
   `lexical_bm25_full_document_fallback`. Fixed: `max_candidates` now
   applies ONLY to the lexical/section portion of `_merge_tiers`'s
   output, applied AFTER (not before) the `min_results` "did we find
   enough" trigger checks (a small `max_candidates` must not itself
   spuriously trigger extra fallback tiers when tier-1 alone already
   found plenty of evidence -- `max_candidates` is an output ceiling, not
   a recall signal). The full-document tier's admission now uses
   `min(full_document_fallback_max_chars, max_evidence_chars)` -- the
   EFFECTIVE budget -- and once admitted is never re-capped by count at
   all, since it is then guaranteed by construction to already fit
   `max_evidence_chars`.
2. **High -- prefix truncation could discard the very evidence that
   produced a chunk's positive score.** `chunk.text[:remaining]` always
   truncates from the start; a 50,022-character chunk whose only matching
   text sat at the END returned 100 characters of pure filler under a
   100-character budget -- retaining the chunk's real BM25 score while
   describing none of the text that earned it. Fixed: `_bm25_scores` now
   also returns which STEMMED query terms actually matched each chunk's
   own BODY text (`_ScoreResult.matched_body_terms` -- deliberately
   excludes section-only matches, see below); a truncated excerpt is now
   a `_windowed_excerpt` centered on the EARLIEST matching term's real
   character position, not a blind prefix, with `start_char`/`end_char`
   describing that window (a genuine SUB-RANGE of the chunk, not the
   chunk's own full offsets -- confirmed the orchestrator's boundary
   validation has no objection to this, since it checks chunk_id
   membership/document identity, never offset-vs-chunk consistency).
   Two cases deliberately still fall back to a plain prefix, not treated
   as a residual gap: a section-metadata-only match (no body position
   exists to anchor a window on) and a genuinely unmatched
   full-document-fallback chunk (included as context, not as scored
   evidence -- there is nothing to preserve).

codex_code_review_eab22bb.md remediation (2026-08-14) -- confirmed the
headline 101-chunk and beginning/middle/end late-hit reproductions fixed,
but found each of round 3's own fixes retained one narrower correctness
path:

1. **High -- a MIXED lexical+full-document result could still be
   count-truncated.** Round 3 exempted the full-document tier from
   `max_candidates` by inspecting each chunk's own method label --
   missing chunks that were ALREADY selected by tier-1/section BEFORE the
   full-document tier fired (e.g. 2 real lexical hits plus a 3rd chunk
   pulled in by full-document fallback): those pre-existing entries kept
   their original method label, so the "not `lexical_bm25_full_document_fallback`"
   filter left them in the still-capped `lexical_and_section` bucket.
   Fixed: admission is now tracked as an explicit `full_document_admitted`
   boolean, not inferred from method labels -- once true, `max_candidates`
   is skipped entirely for the WHOLE selected set (every chunk is already
   guaranteed, by the admission check itself measuring the whole pool's
   canonical length, to fit together), while still preserving each
   chunk's more specific original method label.
2. **High -- window offsets were computed in a normalized copy of the
   text, then applied to the original.** `_earliest_term_occurrence`
   searched an NFC-normalized, lowercased COPY and returned an offset
   into THAT copy; `_windowed_excerpt` then sliced the ORIGINAL string
   with it. Unicode normalization is not length-preserving (decomposed
   `e` + combining acute -> 1 NFC-composed codepoint), so a long enough
   decomposed-Unicode prefix before a real match made the returned window
   land on the wrong region entirely, containing no matching evidence.
   Fixed: `_TOKEN_RE` widened to `[a-zA-Z0-9]+` (was `[a-z0-9]+`) so
   `_earliest_term_match` (renamed, now returns a (start, end) span) can
   search the ORIGINAL, untransformed text directly -- only the matched
   TOKEN itself (pure ASCII by construction) is lowercased for stem
   comparison, which is always length-preserving. Also defined the
   previously-undefined "budget smaller than the matched token itself"
   case explicitly: `_windowed_excerpt` now returns `None` in that case,
   and `_apply_evidence_budget` OMITS the candidate entirely rather than
   emit a positive-scored span whose excerpt cannot possibly contain any
   of the evidence responsible for that score.
3. **Medium -- `_QUERY_VERSION` stayed at `lexical-bm25-3` despite this
   commit changing both scoring output (matched-body-term tracking) and
   emitted-span selection/offset semantics.** Bumped to `lexical-bm25-4`.
   Reaffirmed (not new, but worth restating): `_QUERY_VERSION` is an
   on-span audit value only, never itself part of the cache key --
   `ScreeningConfig.retriever_config_version` is what a deploying caller
   must also bump to prevent a stale cache entry (computed under the
   PRIOR windowing/scoring semantics) from being silently served under
   this one.

codex_code_review_5025df3.md remediation (2026-08-14) -- confirmed EVERY
high-priority finding across all four prior rounds fixed and holding
(mixed-tier admission, decomposed-Unicode offsets, non-finite options,
morphology, source filtering); one medium edge remained:

1. **Medium -- undersized-budget handling checked only the EARLIEST
   matched token, not whether ANY matched token fit.** `_windowed_excerpt`
   asked for the single earliest matching token and immediately gave up
   (returned `None`, omitting the candidate) if THAT ONE token alone
   didn't fit the budget -- even when a LATER matched token in the same
   chunk would have fit comfortably. Exact repro: criterion
   `"randomization trial"` against a chunk containing `"randomization"`
   (13 chars) early and `"trial"` (5 chars) later, `max_evidence_chars=5`
   -- `"trial"` fits exactly, but the omission fired anyway because
   `"randomization"` was checked first and didn't. Fixed:
   `_earliest_fitting_term_match` (renamed from `_earliest_term_match`)
   now walks every matched token in document order and returns the FIRST
   ONE THAT FITS `budget`, only returning `None` when none of them do --
   matching the policy as originally stated ("omit only when the budget
   cannot preserve ANY scoring evidence"), not a stricter one that
   happened to fall out of checking just the first token. Bumped
   `_QUERY_VERSION` again (`lexical-bm25-4` -> `lexical-bm25-5`), since
   this changes which token a truncated span's window is built around.
"""

from __future__ import annotations

import logging
import math
import re
import unicodedata
from collections import Counter
from typing import Any, NamedTuple

from pydantic import BaseModel, ConfigDict, Field

from ft_screening.components import Chunk, ParsedDocument
from ft_screening.schema import CriterionSpec, EvidenceSpan

_logger = logging.getLogger(__name__)

# Bumped whenever candidate-generation semantics change (stemming table,
# scoring formula, fallback policy) -- an on-span audit/provenance value
# only; see the module docstring's "Identity note" for why this alone
# does NOT invalidate the cache (ScreeningConfig.retriever_config_version
# is the field that must be bumped by the deploying caller for that).
_QUERY_VERSION = "lexical-bm25-5"

# Must match document.py's own _CHUNK_SEPARATOR / extracted_characters
# join convention exactly -- the full-document-fallback budget check
# needs the same canonical length the parser itself reports, not an
# under-count that silently admits documents the budget was meant to
# exclude.
_CHUNK_SEPARATOR = "\n\n"

# Case-insensitive character class (not `[a-z0-9]+` + a separate
# lowercasing step) specifically so this ONE regex can be reused both for
# `_tokenize()` (which lowercases the whole text first, for scoring) AND
# for `_earliest_term_match()` (codex_code_review_eab22bb.md finding 2 --
# which must search the ORIGINAL text directly, never a lowercased/
# NFC-normalized copy, since normalization is not length-preserving and
# an offset found in a normalized copy does not reliably identify the
# same character in the original string).
_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")

# A small, deliberately conservative English stopword list -- just common
# function words that would otherwise dominate term-frequency scoring
# without carrying any discriminating signal for a criterion query. Not
# meant to be linguistically exhaustive; it only needs to keep scoring
# from being swamped by words like "the"/"and"/"with".
_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is", "it", "its",
        "of", "on", "or", "that", "the", "this", "to", "was", "were", "will", "with",
    }
)

# Deliberately NOT a full Porter/Snowball stemmer -- a small, deterministic,
# versioned suffix-stripping normalizer tuned for common systematic-review
# screening vocabulary (randomized/randomization/randomly,
# allocation/allocated/allocating, concealed/concealment/concealing, plain
# plurals). Longest/most-specific suffixes are listed first so a compound
# suffix (e.g. "-ization") is matched before its shorter substring
# (e.g. "-ation") would otherwise match instead. Each entry conflates an
# entire derivational family to the SAME stem (verified by
# test_evidence_retriever.py's morphological-variant tests, not assumed):
# "randomized"/"randomizing"/"randomization"/"randomly" -> "random";
# "allocation"/"allocated"/"allocating"/"allocations"/"allocate" -> "alloc".
# Extend this table as new gaps are found in real usage; any change here
# is a candidate-generation semantics change (see the module docstring's
# "Identity note") and must bump `_QUERY_VERSION` plus the deploying
# caller's `retriever_config_version`.
_SUFFIX_RULES: tuple[tuple[str, str], ...] = (
    ("izations", ""),
    ("isations", ""),
    ("ization", ""),
    ("isation", ""),
    ("izing", ""),
    ("izes", ""),
    ("ized", ""),
    ("ize", ""),
    ("ications", ""),
    ("ication", ""),
    ("ations", ""),
    ("ation", ""),
    ("ating", ""),
    ("ates", ""),
    ("ated", ""),
    ("ate", ""),
    ("ments", ""),
    ("ment", ""),
    ("edly", ""),
    ("ing", ""),
    ("ies", "y"),
    ("ied", "y"),
    ("ed", ""),
    ("ly", ""),
)
_MIN_STEM_LEN = 3
# Sibilant/digraph plural endings that need "-es" (not just "-s") stripped
# so the plural and singular forms reduce to the SAME stem: "boxes"/"box",
# "matches"/"match", "classes"/"class", "buses"/"bus".
_ES_PLURAL_SUFFIXES = ("ches", "shes", "ses", "xes", "zes")


def _strip_plural(token: str) -> str:
    if len(token) < _MIN_STEM_LEN + 1:
        return token
    if token.endswith(_ES_PLURAL_SUFFIXES) and len(token) >= _MIN_STEM_LEN + 2:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


# codex_code_review_da65c76.md finding 1: the general suffix-rule engine
# above cannot safely handle two classes of English morphology without a
# dictionary, and attempting to generalize the rules to cover them risks
# over-conflating UNRELATED words (the review's own warning: "the same ad
# hoc rules can over-conflate unrelated words"). Handled instead as an
# explicit, curated, per-word exception table -- checked FIRST, before any
# suffix rule -- exactly the review's own suggested alternative ("make
# criterion keyword/synonym expansions explicit versioned data"):
#
#   1. Consonant-doubling before -ed/-ing ("control"/"controlled"/
#      "controlling" -- English doubles the final consonant here, so a
#      blind "-ed"/"-ing" strip leaves a doubled letter the general rules
#      don't undo). A general undoubling rule was considered and rejected:
#      it would also wrongly mangle words that legitimately end in a
#      doubled consonant on their own ("assess" -> undoubled to "asses").
#   2. Irregular Latin/Greek-derived singular/plural and verb/noun pairs
#      ("analysis"/"analyses"/"analyze", "diagnosis"/"diagnose",
#      "bias"/"biases" -- these do not follow the regular "-s"/"-es"
#      plural pattern a suffix rule can safely generalize from; "bias" is
#      the review's own example of the general plural-"s" rule actively
#      producing DIFFERENT stems for the singular and plural of the same
#      word).
#
# Deliberately NOT the same value for "includ"/"exclud" as any other
# family -- include/exclude are eligibility-criterion ANTONYMS and must
# never collide with each other or with anything else, verified directly
# by test_evidence_retriever.py's negative (non-)pair assertions, not
# just the positive conflation ones. Extend this table one verified word
# family at a time; any change here is a candidate-generation semantics
# change (see the module docstring's "Identity note") and must bump
# `_QUERY_VERSION` plus the deploying caller's `retriever_config_version`.
_IRREGULAR_STEMS: dict[str, str] = {
    # control/controlled/controlling (consonant doubling)
    "control": "control", "controls": "control", "controlled": "control", "controlling": "control",
    # bias/biases/biased (irregular -- see comment above)
    "bias": "bias", "biases": "bias", "biased": "bias",
    # include/included/including/inclusion/inclusive (-d/-sion alternation)
    "include": "includ", "includes": "includ", "included": "includ", "including": "includ",
    "inclusion": "includ", "inclusions": "includ", "inclusive": "includ",
    # exclude/excluded/excluding/exclusion/exclusive (-d/-sion alternation)
    "exclude": "exclud", "excludes": "exclud", "excluded": "exclud", "excluding": "exclud",
    "exclusion": "exclud", "exclusions": "exclud", "exclusive": "exclud",
    # analyze/analyse/analyzed/analyzing/analysis/analyses/analytic(al)
    "analyze": "analy", "analyse": "analy", "analyzes": "analy", "analyses": "analy",
    "analyzed": "analy", "analysed": "analy", "analyzing": "analy", "analysing": "analy",
    "analysis": "analy", "analytic": "analy", "analytical": "analy",
    # diagnose/diagnosed/diagnosing/diagnosis/diagnoses
    "diagnose": "diagnos", "diagnoses": "diagnos", "diagnosed": "diagnos",
    "diagnosing": "diagnos", "diagnosis": "diagnos",
}


def _normalize_token(token: str) -> str:
    irregular = _IRREGULAR_STEMS.get(token)
    if irregular is not None:
        return irregular
    for suffix, replacement in _SUFFIX_RULES:
        if token.endswith(suffix):
            stem = token[: -len(suffix)] + replacement
            # A matched suffix that would strip the token down below the
            # minimum stem length is treated as a false match (e.g. a
            # short word that coincidentally ends in "ly") -- return the
            # token unstemmed rather than over-shortening it.
            return stem if len(stem) >= _MIN_STEM_LEN else token
    return _strip_plural(token)


def _normalize_section_label(value: str) -> str:
    """Same NFC/strip/casefold normalization `CriterionSpec.allowed_sources`
    itself already applies at construction time (schema.py) -- comparing a
    chunk's raw `section` label against an already-normalized
    `allowed_sources` entry requires normalizing the chunk side the same
    way, or an exact match would never occur for anything but
    coincidentally-identical casing/whitespace. Deliberately NOT the same
    function as `_normalize_token`: this normalizes a whole section LABEL
    for exact-string comparison, not an individual word for lexical
    stemming -- conflating the two would let "Methods" and "Method
    Comparison" collide just because they share a stemmed token.
    """
    return unicodedata.normalize("NFC", value).strip().casefold()


class _RetrieverOptionsModel(BaseModel):
    """Strict, `extra="forbid"` validation for `retriever_options` -- same
    pattern as `document.py`'s `_parser_options_model()`: a malformed
    configuration must fail loudly (per-item `retrieval_failed`, via the
    orchestrator boundary) rather than silently coercing or ignoring an
    unknown/mistyped tuning knob. `allow_inf_nan=False`
    (codex_code_review_5087749.md finding 4): a `float` field's `ge`/`gt`
    bound does not exclude `inf`, so without this an infinite tuning
    value would pass validation here and only fail later, conditionally,
    depending on whether a given document/criterion pair actually
    exercised the code path that used it.
    """

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    max_results: int = Field(default=5, ge=1, le=50)
    min_results: int = Field(default=1, ge=0)
    min_score: float = Field(default=0.0, ge=0.0)
    section_match_bonus: float = Field(default=2.0, ge=0.0)
    bm25_k1: float = Field(default=1.5, gt=0.0)
    bm25_b: float = Field(default=0.75, ge=0.0, le=1.0)
    # Admission gate ONLY: whether the full-document tier is attempted at
    # all (if the whole allowed corpus exceeds this, don't offer a
    # "full document" fallback that would actually be a heavily-truncated
    # fragment mislabeled as complete -- return whatever earlier tiers
    # found instead). Does NOT bound the final payload size by itself;
    # see `max_evidence_chars` for that (codex_code_review_da65c76.md
    # finding 2 -- these are deliberately two different bounds).
    full_document_fallback_max_chars: int = Field(default=20000, ge=0)
    # A generous, mostly-defensive ceiling on the NUMBER of spans returned
    # -- codex_code_review_da65c76.md finding 2 found the previous default
    # of 20 silently truncated a legitimate 25-tiny-chunk full-document
    # fallback that fit comfortably within its character budget ("chunk
    # count is not a context-size bound"). Raised so ordinary fallback
    # scenarios essentially never hit it; `max_evidence_chars` below is
    # now the bound that actually matters for context size, applied
    # AFTER this one.
    max_candidates: int = Field(default=100, ge=1, le=500)
    # THE real context-size bound (codex_code_review_da65c76.md finding
    # 2's required fix): a canonical-character budget for the FINAL
    # selected evidence payload, across every tier combined, applied as
    # the very last step. A conservative character budget, not a token
    # estimator -- acceptable per the review's own stated fallback
    # ("a conservative canonical-character budget is acceptable for the
    # current offline layer"); revisit if/when a real token estimator is
    # wired to a specific adjudicator model. See `_apply_evidence_budget`
    # for exactly how this interacts with `max_candidates`: chunks that
    # entirely fit the remaining budget are included whole, in rank
    # order; if the very FIRST (highest-ranked) candidate alone exceeds
    # the budget, it is truncated to fit and its `retrieval_method` gets
    # a `_truncated` suffix rather than being silently dropped or
    # returned unbounded; anything after that point is simply omitted
    # (never a later, smaller candidate reordered ahead of a bigger,
    # higher-ranked one -- the returned set stays a strict rank-ordered
    # prefix).
    max_evidence_chars: int = Field(default=20000, ge=1, le=1_000_000)

    # No min_results <= max_results constraint: `max_results` only caps
    # the first-tier BM25 results; `min_results` is the threshold that
    # triggers the fallback tiers, which are explicitly NOT capped by
    # max_results (per plan_v3.md: a section/full-document fallback can
    # legitimately return more than max_results chunks when the baseline
    # under-retrieves). A caller asking for min_results > max_results is
    # deliberately meaningful: "never trust fewer than N chunks, even if
    # the top-K lexical match alone would have been smaller." The final
    # `max_candidates` ceiling still applies regardless.


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(unicodedata.normalize("NFC", text).lower())


def _query_terms(text: str) -> list[str]:
    """Tokenized, stopword-filtered, stemmed, order-preserving,
    de-duplicated on the STEMMED form -- order only matters for
    readability/determinism of intermediate state, not for the final
    score (BM25 sums over the term set)."""
    seen: list[str] = []
    for token in _tokenize(text):
        if token in _STOPWORDS:
            continue
        stem = _normalize_token(token)
        if stem not in seen:
            seen.append(stem)
    return seen


def _filter_by_allowed_sources(chunks: list[Chunk], allowed_sources: list[str] | None) -> list[Chunk]:
    """codex_code_review_5087749.md finding 1. `allowed_sources` is
    already normalized (NFC/strip/casefold) by `CriterionSpec`'s own
    validator; only the chunk side needs normalizing here. A chunk with
    no section at all (`section=None`) can never satisfy a restriction --
    there is nothing to verify it against, so treating it as
    automatically allowed would defeat the restriction's purpose.
    """
    if allowed_sources is None:
        return chunks
    allowed = set(allowed_sources)
    return [c for c in chunks if c.section is not None and _normalize_section_label(c.section) in allowed]


def _canonical_length(chunks: list[Chunk]) -> int:
    """The same `"\\n\\n"`-joined length `document.py` computes as
    `DocumentCoverage.extracted_characters` (codex_code_review_5087749.md
    finding 3) -- NOT a naive `sum(len(c.text) for c in chunks)`, which
    under-counts by the separator length between every pair of chunks and
    can silently admit a document over budget."""
    if not chunks:
        return 0
    return sum(len(c.text) for c in chunks) + (len(chunks) - 1) * len(_CHUNK_SEPARATOR)


class _ScoreResult(NamedTuple):
    scores: dict[str, float]
    # codex_code_review_2aa9cf0.md finding 2: which STEMMED query terms
    # actually occurred in a chunk's own BODY text (as opposed to only
    # its section label) -- needed later to window a truncated excerpt
    # around real matching evidence instead of blindly taking a prefix.
    # Deliberately excludes section-only matches (a term present ONLY in
    # `chunk.section`, not the body): there is no body-text position to
    # anchor a window on for those, so truncation for that case falls
    # back to a prefix (see `_windowed_excerpt`).
    matched_body_terms: dict[str, frozenset[str]]


def _bm25_scores(chunks: list[Chunk], query_terms: list[str], *, k1: float, b: float, section_bonus: float) -> _ScoreResult:
    """Okapi BM25 (BM25+-style idf, always non-negative) over the
    document's OWN chunks as the corpus, using STEMMED tokens (both body
    text and query terms are stemmed the same way via `_normalize_token`,
    so an inflectional variant in either one still matches). The section-
    match bonus is computed INDEPENDENTLY of body-term frequency
    (codex_code_review_5087749.md finding 2): a chunk earns
    `section_bonus` for each query term present in its OWN section label,
    whether or not that term also appears in the chunk's body -- a body
    chunk under a heading that exactly names the criterion must not score
    zero just because it paraphrases rather than repeats the heading's
    words.
    """
    if not chunks or not query_terms:
        return _ScoreResult({c.chunk_id: 0.0 for c in chunks}, {c.chunk_id: frozenset() for c in chunks})

    token_lists = {
        c.chunk_id: [_normalize_token(t) for t in _tokenize(c.text) if t not in _STOPWORDS] for c in chunks
    }
    lengths = {chunk_id: len(tokens) for chunk_id, tokens in token_lists.items()}
    total_length = sum(lengths.values())
    avgdl = total_length / len(lengths) if lengths else 0.0

    n_docs = len(chunks)
    document_frequency: Counter[str] = Counter()
    for tokens in token_lists.values():
        present = set(tokens)
        for term in query_terms:
            if term in present:
                document_frequency[term] += 1
    idf = {
        term: math.log(1 + (n_docs - document_frequency.get(term, 0) + 0.5) / (document_frequency.get(term, 0) + 0.5))
        for term in query_terms
    }

    scores: dict[str, float] = {}
    matched_body_terms: dict[str, frozenset[str]] = {}
    for chunk in chunks:
        term_frequency = Counter(token_lists[chunk.chunk_id])
        doc_length = lengths[chunk.chunk_id]
        section_tokens = (
            {_normalize_token(t) for t in _tokenize(chunk.section) if t not in _STOPWORDS} if chunk.section else set()
        )
        score = 0.0
        matched: set[str] = set()
        for term in query_terms:
            frequency = term_frequency.get(term, 0)
            if frequency > 0:
                length_norm = (1 - b + b * (doc_length / avgdl)) if avgdl > 0 else 1.0
                denominator = frequency + k1 * length_norm
                score += idf[term] * (frequency * (k1 + 1)) / denominator
                matched.add(term)
            if term in section_tokens:
                score += section_bonus
        scores[chunk.chunk_id] = score
        matched_body_terms[chunk.chunk_id] = frozenset(matched)
    return _ScoreResult(scores, matched_body_terms)


def _sort_key(chunk: Chunk, scores: dict[str, float]) -> tuple[float, int, str]:
    # Deterministic total order: highest score first, then lowest
    # start_char (earlier in the document), then chunk_id -- the last
    # tie-break only matters for chunks sharing both a score AND a
    # start_char, which content-addressed chunk_ids make vanishingly rare
    # but not impossible to rule out structurally.
    return (-scores[chunk.chunk_id], chunk.start_char, chunk.chunk_id)


def _merge_tiers(chunks: list[Chunk], scores: dict[str, float], *, options: _RetrieverOptionsModel) -> list[tuple[Chunk, str]]:
    """Returns a deterministically-ordered, chunk_id-deduplicated list of
    (chunk, retrieval_method) across tiers -- the same dedup mechanism
    plan_v3 describes for unioning a later cheap-model-triage layer's
    candidates with this tier's own.

    codex_code_review_2aa9cf0.md finding 1: `max_candidates` applies
    ONLY when the full-document-fallback tier was NOT admitted -- capping
    by COUNT was what silently truncated a legitimate, fully-fitting
    fallback (25 tiny chunks, then 101 tiny chunks, both well within
    budget) purely because of how many chunks the parser happened to
    emit. The full-document tier's admission is gated by the EFFECTIVE
    final budget -- `min(full_document_fallback_max_chars,
    max_evidence_chars)`, not `full_document_fallback_max_chars` alone
    (the review's "cross-option inconsistency": admitting against the
    looser number, then truncating the result down to the tighter one,
    produced a fragment mislabeled as a complete "full document").

    codex_code_review_eab22bb.md finding 1: admission is now tracked as
    an explicit `full_document_admitted` FLAG, not inferred by inspecting
    each chunk's own method label -- the previous version only exempted
    chunks NEWLY added by the full-document loop from `max_candidates`,
    still count-capping any chunk that had ALREADY been selected by
    tier-1/section fallback before the full-document tier fired (a
    "mixed" case: some real lexical hits plus fallback filling in the
    rest). Once admitted, EVERY selected chunk is guaranteed (by the
    admission check itself, which measures the WHOLE candidate pool's
    canonical length) to fit `max_evidence_chars` together, regardless of
    which tier originally selected it -- so `max_candidates` must not cap
    ANY of them, and existing more-specific method labels
    (`lexical_bm25`/`lexical_bm25_section_fallback`) are deliberately
    preserved rather than overwritten.
    """
    ranked = sorted(chunks, key=lambda c: _sort_key(c, scores))

    tier1 = [c for c in ranked if scores[c.chunk_id] > options.min_score][: options.max_results]
    selected: dict[str, tuple[Chunk, str]] = {c.chunk_id: (c, "lexical_bm25") for c in tier1}

    # The min_results "did we find enough evidence" trigger checks below
    # use the UNCAPPED count -- max_candidates is an OUTPUT ceiling, not a
    # signal that more searching is needed. Applying it before these
    # checks would let a small max_candidates spuriously trigger further
    # fallback tiers even when tier1/section already found plenty of
    # evidence, just less than max_candidates happens to allow through.
    if len(selected) < options.min_results and ranked:
        best = ranked[0]
        if scores[best.chunk_id] > options.min_score and best.section is not None:
            for chunk in chunks:
                if chunk.section == best.section and chunk.chunk_id not in selected:
                    selected[chunk.chunk_id] = (chunk, "lexical_bm25_section_fallback")

    full_document_admitted = False
    if len(selected) < options.min_results:
        effective_full_document_budget = min(options.full_document_fallback_max_chars, options.max_evidence_chars)
        if _canonical_length(chunks) <= effective_full_document_budget:
            full_document_admitted = True
            for chunk in chunks:
                if chunk.chunk_id not in selected:
                    selected[chunk.chunk_id] = (chunk, "lexical_bm25_full_document_fallback")

    ordered = sorted(selected.values(), key=lambda pair: _sort_key(pair[0], scores))
    return ordered if full_document_admitted else ordered[: options.max_candidates]


class _Candidate(NamedTuple):
    chunk: Chunk
    method: str
    excerpt: str
    start_char: int
    end_char: int


def _earliest_fitting_term_match(text: str, matched_terms: frozenset[str], budget: int) -> tuple[int, int] | None:
    """The (start, end) character span, WITHIN THE ORIGINAL `text`
    (never a normalized copy), of the EARLIEST token whose stemmed form
    is in `matched_terms` AND whose own span fits within `budget` -- a
    deterministic anchor for windowing. Returns None only when NO matched
    token fits `budget` at all, not merely when the FIRST one doesn't.

    codex_code_review_5025df3.md: the previous version anchored on the
    single earliest matched token regardless of size, then rejected the
    whole candidate if THAT ONE token didn't fit -- even when a LATER
    matched token in the same chunk would have fit comfortably (e.g.
    criterion "randomization trial" against a chunk containing the long
    word "randomization" early and the short word "trial" later, with a
    5-character budget: "randomization" alone doesn't fit, but "trial"
    does). The defined policy is "omit only when the budget cannot
    preserve ANY scoring evidence," not "omit unless the very first
    match happens to fit."

    codex_code_review_eab22bb.md finding 2 (still the reason this
    searches `text` directly rather than a normalized copy): Unicode
    normalization is not length-preserving, so an offset computed in a
    normalized copy does not reliably identify the same character in the
    original string. `_TOKEN_RE` matches `[a-zA-Z0-9]+` directly against
    `text`, so `match.start()`/`match.end()` are true offsets into `text`
    itself; only the matched TOKEN (already pure ASCII by construction,
    so `.lower()` on it alone is guaranteed length-preserving) is
    transformed, for stem comparison only.
    """
    for match in _TOKEN_RE.finditer(text):
        if match.end() - match.start() > budget:
            continue
        if _normalize_token(match.group().lower()) in matched_terms:
            return match.start(), match.end()
    return None


def _windowed_excerpt(chunk: Chunk, matched_terms: frozenset[str], budget: int) -> tuple[str, int, int] | None:
    """codex_code_review_2aa9cf0.md finding 2: a truncated excerpt must
    retain the evidence that produced the chunk's positive score, not an
    arbitrary prefix that can land far from where the matching term(s)
    actually occur. Returns (excerpt, start_char, end_char) describing a
    budget-bounded WINDOW of `chunk.text` -- not necessarily the chunk's
    own full start/end -- guaranteed to fully contain the earliest
    BUDGET-FITTING matching term's occurrence when `matched_terms` is
    non-empty.

    Returns None -- the "too-small-budget" policy (codex_code_review_eab22bb.md
    finding 2, corrected by codex_code_review_5025df3.md to check every
    matched token, not just the first) -- when NO matched token fits
    `budget`: rather than emit a positive-scored span whose excerpt
    contains none of its own scoring evidence, the caller omits the
    candidate entirely (see `_apply_evidence_budget`).

    A plain prefix window is used, by design, not oversight, ONLY when
    `matched_terms` is empty -- there is no body-text position to anchor
    on at all in that case, for either of two reasons: a section-
    metadata-only match (the chunk's score came entirely from
    `section_match_bonus`, never from its own body), or a genuinely
    unmatched full-document-fallback candidate (nothing in the whole
    document matched at all -- the chunk is included as context, not as
    scored evidence). This is NOT the same case as "matched_terms is
    non-empty but nothing fits the budget," which returns None instead.
    """
    text = chunk.text
    if budget >= len(text):
        return text, chunk.start_char, chunk.end_char

    if not matched_terms:
        window = text[:budget]
        return window, chunk.start_char, chunk.start_char + len(window)

    match = _earliest_fitting_term_match(text, matched_terms, budget)
    if match is None:
        return None

    token_start, token_end = match
    # Center the window on the token, then clamp in two steps so the
    # window both (a) stays within the text's own bounds and (b) still
    # fully contains the token -- step (b) is applied AFTER step (a) and
    # is guaranteed compatible with it (never pushes window_start out of
    # bounds again) precisely because `_earliest_fitting_term_match`
    # already guarantees the token itself fits within `budget`.
    half = budget // 2
    window_start = token_start - half
    window_start = max(0, min(window_start, len(text) - budget))
    window_start = min(window_start, token_start)
    window_start = max(window_start, token_end - budget)
    window_end = window_start + budget
    window = text[window_start:window_end]
    return window, chunk.start_char + window_start, chunk.start_char + window_end


def _apply_evidence_budget(
    ordered: list[tuple[Chunk, str]], *, options: _RetrieverOptionsModel, matched_body_terms: dict[str, frozenset[str]]
) -> list[_Candidate]:
    """The FINAL size bound, applied after every tier has been merged and
    sorted. `ordered` is NOT re-capped by `max_candidates` here
    (codex_code_review_2aa9cf0.md finding 1) -- `_merge_tiers` already
    applied it to the lexical/section portion, and the full-document tier
    (if present) is admitted only when it's already guaranteed to fit
    `max_evidence_chars`, so re-capping here would only ever wrongly
    shrink a legitimately-whole full-document fallback.

    A single candidate whose own text exceeds the remaining budget is
    windowed to fit around its matching evidence (see
    `_windowed_excerpt`), not blindly prefix-truncated
    (codex_code_review_2aa9cf0.md finding 2) -- never silently dropped,
    never returned unbounded, EXCEPT when `_windowed_excerpt` reports the
    budget cannot fit even the matched token itself
    (codex_code_review_eab22bb.md finding 2's "too-small-budget" policy)
    -- in that one case the candidate is omitted entirely rather than
    emit a positive-scored span with no matching evidence in its excerpt.
    ONLY the first (highest-ranked) candidate is ever considered for
    windowed/omitted truncation this way; anything ranked after an
    already-full budget is simply omitted -- the returned set is always a
    strict rank-ordered prefix of `ordered` up to the point the budget
    runs out, never a budget-optimized subset that could promote a
    smaller, lower-ranked candidate ahead of a larger, higher-ranked one.
    """
    candidates: list[_Candidate] = []
    remaining = options.max_evidence_chars

    for chunk, method in ordered:
        if remaining <= 0:
            break
        contribution = len(chunk.text) + (len(_CHUNK_SEPARATOR) if candidates else 0)
        if contribution <= remaining:
            candidates.append(_Candidate(chunk, method, chunk.text, chunk.start_char, chunk.end_char))
            remaining -= contribution
        elif not candidates:
            windowed = _windowed_excerpt(chunk, matched_body_terms.get(chunk.chunk_id, frozenset()), remaining)
            if windowed is not None:
                window_excerpt, window_start, window_end = windowed
                candidates.append(_Candidate(chunk, f"{method}_truncated", window_excerpt, window_start, window_end))
            remaining = 0
        else:
            break

    return candidates


class LexicalEvidenceRetriever:
    """Deterministic, offline, section-aware BM25 evidence retrieval.
    Stateless (holds no per-instance mutable state) -- one shared instance
    is safe to call concurrently across worker threads, unlike
    `document.py`'s `DoclingDocumentParser`, which owns lazily-constructed
    model-backed converters that need their own lock.
    """

    # This project's own BM25/section-bonus/merge-tier retrieval logic
    # version -- there is no third-party package to read a version from
    # (unlike DoclingDocumentParser.installed_parser_version()); bump this
    # literal if the retrieval algorithm itself changes in a way that
    # would produce different results from the same inputs.
    RETRIEVER_CONFIG_VERSION = "1"

    def retrieve(
        self, *, document: ParsedDocument, criterion: CriterionSpec, retriever_options: dict[str, Any]
    ) -> list[EvidenceSpan]:
        try:
            options = _RetrieverOptionsModel.model_validate(retriever_options)
        except Exception as exc:
            # Matches the sanitization already established for
            # document.py's parser_options (codex_code_review_afeba09.md
            # finding 2): a raw pydantic ValidationError's own __str__
            # embeds the invalid input VALUE verbatim, and the
            # orchestrator's generic per-stage failure path
            # (_failed_result) puts str(exc) directly into the PUBLIC
            # ProcessingError.message -- so this must never let the raw
            # exception escape. Full detail goes to the logger only.
            _logger.warning("invalid retriever_options for document_id=%s: %s", document.document_id, exc, exc_info=exc)
            raise ValueError(f"invalid retriever_options: {type(exc).__name__}") from None

        chunks = _filter_by_allowed_sources(list(document.chunks), criterion.allowed_sources)
        if not chunks:
            return []

        query_terms = _query_terms(criterion.eligibility_requirement)
        score_result = _bm25_scores(chunks, query_terms, k1=options.bm25_k1, b=options.bm25_b, section_bonus=options.section_match_bonus)
        ordered = _merge_tiers(chunks, score_result.scores, options=options)
        candidates = _apply_evidence_budget(ordered, options=options, matched_body_terms=score_result.matched_body_terms)

        spans: list[EvidenceSpan] = []
        for rank, candidate in enumerate(candidates, start=1):
            spans.append(
                EvidenceSpan(
                    document_id=document.document_id,
                    document_sha256=document.document_sha256,
                    chunk_id=candidate.chunk.chunk_id,
                    section=candidate.chunk.section,
                    page=candidate.chunk.page,
                    start_char=candidate.start_char,
                    end_char=candidate.end_char,
                    excerpt=candidate.excerpt,
                    stance="relevant",
                    retrieval_method=candidate.method,
                    retrieval_query_version=_QUERY_VERSION,
                    rank=rank,
                    score=score_result.scores[candidate.chunk.chunk_id],
                )
            )
        return spans
