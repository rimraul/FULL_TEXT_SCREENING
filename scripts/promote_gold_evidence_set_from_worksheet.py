"""Promotes the completed human-review worksheet
(outputs/gold_evidence_set_v1/gold_evidence_review_worksheet_v1.csv) into
the FORMAL gold_evidence_set_v1 artifact -- the real consumer the earlier
`finalize_gold_evidence_set_v1.py` never was (that script rebuilds a DRAFT
from hardcoded data and can only ever collision-refuse against an
already-frozen draft; it does not read the worksheet at all).

Does NOT re-parse any PDF and does NOT re-derive original citations from
the worksheet's own display columns -- those are for human readability
only. The single source of truth for `original_supporting/contradicting_*`
is the frozen GoldSet's own `criterion_annotations`, always. The
worksheet's own `original_supporting_passages`/`original_contradicting_passages`
display columns are still cross-checked against that authority (see
`_verify_worksheet_matches_gold_set_citations`) so a hand-edited or
stale worksheet row is rejected rather than silently trusted.

--------------------------------------------------------------------------
DETERMINISTIC RULES applied when building each GoldEvidenceExcerpt (the
worksheet has no per-excerpt stance/evidence_role/extraction_method/
supported_inference columns -- these are the "narrowly documented rules"
substituting for them, verified against the real worksheet content below,
not assumed):

RULE 1 (stance + expansion eligibility): the real GoldSet's 60 keys break
down as 57 support-only, 1 contradiction-only
((10_3846_ijspm_2010_22, I1_review_type), gold_status="violated"), 1 mixed
(both supporting AND contradicting present:
(10_5194_nhess-12-2923-2012, I1_review_type), gold_status="unclear"), and
1 empty (evidence_absent, the attested package). Verified directly: NONE
of those 3 non-support-only rows carries any expanded evidence in this
worksheet. Expanded evidence is therefore permitted ONLY on rows with
`has_original_contradicting=False` -- support-only rows -- where
stance="supports" for the expansion is unambiguous. `_build_entry`
enforces this (not merely documents it): it raises if a future worksheet
ever pairs expanded evidence with ANY contradicting original passage
(contradiction-only OR mixed), rather than guessing which stance the new
evidence should carry.

RULE 2 (evidence_role for RETAINED-ORIGINAL excerpts): reuses
`ft_screening.gold_evidence_set._looks_non_substantive` directly (the
exact same heading/caption heuristic the worksheet itself was screened
and tiered with -- not reimplemented, to avoid drifting from it), and
its SECOND return value (the reason string) to distinguish two different
non-substantive shapes rather than collapsing them into one role:
  - reason="figure/table caption prefix" -> "visual_evidence_locator"
    (points at NON-TEXTUAL evidence -- a figure/table image -- per the
    module's own evidence_role definition; this is NOT "an organized
    discussion focus" claim a structural_heading makes). Confirmed
    against the real data: exactly 2 keys hit this --
    (10_1007_978-3-031-54118-6_8, I2_drr_hazard_framing) ["Fig. 1. Research
    protocol adopted for the literature review."] and
    (10_3390_w16111571, I1_review_type) ["Figure 2. Methodology for the
    review following the PRISMA concept."] -- both were ALSO classified
    visual_evidence_locator by the earlier hand-reviewed
    finalize_gold_evidence_set_v1.py FLAGGED dict; this rule reproduces
    that classification instead of regressing it to structural_heading.
  - any other non-substantive reason (numbered-heading pattern, empty,
    short-with-no-terminal-punctuation) -> "structural_heading".
  - substantive text -> "verbatim_text".

RULE 3 (evidence_role for EXPANDED excerpts): always "verbatim_text" --
every expanded excerpt in this worksheet is real prose (never a bare
heading/caption; that's the whole point of expanding). Not merely
asserted: `GoldEvidenceExcerpt`'s own model validator independently
re-checks substantiveness at construction time and raises loudly if this
assumption is ever wrong for some future row, rather than silently
accepting a heading-shaped "expansion".

RULE 4 (extraction_method for EXPANDED excerpts): "human_transcribed" for
the one row explicitly documented (in its own `review_notes`) as a
manual transcription of TEXT VISIBLE INSIDE A FIGURE IMAGE that Docling's
OCR failed to extract -- (10_1007_978-3-031-54118-6_8, I2_drr_hazard_framing),
see `_FIGURE_TRANSCRIPTION_KEYS` below (an explicit, hardcoded set with
the source review_notes quoted in a comment -- never auto-detected by
sniffing text, which would itself be "inventing" the classification).
Every other expanded excerpt: "expanded_from_pdf_section" (ordinary
additional body text read from the PDF, not a figure).

RULE 5 (supported_inference): the row's own `review_notes` text, verbatim
-- applied identically to EVERY excerpt object built from that row
(retained-original and expanded alike). Never fabricated: it is always
the human reviewer's own recorded words for that row, which is exactly
what `supported_inference` is defined to mean (see gold_evidence_set.py's
own docstring). `verify_ready_for_formal_freeze` already requires
non-empty `review_notes` on every row; `_build_entry` treats a blank
`review_notes` as a hard error too (belt-and-braces, not merely deferred
to the later gate).

RULE 6 (body_excerpt): always None for every excerpt this script builds
-- `heading_or_quotation` alone carries the full excerpt text; the
worksheet's two-column (heading vs body) split from the OLD
finalize_gold_evidence_set_v1.py FLAGGED dict has no equivalent in this
worksheet's schema, so nothing is invented to fill it.

ADDITIONAL HARDENING:
  - Entries are emitted in the frozen GoldSet's own `criterion_annotations`
    ORDER, never worksheet CSV row order -- merely re-sorting the
    worksheet (e.g. by priority_tier, as the worksheet builder itself
    does) must never change `GoldEvidenceSet.content_hash()`.
  - `reviewed_at` must be non-blank on every row (a hard error, not a
    silently-accepted `None`) -- this worksheet's own convention records
    an actual review date on every completed row, so a blank one is a
    data-entry gap, not a valid "unset" state.
  - Every expanded-excerpt page must be `<= ann.document_page_count` -- a
    page number cannot exceed the ORIGINAL annotation's own recorded
    document length (the same bound `GoldCriterionAnnotation` itself
    enforces on original pages).
  - A companion, SELF-HASHING provenance manifest
    (`gold_evidence_set_v1_promotion_manifest.json`, a real frozen
    envelope via `freeze_frozen_model`/`load_frozen_model`, not a plain
    `write_text()` dump) is published alongside the frozen artifact,
    binding it to the exact `worksheet_sha256` and `draft.content_hash()`
    this promotion ran against -- `GoldEvidenceSet` itself only carries
    `gold_set_hash`, so without this companion file the worksheet/draft
    lineage a freeze was produced from would only ever have been printed
    to a terminal, never retained. `publish_evidence_set_and_manifest()`
    publishes the pair RECOVERABLY: both target paths are preflighted
    before either is written, so a process killed between the two writes
    (evidence artifact frozen, manifest not yet written) leaves a state a
    retry resumes from correctly -- it reloads and hash-verifies the
    existing evidence artifact rather than re-freezing it, then writes
    only the still-missing manifest. Any state where the two disagree
    (manifest present without its evidence artifact, or either file's
    hash not matching what the CURRENT run would produce) is treated as
    an inconsistency to raise on, never silently overwritten.
--------------------------------------------------------------------------

Usage:
    python promote_gold_evidence_set_from_worksheet.py             # dry-run / preflight, no freeze
    python promote_gold_evidence_set_from_worksheet.py --freeze    # freezes gold_evidence_set_v1.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

sys.path.insert(0, r"C:/projects/FT_SCREENING/src")

from ft_screening.gold import freeze_frozen_model, load_frozen_gold_set, load_frozen_model  # noqa: E402
from ft_screening.gold import GoldSet  # noqa: E402
from ft_screening.gold_evidence_set import (  # noqa: E402
    GoldEvidenceEntry,
    GoldEvidenceExcerpt,
    GoldEvidenceSet,
    now_iso,
    verify_gold_evidence_set_matches_gold_set,
    verify_ready_for_formal_freeze,
)
from ft_screening.gold_evidence_set import _looks_non_substantive  # noqa: E402
from ft_screening.hashing import canonical_json_bytes  # noqa: E402


class GoldEvidenceSetPromotionManifest(BaseModel):
    """Companion provenance record for a frozen `gold_evidence_set_v1` --
    binds it to the exact worksheet/draft/GoldSet it was promoted from.
    Uses the SAME self-hashing frozen-envelope mechanism
    (`freeze_frozen_model`/`load_frozen_model`) every other artifact in
    this codebase uses, rather than a plain, unverified `write_text()`
    JSON dump."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version_label: str = Field(min_length=1)
    evidence_set_content_hash: str = Field(min_length=64, max_length=64)
    gold_set_hash: str = Field(min_length=64, max_length=64)
    draft_content_hash: str = Field(min_length=64, max_length=64)
    draft_version_label: str = Field(min_length=1)
    worksheet_path: str = Field(min_length=1)
    worksheet_sha256: str = Field(min_length=64, max_length=64)
    promoted_at: str = Field(min_length=1)

    def content_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"promoted_at"})
        return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

FROZEN_GOLD_SET_PATH = Path(r"C:/projects/FT_SCREENING/gold_set/frozen/gold_set_v1.json")
EXPECTED_GOLD_SET_HASH = "6fdfaebfd84e7ed33101d1a9b9ba4960367ab6985f1a52e3a13cabc2e396280a"
DRAFT_PATH = Path(r"C:/projects/FT_SCREENING/outputs/gold_evidence_set_v1/gold_evidence_set_claude_draft_v2.json")
WORKSHEET_PATH = Path(r"C:/projects/FT_SCREENING/outputs/gold_evidence_set_v1/gold_evidence_review_worksheet_v1.csv")
OUTPUT_DIR = Path(r"C:/projects/FT_SCREENING/outputs/gold_evidence_set_v1")
VERSION_LABEL = "gold_evidence_set_v1"  # the FORMAL name -- this script is the only one permitted to produce it.

# Rule 4's explicit set -- see module docstring. Source review_notes (verbatim,
# as recorded in the worksheet): "Docling should have saved the picture of the
# figure and should have extracted the text. Part of the text in this figure
# is: ...". This is the only key in the entire 60-row worksheet with
# figure/OCR-transcribed (rather than ordinary body-text) expanded evidence.
_FIGURE_TRANSCRIPTION_KEYS: frozenset[tuple[str, str]] = frozenset({
    ("10_1007_978-3-031-54118-6_8", "I2_drr_hazard_framing"),
})

_REQUIRED_WORKSHEET_COLUMNS = [
    "document_id", "criterion_id", "original_supporting_passages", "original_contradicting_passages",
    "reviewer_id", "reviewed_at", "human_sufficiency", "human_attestation",
    "expanded_excerpt_text", "expanded_excerpt_page", "review_notes",
]


def _retained_citation_evidence_role(passage: str) -> str:
    """RULE 2. `_looks_non_substantive` returns (bool, reason) -- the
    reason string is what lets a figure/table CAPTION (pointing at
    non-textual evidence) be told apart from an ordinary numbered
    SECTION HEADING (structural evidence of an organized discussion
    focus); collapsing both into one role was the exact regression this
    function exists to prevent."""
    non_substantive, reason = _looks_non_substantive(passage)
    if not non_substantive:
        return "verbatim_text"
    if reason == "figure/table caption prefix":
        return "visual_evidence_locator"
    return "structural_heading"


def _parse_strict_bool(value: str, *, key: tuple[str, str], field_name: str) -> bool:
    normalized = value.strip().upper()
    if normalized == "TRUE":
        return True
    if normalized == "FALSE":
        return False
    raise ValueError(f"{key}: {field_name} must be exactly TRUE or FALSE, got {value!r}")


def _parse_expanded_arrays(text_field: str, page_field: str, *, key: tuple[str, str]) -> list[tuple[str, int]]:
    """Strict JSON parse (no strict=False leniency -- the worksheet has
    already been normalized to real, strictly-valid JSON; a non-strict
    parse succeeding here would mask a regression). Requires equal,
    nonzero lengths whenever expanded evidence is present at all."""
    text_raw, page_raw = text_field.strip(), page_field.strip()
    if not text_raw and not page_raw:
        return []
    if bool(text_raw) != bool(page_raw):
        raise ValueError(f"{key}: expanded_excerpt_text and expanded_excerpt_page must be both-empty or both-set")
    try:
        texts = json.loads(text_raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{key}: expanded_excerpt_text is not valid JSON: {exc}") from exc
    try:
        pages = json.loads(page_raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{key}: expanded_excerpt_page is not valid JSON: {exc}") from exc
    if not (isinstance(texts, list) and texts and all(isinstance(t, str) and t.strip() for t in texts)):
        raise ValueError(f"{key}: expanded_excerpt_text must be a nonempty JSON array of nonblank strings, got {texts!r}")
    if not (isinstance(pages, list) and pages and all(isinstance(p, int) and not isinstance(p, bool) for p in pages)):
        raise ValueError(f"{key}: expanded_excerpt_page must be a nonempty JSON array of integers, got {pages!r}")
    if len(texts) != len(pages):
        raise ValueError(f"{key}: expanded_excerpt_text has {len(texts)} element(s) but expanded_excerpt_page has {len(pages)} -- lengths must match")
    return list(zip(texts, pages))


def _verify_worksheet_matches_gold_set_citations(row: dict[str, str], ann, *, key: tuple[str, str]) -> None:
    """Rejects a worksheet row whose displayed original citation has
    drifted from the frozen GoldSet's own authoritative annotation for
    this key (hand-edited or built from a stale source) -- the display
    columns are never trusted as a data source, only as a tamper check."""
    expected_supporting = " ||| ".join(ann.supporting_passages)
    expected_contradicting = " ||| ".join(ann.contradicting_passages)
    if row["original_supporting_passages"] != expected_supporting:
        raise ValueError(
            f"{key}: worksheet's original_supporting_passages does not match the frozen GoldSet's own "
            f"annotation -- worksheet may be stale or hand-edited (expected={expected_supporting!r}, "
            f"worksheet={row['original_supporting_passages']!r})"
        )
    if row["original_contradicting_passages"] != expected_contradicting:
        raise ValueError(
            f"{key}: worksheet's original_contradicting_passages does not match the frozen GoldSet's own "
            f"annotation -- worksheet may be stale or hand-edited (expected={expected_contradicting!r}, "
            f"worksheet={row['original_contradicting_passages']!r})"
        )


def _build_entry(row: dict[str, str], ann, *, gold_set: GoldSet) -> GoldEvidenceEntry:
    key = (row["document_id"], row["criterion_id"])
    _verify_worksheet_matches_gold_set_citations(row, ann, key=key)

    reviewer_id = row["reviewer_id"].strip()
    if not reviewer_id:
        raise ValueError(f"{key}: no reviewer_id recorded")
    review_notes = row["review_notes"].strip()
    if not review_notes:
        raise ValueError(f"{key}: review_notes is empty -- every reviewed row must carry the reviewer's own rationale")
    reviewed_at = row["reviewed_at"].strip()
    if not reviewed_at:
        raise ValueError(f"{key}: reviewed_at is blank -- every reviewed row must record a review date")
    human_attestation = row["human_attestation"].strip() or None
    if human_attestation and human_attestation != "no_qualifying_evidence_found_after_full_document_review":
        raise ValueError(f"{key}: human_attestation has an unrecognized value {human_attestation!r}")

    sufficient = _parse_strict_bool(row["human_sufficiency"], key=key, field_name="human_sufficiency")

    has_original_supporting = bool(ann.supporting_pages)
    has_original_contradicting = bool(ann.contradicting_pages)
    expanded = _parse_expanded_arrays(row["expanded_excerpt_text"], row["expanded_excerpt_page"], key=key)

    # RULE 1: tightened -- expanded evidence permitted ONLY on support-only
    # rows (has_original_contradicting=False), covering contradiction-only
    # AND mixed rows alike; not merely assumed.
    if expanded and has_original_contradicting:
        raise ValueError(
            f"{key}: has expanded evidence on a row with a contradicting original passage present (contradiction-only "
            "or mixed) -- this script's stance rule (RULE 1) only covers support-only rows; refusing to guess a "
            "stance for this case"
        )

    excerpts: list[GoldEvidenceExcerpt] = []
    for page, passage in zip(ann.supporting_pages, ann.supporting_passages):
        role = _retained_citation_evidence_role(passage)
        excerpts.append(GoldEvidenceExcerpt(
            stance="supports", page=page, evidence_role=role, heading_or_quotation=passage,
            supported_inference=review_notes, extraction_method="retained_original_citation",
        ))
    for page, passage in zip(ann.contradicting_pages, ann.contradicting_passages):
        role = _retained_citation_evidence_role(passage)
        excerpts.append(GoldEvidenceExcerpt(
            stance="contradicts", page=page, evidence_role=role, heading_or_quotation=passage,
            supported_inference=review_notes, extraction_method="retained_original_citation",
        ))

    if expanded:
        source_sha256 = gold_set.document_pdf_sha256.get(row["document_id"])
        if source_sha256 is None:
            raise ValueError(f"{key}: document_id has no entry in gold_set.document_pdf_sha256 -- cannot source-bind expanded evidence")
        extraction_method = "human_transcribed" if key in _FIGURE_TRANSCRIPTION_KEYS else "expanded_from_pdf_section"
        for text, page in expanded:
            if page > ann.document_page_count:
                raise ValueError(
                    f"{key}: expanded excerpt page={page} exceeds ann.document_page_count={ann.document_page_count}"
                )
            excerpts.append(GoldEvidenceExcerpt(
                stance="supports", page=page, evidence_role="verbatim_text", heading_or_quotation=text,
                supported_inference=review_notes, extraction_method=extraction_method,
                source_document_sha256=source_sha256,
            ))

    if not excerpts and human_attestation is None:
        raise ValueError(f"{key}: no excerpts recorded and no human_attestation -- empty package must be explicitly attested")

    return GoldEvidenceEntry(
        document_id=row["document_id"], criterion_id=row["criterion_id"],
        original_supporting_pages=tuple(ann.supporting_pages), original_supporting_passages=tuple(ann.supporting_passages),
        original_contradicting_pages=tuple(ann.contradicting_pages), original_contradicting_passages=tuple(ann.contradicting_passages),
        self_contained_excerpts=tuple(excerpts),
        sufficient_for_independent_adjudication=sufficient, sufficiency_assessed_by=f"human_review:{reviewer_id}",
        sufficiency_rationale=review_notes,
        reviewer_id=reviewer_id, reviewed_at=reviewed_at, review_notes=review_notes,
        human_attestation=human_attestation,
    )


def load_worksheet_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not set(_REQUIRED_WORKSHEET_COLUMNS).issubset(reader.fieldnames):
            missing = set(_REQUIRED_WORKSHEET_COLUMNS) - set(reader.fieldnames or [])
            raise ValueError(f"{csv_path} is missing required column(s): {sorted(missing)}")
        return list(reader)


def build_evidence_set_from_worksheet(gold_set: GoldSet, worksheet_rows: list[dict[str, str]]) -> GoldEvidenceSet:
    expected_keys = {(a.document_id, a.criterion_id) for a in gold_set.criterion_annotations}

    seen: set[tuple[str, str]] = set()
    for row in worksheet_rows:
        key = (row["document_id"], row["criterion_id"])
        if key in seen:
            raise ValueError(f"worksheet has a duplicate row for {key}")
        seen.add(key)
    if seen != expected_keys:
        missing = expected_keys - seen
        extra = seen - expected_keys
        raise ValueError(
            f"worksheet does not cover exactly the GoldSet's 60 keys -- missing={sorted(missing)}, extra={sorted(extra)}"
        )

    annotations_by_key = {(a.document_id, a.criterion_id): a for a in gold_set.criterion_annotations}
    rows_by_key = {(row["document_id"], row["criterion_id"]): row for row in worksheet_rows}
    # Emitted in the frozen GoldSet's OWN annotation order (never worksheet
    # CSV row order) -- see "ADDITIONAL HARDENING" in the module docstring:
    # reordering the worksheet must never change content_hash().
    entries = [_build_entry(rows_by_key[key], ann, gold_set=gold_set) for key, ann in annotations_by_key.items()]

    return GoldEvidenceSet(gold_set_hash=gold_set.content_hash(), version_label=VERSION_LABEL, entries=tuple(entries), frozen_at=now_iso())


def freeze_evidence_set(evidence_set: GoldEvidenceSet, *, output_dir: Path, version_label: str = VERSION_LABEL) -> Path:
    """Low-level, unconditionally-exclusive single write -- raises if the
    target already exists, full stop. Used directly by
    `publish_evidence_set_and_manifest` only in the branch where the
    evidence artifact is confirmed NOT to exist yet; kept as its own
    tested primitive rather than inlined."""
    result_path = output_dir / f"{version_label}.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite an existing frozen artifact: {result_path}")
    freeze_frozen_model(evidence_set, output_dir=output_dir, version_label=version_label, artifact_type="GoldEvidenceSet")
    return result_path


def _build_manifest(
    *, version_label: str, evidence_set: GoldEvidenceSet, gold_set: GoldSet, draft: GoldEvidenceSet,
    worksheet_path: Path, worksheet_sha256: str,
) -> GoldEvidenceSetPromotionManifest:
    return GoldEvidenceSetPromotionManifest(
        version_label=version_label, evidence_set_content_hash=evidence_set.content_hash(),
        gold_set_hash=gold_set.content_hash(), draft_content_hash=draft.content_hash(),
        draft_version_label=draft.version_label, worksheet_path=str(worksheet_path), worksheet_sha256=worksheet_sha256,
        promoted_at=now_iso(),
    )


def publish_evidence_set_and_manifest(
    evidence_set: GoldEvidenceSet, *, output_dir: Path, gold_set: GoldSet, draft: GoldEvidenceSet,
    worksheet_path: Path, worksheet_sha256: str, version_label: str = VERSION_LABEL,
) -> tuple[Path, Path, str]:
    """Recoverable paired publication of the frozen evidence artifact and
    its companion promotion manifest -- BOTH paths are preflighted before
    EITHER is written, so a process killed between the two writes leaves
    a state a retry can cleanly resume from, never re-freezing the
    evidence artifact and never silently accepting a manifest that
    doesn't actually match what's on disk.

    Returns (evidence_path, manifest_path, status) where status is one of
    "wrote_both", "wrote_manifest_only" (recovery case), or
    "already_complete" (both existed and verified consistent -- nothing
    written).

    Four preflight states:
      - neither exists: freeze the evidence artifact, then the manifest.
      - evidence exists, manifest doesn't: RELOAD the existing evidence
        artifact and verify its content_hash equals
        evidence_set.content_hash() (this run's own value) before writing
        ONLY the missing manifest -- never re-freezes the evidence file.
      - both exist: reload and verify both hashes match this run's own
        values, report already-complete, write nothing.
      - inconsistent state (manifest exists without the evidence
        artifact, OR either existing file's hash disagrees with this
        run's own value): raises, refusing to overwrite or guess.
    """
    evidence_path = output_dir / f"{version_label}.json"
    manifest_version_label = f"{version_label}_promotion_manifest"
    manifest_path = output_dir / f"{manifest_version_label}.json"

    evidence_exists = evidence_path.exists()
    manifest_exists = manifest_path.exists()

    if manifest_exists and not evidence_exists:
        raise ValueError(
            f"inconsistent state: promotion manifest {manifest_path} exists but the evidence artifact "
            f"{evidence_path} does not -- refusing to proceed without manual investigation"
        )

    if evidence_exists:
        existing_evidence = load_frozen_model(GoldEvidenceSet, evidence_path, expected_artifact_type="GoldEvidenceSet")
        if existing_evidence.content_hash() != evidence_set.content_hash():
            raise ValueError(
                f"inconsistent state: the frozen evidence artifact at {evidence_path} has content_hash="
                f"{existing_evidence.content_hash()!r}, which does not match this run's evidence_set.content_hash()="
                f"{evidence_set.content_hash()!r} -- refusing to overwrite"
            )

    if manifest_exists:
        existing_manifest = load_frozen_model(GoldEvidenceSetPromotionManifest, manifest_path, expected_artifact_type="GoldEvidenceSetPromotionManifest")
        if existing_manifest.evidence_set_content_hash != evidence_set.content_hash():
            raise ValueError(
                f"inconsistent state: the existing promotion manifest at {manifest_path} records "
                f"evidence_set_content_hash={existing_manifest.evidence_set_content_hash!r}, which does not match "
                f"this run's evidence_set.content_hash()={evidence_set.content_hash()!r} -- refusing to overwrite"
            )

    if evidence_exists and manifest_exists:
        return evidence_path, manifest_path, "already_complete"

    status = "wrote_manifest_only" if evidence_exists else "wrote_both"

    if not evidence_exists:
        freeze_frozen_model(evidence_set, output_dir=output_dir, version_label=version_label, artifact_type="GoldEvidenceSet")

    manifest = _build_manifest(
        version_label=version_label, evidence_set=evidence_set, gold_set=gold_set, draft=draft,
        worksheet_path=worksheet_path, worksheet_sha256=worksheet_sha256,
    )
    freeze_frozen_model(manifest, output_dir=output_dir, version_label=manifest_version_label, artifact_type="GoldEvidenceSetPromotionManifest")

    return evidence_path, manifest_path, status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()

    gold_set = load_frozen_gold_set(FROZEN_GOLD_SET_PATH, expected_hash=EXPECTED_GOLD_SET_HASH)
    draft = load_frozen_model(GoldEvidenceSet, DRAFT_PATH, expected_artifact_type="GoldEvidenceSet")
    if draft.gold_set_hash != gold_set.content_hash():
        raise ValueError("draft.gold_set_hash does not match the current gold_set_v1.json -- refusing to promote against a stale binding")
    draft_keys = {(e.document_id, e.criterion_id) for e in draft.entries}
    gold_set_keys = {(a.document_id, a.criterion_id) for a in gold_set.criterion_annotations}
    if draft_keys != gold_set_keys:
        raise ValueError(f"draft does not cover exactly the GoldSet's keys -- missing={sorted(gold_set_keys - draft_keys)}, extra={sorted(draft_keys - gold_set_keys)}")

    worksheet_bytes = WORKSHEET_PATH.read_bytes()
    worksheet_sha256 = hashlib.sha256(worksheet_bytes).hexdigest()
    worksheet_rows = load_worksheet_rows(WORKSHEET_PATH)

    evidence_set = build_evidence_set_from_worksheet(gold_set, worksheet_rows)
    verify_gold_evidence_set_matches_gold_set(evidence_set, gold_set)
    verify_ready_for_formal_freeze(evidence_set)  # raises with the full problem list if anything is missing

    n_expanded = sum(1 for e in evidence_set.entries if any(x.extraction_method != "retained_original_citation" for x in e.self_contained_excerpts))
    n_figure_transcribed = sum(1 for e in evidence_set.entries if any(x.extraction_method == "human_transcribed" for x in e.self_contained_excerpts))
    n_empty_package = sum(1 for e in evidence_set.entries if not e.self_contained_excerpts)
    n_retained_only = len(evidence_set.entries) - n_expanded - n_empty_package
    n_sufficient = sum(1 for e in evidence_set.entries if e.sufficient_for_independent_adjudication is True)
    n_insufficient = sum(1 for e in evidence_set.entries if e.sufficient_for_independent_adjudication is False)

    print(f"worksheet_path={WORKSHEET_PATH}")
    print(f"worksheet_sha256={worksheet_sha256}")
    print(f"gold_set.content_hash()={gold_set.content_hash()}")
    print(f"draft.content_hash()={draft.content_hash()}")
    print(f"Built {len(evidence_set.entries)} entries: {n_retained_only} retained-only, {n_expanded} with expanded evidence "
          f"(of which {n_figure_transcribed} figure/OCR human_transcribed), {n_empty_package} empty-package (attested).")
    print(f"sufficient=True: {n_sufficient}, sufficient=False: {n_insufficient}")
    print("Readiness gate (verify_ready_for_formal_freeze): PASSED")
    print(f"proposed_formal_content_hash={evidence_set.content_hash()}")

    if not args.freeze:
        print("\nDRY RUN -- not frozen. Pass --freeze to persist as gold_evidence_set_v1.")
        return

    result_path, manifest_path, status = publish_evidence_set_and_manifest(
        evidence_set, output_dir=OUTPUT_DIR, gold_set=gold_set, draft=draft,
        worksheet_path=WORKSHEET_PATH, worksheet_sha256=worksheet_sha256,
    )
    print(f"publish status: {status}")
    print(f"evidence artifact: {result_path}")
    print(f"promotion manifest: {manifest_path}")


if __name__ == "__main__":
    main()
