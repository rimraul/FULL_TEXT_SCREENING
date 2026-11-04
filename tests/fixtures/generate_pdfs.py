"""Generates the deterministic synthetic PDF fixtures committed under
tests/fixtures/pdfs/ -- run this script to (re)produce them, then check
`python generate_pdfs.py --verify` against tests/fixtures/MANIFEST.json
before committing.

Deliberately synthetic, not real papers (per project decision, 2026-08-13):
self-contained, no licensing question, portable to any machine/CI. Real
PDF quirks (genuine multi-column layout, real scan noise, unusual fonts)
are NOT fully represented here -- see test_document_parser.py's
`FT_SCREENING_REAL_PDF_DIR`-gated tests for optional, skipped-by-default,
real-corpus spot checks that fill that gap when a developer has a corpus
available locally.

Determinism: PyMuPDF's default document metadata includes a creation/mod
timestamp ("now"), which would make every regeneration produce different
bytes. Every fixture here has its metadata explicitly pinned to fixed
values before saving, specifically so the MANIFEST's sha256 hashes stay
stable across regenerations (same PyMuPDF version). The hash in the
manifest is the source of truth for "the committed file is intact and
unmodified" -- it is NOT a promise that a different PyMuPDF version will
reproduce identical bytes; regenerate and update the manifest if the
library version changes.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import fitz  # PyMuPDF

FIXTURES_DIR = Path(__file__).parent / "pdfs"
MANIFEST_PATH = Path(__file__).parent / "MANIFEST.json"

_FIXED_METADATA = {
    "title": "", "author": "", "subject": "", "keywords": "",
    "creator": "ft_screening test fixture generator", "producer": "PyMuPDF",
    "creationDate": "D:20250101000000Z", "modDate": "D:20250101000000Z",
    "trapped": "",
}


def _save_deterministic(doc: fitz.Document, path: Path) -> None:
    doc.set_metadata(_FIXED_METADATA)
    # no_new_id=1: PyMuPDF otherwise writes a fresh, effectively-random
    # /ID pair into the trailer on every save() call (the PDF spec's
    # per-revision file-identifier convention) -- with it left at the
    # default, two saves of byte-identical CONTENT still produce
    # different files, breaking the whole point of a committed,
    # hash-pinned fixture. Confirmed empirically: this was the only
    # source of non-determinism found (a byte-level diff of two
    # generation runs isolated it to exactly the trailer's /ID entry).
    doc.save(str(path), garbage=4, deflate=True, clean=True, no_new_id=1)
    doc.close()


def build_simple_two_page() -> None:
    """Baseline: 2 pages, 2 distinct sections, plain paragraph text, no
    tables/images. Exercises: basic parse success, section tracking via
    SECTION_HEADER items, page-to-character-offset mapping, deterministic
    chunk IDs (parse twice, compare)."""
    doc = fitz.open()
    page1 = doc.new_page(width=595, height=842)  # A4
    page1.insert_text((72, 72), "Introduction", fontsize=16)
    page1.insert_text(
        (72, 110),
        "This is the introduction section of a synthetic fixture document.\n"
        "It exists to exercise basic parsing, chunking, and page-to-\n"
        "character-offset mapping without depending on any real paper.",
        fontsize=11,
    )
    page2 = doc.new_page(width=595, height=842)
    page2.insert_text((72, 72), "Conclusion", fontsize=16)
    page2.insert_text(
        (72, 110),
        "This is the conclusion section, on the second page.\n"
        "A correct parser must attribute this text to page 2,\n"
        "not page 1, and to the Conclusion section, not the Introduction.",
        fontsize=11,
    )
    _save_deterministic(doc, FIXTURES_DIR / "simple_two_page.pdf")


def build_multi_section() -> None:
    """4 pages, 4 sections, more text volume -- exercises chunk ordering
    and continuity across page boundaries, and is truncated (below) to
    produce the corrupt-document fixture."""
    doc = fitz.open()
    sections = [
        ("Background", "Prior work in this synthetic field established several findings. " * 6),
        ("Methods", "Participants were recruited using a synthetic sampling procedure. " * 6),
        ("Results", "The synthetic analysis produced the following illustrative results. " * 6),
        ("Discussion", "These synthetic results should be interpreted with appropriate caution. " * 6),
    ]
    for heading, body in sections:
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 72), heading, fontsize=16)
        page.insert_textbox(fitz.Rect(72, 110, 523, 750), body, fontsize=11)
    _save_deterministic(doc, FIXTURES_DIR / "multi_section.pdf")


def build_with_table() -> None:
    """A page with an actual drawn PDF table (grid lines + cell text) --
    exercises Docling's table-structure recognition and
    TableItem.export_to_markdown()."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 72), "Methods", fontsize=16)
    page.insert_text((72, 100), "Table 1. Participant characteristics.", fontsize=11)

    rows = [["Group", "N", "Mean age"], ["Intervention", "42", "35.2"], ["Control", "40", "34.8"]]
    x0, y0 = 72, 130
    col_widths = [180, 90, 120]
    row_height = 24
    n_rows, n_cols = len(rows), len(rows[0])
    for r in range(n_rows + 1):
        y = y0 + r * row_height
        page.draw_line((x0, y), (x0 + sum(col_widths), y))
    x = x0
    for c in range(n_cols + 1):
        page.draw_line((x, y0), (x, y0 + n_rows * row_height))
        if c < n_cols:
            x += col_widths[c]
    x = x0
    for c, width in enumerate(col_widths):
        for r, row in enumerate(rows):
            page.insert_text((x + 6, y0 + r * row_height + 16), row[c], fontsize=10)
        x += width
    _save_deterministic(doc, FIXTURES_DIR / "with_table.pdf")


def build_two_column() -> None:
    """A crude two-column layout -- left column text is a fully separate
    block from right column text, positioned side by side. Exercises
    whether Docling's layout-aware reading order keeps the left column's
    content together and before the right column's, rather than
    interleaving by raw vertical position (which a naive top-to-bottom,
    ignore-x-position extractor would do)."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 72), "Two-Column Layout Test", fontsize=14)
    left_text = "LEFT COLUMN START. " + ("Left column body text continues here. " * 10) + "LEFT COLUMN END."
    right_text = "RIGHT COLUMN START. " + ("Right column body text continues here. " * 10) + "RIGHT COLUMN END."
    # A wide gutter alone was not enough for Docling's layout model to
    # recognize two genuine columns in hands-on testing (it merged both
    # into a single interleaved reading-order flow) -- a drawn vertical
    # rule between the columns is a much stronger, standard visual cue.
    page.draw_line((295, 100), (295, 760), width=0.75)
    page.insert_textbox(fitz.Rect(72, 110, 275, 750), left_text, fontsize=10)
    page.insert_textbox(fitz.Rect(315, 110, 523, 750), right_text, fontsize=10)
    _save_deterministic(doc, FIXTURES_DIR / "two_column.pdf")


def build_scanned_no_text_layer() -> None:
    """A page that is PURELY a rasterized image -- no text objects at
    all, so ordinary text extraction finds nothing and Docling's OCR
    fallback is the only way to recover the content. Exercises
    do_ocr=True and DocumentCoverage.ocr_used detection."""
    render_doc = fitz.open()
    render_page = render_doc.new_page(width=400, height=200)
    render_page.insert_text((20, 100), "SCANNED PAGE OCR TEST CONTENT", fontsize=18)
    pixmap = render_page.get_pixmap(dpi=150)
    image_bytes = pixmap.tobytes("png")
    render_doc.close()

    doc = fitz.open()
    page = doc.new_page(width=400, height=200)
    page.insert_image(page.rect, stream=image_bytes)
    _save_deterministic(doc, FIXTURES_DIR / "scanned_no_text_layer.pdf")


def build_with_appendix() -> None:
    """codex_code_review_1578af2.md finding 4: the acceptance suite's
    prior "appendix" test just renamed an ordinary trailing Discussion
    section -- this is a GENUINE distinct appendix/supplement section
    (its own heading, its own content, structurally separate from the
    main body), the real acceptance-contract case plan_v3 asks for."""
    doc = fitz.open()
    page1 = doc.new_page(width=595, height=842)
    page1.insert_text((72, 72), "Results", fontsize=16)
    page1.insert_textbox(
        fitz.Rect(72, 110, 523, 750),
        "The synthetic study found illustrative results in the main body of the paper. " * 6,
        fontsize=11,
    )
    page2 = doc.new_page(width=595, height=842)
    page2.insert_text((72, 72), "Appendix A: Supplementary Materials", fontsize=16)
    page2.insert_textbox(
        fitz.Rect(72, 110, 523, 750),
        "SUPPLEMENTARY CONTENT MARKER. This appendix contains additional synthetic data tables "
        "and methodological detail not included in the main body. " * 4,
        fontsize=11,
    )
    _save_deterministic(doc, FIXTURES_DIR / "with_appendix.pdf")


def build_truncated() -> None:
    """multi_section.pdf's bytes cut at a fixed offset -- deterministic
    corruption (not a real paper truncated arbitrarily). Exercises
    DoclingDocumentParser.parse()'s corrupt-document handling: Docling's
    convert() raises for this kind of input rather than returning a
    partial result (confirmed via hands-on testing), so parse() must
    catch it and produce a coverage.status == "failed" ParsedDocument."""
    source = FIXTURES_DIR / "multi_section.pdf"
    data = source.read_bytes()
    truncated = data[: len(data) // 3]
    (FIXTURES_DIR / "truncated.pdf").write_bytes(truncated)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


MANIFEST_ENTRIES = {
    "simple_two_page.pdf": {
        "purpose": "Baseline 2-page, 2-section plain-text document.",
        "expected": {"total_pages": 2, "status": "complete", "min_chunks": 2, "sections_include": ["Introduction", "Conclusion"]},
    },
    "multi_section.pdf": {
        "purpose": "4-page, 4-section document with more text volume; also the source for truncated.pdf.",
        "expected": {"total_pages": 4, "status": "complete", "min_chunks": 4, "sections_include": ["Background", "Methods", "Results", "Discussion"]},
    },
    "with_table.pdf": {
        "purpose": "A page with a drawn grid table -- exercises table-structure recognition and markdown export.",
        "expected": {"total_pages": 1, "status": "complete", "min_chunks": 1},
    },
    "two_column.pdf": {
        "purpose": "Crude two-column layout -- exercises layout-aware (not naive top-to-bottom) reading order.",
        "expected": {"total_pages": 1, "status": "complete", "min_chunks": 1},
    },
    "scanned_no_text_layer.pdf": {
        "purpose": "Image-only page, no text layer -- forces the OCR fallback path.",
        "expected": {"total_pages": 1, "requires_ocr": True},
    },
    "with_appendix.pdf": {
        "purpose": "Results section plus a genuinely distinct Appendix/Supplementary Materials section (its own heading and content, not a renamed Discussion).",
        "expected": {"total_pages": 2, "status": "complete", "min_chunks": 2, "sections_include": ["Results", "Appendix A: Supplementary Materials"]},
    },
    "truncated.pdf": {
        "purpose": "multi_section.pdf truncated to 1/3 of its bytes -- deterministic corrupt-document fixture.",
        "expected": {"status": "failed"},
    },
}


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    verify_only = "--verify" in sys.argv

    if not verify_only:
        build_simple_two_page()
        build_multi_section()
        build_with_table()
        build_two_column()
        build_scanned_no_text_layer()
        build_with_appendix()
        build_truncated()  # must run after build_multi_section()

    manifest = {"fixtures": {}}
    for name, entry in MANIFEST_ENTRIES.items():
        path = FIXTURES_DIR / name
        if not path.exists():
            print(f"MISSING fixture: {name}")
            sys.exit(1)
        manifest["fixtures"][name] = {**entry, "sha256": _sha256(path), "size_bytes": path.stat().st_size}

    if verify_only:
        existing = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        mismatches = []
        for name, entry in manifest["fixtures"].items():
            existing_entry = existing["fixtures"].get(name)
            if existing_entry is None or existing_entry["sha256"] != entry["sha256"]:
                mismatches.append(name)
        if mismatches:
            print("MANIFEST MISMATCH for:", mismatches)
            sys.exit(1)
        print("All fixture hashes match MANIFEST.json.")
        return

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(manifest['fixtures'])} fixtures to {FIXTURES_DIR} and {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
