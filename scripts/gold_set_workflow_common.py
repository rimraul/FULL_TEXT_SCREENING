"""Shared plumbing for the gold-set sealed-worksheet workflow scripts
(gold_set_seal_worksheet_pass.py, gold_set_generate_second_pass_worksheets.py,
gold_set_generate_adjudication_worksheet.py, gold_set_merge_annotation_sources.py).

codex_code_review_93bf42d.md finding 3: "the protocol says the scripts
are thin wrappers around validation in ft_screening.gold, but the reveal
and merge security decisions currently live only in permissive script
code." Seal verification is one function (`load_and_verify_seal`,
delegating to `ft_screening.gold.verify_worksheet_pass_seal` -- the
SAME function `freeze_gold_set()` uses internally, per
codex_code_review_73cf5d9.md finding 1), and set-safe publication is one
function (`publish_output_set_exclusive`/`write_generated_csv_set_idempotent`,
finding 4), imported everywhere needed instead of being reimplemented
(or skipped) per script.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, r"C:/projects/FT_SCREENING/src")

from ft_screening.gold import WorksheetPassSeal, load_frozen_worksheet_pass_seal, verify_worksheet_pass_seal  # noqa: E402


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_verify_seal(
    seal_path: Path,
    *,
    expected_pass_name: str,
    criterion_csv_path: Path,
    document_csv_path: Path,
    expected_protocol_version: str,
    expected_criteria_set_hash: str,
    expected_document_ids: set,
    expected_blind_manifest_hash: str,
    expected_sample_manifest_hash: str,
) -> WorksheetPassSeal:
    """Loads a `WorksheetPassSeal` envelope (verifying its own dual-hash
    envelope) and then delegates to `ft_screening.gold.verify_worksheet_pass_seal`
    for EVERY identity check (pass name, protocol version, criteria-set
    hash, exact document-ID set, blind/sample manifest hashes) plus a
    re-hash of the actual criterion/document CSVs. Checking file
    EXISTENCE alone is not verification (codex_code_review_93bf42d.md
    finding 3's exact repro: a file containing the literal text "not a
    valid seal" was enough to unlock the dual-review subset); checking
    only the pass name and CSV hashes is not enough either
    (codex_code_review_73cf5d9.md finding 1's exact repro: a seal for
    the wrong protocol/criteria set, or a second-pass seal covering MORE
    documents than the actual dual-review subset, both verified
    successfully before every identity field was checked)."""
    if not seal_path.exists():
        raise SystemExit(f"refusing to proceed: {seal_path} does not exist or is not a real seal")
    try:
        seal = load_frozen_worksheet_pass_seal(seal_path)
    except Exception as exc:  # noqa: BLE001 -- any parse/hash failure means "not a valid seal"
        raise SystemExit(f"refusing to proceed: {seal_path} is not a valid WorksheetPassSeal envelope ({exc})") from exc
    try:
        verify_worksheet_pass_seal(
            seal, expected_pass_name=expected_pass_name, expected_protocol_version=expected_protocol_version,
            expected_criteria_set_hash=expected_criteria_set_hash, expected_document_ids=expected_document_ids,
            expected_blind_manifest_hash=expected_blind_manifest_hash, expected_sample_manifest_hash=expected_sample_manifest_hash,
            criterion_csv_path=criterion_csv_path, document_csv_path=document_csv_path,
        )
    except ValueError as exc:
        raise SystemExit(f"refusing to proceed: {exc}") from exc
    return seal


def _try_exclusive_write(path: Path, data: bytes) -> bool:
    """Attempts to create `path` EXCLUSIVELY (the same temp-file-then-
    atomic-`os.link` pattern `gold.py`'s own frozen artifacts use,
    reused here per codex_code_review_73cf5d9.md finding 4's explicit
    request), rather than `exists()` followed by `os.replace()` -- the
    latter leaves a window where another process/editor can create or
    change the destination between the check and the write, and
    `os.replace()` would then silently overwrite it. Returns True if
    THIS call created the file. If the destination already exists, its
    content is checked against what was about to be written: identical
    content is a safe no-op (True idempotent rerun); different content
    means something else (a human, a concurrent run) put it there, and
    that is refused, never silently overwritten."""
    tmp_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    tmp_path.write_bytes(data)
    try:
        os.link(tmp_path, path)
        return True
    except FileExistsError:
        if path.read_bytes() != data:
            raise SystemExit(
                f"refusing to overwrite {path} -- it already exists with content different from what would "
                "be generated now (possibly created concurrently, or containing a human-entered answer). "
                "Resolve this manually before rerunning."
            )
        return False
    finally:
        tmp_path.unlink(missing_ok=True)


def safe_unlink_if_matches(path: Path, expected_bytes: bytes) -> None:
    """Deletes `path` ONLY if its current on-disk content still matches
    `expected_bytes` -- never deletes a file that changed underneath us
    (a concurrent process, a human's legitimate edit) even during
    rollback (codex_code_review_6f5dcd0.md finding 1: a prior version
    unlinked by pathname alone, so a file this attempt genuinely
    created but that was then concurrently edited before a LATER output
    failed for an unrelated reason was deleted along with everyone
    else's edit). If the content no longer matches (or the file is
    already gone), this is a silent no-op -- the caller is expected to
    report that manual recovery may be needed, never to delete content
    it can no longer prove it owns. Best-effort: an `OSError` while
    checking/deleting is swallowed so a rollback-time problem never
    masks the original failure that triggered the rollback."""
    try:
        if path.exists() and path.read_bytes() == expected_bytes:
            path.unlink()
    except OSError:
        pass


def publish_output_set_exclusive(outputs: list) -> bool:
    """Publishes MULTIPLE generated files as a single atomic-as-possible
    SET: precomputes and preflights every output's bytes BEFORE writing
    anything, so a partial set can never be created before a LATER
    member is found to differ from what would be generated
    (codex_code_review_73cf5d9.md finding 4's exact repro: the first
    file was created, then the second file's refusal left a
    half-written pair on disk). `outputs`
    is a list of `(path, bytes)` pairs. On a mid-publication failure,
    only the files THIS call actually created -- and that STILL hold
    exactly the bytes this call wrote, verified via
    `safe_unlink_if_matches` -- are removed; a pre-existing, concurrently
    edited, or human-editable file is never touched. Returns True if
    anything was actually written, False if the whole set already
    existed with identical content (a fully idempotent no-op).

    Rollback triggers on `BaseException`, not just `Exception`
    (codex_code_review_6f5dcd0.md finding 1: `_try_exclusive_write()`
    deliberately reports a post-preflight race conflict by raising
    `SystemExit`, which is a `BaseException` -- catching only `Exception`
    let that exact race branch skip rollback entirely, leaving an
    earlier output from this same attempt stranded on disk)."""
    for path, new_bytes in outputs:
        if path.exists() and path.read_bytes() != new_bytes:
            raise SystemExit(
                f"refusing to overwrite {path} -- its content differs from what would be generated now. "
                "It may contain human-entered answers, or belong to a prior run; delete it explicitly "
                "first if you really want to regenerate it from scratch."
            )

    wrote_any = False
    newly_created: list = []
    try:
        for path, new_bytes in outputs:
            if _try_exclusive_write(path, new_bytes):
                wrote_any = True
                newly_created.append((path, new_bytes))
    except BaseException:
        for created_path, expected_bytes in newly_created:
            safe_unlink_if_matches(created_path, expected_bytes)
        raise
    return wrote_any


def write_generated_csv_set_idempotent(outputs: list) -> bool:
    """CSV-rendering convenience wrapper around
    `publish_output_set_exclusive()`. `outputs` is a list of
    `(path, fieldnames, rows)` triples, all published (or refused)
    together as one set."""
    rendered = []
    for path, fieldnames, rows in outputs:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        rendered.append((path, buf.getvalue().encode("utf-8")))
    return publish_output_set_exclusive(rendered)
