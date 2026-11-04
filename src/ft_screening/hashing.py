"""Canonical JSON serialization and the four fingerprint/hash functions.

Implements api_contract_v2.md correction 4 (four separately-defined
hashes, each with an exact formula) and correction 5 (never hash a Python
model instance -- always hash its canonical JSON serialization).

Canonical-JSON algorithm: UTF-8 encoding, sorted object keys, fixed compact
separators, Unicode strings AND object keys normalized to NFC (object keys
per codex_code_review_bae78ac.md finding 1 -- values-only normalization
let canonically-equivalent Unicode keys hash differently), and non-finite
floats (NaN/Infinity) rejected as hash errors rather than silently
serialized.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from ft_screening.schema import CriterionSpec, CriteriaSet, ScreeningConfig, ScreeningItem, ScreeningRequest

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _validate_hash_arg(name: str, value: str) -> str:
    """Validate a caller-supplied hash-shaped argument (criteria_set_hash,
    execution_fingerprint, document_sha256) before it enters a payload.
    These are hashes *produced* by this module's own functions (or by an
    upstream preflight step for document_sha256) -- rejecting a malformed
    one here catches a caller passing an arbitrary string instead of a
    real hash before it silently gets baked into a derived fingerprint.
    """
    normalized = value.lower()
    if not _SHA256_RE.match(normalized):
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters (sha256), got {value!r}")
    return normalized


def _canonicalize(obj: Any) -> Any:
    if isinstance(obj, dict):
        normalized: dict[str, Any] = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise ValueError(f"canonical JSON object keys must be strings, got {type(k).__name__}: {k!r}")
            nk = unicodedata.normalize("NFC", k)
            if nk in normalized:
                raise ValueError(
                    f"canonical JSON key collision after NFC normalization: {nk!r} "
                    "(two distinct original keys normalize to the same key)"
                )
            normalized[nk] = _canonicalize(v)
        return normalized
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(v) for v in obj]
    if isinstance(obj, str):
        return unicodedata.normalize("NFC", obj)
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"non-finite float in canonical payload: {obj!r}")
        return obj
    return obj


def canonical_json_bytes(payload: Any) -> bytes:
    """Deterministic UTF-8 JSON bytes for `payload` -- the input to every hash below."""
    normalized = _canonicalize(payload)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_hex(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def hash_model(model: BaseModel) -> str:
    """Hash a model's canonical JSON serialization. Never hash a model instance directly."""
    return sha256_hex(model.model_dump(mode="json"))


def compute_criteria_set_hash(criteria_set: CriteriaSet) -> str:
    """SHA-256 of canonical JSON of the complete CriteriaSet."""
    return hash_model(criteria_set)


def compute_execution_fingerprint(
    *,
    criteria_set_hash: str,
    config: ScreeningConfig,
    api_version: str,
    schema_version: str,
) -> str:
    """Answers: would this configuration produce the same science?

    Excludes runtime controls (cache_mode, item_error_policy, max_workers,
    duplicate_content_policy), paths, run_id, timestamps, credentials,
    citation metadata -- none of those change the science.
    """
    criteria_set_hash = _validate_hash_arg("criteria_set_hash", criteria_set_hash)
    payload = {
        "api_version": api_version,
        "schema_version": schema_version,
        "criteria_set_hash": criteria_set_hash,
        "parser_name": config.parser_name,
        "parser_version": config.parser_version,
        "chunker_version": config.chunker_version,
        "parser_options": config.parser_options,
        "retriever_name": config.retriever_name,
        "retriever_config_version": config.retriever_config_version,
        "retriever_options": config.retriever_options,
        "adjudicator_provider": config.adjudicator_provider,
        "adjudicator_model": config.adjudicator_model,
        "model_config_id": config.model_config_id,
        "model_options": config.model_options,
        "prompt_template_version": config.prompt_template_version,
        "output_schema_version": config.output_schema_version,
        "aggregation_policy_version": config.aggregation_policy_version,
    }
    return sha256_hex(payload)


def compute_request_fingerprint(
    *,
    execution_fingerprint: str,
    items: Sequence[ScreeningItem],
    document_hashes: Mapping[str, str],
    config: ScreeningConfig,
) -> str:
    """Answers: is this a legitimate resume of the same request, or a different one?

    `document_hashes` maps document_id -> known sha256 (expected or actual,
    whichever is known at call time -- this function takes them as input
    rather than reading PDFs itself, since no PDF I/O exists yet).

    Every item must have a defined binding: either an entry in
    `document_hashes` or a `ScreeningItem.expected_sha256`. Per
    codex_code_review_bae78ac.md finding 7, fingerprint construction is
    disallowed rather than silently proceeding when neither exists --
    preflight must produce a content hash (or the caller must supply
    expected_sha256) first. `document_hashes` must not reference a
    document_id absent from `items`.

    Excludes run_id, output_dir, timestamps, callback. Includes citation
    (it appears in output, so it's part of "the same request") and the
    runtime policies that affect the result envelope shape, even though
    those are excluded from execution_fingerprint. Item order is part of
    the payload deliberately -- reordering items is a different request.
    """
    execution_fingerprint = _validate_hash_arg("execution_fingerprint", execution_fingerprint)
    item_ids = {item.document_id for item in items}
    unexpected_keys = set(document_hashes) - item_ids
    if unexpected_keys:
        raise ValueError(
            f"document_hashes contains document_id(s) not present in items: {sorted(unexpected_keys)}"
        )

    resolved_items = []
    for item in items:
        document_hash = document_hashes.get(item.document_id, item.expected_sha256)
        if document_hash is None:
            raise ValueError(
                f"no document hash binding for document_id={item.document_id!r}: supply "
                "document_hashes[document_id] (actual content hash, from preflight) or set "
                "ScreeningItem.expected_sha256 before computing request_fingerprint"
            )
        document_hash = _validate_hash_arg(f"document_hash for document_id={item.document_id!r}", document_hash)
        resolved_items.append(
            {
                "document_id": item.document_id,
                "report_id": item.report_id,
                "study_id": item.study_id,
                "document_hash": document_hash,
                "citation": item.citation,
            }
        )

    payload = {
        "execution_fingerprint": execution_fingerprint,
        "items": resolved_items,
        "item_error_policy": config.item_error_policy,
        "duplicate_content_policy": config.duplicate_content_policy,
        "max_workers": config.max_workers,
    }
    return sha256_hex(payload)


def compute_provisional_request_binding(
    *,
    execution_fingerprint: str,
    items: Sequence[ScreeningItem],
    config: ScreeningConfig,
) -> str:
    """A cheap, pre-hash binding of "which request is this" -- computable
    entirely from the request object, before Stage A1 has read a single
    byte of any PDF. Exists ONLY to protect the window where a run's
    manifest is provisional (`request_fingerprint` not yet knowable):
    without this, a resume during that window had nothing but
    `execution_fingerprint` to check, which excludes item identities,
    citation, and declared hashes entirely, so an attacker or a mistaken
    caller could resume a provisional run_id with a COMPLETELY DIFFERENT
    request (codex_code_review_810ccf0.md finding 1 -- direct
    reproduction: same run_id, different PDF/document_id/report_id/
    citation/item_error_policy/max_workers, resume silently accepted it).

    Deliberately NOT a replacement for `compute_request_fingerprint`: once
    real content hashes exist (post Stage A1), that remains the
    authoritative check. This covers every one of its inputs that's
    available without file I/O -- ordered document/report/study IDs,
    citation, a declared `expected_sha256` when the caller supplied one,
    and the same result-shaping runtime policies -- plus a normalized
    absolute `pdf_path` as the pre-hash proxy for content identity (the
    final fingerprint uses the real content hash instead; this is only a
    stand-in for the provisional window and is never compared against a
    request_fingerprint).
    """
    execution_fingerprint = _validate_hash_arg("execution_fingerprint", execution_fingerprint)
    resolved_items = [
        {
            "document_id": item.document_id,
            "report_id": item.report_id,
            "study_id": item.study_id,
            "citation": item.citation,
            "expected_sha256": item.expected_sha256,
            "pdf_path": str(item.pdf_path.resolve()),
        }
        for item in items
    ]
    payload = {
        "execution_fingerprint": execution_fingerprint,
        "items": resolved_items,
        "item_error_policy": config.item_error_policy,
        "duplicate_content_policy": config.duplicate_content_policy,
        "max_workers": config.max_workers,
    }
    return sha256_hex(payload)


def compute_criterion_cache_key(
    *,
    document_sha256: str,
    criterion: CriterionSpec,
    config: ScreeningConfig,
) -> str:
    """Per-criterion cache key -- NOT the whole CriteriaSet, so a change to
    one criterion doesn't invalidate every other criterion's cache.

    Excludes aggregation_policy_version deliberately: aggregation consumes
    already-computed CriterionAssessments, it doesn't produce them, so an
    aggregation-policy change invalidates the derived EligibilityResult but
    not these per-criterion cache entries.
    """
    document_sha256 = _validate_hash_arg("document_sha256", document_sha256)
    payload = {
        "document_sha256": document_sha256,
        "criterion": criterion.model_dump(mode="json"),
        "parser_name": config.parser_name,
        "parser_version": config.parser_version,
        "chunker_version": config.chunker_version,
        "parser_options": config.parser_options,
        "retriever_name": config.retriever_name,
        "retriever_config_version": config.retriever_config_version,
        "retriever_options": config.retriever_options,
        "adjudicator_provider": config.adjudicator_provider,
        "adjudicator_model": config.adjudicator_model,
        "model_config_id": config.model_config_id,
        "model_options": config.model_options,
        "prompt_template_version": config.prompt_template_version,
        "output_schema_version": config.output_schema_version,
    }
    return sha256_hex(payload)
