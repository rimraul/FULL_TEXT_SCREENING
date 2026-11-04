"""Golden-vector tests for the four fingerprint/hash functions.

Fixed input -> fixed expected hash, pinned here so a later refactor can't
silently change what these hashes mean without a test failing
(api_contract_v2.md correction 4). If a hash formula genuinely needs to
change, that's a new api_contract version and these vectors are expected
to be regenerated deliberately, not patched quietly.
"""

import math
from pathlib import Path

import pytest

from ft_screening.hashing import (
    canonical_json_bytes,
    compute_criteria_set_hash,
    compute_criterion_cache_key,
    compute_execution_fingerprint,
    compute_request_fingerprint,
    sha256_hex,
)
from ft_screening.schema import CriteriaSet, CriterionSpec, ScreeningConfig, ScreeningItem

GOLDEN_CRITERION = CriterionSpec(
    criterion_id="population",
    version="1",
    eligibility_requirement="study population is adult humans",
    missing_evidence_policy="review",
    exclusion_reason_code="wrong_population",
    exclusion_reason_label="Wrong population",
    priority=1,
)

GOLDEN_CRITERIA_SET = CriteriaSet(
    criteria_set_id="cs-golden",
    version="1",
    criteria=[GOLDEN_CRITERION],
)

GOLDEN_CONFIG = ScreeningConfig(
    parser_name="docling",
    parser_version="1.2.3",
    chunker_version="1",
    retriever_name="lexical",
    retriever_config_version="1",
    adjudicator_provider="stub",
    adjudicator_model="stub-model",
    model_config_id="cfg-1",
    prompt_template_version="1",
    output_schema_version="1",
    aggregation_policy_version="1",
)

GOLDEN_ITEM = ScreeningItem(
    document_id="doc-1",
    report_id="report-1",
    pdf_path=Path("data/pdfs/report-1.pdf"),
    citation={"title": "Example"},
)

# Computed once by running the functions below against the fixed inputs
# above, then pinned. Regenerate deliberately (new api_contract version) if
# a formula intentionally changes -- never patch these to make a test pass.
EXPECTED_CRITERIA_SET_HASH = "e8f46607803b3fa07af67e1cf1a8f7d1959eee9541207413a9e31f9c50492eaa"
EXPECTED_EXECUTION_FINGERPRINT = "98779a5afdd9fdf18ba6baabc3578b1ad672f9ca5739b90968feb89fcc4e8feb"
EXPECTED_REQUEST_FINGERPRINT = "ed981f728b20cf8262af443f5ca37160370f460e9b58f0c98af7160cc8b8584d"
EXPECTED_CRITERION_CACHE_KEY = "3a9a4161820de0b40b325f9f75e82a11e73e42ce9080fc1e520d397c4ee4a641"


def test_criteria_set_hash_golden_vector():
    assert compute_criteria_set_hash(GOLDEN_CRITERIA_SET) == EXPECTED_CRITERIA_SET_HASH


def test_execution_fingerprint_golden_vector():
    fp = compute_execution_fingerprint(
        criteria_set_hash=EXPECTED_CRITERIA_SET_HASH,
        config=GOLDEN_CONFIG,
        api_version="1.0",
        schema_version="1",
    )
    assert fp == EXPECTED_EXECUTION_FINGERPRINT


def test_request_fingerprint_golden_vector():
    fp = compute_request_fingerprint(
        execution_fingerprint=EXPECTED_EXECUTION_FINGERPRINT,
        items=[GOLDEN_ITEM],
        document_hashes={"doc-1": "a" * 64},
        config=GOLDEN_CONFIG,
    )
    assert fp == EXPECTED_REQUEST_FINGERPRINT


def test_criterion_cache_key_golden_vector():
    key = compute_criterion_cache_key(
        document_sha256="a" * 64,
        criterion=GOLDEN_CRITERION,
        config=GOLDEN_CONFIG,
    )
    assert key == EXPECTED_CRITERION_CACHE_KEY


def test_hashes_are_64_char_hex():
    for value in (
        EXPECTED_CRITERIA_SET_HASH,
        EXPECTED_EXECUTION_FINGERPRINT,
        EXPECTED_REQUEST_FINGERPRINT,
        EXPECTED_CRITERION_CACHE_KEY,
    ):
        assert len(value) == 64
        int(value, 16)  # raises if not valid hex


# ---------------------------------------------------------------------------
# Canonical JSON algorithm properties (not tied to the domain models above)
# ---------------------------------------------------------------------------


def test_canonical_json_sorts_keys_regardless_of_input_order():
    a = canonical_json_bytes({"b": 1, "a": 2})
    b = canonical_json_bytes({"a": 2, "b": 1})
    assert a == b
    assert a == b'{"a":2,"b":1}'


def test_canonical_json_compact_separators():
    payload = canonical_json_bytes({"a": [1, 2, 3]})
    assert b" " not in payload


def test_canonical_json_normalizes_unicode_to_nfc():
    # "é" as a single codepoint (NFC) vs "e" + combining acute accent (NFD)
    nfc = "café"
    nfd = "café"
    assert nfc != nfd  # different Python strings, same rendered text
    assert canonical_json_bytes(nfc) == canonical_json_bytes(nfd)


def test_canonical_json_rejects_nan():
    with pytest.raises(ValueError):
        canonical_json_bytes({"score": math.nan})


def test_canonical_json_rejects_infinity():
    with pytest.raises(ValueError):
        canonical_json_bytes({"score": math.inf})


def test_canonical_json_allows_ordinary_floats():
    canonical_json_bytes({"score": 0.87})


def test_sha256_hex_is_deterministic():
    payload = {"x": 1, "y": [1, 2, 3]}
    assert sha256_hex(payload) == sha256_hex(dict(reversed(list(payload.items()))))


def test_criterion_cache_key_excludes_aggregation_policy_version():
    """Changing aggregation_policy_version must not change the per-criterion
    cache key -- aggregation consumes assessments, it doesn't produce them
    (api_contract_v2.md correction 4)."""
    key_a = compute_criterion_cache_key(
        document_sha256="a" * 64, criterion=GOLDEN_CRITERION, config=GOLDEN_CONFIG
    )
    different_policy_config = GOLDEN_CONFIG.model_copy(update={"aggregation_policy_version": "999"})
    key_b = compute_criterion_cache_key(
        document_sha256="a" * 64, criterion=GOLDEN_CRITERION, config=different_policy_config
    )
    assert key_a == key_b


def test_canonical_json_normalizes_dict_keys_to_nfc():
    """codex_code_review_bae78ac.md finding 1: values were normalized but
    keys were not -- canonically-equivalent Unicode keys must hash the same."""
    nfc_key = {"café": 1}  # "é" as one codepoint
    nfd_key = {"café": 1}  # "e" + combining acute accent
    assert canonical_json_bytes(nfc_key) == canonical_json_bytes(nfd_key)


def test_canonical_json_normalizes_nested_dict_keys():
    a = canonical_json_bytes({"outer": {"café": 1}})
    b = canonical_json_bytes({"outer": {"café": 1}})
    assert a == b


def test_canonical_json_rejects_non_string_keys():
    with pytest.raises(ValueError):
        canonical_json_bytes({1: "not a string key"})


def test_canonical_json_detects_key_collision_after_normalization():
    # Two distinct Python dict keys that normalize to the same NFC string --
    # must be a hash error, not a silent overwrite.
    payload = {"café": 1, "café": 2}
    assert len(payload) == 2  # distinct as Python dict keys before normalization
    with pytest.raises(ValueError):
        canonical_json_bytes(payload)


def test_existing_ascii_golden_vectors_unaffected_by_key_normalization():
    # Key-normalization only changes behavior for non-ASCII/colliding keys;
    # the pure-ASCII golden vectors above must still match exactly.
    assert compute_criteria_set_hash(GOLDEN_CRITERIA_SET) == EXPECTED_CRITERIA_SET_HASH


# ---------------------------------------------------------------------------
# compute_request_fingerprint hardening (finding 7)
# ---------------------------------------------------------------------------


def test_request_fingerprint_rejects_unexpected_document_hash_key():
    with pytest.raises(ValueError):
        compute_request_fingerprint(
            execution_fingerprint=EXPECTED_EXECUTION_FINGERPRINT,
            items=[GOLDEN_ITEM],
            document_hashes={"doc-1": "a" * 64, "doc-unknown": "b" * 64},
            config=GOLDEN_CONFIG,
        )


def test_request_fingerprint_requires_a_binding_per_item():
    item_without_expected_hash = ScreeningItem(
        document_id="doc-2", report_id="report-2", pdf_path=Path("data/pdfs/report-2.pdf")
    )
    with pytest.raises(ValueError):
        compute_request_fingerprint(
            execution_fingerprint=EXPECTED_EXECUTION_FINGERPRINT,
            items=[item_without_expected_hash],
            document_hashes={},
            config=GOLDEN_CONFIG,
        )


def test_request_fingerprint_accepts_expected_sha256_as_sole_binding():
    item = ScreeningItem(
        document_id="doc-2",
        report_id="report-2",
        pdf_path=Path("data/pdfs/report-2.pdf"),
        expected_sha256="d" * 64,
    )
    # Must not raise -- expected_sha256 alone is a valid binding.
    compute_request_fingerprint(
        execution_fingerprint=EXPECTED_EXECUTION_FINGERPRINT,
        items=[item],
        document_hashes={},
        config=GOLDEN_CONFIG,
    )


def test_request_fingerprint_rejects_malformed_hash():
    with pytest.raises(ValueError):
        compute_request_fingerprint(
            execution_fingerprint=EXPECTED_EXECUTION_FINGERPRINT,
            items=[GOLDEN_ITEM],
            document_hashes={"doc-1": "not-a-valid-sha256"},
            config=GOLDEN_CONFIG,
        )


def test_request_fingerprint_rejects_malformed_execution_fingerprint_argument():
    """Verification round finding: hash-shaped arguments entering the public
    hash functions themselves (not just values inside document_hashes) must
    be validated too."""
    with pytest.raises(ValueError):
        compute_request_fingerprint(
            execution_fingerprint="not-a-hash",
            items=[GOLDEN_ITEM],
            document_hashes={"doc-1": "a" * 64},
            config=GOLDEN_CONFIG,
        )


def test_execution_fingerprint_rejects_malformed_criteria_set_hash_argument():
    with pytest.raises(ValueError):
        compute_execution_fingerprint(
            criteria_set_hash="not-a-hash",
            config=GOLDEN_CONFIG,
            api_version="1.0",
            schema_version="1",
        )


def test_criterion_cache_key_rejects_malformed_document_sha256_argument():
    with pytest.raises(ValueError):
        compute_criterion_cache_key(
            document_sha256="not-a-hash",
            criterion=GOLDEN_CRITERION,
            config=GOLDEN_CONFIG,
        )


def test_request_fingerprint_changes_when_items_reordered():
    item_a = ScreeningItem(document_id="doc-a", report_id="report-a", pdf_path=Path("a.pdf"))
    item_b = ScreeningItem(document_id="doc-b", report_id="report-b", pdf_path=Path("b.pdf"))
    hashes = {"doc-a": "a" * 64, "doc-b": "b" * 64}
    fp_ab = compute_request_fingerprint(
        execution_fingerprint=EXPECTED_EXECUTION_FINGERPRINT,
        items=[item_a, item_b],
        document_hashes=hashes,
        config=GOLDEN_CONFIG,
    )
    fp_ba = compute_request_fingerprint(
        execution_fingerprint=EXPECTED_EXECUTION_FINGERPRINT,
        items=[item_b, item_a],
        document_hashes=hashes,
        config=GOLDEN_CONFIG,
    )
    assert fp_ab != fp_ba  # item order is part of "the same request" identity


def test_request_fingerprint_changes_when_citation_changes():
    item_no_citation = ScreeningItem(
        document_id="doc-1", report_id="report-1", pdf_path=Path("data/pdfs/report-1.pdf")
    )
    item_with_citation = ScreeningItem(
        document_id="doc-1",
        report_id="report-1",
        pdf_path=Path("data/pdfs/report-1.pdf"),
        citation={"title": "Different title"},
    )
    hashes = {"doc-1": "a" * 64}
    fp_a = compute_request_fingerprint(
        execution_fingerprint=EXPECTED_EXECUTION_FINGERPRINT,
        items=[item_no_citation],
        document_hashes=hashes,
        config=GOLDEN_CONFIG,
    )
    fp_b = compute_request_fingerprint(
        execution_fingerprint=EXPECTED_EXECUTION_FINGERPRINT,
        items=[item_with_citation],
        document_hashes=hashes,
        config=GOLDEN_CONFIG,
    )
    assert fp_a != fp_b


def test_execution_fingerprint_excludes_runtime_controls():
    """cache_mode/item_error_policy/max_workers/duplicate_content_policy
    must not affect execution_fingerprint -- they don't change the science
    (api_contract_v2.md correction 4)."""
    fp_a = compute_execution_fingerprint(
        criteria_set_hash=EXPECTED_CRITERIA_SET_HASH,
        config=GOLDEN_CONFIG,
        api_version="1.0",
        schema_version="1",
    )
    different_runtime_config = GOLDEN_CONFIG.model_copy(
        update={"cache_mode": "off", "item_error_policy": "fail_fast", "max_workers": 8}
    )
    fp_b = compute_execution_fingerprint(
        criteria_set_hash=EXPECTED_CRITERIA_SET_HASH,
        config=different_runtime_config,
        api_version="1.0",
        schema_version="1",
    )
    assert fp_a == fp_b
