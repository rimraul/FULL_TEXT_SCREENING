"""codex_code_review_afeba09.md finding 3: the prior artifact tests only
covered an empty directory, a correct declared digest, and a wrong
declared digest -- not the two explicitly-required mutation cases (one
missing file, one modified file) against a KNOWN-VALID bundle. These use
`validate()`'s `expected_tree_digest` override (added specifically for
this) against a tiny synthetic directory -- never the real ~1.2 GB
bundle -- so they run in the default fast suite, no `@pytest.mark.slow`
needed.
"""

from __future__ import annotations

from ft_screening.docling_artifacts import ArtifactValidationError, compute_tree_digest, validate


def _make_bundle(root, files: dict[str, bytes]) -> None:
    for relative_path, content in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def test_valid_bundle_validates_successfully(tmp_path):
    _make_bundle(tmp_path, {"model_a/weights.bin": b"weights-a", "model_b/config.json": b'{"x": 1}'})
    expected = compute_tree_digest(tmp_path)
    assert validate(tmp_path, expected_tree_digest=expected) == expected


def test_missing_file_is_rejected(tmp_path):
    """A bundle that validated correctly, with one expected file
    subsequently deleted (e.g. an interrupted download, disk cleanup,
    partial corruption) must be rejected -- explicitly required by
    codex_code_review_afeba09.md finding 3."""
    _make_bundle(tmp_path, {"model_a/weights.bin": b"weights-a", "model_b/config.json": b'{"x": 1}'})
    expected = compute_tree_digest(tmp_path)

    (tmp_path / "model_b" / "config.json").unlink()

    try:
        validate(tmp_path, expected_tree_digest=expected)
    except ArtifactValidationError as exc:
        assert expected in str(exc)
    else:
        raise AssertionError("validate() must reject a bundle with a missing file")


def test_modified_file_is_rejected(tmp_path):
    """A bundle where every expected file is PRESENT but one's content
    changed (bit-rot, tampering, a bad partial re-download that didn't
    change the file list) must also be rejected -- distinct from the
    missing-file case, explicitly required by finding 3."""
    _make_bundle(tmp_path, {"model_a/weights.bin": b"weights-a", "model_b/config.json": b'{"x": 1}'})
    expected = compute_tree_digest(tmp_path)

    (tmp_path / "model_a" / "weights.bin").write_bytes(b"weights-a-TAMPERED")

    try:
        validate(tmp_path, expected_tree_digest=expected)
    except ArtifactValidationError:
        pass
    else:
        raise AssertionError("validate() must reject a bundle with a modified file")


def test_extra_unexpected_file_is_rejected(tmp_path):
    """A bundle with every expected file unchanged PLUS an extra,
    unexpected file (e.g. a mixed bootstrap from two different sources)
    must also fail -- the tree digest covers the whole directory listing,
    not just a subset check."""
    _make_bundle(tmp_path, {"model_a/weights.bin": b"weights-a"})
    expected = compute_tree_digest(tmp_path)

    (tmp_path / "model_a" / "unexpected_extra.bin").write_bytes(b"surprise")

    try:
        validate(tmp_path, expected_tree_digest=expected)
    except ArtifactValidationError:
        pass
    else:
        raise AssertionError("validate() must reject a bundle with an unexpected extra file")


def test_bookkeeping_directories_excluded_from_digest(tmp_path):
    """huggingface_hub's own .cache/.locks bookkeeping legitimately
    varies across otherwise-identical downloads (timestamps, lock files)
    -- must not affect the digest, or every bootstrap would produce a
    spuriously different one even with identical model content."""
    _make_bundle(tmp_path, {"model_a/weights.bin": b"weights-a"})
    digest_before = compute_tree_digest(tmp_path)

    _make_bundle(
        tmp_path,
        {
            "model_a/.cache/huggingface/download/weights.bin.metadata": b"some-varying-timestamp-content",
            "model_a/.locks/some.lock": b"lock-content",
        },
    )
    digest_after = compute_tree_digest(tmp_path)
    assert digest_before == digest_after


def test_load_manifest_has_a_real_digest():
    """Sanity check on the actual committed manifest (not a synthetic
    fixture) -- catches the manifest file being malformed/emptied without
    needing the real 1.2 GB bundle to detect it."""
    from ft_screening.docling_artifacts import load_manifest

    manifest = load_manifest()
    assert len(manifest["expected_tree_digest"]) == 64
    assert int(manifest["expected_tree_digest"], 16) >= 0  # valid hex
    assert manifest["pinned_repos"]
