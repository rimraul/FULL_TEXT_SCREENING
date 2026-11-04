"""Direct unit tests for scripts/gold_set_workflow_common.py's
set-publication primitives -- codex_code_review_6f5dcd0.md's exact
fault/race repros against `publish_output_set_exclusive()` and
`safe_unlink_if_matches()`, isolated from the higher-level workflow
scripts that use them.
"""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, r"C:/projects/FT_SCREENING/scripts")

import gold_set_workflow_common as common  # noqa: E402


def test_publish_output_set_exclusive_writes_all_when_none_exist(tmp_path):
    outputs = [(tmp_path / "a.txt", b"AAA"), (tmp_path / "b.txt", b"BBB")]
    wrote = common.publish_output_set_exclusive(outputs)
    assert wrote is True
    assert (tmp_path / "a.txt").read_bytes() == b"AAA"
    assert (tmp_path / "b.txt").read_bytes() == b"BBB"


def test_publish_output_set_exclusive_is_idempotent_when_identical(tmp_path):
    outputs = [(tmp_path / "a.txt", b"AAA"), (tmp_path / "b.txt", b"BBB")]
    common.publish_output_set_exclusive(outputs)
    wrote = common.publish_output_set_exclusive(outputs)
    assert wrote is False


def test_publish_output_set_exclusive_refuses_when_preexisting_content_differs(tmp_path):
    (tmp_path / "a.txt").write_bytes(b"different")
    with pytest.raises(SystemExit):
        common.publish_output_set_exclusive([(tmp_path / "a.txt", b"AAA"), (tmp_path / "b.txt", b"BBB")])
    assert not (tmp_path / "b.txt").exists()


def test_publish_output_set_exclusive_rolls_back_output_one_on_a_post_preflight_race(tmp_path, monkeypatch):
    """Exact repro: a race-conflict on output two raises SystemExit
    (a BaseException, not an Exception) -- output one, created earlier
    in this SAME attempt, must still be rolled back, not stranded."""
    a_path, b_path = tmp_path / "a.txt", tmp_path / "b.txt"
    real_try = common._try_exclusive_write
    call_count = {"n": 0}

    def faulty_try(path, data):
        call_count["n"] += 1
        if call_count["n"] == 2:
            # A concurrent writer creates output two with DIFFERENT
            # content after this call's own preflight already passed.
            path.write_bytes(b"CONCURRENT-CONTENT")
        return real_try(path, data)

    monkeypatch.setattr(common, "_try_exclusive_write", faulty_try)
    with pytest.raises(SystemExit):
        common.publish_output_set_exclusive([(a_path, b"AAA"), (b_path, b"BBB")])

    assert not a_path.exists()  # rolled back
    assert b_path.read_bytes() == b"CONCURRENT-CONTENT"  # the concurrent writer's content is untouched


def test_publish_output_set_exclusive_never_deletes_a_concurrently_edited_output_during_rollback(tmp_path, monkeypatch):
    """Exact repro: output one is created by this attempt, then edited
    by something else (a human, a concurrent process) BEFORE output two
    fails for an unrelated reason -- rollback must preserve the edit,
    not delete it just because this attempt's own bookkeeping still
    names that path."""
    a_path, b_path = tmp_path / "a.txt", tmp_path / "b.txt"
    real_try = common._try_exclusive_write
    call_count = {"n": 0}

    def faulty_try(path, data):
        call_count["n"] += 1
        if call_count["n"] == 1:
            result = real_try(path, data)
            path.write_bytes(b"HUMAN-EDITED-CONTENT")  # edited after we wrote it
            return result
        raise OSError("disk full (simulated)")

    monkeypatch.setattr(common, "_try_exclusive_write", faulty_try)
    with pytest.raises(OSError):
        common.publish_output_set_exclusive([(a_path, b"AAA"), (b_path, b"BBB")])

    assert a_path.read_bytes() == b"HUMAN-EDITED-CONTENT"  # never deleted


def test_publish_output_set_exclusive_rolls_back_on_keyboard_interrupt(tmp_path, monkeypatch):
    a_path, b_path = tmp_path / "a.txt", tmp_path / "b.txt"
    real_try = common._try_exclusive_write
    call_count = {"n": 0}

    def faulty_try(path, data):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise KeyboardInterrupt()
        return real_try(path, data)

    monkeypatch.setattr(common, "_try_exclusive_write", faulty_try)
    with pytest.raises(KeyboardInterrupt):
        common.publish_output_set_exclusive([(a_path, b"AAA"), (b_path, b"BBB")])
    assert not a_path.exists()
    assert not b_path.exists()


def test_safe_unlink_if_matches_deletes_when_content_matches(tmp_path):
    p = tmp_path / "f.txt"
    p.write_bytes(b"AAA")
    common.safe_unlink_if_matches(p, b"AAA")
    assert not p.exists()


def test_safe_unlink_if_matches_preserves_when_content_differs(tmp_path):
    p = tmp_path / "f.txt"
    p.write_bytes(b"BBB")
    common.safe_unlink_if_matches(p, b"AAA")
    assert p.exists()
    assert p.read_bytes() == b"BBB"


def test_safe_unlink_if_matches_is_a_noop_when_file_is_already_gone(tmp_path):
    p = tmp_path / "f.txt"
    common.safe_unlink_if_matches(p, b"AAA")  # must not raise
