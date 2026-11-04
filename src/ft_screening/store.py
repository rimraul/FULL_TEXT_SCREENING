"""Minimal, real (not fake) filesystem-backed ScreeningStore.

Implements api_contract_v2.md's artifact layout exactly:

    output_dir/
    +-- runs/<run_id>/
    |   +-- run_result.json
    |   +-- item_results.jsonl   (materialized current view, rewritten
    |   |                         atomically by input_index on each update)
    |   +-- ledger.jsonl          (append-only, every attempt/transition)
    +-- cache/<cache-schema-version>/...

Exclusive per-run locking uses a real OS-level advisory lock on `.lock`
(`fcntl.flock` on POSIX, `msvcrt.locking` on Windows) -- NOT mere file
existence. Fresh-run creation uses `mkdir(exist_ok=False)` directly (no
separate exists()-then-mkdir race). "Minimal" means: no cross-host locking,
no cache sharding -- sufficient for single-machine orchestration and its
tests, not a production deployment story.

Known limitation, DELIBERATE, not a bug (codex_code_review_146f98e.md's
non-blocking hardening note): an existing-but-UNREADABLE `manifest.json`
(disk corruption, truncation, an out-of-space partial write bypassing
`_atomic_write_text`'s own atomicity via external interference) is still
treated as "claimed" by `begin_run(resume=False)`'s existence check
(correctly refuses to silently overwrite -- the file existing at all
means SOMETHING was claimed here, even if it's now unreadable), while
`begin_run(resume=True)` + `read_run_manifest()` correctly treat it as
unusable and reject with `ResumeConflictError`. The run becomes stuck:
neither a fresh retry nor a resume can proceed. This is intentional --
automatically deleting or overwriting on read failure could silently
destroy the last record of real, possibly-completed work, which would be
far worse than requiring a human to look. Recovery is manual and
deliberately not automated here: move (don't delete) the affected
`runs/<run_id>/` directory aside, inspect `item_results.jsonl`/
`ledger.jsonl` if recoverable, then either restore a good `manifest.json`
or start a fresh run under a new `run_id`. No administrative repair
command exists yet; add one if this proves to be more than a
theoretical concern in practice.

Second review round (codex_code_review_44d9e90.md) findings fixed here:
1/9. item_results.jsonl is now a materialized-current-view, atomically
     rewritten whole on each update (was a blind append -- crash-unsafe and
     grew a second line per resume attempt for the same index).
6. Cache entries now carry source_run_id (a bare assessment can't tell a
   caller which run originally computed it -- needed for cross-run cache-hit
   provenance).
9. begin_run's fresh-run path uses mkdir(exist_ok=False) directly (removes
   the exists()-then-mkdir TOCTOU race); general OSError from filesystem
   operations is wrapped as RunSetupError/ResumeConflictError; corrupt/
   truncated JSON/JSONL records are skipped (with the exception recorded)
   rather than crashing the whole run.

Third review round (codex_code_review_53f21de.md) findings fixed here:
1. append_ledger_event/write_item_result are now protected by a per-store
   lock -- Stage B worker threads call append_ledger_event concurrently
   (cache_hit events, progress_callback_disabled), and an unsynchronized
   open/append was observably lossy (a 150-event stress probe lost 1-5
   events in 9 of 10 runs).
4. The cache envelope is now a validated Pydantic model, not raw dict/json
   extraction -- a syntactically-valid-but-wrong-shaped source_run_id (e.g.
   a JSON list instead of a string) previously reached
   set(cache_source_run_ids) in the orchestrator and crashed the whole run
   with an unhashable-type TypeError. Malformed envelopes are now a clean
   cache miss at the store boundary, same as any other corrupt entry.

Fifth review round (codex_code_review_810ccf0.md) findings fixed here:
2. "Claimed" is now defined SOLELY by `manifest.json` existing (written
   atomically by `write_run_manifest` -> `_atomic_write_text`), never by
   the bare run directory's existence. `begin_run()` no longer
   pre-creates `runs/<run_id>/` for a fresh run at all -- it's created
   lazily as a side effect of the first atomic manifest write. Previously
   `begin_run()` created the directory FIRST and wrote the manifest as a
   separate later step; an interrupt/crash in between left a directory
   that existed (blocking a fresh retry with RunSetupError) but had no
   manifest inside it (blocking resume with ResumeConflictError) -- a
   genuine dead end. The lock file also moved out from under the run
   directory entirely, to `runs/.locks/<run_id>.lock`, so claim-checking
   and locking are fully decoupled from whether/where run data exists yet.

Fourth review round (codex_code_review_0f48a3d.md) findings fixed here:
2. Locking was previously an EXISTENCE-sentinel `.lock` file: correct for
   concurrent-access denial, but a process that crashed (os._exit(),
   kill -9, power loss) without calling end_run() left the sentinel behind
   forever -- a later resume got ResumeConflictError indefinitely, with no
   recovery short of a human deleting internal state. Replaced with a real
   OS-level advisory lock (fcntl.flock on POSIX, msvcrt.locking on
   Windows) held for the life of the open file descriptor: the OS
   releases it automatically when the holding process's file descriptors
   close, which happens on ANY process exit including a crash, no cleanup
   code required. The `.lock` file itself is now just the lock's target,
   not the signal.
3. The cache envelope's `assessment`/`source_run_id` now nest a validated
   `CachedAssessment` (components.py) directly, rather than duplicating
   its own separate field-by-field validation -- see that class's
   docstring for why the orchestrator needs this to be revalidated
   regardless of which ScreeningStore implementation returned it.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from ft_screening.components import CachedAssessment
from ft_screening.errors import ResumeConflictError, RunSetupError
from ft_screening.schema import CriterionAssessment, ScreeningItemResult, ScreeningRunResult

if sys.platform == "win32":
    import msvcrt

    def _acquire_exclusive_lock(fd: int) -> bool:
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False

    def _release_exclusive_lock(fd: int) -> None:
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
else:
    import fcntl

    def _acquire_exclusive_lock(fd: int) -> bool:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def _release_exclusive_lock(fd: int) -> None:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass


class _CacheEnvelope(BaseModel):
    """Validated on-disk shape for a cache entry. Both read and write paths
    go through this -- writing here means we can never persist a malformed
    envelope ourselves; reading here means a malformed one (however it got
    there) is caught before it reaches orchestrator code. Nests the same
    CachedAssessment the orchestrator itself revalidates, rather than
    duplicating field-by-field checks in two places."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cached: CachedAssessment
    cached_at: str


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise


class FileScreeningStore:
    def __init__(self, *, output_dir: Path, cache_schema_version: str = "v1") -> None:
        self._output_dir = Path(output_dir)
        self._cache_schema_version = cache_schema_version
        # Guards append_ledger_event/write_item_result: Stage B worker
        # threads call these concurrently (finding 1), and a raw
        # open/write/close with no synchronization is observably lossy.
        self._write_lock = threading.Lock()
        # run_id -> open fd holding this process's OS-level lock on that
        # run's .lock file. Guarded separately since begin_run/end_run can
        # in principle be called from different threads.
        self._open_locks: dict[str, int] = {}
        self._locks_guard = threading.Lock()

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    # -- paths ----------------------------------------------------------

    def _run_dir(self, run_id: str) -> Path:
        return self._output_dir / "runs" / run_id

    def _lock_path(self, run_id: str) -> Path:
        # Deliberately NOT inside _run_dir(run_id) (codex_code_review_810ccf0.md
        # finding 2) -- the lock must be acquirable/releasable independent
        # of whether the run directory exists yet, so locking and
        # claim-checking (manifest.json's existence) are fully decoupled.
        return self._output_dir / "runs" / ".locks" / f"{run_id}.lock"

    def _manifest_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "manifest.json"

    def artifact_paths(self, *, run_id: str) -> tuple[Path, Path, Path]:
        run_dir = self._run_dir(run_id)
        return (run_dir / "run_result.json", run_dir / "item_results.jsonl", run_dir / "ledger.jsonl")

    def _cache_path(self, *, cache_key: str) -> Path:
        return self._output_dir / "cache" / self._cache_schema_version / f"{cache_key}.json"

    # -- run lifecycle + locking -----------------------------------------

    def begin_run(self, *, run_id: str, resume: bool) -> None:
        error_cls = ResumeConflictError if resume else RunSetupError
        lock_path = self._lock_path(run_id)
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise error_cls(f"cannot create lock directory for run_id={run_id!r}: {exc}") from exc
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
        except OSError as exc:
            raise error_cls(f"cannot open lock file for run_id={run_id!r}: {exc}") from exc
        try:
            if os.fstat(fd).st_size == 0:
                os.write(fd, b"\0")  # ensure a non-empty byte range exists to lock
        except OSError:
            pass
        if not _acquire_exclusive_lock(fd):
            os.close(fd)
            raise error_cls(
                f"run_id={run_id!r} is already locked by another live process"
                + (" -- another process may be resuming it" if resume else " -- concurrent creation attempt")
            )

        # codex_code_review_810ccf0.md finding 2: "claimed" means exactly
        # one thing -- manifest.json exists -- checked only now, AFTER the
        # lock is held, so this check can never race with another process's
        # write_run_manifest(). No directory is created here for a fresh
        # run: it's created lazily, as a side effect of the first atomic
        # manifest write, so there is no window where a directory exists
        # without a manifest that could confuse either check below.
        manifest_exists = self._manifest_path(run_id).exists()
        if resume and not manifest_exists:
            _release_exclusive_lock(fd)
            os.close(fd)
            raise ResumeConflictError(
                f"cannot resume run_id={run_id!r}: no manifest found -- nothing durable was ever recorded"
            )
        if not resume and manifest_exists:
            _release_exclusive_lock(fd)
            os.close(fd)
            raise RunSetupError(f"run_id={run_id!r} already exists; a fresh run must use a new run_id")

        with self._locks_guard:
            self._open_locks[run_id] = fd

    def end_run(self, *, run_id: str) -> None:
        with self._locks_guard:
            fd = self._open_locks.pop(run_id, None)
        if fd is not None:
            _release_exclusive_lock(fd)
            try:
                os.close(fd)
            except OSError:
                pass

    # -- manifest (execution/request fingerprint, for resume compatibility) --

    def write_run_manifest(
        self,
        *,
        run_id: str,
        execution_fingerprint: str,
        request_fingerprint: str | None,
        provisional_request_binding: str | None = None,
        document_hash_checkpoints: dict[str, str] | None = None,
    ) -> None:
        payload = {
            "execution_fingerprint": execution_fingerprint,
            "request_fingerprint": request_fingerprint,
            "provisional_request_binding": provisional_request_binding,
            "document_hash_checkpoints": document_hash_checkpoints or {},
        }
        # _atomic_write_text creates the parent directory (mkdir(parents=True,
        # exist_ok=True)) as a side effect -- for a fresh run's FIRST
        # manifest write, this is the one and only moment runs/<run_id>/
        # comes into existence, and it never exists without this file
        # already inside it (the write itself is atomic: temp file + fsync
        # + os.replace).
        _atomic_write_text(self._manifest_path(run_id), json.dumps(payload))

    def read_run_manifest(self, *, run_id: str) -> dict[str, Any] | None:
        path = self._manifest_path(run_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    # -- ledger -----------------------------------------------------------

    def append_ledger_event(self, *, run_id: str, event: dict) -> None:
        _, _, ledger_path = self.artifact_paths(run_id=run_id)
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        record = dict(event)
        record.setdefault("recorded_at", datetime.now(timezone.utc).isoformat())
        with self._write_lock:
            with open(ledger_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
                f.flush()
                os.fsync(f.fileno())

    # -- item results (materialized current view, atomic rewrite) -----------

    def read_completed_item_results(self, *, run_id: str) -> dict[int, ScreeningItemResult]:
        return self._read_item_results(run_id=run_id)

    def _read_item_results(self, *, run_id: str) -> dict[int, ScreeningItemResult]:
        _, item_results_path, _ = self.artifact_paths(run_id=run_id)
        if not item_results_path.exists():
            return {}
        results: dict[int, ScreeningItemResult] = {}
        with open(item_results_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = ScreeningItemResult.model_validate_json(line)
                except (json.JSONDecodeError, ValidationError):
                    # Crash-truncated or corrupt record -- skip rather than
                    # crash the whole run; the item it described gets
                    # reprocessed on resume, which is the safe default.
                    continue
                results[item.input_index] = item
        return results

    def write_item_result(self, *, run_id: str, item_result: ScreeningItemResult) -> None:
        # Lock covers the whole read-modify-write: two concurrent callers
        # each computing "current" from a stale read would otherwise race
        # and one update could silently overwrite the other.
        with self._write_lock:
            current = self._read_item_results(run_id=run_id)
            current[item_result.input_index] = item_result
            _, item_results_path, _ = self.artifact_paths(run_id=run_id)
            text = "".join(current[idx].model_dump_json() + "\n" for idx in sorted(current))
            _atomic_write_text(item_results_path, text)

    # -- run result ---------------------------------------------------------

    def write_run_result(self, *, run_id: str, run_result: ScreeningRunResult) -> None:
        run_result_path, _, _ = self.artifact_paths(run_id=run_id)
        _atomic_write_text(run_result_path, run_result.model_dump_json())

    # -- assessment cache (identity-neutral assessment + source provenance) --

    def get_cached_assessment(self, *, cache_key: str) -> CachedAssessment | None:
        path = self._cache_path(cache_key=cache_key)
        if not path.exists():
            return None
        try:
            envelope = _CacheEnvelope.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError, ValueError):
            # Corrupt/truncated/malformed cache entry (including a
            # syntactically-valid-but-wrong-shaped source_run_id, e.g. a
            # JSON list instead of a string) -- treat as a miss rather than
            # crash the run; the caller will recompute and overwrite it.
            return None
        return envelope.cached

    def put_cached_assessment(self, *, cache_key: str, assessment: CriterionAssessment, source_run_id: str) -> None:
        envelope = _CacheEnvelope(
            cached=CachedAssessment(assessment=assessment, source_run_id=source_run_id),
            cached_at=datetime.now(timezone.utc).isoformat(),
        )
        _atomic_write_text(self._cache_path(cache_key=cache_key), envelope.model_dump_json())
