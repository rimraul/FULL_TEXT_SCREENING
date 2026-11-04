"""Reusable in-process fault-injection helper for lifecycle/crash-safety
tests (codex_code_review_810ccf0.md finding 4: "the required lifecycle
matrix, general fault-injection framework... were not implemented").

Instead of one bespoke monkeypatch per test (the pattern every prior
hardening round used ad hoc), `Interruptor` wraps a bound method on a
specific object so its Nth call raises `KeyboardInterrupt` instead of
running -- any durable-write or lock-transition method on a `ScreeningStore`
instance, or a module-level function like `_hash_file_streaming`, can be
used as an injection point this way. `tests/contract/test_lifecycle_matrix.py`
uses this to drive one table-driven test across every named hook point
instead of duplicating the setup per finding.

For REAL cross-process crash injection (not just an in-process Python
exception), see `_crash_helper.py` -- a genuine subprocess is required to
exercise OS-level lock release, since threads within one process share
lock ownership.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Interruptor:
    """Wraps `getattr(obj, method_name)` so its `call_number`-th
    invocation raises `KeyboardInterrupt` instead of running; earlier and
    later calls pass through to the real method unchanged. Call `install`
    with a pytest `monkeypatch` fixture to activate it."""

    obj: Any
    method_name: str
    call_number: int = 1
    _real: Callable = field(init=False, repr=False)
    _count: int = field(default=0, init=False)
    triggered: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._real = getattr(self.obj, self.method_name)

    def _wrapper(self, *args, **kwargs):
        self._count += 1
        if self._count == self.call_number:
            self.triggered = True
            raise KeyboardInterrupt()
        return self._real(*args, **kwargs)

    def install(self, monkeypatch) -> "Interruptor":
        monkeypatch.setattr(self.obj, self.method_name, self._wrapper)
        return self


@dataclass
class Failer:
    """Like `Interruptor`, but raises an ordinary exception (default
    `ConnectionError`) instead of `KeyboardInterrupt` -- for injecting a
    non-cancellation store/backend failure (e.g. a cache outage) at a
    named point, rather than simulating user-initiated interruption."""

    obj: Any
    method_name: str
    call_number: int = 1
    exception_factory: Callable[[], Exception] = field(default=lambda: ConnectionError("simulated backend failure"))
    _real: Callable = field(init=False, repr=False)
    _count: int = field(default=0, init=False)
    triggered: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._real = getattr(self.obj, self.method_name)

    def _wrapper(self, *args, **kwargs):
        self._count += 1
        if self._count == self.call_number:
            self.triggered = True
            raise self.exception_factory()
        return self._real(*args, **kwargs)

    def install(self, monkeypatch) -> "Failer":
        monkeypatch.setattr(self.obj, self.method_name, self._wrapper)
        return self
