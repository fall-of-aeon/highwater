"""Time.

The whole system reads time through this module and schedules through asyncio.
That is deliberate: asyncio's event loop already *is* a clock abstraction
(``loop.time()`` drives every timer), so virtualising the loop's clock in
``highwater.sim`` makes every ``asyncio.sleep``, ``wait_for`` and timeout in the
codebase run in simulated time without a single call site changing.

Wall-clock timestamps (the ones that end up in telemetry events) are the one
thing asyncio does not give us, so they get an explicit, swappable source.
"""

from __future__ import annotations

import time
from collections.abc import Callable

__all__ = [
    "monotonic",
    "now",
    "now_ms",
    "set_wall_clock_source",
    "reset_wall_clock_source",
    "iso",
    "Deadline",
]

_wall_source: Callable[[], float] | None = None
_mono_source: Callable[[], float] | None = None


def set_wall_clock_source(fn: Callable[[], float], mono: Callable[[], float] | None = None) -> None:
    """Install clock sources (used by the deterministic simulator)."""
    global _wall_source, _mono_source
    _wall_source = fn
    _mono_source = mono


def reset_wall_clock_source() -> None:
    global _wall_source, _mono_source
    _wall_source = None
    _mono_source = None


def now() -> float:
    """Wall-clock seconds since the epoch (virtualised under simulation)."""
    if _wall_source is not None:
        return _wall_source()
    return time.time()


def now_ms() -> float:
    return now() * 1000.0


def monotonic() -> float:
    """Monotonic seconds.

    Always ``time.monotonic()`` unless the simulator has installed a virtual
    source. Deliberately *not* ``loop.time()``: the event loop's clock epoch is
    implementation-defined -- uvloop starts near zero while asyncio's selector
    loop tracks ``time.monotonic()`` -- so a value captured before the loop
    exists cannot be compared with one captured inside it. That mismatch turns
    every deadline computed across a startup boundary into nonsense, and it
    does so only under uvloop, which is exactly the configuration production
    runs and tests do not.
    """
    if _mono_source is not None:
        return _mono_source()
    return time.monotonic()


def iso(ts: float | None = None) -> str:
    """RFC3339 timestamp with millisecond precision, always UTC."""
    import datetime as _dt

    t = now() if ts is None else ts
    dt = _dt.datetime.fromtimestamp(t, tz=_dt.UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


class Deadline:
    """A monotonic deadline that survives being passed across await points."""

    __slots__ = ("_expires_at",)

    def __init__(self, timeout: float) -> None:
        self._expires_at = monotonic() + timeout

    @property
    def remaining(self) -> float:
        return max(0.0, self._expires_at - monotonic())

    @property
    def expired(self) -> bool:
        return monotonic() >= self._expires_at

    def extend(self, seconds: float) -> None:
        self._expires_at += seconds

    def reset(self, timeout: float) -> None:
        self._expires_at = monotonic() + timeout

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Deadline(remaining={self.remaining:.3f}s)"
