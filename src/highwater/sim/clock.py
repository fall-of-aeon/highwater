"""A virtual-time asyncio event loop.

Consensus bugs are timing bugs. Testing them against a wall clock means either
sleeping for real (slow, so you run few seeds) or mocking timers at every call
site (invasive, and the mock drifts from reality). Both are bad trades.

Instead we virtualise the clock underneath asyncio itself. ``BaseEventLoop``
computes how long to block as ``next_timer - self.time()`` and hands that to
its selector. So if the selector, instead of *sleeping* for that interval,
simply *declares* that the interval elapsed, then every timer in the process
fires in the right order, instantly, and in a reproducible sequence.

The consequence: a five-second election timeout costs nanoseconds, a
sixty-second partition test runs in a few milliseconds, and production code
needs no knowledge that it is being simulated -- ``asyncio.sleep`` is the only
API it ever touches.
"""

from __future__ import annotations

import asyncio
import selectors
from collections.abc import Coroutine
from typing import Any

from ..common import clock as _clock
from ..common import rand as _rand

__all__ = ["VirtualTimeEventLoop", "run_sim", "SimTimeout"]

# Real seconds we are willing to block when the simulated world is completely
# idle. Only reached if every task is parked on I/O that will never arrive,
# which in a pure in-memory simulation means a livelock worth failing on.
_IDLE_POLL = 0.02


class SimTimeout(RuntimeError):
    """Raised when the simulation exceeds its virtual time budget."""


class _VirtualSelector(selectors.DefaultSelector):  # type: ignore[misc]
    """Turns "block for N seconds" into "N seconds have now passed"."""

    def __init__(self) -> None:
        super().__init__()
        self.loop: VirtualTimeEventLoop | None = None
        self.idle_rounds = 0

    def select(self, timeout: float | None = None):  # type: ignore[override]
        # Always give genuinely-ready file descriptors priority; this keeps the
        # loop's self-pipe and any real socket working inside a simulation.
        events = super().select(0)
        if events:
            self.idle_rounds = 0
            return events

        if timeout is None:
            # No ready callbacks and no scheduled timers: nothing in the
            # simulated world can make progress on its own.
            self.idle_rounds += 1
            return super().select(_IDLE_POLL)

        if timeout > 0 and self.loop is not None:
            self.loop.advance(timeout)
        self.idle_rounds = 0
        return []


class VirtualTimeEventLoop(asyncio.SelectorEventLoop):  # type: ignore[misc]
    """A SelectorEventLoop whose notion of ``now`` we control."""

    def __init__(self, start: float = 0.0) -> None:
        self._vtime = float(start)
        self._sim_budget = float("inf")
        selector = _VirtualSelector()
        super().__init__(selector)
        selector.loop = self
        self._sim_selector = selector

    # -- clock ---------------------------------------------------------------
    def time(self) -> float:
        return self._vtime

    def advance(self, seconds: float) -> None:
        if seconds <= 0:
            return
        self._vtime += seconds
        if self._vtime > self._sim_budget:
            raise SimTimeout(
                f"simulation exceeded its virtual budget of {self._sim_budget:.3f}s"
            )

    def set_budget(self, seconds: float) -> None:
        self._sim_budget = self._vtime + seconds

    @property
    def elapsed(self) -> float:
        return self._vtime


def run_sim[T](
    coro: Coroutine[Any, Any, T],
    *,
    seed: int = 0xC0FFEE,
    budget: float = 300.0,
    wall_epoch: float = 1_760_000_000.0,
) -> T:
    """Run ``coro`` to completion in virtual time.

    ``seed`` makes every randomised decision in the system reproducible, so a
    failing run is replayed exactly by re-running with the same seed.
    """
    loop = VirtualTimeEventLoop()
    loop.set_budget(budget)
    _rand.set_root_seed(seed)
    _clock.set_wall_clock_source(lambda: wall_epoch + loop.time(), loop.time)
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        try:
            _cancel_all(loop)
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            asyncio.set_event_loop(None)
            loop.close()
            _clock.reset_wall_clock_source()


def _cancel_all(loop: asyncio.AbstractEventLoop) -> None:
    pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
    for task in pending:
        task.cancel()
    if pending:
        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
