"""Fault-aware delivery scheduling for one directed link.

Subtlety worth being precise about: a naive fault injector that adds jittered
latency by spawning a task per frame will silently *reorder* frames on a TCP
connection, which TCP does not do. Tests written against that injector pass
for the wrong reason.

So delivery times here are clamped to be monotonically non-decreasing per link
-- preserving the in-order delivery a stream socket guarantees -- unless the
``reorder`` fault is explicitly requested, which models a multi-path or
datagram delivery path (see FAILURE_MODES.md). Ordering is a property you have
to opt out of, not one you lose by accident.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from ..common.clock import monotonic
from .faults import FaultTable

__all__ = ["LinkStats", "DeliveryScheduler"]


@dataclass
class LinkStats:
    sent: int = 0
    delivered: int = 0
    dropped: int = 0
    duplicated: int = 0
    reordered: int = 0
    delayed: int = 0
    bytes_sent: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "sent": self.sent,
            "delivered": self.delivered,
            "dropped": self.dropped,
            "duplicated": self.duplicated,
            "reordered": self.reordered,
            "delayed": self.delayed,
            "bytes_sent": self.bytes_sent,
        }


class DeliveryScheduler:
    """Applies a FaultTable to outbound frames and schedules their delivery."""

    def __init__(self, faults: FaultTable, stats: dict[str, LinkStats] | None = None) -> None:
        self.faults = faults
        self.stats: dict[str, LinkStats] = stats if stats is not None else {}
        self._last_release: dict[tuple[str, str], float] = {}
        self._tasks: set[asyncio.Task] = set()

    def stats_for(self, src: str, dst: str) -> LinkStats:
        key = f"{src}->{dst}"
        st = self.stats.get(key)
        if st is None:
            st = LinkStats()
            self.stats[key] = st
        return st

    def submit(
        self,
        src: str,
        dst: str,
        nbytes: int,
        deliver: Callable[[], None],
    ) -> bool:
        """Schedule ``deliver`` subject to the faults in force.

        Returns False if the frame was dropped. ``deliver`` may be invoked more
        than once (duplication) and may be invoked later (latency).
        """
        st = self.stats_for(src, dst)
        st.sent += 1
        st.bytes_sent += nbytes

        decision = self.faults.decide(src, dst)
        if not decision.deliver:
            st.dropped += 1
            return False

        if decision.copies > 1:
            st.duplicated += decision.copies - 1

        key = (src, dst)
        now = monotonic()
        target = now + decision.delay
        if decision.reorder:
            st.reordered += 1
            # Explicitly allow this frame to land before earlier ones.
            target = now + decision.delay * 0.25
        else:
            # Preserve stream ordering: never release before the previous frame.
            target = max(target, self._last_release.get(key, 0.0))
            self._last_release[key] = target

        wait = max(0.0, target - now)
        if wait <= 0.0:
            for _ in range(decision.copies):
                st.delivered += 1
                deliver()
            return True

        st.delayed += 1

        async def _later() -> None:
            await asyncio.sleep(wait)
            for _ in range(decision.copies):
                st.delivered += 1
                deliver()

        task = asyncio.get_running_loop().create_task(_later())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return True

    def cancel_all(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        self._tasks.clear()

    def snapshot(self) -> dict[str, dict[str, int]]:
        return {k: v.to_dict() for k, v in self.stats.items()}
