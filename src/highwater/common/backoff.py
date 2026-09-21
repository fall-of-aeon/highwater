"""Retry policy: exponential backoff with full jitter, and a retry driver.

Full jitter (``sleep = U(0, min(cap, base * 2**n))``) rather than plain
exponential because synchronised retries after a partition heals are a
self-inflicted thundering herd -- the exact failure mode chaos testing
surfaces within seconds.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass

from .errors import HighwaterError, RetriableError
from .rand import stream

__all__ = ["BackoffPolicy", "retry"]


@dataclass(slots=True, frozen=True)
class BackoffPolicy:
    base: float = 0.05
    cap: float = 2.0
    max_attempts: int = 5
    multiplier: float = 2.0
    stream_name: str = "backoff"

    def delay(self, attempt: int) -> float:
        """Full-jitter delay for a zero-indexed ``attempt``."""
        ceiling = min(self.cap, self.base * (self.multiplier**attempt))
        return stream(self.stream_name).uniform(0.0, ceiling)

    def delays(self) -> Iterable[float]:
        for attempt in range(self.max_attempts - 1):
            yield self.delay(attempt)


async def retry[T](
    fn: Callable[[], Awaitable[T]],
    policy: BackoffPolicy | None = None,
    *,
    retry_on: tuple[type[BaseException], ...] = (RetriableError, TimeoutError, OSError),
    on_retry: Callable[[int, BaseException, float], None] | None = None,
) -> T:
    """Call ``fn`` until it succeeds, the policy is exhausted, or it raises a
    non-retriable error. Returns the value; re-raises the last error."""
    pol = policy or BackoffPolicy()
    last: BaseException | None = None
    for attempt in range(pol.max_attempts):
        try:
            return await fn()
        except asyncio.CancelledError:
            raise
        except retry_on as exc:
            last = exc
            if attempt == pol.max_attempts - 1:
                break
            sleep_for = pol.delay(attempt)
            if on_retry is not None:
                on_retry(attempt, exc, sleep_for)
            await asyncio.sleep(sleep_for)
        except HighwaterError:
            raise
    assert last is not None
    raise last
