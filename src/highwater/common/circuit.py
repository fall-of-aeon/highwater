"""Per-peer circuit breaker.

A node that keeps dialling a peer that is dead (or on the far side of a
partition) burns the event loop on connect timeouts and inflates tail latency
for everything else sharing it. The breaker turns that into a fast local
failure, and the half-open probe is what lets the link recover without a
restart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .clock import monotonic
from .errors import CircuitOpen

__all__ = ["CircuitState", "CircuitBreaker"]


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    name: str
    failure_threshold: int = 5
    reset_timeout: float = 1.0
    half_open_max_calls: int = 1

    state: CircuitState = CircuitState.CLOSED
    _failures: int = field(default=0, repr=False)
    _opened_at: float = field(default=0.0, repr=False)
    _half_open_calls: int = field(default=0, repr=False)
    trips: int = field(default=0)

    def allow(self) -> bool:
        """Should the next call be attempted?"""
        if self.state is CircuitState.CLOSED:
            return True
        if self.state is CircuitState.OPEN:
            if monotonic() - self._opened_at >= self.reset_timeout:
                self.state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                return True
            return False
        return self._half_open_calls < self.half_open_max_calls

    def guard(self) -> None:
        if not self.allow():
            raise CircuitOpen(f"circuit open for {self.name}", peer=self.name)
        if self.state is CircuitState.HALF_OPEN:
            self._half_open_calls += 1

    def on_success(self) -> None:
        self._failures = 0
        self._half_open_calls = 0
        self.state = CircuitState.CLOSED

    def on_failure(self) -> None:
        self._failures += 1
        if self.state is CircuitState.HALF_OPEN or self._failures >= self.failure_threshold:
            if self.state is not CircuitState.OPEN:
                self.trips += 1
            self.state = CircuitState.OPEN
            self._opened_at = monotonic()

    def snapshot(self) -> dict[str, object]:
        return {
            "peer": self.name,
            "state": self.state.value,
            "failures": self._failures,
            "trips": self.trips,
        }
