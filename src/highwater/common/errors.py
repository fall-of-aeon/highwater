"""Error taxonomy.

Errors carry a stable string ``code`` because they cross process boundaries:
the wire protocol transmits the code, the UI switches on it, and tests assert
on it. Whether an error is *retriable* is a property of the error, not of the
caller's guesswork.
"""

from __future__ import annotations

__all__ = [
    "HighwaterError",
    "RetriableError",
    "Timeout",
    "TransportClosed",
    "NotLeader",
    "NoQuorum",
    "FencedLeaderEpoch",
    "UnknownTopicOrPartition",
    "NotEnoughReplicas",
    "OffsetOutOfRange",
    "CorruptRecord",
    "DuplicateSequence",
    "OutOfOrderSequence",
    "PartitionUnavailable",
    "CircuitOpen",
    "Backpressure",
    "CheckpointFailed",
    "ShuttingDown",
    "from_code",
]


class HighwaterError(Exception):
    code = "UNKNOWN"
    retriable = False

    def __init__(self, message: str = "", **context: object) -> None:
        super().__init__(message or self.code)
        self.message = message or self.code
        self.context = context

    def to_wire(self) -> dict[str, object]:
        return {"code": self.code, "message": self.message, "context": self.context}


class RetriableError(HighwaterError):
    retriable = True


class Timeout(RetriableError):
    code = "TIMEOUT"


class TransportClosed(RetriableError):
    code = "TRANSPORT_CLOSED"


class NotLeader(RetriableError):
    """Addressed to a non-leader. ``leader_hint`` lets the client redirect."""

    code = "NOT_LEADER"

    def __init__(self, message: str = "", leader_hint: str | None = None, **ctx: object) -> None:
        super().__init__(message, leader_hint=leader_hint, **ctx)
        self.leader_hint = leader_hint


class NoQuorum(RetriableError):
    code = "NO_QUORUM"


class FencedLeaderEpoch(RetriableError):
    """The caller's view of partition leadership is stale; refresh metadata."""

    code = "FENCED_LEADER_EPOCH"


class UnknownTopicOrPartition(RetriableError):
    code = "UNKNOWN_TOPIC_OR_PARTITION"


class NotEnoughReplicas(RetriableError):
    """ISR is smaller than min.insync.replicas: refuse the write rather than
    accept one we cannot honour at the requested durability."""

    code = "NOT_ENOUGH_REPLICAS"


class OffsetOutOfRange(HighwaterError):
    code = "OFFSET_OUT_OF_RANGE"


class CorruptRecord(HighwaterError):
    code = "CORRUPT_RECORD"


class DuplicateSequence(HighwaterError):
    """Idempotent producer replayed a sequence we already committed."""

    code = "DUPLICATE_SEQUENCE"


class OutOfOrderSequence(HighwaterError):
    code = "OUT_OF_ORDER_SEQUENCE"


class PartitionUnavailable(RetriableError):
    """No leader can be safely elected (all ISR down, unclean election off)."""

    code = "PARTITION_UNAVAILABLE"


class CircuitOpen(RetriableError):
    code = "CIRCUIT_OPEN"


class Backpressure(RetriableError):
    code = "BACKPRESSURE"


class CheckpointFailed(RetriableError):
    code = "CHECKPOINT_FAILED"


class ShuttingDown(HighwaterError):
    code = "SHUTTING_DOWN"


_BY_CODE: dict[str, type[HighwaterError]] = {
    cls.code: cls
    for cls in (
        HighwaterError, RetriableError, Timeout, TransportClosed, NotLeader, NoQuorum,
        FencedLeaderEpoch, UnknownTopicOrPartition, NotEnoughReplicas, OffsetOutOfRange,
        CorruptRecord, DuplicateSequence, OutOfOrderSequence, PartitionUnavailable,
        CircuitOpen, Backpressure, CheckpointFailed, ShuttingDown,
    )
}


def from_code(code: str, message: str = "", **ctx: object) -> HighwaterError:
    """Reconstruct a typed exception from a wire error code."""
    return _BY_CODE.get(code, HighwaterError)(message, **ctx)
