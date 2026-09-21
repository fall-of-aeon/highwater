"""Structured logging.

Every log line is a JSON object with a stable set of keys, because these logs
are not only read by humans: the AI incident investigator parses them, and the
test suite asserts on them. ``node``, ``role`` and ``term`` are bound once per
process so every line carries the context needed to reconstruct a distributed
timeline after the fact.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from .clock import iso

_configured = False


def _add_timestamp(_logger: Any, _name: str, event_dict: dict) -> dict:
    # Use the (virtualisable) clock so simulated runs produce coherent logs.
    event_dict["ts"] = iso()
    return event_dict


def configure(
    level: str = "INFO", json_output: bool = True, node: str | None = None
) -> None:
    """Configure structlog once per process."""
    global _configured
    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        _add_timestamp,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    if json_output:
        processors.append(structlog.processors.JSONRenderer(sort_keys=True))
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()))

    logging.basicConfig(
        format="%(message)s", stream=sys.stderr, level=getattr(logging, level.upper(), 20)
    )
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), 20)
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
    if node:
        structlog.contextvars.bind_contextvars(node=node)
    _configured = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    if not _configured:
        configure(json_output=False)
    return structlog.get_logger(name)


def bind(**kwargs: Any) -> None:
    """Bind context onto every subsequent log line in this task tree."""
    structlog.contextvars.bind_contextvars(**kwargs)
