"""Serialisation primitives shared by the wire protocol and the on-disk log."""

from __future__ import annotations

import zlib
from typing import Any

import orjson

__all__ = ["dumps", "loads", "crc32", "varint_encode", "varint_decode", "sizeof_fmt"]


def dumps(obj: Any) -> bytes:
    """Compact JSON bytes.

    ``OPT_SERIALIZE_NUMPY`` keeps benchmark payloads cheap; sort keys so that
    byte-identical logical messages hash identically (useful for dedupe tests).
    """
    return orjson.dumps(obj, option=orjson.OPT_SORT_KEYS | orjson.OPT_SERIALIZE_NUMPY)


def loads(raw: bytes | bytearray | memoryview) -> Any:
    return orjson.loads(bytes(raw) if not isinstance(raw, bytes) else raw)


def crc32(data: bytes | bytearray | memoryview) -> int:
    """CRC-32 (IEEE) used to detect torn writes and bit-rot in log segments."""
    return zlib.crc32(data) & 0xFFFFFFFF


def varint_encode(value: int) -> bytes:
    """Unsigned LEB128. Keeps small offsets/lengths to a single byte."""
    if value < 0:
        raise ValueError("varint_encode expects a non-negative integer")
    out = bytearray()
    while True:
        b = value & 0x7F
        value >>= 7
        if value:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def varint_decode(buf: bytes | memoryview, pos: int = 0) -> tuple[int, int]:
    """Return ``(value, new_pos)``."""
    result = 0
    shift = 0
    while True:
        if pos >= len(buf):
            raise ValueError("truncated varint")
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7
        if shift > 63:
            raise ValueError("varint too long")


def sizeof_fmt(num: float, suffix: str = "B") -> str:
    for unit in ("", "Ki", "Mi", "Gi", "Ti"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Pi{suffix}"
