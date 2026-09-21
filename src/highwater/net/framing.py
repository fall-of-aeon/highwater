"""Wire framing.

    +---------------------------------------------------------------+
    | u32  length      | bytes following this field                  |
    | u8   magic       | 0x48 ('H') -- catches stream desync early   |
    | u8   version     | protocol version                            |
    | u8   kind        | REQUEST | RESPONSE | ERROR | ONEWAY | PING  |
    | u8   flags       | bit0: has_blob                              |
    | u64  corr_id     | correlates a response with its request      |
    | u32  header_len  | length of the JSON header                   |
    | ...  header      | JSON: method + arguments                    |
    | ...  blob        | optional opaque bytes (record batches)      |
    +---------------------------------------------------------------+

The split between a JSON header and an opaque binary blob is deliberate. Log
record batches are already a packed binary format; base64-ing them into JSON
would inflate them ~33% and cost two extra copies on the hottest path in the
system. Control fields stay JSON so the protocol remains inspectable.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from ..common.codec import dumps, loads

__all__ = ["FrameKind", "Frame", "encode_frame", "FrameDecoder", "HEADER_SIZE", "MAGIC"]

MAGIC = 0x48
VERSION = 1
HEADER_SIZE = 16  # after the u32 length prefix: magic..header_len inclusive
_PREFIX = struct.Struct("<I")
_HEAD = struct.Struct("<BBBBQI")
MAX_FRAME = 64 * 1024 * 1024

FLAG_BLOB = 0x01


class FrameKind(IntEnum):
    REQUEST = 1
    RESPONSE = 2
    ERROR = 3
    ONEWAY = 4
    PING = 5
    PONG = 6


@dataclass(slots=True)
class Frame:
    kind: FrameKind
    corr_id: int
    header: dict[str, Any] = field(default_factory=dict)
    blob: bytes = b""

    @property
    def method(self) -> str:
        return str(self.header.get("m", ""))

    @property
    def args(self) -> dict[str, Any]:
        a = self.header.get("a")
        return a if isinstance(a, dict) else {}

    def size(self) -> int:
        return HEADER_SIZE + 4 + len(dumps(self.header)) + len(self.blob)


def encode_frame(frame: Frame) -> bytes:
    header_bytes = dumps(frame.header)
    flags = FLAG_BLOB if frame.blob else 0
    body = _HEAD.pack(
        MAGIC, VERSION, int(frame.kind), flags, frame.corr_id, len(header_bytes)
    ) + header_bytes + frame.blob
    if len(body) > MAX_FRAME:
        raise ValueError(f"frame of {len(body)} bytes exceeds MAX_FRAME")
    return _PREFIX.pack(len(body)) + body


class FrameDecoder:
    """Incremental decoder: feed it arbitrary byte chunks, get whole frames."""

    __slots__ = ("_buf",)

    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, data: bytes) -> list[Frame]:
        self._buf.extend(data)
        return list(self._drain())

    def _drain(self):
        while True:
            if len(self._buf) < 4:
                return
            (length,) = _PREFIX.unpack_from(self._buf, 0)
            if length > MAX_FRAME:
                raise ValueError(f"declared frame length {length} exceeds MAX_FRAME")
            if len(self._buf) < 4 + length:
                return
            body = memoryview(self._buf)[4 : 4 + length]
            magic, version, kind, flags, corr_id, hlen = _HEAD.unpack_from(body, 0)
            if magic != MAGIC:
                raise ValueError(f"bad magic 0x{magic:02x}: stream desynchronised")
            if version != VERSION:
                raise ValueError(f"unsupported protocol version {version}")
            hstart = _HEAD.size
            header = loads(body[hstart : hstart + hlen]) if hlen else {}
            blob = bytes(body[hstart + hlen :]) if flags & FLAG_BLOB else b""
            del body
            del self._buf[: 4 + length]
            yield Frame(FrameKind(kind), corr_id, header, blob)

    @property
    def buffered(self) -> int:
        return len(self._buf)
