"""Transport: request/response RPC with pluggable delivery.

One interface, two implementations:

  * ``TcpTransport``  -- real sockets, real processes, real ``SIGKILL``.
  * ``SimTransport``  -- in-memory delivery on the virtual-time loop.

Node code (Raft, brokers, workers) is written against the interface and has no
idea which one it is running on. That is what lets the same consensus
implementation be both shipped and exhaustively simulation-tested.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import socket
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

from ..common.circuit import CircuitBreaker
from ..common.errors import HighwaterError, Timeout, TransportClosed, from_code
from ..common.logging import get_logger
from .faults import FaultTable
from .framing import Frame, FrameDecoder, FrameKind, encode_frame
from .link import DeliveryScheduler, LinkStats

__all__ = ["Transport", "TcpTransport", "RpcHandler", "Address", "Reply"]

log = get_logger(__name__)

Reply = tuple[dict[str, Any], bytes]
RpcHandler = Callable[[str, str, dict[str, Any], bytes], Awaitable[Reply]]

DEFAULT_TIMEOUT = 2.0


def _set_nodelay(writer: asyncio.StreamWriter) -> None:
    """Disable Nagle's algorithm.

    Without this, the kernel holds a small write back waiting either for more
    data or for the peer's ACK -- and the peer's delayed-ACK timer holds that
    ACK back in turn. The two timers deadlock into a fixed ~40-200 ms stall.

    It is devastating here precisely because the protocol is request/response
    with small frames: an ``acks=all`` produce needs the follower's *next*
    fetch to report its new offset before the high watermark can advance, and
    that fetch is a few dozen bytes. Measured on this machine, enabling
    TCP_NODELAY took acks=all from ~202 ms to sub-millisecond -- a 200x
    difference that no amount of protocol tuning would have found, and that an
    in-memory simulation can never reproduce because it has no TCP underneath.
    """
    sock = writer.get_extra_info("socket")
    if sock is not None:
        with contextlib.suppress(OSError):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


class Address:
    __slots__ = ("host", "port")

    def __init__(self, host: str, port: int) -> None:
        self.host, self.port = host, port

    @classmethod
    def parse(cls, s: str) -> Address:
        host, _, port = s.rpartition(":")
        return cls(host or "127.0.0.1", int(port))

    def __str__(self) -> str:
        return f"{self.host}:{self.port}"


class Transport(ABC):
    """Abstract request/response transport between named nodes."""

    def __init__(self, node_id: str, faults: FaultTable | None = None) -> None:
        self.node_id = node_id
        self.faults = faults if faults is not None else FaultTable()
        self.link_stats: dict[str, LinkStats] = {}
        self.scheduler = DeliveryScheduler(self.faults, self.link_stats)
        self.breakers: dict[str, CircuitBreaker] = {}
        self._handler: RpcHandler | None = None
        self._corr = itertools.count(1)
        self._pending: dict[int, asyncio.Future[Frame]] = {}
        self.rpc_sent = 0
        self.rpc_failed = 0
        self.closed = False

    def set_handler(self, handler: RpcHandler) -> None:
        self._handler = handler

    def breaker(self, peer: str) -> CircuitBreaker:
        cb = self.breakers.get(peer)
        if cb is None:
            cb = CircuitBreaker(peer)
            self.breakers[peer] = cb
        return cb

    def next_corr(self) -> int:
        return next(self._corr)

    # -- API -----------------------------------------------------------------
    async def call(
        self,
        peer: str,
        method: str,
        args: dict[str, Any] | None = None,
        blob: bytes = b"",
        timeout: float = DEFAULT_TIMEOUT,
    ) -> Reply:
        """Send a request and await its reply. Raises on timeout or remote error."""
        if self.closed:
            raise TransportClosed("transport closed")
        cb = self.breaker(peer)
        cb.guard()
        corr = self.next_corr()
        frame = Frame(FrameKind.REQUEST, corr, {"m": method, "a": args or {}}, blob)
        fut: asyncio.Future[Frame] = asyncio.get_running_loop().create_future()
        self._pending[corr] = fut
        self.rpc_sent += 1
        try:
            await self._dispatch(peer, frame)
            reply = await asyncio.wait_for(fut, timeout)
        except TimeoutError as exc:
            self.rpc_failed += 1
            cb.on_failure()
            raise Timeout(f"rpc {method} -> {peer} timed out after {timeout}s") from exc
        except HighwaterError:
            self.rpc_failed += 1
            cb.on_failure()
            raise
        finally:
            self._pending.pop(corr, None)
        cb.on_success()
        if reply.kind is FrameKind.ERROR:
            err = reply.header.get("e", {})
            raise from_code(err.get("code", "UNKNOWN"), err.get("message", ""))
        return reply.header, reply.blob

    async def notify(
        self, peer: str, method: str, args: dict[str, Any] | None = None, blob: bytes = b""
    ) -> None:
        """Fire-and-forget. Used for telemetry and heartbeat-style signals."""
        if self.closed:
            return
        frame = Frame(FrameKind.ONEWAY, self.next_corr(), {"m": method, "a": args or {}}, blob)
        with contextlib.suppress(Exception):
            await self._dispatch(peer, frame)

    # -- to be provided by concrete transports -------------------------------
    @abstractmethod
    async def _dispatch(self, peer: str, frame: Frame) -> None:
        """Hand a frame to the delivery layer (which applies faults)."""

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    # -- inbound -------------------------------------------------------------
    async def _on_frame(self, peer: str, frame: Frame) -> None:
        if frame.kind in (FrameKind.RESPONSE, FrameKind.ERROR):
            fut = self._pending.get(frame.corr_id)
            # A duplicated response finds no pending future the second time --
            # dropping it silently is exactly the idempotent behaviour we want.
            if fut is not None and not fut.done():
                fut.set_result(frame)
            return
        if frame.kind is FrameKind.PING:
            with contextlib.suppress(Exception):
                await self._dispatch(peer, Frame(FrameKind.PONG, frame.corr_id, {}))
            return
        if self._handler is None:
            return
        if frame.kind is FrameKind.ONEWAY:
            with contextlib.suppress(Exception):
                await self._handler(peer, frame.method, frame.args, frame.blob)
            return
        try:
            header, blob = await self._handler(peer, frame.method, frame.args, frame.blob)
            reply = Frame(FrameKind.RESPONSE, frame.corr_id, header, blob)
        except HighwaterError as exc:
            reply = Frame(FrameKind.ERROR, frame.corr_id, {"e": exc.to_wire()})
        except Exception as exc:  # noqa: BLE001 - never let a handler kill the node
            log.exception("rpc_handler_error", method=frame.method, peer=peer)
            reply = Frame(
                FrameKind.ERROR, frame.corr_id, {"e": {"code": "UNKNOWN", "message": str(exc)}}
            )
        # The caller may have died while we were handling its request; failing
        # to deliver a reply to a dead peer is normal, not an error.
        with contextlib.suppress(Exception):
            await self._dispatch(peer, reply)

    def stats(self) -> dict[str, Any]:
        return {
            "node": self.node_id,
            "rpc_sent": self.rpc_sent,
            "rpc_failed": self.rpc_failed,
            "pending": len(self._pending),
            "links": self.scheduler.snapshot(),
            "breakers": [cb.snapshot() for cb in self.breakers.values()],
        }


class TcpTransport(Transport):
    """Length-prefixed frames over asyncio TCP, with faults applied on send.

    Faults are applied on the *sending* side so that a partition means the
    bytes genuinely never reach the peer -- there is no privileged observer
    that could leak information across the partition.
    """

    def __init__(
        self,
        node_id: str,
        bind: Address,
        peers: dict[str, Address] | None = None,
        faults: FaultTable | None = None,
    ) -> None:
        super().__init__(node_id, faults)
        self.bind = bind
        self.peers: dict[str, Address] = dict(peers or {})
        self._server: asyncio.AbstractServer | None = None
        self._writers: dict[str, asyncio.StreamWriter] = {}
        self._dial_locks: dict[str, asyncio.Lock] = {}
        self._reader_tasks: set[asyncio.Task] = set()

    def add_peer(self, node_id: str, addr: Address) -> None:
        self.peers[node_id] = addr

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._accept, self.bind.host, self.bind.port, reuse_address=True
        )
        # Resolve port 0 to the OS-assigned port so tests need no fixed ports.
        sock = self._server.sockets[0]
        self.bind = Address(*sock.getsockname()[:2])
        log.info("transport_listening", node=self.node_id, addr=str(self.bind))

    async def close(self) -> None:
        self.closed = True
        self.scheduler.cancel_all()
        for task in list(self._reader_tasks):
            task.cancel()
        for w in list(self._writers.values()):
            with contextlib.suppress(Exception):
                w.close()
        self._writers.clear()
        if self._server is not None:
            self._server.close()
            with contextlib.suppress(Exception):
                await self._server.wait_closed()
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(TransportClosed("transport closed"))
        self._pending.clear()

    async def _accept(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        peer_id = "?"
        _set_nodelay(writer)
        decoder = FrameDecoder()
        try:
            while not self.closed:
                data = await reader.read(65536)
                if not data:
                    break
                for frame in decoder.feed(data):
                    if frame.kind is FrameKind.ONEWAY and frame.method == "__hello__":
                        peer_id = str(frame.args.get("node_id", "?"))
                        # Reuse the inbound socket for replies so a peer behind
                        # a one-way-reachable path still gets answered.
                        self._writers.setdefault(peer_id, writer)
                        continue
                    await self._on_frame(peer_id, frame)
        except (asyncio.CancelledError, ConnectionResetError):
            pass
        except Exception:  # noqa: BLE001
            log.debug("inbound_connection_error", node=self.node_id, peer=peer_id, exc_info=True)
        finally:
            with contextlib.suppress(Exception):
                writer.close()

    async def _writer_for(self, peer: str) -> asyncio.StreamWriter:
        w = self._writers.get(peer)
        if w is not None and not w.is_closing():
            return w
        lock = self._dial_locks.setdefault(peer, asyncio.Lock())
        async with lock:
            w = self._writers.get(peer)
            if w is not None and not w.is_closing():
                return w
            addr = self.peers.get(peer)
            if addr is None:
                raise TransportClosed(f"unknown peer {peer}")
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(addr.host, addr.port), timeout=1.0
                )
            except (TimeoutError, OSError) as exc:
                raise TransportClosed(f"cannot reach {peer} at {addr}") from exc
            _set_nodelay(writer)
            writer.write(
                encode_frame(
                    Frame(
                        FrameKind.ONEWAY, 0, {"m": "__hello__", "a": {"node_id": self.node_id}}
                    )
                )
            )
            self._writers[peer] = writer
            task = asyncio.get_running_loop().create_task(self._read_loop(peer, reader))
            self._reader_tasks.add(task)
            task.add_done_callback(self._reader_tasks.discard)
            return writer

    async def _read_loop(self, peer: str, reader: asyncio.StreamReader) -> None:
        decoder = FrameDecoder()
        try:
            while not self.closed:
                data = await reader.read(65536)
                if not data:
                    break
                for frame in decoder.feed(data):
                    await self._on_frame(peer, frame)
        except (asyncio.CancelledError, ConnectionResetError):
            pass
        except Exception:  # noqa: BLE001
            log.debug("read_loop_error", node=self.node_id, peer=peer, exc_info=True)
        finally:
            self._writers.pop(peer, None)

    async def _dispatch(self, peer: str, frame: Frame) -> None:
        payload = encode_frame(frame)

        # Check for a partition *before* dialling. Two reasons: a partitioned
        # peer must not even see a TCP SYN, and dialling an unreachable peer
        # would otherwise burn the connect timeout on every frame. We test only
        # the deterministic partition predicate here -- calling decide() would
        # consume randomness twice per frame and break seed reproducibility.
        if self.faults.blocked_by_partition(self.node_id, peer):
            st = self.scheduler.stats_for(self.node_id, peer)
            st.sent += 1
            st.dropped += 1
            return

        try:
            writer = await self._writer_for(peer)
        except TransportClosed:
            st = self.scheduler.stats_for(self.node_id, peer)
            st.sent += 1
            st.dropped += 1
            raise

        def _send() -> None:
            if writer.is_closing():
                return
            try:
                writer.write(payload)
            except Exception:  # noqa: BLE001
                log.debug("write_failed", node=self.node_id, peer=peer)

        self.scheduler.submit(self.node_id, peer, len(payload), _send)
