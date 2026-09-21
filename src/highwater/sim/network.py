"""An in-memory network for deterministic simulation.

``SimTransport`` implements the same ``Transport`` interface as the TCP one, so
Raft, the brokers and the stream workers run unmodified on top of it. What
changes is that delivery happens through the virtual-time loop: a 30 ms
cross-node RPC costs no real time, and the whole cluster lives inside one
process where a test can inspect every node's internals at any instant.

A non-zero ``base_latency`` is on by default. Zero-latency delivery is not just
unrealistic, it is actively misleading: it hides the races that only appear
when a message is in flight while its sender changes state.
"""

from __future__ import annotations

import asyncio
import contextlib

from ..common.errors import TransportClosed
from ..net.faults import FaultTable, LinkFault
from ..net.framing import Frame, encode_frame
from ..net.transport import Transport

__all__ = ["SimNetwork", "SimTransport"]


class SimTransport(Transport):
    def __init__(self, node_id: str, network: SimNetwork) -> None:
        super().__init__(node_id, network.faults)
        self.network = network
        # Share one scheduler cluster-wide so link stats and ordering state are
        # global, exactly as they are for a real network.
        self.scheduler = network.scheduler
        self.link_stats = network.link_stats
        self.crashed = False

    async def start(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(TransportClosed("transport closed"))
        self._pending.clear()

    async def _dispatch(self, peer: str, frame: Frame) -> None:
        if self.crashed or self.closed:
            raise TransportClosed(f"{self.node_id} is down")
        target = self.network.nodes.get(peer)
        nbytes = len(encode_frame(frame))
        if target is None or target.crashed or target.closed:
            st = self.scheduler.stats_for(self.node_id, peer)
            st.sent += 1
            st.dropped += 1
            return

        src = self.node_id

        def _deliver() -> None:
            dst = self.network.nodes.get(peer)
            if dst is None or dst.crashed or dst.closed:
                return
            task = asyncio.get_running_loop().create_task(dst._on_frame(src, frame))
            self.network._tasks.add(task)
            task.add_done_callback(self.network._tasks.discard)

        self.scheduler.submit(src, peer, nbytes, _deliver)


class SimNetwork:
    """Owns the fault table and the set of simulated nodes."""

    def __init__(
        self,
        base_latency_ms: float = 1.0,
        jitter_ms: float = 0.4,
        faults: FaultTable | None = None,
    ) -> None:
        self.faults = faults if faults is not None else FaultTable()
        if self.faults.default.is_clean():
            self.faults.default = LinkFault(latency_ms=base_latency_ms, jitter_ms=jitter_ms)
        self.link_stats: dict = {}
        from ..net.link import DeliveryScheduler

        self.scheduler = DeliveryScheduler(self.faults, self.link_stats)
        self.nodes: dict[str, SimTransport] = {}
        self._tasks: set[asyncio.Task] = set()

    def transport(self, node_id: str) -> SimTransport:
        t = SimTransport(node_id, self)
        self.nodes[node_id] = t
        return t

    def crash(self, node_id: str) -> None:
        """Hard-stop a node: in-flight frames to and from it are lost."""
        t = self.nodes.get(node_id)
        if t is not None:
            t.crashed = True

    def restore(self, node_id: str) -> None:
        t = self.nodes.get(node_id)
        if t is not None:
            t.crashed = False

    def partition(self, *groups: set[str] | list[str]) -> None:
        self.faults.partition_groups(*groups)

    def heal(self) -> None:
        base = self.faults.default
        self.faults.clear()
        self.faults.default = base
        for t in self.nodes.values():
            t.crashed = False

    async def drain(self, rounds: int = 8) -> None:
        """Let every in-flight delivery and handler settle."""
        for _ in range(rounds):
            await asyncio.sleep(0)
            if self._tasks:
                await asyncio.gather(*list(self._tasks), return_exceptions=True)

    async def close(self) -> None:
        self.scheduler.cancel_all()
        for t in list(self.nodes.values()):
            with contextlib.suppress(Exception):
                await t.close()
        for task in list(self._tasks):
            task.cancel()
        self._tasks.clear()
