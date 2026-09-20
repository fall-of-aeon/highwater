"""Link-level fault injection.

Faults are applied per *directed* link (src -> dst), not per node. That choice
buys two things the naive "node is down" model cannot express:

  * asymmetric partitions -- A can hear B but B cannot hear A. This is the
    shape that breaks naive failure detectors and produces the classic
    "everyone thinks everyone else is dead" livelock.
  * partial partitions -- A|B split while both still reach C. Real Raft has to
    survive this; a symmetric model never exercises it.

The table is a plain serialisable object so the gateway can push identical
rules into every node, and the UI can render exactly what is in force.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ..common.rand import stream

__all__ = ["LinkFault", "FaultTable", "FaultDecision"]


@dataclass(slots=True)
class LinkFault:
    partitioned: bool = False
    drop_prob: float = 0.0
    duplicate_prob: float = 0.0
    reorder_prob: float = 0.0
    latency_ms: float = 0.0
    jitter_ms: float = 0.0

    def is_clean(self) -> bool:
        return not (
            self.partitioned
            or self.drop_prob
            or self.duplicate_prob
            or self.reorder_prob
            or self.latency_ms
            or self.jitter_ms
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class FaultDecision:
    """What the transport should do with one outbound frame."""

    deliver: bool = True
    delay: float = 0.0
    copies: int = 1
    reorder: bool = False


_CLEAN = LinkFault()


@dataclass
class FaultTable:
    """Fault rules in force across the cluster."""

    links: dict[tuple[str, str], LinkFault] = field(default_factory=dict)
    default: LinkFault = field(default_factory=LinkFault)
    # Each group is a set of node ids; nodes in different groups cannot talk.
    # An empty list means "no partition in force".
    groups: list[set[str]] = field(default_factory=list)
    # Nodes that are administratively frozen (process paused, not killed).
    frozen: set[str] = field(default_factory=set)

    def clear(self) -> None:
        self.links.clear()
        self.groups.clear()
        self.frozen.clear()
        self.default = LinkFault()

    def partition_groups(self, *groups: set[str] | list[str]) -> None:
        self.groups = [set(g) for g in groups]

    def rule_for(self, src: str, dst: str) -> LinkFault:
        return self.links.get((src, dst), self.default)

    def blocked_by_partition(self, src: str, dst: str) -> bool:
        if src in self.frozen or dst in self.frozen:
            return True
        if not self.groups:
            return False
        src_groups = [i for i, g in enumerate(self.groups) if src in g]
        dst_groups = [i for i, g in enumerate(self.groups) if dst in g]
        # Nodes not mentioned in any group are reachable by everyone; this makes
        # "partition these two off" a one-liner without listing the whole cluster.
        if not src_groups or not dst_groups:
            return False
        return not set(src_groups) & set(dst_groups)

    def decide(self, src: str, dst: str) -> FaultDecision:
        """Decide the fate of a single frame from ``src`` to ``dst``."""
        if self.blocked_by_partition(src, dst):
            return FaultDecision(deliver=False)
        rule = self.rule_for(src, dst)
        if rule is _CLEAN or rule.is_clean():
            return FaultDecision()
        if rule.partitioned:
            return FaultDecision(deliver=False)

        rng = stream(f"fault:{src}->{dst}")
        if rule.drop_prob and rng.random() < rule.drop_prob:
            return FaultDecision(deliver=False)

        delay = 0.0
        if rule.latency_ms:
            delay = rule.latency_ms / 1000.0
        if rule.jitter_ms:
            delay += rng.uniform(0.0, rule.jitter_ms / 1000.0)

        copies = 2 if (rule.duplicate_prob and rng.random() < rule.duplicate_prob) else 1
        reorder = bool(rule.reorder_prob and rng.random() < rule.reorder_prob)
        return FaultDecision(deliver=True, delay=delay, copies=copies, reorder=reorder)

    def to_dict(self) -> dict:
        return {
            "links": [
                {"src": s, "dst": d, **f.to_dict()} for (s, d), f in self.links.items()
            ],
            "default": self.default.to_dict(),
            "groups": [sorted(g) for g in self.groups],
            "frozen": sorted(self.frozen),
        }

    @classmethod
    def from_dict(cls, data: dict) -> FaultTable:
        t = cls()
        for entry in data.get("links", []):
            e = dict(entry)
            src, dst = e.pop("src"), e.pop("dst")
            t.links[(src, dst)] = LinkFault(**e)
        if data.get("default"):
            t.default = LinkFault(**data["default"])
        t.groups = [set(g) for g in data.get("groups", [])]
        t.frozen = set(data.get("frozen", []))
        return t
