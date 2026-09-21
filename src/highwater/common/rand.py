"""Seeded randomness.

Randomised election timeouts and jittered backoff are load-bearing for
correctness *and* for liveness, so they must be reproducible. Every random
decision in HIGHWATER draws from a named stream derived from a single root
seed; the simulator sets the seed, so a failing test replays exactly.
"""

from __future__ import annotations

import hashlib
import random
import struct

__all__ = ["set_root_seed", "get_root_seed", "stream", "jitter", "uniform"]

_root_seed: int = 0xC0FFEE
_streams: dict[str, random.Random] = {}


def set_root_seed(seed: int) -> None:
    """Reset every named stream to a deterministic state derived from ``seed``."""
    global _root_seed
    _root_seed = seed
    _streams.clear()


def get_root_seed() -> int:
    return _root_seed


def stream(name: str) -> random.Random:
    """A named, independent, reproducible random stream.

    Independent streams matter: if every component drew from one generator,
    adding a single random call anywhere would perturb every other component's
    sequence and destroy replay.
    """
    r = _streams.get(name)
    if r is None:
        digest = hashlib.blake2b(
            name.encode("utf-8") + struct.pack("<Q", _root_seed & 0xFFFFFFFFFFFFFFFF),
            digest_size=8,
        ).digest()
        r = random.Random(struct.unpack("<Q", digest)[0])
        _streams[name] = r
    return r


def uniform(name: str, low: float, high: float) -> float:
    return stream(name).uniform(low, high)


def jitter(name: str, value: float, ratio: float = 0.25) -> float:
    """Full-jitter style perturbation of ``value`` by +/- ``ratio``."""
    return value * (1.0 + stream(name).uniform(-ratio, ratio))
