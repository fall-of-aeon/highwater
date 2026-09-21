"""Shared fixtures.

Simulation tests run on a virtual-time event loop, so they need their own
runner rather than pytest-asyncio's. ``sim_test`` is that runner: it takes an
async test body, a seed, and a virtual-time budget, and fails loudly if the
cluster cannot make progress within it.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import pytest

from highwater.common import rand
from highwater.sim.clock import run_sim


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "sim: runs on the virtual-time event loop")


@pytest.fixture
def sim():
    """Run an async body in simulated time.

        def test_x(sim):
            async def body():
                ...
            sim(body, seed=7)
    """

    def _run(body: Callable[[], Any], *, seed: int = 0xC0FFEE, budget: float = 400.0):
        assert inspect.iscoroutinefunction(body), "sim() takes an async function"
        return run_sim(body(), seed=seed, budget=budget)

    return _run


@pytest.fixture(autouse=True)
def _deterministic_seed():
    """Every test starts from the same random state unless it says otherwise."""
    rand.set_root_seed(0xC0FFEE)
    yield
    rand.set_root_seed(0xC0FFEE)
