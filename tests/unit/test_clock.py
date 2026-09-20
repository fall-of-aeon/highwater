"""The virtual-time event loop.

If this breaks, every simulation test silently becomes a real-time test and the
suite goes from seconds to hours -- or hangs. Worth asserting directly.
"""

from __future__ import annotations

import asyncio
import time

from highwater.common import clock
from highwater.sim.clock import SimTimeout, run_sim


def test_sleeps_cost_no_real_time():
    async def body():
        await asyncio.sleep(3600)
        return clock.monotonic()

    started = time.perf_counter()
    simulated = run_sim(body(), seed=1, budget=10_000)
    real = time.perf_counter() - started
    assert simulated >= 3600, "the loop did not advance virtual time"
    assert real < 1.0, f"an hour of simulated sleep took {real:.2f}s of real time"


def test_timers_fire_in_order():
    async def body():
        order: list[str] = []

        async def at(delay: float, name: str) -> None:
            await asyncio.sleep(delay)
            order.append(name)

        await asyncio.gather(at(30, "c"), at(1, "a"), at(10, "b"))
        return order

    assert run_sim(body(), seed=1, budget=100) == ["a", "b", "c"]


def test_wait_for_times_out_in_virtual_time():
    async def body():
        started = clock.monotonic()
        try:
            await asyncio.wait_for(asyncio.sleep(10_000), timeout=5.0)
        except TimeoutError:
            return clock.monotonic() - started
        raise AssertionError("wait_for did not time out")

    assert abs(run_sim(body(), seed=1, budget=100) - 5.0) < 0.01


def test_wall_clock_advances_with_virtual_time():
    async def body():
        before = clock.now()
        await asyncio.sleep(120)
        return clock.now() - before

    assert abs(run_sim(body(), seed=1, budget=1000) - 120) < 0.01


def test_budget_is_enforced():
    """A simulation that cannot make progress must fail loudly rather than
    spinning virtual time forever."""

    async def body():
        await asyncio.sleep(10_000)

    try:
        run_sim(body(), seed=1, budget=5.0)
    except SimTimeout:
        return
    raise AssertionError("the virtual-time budget was not enforced")


def test_runs_are_reproducible_from_a_seed():
    from highwater.common.rand import uniform

    async def body():
        out = []
        for _ in range(5):
            await asyncio.sleep(uniform("t", 0.01, 0.1))
            out.append(round(clock.monotonic(), 6))
        return out

    a = run_sim(body(), seed=4242, budget=100)
    b = run_sim(body(), seed=4242, budget=100)
    c = run_sim(body(), seed=9999, budget=100)
    assert a == b, "the same seed produced a different run"
    assert a != c, "different seeds produced identical runs"
