"""
Tests that a slow request cannot starve the whole API.

Every handler in api.py was declared ``async def`` while doing blocking work:
synchronous ccxt HTTP, ``time.sleep`` in the retry path, and XGBoost fitting.
A blocking call inside a coroutine occupies the event loop thread, so the
server could serve exactly one request at a time and a hanging dependency took
down the entire process. Observed directly: while POST /api/train was stuck
retrying an unreachable exchange, GET / stopped responding too.

FastAPI runs a plain ``def`` handler in a threadpool and an ``async def``
handler on the event loop, so the fix is to declare blocking handlers ``def``.
These tests pin the resulting behaviour rather than the declaration.

TestClient cannot express this: it issues requests sequentially. The ASGI app
is driven directly so two requests really are in flight at once.

Author: BBG-Credit-Momentum
License: MIT
"""

import asyncio
import inspect
import time
import unittest
from unittest.mock import patch

import httpx

import api

SLOW_SECONDS = 3.0
#: A mixed-portfolio request loads one crypto and one credit security, so the
#: stub stalls twice per request. One request therefore costs ~2x SLOW_SECONDS
#: no matter how well the server overlaps work.
STALLS_PER_REQUEST = 2
ONE_REQUEST = SLOW_SECONDS * STALLS_PER_REQUEST
#: GET / must return long before the slow request finishes.
RESPONSIVE_BUDGET = SLOW_SECONDS * 0.6


def _blocking_stall(*args, **kwargs):
    """Stand in for slow synchronous I/O, then fail like a real outage."""
    time.sleep(SLOW_SECONDS)
    raise RuntimeError("simulated upstream outage")


class TestApiStaysResponsive(unittest.TestCase):
    def _run(self, coro):
        return asyncio.run(coro)

    def test_root_finishes_before_a_concurrent_slow_request(self):
        """GET / must complete while the slow POST is still working.

        Completion *order* is the reliable signal here. Measuring a stopwatch
        around GET / is not: if the loop is blocked, even `await
        asyncio.sleep()` overshoots by the full blocking duration, so the
        stopwatch starts after the block has already cleared and the test
        passes for the wrong reason.
        """
        async def scenario():
            transport = httpx.ASGITransport(app=api.app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                marks = {}

                async def slow():
                    try:
                        await client.post(
                            "/api/mixed/train",
                            json=_mixed_payload(),
                            timeout=SLOW_SECONDS * 6,
                        )
                    except Exception:
                        pass
                    marks["slow"] = time.monotonic()

                async def fast():
                    response = await client.get("/", timeout=SLOW_SECONDS * 6)
                    marks["fast"] = time.monotonic()
                    return response

                slow_task = asyncio.create_task(slow())
                fast_task = asyncio.create_task(fast())
                response, _ = await asyncio.gather(fast_task, slow_task)
                return response, marks

        with patch(
            "data_sources.MixedPortfolioDataSource._load_security_data",
            side_effect=_blocking_stall,
        ):
            response, marks = self._run(scenario())

        self.assertEqual(response.status_code, 200)
        self.assertLess(
            marks["fast"],
            marks["slow"] - RESPONSIVE_BUDGET,
            "GET / did not finish until the blocking request released the "
            "event loop thread",
        )

    def test_two_slow_requests_overlap_rather_than_queue(self):
        async def scenario():
            transport = httpx.ASGITransport(app=api.app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                started = time.monotonic()
                await asyncio.gather(
                    client.post(
                        "/api/mixed/train",
                        json=_mixed_payload(),
                        timeout=SLOW_SECONDS * 6,
                    ),
                    client.post(
                        "/api/mixed/train",
                        json=_mixed_payload(),
                        timeout=SLOW_SECONDS * 6,
                    ),
                    return_exceptions=True,
                )
                return time.monotonic() - started

        with patch(
            "data_sources.MixedPortfolioDataSource._load_security_data",
            side_effect=_blocking_stall,
        ):
            elapsed = self._run(scenario())

        # Serialised on the event loop the two requests cost ~2x ONE_REQUEST;
        # run in a threadpool they overlap and the total stays near 1x.
        self.assertLess(
            elapsed,
            ONE_REQUEST * 1.5,
            f"two blocking requests took {elapsed:.2f}s, but one alone costs "
            f"~{ONE_REQUEST:.1f}s -- they serialised instead of overlapping",
        )

    def test_blocking_handlers_are_not_declared_async(self):
        """A guard so the defect cannot be reintroduced by a later edit.

        These handlers all perform synchronous network I/O or model fitting.
        Declaring any of them `async def` puts that work back on the event
        loop thread.
        """
        blocking_handlers = [
            "train_model",
            "predict",
            "backtest_strategy",
            "train_mixed_portfolio",
            "get_cross_asset_analysis",
        ]

        offenders = [
            name
            for name in blocking_handlers
            if inspect.iscoroutinefunction(getattr(api, name))
        ]

        self.assertEqual(
            offenders,
            [],
            f"blocking handlers declared async def: {offenders}",
        )


def _mixed_payload():
    return {
        "crypto_exchange": "binance",
        "crypto_symbols": ["BTC/USDT"],
        "crypto_timeframe": "1h",
        "bloomberg_securities": ["LF98TRUU Index"],
        "bloomberg_fields": ["OAS"],
        "bloomberg_source": "api",
        "start_date": "2024-01-01",
        "target_column": "BTC_USDT_close",
        "model_type": "XGBoost",
        "crypto_features": False,
        "cross_asset_features": False,
    }


if __name__ == "__main__":
    unittest.main()
