"""
Tests for bounded retry behaviour when an exchange is unreachable.

The pagination loop in ``_fetch_symbol_data`` caught ``ccxt.NetworkError``,
slept 5 seconds and ``continue``d -- without advancing ``since`` and without
counting attempts. A *persistent* network error therefore retried forever. This
is not hypothetical: Binance answers 451 from restricted regions, which ccxt
surfaces as a NetworkError, and a single request logged 67 identical retries
before being killed manually.

Because the defect is non-termination, every test here is wrapped in a wall
clock guard. A hanging implementation fails instead of hanging the suite.

Author: BBG-Credit-Momentum
License: MIT
"""

import time
import unittest
from datetime import datetime
from unittest.mock import MagicMock

import ccxt
import pandas as pd

from _crypto_data_sources import CryptoExchangeDataSource


def _source(exchange, **kwargs):
    """
    Build a data source around a stub exchange without running __init__.

    __init__ constructs a real ccxt client and validates the timeframe against
    it, neither of which is under test here. _fetch_symbol_data only reads the
    attributes set below, so bypassing __init__ keeps the test focused on the
    retry loop.
    """
    source = CryptoExchangeDataSource.__new__(CryptoExchangeDataSource)
    source.exchange_id = "binance"
    source.symbols = ["BTC/USDT"]
    source.timeframe = "1d"
    source.start_date = datetime(2024, 1, 1)
    source.end_date = datetime(2024, 3, 1)
    source.limit = 1000
    source.exchange = exchange
    for key, value in kwargs.items():
        setattr(source, key, value)
    return source


def _always_failing_exchange():
    exchange = MagicMock()
    exchange.rateLimit = 0
    exchange.fetch_ohlcv.side_effect = ccxt.NetworkError(
        "binance GET https://api.binance.com/api/v3/exchangeInfo 451"
    )
    return exchange


class TestBoundedRetries(unittest.TestCase):
    TIME_BUDGET = 60.0

    def test_persistent_network_error_gives_up_instead_of_looping_forever(self):
        exchange = _always_failing_exchange()
        source = _source(exchange, retry_backoff=0)

        started = time.monotonic()
        result = source._fetch_symbol_data("BTC/USDT")
        elapsed = time.monotonic() - started

        self.assertLess(
            elapsed,
            self.TIME_BUDGET,
            "_fetch_symbol_data did not terminate; the retry loop is unbounded",
        )
        self.assertIsNone(result)

    def test_retry_attempts_are_capped(self):
        exchange = _always_failing_exchange()
        source = _source(exchange, max_retries=3, retry_backoff=0)

        source._fetch_symbol_data("BTC/USDT")

        # One initial attempt plus at most max_retries retries.
        self.assertLessEqual(exchange.fetch_ohlcv.call_count, 4)
        self.assertGreaterEqual(exchange.fetch_ohlcv.call_count, 2)

    def test_a_transient_error_still_recovers(self):
        # Fail once, then serve one page, then signal end-of-data.
        candle = [1704067200000, 1.0, 2.0, 0.5, 1.5, 100.0]
        exchange = MagicMock()
        exchange.rateLimit = 0
        exchange.fetch_ohlcv.side_effect = [
            ccxt.NetworkError("transient blip"),
            [candle],
            [],
        ]
        source = _source(exchange, retry_backoff=0)

        started = time.monotonic()
        result = source._fetch_symbol_data("BTC/USDT")
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, self.TIME_BUDGET)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty)

    def test_retry_budget_resets_after_a_successful_page(self):
        # A long pagination run that hits an occasional blip must not exhaust a
        # global budget and abort halfway through.
        candles = [
            [1704067200000 + i * 86_400_000, 1.0, 2.0, 0.5, 1.5, 100.0]
            for i in range(4)
        ]
        exchange = MagicMock()
        exchange.rateLimit = 0
        exchange.fetch_ohlcv.side_effect = [
            [candles[0]],
            ccxt.NetworkError("blip 1"),
            [candles[1]],
            ccxt.NetworkError("blip 2"),
            [candles[2]],
            ccxt.NetworkError("blip 3"),
            [candles[3]],
            [],
        ]
        source = _source(exchange, max_retries=2, retry_backoff=0)

        result = source._fetch_symbol_data("BTC/USDT")

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 4)

    def test_exchange_error_still_stops_immediately(self):
        exchange = MagicMock()
        exchange.rateLimit = 0
        exchange.fetch_ohlcv.side_effect = ccxt.ExchangeError("bad symbol")
        source = _source(exchange, retry_backoff=0)

        result = source._fetch_symbol_data("BTC/USDT")

        self.assertIsNone(result)
        self.assertEqual(exchange.fetch_ohlcv.call_count, 1)


if __name__ == "__main__":
    unittest.main()
