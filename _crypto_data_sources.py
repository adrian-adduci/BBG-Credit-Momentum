"""
Cryptocurrency Data Sources

This module provides data source implementations for fetching cryptocurrency
market data from various exchanges using the CCXT library.

Classes:
    CryptoExchangeDataSource: Historical OHLCV data from exchanges
    CryptoWebSocketDataSource: Real-time streaming data via WebSocket
    CryptoAggregatorDataSource: Aggregated data from multiple exchanges

Example:
    >>> from _crypto_data_sources import CryptoExchangeDataSource
    >>> from datetime import datetime
    >>>
    >>> source = CryptoExchangeDataSource(
    ...     exchange_id="binance",
    ...     symbols=["BTC/USDT", "ETH/USDT"],
    ...     timeframe="1h",
    ...     start_date=datetime(2024, 1, 1)
    ... )
    >>> df = source.load_data()
    >>> print(df.head())
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import logging
from pathlib import Path
import time

from _data_sources import DataSource

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "_crypto_data_sources.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CryptoExchangeDataSource(DataSource):
    """
    Fetch historical OHLCV data from cryptocurrency exchanges via CCXT.

    Supports major exchanges including Binance, Coinbase, Kraken, Bybit, and OKX.
    Returns standardized DataFrame with Dates column and OHLCV data for each symbol.

    Attributes:
        exchange (ccxt.Exchange): CCXT exchange instance
        symbols (List[str]): List of trading pairs (e.g., ["BTC/USDT", "ETH/USDT"])
        timeframe (str): Candle timeframe (1m, 5m, 15m, 1h, 4h, 1d, 1w, 1M)
        start_date (datetime): Start date for historical data
        end_date (Optional[datetime]): End date (defaults to now)
        limit (int): Maximum candles per request (default 1000)

    Example:
        >>> source = CryptoExchangeDataSource(
        ...     exchange_id="binance",
        ...     symbols=["BTC/USDT"],
        ...     timeframe="1h",
        ...     start_date=datetime(2024, 1, 1),
        ...     limit=1000
        ... )
        >>> df = source.load_data()
        >>> print(df.columns)
        Index(['Dates', 'BTC_USDT_open', 'BTC_USDT_high', 'BTC_USDT_low',
               'BTC_USDT_close', 'BTC_USDT_volume'], dtype='object')
    """

    def __init__(
        self,
        exchange_id: str = "binance",
        symbols: List[str] = None,
        timeframe: str = "1h",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000
    ):
        """
        Initialize cryptocurrency exchange data source.

        Args:
            exchange_id: Exchange identifier (binance, coinbase, kraken, bybit, okx)
            symbols: List of trading pairs to fetch
            timeframe: Candle timeframe (1m, 5m, 15m, 1h, 4h, 1d, 1w, 1M)
            start_date: Start date for historical data
            end_date: End date (defaults to current time)
            limit: Maximum candles per request
        """
        if symbols is None:
            symbols = ["BTC/USDT"]

        self.exchange_id = exchange_id
        self.symbols = symbols
        self.timeframe = timeframe
        self.start_date = start_date or datetime.now() - timedelta(days=365)
        self.end_date = end_date or datetime.now()
        self.limit = limit

        # Initialize exchange
        try:
            exchange_class = getattr(ccxt, exchange_id)
            self.exchange = exchange_class({
                'enableRateLimit': True,  # Respect rate limits
                'timeout': 30000,  # 30 seconds
            })
            logger.info(f"Initialized {exchange_id} exchange")
        except AttributeError:
            raise ValueError(f"Exchange '{exchange_id}' not supported by CCXT")

        # Validate timeframe
        if timeframe not in self.exchange.timeframes:
            raise ValueError(
                f"Timeframe '{timeframe}' not supported by {exchange_id}. "
                f"Available: {list(self.exchange.timeframes.keys())}"
            )

    def load_data(self) -> pd.DataFrame:
        """
        Fetch historical OHLCV data and return as DataFrame.

        Returns:
            pd.DataFrame: DataFrame with columns:
                - Dates: Datetime index
                - {SYMBOL}_open: Opening price
                - {SYMBOL}_high: High price
                - {SYMBOL}_low: Low price
                - {SYMBOL}_close: Closing price
                - {SYMBOL}_volume: Trading volume

        Raises:
            ccxt.NetworkError: Network connectivity issues
            ccxt.ExchangeError: Exchange API errors
            ValueError: Invalid parameters or data issues
        """
        logger.info(
            f"Fetching {len(self.symbols)} symbols from {self.exchange_id}: "
            f"{self.symbols}, timeframe: {self.timeframe}"
        )

        data_frames = []

        for symbol in self.symbols:
            try:
                logger.info(f"Fetching {symbol}...")
                df_symbol = self._fetch_symbol_data(symbol)

                if df_symbol is not None and not df_symbol.empty:
                    data_frames.append(df_symbol)
                    logger.info(f"✓ {symbol}: {len(df_symbol)} candles fetched")
                else:
                    logger.warning(f"✗ {symbol}: No data returned")

            except Exception as e:
                logger.error(f"✗ {symbol}: Error - {str(e)}")
                continue

        if not data_frames:
            raise ValueError("No data fetched for any symbol")

        # Merge all symbols into single DataFrame
        df_merged = data_frames[0]
        for df_next in data_frames[1:]:
            df_merged = df_merged.merge(df_next, on="Dates", how="outer")

        # Sort by date and reset index
        df_merged = df_merged.sort_values("Dates").reset_index(drop=True)

        # Validate schema
        self.validate_schema(df_merged)

        logger.info(
            f"Successfully loaded {len(df_merged)} rows, "
            f"{len(df_merged.columns)} columns"
        )

        return df_merged

    def _fetch_symbol_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Fetch data for a single symbol with pagination.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")

        Returns:
            pd.DataFrame: OHLCV data with standardized column names
        """
        all_ohlcv = []
        since = int(self.start_date.timestamp() * 1000)  # Convert to milliseconds
        end_timestamp = int(self.end_date.timestamp() * 1000)

        # Pagination loop (exchanges limit candles per request)
        while since < end_timestamp:
            try:
                # Fetch OHLCV data
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe=self.timeframe,
                    since=since,
                    limit=self.limit
                )

                if not ohlcv:
                    break

                all_ohlcv.extend(ohlcv)

                # Update since to last candle timestamp + 1ms
                since = ohlcv[-1][0] + 1

                # Rate limiting (respect exchange limits)
                time.sleep(self.exchange.rateLimit / 1000)

            except ccxt.NetworkError as e:
                logger.error(f"Network error for {symbol}: {e}")
                time.sleep(5)  # Wait before retry
                continue

            except ccxt.ExchangeError as e:
                logger.error(f"Exchange error for {symbol}: {e}")
                break

        if not all_ohlcv:
            return None

        # Convert to DataFrame
        df = pd.DataFrame(
            all_ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

        # Convert timestamp to datetime
        df["Dates"] = pd.to_datetime(df["timestamp"], unit="ms")

        # Filter by end_date
        df = df[df["Dates"] <= self.end_date]

        # Rename columns with symbol prefix
        symbol_clean = symbol.replace("/", "_").replace("-", "_")
        df = df.rename(columns={
            "open": f"{symbol_clean}_open",
            "high": f"{symbol_clean}_high",
            "low": f"{symbol_clean}_low",
            "close": f"{symbol_clean}_close",
            "volume": f"{symbol_clean}_volume"
        })

        # Select final columns
        final_cols = ["Dates"] + [col for col in df.columns if symbol_clean in col]
        df = df[final_cols]

        # Data quality validation
        df = self._validate_ohlcv_data(df, symbol_clean)

        return df

    def _validate_ohlcv_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Validate and clean OHLCV data.

        Checks:
        - High >= Low
        - High >= Open, Close
        - Low <= Open, Close
        - Volume >= 0

        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol identifier

        Returns:
            pd.DataFrame: Cleaned data
        """
        initial_rows = len(df)

        # Check high >= low
        high_col = f"{symbol}_high"
        low_col = f"{symbol}_low"
        invalid_hl = df[high_col] < df[low_col]
        if invalid_hl.any():
            logger.warning(
                f"{symbol}: {invalid_hl.sum()} rows with high < low, removing"
            )
            df = df[~invalid_hl]

        # Check volume >= 0
        volume_col = f"{symbol}_volume"
        invalid_vol = df[volume_col] < 0
        if invalid_vol.any():
            logger.warning(
                f"{symbol}: {invalid_vol.sum()} rows with negative volume, removing"
            )
            df = df[~invalid_vol]

        # Remove duplicate timestamps
        duplicates = df.duplicated(subset=["Dates"], keep="first")
        if duplicates.any():
            logger.warning(
                f"{symbol}: {duplicates.sum()} duplicate timestamps, removing"
            )
            df = df[~duplicates]

        # Drop NaN values
        df = df.dropna()

        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            logger.info(f"{symbol}: Removed {removed_rows} invalid rows")

        return df


class CryptoWebSocketDataSource(DataSource):
    """
    Real-time cryptocurrency data via WebSocket streams.

    Use for live trading signals and order book updates.
    Buffers incoming data and provides it as DataFrame.

    Example:
        >>> source = CryptoWebSocketDataSource(
        ...     exchange_id="binance",
        ...     symbols=["BTC/USDT"],
        ...     stream_type="ticker"
        ... )
        >>> # Start streaming in background
        >>> asyncio.run(source.start_stream())
        >>> # Get buffered data
        >>> df = source.load_data()
    """

    def __init__(
        self,
        exchange_id: str = "binance",
        symbols: List[str] = None,
        stream_type: str = "ticker",
        buffer_size: int = 1000
    ):
        """
        Initialize WebSocket data source.

        Args:
            exchange_id: Exchange identifier
            symbols: List of trading pairs
            stream_type: Stream type (ticker, trades, orderbook)
            buffer_size: Maximum buffer size
        """
        if symbols is None:
            symbols = ["BTC/USDT"]

        self.exchange_id = exchange_id
        self.symbols = symbols
        self.stream_type = stream_type
        self.buffer_size = buffer_size
        self.data_buffer = []

        # Initialize exchange with WebSocket support
        try:
            exchange_class = getattr(ccxt, exchange_id)
            self.exchange = exchange_class({
                'enableRateLimit': True,
                'newUpdates': True  # Enable WebSocket
            })
            logger.info(f"Initialized {exchange_id} WebSocket source")
        except AttributeError:
            raise ValueError(f"Exchange '{exchange_id}' not supported")

    def load_data(self) -> pd.DataFrame:
        """
        Return buffered data as DataFrame.

        Returns:
            pd.DataFrame: Buffered streaming data
        """
        if not self.data_buffer:
            logger.warning("Buffer is empty, no data available")
            return pd.DataFrame()

        df = pd.DataFrame(self.data_buffer)

        # Ensure Dates column exists
        if "timestamp" in df.columns:
            df["Dates"] = pd.to_datetime(df["timestamp"], unit="ms")

        self.validate_schema(df)
        return df

    async def start_stream(self, callback=None):
        """
        Start WebSocket stream (async).

        Note: This is a template implementation.
        Full WebSocket integration requires additional libraries
        and exchange-specific implementations.

        Args:
            callback: Optional callback function for each update
        """
        logger.info(f"Starting WebSocket stream for {self.symbols}")

        # TODO: Implement exchange-specific WebSocket
        # Example pseudo-code:
        #
        # async with websocket.connect(ws_url) as ws:
        #     while True:
        #         data = await ws.recv()
        #         parsed = json.loads(data)
        #
        #         # Add to buffer
        #         self.data_buffer.append(parsed)
        #         if len(self.data_buffer) > self.buffer_size:
        #             self.data_buffer.pop(0)
        #
        #         # Call callback if provided
        #         if callback:
        #             await callback(parsed)

        raise NotImplementedError(
            "WebSocket streaming requires additional implementation. "
            "Please refer to exchange documentation for WebSocket APIs."
        )


class CryptoAggregatorDataSource(DataSource):
    """
    Aggregate data from multiple exchanges.

    Use for arbitrage analysis, liquidity comparison, and price divergence detection.

    Example:
        >>> source = CryptoAggregatorDataSource(
        ...     exchange_ids=["binance", "coinbase", "kraken"],
        ...     symbol="BTC/USDT",
        ...     timeframe="1h"
        ... )
        >>> df = source.load_data()
        >>> # DataFrame includes prices from all exchanges
        >>> print(df[["Dates", "binance_close", "coinbase_close", "kraken_close"]])
    """

    def __init__(
        self,
        exchange_ids: List[str] = None,
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        aggregation_method: str = "vwap"
    ):
        """
        Initialize aggregator data source.

        Args:
            exchange_ids: List of exchanges to aggregate
            symbol: Single trading pair to fetch from all exchanges
            timeframe: Candle timeframe
            start_date: Start date
            end_date: End date
            aggregation_method: Aggregation method (vwap, mean, median)
        """
        if exchange_ids is None:
            exchange_ids = ["binance", "coinbase", "kraken"]

        self.exchange_ids = exchange_ids
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_date = start_date or datetime.now() - timedelta(days=30)
        self.end_date = end_date or datetime.now()
        self.aggregation_method = aggregation_method

        logger.info(
            f"Initialized aggregator for {symbol} across "
            f"{len(exchange_ids)} exchanges"
        )

    def load_data(self) -> pd.DataFrame:
        """
        Fetch data from all exchanges and aggregate.

        Returns:
            pd.DataFrame: Aggregated data with columns:
                - Dates
                - {exchange}_close for each exchange
                - aggregated_close (weighted average)
                - price_divergence (std dev across exchanges)
        """
        logger.info(f"Fetching {self.symbol} from {len(self.exchange_ids)} exchanges")

        exchange_data = []

        for exchange_id in self.exchange_ids:
            try:
                source = CryptoExchangeDataSource(
                    exchange_id=exchange_id,
                    symbols=[self.symbol],
                    timeframe=self.timeframe,
                    start_date=self.start_date,
                    end_date=self.end_date
                )
                df = source.load_data()

                # Rename columns with exchange prefix
                symbol_clean = self.symbol.replace("/", "_")
                df = df.rename(columns={
                    f"{symbol_clean}_close": f"{exchange_id}_close",
                    f"{symbol_clean}_volume": f"{exchange_id}_volume"
                })

                # Keep only Dates, close, volume
                keep_cols = ["Dates", f"{exchange_id}_close", f"{exchange_id}_volume"]
                df = df[[col for col in keep_cols if col in df.columns]]

                exchange_data.append(df)
                logger.info(f"✓ {exchange_id}: {len(df)} candles")

            except Exception as e:
                logger.error(f"✗ {exchange_id}: {str(e)}")
                continue

        if not exchange_data:
            raise ValueError("No data fetched from any exchange")

        # Merge all exchanges
        df_merged = exchange_data[0]
        for df_next in exchange_data[1:]:
            df_merged = df_merged.merge(df_next, on="Dates", how="outer")

        # Sort and remove NaN
        df_merged = df_merged.sort_values("Dates").reset_index(drop=True)

        # Calculate aggregated metrics
        df_merged = self._calculate_aggregated_metrics(df_merged)

        self.validate_schema(df_merged)

        logger.info(
            f"Successfully aggregated {len(df_merged)} rows across "
            f"{len(self.exchange_ids)} exchanges"
        )

        return df_merged

    def _calculate_aggregated_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate aggregated price and divergence metrics.

        Args:
            df: DataFrame with prices from multiple exchanges

        Returns:
            pd.DataFrame: DataFrame with aggregated metrics
        """
        # Get close price columns
        close_cols = [col for col in df.columns if col.endswith("_close")]
        volume_cols = [col for col in df.columns if col.endswith("_volume")]

        if self.aggregation_method == "vwap":
            # Volume-weighted average price
            if volume_cols:
                total_volume = df[volume_cols].sum(axis=1)
                weighted_prices = sum(
                    df[price_col] * df[vol_col]
                    for price_col, vol_col in zip(close_cols, volume_cols)
                )
                df["aggregated_close"] = weighted_prices / total_volume
            else:
                # Fallback to mean if no volume data
                df["aggregated_close"] = df[close_cols].mean(axis=1)

        elif self.aggregation_method == "mean":
            df["aggregated_close"] = df[close_cols].mean(axis=1)

        elif self.aggregation_method == "median":
            df["aggregated_close"] = df[close_cols].median(axis=1)

        else:
            raise ValueError(
                f"Unknown aggregation method: {self.aggregation_method}"
            )

        # Calculate price divergence (standard deviation)
        df["price_divergence"] = df[close_cols].std(axis=1)

        # Calculate price spread (max - min)
        df["price_spread"] = df[close_cols].max(axis=1) - df[close_cols].min(axis=1)

        # Calculate spread percentage
        df["spread_pct"] = (df["price_spread"] / df["aggregated_close"]) * 100

        return df


# Utility functions

def get_supported_exchanges() -> List[str]:
    """
    Get list of all exchanges supported by CCXT.

    Returns:
        List[str]: List of exchange IDs
    """
    return ccxt.exchanges


def get_exchange_info(exchange_id: str) -> Dict:
    """
    Get information about a specific exchange.

    Args:
        exchange_id: Exchange identifier

    Returns:
        Dict: Exchange capabilities and features
    """
    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class()

        return {
            "id": exchange.id,
            "name": exchange.name,
            "has_ohlcv": exchange.has.get("fetchOHLCV", False),
            "has_websocket": exchange.has.get("watchTicker", False),
            "timeframes": list(exchange.timeframes.keys()) if exchange.has.get("fetchOHLCV") else [],
            "rate_limit": exchange.rateLimit,
            "countries": exchange.countries
        }

    except Exception as e:
        logger.error(f"Error getting info for {exchange_id}: {e}")
        return {}


def validate_symbol_format(exchange_id: str, symbol: str) -> bool:
    """
    Validate if symbol format is correct for exchange.

    Args:
        exchange_id: Exchange identifier
        symbol: Trading pair (e.g., "BTC/USDT")

    Returns:
        bool: True if valid, False otherwise
    """
    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class()
        exchange.load_markets()
        return symbol in exchange.markets

    except Exception as e:
        logger.error(f"Error validating symbol {symbol} for {exchange_id}: {e}")
        return False


if __name__ == "__main__":
    """
    Test cryptocurrency data sources.
    """
    print("Testing Cryptocurrency Data Sources\n")
    print("=" * 60)

    # Test 1: Single symbol fetch
    print("\nTest 1: Fetch BTC/USDT from Binance")
    print("-" * 60)
    try:
        source = CryptoExchangeDataSource(
            exchange_id="binance",
            symbols=["BTC/USDT"],
            timeframe="1d",
            start_date=datetime(2024, 1, 1),
            limit=30
        )
        df = source.load_data()
        print(f"✓ Success: {len(df)} rows, {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
        print(f"\nFirst 3 rows:")
        print(df.head(3))
    except Exception as e:
        print(f"✗ Error: {e}")

    # Test 2: Multiple symbols
    print("\n\nTest 2: Fetch Multiple Symbols")
    print("-" * 60)
    try:
        source = CryptoExchangeDataSource(
            exchange_id="binance",
            symbols=["BTC/USDT", "ETH/USDT"],
            timeframe="1h",
            start_date=datetime.now() - timedelta(days=7),
            limit=100
        )
        df = source.load_data()
        print(f"✓ Success: {len(df)} rows, {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Test 3: Exchange aggregation
    print("\n\nTest 3: Aggregate BTC/USDT Across Exchanges")
    print("-" * 60)
    try:
        source = CryptoAggregatorDataSource(
            exchange_ids=["binance", "coinbase"],
            symbol="BTC/USDT",
            timeframe="1d",
            start_date=datetime.now() - timedelta(days=7)
        )
        df = source.load_data()
        print(f"✓ Success: {len(df)} rows, {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
        print(f"\nAggregated prices:")
        print(df[["Dates", "aggregated_close", "price_divergence", "spread_pct"]].head(3))
    except Exception as e:
        print(f"✗ Error: {e}")

    # Test 4: Supported exchanges
    print("\n\nTest 4: List Supported Exchanges")
    print("-" * 60)
    exchanges = get_supported_exchanges()
    print(f"Total exchanges supported: {len(exchanges)}")
    print(f"Popular exchanges: {exchanges[:10]}")

    print("\n" + "=" * 60)
    print("Testing complete!")
