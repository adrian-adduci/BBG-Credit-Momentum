"""
Blockchain Data Sources Module

This module provides data sources for on-chain blockchain metrics and analytics.
It supports multiple providers including Glassnode, CoinMetrics, and custom
blockchain query interfaces for LAN-based databases.

Providers supported:
- Glassnode: On-chain analytics (MVRV, NVT, active addresses, etc.)
- CoinMetrics: Network data (hash rate, difficulty, fees, etc.)
- Custom LAN Database: Query local PostgreSQL for cached blockchain data

Author: BBG-Credit-Momentum Team
License: MIT
"""

import logging
import os
import pathlib
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import time

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Add parent directory to path to import DataSource
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.absolute()))
from _data_sources import DataSource

# Set up structured logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
path = pathlib.Path(__file__).parent.parent.absolute()
handler = logging.FileHandler(path / "logs" / "_blockchain_data_sources.log")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class BlockchainProvider(ABC):
    """
    Abstract base class for blockchain data providers.

    All providers must implement methods to fetch on-chain metrics
    from their respective APIs and return standardized DataFrames.
    """

    @abstractmethod
    def fetch_metric(
        self,
        asset: str,
        metric: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Fetch a specific metric for an asset.

        Args:
            asset: Asset symbol (e.g., "BTC", "ETH")
            metric: Metric name (e.g., "mvrv", "nvt")
            start_date: Start date for data
            end_date: End date for data

        Returns:
            DataFrame with columns: [Dates, {metric_name}]
        """
        pass

    @abstractmethod
    def get_available_metrics(self, asset: str) -> List[str]:
        """
        Get list of available metrics for an asset.

        Args:
            asset: Asset symbol

        Returns:
            List of metric names
        """
        pass


class GlassnodeProvider(BlockchainProvider):
    """
    Glassnode on-chain analytics provider.

    Glassnode provides comprehensive on-chain metrics for Bitcoin, Ethereum,
    and other major cryptocurrencies. Metrics include:
    - MVRV (Market Value to Realized Value)
    - NVT (Network Value to Transactions)
    - Active Addresses
    - Exchange Netflow
    - Supply metrics (held by whales, exchanges, etc.)
    - Price models (stock-to-flow, etc.)

    API Documentation: https://docs.glassnode.com/

    Args:
        api_key: Glassnode API key (get from environment if not provided)
        rate_limit_calls: Max API calls per minute (default: 60 for free tier)
        timeout: Request timeout in seconds (default: 30)

    Example:
        >>> provider = GlassnodeProvider(api_key="your_api_key")
        >>> df = provider.fetch_metric("BTC", "mvrv", start_date, end_date)
    """

    BASE_URL = "https://api.glassnode.com/v1/metrics"

    # Mapping of common metric names to Glassnode API endpoints
    METRIC_MAP = {
        "mvrv": "market/mvrv",
        "nvt": "indicators/nvt",
        "nvt_signal": "indicators/nvts",
        "active_addresses": "addresses/active_count",
        "new_addresses": "addresses/new_non_zero_count",
        "exchange_netflow": "transactions/transfers_volume_exchanges_net",
        "exchange_balance": "distribution/balance_exchanges",
        "whale_balance": "distribution/balance_1k_10k",
        "supply_held_1y": "supply/profit_relative",
        "sopr": "indicators/sopr",
        "realized_cap": "market/marketcap_realized_usd",
        "stock_to_flow": "indicators/stock_to_flow_ratio",
        "puell_multiple": "indicators/puell_multiple",
        "hash_rate": "mining/hash_rate_mean",
        "difficulty": "mining/difficulty_latest",
        "fees_total": "fees/volume_sum",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        rate_limit_calls: int = 60,
        timeout: int = 30,
        debug: bool = False,
    ):
        """Initialize Glassnode provider with API credentials."""
        self.api_key = api_key or os.getenv("GLASSNODE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Glassnode API key required. Set GLASSNODE_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.rate_limit_calls = rate_limit_calls
        self.timeout = timeout
        self.debug = debug

        # Rate limiting state
        self.call_times = []  # Track API call timestamps for rate limiting

        # Setup requests session with retry logic
        self.session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

        logger.info(f"GlassnodeProvider initialized (rate limit: {rate_limit_calls} calls/min)")

        if self.debug:
            logger.setLevel(logging.DEBUG)

    def _rate_limit(self):
        """Implement rate limiting to avoid API throttling."""
        now = time.time()
        # Remove calls older than 1 minute
        self.call_times = [t for t in self.call_times if now - t < 60]

        # If we've hit the rate limit, wait
        if len(self.call_times) >= self.rate_limit_calls:
            sleep_time = 60 - (now - self.call_times[0]) + 1
            logger.warning(f"Rate limit reached. Sleeping for {sleep_time:.1f}s")
            time.sleep(sleep_time)
            self.call_times = []

        self.call_times.append(now)

    def fetch_metric(
        self,
        asset: str,
        metric: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "24h",
    ) -> pd.DataFrame:
        """
        Fetch a specific on-chain metric from Glassnode.

        Args:
            asset: Asset symbol (e.g., "BTC", "ETH")
            metric: Metric name (must be in METRIC_MAP)
            start_date: Start date for data
            end_date: End date for data
            interval: Data interval ("24h", "1h", "10m", etc.)

        Returns:
            DataFrame with columns: [Dates, {asset}_{metric}]

        Raises:
            ValueError: If metric not supported or API error
            requests.RequestException: If network error
        """
        # Validate metric
        if metric not in self.METRIC_MAP:
            raise ValueError(
                f"Metric '{metric}' not supported. Available: {list(self.METRIC_MAP.keys())}"
            )

        endpoint = self.METRIC_MAP[metric]
        logger.info(f"Fetching {metric} for {asset} from {start_date} to {end_date}")

        # Apply rate limiting
        self._rate_limit()

        # Prepare request parameters
        params = {
            "a": asset.upper(),  # Asset symbol
            "s": int(start_date.timestamp()),  # Start timestamp
            "u": int(end_date.timestamp()),  # End timestamp
            "i": interval,  # Interval
            "api_key": self.api_key,
        }

        url = f"{self.BASE_URL}/{endpoint}"

        try:
            # Make API request
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()

            # Parse JSON response
            data = response.json()

            if not data:
                logger.warning(f"No data returned for {metric} on {asset}")
                return pd.DataFrame(columns=["Dates", f"{asset}_{metric}"])

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Glassnode returns: [{"t": timestamp, "v": value}, ...]
            df["Dates"] = pd.to_datetime(df["t"], unit="s")
            df[f"{asset}_{metric}"] = pd.to_numeric(df["v"], errors="coerce")

            # Select and rename columns
            df = df[["Dates", f"{asset}_{metric}"]]

            # Remove any NaN values
            df = df.dropna()

            logger.info(f"Fetched {len(df)} data points for {asset}_{metric}")

            if self.debug:
                logger.debug(f"  Date range: {df['Dates'].min()} to {df['Dates'].max()}")
                logger.debug(f"  Value range: [{df[f'{asset}_{metric}'].min():.2f}, {df[f'{asset}_{metric}'].max():.2f}]")
                logger.debug(f"  Mean: {df[f'{asset}_{metric}'].mean():.2f}")

            return df

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise ValueError("Invalid Glassnode API key")
            elif e.response.status_code == 404:
                raise ValueError(f"Metric {metric} not found for asset {asset}")
            else:
                logger.error(f"HTTP error fetching {metric}: {e}")
                raise

        except Exception as e:
            logger.error(f"Error fetching {metric} for {asset}: {str(e)}")
            raise

    def get_available_metrics(self, asset: str) -> List[str]:
        """
        Get list of available metrics.

        Args:
            asset: Asset symbol (currently unused, but kept for interface compatibility)

        Returns:
            List of metric names
        """
        return list(self.METRIC_MAP.keys())


class CoinMetricsProvider(BlockchainProvider):
    """
    CoinMetrics network data provider.

    CoinMetrics provides blockchain network metrics including:
    - Hash rate and mining difficulty
    - Transaction counts and fees
    - Active addresses and new addresses
    - Supply metrics
    - Market data

    API Documentation: https://docs.coinmetrics.io/

    Args:
        api_key: CoinMetrics API key (optional for free tier, limited to 1000 requests/day)
        rate_limit_calls: Max API calls per minute (default: 10 for free tier)
        timeout: Request timeout in seconds (default: 30)

    Example:
        >>> provider = CoinMetricsProvider(api_key="your_api_key")
        >>> df = provider.fetch_metric("btc", "HashRate", start_date, end_date)
    """

    BASE_URL = "https://api.coinmetrics.io/v4"

    # Mapping of common metric names to CoinMetrics API metric names
    METRIC_MAP = {
        "hash_rate": "HashRate",
        "difficulty": "DiffMean",
        "active_addresses": "AdrActCnt",
        "new_addresses": "AdrActCnt",  # CoinMetrics doesn't distinguish new/active
        "transaction_count": "TxCnt",
        "fees_total": "FeeTotUSD",
        "fees_mean": "FeeMeanUSD",
        "supply_total": "SplyCur",
        "price_usd": "PriceUSD",
        "market_cap": "CapMrktCurUSD",
        "realized_cap": "CapRealUSD",
        "nvt": "NVTAdj",
        "sopr": "SOPR",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        rate_limit_calls: int = 10,
        timeout: int = 30,
        debug: bool = False,
    ):
        """Initialize CoinMetrics provider with API credentials."""
        self.api_key = api_key or os.getenv("COINMETRICS_API_KEY")
        # API key is optional for free tier
        self.rate_limit_calls = rate_limit_calls
        self.timeout = timeout
        self.debug = debug

        # Rate limiting state
        self.call_times = []

        # Setup requests session with retry logic
        self.session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

        logger.info(f"CoinMetricsProvider initialized (rate limit: {rate_limit_calls} calls/min)")

        if self.debug:
            logger.setLevel(logging.DEBUG)

    def _rate_limit(self):
        """Implement rate limiting to avoid API throttling."""
        now = time.time()
        self.call_times = [t for t in self.call_times if now - t < 60]

        if len(self.call_times) >= self.rate_limit_calls:
            sleep_time = 60 - (now - self.call_times[0]) + 1
            logger.warning(f"Rate limit reached. Sleeping for {sleep_time:.1f}s")
            time.sleep(sleep_time)
            self.call_times = []

        self.call_times.append(now)

    def fetch_metric(
        self,
        asset: str,
        metric: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Fetch a specific network metric from CoinMetrics.

        Args:
            asset: Asset symbol (e.g., "btc", "eth")
            metric: Metric name (must be in METRIC_MAP)
            start_date: Start date for data
            end_date: End date for data

        Returns:
            DataFrame with columns: [Dates, {asset}_{metric}]

        Raises:
            ValueError: If metric not supported or API error
            requests.RequestException: If network error
        """
        # Validate metric
        if metric not in self.METRIC_MAP:
            raise ValueError(
                f"Metric '{metric}' not supported. Available: {list(self.METRIC_MAP.keys())}"
            )

        cm_metric = self.METRIC_MAP[metric]
        logger.info(f"Fetching {metric} for {asset} from {start_date} to {end_date}")

        # Apply rate limiting
        self._rate_limit()

        # Prepare request parameters
        params = {
            "assets": asset.lower(),
            "metrics": cm_metric,
            "start_time": start_date.strftime("%Y-%m-%d"),
            "end_time": end_date.strftime("%Y-%m-%d"),
            "frequency": "1d",  # Daily data
        }

        if self.api_key:
            params["api_key"] = self.api_key

        url = f"{self.BASE_URL}/timeseries/asset-metrics"

        try:
            # Make API request
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()

            # Parse JSON response
            result = response.json()

            if "data" not in result or not result["data"]:
                logger.warning(f"No data returned for {metric} on {asset}")
                return pd.DataFrame(columns=["Dates", f"{asset}_{metric}"])

            # Convert to DataFrame
            data_list = result["data"]
            df = pd.DataFrame(data_list)

            # CoinMetrics returns: [{"asset": "btc", "time": "2024-01-01", "HashRate": 123}, ...]
            df["Dates"] = pd.to_datetime(df["time"])
            df[f"{asset}_{metric}"] = pd.to_numeric(df[cm_metric], errors="coerce")

            # Select and rename columns
            df = df[["Dates", f"{asset}_{metric}"]]

            # Remove any NaN values
            df = df.dropna()

            logger.info(f"Fetched {len(df)} data points for {asset}_{metric}")

            if self.debug:
                logger.debug(f"  Date range: {df['Dates'].min()} to {df['Dates'].max()}")
                logger.debug(f"  Value range: [{df[f'{asset}_{metric}'].min():.2e}, {df[f'{asset}_{metric}'].max():.2e}]")

            return df

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise ValueError("Invalid CoinMetrics API key")
            else:
                logger.error(f"HTTP error fetching {metric}: {e}")
                raise

        except Exception as e:
            logger.error(f"Error fetching {metric} for {asset}: {str(e)}")
            raise

    def get_available_metrics(self, asset: str) -> List[str]:
        """
        Get list of available metrics.

        Args:
            asset: Asset symbol

        Returns:
            List of metric names
        """
        return list(self.METRIC_MAP.keys())


class BlockchainDataSource(DataSource):
    """
    Unified blockchain data source that supports multiple providers.

    This class provides a unified interface for fetching on-chain metrics
    from different providers (Glassnode, CoinMetrics, etc.). It handles:
    - Multiple assets and metrics
    - Data merging from multiple providers
    - Standardized DataFrame output
    - Error handling and logging

    Args:
        provider: Provider name ("glassnode" or "coinmetrics")
        assets: List of asset symbols (e.g., ["BTC", "ETH"])
        metrics: List of metric names (e.g., ["mvrv", "nvt"])
        start_date: Start date for data
        end_date: End date for data
        api_key: API key for the provider (optional, uses environment variable)

    Example:
        >>> source = BlockchainDataSource(
        ...     provider="glassnode",
        ...     assets=["BTC", "ETH"],
        ...     metrics=["mvrv", "nvt", "active_addresses"],
        ...     start_date=datetime(2024, 1, 1),
        ...     end_date=datetime.now(),
        ... )
        >>> df = source.load_data()
        >>> print(df.columns)
        # ['Dates', 'BTC_mvrv', 'BTC_nvt', 'BTC_active_addresses', 'ETH_mvrv', ...]
    """

    def __init__(
        self,
        provider: str,
        assets: List[str],
        metrics: List[str],
        start_date: datetime,
        end_date: datetime,
        api_key: Optional[str] = None,
        debug: bool = False,
    ):
        """Initialize blockchain data source."""
        self.provider_name = provider.lower()
        self.assets = assets
        self.metrics = metrics
        self.start_date = start_date
        self.end_date = end_date
        self.api_key = api_key
        self.debug = debug

        # Initialize the appropriate provider
        if self.provider_name == "glassnode":
            self.provider = GlassnodeProvider(api_key=api_key, debug=debug)
        elif self.provider_name == "coinmetrics":
            self.provider = CoinMetricsProvider(api_key=api_key, debug=debug)
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'glassnode' or 'coinmetrics'")

        logger.info(
            f"BlockchainDataSource initialized: provider={provider}, "
            f"assets={assets}, metrics={metrics}"
        )

    def load_data(self) -> pd.DataFrame:
        """
        Load blockchain metrics from the provider.

        Fetches all requested metrics for all assets and merges them into
        a single DataFrame with 'Dates' column and one column per asset-metric pair.

        Returns:
            DataFrame with columns: [Dates, {asset}_{metric}, ...]

        Raises:
            ValueError: If no data could be fetched
            Exception: If API errors occur
        """
        logger.info(
            f"Loading blockchain data for {len(self.assets)} assets "
            f"and {len(self.metrics)} metrics"
        )

        all_dataframes = []

        for asset in self.assets:
            for metric in self.metrics:
                try:
                    # Fetch metric data
                    df = self.provider.fetch_metric(
                        asset=asset,
                        metric=metric,
                        start_date=self.start_date,
                        end_date=self.end_date,
                    )

                    if not df.empty:
                        all_dataframes.append(df)
                    else:
                        logger.warning(f"No data returned for {asset}_{metric}")

                except Exception as e:
                    logger.error(f"Failed to fetch {asset}_{metric}: {str(e)}")
                    # Continue with other metrics instead of failing completely

        if not all_dataframes:
            raise ValueError("No data could be fetched from provider")

        # Merge all DataFrames on 'Dates'
        logger.info(f"Merging {len(all_dataframes)} metric DataFrames")
        merged_df = all_dataframes[0]

        for df in all_dataframes[1:]:
            merged_df = merged_df.merge(df, on="Dates", how="outer")

        # Sort by date
        merged_df = merged_df.sort_values("Dates").reset_index(drop=True)

        logger.info(
            f"Blockchain data loaded: {len(merged_df)} rows, "
            f"{len(merged_df.columns) - 1} metrics"
        )

        if self.debug:
            logger.debug(f"  Date range: {merged_df['Dates'].min()} to {merged_df['Dates'].max()}")
            logger.debug(f"  Columns: {list(merged_df.columns)}")
            logger.debug(f"  Missing values: {merged_df.isnull().sum().sum()}")

        # Validate schema
        self.validate_schema(merged_df)

        return merged_df
