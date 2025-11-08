"""
PostgreSQL Database Client

This module provides a secure PostgreSQL client for storing and retrieving
cryptocurrency trading data, blockchain metrics, and model predictions.

Features:
- Connection pooling for performance
- Parameterized queries to prevent SQL injection
- CRUD operations for all major tables
- Transaction support
- Comprehensive error handling and logging

Security:
- ALL queries use parameterized statements (no string concatenation)
- Input validation on all public methods
- Connection credentials from environment variables only
- No hardcoded passwords or connection strings

Author: BBG-Credit-Momentum Team
License: MIT
"""

import logging
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import pathlib

import psycopg2
from psycopg2 import pool, sql
from psycopg2.extras import RealDictCursor
import pandas as pd

# Set up structured logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
path = pathlib.Path(__file__).parent.parent.absolute()
handler = logging.FileHandler(path / "logs" / "_postgres_client.log")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class PostgreSQLClient:
    """
    PostgreSQL client for cryptocurrency trading database.

    Provides secure, pooled connections to PostgreSQL database with
    CRUD operations for OHLCV data, blockchain metrics, and model predictions.

    All database operations use parameterized queries to prevent SQL injection.

    Args:
        host: Database host (default: from DB_HOST environment variable)
        port: Database port (default: from DB_PORT or 5432)
        database: Database name (default: from DB_NAME)
        user: Database user (default: from DB_USER)
        password: Database password (default: from DB_PASSWORD)
        min_connections: Minimum connections in pool (default: 1)
        max_connections: Maximum connections in pool (default: 10)

    Example:
        >>> client = PostgreSQLClient()
        >>> df = client.get_ohlcv_data("BTC/USDT", "binance", start_date, end_date)
        >>> client.insert_ohlcv_batch(df)
        >>> client.close()
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        min_connections: int = 1,
        max_connections: int = 10,
    ):
        """Initialize PostgreSQL client with connection pool."""
        # Get credentials from environment variables (secure practice)
        self.host = host or os.getenv("DB_HOST", "localhost")
        self.port = port or int(os.getenv("DB_PORT", "5432"))
        self.database = database or os.getenv("DB_NAME", "crypto_trading")
        self.user = user or os.getenv("DB_USER")
        self.password = password or os.getenv("DB_PASSWORD")

        # Validate required credentials
        if not self.user or not self.password:
            raise ValueError(
                "Database credentials required. Set DB_USER and DB_PASSWORD "
                "environment variables or pass user/password parameters."
            )

        # Create connection pool
        try:
            self.pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=min_connections,
                maxconn=max_connections,
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
            )
            logger.info(
                f"PostgreSQL connection pool created: {self.user}@{self.host}:{self.port}/{self.database}"
            )
        except Exception as e:
            logger.error(f"Failed to create connection pool: {str(e)}")
            raise

    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections.

        Automatically returns connection to pool when done.
        Handles transactions with automatic commit/rollback.

        Yields:
            Connection object from the pool

        Example:
            >>> with client.get_connection() as conn:
            ...     with conn.cursor() as cur:
            ...         cur.execute("SELECT * FROM crypto_ohlcv LIMIT 1")
        """
        conn = self.pool.getconn()
        try:
            yield conn
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error, rolling back: {str(e)}")
            raise
        finally:
            self.pool.putconn(conn)

    def close(self):
        """Close all connections in the pool."""
        if self.pool:
            self.pool.closeall()
            logger.info("PostgreSQL connection pool closed")

    # ==========================================================================
    # OHLCV Data Operations
    # ==========================================================================

    def insert_ohlcv(
        self,
        symbol: str,
        exchange: str,
        timestamp: datetime,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> bool:
        """
        Insert a single OHLCV candle into the database.

        Uses parameterized query to prevent SQL injection.
        Handles duplicate timestamps with ON CONFLICT DO NOTHING.

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            exchange: Exchange name (e.g., "binance")
            timestamp: Candle timestamp
            open_price: Opening price
            high: High price
            low: Low price
            close: Closing price
            volume: Trading volume

        Returns:
            True if inserted successfully, False if duplicate

        Raises:
            ValueError: If input validation fails
            psycopg2.Error: If database error occurs
        """
        # Input validation
        if not symbol or not exchange:
            raise ValueError("Symbol and exchange are required")

        if high < low:
            raise ValueError(f"High price ({high}) cannot be less than low price ({low})")

        if volume < 0:
            raise ValueError(f"Volume cannot be negative: {volume}")

        # Parameterized query (SAFE - prevents SQL injection)
        query = """
            INSERT INTO crypto_ohlcv (symbol, exchange, timestamp, open, high, low, close, volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol, exchange, timestamp) DO NOTHING
            RETURNING id;
        """

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        query,
                        (symbol, exchange, timestamp, open_price, high, low, close, volume),
                    )
                    result = cur.fetchone()
                    conn.commit()

                    if result:
                        logger.debug(f"Inserted OHLCV: {symbol} {exchange} {timestamp}")
                        return True
                    else:
                        logger.debug(f"Duplicate OHLCV skipped: {symbol} {exchange} {timestamp}")
                        return False

        except Exception as e:
            logger.error(f"Error inserting OHLCV: {str(e)}")
            raise

    def insert_ohlcv_batch(self, df: pd.DataFrame, exchange: str) -> int:
        """
        Insert multiple OHLCV candles from a DataFrame.

        More efficient than individual inserts for large datasets.
        Uses COPY FROM or batch INSERT for performance.

        Args:
            df: DataFrame with columns [Dates, {symbol}_open, {symbol}_high, {symbol}_low, {symbol}_close, {symbol}_volume]
            exchange: Exchange name

        Returns:
            Number of rows inserted

        Raises:
            ValueError: If DataFrame schema is invalid
        """
        if "Dates" not in df.columns:
            raise ValueError("DataFrame must contain 'Dates' column")

        # Find all symbols in the DataFrame
        close_cols = [col for col in df.columns if col.endswith("_close")]
        symbols = [col.replace("_close", "") for col in close_cols]

        inserted_count = 0

        for symbol in symbols:
            # Check if all required columns exist for this symbol
            required_cols = [
                f"{symbol}_open",
                f"{symbol}_high",
                f"{symbol}_low",
                f"{symbol}_close",
                f"{symbol}_volume",
            ]

            if not all(col in df.columns for col in required_cols):
                logger.warning(f"Skipping {symbol}: missing required columns")
                continue

            # Prepare data for batch insert
            records = []
            for _, row in df.iterrows():
                records.append(
                    (
                        symbol.replace("_", "/"),  # Convert BTC_USDT to BTC/USDT
                        exchange,
                        row["Dates"],
                        row[f"{symbol}_open"],
                        row[f"{symbol}_high"],
                        row[f"{symbol}_low"],
                        row[f"{symbol}_close"],
                        row[f"{symbol}_volume"],
                    )
                )

            # Batch insert using execute_batch for performance
            query = """
                INSERT INTO crypto_ohlcv (symbol, exchange, timestamp, open, high, low, close, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, exchange, timestamp) DO NOTHING;
            """

            try:
                with self.get_connection() as conn:
                    with conn.cursor() as cur:
                        psycopg2.extras.execute_batch(cur, query, records, page_size=1000)
                        conn.commit()
                        inserted_count += len(records)

                logger.info(f"Inserted {len(records)} OHLCV records for {symbol} from {exchange}")

            except Exception as e:
                logger.error(f"Error batch inserting OHLCV for {symbol}: {str(e)}")
                raise

        return inserted_count

    def get_ohlcv_data(
        self,
        symbol: str,
        exchange: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Retrieve OHLCV data from database.

        Uses parameterized query to prevent SQL injection.

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            exchange: Exchange name
            start_date: Start date for data (optional)
            end_date: End date for data (optional)
            limit: Maximum number of rows to return (optional)

        Returns:
            DataFrame with OHLCV data

        Raises:
            ValueError: If input validation fails
        """
        # Input validation
        if not symbol or not exchange:
            raise ValueError("Symbol and exchange are required")

        # Build parameterized query (SAFE)
        query = """
            SELECT timestamp as "Dates", open, high, low, close, volume
            FROM crypto_ohlcv
            WHERE symbol = %s AND exchange = %s
        """
        params = [symbol, exchange]

        if start_date:
            query += " AND timestamp >= %s"
            params.append(start_date)

        if end_date:
            query += " AND timestamp <= %s"
            params.append(end_date)

        query += " ORDER BY timestamp ASC"

        if limit:
            query += " LIMIT %s"
            params.append(limit)

        try:
            with self.get_connection() as conn:
                df = pd.read_sql_query(query, conn, params=params)

                logger.info(f"Retrieved {len(df)} OHLCV records for {symbol} from {exchange}")

                # Rename columns to match expected format
                symbol_base = symbol.replace("/", "_")
                df = df.rename(
                    columns={
                        "open": f"{symbol_base}_open",
                        "high": f"{symbol_base}_high",
                        "low": f"{symbol_base}_low",
                        "close": f"{symbol_base}_close",
                        "volume": f"{symbol_base}_volume",
                    }
                )

                return df

        except Exception as e:
            logger.error(f"Error retrieving OHLCV data: {str(e)}")
            raise

    # ==========================================================================
    # Blockchain Metrics Operations
    # ==========================================================================

    def insert_blockchain_metric(
        self,
        asset: str,
        metric_name: str,
        timestamp: datetime,
        value: float,
    ) -> bool:
        """
        Insert a single blockchain metric into the database.

        Uses parameterized query to prevent SQL injection.

        Args:
            asset: Asset symbol (e.g., "BTC")
            metric_name: Metric name (e.g., "mvrv")
            timestamp: Metric timestamp
            value: Metric value

        Returns:
            True if inserted successfully, False if duplicate
        """
        # Input validation
        if not asset or not metric_name:
            raise ValueError("Asset and metric_name are required")

        # Parameterized query (SAFE)
        query = """
            INSERT INTO blockchain_metrics (asset, metric_name, timestamp, value)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (asset, metric_name, timestamp) DO NOTHING
            RETURNING id;
        """

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (asset, metric_name, timestamp, value))
                    result = cur.fetchone()
                    conn.commit()

                    if result:
                        logger.debug(f"Inserted metric: {asset} {metric_name} {timestamp}")
                        return True
                    else:
                        logger.debug(f"Duplicate metric skipped: {asset} {metric_name} {timestamp}")
                        return False

        except Exception as e:
            logger.error(f"Error inserting blockchain metric: {str(e)}")
            raise

    def insert_blockchain_metrics_batch(self, df: pd.DataFrame) -> int:
        """
        Insert multiple blockchain metrics from a DataFrame.

        Args:
            df: DataFrame with columns [Dates, {asset}_{metric}, ...]

        Returns:
            Number of rows inserted
        """
        if "Dates" not in df.columns:
            raise ValueError("DataFrame must contain 'Dates' column")

        inserted_count = 0

        # Parse asset and metric from column names
        metric_cols = [col for col in df.columns if col != "Dates"]

        for col in metric_cols:
            # Column format: BTC_mvrv, ETH_nvt, etc.
            parts = col.split("_", 1)
            if len(parts) != 2:
                logger.warning(f"Skipping column with invalid format: {col}")
                continue

            asset, metric = parts

            # Prepare batch records
            records = [
                (asset, metric, row["Dates"], row[col])
                for _, row in df.iterrows()
                if pd.notna(row[col])
            ]

            query = """
                INSERT INTO blockchain_metrics (asset, metric_name, timestamp, value)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (asset, metric_name, timestamp) DO NOTHING;
            """

            try:
                with self.get_connection() as conn:
                    with conn.cursor() as cur:
                        psycopg2.extras.execute_batch(cur, query, records, page_size=1000)
                        conn.commit()
                        inserted_count += len(records)

                logger.info(f"Inserted {len(records)} records for {asset}_{metric}")

            except Exception as e:
                logger.error(f"Error batch inserting metrics for {asset}_{metric}: {str(e)}")
                raise

        return inserted_count

    def get_blockchain_metrics(
        self,
        asset: str,
        metric_name: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Retrieve blockchain metrics from database.

        Args:
            asset: Asset symbol
            metric_name: Metric name
            start_date: Start date (optional)
            end_date: End date (optional)

        Returns:
            DataFrame with metric data
        """
        # Parameterized query (SAFE)
        query = """
            SELECT timestamp as "Dates", value
            FROM blockchain_metrics
            WHERE asset = %s AND metric_name = %s
        """
        params = [asset, metric_name]

        if start_date:
            query += " AND timestamp >= %s"
            params.append(start_date)

        if end_date:
            query += " AND timestamp <= %s"
            params.append(end_date)

        query += " ORDER BY timestamp ASC"

        try:
            with self.get_connection() as conn:
                df = pd.read_sql_query(query, conn, params=params)

                logger.info(f"Retrieved {len(df)} records for {asset}_{metric_name}")

                # Rename value column to match format
                df = df.rename(columns={"value": f"{asset}_{metric_name}"})

                return df

        except Exception as e:
            logger.error(f"Error retrieving blockchain metrics: {str(e)}")
            raise

    # ==========================================================================
    # Model Predictions Operations
    # ==========================================================================

    def insert_prediction(
        self,
        model_id: str,
        symbol: str,
        prediction_timestamp: datetime,
        forecast_timestamp: datetime,
        predicted_value: float,
        actual_value: Optional[float] = None,
        mae: Optional[float] = None,
    ) -> int:
        """
        Insert a model prediction into the database.

        Args:
            model_id: Model identifier
            symbol: Trading pair symbol
            prediction_timestamp: When the prediction was made
            forecast_timestamp: What time the prediction is for
            predicted_value: Predicted value
            actual_value: Actual value (if known)
            mae: Mean absolute error (if known)

        Returns:
            ID of inserted row
        """
        # Parameterized query (SAFE)
        query = """
            INSERT INTO model_predictions
            (model_id, symbol, prediction_timestamp, forecast_timestamp, predicted_value, actual_value, mae)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id;
        """

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        query,
                        (
                            model_id,
                            symbol,
                            prediction_timestamp,
                            forecast_timestamp,
                            predicted_value,
                            actual_value,
                            mae,
                        ),
                    )
                    result = cur.fetchone()
                    conn.commit()

                    logger.debug(f"Inserted prediction: {model_id} for {symbol}")
                    return result[0]

        except Exception as e:
            logger.error(f"Error inserting prediction: {str(e)}")
            raise

    def get_model_performance(self, model_id: str) -> pd.DataFrame:
        """
        Get performance metrics for a model.

        Args:
            model_id: Model identifier

        Returns:
            DataFrame with prediction vs actual values and errors
        """
        # Parameterized query (SAFE)
        query = """
            SELECT
                symbol,
                prediction_timestamp,
                forecast_timestamp,
                predicted_value,
                actual_value,
                mae,
                created_at
            FROM model_predictions
            WHERE model_id = %s AND actual_value IS NOT NULL
            ORDER BY prediction_timestamp DESC;
        """

        try:
            with self.get_connection() as conn:
                df = pd.read_sql_query(query, conn, params=[model_id])

                logger.info(f"Retrieved {len(df)} predictions for model {model_id}")

                return df

        except Exception as e:
            logger.error(f"Error retrieving model performance: {str(e)}")
            raise
