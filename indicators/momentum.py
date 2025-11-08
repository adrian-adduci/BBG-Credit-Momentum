"""
Momentum Indicators Module

This module implements momentum-based technical indicators for trading analysis.
Momentum indicators measure the rate of change in price movements and help
identify trend strength, potential reversals, and volatility.

Indicators implemented:
- Rate of Change (ROC)
- Commodity Channel Index (CCI)
- Average True Range (ATR)
- On-Balance Volume (OBV)

Author: BBG-Credit-Momentum Team
License: MIT
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Union

# Set up structured logging
logger = logging.getLogger(__name__)


class MomentumIndicators:
    """
    Calculate momentum-based technical indicators.

    This class provides methods for calculating various momentum indicators
    that help identify trend strength, volatility, and volume-based signals.

    All methods follow the same pattern:
    - Accept pandas Series or DataFrame column as input
    - Return pandas Series with indicator values
    - Handle NaN values gracefully
    - Log calculation metrics for debugging
    """

    def __init__(self, debug: bool = False):
        """
        Initialize MomentumIndicators calculator.

        Args:
            debug: Enable debug logging with calculation metrics
        """
        self.debug = debug
        self.calculation_counter = 0  # Track number of calculations for metrics

        if self.debug:
            logger.setLevel(logging.DEBUG)
            logger.debug("MomentumIndicators initialized with debug mode enabled")

    def rate_of_change(
        self,
        close: pd.Series,
        window: int = 10,
    ) -> pd.Series:
        """
        Calculate Rate of Change (ROC) indicator.

        ROC measures the percentage change in price over a specified period.
        It's a pure momentum oscillator that shows the speed of price movement.

        Formula:
            ROC = 100 * (Price_today - Price_n_days_ago) / Price_n_days_ago

        Interpretation:
            > 0: Upward momentum (bullish)
            < 0: Downward momentum (bearish)
            Crossing zero line: Potential trend change
            Extreme values (> ±10%): Overbought/oversold conditions

        Args:
            close: Closing prices
            window: Lookback period for comparison (default: 10)

        Returns:
            ROC values as pandas Series (percentage)

        Raises:
            ValueError: If window is invalid or series too short
            TypeError: If input is not pandas Series

        Example:
            >>> close = pd.Series([100, 102, 105, 103, 108])
            >>> roc = calculator.rate_of_change(close, window=3)
            >>> # roc[3] = 100 * (103 - 100) / 100 = 3.0
        """
        # Input validation
        if not isinstance(close, pd.Series):
            raise TypeError("Input must be pandas Series")

        if window < 1:
            raise ValueError(f"Window size must be positive. Got window={window}")

        if len(close) <= window:
            raise ValueError(f"Input series too short. Need > {window} points, got {len(close)}")

        logger.info(f"Calculating Rate of Change: window={window}")

        try:
            # ROC = 100 * (Price - Price_shifted) / Price_shifted
            shifted_close = close.shift(window)
            epsilon = 1e-10  # Prevent division by zero
            roc = 100 * (close - shifted_close) / (shifted_close + epsilon)

            # Count valid values and calculate statistics
            valid_values = roc.notna().sum()

            self.calculation_counter += 1

            if self.debug:
                logger.debug(f"ROC calculation #{self.calculation_counter}:")
                logger.debug(f"  Input length: {len(close)}")
                logger.debug(f"  Valid values: {valid_values}/{len(roc)} ({valid_values/len(roc)*100:.1f}%)")
                logger.debug(f"  Range: [{roc.min():.2f}%, {roc.max():.2f}%]")
                logger.debug(f"  Mean: {roc.mean():.2f}%, Std: {roc.std():.2f}%")

                # Count extreme values
                extreme_high = (roc > 10).sum()
                extreme_low = (roc < -10).sum()
                logger.debug(f"  Extreme values: {extreme_high} high (>10%), {extreme_low} low (<-10%)")

            return roc

        except Exception as e:
            logger.error(f"Error calculating Rate of Change: {str(e)}")
            raise

    def commodity_channel_index(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 20,
        constant: float = 0.015,
    ) -> pd.Series:
        """
        Calculate Commodity Channel Index (CCI).

        CCI measures the deviation of price from its statistical mean. It's useful
        for identifying cyclical trends and overbought/oversold conditions. Originally
        designed for commodities but works well for any asset.

        Formula:
            1. Typical Price (TP) = (High + Low + Close) / 3
            2. SMA = Simple Moving Average of TP
            3. Mean Deviation = Average of |TP - SMA|
            4. CCI = (TP - SMA) / (constant × Mean Deviation)

        Interpretation:
            > +100: Overbought (strong uptrend)
            < -100: Oversold (strong downtrend)
            Between -100 and +100: Normal range
            Crossing zero: Potential trend change

        Args:
            high: High prices
            low: Low prices
            close: Closing prices
            window: Period for moving average (default: 20)
            constant: Scaling constant (default: 0.015, typically 0.015)

        Returns:
            CCI values as pandas Series

        Raises:
            ValueError: If input series have different lengths or invalid parameters
            TypeError: If inputs are not pandas Series

        Example:
            >>> high = pd.Series([105, 110, 108])
            >>> low = pd.Series([100, 105, 103])
            >>> close = pd.Series([104, 109, 106])
            >>> cci = calculator.commodity_channel_index(high, low, close)
        """
        # Input validation
        if not isinstance(high, pd.Series) or not isinstance(low, pd.Series) or not isinstance(close, pd.Series):
            raise TypeError("Inputs must be pandas Series")

        if len(high) != len(low) or len(high) != len(close):
            raise ValueError(f"Input series must have same length. Got high={len(high)}, low={len(low)}, close={len(close)}")

        if window < 1:
            raise ValueError(f"Window size must be positive. Got window={window}")

        if constant <= 0:
            raise ValueError(f"Constant must be positive. Got constant={constant}")

        logger.info(f"Calculating CCI: window={window}, constant={constant}")

        try:
            # Step 1: Calculate Typical Price
            typical_price = (high + low + close) / 3

            # Step 2: Calculate SMA of Typical Price
            sma = typical_price.rolling(window=window, min_periods=window).mean()

            # Step 3: Calculate Mean Deviation
            # Mean Deviation = average of absolute deviations from SMA
            mean_deviation = (
                typical_price.rolling(window=window, min_periods=window)
                .apply(lambda x: np.abs(x - x.mean()).mean(), raw=False)
            )

            # Step 4: Calculate CCI
            epsilon = 1e-10  # Prevent division by zero
            cci = (typical_price - sma) / (constant * mean_deviation + epsilon)

            # Count valid values and statistics
            valid_values = cci.notna().sum()

            self.calculation_counter += 1

            if self.debug:
                logger.debug(f"CCI calculation #{self.calculation_counter}:")
                logger.debug(f"  Input length: {len(close)}")
                logger.debug(f"  Valid values: {valid_values}/{len(cci)} ({valid_values/len(cci)*100:.1f}%)")
                logger.debug(f"  Range: [{cci.min():.2f}, {cci.max():.2f}]")
                logger.debug(f"  Mean: {cci.mean():.2f}, Std: {cci.std():.2f}")

                # Count overbought/oversold conditions
                overbought = (cci > 100).sum()
                oversold = (cci < -100).sum()
                logger.debug(f"  Overbought periods (>100): {overbought} ({overbought/valid_values*100:.1f}%)")
                logger.debug(f"  Oversold periods (<-100): {oversold} ({oversold/valid_values*100:.1f}%)")

            return cci

        except Exception as e:
            logger.error(f"Error calculating CCI: {str(e)}")
            raise

    def average_true_range(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14,
    ) -> pd.Series:
        """
        Calculate Average True Range (ATR).

        ATR measures market volatility by decomposing the entire range of price
        movement. Unlike other volatility measures, ATR accounts for gaps in price.
        It's widely used for position sizing and stop-loss placement.

        Formula:
            1. True Range (TR) = max of:
               - High - Low
               - |High - Previous Close|
               - |Low - Previous Close|
            2. ATR = Moving Average of TR (typically EMA)

        Interpretation:
            Higher ATR: Increased volatility, wider stops needed
            Lower ATR: Decreased volatility, tighter stops possible
            Rising ATR: Increasing volatility (often at trend start)
            Falling ATR: Decreasing volatility (often in consolidation)

        Args:
            high: High prices
            low: Low prices
            close: Closing prices
            window: Period for moving average (default: 14)

        Returns:
            ATR values as pandas Series

        Raises:
            ValueError: If input series have different lengths or invalid window
            TypeError: If inputs are not pandas Series

        Example:
            >>> high = pd.Series([105, 110, 108, 112])
            >>> low = pd.Series([100, 105, 103, 107])
            >>> close = pd.Series([104, 109, 106, 111])
            >>> atr = calculator.average_true_range(high, low, close)
        """
        # Input validation
        if not isinstance(high, pd.Series) or not isinstance(low, pd.Series) or not isinstance(close, pd.Series):
            raise TypeError("Inputs must be pandas Series")

        if len(high) != len(low) or len(high) != len(close):
            raise ValueError(f"Input series must have same length. Got high={len(high)}, low={len(low)}, close={len(close)}")

        if window < 1:
            raise ValueError(f"Window size must be positive. Got window={window}")

        logger.info(f"Calculating ATR: window={window}")

        try:
            # Calculate the three components of True Range
            tr1 = high - low  # Current high - low
            tr2 = abs(high - close.shift())  # Current high - previous close
            tr3 = abs(low - close.shift())  # Current low - previous close

            # True Range is the maximum of the three
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # ATR is the exponential moving average of True Range
            # Using EMA instead of SMA for smoother results
            atr = true_range.ewm(span=window, min_periods=window, adjust=False).mean()

            # Count valid values and statistics
            valid_values = atr.notna().sum()

            self.calculation_counter += 1

            if self.debug:
                logger.debug(f"ATR calculation #{self.calculation_counter}:")
                logger.debug(f"  Input length: {len(close)}")
                logger.debug(f"  Valid values: {valid_values}/{len(atr)} ({valid_values/len(atr)*100:.1f}%)")
                logger.debug(f"  Range: [{atr.min():.2f}, {atr.max():.2f}]")
                logger.debug(f"  Mean: {atr.mean():.2f}, Std: {atr.std():.2f}")
                logger.debug(f"  Recent ATR: {atr.iloc[-1]:.2f}")

                # Calculate ATR as percentage of price for context
                atr_pct = (atr / close * 100).mean()
                logger.debug(f"  Average ATR as % of price: {atr_pct:.2f}%")

            return atr

        except Exception as e:
            logger.error(f"Error calculating ATR: {str(e)}")
            raise

    def on_balance_volume(
        self,
        close: pd.Series,
        volume: pd.Series,
    ) -> pd.Series:
        """
        Calculate On-Balance Volume (OBV).

        OBV is a volume-based momentum indicator that relates volume to price
        change. It's based on the theory that volume precedes price movement:
        smart money accumulates during up-volume days and distributes during
        down-volume days.

        Formula:
            If Close > Previous Close: OBV = Previous OBV + Volume
            If Close < Previous Close: OBV = Previous OBV - Volume
            If Close = Previous Close: OBV = Previous OBV

        Interpretation:
            Rising OBV: Accumulation (bullish)
            Falling OBV: Distribution (bearish)
            OBV confirms trend: Both price and OBV rising (strong uptrend)
            OBV divergence: Price rising but OBV falling (warning signal)

        Args:
            close: Closing prices
            volume: Trading volume

        Returns:
            OBV values as pandas Series (cumulative volume)

        Raises:
            ValueError: If input series have different lengths
            TypeError: If inputs are not pandas Series

        Example:
            >>> close = pd.Series([100, 102, 101, 105])
            >>> volume = pd.Series([1000, 1200, 900, 1500])
            >>> obv = calculator.on_balance_volume(close, volume)
        """
        # Input validation
        if not isinstance(close, pd.Series) or not isinstance(volume, pd.Series):
            raise TypeError("Inputs must be pandas Series")

        if len(close) != len(volume):
            raise ValueError(f"Input series must have same length. Got close={len(close)}, volume={len(volume)}")

        logger.info("Calculating On-Balance Volume")

        try:
            # Calculate price direction: 1 (up), -1 (down), 0 (unchanged)
            price_change = close.diff()
            direction = np.sign(price_change)

            # OBV = cumulative sum of (direction × volume)
            obv = (direction * volume).cumsum()

            # Count valid values and statistics
            valid_values = obv.notna().sum()

            self.calculation_counter += 1

            if self.debug:
                logger.debug(f"OBV calculation #{self.calculation_counter}:")
                logger.debug(f"  Input length: {len(close)}")
                logger.debug(f"  Valid values: {valid_values}/{len(obv)} ({valid_values/len(obv)*100:.1f}%)")
                logger.debug(f"  Range: [{obv.min():.0f}, {obv.max():.0f}]")
                logger.debug(f"  Final OBV: {obv.iloc[-1]:.0f}")

                # Count up/down days
                up_days = (direction > 0).sum()
                down_days = (direction < 0).sum()
                unchanged = (direction == 0).sum()
                logger.debug(f"  Up days: {up_days}, Down days: {down_days}, Unchanged: {unchanged}")

                # Check for divergence (simplified check)
                recent_price_trend = (close.iloc[-20:].iloc[-1] - close.iloc[-20:].iloc[0]) / close.iloc[-20:].iloc[0]
                recent_obv_trend = (obv.iloc[-20:].iloc[-1] - obv.iloc[-20:].iloc[0]) / abs(obv.iloc[-20:].iloc[0] + 1e-10)

                if (recent_price_trend > 0 and recent_obv_trend < 0) or (recent_price_trend < 0 and recent_obv_trend > 0):
                    logger.debug("  WARNING: Potential price-OBV divergence detected!")

            return obv

        except Exception as e:
            logger.error(f"Error calculating On-Balance Volume: {str(e)}")
            raise


# Convenience functions for quick calculations without instantiating the class
def calculate_roc(
    close: pd.Series,
    window: int = 10,
    debug: bool = False,
) -> pd.Series:
    """
    Quick function to calculate Rate of Change.

    See MomentumIndicators.rate_of_change() for detailed documentation.
    """
    calculator = MomentumIndicators(debug=debug)
    return calculator.rate_of_change(close, window)


def calculate_cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
    constant: float = 0.015,
    debug: bool = False,
) -> pd.Series:
    """
    Quick function to calculate Commodity Channel Index.

    See MomentumIndicators.commodity_channel_index() for detailed documentation.
    """
    calculator = MomentumIndicators(debug=debug)
    return calculator.commodity_channel_index(high, low, close, window, constant)


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
    debug: bool = False,
) -> pd.Series:
    """
    Quick function to calculate Average True Range.

    See MomentumIndicators.average_true_range() for detailed documentation.
    """
    calculator = MomentumIndicators(debug=debug)
    return calculator.average_true_range(high, low, close, window)


def calculate_obv(
    close: pd.Series,
    volume: pd.Series,
    debug: bool = False,
) -> pd.Series:
    """
    Quick function to calculate On-Balance Volume.

    See MomentumIndicators.on_balance_volume() for detailed documentation.
    """
    calculator = MomentumIndicators(debug=debug)
    return calculator.on_balance_volume(close, volume)
