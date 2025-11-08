"""
Stochastic Indicators Module

This module implements stochastic-based technical indicators for trading analysis.
Stochastic indicators measure momentum by comparing closing price to price range
over a period. They're particularly useful for identifying overbought/oversold
conditions and potential reversal points.

Indicators implemented:
- Stochastic Oscillator (%K and %D)
- Stochastic RSI (combines RSI with Stochastic)
- Williams %R (inverse of Stochastic)

Author: BBG-Credit-Momentum Team
License: MIT
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union

# Set up structured logging
logger = logging.getLogger(__name__)


class StochasticIndicators:
    """
    Calculate stochastic-based momentum indicators.

    This class provides methods for calculating various stochastic indicators
    that help identify momentum and overbought/oversold conditions in price data.

    All methods follow the same pattern:
    - Accept pandas Series or DataFrame column as input
    - Return pandas Series with indicator values
    - Handle NaN values gracefully
    - Log calculation metrics for debugging
    """

    def __init__(self, debug: bool = False):
        """
        Initialize StochasticIndicators calculator.

        Args:
            debug: Enable debug logging with calculation metrics
        """
        self.debug = debug
        self.calculation_counter = 0  # Track number of calculations for metrics

        if self.debug:
            logger.setLevel(logging.DEBUG)
            logger.debug("StochasticIndicators initialized with debug mode enabled")

    def stochastic_oscillator(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_window: int = 14,
        d_window: int = 3,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator (%K and %D).

        The Stochastic Oscillator compares the closing price to the price range
        over a given period. It consists of two lines:
        - %K (fast): Current close position within the range
        - %D (slow): Moving average of %K, acts as signal line

        Formula:
            %K = 100 * (Close - Low_n) / (High_n - Low_n)
            %D = SMA(%K, d_window)

        Interpretation:
            >80: Overbought condition (potential sell signal)
            <20: Oversold condition (potential buy signal)
            %K crosses above %D: Bullish signal
            %K crosses below %D: Bearish signal

        Args:
            high: High prices
            low: Low prices
            close: Closing prices
            k_window: Lookback period for %K calculation (default: 14)
            d_window: Moving average period for %D (default: 3)

        Returns:
            Tuple of (%K, %D) as pandas Series

        Raises:
            ValueError: If input series have different lengths or invalid windows
            TypeError: If inputs are not pandas Series

        Example:
            >>> high = pd.Series([100, 105, 110, 108, 112])
            >>> low = pd.Series([95, 100, 105, 103, 107])
            >>> close = pd.Series([98, 104, 109, 106, 111])
            >>> stoch_k, stoch_d = calculator.stochastic_oscillator(high, low, close)
        """
        # Input validation
        if not isinstance(high, pd.Series) or not isinstance(low, pd.Series) or not isinstance(close, pd.Series):
            raise TypeError("Inputs must be pandas Series")

        if len(high) != len(low) or len(high) != len(close):
            raise ValueError(f"Input series must have same length. Got high={len(high)}, low={len(low)}, close={len(close)}")

        if k_window < 1 or d_window < 1:
            raise ValueError(f"Window sizes must be positive. Got k_window={k_window}, d_window={d_window}")

        logger.info(f"Calculating Stochastic Oscillator: k_window={k_window}, d_window={d_window}")

        try:
            # Calculate lowest low and highest high over the window
            low_min = low.rolling(window=k_window, min_periods=k_window).min()
            high_max = high.rolling(window=k_window, min_periods=k_window).max()

            # Calculate %K = 100 * (Close - Low_n) / (High_n - Low_n)
            # Add small epsilon to denominator to avoid division by zero
            epsilon = 1e-10
            stoch_k = 100 * (close - low_min) / (high_max - low_min + epsilon)

            # Calculate %D = SMA of %K
            stoch_d = stoch_k.rolling(window=d_window, min_periods=d_window).mean()

            # Count valid values for metrics
            valid_k = stoch_k.notna().sum()
            valid_d = stoch_d.notna().sum()

            self.calculation_counter += 1

            if self.debug:
                logger.debug(f"Stochastic calculation #{self.calculation_counter}:")
                logger.debug(f"  Input length: {len(close)}")
                logger.debug(f"  Valid %K values: {valid_k}/{len(stoch_k)} ({valid_k/len(stoch_k)*100:.1f}%)")
                logger.debug(f"  Valid %D values: {valid_d}/{len(stoch_d)} ({valid_d/len(stoch_d)*100:.1f}%)")
                logger.debug(f"  %K range: [{stoch_k.min():.2f}, {stoch_k.max():.2f}]")
                logger.debug(f"  %D range: [{stoch_d.min():.2f}, {stoch_d.max():.2f}]")

            return stoch_k, stoch_d

        except Exception as e:
            logger.error(f"Error calculating Stochastic Oscillator: {str(e)}")
            raise

    def stochastic_rsi(
        self,
        close: pd.Series,
        rsi_window: int = 14,
        stoch_window: int = 14,
        k_window: int = 3,
        d_window: int = 3,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic RSI (StochRSI).

        StochRSI applies the Stochastic Oscillator formula to RSI values instead
        of price. This creates a more sensitive indicator that oscillates between
        0 and 100, useful for identifying overbought/oversold conditions earlier.

        Formula:
            1. Calculate RSI(close, rsi_window)
            2. StochRSI = 100 * (RSI - RSI_low) / (RSI_high - RSI_low)
            3. %K = SMA(StochRSI, k_window)
            4. %D = SMA(%K, d_window)

        Interpretation:
            >80: Overbought (more sensitive than regular RSI)
            <20: Oversold (more sensitive than regular RSI)
            Faster signals than regular RSI, but more false signals

        Args:
            close: Closing prices
            rsi_window: Period for RSI calculation (default: 14)
            stoch_window: Lookback period for Stochastic calculation (default: 14)
            k_window: Smoothing period for %K (default: 3)
            d_window: Smoothing period for %D (default: 3)

        Returns:
            Tuple of (StochRSI %K, StochRSI %D) as pandas Series

        Raises:
            ValueError: If input series is too short or invalid windows
            TypeError: If input is not pandas Series

        Example:
            >>> close = pd.Series([...])  # Price data
            >>> stoch_k, stoch_d = calculator.stochastic_rsi(close)
        """
        # Input validation
        if not isinstance(close, pd.Series):
            raise TypeError("Input must be pandas Series")

        if rsi_window < 2 or stoch_window < 1 or k_window < 1 or d_window < 1:
            raise ValueError("All window sizes must be positive (rsi_window >= 2)")

        min_length = rsi_window + stoch_window + k_window
        if len(close) < min_length:
            raise ValueError(f"Input series too short. Need at least {min_length} points, got {len(close)}")

        logger.info(f"Calculating Stochastic RSI: rsi_window={rsi_window}, stoch_window={stoch_window}")

        try:
            # Step 1: Calculate RSI
            rsi = self._calculate_rsi(close, window=rsi_window)

            # Step 2: Apply Stochastic formula to RSI
            rsi_min = rsi.rolling(window=stoch_window, min_periods=stoch_window).min()
            rsi_max = rsi.rolling(window=stoch_window, min_periods=stoch_window).max()

            epsilon = 1e-10
            stoch_rsi = 100 * (rsi - rsi_min) / (rsi_max - rsi_min + epsilon)

            # Step 3: Smooth with %K
            stoch_k = stoch_rsi.rolling(window=k_window, min_periods=k_window).mean()

            # Step 4: Calculate %D signal line
            stoch_d = stoch_k.rolling(window=d_window, min_periods=d_window).mean()

            # Metrics
            valid_k = stoch_k.notna().sum()
            valid_d = stoch_d.notna().sum()

            self.calculation_counter += 1

            if self.debug:
                logger.debug(f"StochRSI calculation #{self.calculation_counter}:")
                logger.debug(f"  Input length: {len(close)}")
                logger.debug(f"  Valid RSI values: {rsi.notna().sum()}/{len(rsi)}")
                logger.debug(f"  Valid %K values: {valid_k}/{len(stoch_k)}")
                logger.debug(f"  Valid %D values: {valid_d}/{len(stoch_d)}")
                logger.debug(f"  %K range: [{stoch_k.min():.2f}, {stoch_k.max():.2f}]")

            return stoch_k, stoch_d

        except Exception as e:
            logger.error(f"Error calculating Stochastic RSI: {str(e)}")
            raise

    def williams_r(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14,
    ) -> pd.Series:
        """
        Calculate Williams %R indicator.

        Williams %R is essentially the inverse of the Stochastic Oscillator.
        It measures overbought/oversold levels by comparing the close to the
        recent high-low range. Values are negative, ranging from 0 to -100.

        Formula:
            %R = -100 * (High_n - Close) / (High_n - Low_n)

        Interpretation:
            0 to -20: Overbought (potential sell signal)
            -80 to -100: Oversold (potential buy signal)
            Crosses above -50: Bullish momentum
            Crosses below -50: Bearish momentum

        Args:
            high: High prices
            low: Low prices
            close: Closing prices
            window: Lookback period (default: 14)

        Returns:
            Williams %R values as pandas Series (range: -100 to 0)

        Raises:
            ValueError: If input series have different lengths or invalid window
            TypeError: If inputs are not pandas Series

        Example:
            >>> high = pd.Series([100, 105, 110])
            >>> low = pd.Series([95, 100, 105])
            >>> close = pd.Series([98, 104, 109])
            >>> williams = calculator.williams_r(high, low, close, window=14)
        """
        # Input validation
        if not isinstance(high, pd.Series) or not isinstance(low, pd.Series) or not isinstance(close, pd.Series):
            raise TypeError("Inputs must be pandas Series")

        if len(high) != len(low) or len(high) != len(close):
            raise ValueError(f"Input series must have same length. Got high={len(high)}, low={len(low)}, close={len(close)}")

        if window < 1:
            raise ValueError(f"Window size must be positive. Got window={window}")

        logger.info(f"Calculating Williams %R: window={window}")

        try:
            # Calculate highest high and lowest low over the window
            high_max = high.rolling(window=window, min_periods=window).max()
            low_min = low.rolling(window=window, min_periods=window).min()

            # Calculate %R = -100 * (High_n - Close) / (High_n - Low_n)
            epsilon = 1e-10
            williams = -100 * (high_max - close) / (high_max - low_min + epsilon)

            # Count valid values
            valid_values = williams.notna().sum()

            self.calculation_counter += 1

            if self.debug:
                logger.debug(f"Williams %R calculation #{self.calculation_counter}:")
                logger.debug(f"  Input length: {len(close)}")
                logger.debug(f"  Valid values: {valid_values}/{len(williams)} ({valid_values/len(williams)*100:.1f}%)")
                logger.debug(f"  Range: [{williams.min():.2f}, {williams.max():.2f}]")
                logger.debug(f"  Mean: {williams.mean():.2f}, Std: {williams.std():.2f}")

            return williams

        except Exception as e:
            logger.error(f"Error calculating Williams %R: {str(e)}")
            raise

    def _calculate_rsi(self, close: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).

        Helper method used by stochastic_rsi. Uses the standard RSI formula
        with exponential moving average for smoothing.

        Formula:
            1. Calculate price changes (deltas)
            2. Separate gains and losses
            3. Calculate average gain and average loss (EMA)
            4. RS = Average Gain / Average Loss
            5. RSI = 100 - (100 / (1 + RS))

        Args:
            close: Closing prices
            window: RSI period (default: 14)

        Returns:
            RSI values as pandas Series (range: 0-100)
        """
        # Calculate price changes
        delta = close.diff()

        # Separate gains and losses
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        # Calculate exponential moving average of gains and losses
        avg_gain = gain.ewm(com=window - 1, min_periods=window, adjust=False).mean()
        avg_loss = loss.ewm(com=window - 1, min_periods=window, adjust=False).mean()

        # Calculate RS and RSI
        epsilon = 1e-10
        rs = avg_gain / (avg_loss + epsilon)
        rsi = 100 - (100 / (1 + rs))

        return rsi


# Convenience functions for quick calculations without instantiating the class
def calculate_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_window: int = 14,
    d_window: int = 3,
    debug: bool = False,
) -> Tuple[pd.Series, pd.Series]:
    """
    Quick function to calculate Stochastic Oscillator.

    See StochasticIndicators.stochastic_oscillator() for detailed documentation.
    """
    calculator = StochasticIndicators(debug=debug)
    return calculator.stochastic_oscillator(high, low, close, k_window, d_window)


def calculate_stochastic_rsi(
    close: pd.Series,
    rsi_window: int = 14,
    stoch_window: int = 14,
    k_window: int = 3,
    d_window: int = 3,
    debug: bool = False,
) -> Tuple[pd.Series, pd.Series]:
    """
    Quick function to calculate Stochastic RSI.

    See StochasticIndicators.stochastic_rsi() for detailed documentation.
    """
    calculator = StochasticIndicators(debug=debug)
    return calculator.stochastic_rsi(close, rsi_window, stoch_window, k_window, d_window)


def calculate_williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
    debug: bool = False,
) -> pd.Series:
    """
    Quick function to calculate Williams %R.

    See StochasticIndicators.williams_r() for detailed documentation.
    """
    calculator = StochasticIndicators(debug=debug)
    return calculator.williams_r(high, low, close, window)
