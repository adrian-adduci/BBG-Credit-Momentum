################################################################################
# Cross-Asset Feature Engineering
# Author: Adrian Adduci
# Email: FAA2160@columbia.edu
################################################################################

"""
Cross-asset feature engineering for mixed crypto and credit portfolios.

This module provides functions to calculate cross-asset relationships including:
- Correlation tracking between crypto and traditional securities
- Regime detection (risk-on vs risk-off)
- Momentum divergence indicators
- Flight-to-quality signals
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class CrossAssetIndicators:
    """
    Cross-asset feature engineering for unified crypto + credit analysis.

    This class calculates features that capture relationships between different
    asset classes (e.g., crypto vs credit spreads) to identify regime changes
    and momentum divergences.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize cross-asset indicators calculator.

        Args:
            df: DataFrame with columns from multiple asset classes
        """
        self.df = df.copy()

    def add_rolling_correlation(
        self,
        col1: str,
        col2: str,
        windows: List[int] = [20, 60, 120],
        min_periods: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Calculate rolling correlation between two columns.

        Useful for tracking evolving relationships between crypto and credit markets.

        Args:
            col1: First column name (e.g., "BTC_USDT_close")
            col2: Second column name (e.g., "LF98TRUU_OAS")
            windows: List of rolling window sizes in periods
            min_periods: Minimum periods required for calculation (default: window size)

        Returns:
            DataFrame with new correlation columns

        Example:
            >>> indicators = CrossAssetIndicators(df)
            >>> df = indicators.add_rolling_correlation(
            ...     "BTC_USDT_close",
            ...     "LF98TRUU_OAS",
            ...     windows=[20, 60]
            ... )
            >>> # Creates: corr_BTC_USDT_close_LF98TRUU_OAS_20, corr_..._60
        """
        if col1 not in self.df.columns:
            raise ValueError(f"Column {col1} not found in DataFrame")
        if col2 not in self.df.columns:
            raise ValueError(f"Column {col2} not found in DataFrame")

        for window in windows:
            col_name = f"corr_{col1}_{col2}_{window}"

            # Calculate rolling correlation
            self.df[col_name] = (
                self.df[col1]
                .rolling(window=window, min_periods=min_periods or window)
                .corr(self.df[col2])
            )

            logger.info(f"Added rolling correlation: {col_name}")

        return self.df

    def add_correlation_regime(
        self,
        col1: str,
        col2: str,
        window: int = 60,
        threshold_positive: float = 0.3,
        threshold_negative: float = -0.3
    ) -> pd.DataFrame:
        """
        Detect correlation regime (positive, negative, or neutral).

        Classifies the correlation regime between two assets:
        - 1: Positive correlation (risk-on, assets moving together)
        - -1: Negative correlation (flight-to-quality, divergence)
        - 0: Neutral/decorrelated

        Args:
            col1: First column name
            col2: Second column name
            window: Rolling window for correlation calculation
            threshold_positive: Threshold for positive correlation regime
            threshold_negative: Threshold for negative correlation regime

        Returns:
            DataFrame with correlation regime column

        Example:
            When BTC and credit spreads are positively correlated (both rising),
            this indicates risk-off regime. When negatively correlated, indicates
            rotation between asset classes.
        """
        # First calculate correlation
        corr_col = f"corr_{col1}_{col2}_{window}"
        if corr_col not in self.df.columns:
            self.add_rolling_correlation(col1, col2, windows=[window])

        regime_col = f"regime_{col1}_{col2}_{window}"

        # Classify regime
        self.df[regime_col] = 0  # Default: neutral
        self.df.loc[self.df[corr_col] > threshold_positive, regime_col] = 1  # Positive
        self.df.loc[self.df[corr_col] < threshold_negative, regime_col] = -1  # Negative

        logger.info(f"Added correlation regime: {regime_col}")

        return self.df

    def add_momentum_divergence(
        self,
        crypto_col: str,
        credit_col: str,
        momentum_window: int = 20,
        divergence_threshold: float = 2.0
    ) -> pd.DataFrame:
        """
        Calculate momentum divergence between crypto and credit markets.

        Identifies when crypto and credit markets are moving in opposite directions,
        which can signal regime changes or market stress.

        Args:
            crypto_col: Crypto price column (e.g., "BTC_USDT_close")
            credit_col: Credit spread column (e.g., "LF98TRUU_OAS")
            momentum_window: Window for momentum calculation
            divergence_threshold: Z-score threshold for significant divergence

        Returns:
            DataFrame with divergence indicators

        Creates:
            - momentum_{crypto_col}: Crypto momentum
            - momentum_{credit_col}: Credit momentum
            - divergence_{crypto_col}_{credit_col}: Divergence score
            - divergence_signal_{crypto_col}_{credit_col}: Binary signal (1=diverging)
        """
        # Calculate momentum for both assets
        crypto_momentum = self.df[crypto_col].pct_change(momentum_window)
        credit_momentum = self.df[credit_col].pct_change(momentum_window)

        # Store momentum
        self.df[f"momentum_{crypto_col}_{momentum_window}"] = crypto_momentum
        self.df[f"momentum_{credit_col}_{momentum_window}"] = credit_momentum

        # Calculate divergence (normalized difference)
        # Positive divergence: crypto rising, credit falling (risk-on)
        # Negative divergence: crypto falling, credit rising (risk-off)
        divergence = crypto_momentum - credit_momentum

        # Normalize to z-score
        divergence_mean = divergence.rolling(window=60).mean()
        divergence_std = divergence.rolling(window=60).std()
        divergence_zscore = (divergence - divergence_mean) / (divergence_std + 1e-8)

        divergence_col = f"divergence_{crypto_col}_{credit_col}"
        signal_col = f"divergence_signal_{crypto_col}_{credit_col}"

        self.df[divergence_col] = divergence_zscore
        self.df[signal_col] = (np.abs(divergence_zscore) > divergence_threshold).astype(int)

        logger.info(f"Added momentum divergence: {divergence_col}, {signal_col}")

        return self.df

    def add_flight_to_quality_indicator(
        self,
        crypto_cols: List[str],
        credit_spread_col: str,
        window: int = 20
    ) -> pd.DataFrame:
        """
        Calculate flight-to-quality indicator.

        Measures when money is flowing from risky assets (crypto) to safer
        assets (credit tightening = lower spreads). This typically happens
        during market stress.

        Args:
            crypto_cols: List of crypto price columns
            credit_spread_col: Credit spread column (OAS)
            window: Rolling window for calculation

        Returns:
            DataFrame with flight-to-quality indicator

        Creates:
            - ftq_indicator: Flight-to-quality score
            - ftq_signal: Binary signal (1 = flight to quality occurring)

        Interpretation:
            Positive FTQ: Crypto falling + credit spreads tightening = flight to safety
            Negative FTQ: Crypto rising + credit spreads widening = risk-on
        """
        # Calculate average crypto momentum
        crypto_returns = pd.DataFrame()
        for col in crypto_cols:
            if col in self.df.columns:
                crypto_returns[col] = self.df[col].pct_change(window)

        avg_crypto_return = crypto_returns.mean(axis=1)

        # Calculate credit spread change (negative = tightening = safer)
        credit_change = self.df[credit_spread_col].pct_change(window)

        # Flight to quality: crypto falling AND spreads tightening
        # FTQ score: -crypto_return - credit_change
        # High FTQ: crypto down, spreads down (flight to safety)
        ftq_score = -avg_crypto_return - credit_change

        # Normalize to z-score
        ftq_mean = ftq_score.rolling(window=60).mean()
        ftq_std = ftq_score.rolling(window=60).std()
        ftq_zscore = (ftq_score - ftq_mean) / (ftq_std + 1e-8)

        self.df["ftq_indicator"] = ftq_zscore
        self.df["ftq_signal"] = (ftq_zscore > 1.5).astype(int)  # Significant flight to quality

        logger.info("Added flight-to-quality indicator")

        return self.df

    def add_cross_asset_volatility_ratio(
        self,
        crypto_col: str,
        credit_col: str,
        window: int = 20
    ) -> pd.DataFrame:
        """
        Calculate ratio of crypto volatility to credit volatility.

        High ratio indicates crypto market more volatile than credit (normal).
        Decreasing ratio can signal crypto stabilizing or credit stress increasing.

        Args:
            crypto_col: Crypto price column
            credit_col: Credit spread column
            window: Rolling window for volatility calculation

        Returns:
            DataFrame with volatility ratio
        """
        crypto_vol = self.df[crypto_col].pct_change().rolling(window).std()
        credit_vol = self.df[credit_col].pct_change().rolling(window).std()

        vol_ratio_col = f"vol_ratio_{crypto_col}_{credit_col}_{window}"
        self.df[vol_ratio_col] = crypto_vol / (credit_vol + 1e-8)

        # Z-score of volatility ratio (detect unusual regimes)
        vol_ratio_mean = self.df[vol_ratio_col].rolling(window=60).mean()
        vol_ratio_std = self.df[vol_ratio_col].rolling(window=60).std()
        self.df[f"{vol_ratio_col}_zscore"] = (
            (self.df[vol_ratio_col] - vol_ratio_mean) / (vol_ratio_std + 1e-8)
        )

        logger.info(f"Added volatility ratio: {vol_ratio_col}")

        return self.df

    def add_regime_detection(
        self,
        crypto_col: str,
        credit_col: str,
        lookback: int = 60
    ) -> pd.DataFrame:
        """
        Comprehensive regime detection combining multiple signals.

        Detects market regimes:
        - 2: Strong risk-on (crypto rising, spreads tightening)
        - 1: Mild risk-on
        - 0: Neutral
        - -1: Mild risk-off
        - -2: Strong risk-off (crypto falling, spreads widening)

        Args:
            crypto_col: Crypto price column
            credit_col: Credit spread column
            lookback: Lookback window for regime classification

        Returns:
            DataFrame with regime column
        """
        # Calculate components
        crypto_momentum = self.df[crypto_col].pct_change(lookback)
        credit_momentum = self.df[credit_col].pct_change(lookback)

        # Normalize to [-1, 1] range
        crypto_signal = np.sign(crypto_momentum) * np.minimum(np.abs(crypto_momentum) / 0.1, 1)
        credit_signal = -np.sign(credit_momentum) * np.minimum(np.abs(credit_momentum) / 0.1, 1)

        # Combine signals
        regime_score = crypto_signal + credit_signal

        # Classify into 5 regimes
        regime = pd.Series(0, index=self.df.index)
        regime[regime_score > 1.5] = 2  # Strong risk-on
        regime[(regime_score > 0.5) & (regime_score <= 1.5)] = 1  # Mild risk-on
        regime[(regime_score >= -0.5) & (regime_score <= 0.5)] = 0  # Neutral
        regime[(regime_score >= -1.5) & (regime_score < -0.5)] = -1  # Mild risk-off
        regime[regime_score < -1.5] = -2  # Strong risk-off

        self.df[f"regime_{crypto_col}_{credit_col}"] = regime

        logger.info(f"Added comprehensive regime detection: regime_{crypto_col}_{credit_col}")

        return self.df

    def add_all_cross_asset_features(
        self,
        crypto_cols: List[str],
        credit_cols: List[str],
        correlation_windows: List[int] = [20, 60, 120],
        momentum_window: int = 20
    ) -> pd.DataFrame:
        """
        Add all cross-asset features for comprehensive analysis.

        This is a convenience function that adds all relevant cross-asset
        features for each crypto-credit pair.

        Args:
            crypto_cols: List of crypto price columns
            credit_cols: List of credit spread columns
            correlation_windows: Windows for correlation analysis
            momentum_window: Window for momentum calculations

        Returns:
            DataFrame with all cross-asset features
        """
        logger.info("Adding comprehensive cross-asset features...")

        # For each crypto-credit pair
        for crypto_col in crypto_cols:
            for credit_col in credit_cols:
                if crypto_col in self.df.columns and credit_col in self.df.columns:
                    logger.info(f"Processing pair: {crypto_col} vs {credit_col}")

                    # Add correlation features
                    self.add_rolling_correlation(crypto_col, credit_col, windows=correlation_windows)
                    self.add_correlation_regime(crypto_col, credit_col)

                    # Add momentum features
                    self.add_momentum_divergence(crypto_col, credit_col, momentum_window=momentum_window)

                    # Add volatility features
                    self.add_cross_asset_volatility_ratio(crypto_col, credit_col)

                    # Add regime detection
                    self.add_regime_detection(crypto_col, credit_col)

        # Add flight-to-quality indicator (uses all crypto vs primary credit spread)
        if crypto_cols and credit_cols:
            primary_credit = credit_cols[0]
            if primary_credit in self.df.columns:
                valid_crypto_cols = [col for col in crypto_cols if col in self.df.columns]
                if valid_crypto_cols:
                    self.add_flight_to_quality_indicator(valid_crypto_cols, primary_credit)

        logger.info("Completed adding cross-asset features")

        return self.df

    def get_feature_summary(self) -> pd.DataFrame:
        """
        Get summary statistics of all cross-asset features.

        Returns:
            DataFrame with feature names, mean, std, min, max
        """
        # Find all cross-asset feature columns
        cross_asset_cols = [
            col for col in self.df.columns
            if any(prefix in col for prefix in ['corr_', 'regime_', 'divergence_', 'ftq_', 'vol_ratio_', 'momentum_'])
        ]

        if not cross_asset_cols:
            return pd.DataFrame()

        summary = self.df[cross_asset_cols].describe().T
        summary['null_count'] = self.df[cross_asset_cols].isnull().sum()
        summary['null_pct'] = (summary['null_count'] / len(self.df) * 100).round(2)

        return summary


def identify_crypto_credit_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Automatically identify crypto and credit columns from a DataFrame.

    Args:
        df: DataFrame with mixed crypto and credit data

    Returns:
        Tuple of (crypto_columns, credit_columns)

    Example:
        >>> crypto_cols, credit_cols = identify_crypto_credit_columns(df)
        >>> print(crypto_cols)  # ['BTC_USDT_close', 'ETH_USDT_close']
        >>> print(credit_cols)  # ['LF98TRUU_OAS', 'LUACTRUU_OAS']
    """
    crypto_cols = []
    credit_cols = []

    for col in df.columns:
        # Crypto patterns: contains USDT, BTC, ETH, or common crypto pairs
        if any(token in col.upper() for token in ['USDT', 'BTC', 'ETH', 'SOL', 'ADA', 'XRP', 'DOT']):
            # Prefer close prices for analysis
            if 'close' in col.lower() or 'price' in col.lower():
                crypto_cols.append(col)

        # Credit patterns: contains OAS, DTS, YIELD, or Bloomberg index notation
        elif any(token in col.upper() for token in ['OAS', 'DTS', 'YIELD', 'SPREAD', 'INDEX']):
            credit_cols.append(col)

    return crypto_cols, credit_cols
