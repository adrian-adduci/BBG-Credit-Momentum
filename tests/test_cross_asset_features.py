"""
Unit tests for cross-asset feature engineering.

Tests the indicators/cross_asset.py module which calculates relationships
between crypto and traditional securities.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from indicators.cross_asset import (
    CrossAssetIndicators,
    identify_crypto_credit_columns
)


class TestCrossAssetIndicators:
    """Test suite for CrossAssetIndicators class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample mixed portfolio data for testing."""
        dates = pd.date_range(start='2024-01-01', periods=200, freq='D')

        # Create correlated crypto and credit data
        np.random.seed(42)
        crypto_trend = np.cumsum(np.random.randn(200) * 10) + 50000
        credit_spread = 100 + np.cumsum(np.random.randn(200) * 2)

        df = pd.DataFrame({
            'Dates': dates,
            'BTC_USDT_close': crypto_trend + np.random.randn(200) * 100,
            'ETH_USDT_close': crypto_trend * 0.05 + np.random.randn(200) * 10,
            'LF98TRUU_Index_OAS': credit_spread + np.random.randn(200) * 5,
            'LUACTRUU_Index_OAS': credit_spread * 1.2 + np.random.randn(200) * 8,
        })

        return df

    def test_initialization(self, sample_data):
        """Test CrossAssetIndicators initialization."""
        indicators = CrossAssetIndicators(sample_data)
        assert indicators.df is not None
        assert len(indicators.df) == len(sample_data)
        assert not indicators.df.equals(sample_data)  # Should be a copy

    def test_rolling_correlation(self, sample_data):
        """Test rolling correlation calculation."""
        indicators = CrossAssetIndicators(sample_data)

        df_result = indicators.add_rolling_correlation(
            col1='BTC_USDT_close',
            col2='LF98TRUU_Index_OAS',
            windows=[20, 60]
        )

        # Check new columns created
        assert 'corr_BTC_USDT_close_LF98TRUU_Index_OAS_20' in df_result.columns
        assert 'corr_BTC_USDT_close_LF98TRUU_Index_OAS_60' in df_result.columns

        # Check correlation values are in valid range [-1, 1]
        corr_20 = df_result['corr_BTC_USDT_close_LF98TRUU_Index_OAS_20'].dropna()
        assert (corr_20 >= -1).all() and (corr_20 <= 1).all()

    def test_rolling_correlation_invalid_column(self, sample_data):
        """Test rolling correlation with invalid column name."""
        indicators = CrossAssetIndicators(sample_data)

        with pytest.raises(ValueError, match="Column .* not found"):
            indicators.add_rolling_correlation(
                col1='INVALID_COLUMN',
                col2='LF98TRUU_Index_OAS',
                windows=[20]
            )

    def test_correlation_regime(self, sample_data):
        """Test correlation regime detection."""
        indicators = CrossAssetIndicators(sample_data)

        df_result = indicators.add_correlation_regime(
            col1='BTC_USDT_close',
            col2='LF98TRUU_Index_OAS',
            window=60,
            threshold_positive=0.3,
            threshold_negative=-0.3
        )

        # Check regime column created
        regime_col = 'regime_BTC_USDT_close_LF98TRUU_Index_OAS_60'
        assert regime_col in df_result.columns

        # Check regime values are -1, 0, or 1
        regime_values = df_result[regime_col].dropna().unique()
        assert all(val in [-1, 0, 1] for val in regime_values)

    def test_momentum_divergence(self, sample_data):
        """Test momentum divergence calculation."""
        indicators = CrossAssetIndicators(sample_data)

        df_result = indicators.add_momentum_divergence(
            crypto_col='BTC_USDT_close',
            credit_col='LF98TRUU_Index_OAS',
            momentum_window=20
        )

        # Check columns created
        assert 'momentum_BTC_USDT_close_20' in df_result.columns
        assert 'momentum_LF98TRUU_Index_OAS_20' in df_result.columns
        assert 'divergence_BTC_USDT_close_LF98TRUU_Index_OAS' in df_result.columns
        assert 'divergence_signal_BTC_USDT_close_LF98TRUU_Index_OAS' in df_result.columns

        # Check signal is binary
        signal_col = 'divergence_signal_BTC_USDT_close_LF98TRUU_Index_OAS'
        signal_values = df_result[signal_col].dropna().unique()
        assert all(val in [0, 1] for val in signal_values)

    def test_flight_to_quality_indicator(self, sample_data):
        """Test flight-to-quality indicator."""
        indicators = CrossAssetIndicators(sample_data)

        df_result = indicators.add_flight_to_quality_indicator(
            crypto_cols=['BTC_USDT_close', 'ETH_USDT_close'],
            credit_spread_col='LF98TRUU_Index_OAS',
            window=20
        )

        # Check columns created
        assert 'ftq_indicator' in df_result.columns
        assert 'ftq_signal' in df_result.columns

        # Check signal is binary
        ftq_signal = df_result['ftq_signal'].dropna().unique()
        assert all(val in [0, 1] for val in ftq_signal)

    def test_volatility_ratio(self, sample_data):
        """Test cross-asset volatility ratio."""
        indicators = CrossAssetIndicators(sample_data)

        df_result = indicators.add_cross_asset_volatility_ratio(
            crypto_col='BTC_USDT_close',
            credit_col='LF98TRUU_Index_OAS',
            window=20
        )

        # Check columns created
        vol_ratio_col = 'vol_ratio_BTC_USDT_close_LF98TRUU_Index_OAS_20'
        assert vol_ratio_col in df_result.columns
        assert f'{vol_ratio_col}_zscore' in df_result.columns

        # Check volatility ratio is positive
        vol_ratio = df_result[vol_ratio_col].dropna()
        assert (vol_ratio > 0).all()

    def test_regime_detection(self, sample_data):
        """Test comprehensive regime detection."""
        indicators = CrossAssetIndicators(sample_data)

        df_result = indicators.add_regime_detection(
            crypto_col='BTC_USDT_close',
            credit_col='LF98TRUU_Index_OAS',
            lookback=60
        )

        # Check regime column created
        regime_col = 'regime_BTC_USDT_close_LF98TRUU_Index_OAS'
        assert regime_col in df_result.columns

        # Check regime values are in [-2, -1, 0, 1, 2]
        regime_values = df_result[regime_col].dropna().unique()
        assert all(val in [-2, -1, 0, 1, 2] for val in regime_values)

    def test_add_all_cross_asset_features(self, sample_data):
        """Test adding all cross-asset features at once."""
        indicators = CrossAssetIndicators(sample_data)

        df_result = indicators.add_all_cross_asset_features(
            crypto_cols=['BTC_USDT_close', 'ETH_USDT_close'],
            credit_cols=['LF98TRUU_Index_OAS', 'LUACTRUU_Index_OAS'],
            correlation_windows=[20, 60],
            momentum_window=20
        )

        # Check various feature types were added
        assert any(col.startswith('corr_') for col in df_result.columns)
        assert any('regime' in col for col in df_result.columns)
        assert any('divergence' in col for col in df_result.columns)
        assert any('vol_ratio' in col for col in df_result.columns)
        assert 'ftq_indicator' in df_result.columns

    def test_feature_summary(self, sample_data):
        """Test feature summary generation."""
        indicators = CrossAssetIndicators(sample_data)

        # Add some features first
        indicators.add_rolling_correlation(
            'BTC_USDT_close',
            'LF98TRUU_Index_OAS',
            windows=[20]
        )

        summary = indicators.get_feature_summary()

        # Check summary contains expected columns
        assert 'mean' in summary.columns
        assert 'std' in summary.columns
        assert 'null_count' in summary.columns
        assert 'null_pct' in summary.columns

        # Check at least one feature in summary
        assert len(summary) > 0


class TestIdentifyCryptoCreditColumns:
    """Test suite for column identification helper."""

    def test_identify_crypto_columns(self):
        """Test identification of crypto columns."""
        df = pd.DataFrame({
            'BTC_USDT_close': [1, 2, 3],
            'ETH_USDT_close': [1, 2, 3],
            'BTC_USDT_volume': [100, 200, 300],
            'LF98TRUU_Index_OAS': [50, 51, 52],
        })

        crypto_cols, credit_cols = identify_crypto_credit_columns(df)

        assert 'BTC_USDT_close' in crypto_cols
        assert 'ETH_USDT_close' in crypto_cols
        assert 'BTC_USDT_volume' not in crypto_cols  # Only close/price columns

    def test_identify_credit_columns(self):
        """Test identification of credit columns."""
        df = pd.DataFrame({
            'BTC_USDT_close': [1, 2, 3],
            'LF98TRUU_Index_OAS': [50, 51, 52],
            'LUACTRUU_Index_DTS': [5, 5.1, 5.2],
            'YIELD_CURVE': [2.5, 2.6, 2.7],
        })

        crypto_cols, credit_cols = identify_crypto_credit_columns(df)

        assert 'LF98TRUU_Index_OAS' in credit_cols
        assert 'LUACTRUU_Index_DTS' in credit_cols
        assert 'YIELD_CURVE' in credit_cols

    def test_no_crypto_or_credit(self):
        """Test when no crypto or credit columns present."""
        df = pd.DataFrame({
            'date': [1, 2, 3],
            'random_column': [100, 200, 300],
        })

        crypto_cols, credit_cols = identify_crypto_credit_columns(df)

        assert len(crypto_cols) == 0
        assert len(credit_cols) == 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        indicators = CrossAssetIndicators(df)

        # Should not raise error, but features won't be added
        assert len(indicators.df) == 0

    def test_insufficient_data_for_windows(self):
        """Test with insufficient data for rolling windows."""
        df = pd.DataFrame({
            'BTC_USDT_close': [1, 2, 3, 4, 5],
            'LF98TRUU_Index_OAS': [50, 51, 52, 53, 54],
        })

        indicators = CrossAssetIndicators(df)

        # Should work but have many NaN values
        df_result = indicators.add_rolling_correlation(
            'BTC_USDT_close',
            'LF98TRUU_Index_OAS',
            windows=[20]  # More than available data
        )

        # Check column exists but contains NaN
        corr_col = 'corr_BTC_USDT_close_LF98TRUU_Index_OAS_20'
        assert corr_col in df_result.columns
        assert df_result[corr_col].isna().all()  # All NaN due to insufficient data

    def test_missing_values_handling(self):
        """Test handling of missing values in input data."""
        df = pd.DataFrame({
            'BTC_USDT_close': [1, 2, np.nan, 4, 5, 6, 7, 8],
            'LF98TRUU_Index_OAS': [50, np.nan, 52, 53, 54, 55, 56, 57],
        })

        indicators = CrossAssetIndicators(df)

        # Should handle NaN gracefully
        df_result = indicators.add_rolling_correlation(
            'BTC_USDT_close',
            'LF98TRUU_Index_OAS',
            windows=[3]
        )

        # Correlation calculation should skip NaN
        assert 'corr_BTC_USDT_close_LF98TRUU_Index_OAS_3' in df_result.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
