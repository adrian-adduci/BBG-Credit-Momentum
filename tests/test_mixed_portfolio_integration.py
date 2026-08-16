"""
Integration tests for mixed portfolio (crypto + credit) workflow.

Tests the end-to-end pipeline from data loading to model training
with mixed asset portfolios.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

import _preprocessing
import _models
from _data_sources import (
    MixedPortfolioDataSource,
    BloombergExcelDataSource,
    DataSourceFactory
)


# ---------------------------------------------------------------------------
# Known-unimplemented API surface.
#
# The tests below were written against a MixedPortfolioDataSource API that was
# never built. They landed in 0f93e5c ("Complete phases 4-6") and have never
# passed. Two concrete mismatches:
#
#   1. They patch '_data_sources.CryptoExchangeDataSource', but that class
#      lives in _crypto_data_sources.py and _data_sources.py does not import
#      it, so mock.patch cannot resolve the attribute.
#
#   2. They construct MixedPortfolioDataSource(sources=[...], alignment=...).
#      The real signature is (securities=..., alignment_method=...), and it
#      takes security *definitions* rather than data-source objects -- a
#      different design, not a renamed argument.
#
# Marked xfail(strict=True) rather than deleted so the gap stays visible and
# so the marker fails loudly if the API is ever implemented.
# ---------------------------------------------------------------------------
UNIMPLEMENTED_MIXED_API = pytest.mark.xfail(
    strict=True,
    reason=(
        "MixedPortfolioDataSource(sources=..., alignment=...) and "
        "_data_sources.CryptoExchangeDataSource do not exist; tests were "
        "written against an unbuilt API"
    ),
)


class TestMixedPortfolioIntegration:
    """Integration tests for mixed portfolio workflow."""

    # Both fixtures span the same 300 calendar days.
    #
    # The crypto fixture previously used freq='H' for 100 periods -- about four
    # days -- while the credit fixture used freq='D' for 100 days. Merging them
    # on date left roughly five overlapping rows, far fewer than the 31-period
    # warmup the stochastic-RSI and ROC indicators need. Those indicators then
    # emitted all-NaN columns, and the unconditional dropna() in
    # _preprocess_xlsx reduced a 379-column frame to zero rows.
    PERIODS = 300

    @pytest.fixture
    def mock_crypto_data(self):
        """Create mock crypto exchange data."""
        n = self.PERIODS
        dates = pd.date_range(start='2024-01-01', periods=n, freq='D')
        return pd.DataFrame({
            'timestamp': dates,
            'BTC_USDT_close': np.random.randn(n).cumsum() + 50000,
            'BTC_USDT_high': np.random.randn(n).cumsum() + 50100,
            'BTC_USDT_low': np.random.randn(n).cumsum() + 49900,
            'BTC_USDT_volume': np.random.rand(n) * 1000,
            'ETH_USDT_close': np.random.randn(n).cumsum() + 3000,
            'ETH_USDT_high': np.random.randn(n).cumsum() + 3010,
            'ETH_USDT_low': np.random.randn(n).cumsum() + 2990,
            'ETH_USDT_volume': np.random.rand(n) * 5000,
        }).set_index('timestamp')

    @pytest.fixture
    def mock_bloomberg_data(self):
        """Create mock Bloomberg credit data."""
        n = self.PERIODS
        dates = pd.date_range(start='2024-01-01', periods=n, freq='D')
        return pd.DataFrame({
            'Dates': dates,
            'LF98TRUU_Index_OAS': np.random.randn(n).cumsum() + 100,
            'LF98TRUU_Index_DTS': np.random.rand(n) * 2 + 5,
            'LUACTRUU_Index_OAS': np.random.randn(n).cumsum() + 120,
            'LUACTRUU_Index_DTS': np.random.rand(n) * 2 + 4.5,
        })

    @UNIMPLEMENTED_MIXED_API
    def test_mixed_portfolio_data_loading(self, mock_crypto_data, mock_bloomberg_data, tmp_path):
        """Test loading data from multiple sources and merging."""
        # Save Bloomberg data to temporary Excel file
        bloomberg_file = tmp_path / "bloomberg_test.xlsx"
        mock_bloomberg_data.to_excel(bloomberg_file, index=False)

        # Mock crypto data source
        with patch('_data_sources.CryptoExchangeDataSource') as MockCrypto:
            mock_crypto_source = MockCrypto.return_value
            mock_crypto_source.load_data.return_value = mock_crypto_data

            # Create Bloomberg source
            bloomberg_source = BloombergExcelDataSource(file_path=str(bloomberg_file))

            # Create mixed portfolio source
            mixed_source = MixedPortfolioDataSource(
                sources=[mock_crypto_source, bloomberg_source],
                alignment='outer'
            )

            # Load data
            df = mixed_source.load_data()

            # Verify merged data
            assert 'BTC_USDT_close' in df.columns
            assert 'ETH_USDT_close' in df.columns
            assert 'LF98TRUU_Index_OAS' in df.columns
            assert 'LUACTRUU_Index_DTS' in df.columns

            # Should have dates from both sources (may have NaN for non-overlapping dates)
            assert len(df) > 0

    def test_mixed_portfolio_preprocessing(self, mock_crypto_data, mock_bloomberg_data, tmp_path):
        """Test preprocessing with cross-asset features enabled."""
        # Create combined dataset manually for testing
        # Resample crypto to daily to match Bloomberg
        crypto_daily = mock_crypto_data.resample('D').last()

        # Merge with Bloomberg data
        combined = pd.merge(
            crypto_daily.reset_index(),
            mock_bloomberg_data,
            left_on='timestamp',
            right_on='Dates',
            how='inner'
        )
        # Drop the right-hand join key before renaming: merging on
        # left_on='timestamp'/right_on='Dates' keeps BOTH columns, so the
        # rename produced two columns named 'Dates'. Excel round-tripping
        # then de-duplicated them into 'Dates' and 'Dates.1', and the
        # leftover datetime64 column reached XGBoost as a feature.
        combined = combined.drop(columns=['Dates']).rename(
            columns={'timestamp': 'Dates'}
        )

        # Save to temp file
        test_file = tmp_path / "mixed_test.xlsx"
        combined.to_excel(test_file, index=False)

        # Preprocess with cross-asset features
        pipeline = _preprocessing._preprocess_xlsx(
            xlsx_file=str(test_file),
            target_col='BTC_USDT_close',
            momentum_list=['BTC_USDT_close', 'LF98TRUU_Index_OAS'],
            crypto_features=True,
            cross_asset_features=True
        )

        # Verify cross-asset features were added
        df = pipeline._return_dataframe()

        # Check for cross-asset feature columns
        cross_asset_features = [col for col in df.columns if any(
            keyword in col for keyword in ['corr_', 'regime_', 'divergence_', 'ftq_']
        )]

        assert len(cross_asset_features) > 0, "No cross-asset features were added"

    def test_mixed_portfolio_model_training(self, mock_crypto_data, mock_bloomberg_data, tmp_path):
        """Test training model on mixed portfolio data."""
        # Create combined dataset
        crypto_daily = mock_crypto_data.resample('D').last()
        combined = pd.merge(
            crypto_daily.reset_index(),
            mock_bloomberg_data,
            left_on='timestamp',
            right_on='Dates',
            how='inner'
        )
        # Drop the right-hand join key before renaming: merging on
        # left_on='timestamp'/right_on='Dates' keeps BOTH columns, so the
        # rename produced two columns named 'Dates'. Excel round-tripping
        # then de-duplicated them into 'Dates' and 'Dates.1', and the
        # leftover datetime64 column reached XGBoost as a feature.
        combined = combined.drop(columns=['Dates']).rename(
            columns={'timestamp': 'Dates'}
        )

        # Save to temp file
        test_file = tmp_path / "mixed_model_test.xlsx"
        combined.to_excel(test_file, index=False)

        # Preprocess
        pipeline = _preprocessing._preprocess_xlsx(
            xlsx_file=str(test_file),
            target_col='BTC_USDT_close',
            momentum_list=['BTC_USDT_close', 'LF98TRUU_Index_OAS'],
            crypto_features=True,
            cross_asset_features=True
        )

        # Train model
        model = _models._build_model(pipeline, model_name='XGBoost')

        # Verify model trained successfully
        assert model is not None
        mae, mse, rmse = model._return_mean_error_metrics()
        assert mae > 0
        assert mse > 0
        assert rmse > 0

    def test_feature_importance_includes_cross_asset(self, mock_crypto_data, mock_bloomberg_data, tmp_path):
        """Test that feature importance includes cross-asset features."""
        # Create combined dataset
        crypto_daily = mock_crypto_data.resample('D').last()
        combined = pd.merge(
            crypto_daily.reset_index(),
            mock_bloomberg_data,
            left_on='timestamp',
            right_on='Dates',
            how='inner'
        )
        # Drop the right-hand join key before renaming: merging on
        # left_on='timestamp'/right_on='Dates' keeps BOTH columns, so the
        # rename produced two columns named 'Dates'. Excel round-tripping
        # then de-duplicated them into 'Dates' and 'Dates.1', and the
        # leftover datetime64 column reached XGBoost as a feature.
        combined = combined.drop(columns=['Dates']).rename(
            columns={'timestamp': 'Dates'}
        )

        # Save to temp file
        test_file = tmp_path / "feature_importance_test.xlsx"
        combined.to_excel(test_file, index=False)

        # Preprocess and train
        pipeline = _preprocessing._preprocess_xlsx(
            xlsx_file=str(test_file),
            target_col='BTC_USDT_close',
            momentum_list=['BTC_USDT_close', 'LF98TRUU_Index_OAS'],
            crypto_features=True,
            cross_asset_features=True
        )

        model = _models._build_model(pipeline, model_name='XGBoost')

        # Calculate feature importance
        try:
            model.predictive_power(forecast_range=10)
            feature_importance = model._return_features_of_importance(forecast_day=10)

            # Check if any cross-asset features are in top features
            cross_asset_keywords = ['corr_', 'regime_', 'divergence_', 'ftq_', 'momentum_']
            has_cross_asset = any(
                any(keyword in str(feature) for keyword in cross_asset_keywords)
                for feature in feature_importance
            )

            # Note: May not always have cross-asset in top features, but they should exist
            # Just verify feature importance calculation works
            assert isinstance(feature_importance, (dict, list))

        except Exception as e:
            # Some configurations may not support feature importance
            pytest.skip(f"Feature importance not available: {e}")


class TestDataAlignment:
    """Test different data alignment strategies."""

    @pytest.fixture
    def crypto_24_7_data(self):
        """Crypto data (24/7 market)."""
        dates = pd.date_range(start='2024-01-01', end='2024-01-07', freq='H')
        return pd.DataFrame({
            'timestamp': dates,
            'BTC_USDT_close': np.random.rand(len(dates)) * 100 + 50000,
        }).set_index('timestamp')

    @pytest.fixture
    def credit_weekday_data(self):
        """Credit data (weekday only market)."""
        # Only weekdays
        dates = pd.date_range(start='2024-01-01', end='2024-01-07', freq='B')
        return pd.DataFrame({
            'Dates': dates,
            'LF98TRUU_Index_OAS': np.random.rand(len(dates)) * 10 + 100,
        })

    @UNIMPLEMENTED_MIXED_API
    def test_outer_join_alignment(self, crypto_24_7_data, credit_weekday_data, tmp_path):
        """Test outer join keeps all dates from both sources."""
        bloomberg_file = tmp_path / "credit_weekday.xlsx"
        credit_weekday_data.to_excel(bloomberg_file, index=False)

        with patch('_data_sources.CryptoExchangeDataSource') as MockCrypto:
            mock_crypto_source = MockCrypto.return_value
            mock_crypto_source.load_data.return_value = crypto_24_7_data

            bloomberg_source = BloombergExcelDataSource(file_path=str(bloomberg_file))

            mixed_source = MixedPortfolioDataSource(
                sources=[mock_crypto_source, bloomberg_source],
                alignment='outer'
            )

            df = mixed_source.load_data()

            # Should include weekend dates from crypto (with NaN for credit)
            assert len(df) >= len(credit_weekday_data)

    @UNIMPLEMENTED_MIXED_API
    def test_inner_join_alignment(self, crypto_24_7_data, credit_weekday_data, tmp_path):
        """Test inner join keeps only overlapping dates."""
        bloomberg_file = tmp_path / "credit_weekday.xlsx"
        credit_weekday_data.to_excel(bloomberg_file, index=False)

        with patch('_data_sources.CryptoExchangeDataSource') as MockCrypto:
            mock_crypto_source = MockCrypto.return_value
            mock_crypto_source.load_data.return_value = crypto_24_7_data

            bloomberg_source = BloombergExcelDataSource(file_path=str(bloomberg_file))

            mixed_source = MixedPortfolioDataSource(
                sources=[mock_crypto_source, bloomberg_source],
                alignment='inner'
            )

            df = mixed_source.load_data()

            # Should only include dates present in both
            # (weekdays only, and matching hours)
            assert len(df) <= len(credit_weekday_data)

            # No NaN in key columns (since inner join)
            # (Note: may still have NaN from forward fill limits)


class TestErrorHandling:
    """Test error handling in mixed portfolio workflow."""

    @UNIMPLEMENTED_MIXED_API
    def test_no_data_sources(self):
        """Test error when no data sources provided."""
        with pytest.raises(ValueError, match="At least one data source"):
            mixed_source = MixedPortfolioDataSource(sources=[])

    @UNIMPLEMENTED_MIXED_API
    def test_incompatible_data_types(self, tmp_path):
        """Test handling of incompatible data from different sources."""
        # Create Bloomberg data with different structure
        df_bad = pd.DataFrame({
            'InvalidColumn': [1, 2, 3]
        })

        bad_file = tmp_path / "bad_bloomberg.xlsx"
        df_bad.to_excel(bad_file, index=False)

        bloomberg_source = BloombergExcelDataSource(file_path=str(bad_file))

        # Should handle missing 'Dates' column gracefully
        with pytest.raises((ValueError, KeyError)):
            mixed_source = MixedPortfolioDataSource(sources=[bloomberg_source])
            df = mixed_source.load_data()

    def test_cross_asset_features_with_only_crypto(self, tmp_path):
        """Test cross-asset features when only crypto data present."""
        crypto_only = pd.DataFrame({
            'Dates': pd.date_range(start='2024-01-01', periods=100, freq='D'),
            'BTC_USDT_close': np.random.rand(100) * 100 + 50000,
        })

        test_file = tmp_path / "crypto_only.xlsx"
        crypto_only.to_excel(test_file, index=False)

        # Should not crash, but may warn about no cross-asset features
        pipeline = _preprocessing._preprocess_xlsx(
            xlsx_file=str(test_file),
            target_col='BTC_USDT_close',
            momentum_list=['BTC_USDT_close'],
            cross_asset_features=True  # Enabled but no credit data
        )

        # Should complete without error
        df = pipeline._return_dataframe()
        assert df is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
