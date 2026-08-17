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
    DataSourceFactory,
    Security,
)


# ---------------------------------------------------------------------------
# Helpers for driving the real MixedPortfolioDataSource API.
#
# These tests previously targeted an API that was never built: they patched
# _data_sources.CryptoExchangeDataSource (that class lives in
# _crypto_data_sources.py) and constructed
# MixedPortfolioDataSource(sources=[...], alignment=...).
#
# The real constructor takes Security *definitions* plus a date range, and
# load_data() resolves each one through self._load_security_data(). That
# method is the seam, so it is what these tests patch -- no network, no
# exchange credentials, and no dependence on which concrete source class a
# given security happens to route to.
# ---------------------------------------------------------------------------
START = datetime(2024, 1, 1)
END = datetime(2024, 12, 31)


def crypto_security(identifier="BTC/USDT"):
    return Security(
        identifier=identifier,
        security_type="crypto_spot",
        source="binance",
        fields=["close"],
    )


def credit_security(identifier="LF98TRUU Index"):
    return Security(
        identifier=identifier,
        security_type="credit_index",
        source="bloomberg",
        fields=["OAS"],
    )


def mixed_source(securities, **kwargs):
    kwargs.setdefault("alignment_method", "outer")
    kwargs.setdefault("validate", False)
    return MixedPortfolioDataSource(
        securities=securities, start_date=START, end_date=END, **kwargs
    )


def load_with(securities, frames_by_identifier, **kwargs):
    """Run load_data() with _load_security_data stubbed per security."""
    source = mixed_source(securities, **kwargs)
    with patch.object(
        MixedPortfolioDataSource,
        "_load_security_data",
        side_effect=lambda sec: frames_by_identifier[sec.identifier],
    ):
        return source.load_data()


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

    def test_mixed_portfolio_data_loading(self, mock_crypto_data, mock_bloomberg_data):
        """Securities from different sources merge into one aligned frame."""
        crypto = crypto_security()
        credit = credit_security()

        crypto_frame = mock_crypto_data.reset_index().rename(
            columns={"timestamp": "Dates"}
        )[["Dates", "BTC_USDT_close", "ETH_USDT_close"]]
        credit_frame = mock_bloomberg_data[
            ["Dates", "LF98TRUU_Index_OAS", "LUACTRUU_Index_DTS"]
        ]

        df = load_with(
            [crypto, credit],
            {crypto.identifier: crypto_frame, credit.identifier: credit_frame},
        )

        for column in (
            "BTC_USDT_close",
            "ETH_USDT_close",
            "LF98TRUU_Index_OAS",
            "LUACTRUU_Index_DTS",
        ):
            assert column in df.columns

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

        # Calculate feature importance.
        #
        # This block used to be wrapped in `except Exception: pytest.skip(...)`,
        # which turned real defects into a green run. It was concealing two of
        # them: _return_features_of_importance did not exist at all, and
        # predictive_power handed ppscore a frame that did not contain the
        # column it was told to score. Let failures fail.
        scores = model.predictive_power(forecast_range=10, plot=False)
        assert not scores.empty

        feature_importance = model._return_features_of_importance(forecast_day=10)
        assert isinstance(feature_importance, dict)

        # The raw target must never be reported as a predictor of itself.
        assert 'BTC_USDT_close' not in feature_importance

        # Cross-asset engineering was requested, so those features must at
        # least be present as candidates in the scored set.
        cross_asset_keywords = ['corr_', 'regime_', 'divergence_', 'ftq_', 'momentum_']
        scored_features = set(scores['x'].unique())
        assert any(
            keyword in str(feature)
            for feature in scored_features
            for keyword in cross_asset_keywords
        ), f"no cross-asset features among {len(scored_features)} scored candidates"


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

    def test_outer_join_alignment(self, crypto_24_7_data, credit_weekday_data):
        """Outer join keeps every date from both calendars."""
        crypto = crypto_security()
        credit = credit_security()
        crypto_frame = crypto_24_7_data.reset_index().rename(
            columns={"timestamp": "Dates"}
        )

        df = load_with(
            [crypto, credit],
            {crypto.identifier: crypto_frame, credit.identifier: credit_weekday_data},
            alignment_method="outer",
        )

        # Crypto trades weekends; credit does not. An outer join must retain
        # at least the union's larger side.
        assert len(df) >= len(credit_weekday_data)
        assert len(df) >= len(crypto_frame)

    def test_inner_join_alignment(self, crypto_24_7_data, credit_weekday_data):
        """Inner join keeps only dates present in both calendars."""
        crypto = crypto_security()
        credit = credit_security()
        crypto_frame = crypto_24_7_data.reset_index().rename(
            columns={"timestamp": "Dates"}
        )

        df = load_with(
            [crypto, credit],
            {crypto.identifier: crypto_frame, credit.identifier: credit_weekday_data},
            alignment_method="inner",
        )

        assert len(df) <= len(credit_weekday_data)

    def test_inner_join_is_a_subset_of_outer_join(
        self, crypto_24_7_data, credit_weekday_data
    ):
        """The two alignment modes must be consistent with each other."""
        crypto = crypto_security()
        credit = credit_security()
        crypto_frame = crypto_24_7_data.reset_index().rename(
            columns={"timestamp": "Dates"}
        )
        frames = {
            crypto.identifier: crypto_frame,
            credit.identifier: credit_weekday_data,
        }

        outer = load_with([crypto, credit], frames, alignment_method="outer")
        inner = load_with([crypto, credit], frames, alignment_method="inner")

        assert len(inner) <= len(outer)
        assert set(inner["Dates"]).issubset(set(outer["Dates"]))


class TestErrorHandling:
    """Test error handling in mixed portfolio workflow."""

    def test_no_securities_raises(self):
        """An empty universe is rejected when loading, not at construction."""
        source = mixed_source([])

        with pytest.raises(ValueError, match="No securities provided"):
            source.load_data()

    def test_a_failing_security_does_not_pass_silently(self):
        """If every source fails, load_data must raise rather than return empty."""
        crypto = crypto_security()
        source = mixed_source([crypto])

        with patch.object(
            MixedPortfolioDataSource,
            "_load_security_data",
            side_effect=KeyError("Dates"),
        ), pytest.raises((ValueError, KeyError)):
            source.load_data()

    def test_frame_missing_dates_column_is_rejected(self, tmp_path):
        """A source returning a frame with no 'Dates' column cannot be merged."""
        credit = credit_security()
        malformed = pd.DataFrame({"InvalidColumn": [1, 2, 3]})

        with pytest.raises((ValueError, KeyError)):
            load_with([credit], {credit.identifier: malformed})

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
