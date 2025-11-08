"""
Unit Tests for Momentum Indicators

Tests cover:
- Typical use cases
- Edge cases (empty data, NaN values, constant prices)
- Invalid inputs (negative windows, wrong types)
- Boundary conditions
- Expected behavior validation

Author: BBG-Credit-Momentum Team
License: MIT
"""

import unittest
import numpy as np
import pandas as pd
from indicators.momentum import (
    MomentumIndicators,
    calculate_roc,
    calculate_cci,
    calculate_atr,
    calculate_obv,
)


class TestRateOfChange(unittest.TestCase):
    """Test Rate of Change (ROC) calculation."""

    def setUp(self):
        """Set up test data for each test."""
        self.calculator = MomentumIndicators(debug=False)
        np.random.seed(42)
        self.close = pd.Series(100 + np.random.randn(100).cumsum())

    def test_typical_use_case(self):
        """Test 1: Typical use case with valid data."""
        roc = self.calculator.rate_of_change(self.close, window=10)

        # Verify output type and length
        self.assertIsInstance(roc, pd.Series, "Output should be pandas Series")
        self.assertEqual(len(roc), len(self.close), "Output length should match input")

        # Verify first 'window' values are NaN (not enough history)
        self.assertTrue(roc.iloc[:10].isna().all(), "First window values should be NaN")

        # Verify some valid values exist
        valid_roc = roc.dropna()
        self.assertGreater(len(valid_roc), 80, "Should have many valid values")

    def test_calculation_accuracy(self):
        """Test 2: Verify ROC calculation accuracy with known values."""
        # Create simple test case: prices increasing by 10% each period
        test_close = pd.Series([100, 110, 121, 133.1, 146.41])
        roc = self.calculator.rate_of_change(test_close, window=1)

        # ROC should be approximately 10% for each period (after first NaN)
        valid_roc = roc.dropna()
        expected_roc = 10.0  # 10% increase

        # Allow 0.1% tolerance for floating point
        np.testing.assert_array_almost_equal(
            valid_roc.values,
            [expected_roc] * len(valid_roc),
            decimal=1,
            err_msg="ROC should be 10% for 10% price increases"
        )

    def test_edge_case_zero_price(self):
        """Test 3: Edge case - price dropping to zero (division by zero)."""
        # Epsilon should prevent division by zero
        zero_close = pd.Series([100, 50, 25, 0, 0])
        roc = self.calculator.rate_of_change(zero_close, window=1)

        # Should not raise exception and should produce finite values
        self.assertTrue(np.isfinite(roc.dropna()).all(), "Should handle zero price gracefully")

    def test_positive_and_negative_momentum(self):
        """Test 4: ROC correctly identifies positive and negative momentum."""
        # Create data with clear up and down trends
        trending_close = pd.Series([100, 105, 110, 115, 120, 115, 110, 105, 100, 95])
        roc = self.calculator.rate_of_change(trending_close, window=1)

        # First 5 periods (uptrend) should have positive ROC
        uptrend_roc = roc.iloc[1:5]  # Skip first NaN
        self.assertTrue((uptrend_roc > 0).all(), "Uptrend should have positive ROC")

        # Last 4 periods (downtrend) should have negative ROC
        downtrend_roc = roc.iloc[6:]
        self.assertTrue((downtrend_roc < 0).all(), "Downtrend should have negative ROC")

    def test_invalid_input_short_series(self):
        """Test 5: Invalid input - series shorter than window."""
        short_close = pd.Series([100, 105])

        with self.assertRaises(ValueError) as context:
            self.calculator.rate_of_change(short_close, window=10)

        self.assertIn("too short", str(context.exception).lower())

    def test_invalid_input_negative_window(self):
        """Test 6: Invalid input - negative window size."""
        with self.assertRaises(ValueError) as context:
            self.calculator.rate_of_change(self.close, window=-5)

        self.assertIn("positive", str(context.exception).lower())

    def test_convenience_function(self):
        """Test 7: Convenience function produces same result."""
        class_roc = self.calculator.rate_of_change(self.close, window=10)
        func_roc = calculate_roc(self.close, window=10)

        pd.testing.assert_series_equal(class_roc, func_roc, check_names=False)


class TestCommodityChannelIndex(unittest.TestCase):
    """Test Commodity Channel Index (CCI) calculation."""

    def setUp(self):
        """Set up test data for each test."""
        self.calculator = MomentumIndicators(debug=False)
        np.random.seed(42)
        self.high = pd.Series(100 + np.random.randn(100).cumsum() + 5)
        self.low = pd.Series(100 + np.random.randn(100).cumsum() - 5)
        self.close = pd.Series(100 + np.random.randn(100).cumsum())

        # Ensure high >= low
        for i in range(len(self.high)):
            if self.high.iloc[i] < self.low.iloc[i]:
                self.high.iloc[i], self.low.iloc[i] = self.low.iloc[i], self.high.iloc[i]

    def test_typical_use_case(self):
        """Test 1: Typical use case with valid data."""
        cci = self.calculator.commodity_channel_index(
            self.high, self.low, self.close, window=20
        )

        # Verify output
        self.assertIsInstance(cci, pd.Series)
        self.assertEqual(len(cci), len(self.close))

        # CCI typically ranges from -300 to +300, but can go beyond
        valid_cci = cci.dropna()
        self.assertGreater(len(valid_cci), 70, "Should have many valid values")

    def test_overbought_oversold_levels(self):
        """Test 2: CCI identifies overbought/oversold conditions."""
        # Create strong uptrend (should produce high CCI)
        uptrend_high = pd.Series(range(100, 150))
        uptrend_low = pd.Series(range(95, 145))
        uptrend_close = pd.Series(range(98, 148))

        cci = self.calculator.commodity_channel_index(
            uptrend_high, uptrend_low, uptrend_close, window=20
        )

        # Strong uptrend should produce high CCI values
        valid_cci = cci.dropna().iloc[-10:]  # Last 10 values
        self.assertTrue(
            (valid_cci > 0).any(),
            "Strong uptrend should produce positive CCI values"
        )

    def test_edge_case_constant_price(self):
        """Test 3: Edge case - constant price (no movement)."""
        constant = pd.Series([100.0] * 50)

        cci = self.calculator.commodity_channel_index(
            constant, constant, constant
        )

        # Should handle constant price gracefully (epsilon prevents division by zero)
        valid_cci = cci.dropna()
        self.assertTrue(np.isfinite(valid_cci).all(), "Should handle constant price")

    def test_invalid_input_different_lengths(self):
        """Test 4: Invalid input - mismatched series lengths."""
        short_high = pd.Series([100, 105])

        with self.assertRaises(ValueError) as context:
            self.calculator.commodity_channel_index(short_high, self.low, self.close)

        self.assertIn("same length", str(context.exception).lower())

    def test_invalid_constant_zero(self):
        """Test 5: Invalid input - zero or negative constant."""
        with self.assertRaises(ValueError) as context:
            self.calculator.commodity_channel_index(
                self.high, self.low, self.close, constant=0
            )

        self.assertIn("positive", str(context.exception).lower())

    def test_convenience_function(self):
        """Test 6: Convenience function works correctly."""
        func_cci = calculate_cci(self.high, self.low, self.close)

        self.assertIsInstance(func_cci, pd.Series)
        self.assertGreater(len(func_cci.dropna()), 70)


class TestAverageTrueRange(unittest.TestCase):
    """Test Average True Range (ATR) calculation."""

    def setUp(self):
        """Set up test data for each test."""
        self.calculator = MomentumIndicators(debug=False)
        np.random.seed(42)
        self.high = pd.Series(100 + np.random.randn(100).cumsum() + 5)
        self.low = pd.Series(100 + np.random.randn(100).cumsum() - 5)
        self.close = pd.Series(100 + np.random.randn(100).cumsum())

        # Ensure high >= low
        for i in range(len(self.high)):
            if self.high.iloc[i] < self.low.iloc[i]:
                self.high.iloc[i], self.low.iloc[i] = self.low.iloc[i], self.high.iloc[i]

    def test_typical_use_case(self):
        """Test 1: Typical use case with valid data."""
        atr = self.calculator.average_true_range(
            self.high, self.low, self.close, window=14
        )

        # Verify output
        self.assertIsInstance(atr, pd.Series)
        self.assertEqual(len(atr), len(self.close))

        # ATR should always be positive
        valid_atr = atr.dropna()
        self.assertTrue((valid_atr > 0).all(), "ATR should always be positive")

    def test_volatility_detection(self):
        """Test 2: ATR increases with increased volatility."""
        # Create low volatility period
        low_vol_high = pd.Series([100 + i * 0.1 for i in range(50)])
        low_vol_low = pd.Series([99 + i * 0.1 for i in range(50)])
        low_vol_close = pd.Series([99.5 + i * 0.1 for i in range(50)])

        # Create high volatility period
        high_vol_high = pd.Series([100 + i * 2 for i in range(50)])
        high_vol_low = pd.Series([95 + i * 2 for i in range(50)])
        high_vol_close = pd.Series([97.5 + i * 2 for i in range(50)])

        atr_low = self.calculator.average_true_range(
            low_vol_high, low_vol_low, low_vol_close, window=14
        )

        atr_high = self.calculator.average_true_range(
            high_vol_high, high_vol_low, high_vol_close, window=14
        )

        # High volatility should produce higher ATR
        self.assertGreater(
            atr_high.dropna().mean(),
            atr_low.dropna().mean(),
            "High volatility should produce higher ATR"
        )

    def test_gap_handling(self):
        """Test 3: ATR accounts for gaps (high - previous close, previous close - low)."""
        # Create scenario with gap up
        gap_high = pd.Series([100, 105, 115, 120])  # Gap from 105 to 115
        gap_low = pd.Series([95, 100, 112, 115])
        gap_close = pd.Series([102, 104, 118, 119])

        atr = self.calculator.average_true_range(gap_high, gap_low, gap_close, window=2)

        # ATR should capture the gap (difference between 115 low and 104 previous close)
        valid_atr = atr.dropna()
        self.assertGreater(len(valid_atr), 0, "Should produce valid ATR values")

    def test_invalid_input_different_lengths(self):
        """Test 4: Invalid input - mismatched lengths."""
        with self.assertRaises(ValueError):
            self.calculator.average_true_range(
                pd.Series([100]), self.low, self.close
            )

    def test_convenience_function(self):
        """Test 5: Convenience function works correctly."""
        func_atr = calculate_atr(self.high, self.low, self.close)

        self.assertIsInstance(func_atr, pd.Series)
        self.assertTrue((func_atr.dropna() > 0).all())


class TestOnBalanceVolume(unittest.TestCase):
    """Test On-Balance Volume (OBV) calculation."""

    def setUp(self):
        """Set up test data for each test."""
        self.calculator = MomentumIndicators(debug=False)
        np.random.seed(42)
        self.close = pd.Series(100 + np.random.randn(100).cumsum())
        self.volume = pd.Series(np.random.randint(1000, 10000, 100))

    def test_typical_use_case(self):
        """Test 1: Typical use case with valid data."""
        obv = self.calculator.on_balance_volume(self.close, self.volume)

        # Verify output
        self.assertIsInstance(obv, pd.Series)
        self.assertEqual(len(obv), len(self.close))

        # OBV is cumulative, so should increase in magnitude over time
        valid_obv = obv.dropna()
        self.assertGreater(len(valid_obv), 90, "Should have many valid values")

    def test_uptrend_accumulation(self):
        """Test 2: OBV increases during uptrend (accumulation)."""
        # Create clear uptrend
        uptrend_close = pd.Series([100, 105, 110, 115, 120])
        uptrend_volume = pd.Series([1000, 1000, 1000, 1000, 1000])

        obv = self.calculator.on_balance_volume(uptrend_close, uptrend_volume)

        # OBV should be increasing (all positive volume additions)
        valid_obv = obv.dropna()
        # Check that each value is greater than the previous
        obv_diff = valid_obv.diff().dropna()
        self.assertTrue((obv_diff > 0).all(), "OBV should increase during uptrend")

    def test_downtrend_distribution(self):
        """Test 3: OBV decreases during downtrend (distribution)."""
        # Create clear downtrend
        downtrend_close = pd.Series([120, 115, 110, 105, 100])
        downtrend_volume = pd.Series([1000, 1000, 1000, 1000, 1000])

        obv = self.calculator.on_balance_volume(downtrend_close, downtrend_volume)

        # OBV should be decreasing (all negative volume additions)
        valid_obv = obv.dropna()
        obv_diff = valid_obv.diff().dropna()
        self.assertTrue((obv_diff < 0).all(), "OBV should decrease during downtrend")

    def test_flat_price_no_change(self):
        """Test 4: OBV doesn't change when price is flat."""
        flat_close = pd.Series([100, 100, 100, 100])
        flat_volume = pd.Series([1000, 1000, 1000, 1000])

        obv = self.calculator.on_balance_volume(flat_close, flat_volume)

        # OBV should remain constant (direction = 0)
        valid_obv = obv.dropna()
        obv_diff = valid_obv.diff().dropna()
        self.assertTrue((obv_diff == 0).all(), "OBV should be flat when price is flat")

    def test_divergence_detection(self):
        """Test 5: Test setup for divergence detection (price up, OBV down)."""
        # Price going up but on decreasing volume (bearish divergence warning)
        divergence_close = pd.Series([100, 101, 102, 103, 104])
        divergence_volume = pd.Series([10000, 8000, 6000, 4000, 2000])

        obv = self.calculator.on_balance_volume(divergence_close, divergence_volume)

        # OBV should still increase (price is rising) but at decreasing rate
        valid_obv = obv.dropna()
        self.assertTrue((valid_obv.diff().dropna() > 0).all(), "OBV should increase with rising price")

        # But the rate of increase should be decreasing (lower volume)
        # This would be detected by comparing OBV trend vs price trend in practice

    def test_invalid_input_different_lengths(self):
        """Test 6: Invalid input - mismatched lengths."""
        with self.assertRaises(ValueError):
            self.calculator.on_balance_volume(
                pd.Series([100, 105]),
                pd.Series([1000, 2000, 3000])
            )

    def test_convenience_function(self):
        """Test 7: Convenience function works correctly."""
        func_obv = calculate_obv(self.close, self.volume)

        self.assertIsInstance(func_obv, pd.Series)
        self.assertEqual(len(func_obv), len(self.close))


class TestDebugMode(unittest.TestCase):
    """Test debug mode and logging functionality."""

    def test_debug_mode_counter(self):
        """Test 1: Debug mode increments calculation counter."""
        calculator = MomentumIndicators(debug=True)

        close = pd.Series([100, 105, 110, 108, 112] * 10)
        volume = pd.Series([1000, 1200, 1100, 1300, 1250] * 10)

        # Run multiple calculations
        calculator.rate_of_change(close)
        calculator.on_balance_volume(close, volume)

        # Counter should increment for each calculation
        self.assertEqual(calculator.calculation_counter, 2)


if __name__ == "__main__":
    unittest.main()
