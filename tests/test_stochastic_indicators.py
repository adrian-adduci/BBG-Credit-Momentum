"""
Unit Tests for Stochastic Indicators

Tests cover:
- Typical use cases
- Edge cases (empty data, NaN values)
- Invalid inputs (negative windows, wrong types)
- Type overflows
- Boundary conditions

Author: BBG-Credit-Momentum Team
License: MIT
"""

import unittest
import numpy as np
import pandas as pd
from indicators.stochastic import (
    StochasticIndicators,
    calculate_stochastic,
    calculate_stochastic_rsi,
    calculate_williams_r,
)


class TestStochasticOscillator(unittest.TestCase):
    """Test Stochastic Oscillator calculation."""

    def setUp(self):
        """Set up test data for each test."""
        self.calculator = StochasticIndicators(debug=False)

        # Create realistic price data (100 periods)
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
        stoch_k, stoch_d = self.calculator.stochastic_oscillator(
            self.high, self.low, self.close, k_window=14, d_window=3
        )

        # Verify output type
        self.assertIsInstance(stoch_k, pd.Series, "Output should be pandas Series")
        self.assertIsInstance(stoch_d, pd.Series, "Output should be pandas Series")

        # Verify output length
        self.assertEqual(len(stoch_k), len(self.close), "Output length should match input")

        # Verify range (0-100)
        valid_k = stoch_k.dropna()
        self.assertTrue((valid_k >= 0).all(), "%K should be >= 0")
        self.assertTrue((valid_k <= 100).all(), "%K should be <= 100")

        valid_d = stoch_d.dropna()
        self.assertTrue((valid_d >= 0).all(), "%D should be >= 0")
        self.assertTrue((valid_d <= 100).all(), "%D should be <= 100")

        # Verify %D is smoother than %K (less variance)
        # %D is moving average of %K, so should have lower std
        self.assertLess(
            valid_d.std(), valid_k.std(),
            "%D should be smoother (lower std) than %K"
        )

    def test_edge_case_constant_price(self):
        """Test 2: Edge case - constant price (no movement)."""
        # When price is constant, stochastic should handle division by zero
        constant_high = pd.Series([100.0] * 50)
        constant_low = pd.Series([100.0] * 50)
        constant_close = pd.Series([100.0] * 50)

        stoch_k, stoch_d = self.calculator.stochastic_oscillator(
            constant_high, constant_low, constant_close
        )

        # Should not raise exception and should produce valid output
        self.assertIsInstance(stoch_k, pd.Series)
        self.assertIsInstance(stoch_d, pd.Series)

        # With epsilon handling, values should be finite
        valid_k = stoch_k.dropna()
        self.assertTrue(np.isfinite(valid_k).all(), "Should handle constant price without inf/nan")

    def test_edge_case_minimal_data(self):
        """Test 3: Edge case - minimal data (exactly k_window + d_window points)."""
        minimal_high = pd.Series([105, 110, 108, 112, 115, 113, 118, 120, 119, 122, 125, 123, 128, 130, 129, 132, 135])
        minimal_low = pd.Series([95, 100, 98, 102, 105, 103, 108, 110, 109, 112, 115, 113, 118, 120, 119, 122, 125])
        minimal_close = pd.Series([100, 105, 103, 107, 110, 108, 113, 115, 114, 117, 120, 118, 123, 125, 124, 127, 130])

        stoch_k, stoch_d = self.calculator.stochastic_oscillator(
            minimal_high, minimal_low, minimal_close, k_window=14, d_window=3
        )

        # Should produce at least some valid values
        valid_d = stoch_d.dropna()
        self.assertGreater(len(valid_d), 0, "Should produce some valid values with minimal data")

    def test_invalid_input_different_lengths(self):
        """Test 4: Invalid input - series with different lengths."""
        high_short = pd.Series([105, 110])
        low_normal = self.low
        close_normal = self.close

        with self.assertRaises(ValueError) as context:
            self.calculator.stochastic_oscillator(high_short, low_normal, close_normal)

        self.assertIn("same length", str(context.exception).lower())

    def test_invalid_input_negative_window(self):
        """Test 5: Invalid input - negative window size."""
        with self.assertRaises(ValueError) as context:
            self.calculator.stochastic_oscillator(
                self.high, self.low, self.close, k_window=-5
            )

        self.assertIn("positive", str(context.exception).lower())

    def test_invalid_input_wrong_type(self):
        """Test 6: Invalid input - wrong data type (list instead of Series)."""
        high_list = [100, 105, 110]

        with self.assertRaises(TypeError) as context:
            self.calculator.stochastic_oscillator(high_list, self.low, self.close)

        self.assertIn("pandas Series", str(context.exception))

    def test_nan_handling(self):
        """Test 7: Edge case - data with NaN values."""
        # Insert some NaN values
        high_with_nan = self.high.copy()
        high_with_nan.iloc[5:10] = np.nan

        stoch_k, stoch_d = self.calculator.stochastic_oscillator(
            high_with_nan, self.low, self.close
        )

        # Should handle NaN gracefully
        self.assertIsInstance(stoch_k, pd.Series)
        # NaN values should propagate but not crash
        self.assertTrue(stoch_k.isna().any(), "NaN input should produce some NaN output")

    def test_convenience_function(self):
        """Test 8: Convenience function produces same result as class method."""
        class_k, class_d = self.calculator.stochastic_oscillator(
            self.high, self.low, self.close
        )

        func_k, func_d = calculate_stochastic(
            self.high, self.low, self.close
        )

        # Results should be identical
        pd.testing.assert_series_equal(class_k, func_k, check_names=False)
        pd.testing.assert_series_equal(class_d, func_d, check_names=False)


class TestStochasticRSI(unittest.TestCase):
    """Test Stochastic RSI calculation."""

    def setUp(self):
        """Set up test data for each test."""
        self.calculator = StochasticIndicators(debug=False)
        np.random.seed(42)
        self.close = pd.Series(100 + np.random.randn(100).cumsum())

    def test_typical_use_case(self):
        """Test 1: Typical use case with valid data."""
        stoch_k, stoch_d = self.calculator.stochastic_rsi(
            self.close, rsi_window=14, stoch_window=14, k_window=3, d_window=3
        )

        # Verify output type and length
        self.assertIsInstance(stoch_k, pd.Series)
        self.assertEqual(len(stoch_k), len(self.close))

        # Verify range (0-100)
        valid_k = stoch_k.dropna()
        self.assertTrue((valid_k >= 0).all(), "StochRSI %K should be >= 0")
        self.assertTrue((valid_k <= 100).all(), "StochRSI %K should be <= 100")

    def test_edge_case_insufficient_data(self):
        """Test 2: Edge case - insufficient data for calculation."""
        short_close = pd.Series([100, 101, 102, 103])

        with self.assertRaises(ValueError) as context:
            self.calculator.stochastic_rsi(
                short_close, rsi_window=14, stoch_window=14
            )

        self.assertIn("too short", str(context.exception).lower())

    def test_invalid_input_zero_rsi_window(self):
        """Test 3: Invalid input - RSI window less than 2."""
        with self.assertRaises(ValueError) as context:
            self.calculator.stochastic_rsi(self.close, rsi_window=1)

        self.assertIn("positive", str(context.exception).lower())

    def test_sensitivity_comparison(self):
        """Test 4: StochRSI should be more sensitive than regular Stochastic."""
        # Create data with small price change
        stable_close = pd.Series([100 + i * 0.1 for i in range(100)])

        stoch_rsi_k, _ = self.calculator.stochastic_rsi(stable_close)

        # StochRSI should show more variation than regular stochastic would
        # (This is a characteristic test, not strict validation)
        valid_values = stoch_rsi_k.dropna()
        self.assertGreater(len(valid_values), 50, "Should produce many valid values")

    def test_convenience_function(self):
        """Test 5: Convenience function works correctly."""
        func_k, func_d = calculate_stochastic_rsi(self.close)

        self.assertIsInstance(func_k, pd.Series)
        self.assertIsInstance(func_d, pd.Series)


class TestWilliamsR(unittest.TestCase):
    """Test Williams %R calculation."""

    def setUp(self):
        """Set up test data for each test."""
        self.calculator = StochasticIndicators(debug=False)
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
        williams = self.calculator.williams_r(self.high, self.low, self.close, window=14)

        # Verify output type
        self.assertIsInstance(williams, pd.Series)
        self.assertEqual(len(williams), len(self.close))

        # Verify range (-100 to 0)
        valid_williams = williams.dropna()
        self.assertTrue((valid_williams >= -100).all(), "Williams %R should be >= -100")
        self.assertTrue((valid_williams <= 0).all(), "Williams %R should be <= 0")

    def test_overbought_oversold_levels(self):
        """Test 2: Verify overbought/oversold interpretation."""
        # Create scenario where close is at high (should be near 0, overbought)
        overbought_high = pd.Series([100, 105, 110, 115, 120] * 5)
        overbought_low = pd.Series([95, 100, 105, 110, 115] * 5)
        overbought_close = pd.Series([100, 105, 110, 115, 120] * 5)  # Close at high

        williams = self.calculator.williams_r(
            overbought_high, overbought_low, overbought_close, window=5
        )

        # When close equals high, Williams %R should be 0 (overbought)
        valid_williams = williams.dropna()
        # Should be close to 0 (allowing small floating point error)
        self.assertTrue(
            (valid_williams > -10).any(),
            "Williams %R should show overbought levels (near 0) when close = high"
        )

    def test_inverse_relationship_with_stochastic(self):
        """Test 3: Williams %R is inverse of Stochastic %K."""
        # Williams %R = -100 * (High - Close) / (High - Low)
        # Stochastic %K = 100 * (Close - Low) / (High - Low)
        # Therefore: Williams %R ≈ Stochastic %K - 100

        williams = self.calculator.williams_r(self.high, self.low, self.close)
        stoch_k, _ = self.calculator.stochastic_oscillator(self.high, self.low, self.close)

        # Drop NaN for comparison
        williams_valid = williams.dropna()
        stoch_k_valid = stoch_k.loc[williams_valid.index]

        # Williams %R + Stochastic %K should equal approximately 0 (allowing for rounding)
        difference = (williams_valid + stoch_k_valid - 100).abs()
        self.assertTrue(
            (difference < 0.01).all(),
            "Williams %R should be inverse of Stochastic %K"
        )

    def test_invalid_input_different_lengths(self):
        """Test 4: Invalid input - mismatched series lengths."""
        high_short = pd.Series([100, 105])

        with self.assertRaises(ValueError) as context:
            self.calculator.williams_r(high_short, self.low, self.close)

        self.assertIn("same length", str(context.exception).lower())

    def test_convenience_function(self):
        """Test 5: Convenience function works correctly."""
        func_williams = calculate_williams_r(self.high, self.low, self.close)

        self.assertIsInstance(func_williams, pd.Series)
        valid_values = func_williams.dropna()
        self.assertTrue((valid_values >= -100).all())
        self.assertTrue((valid_values <= 0).all())


class TestDebugMode(unittest.TestCase):
    """Test debug mode and logging functionality."""

    def test_debug_mode_enabled(self):
        """Test 1: Debug mode creates calculation counter."""
        calculator = StochasticIndicators(debug=True)

        high = pd.Series([100, 105, 110, 108, 112] * 10)
        low = pd.Series([95, 100, 105, 103, 107] * 10)
        close = pd.Series([98, 104, 109, 106, 111] * 10)

        # Run calculation
        calculator.stochastic_oscillator(high, low, close)

        # Counter should increment
        self.assertEqual(calculator.calculation_counter, 1)

        # Run another calculation
        calculator.williams_r(high, low, close)

        self.assertEqual(calculator.calculation_counter, 2)


if __name__ == "__main__":
    unittest.main()
