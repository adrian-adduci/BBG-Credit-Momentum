"""
Regression tests for the all-NaN column guard in BloombergPreprocessor.

_add_custom_features() can generate several hundred indicator columns. An
indicator that cannot warm up on a short series emits an all-NaN column, and
the unconditional row-wise dropna() that follows then removes *every* row --
turning one unusable column into an empty dataset with no explanation.

These pin the behaviour that all-NaN columns are dropped (with a warning)
before rows are filtered, and that a genuinely empty result raises something
legible rather than surfacing as a confusing downstream error.

Author: BBG-Credit-Momentum
License: MIT
"""

import pathlib
import tempfile
import unittest

import numpy as np
import pandas as pd

import preprocessing


def _frame(n=120, seed=7):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "Dates": pd.date_range("2020-01-01", periods=n, freq="D"),
            "LF98TRUU_Index_OAS": 300 + rng.normal(size=n).cumsum(),
            "LUACTRUU_Index_OAS": 120 + rng.normal(size=n).cumsum(),
            "VIX": 15 + rng.normal(size=n).cumsum() * 0.1,
        }
    )


class TestAllNaNColumnGuard(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, df):
        path = self.dir / "econ.xlsx"
        df.to_excel(path, index=False)
        return path

    def test_all_nan_column_does_not_empty_the_dataset(self):
        """One unusable column must not wipe out every row."""
        df = _frame()
        df["BROKEN_INDICATOR"] = np.nan
        path = self._write(df)

        pipeline = preprocessing.BloombergPreprocessor(
            path, target_col="LF98TRUU_Index_OAS", momentum_list=[]
        )

        self.assertGreater(
            len(pipeline.complete_data),
            0,
            "an all-NaN column collapsed the dataset to zero rows",
        )

    def test_all_nan_column_is_removed_from_features(self):
        df = _frame()
        df["BROKEN_INDICATOR"] = np.nan
        path = self._write(df)

        pipeline = preprocessing.BloombergPreprocessor(
            path, target_col="LF98TRUU_Index_OAS", momentum_list=[]
        )

        self.assertNotIn("BROKEN_INDICATOR", pipeline.complete_data.columns)
        self.assertNotIn("BROKEN_INDICATOR", list(pipeline.feature_cols))

    def test_usable_columns_are_retained(self):
        """Dropping all-NaN columns must not disturb the good ones."""
        df = _frame()
        df["BROKEN_INDICATOR"] = np.nan
        path = self._write(df)

        pipeline = preprocessing.BloombergPreprocessor(
            path, target_col="LF98TRUU_Index_OAS", momentum_list=[]
        )

        for col in ("LUACTRUU_Index_OAS", "VIX"):
            self.assertIn(col, pipeline.complete_data.columns)

    def test_partially_nan_column_is_kept(self):
        """Only *entirely* empty columns are dropped; warmup NaNs are normal."""
        df = _frame()
        df["WARMUP_INDICATOR"] = [np.nan] * 30 + list(range(len(df) - 30))
        path = self._write(df)

        pipeline = preprocessing.BloombergPreprocessor(
            path, target_col="LF98TRUU_Index_OAS", momentum_list=[]
        )

        self.assertIn("WARMUP_INDICATOR", pipeline.complete_data.columns)

    def test_warns_about_the_columns_it_drops(self):
        df = _frame()
        df["BROKEN_INDICATOR"] = np.nan
        path = self._write(df)

        with self.assertLogs("BloombergPreprocessor", level="WARNING") as captured:
            preprocessing.BloombergPreprocessor(
                path, target_col="LF98TRUU_Index_OAS", momentum_list=[]
            )

        self.assertTrue(
            any("BROKEN_INDICATOR" in line for line in captured.output),
            f"dropped column was not named in the warning: {captured.output}",
        )

    def test_empty_result_raises_a_legible_error(self):
        """A dataset with no usable rows must say so, not fail downstream."""
        df = _frame(n=12)
        # Every row carries a NaN somewhere, so no row survives.
        df.loc[::2, "VIX"] = np.nan
        df.loc[1::2, "LUACTRUU_Index_OAS"] = np.nan
        path = self._write(df)

        with self.assertRaises(ValueError) as ctx:
            preprocessing.BloombergPreprocessor(
                path, target_col="LF98TRUU_Index_OAS", momentum_list=[]
            )

        self.assertIn("no rows", str(ctx.exception).lower())
