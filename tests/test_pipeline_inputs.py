"""
Tests for the inputs ``BloombergPreprocessor`` must accept.

The file-existence guard added during the 2025 refactor called
``pathlib.Path(xlsx_file).is_file()`` unconditionally. That raises TypeError
for anything that is not a path, which is exactly what the two real callers
pass: ``webapp.py`` hands over a Streamlit upload buffer and ``api.py`` hands
over an in-memory DataFrame. Both primary entry points were broken.

Author: BBG-Credit-Momentum
License: MIT
"""

import io
import pathlib
import tempfile
import unittest

import numpy as np
import pandas as pd

import preprocessing


def _frame(n=120, seed=5):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "Dates": pd.date_range("2020-01-01", periods=n, freq="D"),
            "LF98TRUU_Index_OAS": 300 + rng.normal(size=n).cumsum(),
            "VIX": 15 + rng.normal(size=n).cumsum() * 0.1,
        }
    )


class TestAcceptedInputs(unittest.TestCase):
    TARGET = "LF98TRUU_Index_OAS"

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        self.frame = _frame()

    def tearDown(self):
        self._tmp.cleanup()

    def test_accepts_a_filesystem_path(self):
        path = self.root / "econ.xlsx"
        self.frame.to_excel(path, index=False)

        pipeline = preprocessing.BloombergPreprocessor(path, target_col=self.TARGET)

        self.assertGreater(len(pipeline.get_dataframe()), 0)

    def test_accepts_an_in_memory_dataframe(self):
        # api.py passes a DataFrame directly.
        pipeline = preprocessing.BloombergPreprocessor(
            self.frame, target_col=self.TARGET
        )

        self.assertGreater(len(pipeline.get_dataframe()), 0)

    def test_accepts_a_file_like_buffer(self):
        # webapp.py passes a Streamlit upload buffer.
        buffer = io.BytesIO()
        self.frame.to_excel(buffer, index=False)
        buffer.seek(0)

        pipeline = preprocessing.BloombergPreprocessor(buffer, target_col=self.TARGET)

        self.assertGreater(len(pipeline.get_dataframe()), 0)

    def test_missing_file_still_reports_a_clear_error(self):
        with self.assertRaises(FileNotFoundError):
            preprocessing.BloombergPreprocessor(
                self.root / "absent.xlsx", target_col=self.TARGET
            )

    def test_missing_target_column_still_reports_a_clear_error(self):
        with self.assertRaises(ValueError):
            preprocessing.BloombergPreprocessor(self.frame, target_col="nope")


if __name__ == "__main__":
    unittest.main()
