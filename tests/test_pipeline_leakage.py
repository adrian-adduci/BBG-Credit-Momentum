"""
Regression tests for leakage defects in the legacy preprocessing/model classes.

These pin the behaviour of ``BloombergPreprocessor`` and ``MomentumModel`` -- the
classes ``webapp.py`` and ``api.py`` actually run -- so the defects described
in tests/test_forecasting.py cannot reappear through the older code path.

Author: BBG-Credit-Momentum
License: MIT
"""

import unittest

import numpy as np
import pandas as pd

import models
import preprocessing


def _write_frame(tmpdir, n=180, seed=3):
    """Write a Bloomberg-shaped xlsx and return its path."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "Dates": pd.date_range("2020-01-01", periods=n, freq="D"),
            "LF98TRUU_Index_OAS": 300 + rng.normal(size=n).cumsum(),
            "LUACTRUU_Index_OAS": 120 + rng.normal(size=n).cumsum(),
            "VIX": 15 + rng.normal(size=n).cumsum() * 0.1,
        }
    )
    path = tmpdir / "econ.xlsx"
    df.to_excel(path, index=False)
    return path, df


class TestDaysAheadActuals(unittest.TestCase):
    """``get_data_with_dh_actuals`` must look forward, not backward."""

    @classmethod
    def setUpClass(cls):
        import tempfile
        import pathlib

        cls._tmp = tempfile.TemporaryDirectory()
        cls.path, cls.source = _write_frame(pathlib.Path(cls._tmp.name))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _pipeline(self):
        return preprocessing.BloombergPreprocessor(
            self.path, target_col="LF98TRUU_Index_OAS", momentum_list=[]
        )

    def test_d_ahead_actual_holds_a_future_value(self):
        pipeline = self._pipeline()

        data, _, y_dict = pipeline.get_data_with_dh_actuals(days_ahead=3)

        frame = data[3]
        col = "LF98TRUU_Index_OAS_3D_Ahead_Actual"
        target = "LF98TRUU_Index_OAS"

        # Row i's "3 day ahead actual" must equal the target three rows later
        # in the source ordering -- not three rows earlier.
        ordered = frame.sort_values("Dates").reset_index(drop=True)
        lookup = (
            self.source.set_index("Dates")["LF98TRUU_Index_OAS"]
        )
        for _, row in ordered.head(20).iterrows():
            expected = lookup[row["Dates"] + pd.Timedelta(days=3)]
            self.assertAlmostEqual(row[col], expected, places=8)
            self.assertNotAlmostEqual(row[col], row[target], places=8)

    def test_raw_target_is_not_a_feature(self):
        pipeline = self._pipeline()

        _, x_dict, _ = pipeline.get_data_with_dh_actuals(days_ahead=3)

        self.assertNotIn("LF98TRUU_Index_OAS", x_dict[3].columns)
        self.assertNotIn("Dates", x_dict[3].columns)
        self.assertIn("VIX", x_dict[3].columns)

    def test_no_rows_carry_an_unobservable_future(self):
        pipeline = self._pipeline()

        _, _, y_dict = pipeline.get_data_with_dh_actuals(days_ahead=5)

        for horizon, labels in y_dict.items():
            self.assertFalse(
                labels.isna().any().any(), f"NaN labels at horizon {horizon}"
            )


class TestChronologicalSplit(unittest.TestCase):
    """The train/test split must never interleave future and past."""

    @classmethod
    def setUpClass(cls):
        import tempfile
        import pathlib

        cls._tmp = tempfile.TemporaryDirectory()
        cls.path, cls.source = _write_frame(pathlib.Path(cls._tmp.name))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_training_rows_all_precede_test_rows(self):
        pipeline = preprocessing.BloombergPreprocessor(
            self.path, target_col="LF98TRUU_Index_OAS", momentum_list=[]
        )

        X_train, X_test, _, _ = pipeline.get_test_and_train_data()

        train_dates = pipeline.get_train_dates()
        test_dates = pipeline.get_test_dates()
        self.assertLess(train_dates.max(), test_dates.min())
        self.assertEqual(len(X_train) + len(X_test), len(pipeline.get_X_Y_dataframe()[0]))

    def test_sequential_flag_cannot_reenable_shuffling(self):
        # The old parameter wired `sequential` straight into `shuffle`, so
        # sequential=True produced a shuffled split. Whatever is passed, the
        # split must stay chronological.
        pipeline = preprocessing.BloombergPreprocessor(
            self.path,
            target_col="LF98TRUU_Index_OAS",
            momentum_list=[],
            sequential=True,
        )

        self.assertLess(
            pipeline.get_train_dates().max(),
            pipeline.get_test_dates().min(),
        )


class TestModelIntegrity(unittest.TestCase):
    """``MomentumModel`` must not corrupt its own reported metrics."""

    @classmethod
    def setUpClass(cls):
        import tempfile
        import pathlib

        cls._tmp = tempfile.TemporaryDirectory()
        cls.path, cls.source = _write_frame(pathlib.Path(cls._tmp.name))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _model(self):
        pipeline = preprocessing.BloombergPreprocessor(
            self.path, target_col="LF98TRUU_Index_OAS", momentum_list=[]
        )
        return models.MomentumModel(pipeline, model_name="XGBoost", estimators=25)

    def test_feature_importance_does_not_refit_the_reported_model(self):
        # The original code called self.model.fit(...) inside the importance
        # loop, so the estimator returned by get_model() was no longer the
        # one whose metrics had been reported.
        model = self._model()
        before = model.get_model().predict(model.X_test)

        model.feature_importance(forecast_range=2, plot=False)
        after = model.get_model().predict(model.X_test)

        np.testing.assert_allclose(before, after)

    def test_model_is_given_a_concrete_random_seed(self):
        # random_state defaulted to random.seed(), which returns None.
        model = self._model()

        seed = model.get_model().random_state
        self.assertIsInstance(seed, int)

    def test_squared_error_series_is_actually_squared(self):
        model = self._model()

        errors = model.get_squared_errors()
        residuals = np.asarray(model.Y_test) - np.asarray(model.get_preds())

        np.testing.assert_allclose(np.asarray(errors), residuals ** 2)
        self.assertTrue((np.asarray(errors) >= 0).all())


if __name__ == "__main__":
    unittest.main()


class TestNoClassificationMachineryOnARegressionTarget(unittest.TestCase):
    """Credit spreads are continuous; label-encoding them is meaningless.

    BloombergPreprocessor ran preprocessing.LabelEncoder().fit_transform() over the
    float target, producing roughly as many "classes" as rows. Worse, it fitted
    the encoder three separate times -- on Y, Y_train and Y_test -- so the three
    encodings did not even share a mapping. Nothing consumed the result once the
    dead ROC/precision-recall method was removed.
    """

    @classmethod
    def setUpClass(cls):
        import tempfile
        import pathlib

        cls._tmp = tempfile.TemporaryDirectory()
        cls.path, cls.source = _write_frame(pathlib.Path(cls._tmp.name))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _pipeline(self):
        return preprocessing.BloombergPreprocessor(
            self.path, target_col="LF98TRUU_Index_OAS", momentum_list=[]
        )

    def test_pipeline_exposes_no_label_encoded_target(self):
        pipeline = self._pipeline()

        self.assertFalse(hasattr(pipeline, "get_Y_encoded"))
        self.assertFalse(hasattr(pipeline, "Y_encoded"))
        self.assertFalse(hasattr(pipeline, "label_encoder"))

    def test_model_still_builds_without_the_encoded_target(self):
        pipeline = self._pipeline()

        model = models.MomentumModel(pipeline, model_name="XGBoost", estimators=25)

        self.assertEqual(len(model.get_preds()), len(pipeline.X_test))
