"""
Tests for feature-importance reporting on MomentumModel.

Two defects motivated these tests:

1. ``get_features_of_importance`` did not exist, yet ``api.py`` called it in
   two endpoints. Both call sites wrap it in ``except Exception`` and log a
   warning, so every /api/train and /api/mixed/train response silently carried
   ``feature_importance: null`` instead of failing loudly.

2. ``predictive_power`` passed the feature matrix to ``ppscore`` while naming
   the raw target as ``y``. Once the target was correctly removed from the
   feature matrix, ppscore could no longer find that column and raised. The
   deeper problem is that scoring predictors against the *contemporaneous*
   target measures a nowcast; the score must be taken against the future label.

Author: BBG-Credit-Momentum
License: MIT
"""

import pathlib
import tempfile
import unittest

import numpy as np
import pandas as pd

import models
import preprocessing


def _frame(n=200, seed=11):
    rng = np.random.default_rng(seed)
    n_steps = rng.normal(size=n).cumsum()
    return pd.DataFrame(
        {
            "Dates": pd.date_range("2020-01-01", periods=n, freq="D"),
            "LF98TRUU_Index_OAS": 300 + n_steps,
            "LUACTRUU_Index_OAS": 120 + rng.normal(size=n).cumsum(),
            "VIX": 15 + rng.normal(size=n).cumsum() * 0.1,
        }
    )


class TestFeatureImportanceReporting(unittest.TestCase):
    TARGET = "LF98TRUU_Index_OAS"

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.plot_dir = pathlib.Path(cls._tmp.name)
        cls.frame = _frame()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _model(self):
        pipeline = preprocessing.BloombergPreprocessor(
            self.frame, target_col=self.TARGET, horizon=1
        )
        return models.MomentumModel(
            pipeline,
            model_name="XGBoost",
            estimators=25,
            plot_dir=self.plot_dir,
        )

    def test_features_of_importance_returns_a_feature_to_score_mapping(self):
        model = self._model()

        importance = model.get_features_of_importance(forecast_day=3)

        self.assertIsInstance(importance, dict)
        self.assertGreater(len(importance), 0)
        for feature, score in importance.items():
            self.assertIsInstance(feature, str)
            self.assertIsInstance(score, float)

    def test_features_of_importance_are_ranked_highest_first(self):
        model = self._model()

        scores = list(model.get_features_of_importance(forecast_day=3).values())

        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_features_of_importance_never_reports_the_raw_target(self):
        model = self._model()

        importance = model.get_features_of_importance(forecast_day=3)

        self.assertNotIn(self.TARGET, importance)

    def test_features_of_importance_computes_the_day_on_demand(self):
        # api.py calls this without calling feature_importance first.
        model = self._model()
        self.assertEqual(model.features_over_time_dict, {})

        importance = model.get_features_of_importance(forecast_day=2)

        self.assertGreater(len(importance), 0)

    def test_unknown_forecast_day_is_rejected(self):
        model = self._model()

        with self.assertRaises(ValueError):
            model.get_features_of_importance(forecast_day=0)

    def test_predictive_power_scores_against_the_future_label(self):
        model = self._model()

        scores = model.predictive_power(forecast_range=3, plot=False)

        self.assertIsInstance(scores, pd.DataFrame)
        self.assertIn("x", scores.columns)
        self.assertIn("ppscore", scores.columns)
        # ppscore records the scored target in its "y" column. It must be the
        # forward-shifted label, not the contemporaneous target.
        scored_target = set(scores["y"].unique())
        self.assertEqual(scored_target, {f"{self.TARGET}_3D_Ahead_Actual"})
        self.assertNotIn(self.TARGET, scored_target)

    def test_predictive_power_does_not_rank_the_raw_target(self):
        model = self._model()

        scores = model.predictive_power(forecast_range=3, plot=False)

        self.assertNotIn(self.TARGET, set(scores["x"].unique()))

    def test_predictive_power_writes_its_chart(self):
        model = self._model()

        model.predictive_power(forecast_range=3, plot=False)

        self.assertTrue((self.plot_dir / "predictive_power.png").is_file())


if __name__ == "__main__":
    unittest.main()
