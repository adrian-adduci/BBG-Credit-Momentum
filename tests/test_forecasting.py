"""
Unit tests for the forecasting primitives.

These tests encode the three defects that made the original pipeline report
optimistic results, so that a regression is caught immediately:

1. Targets were taken from the *same* row as the features (a nowcast), and
   where a horizon was applied it used ``shift(+h)`` -- the value from ``h``
   days *ago* -- while being labelled "_D_Ahead_Actual".
2. The raw target column was left in the feature matrix, so feature-importance
   and predictive-power rankings were measuring leakage.
3. The train/test split was produced by ``train_test_split`` whose ``shuffle``
   argument was wired to a parameter named ``sequential``, so asking for a
   sequential split produced a shuffled one -- training on the future.

Author: BBG-Credit-Momentum
License: MIT
"""

import unittest

import numpy as np
import pandas as pd

from forecasting import (
    make_supervised,
    random_walk_baseline,
    time_ordered_split,
    walk_forward_backtest,
)


def _frame(n=100, seed=0):
    """A small deterministic frame shaped like a Bloomberg export."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "Dates": pd.date_range("2020-01-01", periods=n, freq="D"),
            "target": np.arange(n, dtype=float),
            "driver_a": rng.normal(size=n),
            "driver_b": rng.normal(size=n),
        }
    )


class TestMakeSupervised(unittest.TestCase):
    """The forward-shift contract: y[t] must be the target at t + horizon."""

    def test_target_is_taken_from_the_future_not_the_past(self):
        df = _frame(n=50)

        X, y, dates = make_supervised(df, target_col="target", horizon=5)

        # Row 0 is dated 2020-01-01; its label must be the target on 2020-01-06.
        self.assertEqual(dates.iloc[0], pd.Timestamp("2020-01-01"))
        self.assertEqual(y.iloc[0], df["target"].iloc[5])
        # And emphatically NOT the value from five days earlier.
        self.assertNotEqual(y.iloc[0], df["target"].iloc[0])

    def test_every_label_leads_its_features_by_the_horizon(self):
        df = _frame(n=50)
        horizon = 7

        _, y, dates = make_supervised(df, target_col="target", horizon=horizon)

        lookup = df.set_index("Dates")["target"]
        for feature_date, label in zip(dates, y):
            expected_date = feature_date + pd.Timedelta(days=horizon)
            self.assertEqual(label, lookup[expected_date])

    def test_rows_without_a_known_future_are_dropped(self):
        df = _frame(n=50)

        X, y, dates = make_supervised(df, target_col="target", horizon=5)

        # The final 5 rows have no observable t+5 value, so they cannot be
        # training examples.
        self.assertEqual(len(X), 45)
        self.assertEqual(len(y), 45)
        self.assertEqual(len(dates), 45)
        self.assertFalse(y.isna().any())

    def test_raw_target_is_excluded_from_the_feature_matrix(self):
        df = _frame(n=50)

        X, _, _ = make_supervised(df, target_col="target", horizon=5)

        self.assertNotIn("target", X.columns)
        self.assertNotIn("Dates", X.columns)
        self.assertIn("driver_a", X.columns)
        self.assertIn("driver_b", X.columns)

    def test_lagged_target_features_are_opt_in_and_never_contemporaneous(self):
        df = _frame(n=50)

        X, _, _ = make_supervised(
            df, target_col="target", horizon=5, target_lags=[1, 2]
        )

        self.assertIn("target_lag_1", X.columns)
        self.assertIn("target_lag_2", X.columns)
        # A lag of 0 would reintroduce the contemporaneous target.
        self.assertNotIn("target_lag_0", X.columns)
        self.assertNotIn("target", X.columns)

    def test_horizon_must_be_a_positive_number_of_periods(self):
        df = _frame(n=50)

        for bad_horizon in (0, -1):
            with self.assertRaises(ValueError):
                make_supervised(df, target_col="target", horizon=bad_horizon)

    def test_missing_target_column_is_rejected(self):
        df = _frame(n=50)

        with self.assertRaises(ValueError):
            make_supervised(df, target_col="not_a_column", horizon=5)


class TestTargetMode(unittest.TestCase):
    """Levels vs changes: tree models cannot extrapolate a trending level."""

    def test_change_mode_labels_the_future_difference(self):
        df = _frame(n=50)

        _, y, _ = make_supervised(
            df, target_col="target", horizon=5, target_mode="change"
        )

        # target is 0,1,2,... so a 5-day-ahead change is always +5.
        self.assertAlmostEqual(y.iloc[0], 5.0)
        self.assertTrue((y == 5.0).all())

    def test_level_mode_remains_the_default(self):
        df = _frame(n=50)

        _, y_default, _ = make_supervised(df, target_col="target", horizon=5)
        _, y_level, _ = make_supervised(
            df, target_col="target", horizon=5, target_mode="level"
        )

        pd.testing.assert_series_equal(y_default, y_level)

    def test_unknown_target_mode_is_rejected(self):
        df = _frame(n=50)

        with self.assertRaises(ValueError):
            make_supervised(
                df, target_col="target", horizon=5, target_mode="sideways"
            )

    def test_change_mode_baseline_is_a_forecast_of_no_change(self):
        rng = np.random.default_rng(5)
        n = 400
        walk = 100 + rng.normal(size=n).cumsum()
        df = pd.DataFrame(
            {
                "Dates": pd.date_range("2020-01-01", periods=n, freq="D"),
                "target": walk,
                "noise": rng.normal(size=n),
            }
        )

        result = walk_forward_backtest(
            df, target_col="target", horizon=1, n_splits=4, target_mode="change"
        )

        # Predicting "no change" on a random walk has RMSE ~1 (one step of
        # unit noise), and no model can genuinely beat it.
        self.assertAlmostEqual(result.baseline_rmse, 1.0, delta=0.35)
        self.assertLess(result.skill_score, 0.5)


class TestTimeOrderedSplit(unittest.TestCase):
    """The split contract: no training example may postdate a test example."""

    def test_all_training_dates_precede_all_test_dates(self):
        df = _frame(n=100)
        X, y, dates = make_supervised(df, target_col="target", horizon=1)

        split = time_ordered_split(X, y, dates, test_size=0.2)

        self.assertLess(split.train_dates.max(), split.test_dates.min())

    def test_split_preserves_every_row_exactly_once(self):
        df = _frame(n=100)
        X, y, dates = make_supervised(df, target_col="target", horizon=1)

        split = time_ordered_split(X, y, dates, test_size=0.2)

        self.assertEqual(len(split.X_train) + len(split.X_test), len(X))
        self.assertEqual(len(split.y_train) + len(split.y_test), len(y))

    def test_test_size_controls_the_holdout_fraction(self):
        df = _frame(n=100)
        X, y, dates = make_supervised(df, target_col="target", horizon=1)

        split = time_ordered_split(X, y, dates, test_size=0.25)

        self.assertEqual(len(split.X_test), 25)
        self.assertEqual(len(split.X_train), 74)

    def test_features_and_labels_stay_aligned_after_splitting(self):
        df = _frame(n=100)
        X, y, dates = make_supervised(df, target_col="target", horizon=1)

        split = time_ordered_split(X, y, dates, test_size=0.2)

        # driver_a is unique per row, so it identifies the source row.
        for driver_value, label in zip(split.X_test["driver_a"], split.y_test):
            source_row = X.index[X["driver_a"] == driver_value][0]
            self.assertEqual(label, y.loc[source_row])


class TestRandomWalkBaseline(unittest.TestCase):
    """The honest benchmark: tomorrow's spread is today's spread."""

    def test_baseline_predicts_the_last_observed_value(self):
        df = _frame(n=50)
        last_known = df["target"].iloc[:45]

        preds = random_walk_baseline(last_known)

        np.testing.assert_array_equal(np.asarray(preds), np.asarray(last_known))

    def test_baseline_beats_a_model_that_has_no_signal(self):
        # A pure random walk is unpredictable by construction: the naive
        # forecast should not be beaten by much, if at all.
        rng = np.random.default_rng(7)
        walk = pd.Series(100 + rng.normal(size=500).cumsum())

        preds = random_walk_baseline(walk[:-1])
        actual = walk.shift(-1).dropna()

        rmse = float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(preds)) ** 2)))
        # One step of unit-variance noise.
        self.assertLess(rmse, 1.5)


class TestWalkForwardBacktest(unittest.TestCase):
    """The evaluation contract: expanding window, always trained on the past."""

    def test_each_fold_trains_only_on_data_preceding_its_test_window(self):
        df = _frame(n=200)

        result = walk_forward_backtest(
            df, target_col="target", horizon=1, n_splits=4
        )

        self.assertEqual(len(result.folds), 4)
        for fold in result.folds:
            self.assertLess(fold.train_end_date, fold.test_start_date)

    def test_backtest_reports_model_and_baseline_side_by_side(self):
        df = _frame(n=200)

        result = walk_forward_backtest(
            df, target_col="target", horizon=1, n_splits=3
        )

        for fold in result.folds:
            self.assertIsInstance(fold.model_rmse, float)
            self.assertIsInstance(fold.baseline_rmse, float)
        self.assertIsInstance(result.skill_score, float)

    def test_no_skill_against_a_random_walk_is_reported_honestly(self):
        # On an unpredictable series the model must NOT show large skill.
        # If this fails, leakage has crept back in.
        rng = np.random.default_rng(11)
        n = 400
        walk = 100 + rng.normal(size=n).cumsum()
        df = pd.DataFrame(
            {
                "Dates": pd.date_range("2020-01-01", periods=n, freq="D"),
                "target": walk,
                "noise_a": rng.normal(size=n),
                "noise_b": rng.normal(size=n),
            }
        )

        result = walk_forward_backtest(
            df, target_col="target", horizon=1, n_splits=4, target_lags=[1, 2, 3]
        )

        # Skill score is 1 - (model_rmse / baseline_rmse). Genuine predictive
        # skill on white noise is impossible; anything above 0.5 means the
        # model is seeing the answer.
        self.assertLess(result.skill_score, 0.5)


if __name__ == "__main__":
    unittest.main()
