################################################################################
# Author: Adrian Adduci
# Email: FAA2160@columbia.edu
################################################################################
import logging
import os
import pathlib
import warnings

import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({"figure.autolayout": True})
import numpy as np
import pandas as pd
import ppscore as pps
import seaborn as sns
from sklearn import linear_model, metrics
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from forecasting import DEFAULT_RANDOM_STATE
from logging_setup import get_logger

path = pathlib.Path(__file__).parent.absolute()

#: Where charts are written when no directory is given.
#:
#: This used to be hardcoded at each savefig call site. Because it is a
#: tracked location, merely running the test suite overwrote committed
#: repository assets with charts rendered from fixture data, leaving the
#: working tree dirty after every run.
DEFAULT_PLOT_DIR = path / "_img"


def _resolve_plot_dir(plot_dir):
    """Return the directory to write charts into, creating it if needed."""
    target = pathlib.Path(plot_dir) if plot_dir is not None else DEFAULT_PLOT_DIR
    target.mkdir(parents=True, exist_ok=True)
    return target


# Silence only the noise this module actually provokes. The previous blanket
# warnings.filterwarnings("ignore") applied process-wide to anything that
# imported this module, hiding genuine pandas and sklearn deprecations from the
# app, the API and the test suite alike.
warnings.filterwarnings(
    "ignore", category=UserWarning, module=r"ppscore\..*"
)
warnings.filterwarnings(
    "ignore", category=FutureWarning, module=r"(seaborn|ppscore)\..*"
)

logger = get_logger("_model", "_model.log")
os.environ["NUMEXPR_MAX_THREADS"] = "16"
################################################################################
# Fit and Predict a Chosen Model
################################################################################


class MomentumModel:
    """
    Builds and trains machine learning models for time series forecasting.

    Trains models on preprocessed data, generates predictions, and analyzes
    feature importance over multiple forecast horizons. Supports various
    sklearn models including XGBoost, CART, and regression models.

    Args:
        pipeline: BloombergPreprocessor object containing preprocessed data
        model_name: Name of the model to use (default: "XGBoost")
            Options: "XGBoost", "CART", "AdaBoostClassifier",
                     "LogisticRegression", "Quadratic Regression", "KNeighborsRegressor"
        estimators: Number of estimators for ensemble models (default: 1000)
        random_state: Random seed for reproducibility
        max_forecast: Maximum forecast horizon in days (default: 30)

    Attributes:
        model: Trained sklearn model
        model_preds: Predictions on test set
        X_train, X_test, Y_train, Y_test: Training and test data splits
        final_data: Complete dataset with forecasts
        scores: Cross-validation scores
        features_over_time_dict: Feature importance across forecast horizons

    Example:
        >>> pipeline = BloombergPreprocessor("data.xlsx", "target_col")
        >>> model = MomentumModel(pipeline, model_name="XGBoost")
        >>> model.predictive_power(forecast_range=30)
        >>> mae, mse, rmse = model.get_mean_error_metrics()
    """
    def __init__(
        self,
        pipeline,
        model_name="XGBoost",
        estimators=1000,
        random_state=DEFAULT_RANDOM_STATE,
        max_forecast=30,
        plot_dir=None,
    ):

        try:
            if model_name != None:
                pass
        except ValueError:
            self.logger.debug(" Must specify a model type")

        self.pipeline = pipeline
        self.plot_dir = _resolve_plot_dir(plot_dir)
        self.timeseries_splits = 5
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        # Return Processesed Data
        (
            self.X_train,
            self.X_test,
            self.Y_train,
            self.Y_test,
        ) = pipeline.get_test_and_train_data()
        self.X_df, self.Y_df = pipeline.get_X_Y_dataframe()

        logger.info(f" Selecting model {model_name}")

        # Default Models
        self.models_available = {
            "CART": DecisionTreeRegressor(random_state=random_state),
            "XGBoost": XGBRegressor(n_estimators=estimators, random_state=random_state),
            "AdaBoostClassifier": AdaBoostClassifier(
                n_estimators=30, learning_rate=0.50, random_state=random_state
            ),
            "LogisticRegression": linear_model.LogisticRegression(),
            "Quadratic Regression": make_pipeline(PolynomialFeatures(3), Ridge()),
            "KNeighborsRegressor": KNeighborsRegressor(n_neighbors=2),
        }
        self.model = self.models_available.get(model_name)
        if self.model is None:
            raise ValueError(
                f"Unknown model {model_name!r}. Available: "
                f"{sorted(self.models_available)}"
            )
        tss = TimeSeriesSplit(n_splits=self.timeseries_splits)

        logger.info(f" Fitting model: \n {self.model}")
        self.model.fit(self.X_train, self.Y_train)

        # Predict
        logger.info(f" Predicting with model: \n {self.model}")
        self.model_preds = self.model.predict(self.X_test)

        # Forecasts are produced from the aligned feature frame, whose rows are
        # exactly those with an observable future value.
        X_all, _ = self.pipeline.get_feature_frame()
        final = self.pipeline.get_complete_data().loc[X_all.index].copy()
        forecast_col = f"{self.pipeline.get_target_col()}_Forecast"
        final[forecast_col] = self.model.predict(X_all)
        self.final_data = final.sort_values("Dates", ascending=False).set_index("Dates")

        # cross_val_score returns R^2 for regressors, not accuracy. Scored on
        # the training portion only so the held-out set stays untouched.
        self.scores = cross_val_score(
            clone(self.model), self.X_train, self.Y_train, cv=tss
        )
        logger.info(
            " Mean cross-validation R^2 (train only): %0.2f (+/- %0.3f)"
            % (self.scores.mean(), self.scores.std())
        )
        self.forecast_horizon = max(pipeline.forecast_list)
        self.features_over_time_dict = {}

    ################################################################################
    # Class Methods
    ################################################################################
    def predictive_power(self, forecast_range=30, plot=True):
        """
        Calculate and visualize predictive power scores for all features.

        Uses the ppscore library to identify which features have the highest
        predictive power for the target variable at a specific forecast horizon.

        Scores are taken against the *future* label at ``forecast_range`` days
        ahead. Scoring against the contemporaneous target would measure a
        nowcast, and since the raw target is deliberately absent from the
        feature matrix, naming it as ``y`` also raised a ValueError inside
        ppscore.

        Args:
            forecast_range: Number of days ahead to forecast (default: 30)
            plot: Whether to display the plot (default: True)

        Returns:
            pd.DataFrame: The ppscore predictors table, ranked by score. Its
            ``y`` column holds the forward-shifted label that was scored.

        Creates:
            PNG file saved to ``<plot_dir>/predictive_power.png``.
        """

        all_data, X_data, Y_data = self.pipeline.get_data_with_dh_actuals(
            days_ahead=forecast_range
        )

        pipeline_target = self.pipeline.get_target_col()
        label_col = f"{pipeline_target}_{forecast_range}D_Ahead_Actual"

        # ppscore needs the scored column inside the frame it is handed, so
        # attach the future label to the features rather than the raw target.
        feats = pd.DataFrame(data=X_data[forecast_range]).copy()
        feats[label_col] = Y_data[forecast_range][label_col]

        predictors_df = pps.predictors(feats, y=label_col)

        strong = predictors_df[predictors_df["ppscore"] > 0.5]

        f, ax = plt.subplots(figsize=(16, 5))
        ax.set_title(
            f"Predicative Power for {pipeline_target} at {forecast_range} Days"
        )
        if not strong.empty:
            sns.barplot(data=strong, y="x", x="ppscore", palette="rocket")
        else:
            ax.text(
                0.5,
                0.5,
                f"No feature scored above 0.5 at {forecast_range} days",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        plt.savefig(self.plot_dir / "predictive_power.png", bbox_inches="tight")
        if plot:
            f.show()
        else:
            plt.close(f)

        return predictors_df

    def feature_importance(self, forecast_range=30, plot=True):
        """
        Calculate and visualize feature importance for each forecast day.

        Refits the model for each day in the forecast range and extracts
        feature importances. Only features with importance > 0.05 are retained.

        Args:
            forecast_range: Number of days ahead to forecast (default: 30)
            plot: Whether to display the plot (default: True)

        Updates:
            self.features_over_time_dict: Dictionary mapping each day to
                feature importances for that forecast horizon

        Creates:
            PNG file saved to _img/feats_importance.png showing bar chart
            of top features for the final forecast day
        """

        pipeline_target = self.pipeline.get_target_col()

        all_data, X_data, Y_data = self.pipeline.get_data_with_dh_actuals(
            days_ahead=forecast_range, target=pipeline_target
        )

        for day in range(1, (forecast_range + 1)):

            X_feature_cols = X_data[day].columns

            # No MinMaxScaler here: the supported importance models are all
            # tree-based and scale-invariant, and fitting a scaler across the
            # full dataset leaked test-set range into the transform.
            X_day = X_data[day]

            logger.info(f" Fitting importance model: Day {day}")
            # clone() gives a fresh unfitted estimator. Calling self.model.fit
            # here refit the very model whose metrics had already been
            # reported, so results changed depending on call order.
            model_scaled = clone(self.model).fit(X_day, Y_data[day])
            importances = model_scaled.feature_importances_

            feats = {}
            feats_model_by_day = {}
            threshold = 0.05

            if day not in self.features_over_time_dict.keys():
                self.features_over_time_dict[day] = {
                    feature: None for feature in X_feature_cols
                }

            for feature, importance in zip(X_feature_cols, importances):

                if importance > threshold:
                    feats[feature] = importance

                if self.features_over_time_dict[day][feature] == None:
                    self.features_over_time_dict[day][feature] = [importance]

                else:
                    self.features_over_time_dict[day][feature].append(importance)

            feats = sorted(feats.items(), key=lambda x: x[1], reverse=True)
            feats = dict(feats)
            feats_model_by_day[day] = feats

            if day == forecast_range:
                for target, feature in feats_model_by_day.items():
                    width = 1
                    keys = feature.keys()
                    values = feature.values()
                    if target == day:
                        f, ax = plt.subplots(figsize=(16, 5))
                        ax.set_title(
                            f"Feature Importance for {target} Day Forecast: {pipeline_target}"
                        )
                        sns.barplot(y=list(keys), x=list(values), palette="rocket")
                        plt.savefig(self.plot_dir / "feats_importance.png")
                        if plot:
                            f.show()

    def feature_importance_over_time(
        self, plot=True, forecast_range=30, usefulness_threshold=0.2
    ):

        pipeline_target = self.pipeline.get_target_col()

        if not self.features_over_time_dict:
            self.feature_importance(forecast_range, plot=False)

        list_of_days_to_forecast = list(range(1, forecast_range + 1))
        df = pd.DataFrame()
        column_names = []

        for day in list_of_days_to_forecast:
            if day == 1:
                df = pd.DataFrame.from_dict(self.features_over_time_dict[day])
                column_names = list(df.columns)
            else:
                feature_dict = pd.DataFrame.from_dict(self.features_over_time_dict[day])
                df = pd.concat([df, feature_dict], ignore_index=True)
        df["day"] = list_of_days_to_forecast

        remove_list = []

        for feat in column_names:
            usefulness = df[feat].max()

            if usefulness < usefulness_threshold:
                logger.info("feat: {}, usseful-max: {:.5f}".format(feat, usefulness))
                df.drop([feat], axis=1)
                remove_list.append(feat)

        for x in remove_list:
            column_names.remove(x)

        sns.set_palette(sns.color_palette("rocket"))
        f, ax = plt.subplots(figsize=(14, 6))
        for feat in column_names:
            sns.lineplot(data=df, x="day", y=df[feat], dashes=False).set_title(
                f"{pipeline_target} Feature Importance By Time"
            )
        sns.set_style("whitegrid")
        ax.grid(True)
        ax.set(xlabel="Days Out", ylabel="Predictive Importance")
        ax.set(xticks=list(range(1, forecast_range + 1)))
        ax.legend(column_names)
        plt.savefig(
            self.plot_dir / "feats_importance_over_time.png", bbox_inches="tight"
        )

    # NOTE: get_roc_and_precision_recall_curves was removed. It called
    # plot_roc_curve / plot_precision_recall_curve, which were never imported
    # and were deleted from scikit-learn in 1.2, and it indexed a 1-D axes
    # array as axes[0, 0]. It could never have run. For classification use
    # sklearn.metrics.RocCurveDisplay.from_estimator instead.

    def get_mean_error_metrics(self):
        """
        Calculate and return model error metrics.

        Computes Mean Absolute Error (MAE), Mean Squared Error (MSE),
        and Root Mean Squared Error (RMSE) for the test set predictions.

        Returns:
            tuple: (MAE, MSE, RMSE) as floats

        Example:
            >>> mae, mse, rmse = model.get_mean_error_metrics()
            >>> print(f"Model RMSE: {rmse:.4f}")
        """
        MAE = metrics.mean_absolute_error(self.Y_test, self.model_preds)
        MSE = metrics.mean_squared_error(self.Y_test, self.model_preds)
        RMSE = np.sqrt(metrics.mean_squared_error(self.Y_test, self.model_preds))
        logger.info(f"MAE: {MAE:.4}")
        logger.info(f"MSE: {MSE:.4}")
        logger.info(f"RMSE: {RMSE:.4}")

        num_predictions = [int(num) for num in range(1, len(self.model_preds) + 1)]

        # Squared, not doubled. The original computed `residual * 2` and
        # labelled the series "MSE", which also let it go negative.
        errors = self.get_squared_errors().tolist()

        err_MSE_df = pd.DataFrame(
            list(zip(num_predictions, errors)), columns=["Prediction", "MSE"]
        )
        sns.set_palette(sns.color_palette("rocket"))
        sns.set_style("whitegrid")
        sns.lineplot(data=err_MSE_df, y="MSE", x="Prediction", dashes=False).set_title(
            "Mean Squared Error"
        )
        return MAE, MSE, RMSE

    def get_features_of_importance(
        self, forecast_day=30, threshold=0.05
    ):
        """
        Return the features that matter at a given forecast horizon.

        This method is called by ``api.py`` in both training endpoints but was
        never implemented. Because those call sites catch every exception and
        only log a warning, the API returned ``feature_importance: null`` on
        every request instead of surfacing the AttributeError.

        Importances for the requested day are computed on demand if
        ``feature_importance`` has not already run.

        Args:
            forecast_day: Forecast horizon in days (default: 30)
            threshold: Minimum importance to report (default: 0.05)

        Returns:
            dict: ``{feature_name: importance}`` ranked highest first. Empty if
            no feature clears ``threshold``.

        Raises:
            ValueError: If ``forecast_day`` < 1.
        """
        if not isinstance(forecast_day, (int, np.integer)) or forecast_day < 1:
            raise ValueError(
                f"forecast_day must be an integer >= 1, got {forecast_day!r}"
            )

        if forecast_day not in self.features_over_time_dict:
            self.feature_importance(forecast_range=forecast_day, plot=False)

        if forecast_day not in self.features_over_time_dict:
            return {}

        scored = {}
        for feature, values in self.features_over_time_dict[forecast_day].items():
            if not values:
                continue
            # Values accumulate across repeated runs; the latest is current.
            importance = float(values[-1])
            if importance > threshold:
                scored[feature] = importance

        return dict(sorted(scored.items(), key=lambda kv: kv[1], reverse=True))

    def get_squared_errors(self):
        """Per-prediction squared residuals on the held-out set."""
        residuals = np.asarray(self.Y_test) - np.asarray(self.model_preds)
        return residuals ** 2

    def get_preds(self):
        return self.model_preds

    def get_preds_with_dates(self):
        self.model_preds["Dates"] = self.X_test_dates
        return self.model_preds

    def get_final_data(self):
        return self.final_data

    def get_model(self):
        return self.model
