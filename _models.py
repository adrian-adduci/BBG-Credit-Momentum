################################################################################
# Author: Adrian Adduci
# Email: FAA2160@columbia.edu
################################################################################
from dotenv import load_dotenv
import os
import logging
import os
import pathlib
import random
import warnings

import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({"figure.autolayout": True})
import numpy as np
import pandas as pd
import ppscore as pps
import seaborn as sns
import streamlit as st
from sklearn import linear_model, metrics
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (mean_squared_error, plot_precision_recall_curve,
                             plot_roc_curve)
from sklearn.model_selection import (TimeSeriesSplit, cross_val_score,
                                     train_test_split)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sktime.forecasting.model_selection import SingleWindowSplitter
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from xgboost import XGBRegressor, plot_importance, plot_tree

path = pathlib.Path(__file__).parent.absolute()

# Debug and logger
warnings.filterwarnings("ignore")
logger = logging.getLogger("_model")
logger.setLevel(logging.INFO)
handler = logging.FileHandler(str(path) + "\\logs\\_model.log")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
os.environ["NUMEXPR_MAX_THREADS"] = "16"
################################################################################
# Fit and Predict a Chosen Model
################################################################################


class _build_model:
    def __init__(
        self,
        pipeline,
        model_name="XGBoost",
        estimators=1000,
        random_state=random.seed(),
        max_forecast=30,
    ):

        try:
            if model_name != None:
                pass
        except ValueError:
            self.logger.debug(" Must specify a model type")

        self.pipeline = pipeline
        self.timeseries_splits = 5
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        # Return Processesed Data
        (
            self.X_train,
            self.X_test,
            self.Y_train,
            self.Y_test,
        ) = pipeline._return_test_and_train_data()
        (
            self.Y_encoded,
            self.Y_train_encoded,
            self.Y_test_encoded,
        ) = pipeline._return_Y_encoded()
        self.X_df, self.Y_df = pipeline._return_X_Y_dataframe()

        logger.info(" Selecting model {}".format(model_name))

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
        tss = TimeSeriesSplit(n_splits=self.timeseries_splits).split(self.X_df)

        logger.info(" Fitting model: \n {}".format(self.model))
        self.model.fit(self.X_train, self.Y_train)

        # Predict
        logger.info(" Predicting with model: \n {}".format(self.model))
        self.model_preds = self.model.predict(self.X_test)

        self.final_data = self.pipeline._return_complete_data()
        self.final_data[
            (self.pipeline._return_target_col()) + "_Forecast"
        ] = self.model.predict(self.final_data[self.pipeline._return_feature_names()])
        self.final_data = self.final_data.sort_values(
            "Dates", ascending=False
        ).set_index("Dates")

        self.scores = cross_val_score(self.model, self.X_df, self.Y_df, cv=tss)
        logger.info(
            " Mean cross-validataion Accuracy: %0.2f (+/- %0.3f)"
            % (self.scores.mean(), self.scores.std())
        )
        self.forecast_horizon = max(pipeline.forecast_list)
        self.features_over_time_dict = {}

    ################################################################################
    # Class Methods
    ################################################################################
    def predictive_power(self, forecast_range=30, plot=True):

        all_data, X_data, Y_data = self.pipeline._return_data_with_dh_actuals(
            forecast_range
        )

        pipeline_target = self.pipeline._return_target_col()

        feats = pd.DataFrame(data=X_data[forecast_range])

        predictors_df = pps.predictors(feats, y=pipeline_target)

        predictors_df = predictors_df[predictors_df["ppscore"] > 0.5]

        f, ax = plt.subplots(figsize=(16, 5))
        ax.set_title(
            "Predicative Power for {0} at {1} Days".format(
                pipeline_target, forecast_range
            )
        )
        sns.barplot(data=predictors_df, y="x", x="ppscore", palette="rocket")
        plt.savefig((str(path)) + "\\_img\\predictive_power.png", bbox_inches="tight")
        if plot:
            f.show()

    def _feature_importance(self, forecast_range=30, plot=True):

        pipeline_target = self.pipeline._return_target_col()

        all_data, X_data, Y_data = self.pipeline._return_data_with_dh_actuals(
            days_ahead=forecast_range, target=pipeline_target
        )

        for day in range(1, (forecast_range + 1)):

            X_feature_cols = X_data[day].columns

            # Scale and add back to df with column names
            X_scaled = self.scaler.fit_transform(X_data[day])
            X_scaled = pd.DataFrame(X_scaled, columns=X_feature_cols)

            logger.info(" Fitting Scaled Model: Day {}".format(day))
            model_scaled = self.model.fit(X_scaled, Y_data[day])
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
                            "Feature Importance for {0} Day Forecast: {1}".format(
                                target, pipeline_target
                            )
                        )
                        sns.barplot(y=list(keys), x=list(values), palette="rocket")
                        plt.savefig((str(path)) + "\\_img\\feats_importance.png")
                        if plot:
                            f.show()

    def _feature_importance_over_time(
        self, plot=True, forecast_range=30, usefulness_threshold=0.2
    ):

        pipeline_target = self.pipeline._return_target_col()

        if not self.features_over_time_dict:
            self._feature_importance(forecast_range, plot=False)

        list_of_days_to_forecast = list(range(1, forecast_range + 1))
        df = pd.DataFrame()
        column_names = []

        for day in list_of_days_to_forecast:
            if day == 1:
                df = pd.DataFrame.from_dict(self.features_over_time_dict[day])
                column_names = list(df.columns)
            else:
                feature_dict = pd.DataFrame.from_dict(self.features_over_time_dict[day])
                df = df.append(feature_dict, ignore_index=True)
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
                "{0} Feature Importance By Time".format(pipeline_target)
            )
        sns.set_style("whitegrid")
        ax.grid(True)
        ax.set(xlabel="Days Out", ylabel="Predictive Importance")
        ax.set(xticks=list(range(1, forecast_range + 1)))
        ax.legend(column_names)
        plt.savefig(
            (str(path)) + "\\_img\\feats_importance_over_time.png", bbox_inches="tight"
        )

    # Classification only, n/a for regression models
    def _return_roc_and_precision_recall_curves(self):
        sns.set_palette(sns.color_palette("rocket"))
        pipeline_target = self.pipeline._return_target_col()
        accuracy = metrics.accuracy_score(self.Y_test_encoded, self.model_preds)
        fig_roc, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))
        roc_plot = plot_roc_curve(
            self.model, self.X_test, self.Y_test_encoded, ax=axes[0, 0]
        )
        axes[0, 0].title.set_text(
            "{} Prediction ROC [Accuracy: {}]".format(pipeline_target, accuracy)
        )
        pr_plot = plot_precision_recall_curve(
            self.model, self.X_test, self.Y_test_encoded, ax=axes[1, 0]
        )
        axes[1, 0].title.set_text(
            "{} Prediction Precision-Recall Curve".format(pipeline_target)
        )
        fig_roc.tight_layout(pad=3.0)
        fig_roc.show()
        return fig_roc

    def _return_mean_error_metrics(self):
        MAE = metrics.mean_absolute_error(self.Y_test, self.model_preds)
        MSE = metrics.mean_squared_error(self.Y_test, self.model_preds)
        RMSE = np.sqrt(metrics.mean_squared_error(self.Y_test, self.model_preds))
        logger.info("MAE: {:.4}".format(MAE))
        logger.info("MSE: {:.4}".format(MSE))
        logger.info("RMSE: {:.4}".format(RMSE))

        errors_MAE = list()
        errors = list()
        errors_RMSE = list()
        num_predictions = [int(num) for num in range(1, len(self.model_preds) + 1)]

        pipeline_target = self.pipeline._return_target_col()

        for i in range(0, len(self.Y_test)):
            err = (list(self.Y_test)[i] - list(self.model_preds)[i]) * 2
            errors.append(err)

        err_MSE_df = pd.DataFrame(
            list(zip(num_predictions, errors)), columns=["Prediction", "MSE"]
        )
        sns.set_palette(sns.color_palette("rocket"))
        sns.set_style("whitegrid")
        sns.lineplot(data=err_MSE_df, y="MSE", x="Prediction", dashes=False).set_title(
            "Mean Squared Error"
        )
        return MAE, MSE, RMSE

    def _return_preds(self):
        return self.model_preds

    def _return_preds_with_dates(self):
        self.model_preds["Dates"] = self.X_test_dates
        return self.model_preds

    def _return_final_data(self):
        return self.final_data

    def _return_model(self):
        return self.model
