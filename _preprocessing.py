################################################################################
# Author: Adrian Adduci
# Email: FAA2160@columbia.edu
################################################################################

# WIP: Current DF return begins 1-month after start date/ need to decrease
#   for shorter analysis windows

import logging
import os
import pathlib

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tabulate import tabulate

path = pathlib.Path(__file__).parent.absolute()
logger = logging.getLogger("_preprocess_xlsx")
logger.setLevel(logging.INFO)
handler = logging.FileHandler(path / "logs" / "_preprocess.log")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
os.environ["NUMEXPR_MAX_THREADS"] = "16"
################################################################################
# Pre-Processing of XLSX Into Pandas Dataframe
################################################################################


class _preprocess_xlsx:
    """
    Preprocesses Excel data for machine learning model training.

    Loads Bloomberg economic data from Excel files, creates momentum features,
    splits data into train/test sets, and prepares features for model training.

    Args:
        xlsx_file: Path to the Excel file or file-like object
        target_col: Name of the column to use as the target variable
        forecast_list: List of forecast horizons in days (default: [1, 3, 7, 15, 30])
        momentum_list: List of column names to calculate momentum features for
        split_percentage: Percentage of data to use for testing (default: 0.20)
        sequential: Whether to shuffle data before splitting (default: False)
        momentum_X_days: Short-term windows for momentum calculation (default: [5, 10, 15])
        momentum_Y_days: Long-term baseline window for momentum (default: 30)

    Raises:
        FileNotFoundError: If xlsx_file doesn't exist
        ValueError: If required columns are missing or data is invalid

    Attributes:
        df: Raw DataFrame loaded from Excel
        complete_data: Cleaned DataFrame with NaN values removed
        X_train, X_test: Training and test feature sets
        Y_train, Y_test: Training and test target sets
        feature_cols: List of feature column names
    """
    def __init__(
        self,
        xlsx_file,
        target_col,
        forecast_list=[1, 3, 7, 15, 30],
        momentum_list=[],
        split_percentage=0.20,
        sequential=False,
        momentum_X_days=[5, 10, 15],
        momentum_Y_days=30,
    ):

        logger.info(f" Preprocessing, using XLSX: {xlsx_file} and target(s): {target_col}")

        # Validate Excel file exists
        if not pathlib.Path(xlsx_file).is_file():
            error_msg = f"Excel file not found: {xlsx_file}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Load Excel file with error handling
        try:
            self.df = pd.read_excel(xlsx_file)
        except Exception as e:
            error_msg = f"Failed to read Excel file: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Validate required columns
        if "Dates" not in self.df.columns:
            error_msg = "Excel file must contain a 'Dates' column"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if target_col not in self.df.columns:
            error_msg = f"Target column '{target_col}' not found in Excel file"
            logger.error(error_msg)
            raise ValueError(error_msg)
        self.label_encoder = preprocessing.LabelEncoder()
        self.target_col = target_col
        self.forecast_list = forecast_list
        self.momentum_list = momentum_list
        self.split_percentage = split_percentage
        self.sequential = sequential
        self.momentum_X_days = momentum_X_days
        self.momentum_Y_days = momentum_Y_days

        self._add_custom_features()

        self.complete_data = self.df.dropna().copy()

        self.X = self.complete_data.drop([self.target_col, "Dates"], axis=1)

        self.feature_cols = self.X.columns

        self.Y = self.complete_data[self.target_col]

        logger.debug(" Splitting Test and Training Data")

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X, self.Y, test_size=self.split_percentage, shuffle=self.sequential
        )

        self._find_entropy_of_feature(self.Y)

        # Encode Target
        logger.debug(" Encoding target training and test data")
        self.Y_encoded = self.label_encoder.fit_transform(self.Y)
        self.Y_train_encoded = self.label_encoder.fit_transform(self.Y_train)
        self.Y_test_encoded = self.label_encoder.fit_transform(self.Y_test)

    ################################################################################
    # Class Methods
    ################################################################################

    def _find_entropy_of_feature(self, df_target_col):
        """
        Calculate the entropy of a target feature.

        Entropy measures the uncertainty/information content in a feature.
        Formula: -Σ(p_i * log2(p_i)) for each unique value i

        Args:
            df_target_col: Pandas Series containing the target feature values

        Returns:
            float: Entropy value (higher = more uncertainty)
        """
        target_counts = df_target_col.value_counts().astype(float).values
        total = df_target_col.count()
        probas = target_counts / total
        entropy_components = probas * np.log2(probas)
        entropy = -entropy_components.sum()
        logger.info(f" Entropy of target feature is {entropy}")
        return entropy

    def _information_gain(self, info_column, target_col, threshold=0.5):
        """
        Calculate information gain of a feature relative to a target.

        Information gain measures how much knowing a feature reduces uncertainty
        about the target. Formula: H(target) - H(target|feature)

        Args:
            info_column: Name of the feature column to evaluate
            target_col: Name of the target column
            threshold: Split threshold for the feature (default: 0.5)

        Returns:
            float: Information gain value (higher = more predictive power)
        """

        data_above_thresh = self.df[self.df[info_column] > threshold]
        data_below_thresh = self.df[self.df[info_column] <= threshold]

        entropy_target_col = self._find_entropy_of_feature(self.df[target_col])
        entropy_above = self._find_entropy_of_feature(data_above_thresh[target_col])
        entropy_below = self._find_entropy_of_feature(data_below_thresh[target_col])

        ct_above = data_above_thresh.shape[0]
        ct_below = data_below_thresh.shape[0]
        tot = float(self.df.shape[0])
        IG = (
            entropy_target_col
            - entropy_above * ct_above / tot
            - entropy_below * ct_below / tot
        )
        logger.info(" IG of {} and {} at threshold {} is {}").format(
            info_column, target_col, threshold, IG
        )
        return IG

    def best_threshold(self, info_column, target_col, criteria=_information_gain):
        maximum_ig = 0
        maximum_threshold = 0

        for thresh in self.df[info_column]:
            IG = criteria(info_column, target_col, thresh)
            if IG > maximum_ig:
                maximum_ig = IG
                maximum_threshold = thresh

        return (maximum_threshold, maximum_ig)

    ################################################################################
    # Customize Import Data For Bloomberg
    ################################################################################

    def _add_custom_features(self):
        # Optional: Convert EARN_DOWN and EARN_UP if they exist
        if "EARN_DOWN" in self.df.columns:
            try:
                self.df["EARN_DOWN"] = self.df["EARN_DOWN"].astype(np.float16)
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not convert EARN_DOWN to float16: {str(e)}")
        else:
            logger.debug("EARN_DOWN not included in Excel file (optional column)")

        if "EARN_UP" in self.df.columns:
            try:
                self.df["EARN_UP"] = self.df["EARN_UP"].astype(np.float16)
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not convert EARN_UP to float16: {str(e)}")
        else:
            logger.debug("EARN_UP not included in Excel file (optional column)")

        # Add new momentum features
        self._add_momentum(
            self.momentum_list, self.momentum_X_days, self.momentum_Y_days
        )

    def _add_momentum(self, momentum_list, momentum_X_days, momentum_Y_days):
        """
        Add momentum features using rolling window averages.

        Momentum is calculated as the percent change from short-term average
        to long-term average: (short_avg - long_avg) / long_avg

        Args:
            momentum_list: List of column names to calculate momentum for
            momentum_X_days: List of short-term window sizes (e.g., [5, 10, 15])
            momentum_Y_days: Long-term baseline window size (e.g., 30)

        Raises:
            ValueError: If momentum columns don't exist in the DataFrame

        Creates:
            New columns named "{column}_{window}day_rolling_average" for each
            combination of column and window size
        """

        logger.info(f" momentum_list: {momentum_list}")

        if not momentum_list:
            return

        # Validate momentum columns exist
        missing_cols = [col for col in momentum_list if col not in self.df.columns]
        if missing_cols:
            error_msg = f"Momentum columns not found in Excel: {missing_cols}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if momentum_X_days is None:
            momentum_X_days = self.momentum_X_days
        if momentum_Y_days is None:
            momentum_Y_days = self.momentum_Y_days

        for item in momentum_list:
            for win in momentum_X_days:
                new_item = f"{item}_{win}day_rolling_average"
                try:
                    # Cache rolling calculations to avoid redundant computation
                    y_day_avg = self.df[item].rolling(window=momentum_Y_days).mean()
                    x_day_avg = self.df[item].rolling(window=win).mean()
                    # Calculate momentum: (short_term - long_term) / long_term
                    self.df[new_item] = (x_day_avg - y_day_avg) / y_day_avg
                    logger.info(f" Adding new col for {new_item}")
                except Exception as e:
                    logger.error(f"Failed to calculate momentum for {item}: {str(e)}")
                    raise

    # Add column to df with net change from day to dh in future
    def _change_over_days(self, dh=None):
        if dh == None:
            for dh in self.forecast_list:

                logger.debug(f" Processing for {dh} days ahead")
                self.target_change = f"{self.target_col}_{dh}_Day_Change"
                self.df[str(self.target_change)] = self.df[self.target_col] - self.df[
                    self.target_col
                ].shift(dh)
        else:
            for d in dh:

                logger.debug(f" Processing for {d} days ahead")
                self.target_change = f"{self.target_col}_{d}_Day_Change"
                self.df[str(self.target_change)] = self.df[self.target_col] - self.df[
                    self.target_col
                ].shift(d)

    ################################################################################
    # Returns
    ################################################################################

    # Add column to df with actual value of target in days ahead
    def _return_data_with_dh_actuals(self, days_ahead=None, target=None):
        max_forecast = days_ahead
        if max_forecast == None:
            max_forecast = max(self.forecast_list)
        days_to_go = list(range(1, max_forecast + 1))
        data_dict, X_dict, Y_dict = {}, {}, {}

        for dh in days_to_go:

            temp_data = self.complete_data.copy()

            # Add predicative column for days ahead (d)
            forecast_name = "{0}_{1}D_Ahead_Actual".format(self.target_col, dh)
            # logger.info("Adding {0} ".format(forecast_name))

            temp_data[forecast_name] = temp_data[self.target_col].shift(dh)
            temp_data = temp_data.dropna()
            Y_dict[dh] = temp_data[[forecast_name]]

            data_dict[dh] = temp_data

            temp_data = temp_data.drop([forecast_name, "Dates"], axis=1)
            X_dict[dh] = temp_data

        return data_dict, X_dict, Y_dict

    def _set_feature_names(self, new_features):
        self.feature_cols = new_features

    def _set_target_col(self, target_col):
        self.target_col = target_col

    def _print_target_col(self):
        print(tabulate(self.target_col))

    def _return_target_col(self):
        return self.target_col

    def _return_feature_names(self):
        return self.feature_cols

    def _return_complete_data(self):
        return self.complete_data

    def _return_test_and_train_data(self):
        return self.X_train, self.X_test, self.Y_train, self.Y_test

    def _return_Y_encoded(self):
        return self.Y_encoded, self.Y_train_encoded, self.Y_test_encoded

    def _return_dataframe(self):
        return self.complete_data

    def _return_X_Y_dataframe(self):
        return self.X, self.Y

    def _return_forecast_list(self):
        return self.forecast_list

    def __str__(self):
        return print(tabulate(self.df, headers="keys", tablefmt="psql"))
