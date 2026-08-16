################################################################################
# Author: Adrian Adduci
# Email: FAA2160@columbia.edu
################################################################################
"""
Forecasting primitives with leakage-safe defaults.

This module exists because the original pipeline built its supervised learning
problem in a way that quietly guaranteed optimistic results:

* features and labels were drawn from the same timestamp (a nowcast, not a
  forecast);
* where a horizon *was* applied it used ``shift(+h)``, which yields the value
  from ``h`` periods **ago**, while the column was named ``_D_Ahead_Actual``;
* the raw target was left in the feature matrix, so importance rankings were
  measuring the target against itself;
* the train/test split ran through ``train_test_split(shuffle=...)``, which
  interleaves future and past.

Every function here is written so that the safe behaviour is the only
behaviour. There is no flag that turns leakage back on.

Example:
    >>> X, y, dates = make_supervised(df, "LF98TRUU_Index_OAS", horizon=5)
    >>> split = time_ordered_split(X, y, dates, test_size=0.2)
    >>> result = walk_forward_backtest(df, "LF98TRUU_Index_OAS", horizon=5)
    >>> print(result.summary())
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

__all__ = [
    "make_supervised",
    "time_ordered_split",
    "random_walk_baseline",
    "walk_forward_backtest",
    "SplitResult",
    "FoldResult",
    "BacktestResult",
    "default_model_factory",
]

DEFAULT_RANDOM_STATE = 42


def default_model_factory() -> XGBRegressor:
    """A deterministic gradient-boosted regressor used when none is supplied."""
    return XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=DEFAULT_RANDOM_STATE,
        n_jobs=1,
        verbosity=0,
    )


################################################################################
# Supervised problem construction
################################################################################


def make_supervised(
    df: pd.DataFrame,
    target_col: str,
    horizon: int,
    feature_cols: Optional[Sequence[str]] = None,
    date_col: str = "Dates",
    target_lags: Optional[Sequence[int]] = None,
    target_mode: str = "level",
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Build a leakage-safe supervised learning problem from a time-indexed frame.

    The label for the row observed at time ``t`` is the target at ``t +
    horizon``. Rows whose future value is not observable are dropped rather
    than imputed, because an imputed future is an invented one.

    The raw target column is **always** removed from the feature matrix. Past
    values of the target are legitimate predictors, but only via ``target_lags``
    which forces an explicit, strictly positive lag.

    Args:
        df: Source frame. Must contain ``target_col``; ``date_col`` is used for
            ordering and is excluded from the features.
        target_col: Column to forecast.
        horizon: Number of periods ahead to forecast. Must be >= 1.
        feature_cols: Explicit feature whitelist. Defaults to every column
            except the target and the date column.
        date_col: Name of the date column (default: "Dates").
        target_lags: Optional list of strictly positive lags of the target to
            add as features, e.g. ``[1, 5]``.
        target_mode: ``"level"`` (default) labels the raw value at t+horizon.
            ``"change"`` labels ``target[t+horizon] - target[t]``. For series
            that behave like a random walk, "change" is usually the correct
            framing: gradient-boosted trees predict a weighted average of
            training leaf values and so cannot extrapolate a drifting level at
            all, which makes a "level" model lose to the naive forecast for
            reasons that have nothing to do with market predictability.

    Returns:
        tuple: ``(X, y, dates)`` sharing a common index, where ``dates`` holds
        the timestamp at which each row's features were observed.

    Raises:
        ValueError: If ``horizon`` < 1, a lag is < 1, ``target_mode`` is not
            recognised, or a column is missing.
    """
    if target_mode not in ("level", "change"):
        raise ValueError(
            f"target_mode must be 'level' or 'change', got {target_mode!r}"
        )

    if not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError(
            f"horizon must be an integer >= 1 (got {horizon!r}). A horizon of 0 "
            "produces a nowcast, not a forecast."
        )

    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col!r} not found in DataFrame")

    frame = df.copy()

    if date_col in frame.columns:
        frame = frame.sort_values(date_col)
        dates = frame[date_col]
    else:
        dates = pd.Series(frame.index, index=frame.index, name="index")

    # The label leads the features: shift(-horizon) reaches forward in time.
    future = frame[target_col].shift(-horizon)
    if target_mode == "change":
        y = future - frame[target_col]
        y.name = f"{target_col}_change_over_{horizon}"
    else:
        y = future
        y.name = f"{target_col}_t_plus_{horizon}"

    if feature_cols is None:
        excluded = {target_col, date_col}
        features = [col for col in frame.columns if col not in excluded]
    else:
        missing = [col for col in feature_cols if col not in frame.columns]
        if missing:
            raise ValueError(f"Feature columns not found in DataFrame: {missing}")
        features = [col for col in feature_cols if col not in {target_col, date_col}]

    X = frame[features].copy()

    if target_lags:
        for lag in target_lags:
            if not isinstance(lag, (int, np.integer)) or lag < 1:
                raise ValueError(
                    f"target_lags must be integers >= 1 (got {lag!r}). A lag of 0 "
                    "is the contemporaneous target and would leak."
                )
            X[f"{target_col}_lag_{lag}"] = frame[target_col].shift(lag)

    # Drop rows with an unobservable future or an incomplete lag window.
    usable = y.notna() & X.notna().all(axis=1)

    return X.loc[usable], y.loc[usable], dates.loc[usable]


################################################################################
# Splitting
################################################################################


@dataclass
class SplitResult:
    """A chronological train/test split."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_dates: pd.Series
    test_dates: pd.Series


def time_ordered_split(
    X: pd.DataFrame,
    y: pd.Series,
    dates: Optional[pd.Series] = None,
    test_size: float = 0.2,
) -> SplitResult:
    """
    Split chronologically so that no training row postdates a test row.

    This deliberately offers no ``shuffle`` argument. Shuffling a time series
    trains the model on the future, which is the single easiest way to produce
    a backtest that cannot be reproduced in live trading.

    Args:
        X: Feature matrix, already in chronological order.
        y: Labels aligned to ``X``.
        dates: Optional observation timestamps aligned to ``X``.
        test_size: Fraction of the most recent rows held out (default: 0.2).

    Returns:
        SplitResult: The four arrays plus the train and test date ranges.

    Raises:
        ValueError: If ``test_size`` is not strictly between 0 and 1, or the
            split would leave either side empty.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size!r}")

    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")

    n_test = int(round(len(X) * test_size))
    if n_test < 1 or n_test >= len(X):
        raise ValueError(
            f"test_size={test_size} yields {n_test} test rows out of {len(X)}"
        )

    cut = len(X) - n_test

    if dates is None:
        dates = pd.Series(X.index, index=X.index)

    return SplitResult(
        X_train=X.iloc[:cut],
        X_test=X.iloc[cut:],
        y_train=y.iloc[:cut],
        y_test=y.iloc[cut:],
        train_dates=dates.iloc[:cut],
        test_dates=dates.iloc[cut:],
    )


################################################################################
# Baseline
################################################################################


def random_walk_baseline(last_known: pd.Series) -> pd.Series:
    """
    The naive forecast: the best estimate of a future level is the current one.

    For most credit spread series this is a genuinely hard benchmark to beat.
    A model that cannot beat it has no demonstrated skill, and reporting model
    error without this comparison makes noise look like signal.

    Args:
        last_known: The target value observed at each feature timestamp.

    Returns:
        pd.Series: Predictions equal to the last observed value.
    """
    return last_known.copy()


################################################################################
# Walk-forward evaluation
################################################################################


@dataclass
class FoldResult:
    """Metrics for a single walk-forward fold."""

    fold: int
    n_train: int
    n_test: int
    train_end_date: object
    test_start_date: object
    model_rmse: float
    model_mae: float
    baseline_rmse: float
    baseline_mae: float

    @property
    def skill_score(self) -> float:
        """1 - model_rmse / baseline_rmse. Positive means it beat the naive."""
        if self.baseline_rmse == 0:
            return float("nan")
        return 1.0 - (self.model_rmse / self.baseline_rmse)


@dataclass
class BacktestResult:
    """Aggregate walk-forward results across all folds."""

    folds: List[FoldResult] = field(default_factory=list)
    target_col: str = ""
    horizon: int = 1

    @property
    def model_rmse(self) -> float:
        return float(np.mean([f.model_rmse for f in self.folds]))

    @property
    def baseline_rmse(self) -> float:
        return float(np.mean([f.baseline_rmse for f in self.folds]))

    @property
    def skill_score(self) -> float:
        """
        Aggregate skill against the random walk.

        Positive means the model beat the naive forecast; zero or negative
        means it did not, which is a legitimate and publishable result.
        """
        if self.baseline_rmse == 0:
            return float("nan")
        return float(1.0 - (self.model_rmse / self.baseline_rmse))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "fold": f.fold,
                    "n_train": f.n_train,
                    "n_test": f.n_test,
                    "train_end": f.train_end_date,
                    "test_start": f.test_start_date,
                    "model_rmse": f.model_rmse,
                    "baseline_rmse": f.baseline_rmse,
                    "skill_score": f.skill_score,
                }
                for f in self.folds
            ]
        )

    def summary(self) -> str:
        verdict = (
            "model beats random walk"
            if self.skill_score > 0
            else "NO SKILL vs random walk"
        )
        return (
            f"{self.target_col} @ {self.horizon}d over {len(self.folds)} folds: "
            f"model RMSE {self.model_rmse:.4f} vs baseline RMSE "
            f"{self.baseline_rmse:.4f} -> skill {self.skill_score:+.3f} ({verdict})"
        )


def walk_forward_backtest(
    df: pd.DataFrame,
    target_col: str,
    horizon: int = 1,
    n_splits: int = 5,
    model_factory: Optional[Callable[[], object]] = None,
    feature_cols: Optional[Sequence[str]] = None,
    date_col: str = "Dates",
    target_lags: Optional[Sequence[int]] = None,
    target_mode: str = "level",
) -> BacktestResult:
    """
    Evaluate a model with an expanding-window walk-forward backtest.

    Each fold trains only on data preceding its test window, and an embargo of
    ``horizon`` rows is removed from the end of every training set. Without
    that embargo the last training labels reach into the test window, which
    leaks the very thing being measured.

    Every fold is scored against :func:`random_walk_baseline` so the result is
    always reported as skill *relative to doing nothing*.

    Args:
        df: Source frame.
        target_col: Column to forecast.
        horizon: Periods ahead to forecast (default: 1).
        n_splits: Number of walk-forward folds (default: 5).
        model_factory: Zero-argument callable returning an unfitted estimator.
            Defaults to :func:`default_model_factory`.
        feature_cols: Optional explicit feature whitelist.
        date_col: Name of the date column (default: "Dates").
        target_lags: Optional strictly positive lags of the target to include.

    Returns:
        BacktestResult: Per-fold and aggregate metrics.

    Raises:
        ValueError: If there is not enough usable data for ``n_splits`` folds.
    """
    if model_factory is None:
        model_factory = default_model_factory

    X, y, dates = make_supervised(
        df,
        target_col=target_col,
        horizon=horizon,
        feature_cols=feature_cols,
        date_col=date_col,
        target_lags=target_lags,
        target_mode=target_mode,
    )

    if len(X) < (n_splits + 1) * (horizon + 1):
        raise ValueError(
            f"Not enough usable rows ({len(X)}) for {n_splits} folds at "
            f"horizon {horizon}"
        )

    # The naive forecast. Predicting the level, it is the last observed value;
    # predicting the change, it is zero -- "nothing will happen". Both express
    # the same random-walk hypothesis, so skill scores stay comparable.
    if target_mode == "change":
        naive = pd.Series(0.0, index=X.index)
    else:
        naive = df[target_col].loc[X.index]

    result = BacktestResult(target_col=target_col, horizon=horizon)
    splitter = TimeSeriesSplit(n_splits=n_splits)

    for fold_number, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        # Embargo: a training row at position i carries a label from i+horizon,
        # so the final `horizon` training rows overlap the test window.
        if horizon > 0:
            train_idx = train_idx[: max(len(train_idx) - horizon, 0)]
        if len(train_idx) == 0:
            continue

        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        # A fresh estimator per fold: refitting one shared instance is how the
        # original code silently mutated the model between reported metrics.
        model = model_factory()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        baseline_preds = random_walk_baseline(naive.iloc[test_idx])

        result.folds.append(
            FoldResult(
                fold=fold_number,
                n_train=len(train_idx),
                n_test=len(test_idx),
                train_end_date=dates.iloc[train_idx[-1]],
                test_start_date=dates.iloc[test_idx[0]],
                model_rmse=float(np.sqrt(mean_squared_error(y_test, preds))),
                model_mae=float(mean_absolute_error(y_test, preds)),
                baseline_rmse=float(
                    np.sqrt(mean_squared_error(y_test, baseline_preds))
                ),
                baseline_mae=float(mean_absolute_error(y_test, baseline_preds)),
            )
        )

    return result
