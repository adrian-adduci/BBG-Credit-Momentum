################################################################################
# Author: Adrian Adduci
# Email: FAA2160@columbia.edu
################################################################################
"""
Command-line walk-forward backtest.

Evaluates the forecasting model against a random-walk baseline across several
horizons and prints a skill score for each. A negative skill score means the
model does not beat "tomorrow's spread is today's spread", which is a
legitimate result and the only honest way to read an error metric on a series
that is close to a random walk.

Usage:
    python backtest.py data/Economic_Data_2020_08_01.xlsx \\
        --target LF98TRUU_Index_OAS \\
        --horizons 1 5 10 30 \\
        --mode change

    # Both framings side by side
    python backtest.py data/Economic_Data_2020_08_01.xlsx \\
        --target LF98TRUU_Index_OAS --mode both
"""

import argparse
import sys
import warnings

import pandas as pd

from forecasting import walk_forward_backtest

warnings.filterwarnings("ignore", category=FutureWarning)


def _load(path):
    """Read an Excel or CSV export into a DataFrame."""
    if str(path).lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)


def _run_mode(df, target, horizons, mode, n_splits, lags):
    print(f"\n=== target_mode = {mode!r} ===")
    print(
        f"{'horizon':>8} {'model RMSE':>12} {'naive RMSE':>12} "
        f"{'skill':>9}  verdict"
    )
    print("-" * 64)

    for horizon in horizons:
        result = walk_forward_backtest(
            df,
            target_col=target,
            horizon=horizon,
            n_splits=n_splits,
            target_lags=lags,
            target_mode=mode,
        )
        verdict = (
            "beats naive" if result.skill_score > 0 else "NO SKILL vs random walk"
        )
        print(
            f"{horizon:>8} {result.model_rmse:>12.4f} "
            f"{result.baseline_rmse:>12.4f} {result.skill_score:>+9.3f}  {verdict}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Walk-forward backtest against a random-walk baseline."
    )
    parser.add_argument("data", help="Path to the Excel or CSV export")
    parser.add_argument(
        "--target", required=True, help="Column to forecast, e.g. LF98TRUU_Index_OAS"
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[1, 5, 10, 30],
        help="Forecast horizons in periods (default: 1 5 10 30)",
    )
    parser.add_argument(
        "--mode",
        choices=["level", "change", "both"],
        default="change",
        help=(
            "Predict the future level, the future change, or both. 'change' is "
            "the right default for spread series: tree models cannot "
            "extrapolate a drifting level (default: change)"
        ),
    )
    parser.add_argument(
        "--splits", type=int, default=5, help="Walk-forward folds (default: 5)"
    )
    parser.add_argument(
        "--lags",
        type=int,
        nargs="*",
        default=[1, 2, 5],
        help="Strictly positive target lags to use as features (default: 1 2 5)",
    )
    args = parser.parse_args(argv)

    df = _load(args.data)

    if args.target not in df.columns:
        parser.error(
            f"Target column {args.target!r} not found. Available: "
            f"{sorted(df.columns)}"
        )

    print(f"Loaded {len(df)} rows from {args.data}")
    print(f"Target: {args.target}")

    modes = ["level", "change"] if args.mode == "both" else [args.mode]
    for mode in modes:
        _run_mode(df, args.target, args.horizons, mode, args.splits, args.lags)

    print(
        "\nSkill score = 1 - model_RMSE / naive_RMSE. Positive means the model "
        "beat\nthe random walk; zero or negative means it did not."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
