"""
Mixed Portfolio Example - Crypto + Traditional Securities

This example demonstrates how to use the unified security framework
to analyze both cryptocurrency and traditional credit securities together.

Features demonstrated:
1. Loading mixed portfolio (crypto + Bloomberg credit)
2. Data alignment across different market calendars
3. Cross-asset feature engineering
4. Unified model training
5. Multi-asset analysis

Usage:
    python examples/mixed_portfolio_example.py
"""

import sys
sys.path.insert(0, '..')

from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

from _data_sources import Security, MixedPortfolioDataSource
from _preprocessing import _preprocess_xlsx
from _models import _build_model


def create_mixed_portfolio():
    """
    Create a mixed portfolio with crypto and traditional securities.

    Returns:
        List[Security]: List of Security objects
    """
    securities = [
        # Cryptocurrency - Bitcoin
        Security(
            identifier="BTC/USDT",
            security_type="crypto_spot",
            source="binance",
            fields=["close", "volume"],
            metadata={
                "exchange": "binance",
                "timeframe": "1d",  # Daily candles
                "display_name": "Bitcoin"
            },
            features={
                "technical_indicators": True
            }
        ),

        # Cryptocurrency - Ethereum
        Security(
            identifier="ETH/USDT",
            security_type="crypto_spot",
            source="binance",
            fields=["close", "volume"],
            metadata={
                "exchange": "binance",
                "timeframe": "1d",
                "display_name": "Ethereum"
            },
            features={
                "technical_indicators": True
            }
        ),

        # Traditional Credit - US Aggregate Bond Index
        Security(
            identifier="LF98TRUU Index",
            security_type="credit_index",
            source="bloomberg",
            fields=["OAS"],
            metadata={
                "display_name": "US Aggregate Bond Index",
                "excel_fallback": "data/bloomberg_export.xlsx"
            },
            features={
                "momentum": True
            }
        ),

        # Traditional Credit - US Corporate Bond Index
        Security(
            identifier="LUACTRUU Index",
            security_type="credit_index",
            source="bloomberg",
            fields=["OAS"],
            metadata={
                "display_name": "US Corporate Bond Index",
                "excel_fallback": "data/bloomberg_export.xlsx"
            },
            features={
                "momentum": True
            }
        ),
    ]

    return securities


def load_mixed_portfolio_data(securities, start_date, end_date):
    """
    Load data for mixed portfolio.

    Args:
        securities: List of Security objects
        start_date: Start date for data
        end_date: End date for data

    Returns:
        pd.DataFrame: Unified DataFrame with all securities
    """
    print("Loading mixed portfolio data...")
    print(f"Securities: {len(securities)}")
    print(f"  - Crypto: {sum(1 for s in securities if s.is_crypto())}")
    print(f"  - Credit: {sum(1 for s in securities if s.is_credit())}")
    print(f"Date range: {start_date.date()} to {end_date.date()}\n")

    # Create mixed portfolio data source
    source = MixedPortfolioDataSource(
        securities=securities,
        start_date=start_date,
        end_date=end_date,
        alignment_method="outer",  # Keep all dates
        fill_method="ffill",       # Forward fill missing values
        fill_limit=5,              # Max 5 days forward fill
        validate=True              # Validate data quality
    )

    # Load data
    df = source.load_data()

    print(f"\nLoaded {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}\n")

    return df


def analyze_cross_asset_correlations(df):
    """
    Analyze correlations between crypto and credit securities.

    Args:
        df: DataFrame with mixed portfolio data

    Returns:
        pd.DataFrame: Correlation matrix
    """
    print("=" * 80)
    print("Cross-Asset Correlation Analysis")
    print("=" * 80)

    # Select crypto and credit columns
    crypto_cols = [c for c in df.columns if c.startswith(("BTC_", "ETH_"))]
    credit_cols = [c for c in df.columns if "_OAS" in c]

    # Calculate correlations
    corr_data = df[crypto_cols + credit_cols].corr()

    print("\nCorrelation Matrix:")
    print(corr_data)

    # Highlight key correlations
    print("\n\nKey Cross-Asset Correlations:")
    for crypto_col in crypto_cols:
        for credit_col in credit_cols:
            corr = corr_data.loc[crypto_col, credit_col]
            print(f"{crypto_col} vs {credit_col}: {corr:.3f}")

    return corr_data


def detect_regime_changes(df):
    """
    Detect regime changes (risk-on vs risk-off).

    Risk-On: Crypto up, Credit spreads tightening (down)
    Risk-Off: Crypto down, Credit spreads widening (up)

    Args:
        df: DataFrame with mixed portfolio data

    Returns:
        pd.Series: Regime indicator (-1: risk-off, 0: neutral, 1: risk-on)
    """
    print("\n" + "=" * 80)
    print("Regime Detection (Risk-On / Risk-Off)")
    print("=" * 80)

    # Calculate momentum for BTC and credit spreads
    window = 20

    if "BTC_USDT_close" in df.columns:
        btc_momentum = df["BTC_USDT_close"].pct_change(window)
    else:
        print("Warning: BTC_USDT_close not found")
        return pd.Series(index=df.index, data=0)

    if "LF98TRUU_Index_OAS" in df.columns:
        spread_momentum = df["LF98TRUU_Index_OAS"].pct_change(window)
    else:
        print("Warning: LF98TRUU_Index_OAS not found")
        return pd.Series(index=df.index, data=0)

    # Regime logic:
    # Risk-On: BTC up (>0) and spreads down (<0)
    # Risk-Off: BTC down (<0) and spreads up (>0)
    regime = pd.Series(index=df.index, data=0)
    regime[(btc_momentum > 0) & (spread_momentum < 0)] = 1   # Risk-On
    regime[(btc_momentum < 0) & (spread_momentum > 0)] = -1  # Risk-Off

    # Count regimes
    risk_on_count = (regime == 1).sum()
    risk_off_count = (regime == -1).sum()
    neutral_count = (regime == 0).sum()

    print(f"\nRegime Distribution:")
    print(f"  Risk-On:  {risk_on_count} days ({risk_on_count/len(regime)*100:.1f}%)")
    print(f"  Risk-Off: {risk_off_count} days ({risk_off_count/len(regime)*100:.1f}%)")
    print(f"  Neutral:  {neutral_count} days ({neutral_count/len(regime)*100:.1f}%)")

    return regime


def train_unified_model(df, target_col):
    """
    Train a unified model on mixed portfolio data.

    Args:
        df: DataFrame with mixed portfolio data
        target_col: Target column for prediction

    Returns:
        _build_model: Trained model object
    """
    print("\n" + "=" * 80)
    print(f"Training Unified Model (Target: {target_col})")
    print("=" * 80)

    # Prepare momentum columns (all numeric columns except Dates and target)
    momentum_cols = [
        c for c in df.columns
        if c != "Dates" and c != target_col and df[c].dtype in ['float64', 'int64']
    ]

    print(f"\nMomentum columns ({len(momentum_cols)}):")
    for col in momentum_cols[:5]:  # Show first 5
        print(f"  - {col}")
    if len(momentum_cols) > 5:
        print(f"  ... and {len(momentum_cols) - 5} more")

    # Preprocess data
    preprocessor = _preprocess_xlsx(
        data=df,
        target_col=target_col,
        momentum_list=momentum_cols,
        momentum_X_days=[5, 10, 15],
        momentum_Y_days=30,
        forecast_list=[1, 3, 7, 15, 30],
        split_percentage=0.20,
        sequential=False
    )

    # Train model
    model = _build_model(
        preprocessor=preprocessor,
        model_name="XGBoost",
        estimators=1000
    )

    print("\nModel training complete!")

    return model


def visualize_mixed_portfolio(df):
    """
    Create visualization of mixed portfolio.

    Args:
        df: DataFrame with mixed portfolio data
    """
    print("\n" + "=" * 80)
    print("Generating Visualizations")
    print("=" * 80)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Crypto prices
    if "BTC_USDT_close" in df.columns:
        axes[0].plot(df["Dates"], df["BTC_USDT_close"], label="BTC/USDT", linewidth=2)
    if "ETH_USDT_close" in df.columns:
        axes[0].plot(df["Dates"], df["ETH_USDT_close"], label="ETH/USDT", linewidth=2)

    axes[0].set_title("Cryptocurrency Prices", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Price (USDT)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Credit spreads
    if "LF98TRUU_Index_OAS" in df.columns:
        axes[1].plot(df["Dates"], df["LF98TRUU_Index_OAS"],
                     label="US Aggregate OAS", linewidth=2)
    if "LUACTRUU_Index_OAS" in df.columns:
        axes[1].plot(df["Dates"], df["LUACTRUU_Index_OAS"],
                     label="US Corporate OAS", linewidth=2)

    axes[1].set_title("Credit Spreads (OAS)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("OAS (basis points)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("_img/mixed_portfolio_overview.png", dpi=300, bbox_inches="tight")
    print("\nSaved visualization to _img/mixed_portfolio_overview.png")

    plt.close()


def main():
    """Main execution function."""
    print("=" * 80)
    print("BBG Credit Momentum - Mixed Portfolio Example")
    print("=" * 80)
    print("\nThis example demonstrates unified analysis of crypto + traditional securities\n")

    # Define date range
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)

    try:
        # 1. Create portfolio
        securities = create_mixed_portfolio()

        # 2. Load data
        df = load_mixed_portfolio_data(securities, start_date, end_date)

        # 3. Analyze correlations
        correlations = analyze_cross_asset_correlations(df)

        # 4. Detect regime changes
        regime = detect_regime_changes(df)

        # 5. Train unified model
        # Choose target based on available columns
        if "BTC_USDT_close" in df.columns:
            target = "BTC_USDT_close"
        elif "LF98TRUU_Index_OAS" in df.columns:
            target = "LF98TRUU_Index_OAS"
        else:
            print("\nWarning: No suitable target column found")
            return

        model = train_unified_model(df, target)

        # 6. Visualize
        visualize_mixed_portfolio(df)

        print("\n" + "=" * 80)
        print("Example Complete!")
        print("=" * 80)
        print("\nKey Takeaways:")
        print("1. Successfully loaded mixed portfolio (crypto + credit)")
        print("2. Analyzed cross-asset correlations")
        print("3. Detected risk-on/risk-off regimes")
        print("4. Trained unified model on combined features")
        print("5. Generated visualizations")
        print("\nNext steps:")
        print("- Modify securities list in create_mixed_portfolio()")
        print("- Adjust date range")
        print("- Experiment with different models")
        print("- Add more cross-asset features")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
