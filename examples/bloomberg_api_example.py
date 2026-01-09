"""
Bloomberg API Usage Examples

This script demonstrates how to use the Bloomberg Terminal API integration
for fetching credit market data.

Examples covered:
1. Basic Bloomberg API connection and data fetching
2. Fetching multiple securities and fields
3. Error handling and retry logic
4. Integration with analysis pipeline
5. Batch processing for large security lists

Prerequisites:
- Bloomberg Terminal installed and running
- blpapi library installed: pip install --index-url=https://bcms.bloomberg.com/pip/simple blpapi
- DAPI configured in Bloomberg Terminal (type DAPI<GO>)

Usage:
    python examples/bloomberg_api_example.py
"""

import sys
sys.path.insert(0, '..')

from datetime import datetime
import pandas as pd

from _data_sources import (
    BloombergAPIDataSource,
    BloombergAPIError,
    BloombergTerminalNotRunning
)
from _preprocessing import _preprocess_xlsx
from _models import _build_model


def example_1_basic_connection():
    """
    Example 1: Basic Bloomberg API connection and data fetching.

    This example demonstrates the simplest use case: connecting to Bloomberg
    and fetching OAS data for a single credit index.
    """
    print("=" * 80)
    print("Example 1: Basic Bloomberg API Connection")
    print("=" * 80)

    try:
        # Define what we want to fetch
        securities = ["LF98TRUU Index"]  # US Aggregate Bond Index
        fields = ["OAS"]                 # Option-Adjusted Spread
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2024, 12, 31)

        print(f"\nFetching data for: {securities}")
        print(f"Fields: {fields}")
        print(f"Date range: {start_date.date()} to {end_date.date()}")

        # Create Bloomberg API data source
        source = BloombergAPIDataSource(
            securities=securities,
            fields=fields,
            start_date=start_date,
            end_date=end_date
        )

        # Load data (connects to Bloomberg Terminal)
        print("\nConnecting to Bloomberg Terminal...")
        df = source.load_data()

        # Display results
        print(f"\n✓ Successfully loaded {len(df)} rows")
        print(f"✓ Columns: {list(df.columns)}")
        print("\nFirst 5 rows:")
        print(df.head())

        print("\nLast 5 rows:")
        print(df.tail())

        # Basic statistics
        print("\nData Statistics:")
        print(df.describe())

        return df

    except BloombergTerminalNotRunning as e:
        print(f"\n✗ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure Bloomberg Terminal is running")
        print("2. Type DAPI<GO> in Bloomberg to verify API is configured")
        print("3. Check that API is listening on localhost:8194")
        return None

    except ImportError as e:
        print(f"\n✗ Error: {e}")
        print("\nInstall blpapi with:")
        print("pip install --index-url=https://bcms.bloomberg.com/pip/simple blpapi")
        return None

    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None


def example_2_multiple_securities_fields():
    """
    Example 2: Fetching multiple securities and fields.

    This example shows how to fetch data for multiple credit indices
    with multiple fields (OAS, price, duration, etc.).
    """
    print("\n" + "=" * 80)
    print("Example 2: Multiple Securities and Fields")
    print("=" * 80)

    try:
        # Define multiple securities
        securities = [
            "LF98TRUU Index",    # US Aggregate
            "LUACTRUU Index",    # US Corporate
            "LF98TRHY Index",    # US High Yield
        ]

        # Define multiple fields
        fields = [
            "OAS",           # Option-Adjusted Spread
            "PX_LAST",       # Last Price
            "DTS",           # Duration to Worst
            "YLD_YTM_MID"    # Yield to Maturity
        ]

        print(f"\nFetching {len(securities)} securities:")
        for sec in securities:
            print(f"  - {sec}")

        print(f"\nFields ({len(fields)}):")
        for field in fields:
            print(f"  - {field}")

        # Create data source
        source = BloombergAPIDataSource(
            securities=securities,
            fields=fields,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31)
        )

        # Load data
        print("\nFetching data from Bloomberg...")
        df = source.load_data()

        # Display results
        print(f"\n✓ Successfully loaded {len(df)} rows, {len(df.columns)-1} data columns")

        # Show column naming convention
        print("\nColumn naming convention (security_field):")
        for col in df.columns:
            if col != "Dates":
                print(f"  - {col}")

        # Show sample data
        print("\nSample data (first 3 rows):")
        print(df.head(3).to_string())

        # Compare spreads across indices
        if "LF98TRUU_Index_OAS" in df.columns and "LF98TRHY_Index_OAS" in df.columns:
            print("\nSpread Comparison:")
            print(f"Average US Aggregate OAS: {df['LF98TRUU_Index_OAS'].mean():.2f} bps")
            print(f"Average US High Yield OAS: {df['LF98TRHY_Index_OAS'].mean():.2f} bps")
            spread_diff = df['LF98TRHY_Index_OAS'].mean() - df['LF98TRUU_Index_OAS'].mean()
            print(f"High Yield Premium: {spread_diff:.2f} bps")

        return df

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def example_3_batch_processing():
    """
    Example 3: Batch processing for large security lists.

    Bloomberg API has limits on requests. This example shows how
    batch processing works automatically for 100+ securities.
    """
    print("\n" + "=" * 80)
    print("Example 3: Batch Processing")
    print("=" * 80)

    try:
        # Simulate a large list of securities
        # In practice, you might have 100+ corporate bonds
        securities = [
            "LF98TRUU Index",
            "LUACTRUU Index",
            "LF98TRHY Index",
            "LD01TRUU Index",  # 1-3 Year
            "LD05TRUU Index",  # 5-7 Year
            "LD10TRUU Index",  # 7-10 Year
        ]

        fields = ["OAS"]

        print(f"\nFetching {len(securities)} securities")
        print(f"Batch size: 100 securities per request (configurable)")

        # Create source with custom batch size
        source = BloombergAPIDataSource(
            securities=securities,
            fields=fields,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 3, 31),  # 3 months
            batch_size=3  # Small batch for demonstration
        )

        print("\nFetching data (watch for batch processing)...")
        df = source.load_data()

        print(f"\n✓ Successfully processed all securities in batches")
        print(f"✓ Total rows: {len(df)}")
        print(f"✓ Total columns: {len(df.columns) - 1}")

        return df

    except Exception as e:
        print(f"\n✗ Error: {e}")
        return None


def example_4_retry_logic():
    """
    Example 4: Automatic retry logic for network issues.

    The Bloomberg API integration includes automatic retry with
    exponential backoff for transient failures.
    """
    print("\n" + "=" * 80)
    print("Example 4: Retry Logic")
    print("=" * 80)

    print("\nThe Bloomberg API source includes automatic retry logic:")
    print("- Max retries: 3 (configurable)")
    print("- Backoff: Exponential (2s, 4s, 8s)")
    print("- Retries on: Network errors, timeouts, transient failures")
    print("- No retry on: Terminal not running, invalid securities")

    try:
        source = BloombergAPIDataSource(
            securities=["LF98TRUU Index"],
            fields=["OAS"],
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            timeout=30000,    # 30 second timeout
            max_retries=3     # Try up to 3 times
        )

        print("\nAttempting connection...")
        df = source.load_data()

        print(f"✓ Success after 1 attempt (or after retries if there were issues)")

        return df

    except Exception as e:
        print(f"\n✗ All retries exhausted: {e}")
        return None


def example_5_integration_with_analysis():
    """
    Example 5: End-to-end integration with analysis pipeline.

    This example shows how to fetch Bloomberg data and immediately
    use it for model training and prediction.
    """
    print("\n" + "=" * 80)
    print("Example 5: Integration with Analysis Pipeline")
    print("=" * 80)

    try:
        # Step 1: Fetch Bloomberg data
        print("\nStep 1: Fetching Bloomberg data...")

        source = BloombergAPIDataSource(
            securities=["LF98TRUU Index", "LUACTRUU Index"],
            fields=["OAS"],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2024, 12, 31)
        )

        df = source.load_data()
        print(f"✓ Loaded {len(df)} rows")

        # Step 2: Preprocess data
        print("\nStep 2: Preprocessing and feature engineering...")

        preprocessor = _preprocess_xlsx(
            data=df,
            target_col="LF98TRUU_Index_OAS",
            momentum_list=["LF98TRUU_Index_OAS", "LUACTRUU_Index_OAS"],
            momentum_X_days=[5, 10, 15],
            momentum_Y_days=30,
            forecast_list=[1, 3, 7, 15, 30],
            split_percentage=0.20,
            sequential=False
        )

        print(f"✓ Created {preprocessor.X_train.shape[1]} features")

        # Step 3: Train model
        print("\nStep 3: Training XGBoost model...")

        model = _build_model(
            preprocessor=preprocessor,
            model_name="XGBoost",
            estimators=500  # Reduced for demo
        )

        print("✓ Model trained successfully")

        # Step 4: Generate predictions
        print("\nStep 4: Generating predictions...")

        predictions = model._return_final_data()

        print("\nPredictions (last 5 rows):")
        print(predictions.tail())

        # Step 5: Analyze feature importance
        print("\nStep 5: Top 5 Most Important Features:")

        importance_df = pd.DataFrame({
            'Feature': model.X_train.columns,
            'Importance': model.model.feature_importances_
        }).sort_values('Importance', ascending=False)

        print(importance_df.head())

        print("\n" + "=" * 80)
        print("Complete Pipeline Success!")
        print("=" * 80)
        print("\nWorkflow summary:")
        print("1. ✓ Fetched data from Bloomberg Terminal API")
        print("2. ✓ Engineered momentum features")
        print("3. ✓ Trained XGBoost model")
        print("4. ✓ Generated multi-horizon predictions")
        print("5. ✓ Analyzed feature importance")

        return predictions

    except Exception as e:
        print(f"\n✗ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return None


def example_6_custom_configuration():
    """
    Example 6: Custom configuration options.

    Shows all available configuration options for fine-tuning
    Bloomberg API behavior.
    """
    print("\n" + "=" * 80)
    print("Example 6: Custom Configuration Options")
    print("=" * 80)

    print("\nAll available configuration options:")

    print("""
    source = BloombergAPIDataSource(
        # Required parameters
        securities=["LF98TRUU Index", ...],
        fields=["OAS", "PX_LAST", ...],
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2024, 12, 31),

        # Connection settings
        host="localhost",          # Bloomberg API host
        port=8194,                 # Bloomberg API port (default)

        # Performance tuning
        timeout=30000,             # Request timeout (30 seconds)
        max_retries=3,             # Number of retry attempts
        batch_size=100,            # Securities per request
    )
    """)

    print("\nSupported Bloomberg fields:")
    print("Financial Fields:")
    print("  - OAS: Option-Adjusted Spread")
    print("  - PX_LAST: Last Price")
    print("  - PX_OPEN/HIGH/LOW: OHLC Prices")
    print("  - PX_VOLUME: Volume")
    print("  - YLD_YTM_MID: Yield to Maturity")
    print("  - DTS: Duration to Worst")
    print("  - AMOUNT_OUTSTANDING: Outstanding Amount")
    print("\nRating Fields:")
    print("  - RTG_MOODY: Moody's Rating")
    print("  - RTG_SP: S&P Rating")


def main():
    """Main execution function."""
    print("=" * 80)
    print("BBG Credit Momentum - Bloomberg API Examples")
    print("=" * 80)
    print("\nThis script demonstrates various Bloomberg API usage patterns.")
    print("\nIMPORTANT: Requires Bloomberg Terminal running with DAPI configured.")

    # Run examples
    examples = [
        ("Basic Connection", example_1_basic_connection),
        ("Multiple Securities & Fields", example_2_multiple_securities_fields),
        ("Batch Processing", example_3_batch_processing),
        ("Retry Logic", example_4_retry_logic),
        ("Integration with Analysis", example_5_integration_with_analysis),
        ("Custom Configuration", example_6_custom_configuration),
    ]

    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"{i}. {name}")

    print("\nRunning all examples...")

    results = {}
    for name, example_func in examples:
        try:
            result = example_func()
            results[name] = result
        except KeyboardInterrupt:
            print("\n\nExecution interrupted by user.")
            break
        except Exception as e:
            print(f"\n✗ Example '{name}' failed: {e}")
            results[name] = None

    # Summary
    print("\n" + "=" * 80)
    print("Examples Summary")
    print("=" * 80)

    success_count = sum(1 for v in results.values() if v is not None)
    total_count = len(results)

    print(f"\nCompleted: {success_count}/{total_count} examples")

    for name, result in results.items():
        status = "✓ Success" if result is not None else "✗ Failed"
        print(f"{status}: {name}")

    print("\n" + "=" * 80)
    print("For more information:")
    print("- Bloomberg API documentation: docs/BLOOMBERG_INTEGRATION.md")
    print("- Configuration reference: config.example.yaml")
    print("- Troubleshooting: docs/BLOOMBERG_INTEGRATION.md#troubleshooting")
    print("=" * 80)


if __name__ == "__main__":
    main()
