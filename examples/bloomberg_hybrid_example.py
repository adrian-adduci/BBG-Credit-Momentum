"""
Bloomberg Hybrid Mode Examples

This script demonstrates the hybrid Bloomberg data source that automatically
falls back between API and Excel for maximum reliability.

Examples covered:
1. Basic hybrid mode usage
2. API-first with Excel fallback
3. Excel-first with API fallback
4. Handling Terminal unavailability
5. Production deployment patterns

Prerequisites:
- Bloomberg Terminal (optional - will fall back to Excel)
- Excel file with Bloomberg data (optional - will try API first)

Usage:
    python examples/bloomberg_hybrid_example.py
"""

import sys
sys.path.insert(0, '..')

from datetime import datetime
import pandas as pd
import os

from _data_sources import (
    HybridBloombergDataSource,
    BloombergTerminalNotRunning
)
from _preprocessing import _preprocess_xlsx
from _models import _build_model


def example_1_basic_hybrid():
    """
    Example 1: Basic hybrid mode usage.

    Hybrid mode tries Bloomberg API first, then falls back to Excel
    if the Terminal is not available.
    """
    print("=" * 80)
    print("Example 1: Basic Hybrid Mode")
    print("=" * 80)

    print("\nHybrid mode workflow:")
    print("1. Try Bloomberg Terminal API")
    print("2. If Terminal not available → Fall back to Excel")
    print("3. If both fail → Raise error with details")

    try:
        # Define securities and fields
        securities = ["LF98TRUU Index"]
        fields = ["OAS"]
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 3, 31)

        # Excel fallback file
        excel_fallback = "data/Economic_Data_2020_08_01.xlsx"

        print(f"\nConfiguration:")
        print(f"  Securities: {securities}")
        print(f"  Fields: {fields}")
        print(f"  Date range: {start_date.date()} to {end_date.date()}")
        print(f"  Excel fallback: {excel_fallback}")

        # Create hybrid source
        source = HybridBloombergDataSource(
            securities=securities,
            fields=fields,
            start_date=start_date,
            end_date=end_date,
            excel_fallback_path=excel_fallback,
            prefer_api=True  # Try API first
        )

        # Load data (will use whichever source is available)
        print("\nAttempting to load data...")
        print("(Watch for automatic fallback if Terminal not available)")

        df = source.load_data()

        print(f"\n✓ Successfully loaded {len(df)} rows from available source")
        print(f"✓ Columns: {list(df.columns)}")

        print("\nFirst 5 rows:")
        print(df.head())

        return df

    except Exception as e:
        print(f"\n✗ Both sources failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def example_2_api_first_mode():
    """
    Example 2: API-first mode (default).

    Prioritizes real-time Bloomberg API data, falls back to Excel
    only if API unavailable.
    """
    print("\n" + "=" * 80)
    print("Example 2: API-First Mode (Default)")
    print("=" * 80)

    print("\nAPI-first mode:")
    print("✓ Use cases: Real-time analysis, automated trading, fresh data")
    print("✓ Advantage: Always get latest data when Terminal available")
    print("✓ Fallback: Excel file if Terminal down/busy")

    try:
        source = HybridBloombergDataSource(
            securities=["LF98TRUU Index", "LUACTRUU Index"],
            fields=["OAS", "PX_LAST"],
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 3, 31),
            excel_fallback_path="data/Economic_Data_2020_08_01.xlsx",
            prefer_api=True  # Explicit API-first (default)
        )

        print("\nLoading data (API-first)...")
        df = source.load_data()

        print(f"\n✓ Loaded {len(df)} rows")

        # Check data freshness
        latest_date = df['Dates'].max()
        print(f"✓ Latest data point: {latest_date.date()}")

        return df

    except Exception as e:
        print(f"\n✗ Error: {e}")
        return None


def example_3_excel_first_mode():
    """
    Example 3: Excel-first mode.

    Prioritizes Excel file (faster, offline), falls back to API
    only if Excel not available or corrupted.
    """
    print("\n" + "=" * 80)
    print("Example 3: Excel-First Mode")
    print("=" * 80)

    print("\nExcel-first mode:")
    print("✓ Use cases: Offline analysis, stable data snapshots, performance")
    print("✓ Advantage: Faster loading, no API rate limits, works offline")
    print("✓ Fallback: Bloomberg API if Excel missing/corrupted")

    try:
        source = HybridBloombergDataSource(
            securities=["LF98TRUU Index"],
            fields=["OAS"],
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
            excel_fallback_path="data/Economic_Data_2020_08_01.xlsx",
            prefer_api=False  # Excel-first mode
        )

        print("\nLoading data (Excel-first)...")
        df = source.load_data()

        print(f"\n✓ Loaded {len(df)} rows from Excel (or API fallback)")

        return df

    except Exception as e:
        print(f"\n✗ Error: {e}")
        return None


def example_4_terminal_unavailable_simulation():
    """
    Example 4: Handling Terminal unavailability.

    Simulates what happens when Bloomberg Terminal is not running.
    """
    print("\n" + "=" * 80)
    print("Example 4: Terminal Unavailability Handling")
    print("=" * 80)

    print("\nScenarios where API fails:")
    print("1. Bloomberg Terminal not running")
    print("2. Terminal is busy/unresponsive")
    print("3. Network issues between API and Terminal")
    print("4. DAPI not configured")

    print("\nIn all cases, hybrid mode will:")
    print("- Log a warning about API failure")
    print("- Automatically try Excel fallback")
    print("- Only raise error if both sources fail")

    # This example uses a non-existent Excel to show both failures
    print("\nSimulating both sources unavailable...")

    try:
        source = HybridBloombergDataSource(
            securities=["LF98TRUU Index"],
            fields=["OAS"],
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 3, 31),
            excel_fallback_path="nonexistent_file.xlsx",  # Intentionally wrong
            prefer_api=True
        )

        df = source.load_data()

        print("✗ Should have failed (both sources unavailable)")

    except Exception as e:
        print(f"\n✓ Correctly handled unavailability:")
        print(f"   Error: {str(e)[:100]}...")
        print("\nError message includes:")
        print("- Which source failed first (API)")
        print("- Why it failed")
        print("- That fallback was attempted")
        print("- Why fallback failed")


def example_5_production_deployment():
    """
    Example 5: Production deployment patterns.

    Shows recommended patterns for production use with monitoring
    and alerting.
    """
    print("\n" + "=" * 80)
    print("Example 5: Production Deployment Patterns")
    print("=" * 80)

    print("\nRecommended production setup:")

    print("""
    # 1. Configuration
    config.yaml:
    -----------
    data_source:
      type: "bloomberg_hybrid"
      bloomberg:
        securities: ["LF98TRUU Index", "LUACTRUU Index"]
        fields: ["OAS", "PX_LAST"]
        start_date: "2020-01-01"
        end_date: "2024-12-31"
        excel_fallback: "/data/bloomberg_export.xlsx"
        prefer_api: true

    # 2. Scheduled Excel Updates
    # Cron job to update Excel fallback weekly
    0 0 * * 0 /scripts/update_bloomberg_excel.sh

    # 3. Monitoring
    import logging
    logger = logging.getLogger("bloomberg_hybrid")

    def load_bloomberg_data_with_monitoring():
        source = HybridBloombergDataSource(...)

        try:
            df = source.load_data()
            logger.info(f"✓ Loaded {len(df)} rows")

            # Alert if using fallback
            if "Excel" in str(df.attrs.get("source", "")):
                logger.warning("Using Excel fallback - Bloomberg API unavailable")
                # Send alert to monitoring system

            return df

        except Exception as e:
            logger.error(f"✗ Both Bloomberg sources failed: {e}")
            # Send critical alert
            raise

    # 4. Health Check Endpoint
    @app.get("/health/bloomberg")
    def bloomberg_health():
        try:
            source = HybridBloombergDataSource(
                securities=["LF98TRUU Index"],
                fields=["OAS"],
                start_date=datetime.now() - timedelta(days=7),
                end_date=datetime.now(),
                excel_fallback_path=config.excel_fallback
            )
            df = source.load_data()
            return {"status": "ok", "rows": len(df)}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # 5. Graceful Degradation
    def get_data_with_degradation():
        # Primary: Real-time Bloomberg
        try:
            df = hybrid_source.load_data()
            return df, "realtime"
        except Exception:
            # Fallback 1: Cached data
            if cache_exists():
                return load_from_cache(), "cached"
            # Fallback 2: Historical data
            else:
                return load_historical(), "historical"
    """)

    print("\nProduction checklist:")
    print("✓ Hybrid mode enabled")
    print("✓ Excel fallback updated regularly (weekly/daily)")
    print("✓ Monitoring for source failures")
    print("✓ Alerts configured")
    print("✓ Health check endpoint")
    print("✓ Graceful degradation strategy")


def example_6_integration_comparison():
    """
    Example 6: Comparing data from both sources.

    For validation, you can load from both sources and compare
    to ensure consistency.
    """
    print("\n" + "=" * 80)
    print("Example 6: Data Source Comparison")
    print("=" * 80)

    print("\nValidation workflow:")
    print("1. Load data via hybrid mode (gets from one source)")
    print("2. Force load from other source")
    print("3. Compare results")
    print("4. Alert if significant differences")

    print("\nPseudo-code for validation:")
    print("""
    # Load via hybrid (automatic source selection)
    hybrid_source = HybridBloombergDataSource(...)
    df_hybrid = hybrid_source.load_data()

    # Force load from API
    api_source = BloombergAPIDataSource(...)
    df_api = api_source.load_data()

    # Force load from Excel
    excel_source = BloombergExcelDataSource(...)
    df_excel = excel_source.load_data()

    # Compare
    def compare_sources(df1, df2, name1, name2):
        merged = df1.merge(df2, on="Dates", suffixes=(f"_{name1}", f"_{name2}"))

        for col in df1.columns:
            if col != "Dates":
                col1 = f"{col}_{name1}"
                col2 = f"{col}_{name2}"

                diff = (merged[col1] - merged[col2]).abs()
                max_diff = diff.max()
                mean_diff = diff.mean()

                print(f"{col}:")
                print(f"  Max diff: {max_diff:.6f}")
                print(f"  Mean diff: {mean_diff:.6f}")

                if max_diff > 0.01:  # Threshold
                    logger.warning(f"Significant difference in {col}")

    compare_sources(df_api, df_excel, "api", "excel")
    """)


def example_7_complete_pipeline():
    """
    Example 7: Complete production pipeline with hybrid mode.

    End-to-end example showing hybrid mode in a full analysis pipeline.
    """
    print("\n" + "=" * 80)
    print("Example 7: Complete Production Pipeline")
    print("=" * 80)

    try:
        # Step 1: Load data with hybrid mode
        print("\nStep 1: Loading Bloomberg data (hybrid mode)...")

        source = HybridBloombergDataSource(
            securities=["LF98TRUU Index", "LUACTRUU Index"],
            fields=["OAS"],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2024, 12, 31),
            excel_fallback_path="data/Economic_Data_2020_08_01.xlsx",
            prefer_api=True
        )

        df = source.load_data()
        print(f"✓ Loaded {len(df)} rows")

        # Step 2: Validate data quality
        print("\nStep 2: Validating data quality...")

        # Check for sufficient data
        if len(df) < 100:
            print("⚠ Warning: Less than 100 data points")

        # Check for missing values
        missing_pct = df.isna().sum().sum() / (len(df) * len(df.columns)) * 100
        print(f"Missing data: {missing_pct:.2f}%")

        if missing_pct > 10:
            print("⚠ Warning: High percentage of missing data")

        # Step 3: Preprocess
        print("\nStep 3: Preprocessing...")

        preprocessor = _preprocess_xlsx(
            data=df,
            target_col="LF98TRUU_Index_OAS",
            momentum_list=["LF98TRUU_Index_OAS", "LUACTRUU_Index_OAS"],
            momentum_X_days=[5, 10, 15],
            momentum_Y_days=30,
            forecast_list=[1, 3, 7, 15, 30],
            split_percentage=0.20
        )

        print(f"✓ Features: {preprocessor.X_train.shape[1]}")

        # Step 4: Train model
        print("\nStep 4: Training model...")

        model = _build_model(
            preprocessor=preprocessor,
            model_name="XGBoost",
            estimators=500
        )

        print("✓ Model trained")

        # Step 5: Generate predictions
        print("\nStep 5: Generating predictions...")

        predictions = model._return_final_data()

        print("\nLatest predictions:")
        print(predictions[["Dates", "LF98TRUU_Index_OAS"]].tail())

        # Step 6: Save results
        print("\nStep 6: Saving results...")

        output_file = "predictions_hybrid.csv"
        predictions.to_csv(output_file, index=False)
        print(f"✓ Saved to: {output_file}")

        print("\n" + "=" * 80)
        print("Production Pipeline Complete!")
        print("=" * 80)
        print("\nDeployment ready:")
        print("✓ Hybrid mode ensures reliability")
        print("✓ Data quality validated")
        print("✓ Model trained and predictions generated")
        print("✓ Results saved for downstream use")

        return predictions

    except Exception as e:
        print(f"\n✗ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main execution function."""
    print("=" * 80)
    print("BBG Credit Momentum - Bloomberg Hybrid Mode Examples")
    print("=" * 80)
    print("\nHybrid mode provides maximum reliability by automatically")
    print("falling back between Bloomberg API and Excel exports.")

    # Run examples
    examples = [
        ("Basic Hybrid Mode", example_1_basic_hybrid),
        ("API-First Mode", example_2_api_first_mode),
        ("Excel-First Mode", example_3_excel_first_mode),
        ("Terminal Unavailability", example_4_terminal_unavailable_simulation),
        ("Production Deployment", example_5_production_deployment),
        ("Data Source Comparison", example_6_integration_comparison),
        ("Complete Pipeline", example_7_complete_pipeline),
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
        status = "✓ Success" if result is not None else "✗ Failed/Skipped"
        print(f"{status}: {name}")

    print("\n" + "=" * 80)
    print("Recommendations:")
    print("=" * 80)
    print("\n1. For Production: Use hybrid mode (prefer_api=True)")
    print("   - Gets latest data when possible")
    print("   - Falls back gracefully when Terminal unavailable")
    print("\n2. For Development: Use Excel-first mode (prefer_api=False)")
    print("   - Faster iteration")
    print("   - No API rate limits")
    print("\n3. Always keep Excel fallback updated")
    print("   - Weekly updates recommended")
    print("   - Automated via cron/scheduled task")

    print("\n" + "=" * 80)
    print("For more information:")
    print("- Hybrid mode documentation: docs/BLOOMBERG_INTEGRATION.md#hybrid-mode-recommended")
    print("- Configuration: config.example.yaml")
    print("=" * 80)


if __name__ == "__main__":
    main()
