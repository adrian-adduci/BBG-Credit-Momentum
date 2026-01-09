"""
Bloomberg Excel Usage Examples

This script demonstrates how to use Bloomberg Excel exports with the
BBG-Credit-Momentum system.

Examples covered:
1. Loading Bloomberg Excel exports
2. Handling Bloomberg error values (#N/A, #VALUE!)
3. Schema validation
4. Multi-sheet workbooks
5. Integration with analysis pipeline
6. Creating Excel templates

Prerequisites:
- Microsoft Excel with Bloomberg Excel Add-in
- OR pre-generated Excel files from Bloomberg Terminal

Usage:
    python examples/bloomberg_excel_example.py
"""

import sys
sys.path.insert(0, '..')

from datetime import datetime
import pandas as pd
import os

from _data_sources import BloombergExcelDataSource, ExcelDataSource
from _preprocessing import _preprocess_xlsx
from _models import _build_model


def example_1_basic_excel_loading():
    """
    Example 1: Basic Bloomberg Excel file loading.

    This example shows how to load a Bloomberg Excel export file.
    The file should be generated using Bloomberg Excel formulas (BDH).
    """
    print("=" * 80)
    print("Example 1: Basic Bloomberg Excel Loading")
    print("=" * 80)

    # Check if example file exists
    excel_file = "data/Economic_Data_2020_08_01.xlsx"

    if not os.path.exists(excel_file):
        print(f"\n✗ Excel file not found: {excel_file}")
        print("\nTo create a Bloomberg Excel export:")
        print("1. Open Excel with Bloomberg Add-in")
        print("2. Use =BDH() formula to fetch historical data")
        print("3. Save as .xlsx file")
        print("\nSee example below for formula structure.")
        return None

    try:
        print(f"\nLoading Excel file: {excel_file}")

        # Create Bloomberg Excel data source
        source = BloombergExcelDataSource(
            file_path=excel_file,
            sheet_name=0,           # First sheet
            date_column="Dates",    # Column with dates
            handle_errors=True      # Convert #N/A to NaN
        )

        # Load data
        df = source.load_data()

        print(f"\n✓ Successfully loaded {len(df)} rows")
        print(f"✓ Columns: {list(df.columns)}")

        # Display sample data
        print("\nFirst 5 rows:")
        print(df.head())

        # Check for any NaN values (converted from Bloomberg errors)
        nan_counts = df.isna().sum()
        if nan_counts.sum() > 0:
            print("\nColumns with missing values:")
            for col, count in nan_counts.items():
                if count > 0 and col != "Dates":
                    print(f"  - {col}: {count} missing ({count/len(df)*100:.1f}%)")

        return df

    except Exception as e:
        print(f"\n✗ Error loading Excel: {e}")
        import traceback
        traceback.print_exc()
        return None


def example_2_error_handling():
    """
    Example 2: Handling Bloomberg error values.

    Bloomberg Excel formulas can return error values like:
    - #N/A N/A (field not available)
    - #VALUE! (calculation error)
    - #N/A Field Not Applicable

    This example shows how these are automatically handled.
    """
    print("\n" + "=" * 80)
    print("Example 2: Bloomberg Error Value Handling")
    print("=" * 80)

    print("\nBloomberg Excel error values that are automatically handled:")

    error_values = [
        "#N/A N/A",
        "#N/A Field Not Applicable",
        "#N/A Invalid Security",
        "#VALUE!",
        "#REF!",
        "#NAME?",
        "#NUM!",
        "#DIV/0!",
    ]

    for error in error_values:
        print(f"  - {error} → NaN")

    print("\nCreating sample Excel with errors...")

    # Create sample DataFrame with Bloomberg errors
    sample_data = {
        "Dates": pd.date_range("2024-01-01", periods=5),
        "LF98TRUU_Index_OAS": [100.0, 105.0, "#N/A N/A", 110.0, 115.0],
        "LUACTRUU_Index_OAS": [200.0, "#VALUE!", 210.0, 215.0, "#N/A Field Not Applicable"]
    }

    df_with_errors = pd.DataFrame(sample_data)

    # Save to temporary file
    temp_file = "temp_bloomberg_errors.xlsx"
    df_with_errors.to_excel(temp_file, index=False)

    print(f"\nSaved sample file to: {temp_file}")
    print("\nOriginal data (with Bloomberg errors):")
    print(df_with_errors)

    # Load with error handling
    source = BloombergExcelDataSource(
        file_path=temp_file,
        handle_errors=True
    )

    df_cleaned = source.load_data()

    print("\nAfter loading with BloombergExcelDataSource:")
    print(df_cleaned)

    print("\nError conversion summary:")
    print(f"  - Errors converted to NaN: 3")
    print(f"  - Valid data points preserved: 7")

    # Cleanup
    os.remove(temp_file)
    print(f"\nCleaned up temporary file: {temp_file}")

    return df_cleaned


def example_3_schema_validation():
    """
    Example 3: Schema validation for Bloomberg Excel files.

    The BloombergExcelDataSource validates:
    - Dates column exists
    - Dates in ascending order
    - No duplicate dates
    - At least one data column
    """
    print("\n" + "=" * 80)
    print("Example 3: Schema Validation")
    print("=" * 80)

    print("\nSchema validation checks:")
    print("1. ✓ 'Dates' column exists")
    print("2. ✓ Dates are in ascending order")
    print("3. ✓ No duplicate dates")
    print("4. ✓ At least one data column present")
    print("5. ⚠ Warning for columns with all NaN values")

    # Example 1: Valid schema
    print("\n--- Test 1: Valid Schema ---")
    valid_data = {
        "Dates": pd.date_range("2024-01-01", periods=5),
        "LF98TRUU_Index_OAS": [100, 105, 110, 115, 120]
    }
    df_valid = pd.DataFrame(valid_data)

    temp_file = "temp_valid.xlsx"
    df_valid.to_excel(temp_file, index=False)

    try:
        source = BloombergExcelDataSource(file_path=temp_file)
        df = source.load_data()
        print("✓ Schema validation passed")
    except Exception as e:
        print(f"✗ Validation failed: {e}")
    finally:
        os.remove(temp_file)

    # Example 2: Duplicate dates (should fail)
    print("\n--- Test 2: Duplicate Dates (Should Fail) ---")
    invalid_data = {
        "Dates": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-03"]),
        "LF98TRUU_Index_OAS": [100, 105, 110]
    }
    df_invalid = pd.DataFrame(invalid_data)

    temp_file = "temp_invalid.xlsx"
    df_invalid.to_excel(temp_file, index=False)

    try:
        source = BloombergExcelDataSource(file_path=temp_file)
        df = source.load_data()
        print("✗ Should have failed validation")
    except ValueError as e:
        print(f"✓ Validation correctly failed: {e}")
    finally:
        os.remove(temp_file)


def example_4_creating_excel_template():
    """
    Example 4: How to create a Bloomberg Excel template.

    This example shows the recommended structure for Bloomberg
    Excel exports and provides formulas to use.
    """
    print("\n" + "=" * 80)
    print("Example 4: Creating Bloomberg Excel Template")
    print("=" * 80)

    print("\nRecommended Excel structure:")
    print("""
    Sheet 1: Timeseries Data
    ------------------------
    | Dates      | LF98TRUU_Index_OAS | LUACTRUU_Index_OAS | ...
    |------------|--------------------|--------------------|-----
    | 2024-01-01 | 123.45             | 234.56             | ...
    | 2024-01-02 | 124.56             | 235.67             | ...
    | ...        | ...                | ...                | ...

    Sheet 2 (optional): Security Metadata
    --------------------------------------
    | Ticker           | Display_Name              | Asset_Class
    |------------------|---------------------------|-------------
    | LF98TRUU Index   | US Aggregate Bond Index   | Credit
    | LUACTRUU Index   | US Corporate Bond Index   | Credit
    """)

    print("\nBloomberg Excel formulas:")

    print("""
    For historical data (BDH - Bloomberg Data History):
    ---------------------------------------------------
    Cell A1: "Dates"
    Cell B1: "LF98TRUU_Index_OAS"

    Cell A2: [Start date, e.g., 2024-01-01]
    Cell B2: =BDH("LF98TRUU Index", "OAS", $A$2, $A$100, "Dir=V")

    Formula breakdown:
    - "LF98TRUU Index": Security identifier
    - "OAS": Field name
    - $A$2: Start date (first date in column A)
    - $A$100: End date (last date in column A)
    - "Dir=V": Direction vertical (dates down rows)

    For current data (BDP - Bloomberg Data Point):
    -----------------------------------------------
    =BDP("LF98TRUU Index", "OAS")

    This returns the current value (not historical).
    """)

    # Create an example template
    print("\nCreating example template file...")

    # Create template structure
    template_data = {
        "Dates": pd.date_range("2024-01-01", periods=10),
        "LF98TRUU_Index_OAS": [None] * 10,  # To be filled by Bloomberg
        "LUACTRUU_Index_OAS": [None] * 10,
    }

    df_template = pd.DataFrame(template_data)

    template_file = "templates/bloomberg_credit_template.xlsx"
    os.makedirs("templates", exist_ok=True)

    df_template.to_excel(template_file, index=False)

    print(f"✓ Created template: {template_file}")
    print("\nNext steps:")
    print("1. Open template in Excel with Bloomberg Add-in")
    print("2. Replace column formulas with Bloomberg formulas (see above)")
    print("3. Wait for Bloomberg to populate data")
    print("4. Save and use with BloombergExcelDataSource")


def example_5_integration_with_analysis():
    """
    Example 5: End-to-end Excel to analysis pipeline.

    Shows complete workflow from Excel file to trained model.
    """
    print("\n" + "=" * 80)
    print("Example 5: Excel to Analysis Pipeline")
    print("=" * 80)

    excel_file = "data/Economic_Data_2020_08_01.xlsx"

    if not os.path.exists(excel_file):
        print(f"\n✗ Excel file not found: {excel_file}")
        print("Run Example 1 first or provide an Excel file.")
        return None

    try:
        # Step 1: Load Bloomberg Excel data
        print("\nStep 1: Loading Bloomberg Excel file...")

        source = BloombergExcelDataSource(
            file_path=excel_file,
            handle_errors=True
        )

        df = source.load_data()
        print(f"✓ Loaded {len(df)} rows, {len(df.columns)-1} columns")

        # Step 2: Preprocess
        print("\nStep 2: Preprocessing...")

        # Find suitable columns for momentum
        data_cols = [c for c in df.columns if c != "Dates" and "_OAS" in c]

        if len(data_cols) == 0:
            print("✗ No OAS columns found in Excel file")
            return None

        target_col = data_cols[0]
        momentum_cols = data_cols[:min(2, len(data_cols))]

        print(f"Target: {target_col}")
        print(f"Momentum columns: {momentum_cols}")

        preprocessor = _preprocess_xlsx(
            data=df,
            target_col=target_col,
            momentum_list=momentum_cols,
            momentum_X_days=[5, 10, 15],
            momentum_Y_days=30,
            forecast_list=[1, 3, 7, 15, 30],
            split_percentage=0.20
        )

        print(f"✓ Created {preprocessor.X_train.shape[1]} features")

        # Step 3: Train model
        print("\nStep 3: Training model...")

        model = _build_model(
            preprocessor=preprocessor,
            model_name="XGBoost",
            estimators=500
        )

        print("✓ Model trained")

        # Step 4: Results
        print("\nStep 4: Results...")

        predictions = model._return_final_data()

        print("\nPredictions (last 5 rows):")
        print(predictions[["Dates", target_col]].tail())

        # Feature importance
        importance_df = pd.DataFrame({
            'Feature': model.X_train.columns,
            'Importance': model.model.feature_importances_
        }).sort_values('Importance', ascending=False)

        print("\nTop 5 Features:")
        print(importance_df.head())

        print("\n" + "=" * 80)
        print("Pipeline Complete!")
        print("=" * 80)
        print("\nYou can now:")
        print("- Export predictions to Excel: predictions.to_excel('output.xlsx')")
        print("- Visualize results in Streamlit UI")
        print("- Use predictions for trading decisions")

        return predictions

    except Exception as e:
        print(f"\n✗ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return None


def example_6_hybrid_comparison():
    """
    Example 6: Comparing Excel vs API data (if both available).

    Shows how to verify that Excel export matches API data.
    """
    print("\n" + "=" * 80)
    print("Example 6: Excel vs API Data Comparison")
    print("=" * 80)

    print("\nThis example would compare:")
    print("- Data from Bloomberg Excel export")
    print("- Same data from Bloomberg API")
    print("\nTo verify they match and identify any discrepancies.")

    print("\nPseudo-code:")
    print("""
    # Load from Excel
    excel_source = BloombergExcelDataSource("data.xlsx")
    df_excel = excel_source.load_data()

    # Load from API
    api_source = BloombergAPIDataSource(
        securities=["LF98TRUU Index"],
        fields=["OAS"],
        start_date=df_excel['Dates'].min(),
        end_date=df_excel['Dates'].max()
    )
    df_api = api_source.load_data()

    # Compare
    merged = df_excel.merge(df_api, on="Dates", suffixes=("_excel", "_api"))
    diff = merged["LF98TRUU_Index_OAS_excel"] - merged["LF98TRUU_Index_OAS_api"]

    print(f"Mean difference: {diff.mean():.6f}")
    print(f"Max difference: {diff.abs().max():.6f}")
    """)


def main():
    """Main execution function."""
    print("=" * 80)
    print("BBG Credit Momentum - Bloomberg Excel Examples")
    print("=" * 80)
    print("\nThis script demonstrates Bloomberg Excel export usage patterns.")

    # Run examples
    examples = [
        ("Basic Excel Loading", example_1_basic_excel_loading),
        ("Error Value Handling", example_2_error_handling),
        ("Schema Validation", example_3_schema_validation),
        ("Creating Excel Template", example_4_creating_excel_template),
        ("Integration with Analysis", example_5_integration_with_analysis),
        ("Excel vs API Comparison", example_6_hybrid_comparison),
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
    print("For more information:")
    print("- Bloomberg Excel guide: docs/BLOOMBERG_INTEGRATION.md#bloomberg-excel-export")
    print("- Excel template: templates/bloomberg_credit_template.xlsx")
    print("- Configuration: config.example.yaml")
    print("=" * 80)


if __name__ == "__main__":
    main()
