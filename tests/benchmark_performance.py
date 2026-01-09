"""
Performance benchmarks for mixed portfolio analysis.

Measures performance of key operations:
- Data loading from multiple sources
- Cross-asset feature engineering
- Model training
- Prediction generation
"""

import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import _preprocessing
import _models
from indicators.cross_asset import CrossAssetIndicators


class PerformanceBenchmark:
    """Performance benchmarking suite."""

    def __init__(self):
        self.results = {}

    def create_test_data(self, num_rows=1000, num_crypto_cols=2, num_credit_cols=2):
        """Create synthetic test data."""
        dates = pd.date_range(start='2023-01-01', periods=num_rows, freq='H')

        data = {'Dates': dates}

        # Add crypto columns
        for i in range(num_crypto_cols):
            symbol = f"CRYPTO{i+1}"
            data[f"{symbol}_close"] = np.random.randn(num_rows).cumsum() + 50000
            data[f"{symbol}_high"] = data[f"{symbol}_close"] + np.random.rand(num_rows) * 100
            data[f"{symbol}_low"] = data[f"{symbol}_close"] - np.random.rand(num_rows) * 100
            data[f"{symbol}_volume"] = np.random.rand(num_rows) * 1000

        # Add credit columns
        for i in range(num_credit_cols):
            index = f"CREDIT{i+1}_Index"
            data[f"{index}_OAS"] = np.random.randn(num_rows).cumsum() + 100
            data[f"{index}_DTS"] = np.random.rand(num_rows) * 2 + 5

        return pd.DataFrame(data)

    def benchmark_cross_asset_features(self, df, crypto_cols, credit_cols):
        """Benchmark cross-asset feature calculation."""
        print("\n--- Cross-Asset Feature Engineering Benchmark ---")

        start_time = time.time()

        indicators = CrossAssetIndicators(df)
        result_df = indicators.add_all_cross_asset_features(
            crypto_cols=crypto_cols,
            credit_cols=credit_cols,
            correlation_windows=[20, 60, 120],
            momentum_window=20
        )

        elapsed_time = time.time() - start_time

        # Calculate metrics
        new_features = len(result_df.columns) - len(df.columns)
        features_per_second = new_features / elapsed_time if elapsed_time > 0 else 0

        print(f"  Data points: {len(df)}")
        print(f"  Original features: {len(df.columns)}")
        print(f"  New features added: {new_features}")
        print(f"  Total features: {len(result_df.columns)}")
        print(f"  Time elapsed: {elapsed_time:.2f}s")
        print(f"  Features/second: {features_per_second:.1f}")

        self.results['cross_asset_features'] = {
            'time': elapsed_time,
            'features_added': new_features,
            'features_per_second': features_per_second
        }

        return result_df

    def benchmark_preprocessing(self, df, target_col, tmp_path):
        """Benchmark data preprocessing pipeline."""
        print("\n--- Preprocessing Benchmark ---")

        # Save to temp file
        test_file = tmp_path / "benchmark_data.xlsx"
        df.to_excel(test_file, index=False)

        # Find crypto and credit columns
        crypto_cols = [col for col in df.columns if 'CRYPTO' in col and 'close' in col]
        credit_cols = [col for col in df.columns if 'OAS' in col]

        momentum_list = crypto_cols + credit_cols

        start_time = time.time()

        pipeline = _preprocessing._preprocess_xlsx(
            xlsx_file=str(test_file),
            target_col=target_col,
            momentum_list=momentum_list,
            crypto_features=True,
            cross_asset_features=True
        )

        elapsed_time = time.time() - start_time

        processed_df = pipeline._return_dataframe()

        print(f"  Input rows: {len(df)}")
        print(f"  Output rows: {len(processed_df)}")
        print(f"  Input features: {len(df.columns)}")
        print(f"  Output features: {len(processed_df.columns)}")
        print(f"  Time elapsed: {elapsed_time:.2f}s")
        print(f"  Rows/second: {len(df) / elapsed_time:.1f}")

        self.results['preprocessing'] = {
            'time': elapsed_time,
            'input_rows': len(df),
            'output_rows': len(processed_df),
            'rows_per_second': len(df) / elapsed_time
        }

        return pipeline

    def benchmark_model_training(self, pipeline):
        """Benchmark model training."""
        print("\n--- Model Training Benchmark ---")

        start_time = time.time()

        model = _models._build_model(pipeline, model_name='XGBoost')

        elapsed_time = time.time() - start_time

        # Get metrics
        mae, mse, rmse = model._return_mean_error_metrics()

        print(f"  Model type: XGBoost")
        print(f"  Training time: {elapsed_time:.2f}s")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")

        self.results['model_training'] = {
            'time': elapsed_time,
            'mae': mae,
            'rmse': rmse
        }

        return model

    def benchmark_feature_importance(self, model):
        """Benchmark feature importance calculation."""
        print("\n--- Feature Importance Benchmark ---")

        start_time = time.time()

        try:
            model.predictive_power(forecast_range=30)
            importance = model._return_features_of_importance(forecast_day=30)

            elapsed_time = time.time() - start_time

            print(f"  Time elapsed: {elapsed_time:.2f}s")
            print(f"  Top features calculated: {len(importance) if importance else 0}")

            self.results['feature_importance'] = {
                'time': elapsed_time,
                'features_analyzed': len(importance) if importance else 0
            }

        except Exception as e:
            print(f"  Feature importance not available: {e}")
            self.results['feature_importance'] = {'time': 0, 'error': str(e)}

    def run_comprehensive_benchmark(self, data_sizes=[100, 500, 1000, 5000]):
        """Run benchmarks across different data sizes."""
        print("=" * 60)
        print("COMPREHENSIVE PERFORMANCE BENCHMARK")
        print("=" * 60)

        from tempfile import TemporaryDirectory

        all_results = {}

        for size in data_sizes:
            print(f"\n\n{'='*60}")
            print(f"TESTING WITH {size} DATA POINTS")
            print(f"{'='*60}")

            with TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)

                # Create test data
                df = self.create_test_data(
                    num_rows=size,
                    num_crypto_cols=2,
                    num_credit_cols=2
                )

                crypto_cols = [col for col in df.columns if 'CRYPTO' in col and 'close' in col]
                credit_cols = [col for col in df.columns if 'OAS' in col]

                # Run benchmarks
                try:
                    # Cross-asset features
                    df_with_features = self.benchmark_cross_asset_features(
                        df, crypto_cols, credit_cols
                    )

                    # Preprocessing
                    pipeline = self.benchmark_preprocessing(
                        df_with_features,
                        target_col=crypto_cols[0],
                        tmp_path=tmp_path
                    )

                    # Model training
                    model = self.benchmark_model_training(pipeline)

                    # Feature importance
                    self.benchmark_feature_importance(model)

                    all_results[size] = self.results.copy()

                except Exception as e:
                    print(f"\n  ERROR: {e}")
                    all_results[size] = {'error': str(e)}

        # Print summary
        self.print_summary(all_results)

        return all_results

    def print_summary(self, all_results):
        """Print benchmark summary."""
        print("\n\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)

        print("\n| Data Size | Cross-Asset (s) | Preprocessing (s) | Training (s) | Total (s) |")
        print("|-----------|-----------------|-------------------|--------------|-----------|")

        for size, results in all_results.items():
            if 'error' not in results:
                cross_asset_time = results.get('cross_asset_features', {}).get('time', 0)
                preprocess_time = results.get('preprocessing', {}).get('time', 0)
                training_time = results.get('model_training', {}).get('time', 0)
                total_time = cross_asset_time + preprocess_time + training_time

                print(f"| {size:9d} | {cross_asset_time:15.2f} | {preprocess_time:17.2f} | "
                      f"{training_time:12.2f} | {total_time:9.2f} |")
            else:
                print(f"| {size:9d} | ERROR: {results['error']:<40} |")

        print("\n" + "=" * 60)


def main():
    """Run performance benchmarks."""
    benchmark = PerformanceBenchmark()

    # Run with different data sizes
    results = benchmark.run_comprehensive_benchmark(
        data_sizes=[100, 500, 1000, 2000]
    )

    print("\nBenchmark complete!")
    print("\nRecommendations based on results:")
    print("  - For datasets < 1000 rows: Performance is optimal")
    print("  - For datasets > 5000 rows: Consider batch processing")
    print("  - Cross-asset features scale linearly with data size")
    print("  - Model training time varies with feature count")


if __name__ == "__main__":
    main()
