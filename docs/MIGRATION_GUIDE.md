# Migration Guide: Crypto Securities Integration

**Version:** 2.0
**Date:** 2026-01-07
**Status:** Complete

---

## Overview

This guide helps you migrate from the legacy credit-only system to the new unified crypto + credit securities framework. The migration preserves backward compatibility while adding powerful cross-asset analysis capabilities.

---

## Table of Contents

1. [Breaking Changes](#breaking-changes)
2. [Configuration Migration](#configuration-migration)
3. [Code Migration](#code-migration)
4. [Data Pipeline Migration](#data-pipeline-migration)
5. [API Migration](#api-migration)
6. [Testing Migration](#testing-migration)
7. [Troubleshooting](#troubleshooting)

---

## Breaking Changes

### None!

**The migration is fully backward compatible.** All existing code, configurations, and workflows continue to work without modification.

### What's New (Opt-In Features)

1. **New Data Sources**: Bloomberg API, Hybrid Bloomberg, Mixed Portfolio
2. **New Features**: Cross-asset indicators (correlations, regime detection, divergence)
3. **New API Endpoints**: `/api/mixed/train`, `/api/mixed/analysis/{model_id}`
4. **New Preprocessing Options**: `cross_asset_features=True`

---

## Configuration Migration

### Legacy Configuration (Still Works)

Your existing `config.yaml` for credit-only analysis continues to work:

```yaml
# config.yaml (Legacy - Still Supported)
data_source:
  type: excel
  file_path: data/Economic_Data_2020_08_01.xlsx

model:
  type: XGBoost
  target: LF98TRUU_Index_OAS
```

### New Unified Configuration

For mixed portfolios, use the new format:

```yaml
# config.unified.yaml (New Format)
securities:
  # Crypto securities
  - identifier: "BTC/USDT"
    type: "crypto_spot"
    source:
      type: "crypto_exchange"
      exchange: "binance"
      timeframe: "1h"
    fields: ["close", "volume"]

  # Credit securities
  - identifier: "LF98TRUU Index"
    type: "credit_index"
    source:
      type: "bloomberg"
      mode: "hybrid"  # API with Excel fallback
      excel_path: "data/bloomberg_export.xlsx"
    fields: ["OAS", "DTS"]

model:
  target: "BTC_USDT_close"
  features:
    crypto_features: true
    cross_asset_features: true  # NEW!
```

### Environment Variables (.env)

Add Bloomberg credentials to your `.env` file:

```bash
# Copy .env.example to .env
cp .env.example .env

# Edit .env and add:
BLOOMBERG_HOST=localhost
BLOOMBERG_PORT=8194
BLOOMBERG_TIMEOUT=30000
BLOOMBERG_MAX_RETRIES=3
```

---

## Code Migration

### 1. Data Loading

#### Before (Credit Only)

```python
from _data_sources import ExcelDataSource

# Load credit data only
source = ExcelDataSource(file_path="data/bloomberg_export.xlsx")
df = source.load_data()
```

#### After (Mixed Portfolio)

```python
from _data_sources import (
    MixedPortfolioDataSource,
    BloombergExcelDataSource,
    DataSourceFactory
)

# Load crypto data
crypto_source = DataSourceFactory.create(
    "crypto",
    exchange_id="binance",
    symbols=["BTC/USDT", "ETH/USDT"],
    timeframe="1h",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31)
)

# Load credit data
credit_source = BloombergExcelDataSource(
    file_path="data/bloomberg_export.xlsx"
)

# Combine both
mixed_source = MixedPortfolioDataSource(
    sources=[crypto_source, credit_source],
    alignment='outer',
    fill_method='ffill'
)

df = mixed_source.load_data()
```

### 2. Preprocessing

#### Before

```python
from _preprocessing import _preprocess_xlsx

pipeline = _preprocess_xlsx(
    xlsx_file="data/economic_data.xlsx",
    target_col="LF98TRUU_Index_OAS",
    momentum_list=["LF98TRUU_Index_OAS"],
)
```

#### After (With Cross-Asset Features)

```python
from _preprocessing import _preprocess_xlsx

pipeline = _preprocess_xlsx(
    xlsx_file=df,  # Can pass DataFrame directly
    target_col="BTC_USDT_close",
    momentum_list=[
        "BTC_USDT_close",
        "LF98TRUU_Index_OAS"
    ],
    crypto_features=True,        # Enable crypto indicators
    cross_asset_features=True    # Enable cross-asset features (NEW!)
)
```

### 3. Model Training

No changes required! Model training works the same:

```python
from _models import _build_model

# Same as before
model = _build_model(pipeline, model_name='XGBoost')
```

---

## Data Pipeline Migration

### Step-by-Step Migration

#### Step 1: Backup Your Current Setup

```bash
# Backup your data and config
cp -r data data_backup
cp config.yaml config.yaml.backup
```

#### Step 2: Install New Dependencies

```bash
# Update requirements
pip install -r requirements.txt

# Optional: Bloomberg API (requires Bloomberg Terminal)
# pip install --index-url=https://bcms.bloomberg.com/pip/simple blpapi
```

#### Step 3: Test Legacy Workflow

```bash
# Verify existing code still works
python -c "from _data_sources import ExcelDataSource; print('✓ Legacy imports work')"
```

#### Step 4: Add New Data Sources Gradually

Start by adding crypto data alongside your existing credit data:

```python
# test_mixed.py
from _data_sources import MixedPortfolioDataSource, ExcelDataSource, DataSourceFactory
from datetime import datetime

# Your existing credit source (no changes)
credit_source = ExcelDataSource(file_path="data/bloomberg_export.xlsx")

# Add crypto source
crypto_source = DataSourceFactory.create(
    "crypto",
    exchange_id="binance",
    symbols=["BTC/USDT"],
    timeframe="1d",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31)
)

# Combine
mixed_source = MixedPortfolioDataSource(
    sources=[credit_source, crypto_source],
    alignment='outer'
)

df = mixed_source.load_data()
print(f"✓ Mixed data loaded: {len(df)} rows, {len(df.columns)} columns")
```

#### Step 5: Enable Cross-Asset Features

```python
# In your preprocessing code, add one parameter:
pipeline = _preprocess_xlsx(
    xlsx_file=df,
    target_col="BTC_USDT_close",
    momentum_list=["BTC_USDT_close", "LF98TRUU_Index_OAS"],
    crypto_features=True,
    cross_asset_features=True  # ← ADD THIS
)
```

#### Step 6: Verify Cross-Asset Features

```python
# Check that cross-asset features were added
processed_df = pipeline._return_dataframe()

cross_asset_features = [
    col for col in processed_df.columns
    if any(keyword in col for keyword in ['corr_', 'regime_', 'divergence_', 'ftq_'])
]

print(f"✓ Cross-asset features added: {len(cross_asset_features)}")
print(f"  Examples: {cross_asset_features[:5]}")
```

---

## API Migration

### Legacy API Endpoints (Still Work)

```bash
# These endpoints are unchanged
POST /api/train
POST /api/predict
GET /api/strategies
POST /api/backtest
```

### New Mixed Portfolio Endpoints

```bash
# New endpoints for mixed portfolios
POST /api/mixed/train
GET /api/mixed/analysis/{model_id}
```

#### Example: Train Mixed Portfolio Model

```bash
curl -X POST "http://localhost:8000/api/mixed/train" \
  -H "Content-Type: application/json" \
  -d '{
    "crypto_exchange": "binance",
    "crypto_symbols": ["BTC/USDT", "ETH/USDT"],
    "bloomberg_securities": ["LF98TRUU Index"],
    "bloomberg_source": "excel",
    "bloomberg_excel_path": "data/bloomberg_export.xlsx",
    "start_date": "2024-01-01",
    "target_column": "BTC_USDT_close",
    "crypto_features": true,
    "cross_asset_features": true
  }'
```

#### Example: Get Cross-Asset Analysis

```bash
curl -X GET "http://localhost:8000/api/mixed/analysis/mixed_XGBoost_20260107_123456"
```

Response:
```json
{
  "success": true,
  "correlations": {
    "corr_BTC_USDT_close_LF98TRUU_Index_OAS_60": 0.45
  },
  "regime": "risk-on",
  "divergence_signals": {
    "divergence_signal_BTC_USDT_close_LF98TRUU_Index_OAS": false
  },
  "flight_to_quality": -0.23,
  "timestamp": "2026-01-07T12:34:56.789Z"
}
```

---

## Testing Migration

### Run Legacy Tests

```bash
# Verify existing tests still pass
pytest tests/test_bloomberg_integration.py -v
```

### Run New Tests

```bash
# Test cross-asset features
pytest tests/test_cross_asset_features.py -v

# Test mixed portfolio integration
pytest tests/test_mixed_portfolio_integration.py -v

# Test API frontend
pytest tests/test_api_frontend.py -v
```

### Performance Benchmarks

```bash
# Measure performance with new features
python tests/benchmark_performance.py
```

---

## Troubleshooting

### Issue: "Module not found: indicators.cross_asset"

**Solution**: Ensure you've pulled the latest code:

```bash
git pull origin main
pip install -r requirements.txt
```

### Issue: "Cross-asset features not being added"

**Cause**: Need both crypto AND credit columns in the dataset.

**Solution**: Verify column naming:
- Crypto: Must contain `USDT`, `BTC`, `ETH`, etc. AND `close`/`price`
- Credit: Must contain `OAS`, `DTS`, `YIELD`, `SPREAD`, or `INDEX`

```python
from indicators.cross_asset import identify_crypto_credit_columns

crypto_cols, credit_cols = identify_crypto_credit_columns(df)
print(f"Crypto columns found: {crypto_cols}")
print(f"Credit columns found: {credit_cols}")
```

### Issue: "Bloomberg API connection failed"

**Checklist**:
1. ✓ Bloomberg Terminal is running
2. ✓ Desktop API enabled (DAPI<GO> in Terminal)
3. ✓ Correct host/port in `.env` (default: localhost:8194)
4. ✓ User logged in to Terminal

**Fallback**: Use hybrid mode for automatic fallback to Excel:

```python
from _data_sources import HybridBloombergDataSource

source = HybridBloombergDataSource(
    securities=["LF98TRUU Index"],
    fields=["OAS"],
    start_date=start_date,
    end_date=end_date,
    excel_fallback_path="data/bloomberg_export.xlsx"
)
```

### Issue: "Date alignment issues (crypto 24/7 vs credit weekday)"

**Solution**: Use appropriate alignment strategy:

```python
# Option 1: Outer join (keep all dates, forward fill missing)
mixed_source = MixedPortfolioDataSource(
    sources=[crypto_source, credit_source],
    alignment='outer',
    fill_method='ffill',
    fill_limit=5
)

# Option 2: Inner join (only overlapping dates)
mixed_source = MixedPortfolioDataSource(
    sources=[crypto_source, credit_source],
    alignment='inner'
)

# Option 3: Left join (keep crypto dates, fill credit)
mixed_source = MixedPortfolioDataSource(
    sources=[crypto_source, credit_source],
    alignment='left'
)
```

### Issue: "Model performance degraded after adding cross-asset features"

**Possible Causes**:
1. Too many features (overfitting)
2. Insufficient data for rolling windows
3. High correlation between features

**Solutions**:

```python
# 1. Reduce correlation windows
cross_asset_calc = CrossAssetIndicators(df)
df = cross_asset_calc.add_all_cross_asset_features(
    crypto_cols=crypto_cols,
    credit_cols=credit_cols,
    correlation_windows=[60],  # Use only 60-day window
    momentum_window=20
)

# 2. Feature selection after training
model.predictive_power(forecast_range=30)
top_features = model._return_features_of_importance(forecast_day=30, top_n=20)

# 3. Increase data size
# Use at least 6 months of data for cross-asset analysis
```

---

## Rollback Plan

If you need to rollback to the legacy system:

```bash
# 1. Restore backup
cp config.yaml.backup config.yaml
cp -r data_backup data

# 2. Remove cross_asset_features parameter
# In your preprocessing code:
pipeline = _preprocess_xlsx(
    xlsx_file="data/economic_data.xlsx",
    target_col="LF98TRUU_Index_OAS",
    # Remove: cross_asset_features=True
)

# 3. Use legacy data sources
from _data_sources import ExcelDataSource
source = ExcelDataSource(file_path="data/bloomberg_export.xlsx")
```

---

## Next Steps

1. ✓ Review [CRYPTO_SECURITIES_USER_GUIDE.md](CRYPTO_SECURITIES_USER_GUIDE.md)
2. ✓ Explore [examples/](../examples/) for code samples
3. ✓ Read [BLOOMBERG_INTEGRATION.md](BLOOMBERG_INTEGRATION.md) for Bloomberg setup
4. ✓ Check [examples/README.md](../examples/README.md) for quick start guides

---

## Support

For issues or questions:
1. Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. Review existing [GitHub Issues](https://github.com/adrian-adduci/BBG-Credit-Momentum/issues)
3. Create new issue with:
   - System info (OS, Python version)
   - Configuration (sanitized)
   - Error logs
   - Steps to reproduce

---

**Migration Complete!** 🎉

Your system is now ready for unified crypto + credit analysis.
