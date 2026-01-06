# Bloomberg Integration Guide

**Version:** 1.0
**Date:** 2026-01-06
**Applicable to:** BBG-Credit-Momentum v2.0+

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Integration Options](#integration-options)
4. [Bloomberg Terminal API Setup](#bloomberg-terminal-api-setup)
5. [Bloomberg Excel Export](#bloomberg-excel-export)
6. [Hybrid Mode (Recommended)](#hybrid-mode-recommended)
7. [Configuration](#configuration)
8. [Usage Examples](#usage-examples)
9. [Supported Securities & Fields](#supported-securities--fields)
10. [Troubleshooting](#troubleshooting)
11. [Best Practices](#best-practices)

---

## Overview

BBG-Credit-Momentum supports three methods for integrating Bloomberg data:

1. **Bloomberg Terminal API** - Real-time data access via `blpapi` library
2. **Bloomberg Excel Export** - Pre-generated Excel files with Bloomberg plugins
3. **Hybrid Mode** - Automatic fallback between API and Excel (recommended)

All three methods provide the same data quality and are designed to work seamlessly with the existing analysis pipeline.

---

## Prerequisites

### For All Methods

- Python 3.8 or higher
- BBG-Credit-Momentum installed
- Access to Bloomberg credit market data

### For Bloomberg Terminal API

- **Bloomberg Terminal** installed and running on your system
- **DAPI<GO>** configured in Bloomberg Terminal
- Bloomberg Terminal license with API access
- `blpapi` Python library installed

### For Excel Export

- Microsoft Excel with Bloomberg Excel Add-in installed
- OR pre-generated Excel files from Bloomberg Terminal

---

## Integration Options

### Option 1: Bloomberg Terminal API

**Advantages:**
- Real-time data access
- Automated data fetching
- No manual export required
- Always up-to-date

**Disadvantages:**
- Requires Bloomberg Terminal running
- Requires Terminal license
- More complex setup
- Potential connection issues

**When to Use:**
- Automated trading systems
- Real-time analysis
- When Terminal is always available
- Large-scale data requirements

### Option 2: Bloomberg Excel Export

**Advantages:**
- No Terminal required after export
- Reliable and tested workflow
- Familiar to Bloomberg users
- Works offline
- No API complexity

**Disadvantages:**
- Manual export process
- Data can become stale
- Requires periodic updates
- Manual file management

**When to Use:**
- Manual analysis workflows
- Terminal access limited
- Occasional data updates
- Educational/research purposes

### Option 3: Hybrid Mode (Recommended)

**Advantages:**
- Best of both worlds
- Automatic failover
- Maximum reliability
- Production-ready

**Disadvantages:**
- Requires both setups
- Slightly more configuration

**When to Use:**
- Production environments
- Critical trading systems
- When reliability is paramount

---

## Bloomberg Terminal API Setup

### Step 1: Install Bloomberg Terminal

Ensure Bloomberg Terminal is installed on your system. Download from:
- Bloomberg Professional Services
- Contact your Bloomberg sales representative

### Step 2: Configure DAPI

1. Open Bloomberg Terminal
2. Type `DAPI<GO>` and press Enter
3. Follow the setup wizard to configure desktop API access
4. Verify API is enabled and running

### Step 3: Install blpapi Library

```bash
# Install blpapi from Bloomberg repository
pip install --index-url=https://bcms.bloomberg.com/pip/simple blpapi
```

**Note:** If the above fails, you may need to install from Bloomberg software repository:

```bash
# Alternative installation (Windows)
# Download blpapi from Bloomberg Terminal installation directory
# Usually located at: C:\blp\DAPI\APIv3\Python\

# Navigate to the Python directory and install
cd "C:\blp\DAPI\APIv3\Python\"
pip install blpapi-3.18.4-py3-none-any.whl  # Version may vary
```

### Step 4: Verify Installation

```python
# Test blpapi installation
import blpapi
print(f"blpapi version: {blpapi.__version__}")
```

### Step 5: Test Connection

Run the test script:

```bash
python scripts/test_bloomberg_connection.py
```

Expected output:
```
Connecting to Bloomberg Terminal at localhost:8194
Successfully connected to Bloomberg Terminal
Connection test passed!
```

### Step 6: Configure Application

Edit `config.yaml`:

```yaml
data_source:
  type: "bloomberg"  # Use Bloomberg API

  bloomberg:
    securities:
      - "LF98TRUU Index"     # US Aggregate Bond Index
      - "LUACTRUU Index"     # US Corporate Bond Index
    fields:
      - "OAS"                # Option-Adjusted Spread
      - "PX_LAST"            # Last Price
      - "DTS"                # Duration to Worst
    start_date: "2020-01-01"
    end_date: "2024-12-31"
    host: "localhost"
    port: 8194
    timeout: 30000
    max_retries: 3
```

---

## Bloomberg Excel Export

### Step 1: Prepare Excel Template

Use the provided template: `templates/bloomberg_credit_template.xlsx`

Or create your own:

1. Open Excel with Bloomberg Add-in
2. Create column headers: `Dates`, `LF98TRUU_Index_OAS`, etc.
3. Use Bloomberg formulas to populate data

### Step 2: Bloomberg Excel Formulas

For historical data, use `=BDH()` (Bloomberg Data History):

```excel
=BDH("LF98TRUU Index", "OAS", "2020-01-01", "2024-12-31", "Dir=V")
```

Formula parameters:
- `"LF98TRUU Index"`: Security identifier
- `"OAS"`: Field name
- `"2020-01-01"`: Start date
- `"2024-12-31"`: End date
- `"Dir=V"`: Direction vertical (dates down rows)

### Step 3: Export Data

1. Populate Excel with Bloomberg formulas
2. Wait for data to load (Bloomberg must be running)
3. Save As Excel file (.xlsx)
4. Place in `data/` directory

**Important:** Excel file must contain:
- `Dates` column with datetime values
- At least one data column
- No duplicate dates
- Dates in ascending order

### Step 4: Configure Application

Edit `config.yaml`:

```yaml
data_source:
  type: "bloomberg_excel"  # Use Bloomberg Excel source

  bloomberg_excel:
    file_path: "data/bloomberg_export.xlsx"
    sheet_name: 0  # First sheet
    date_column: "Dates"
    handle_errors: true  # Convert #N/A to NaN
```

---

## Hybrid Mode (Recommended)

Hybrid mode automatically tries Bloomberg API first, then falls back to Excel if API is unavailable.

### Setup

1. Complete both Bloomberg API and Excel setup
2. Configure both sources in `config.yaml`

```yaml
data_source:
  type: "bloomberg_hybrid"  # Use hybrid mode

  bloomberg:
    securities:
      - "LF98TRUU Index"
      - "LUACTRUU Index"
    fields:
      - "OAS"
      - "PX_LAST"
    start_date: "2020-01-01"
    end_date: "2024-12-31"
    host: "localhost"
    port: 8194
    timeout: 30000
    max_retries: 3
    excel_fallback: "data/bloomberg_export.xlsx"  # Fallback file
```

### How It Works

1. Application attempts Bloomberg API connection
2. If Terminal running: Uses API (real-time data)
3. If Terminal not running: Falls back to Excel (cached data)
4. If both fail: Raises clear error message

### Fallback Behavior

```
┌─────────────────────────┐
│  Start Load Data        │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Try Bloomberg API      │
└───────────┬─────────────┘
            │
    ┌───────┴───────┐
    │               │
    ▼               ▼
  Success        Failure
    │               │
    │               ▼
    │         ┌─────────────────┐
    │         │  Log Warning    │
    │         └────────┬────────┘
    │                  │
    │                  ▼
    │         ┌─────────────────┐
    │         │  Try Excel      │
    │         └────────┬────────┘
    │                  │
    │          ┌───────┴───────┐
    │          │               │
    │          ▼               ▼
    │        Success        Failure
    │          │               │
    │          │               ▼
    │          │         ┌──────────┐
    │          │         │  Error   │
    │          │         └──────────┘
    ▼          ▼
┌─────────────────────────┐
│  Return DataFrame       │
└─────────────────────────┘
```

---

## Configuration

### Environment Variables

Bloomberg settings can be overridden via environment variables:

```bash
# .env file
BLOOMBERG_API_ENABLED=true
BLOOMBERG_HOST=localhost
BLOOMBERG_PORT=8194
BLOOMBERG_TIMEOUT=30000
BLOOMBERG_MAX_RETRIES=3
BLOOMBERG_EXCEL_FALLBACK=data/bloomberg_export.xlsx
```

### Configuration Precedence

1. **Environment Variables** (highest priority)
2. **config.yaml** file
3. **Default values** (lowest priority)

### Full Configuration Reference

```yaml
data_source:
  # Data source type
  type: "bloomberg"  # or "bloomberg_excel" or "bloomberg_hybrid"

  bloomberg:
    # Securities to fetch (Bloomberg tickers)
    securities:
      - "LF98TRUU Index"      # US Aggregate
      - "LUACTRUU Index"      # US Corporate
      - "LF98TRHY Index"      # US High Yield

    # Fields to retrieve
    fields:
      - "OAS"                 # Option-Adjusted Spread
      - "PX_LAST"             # Last Price
      - "DTS"                 # Duration to Worst
      - "YLD_YTM_MID"         # Yield to Maturity
      - "AMOUNT_OUTSTANDING"  # Outstanding Amount

    # Date range
    start_date: "2020-01-01"
    end_date: "2024-12-31"

    # Connection settings
    host: "localhost"
    port: 8194
    timeout: 30000           # milliseconds
    max_retries: 3           # retry attempts
    batch_size: 100          # securities per request

    # Excel fallback (for hybrid mode)
    excel_fallback: "data/bloomberg_export.xlsx"

  bloomberg_excel:
    file_path: "data/bloomberg_export.xlsx"
    sheet_name: 0            # or sheet name as string
    date_column: "Dates"
    handle_errors: true      # Convert #N/A to NaN
    validate_formulas: false # Check for Bloomberg formulas
```

---

## Usage Examples

### Example 1: Basic Bloomberg API Usage

```python
from _data_sources import BloombergAPIDataSource
from datetime import datetime

# Initialize source
source = BloombergAPIDataSource(
    securities=["LF98TRUU Index", "LUACTRUU Index"],
    fields=["OAS", "PX_LAST"],
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2024, 12, 31)
)

# Load data
df = source.load_data()

print(df.head())
#         Dates  LF98TRUU_Index_OAS  LF98TRUU_Index_PX_LAST  ...
# 0  2020-01-01              123.45                  100.50  ...
# 1  2020-01-02              124.56                  100.75  ...
```

### Example 2: Bloomberg Excel with Error Handling

```python
from _data_sources import BloombergExcelDataSource

# Initialize source
source = BloombergExcelDataSource(
    file_path="data/bloomberg_export.xlsx",
    handle_errors=True  # Convert #N/A to NaN
)

# Load data
df = source.load_data()

# Check for any remaining errors
print(f"Missing values: {df.isna().sum().sum()}")
```

### Example 3: Hybrid Mode with Fallback

```python
from _data_sources import HybridBloombergDataSource
from datetime import datetime

# Initialize hybrid source
source = HybridBloombergDataSource(
    securities=["LF98TRUU Index"],
    fields=["OAS"],
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2024, 12, 31),
    excel_fallback_path="data/bloomberg_export.xlsx",
    prefer_api=True  # Try API first
)

# Load data (auto-fallback if API unavailable)
df = source.load_data()
```

### Example 4: Using DataSourceFactory

```python
from _data_sources import DataSourceFactory
from datetime import datetime

# Create Bloomberg API source via factory
source = DataSourceFactory.create(
    "bloomberg",
    securities=["LF98TRUU Index"],
    fields=["OAS"],
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2024, 12, 31)
)

df = source.load_data()
```

### Example 5: Integration with Analysis Pipeline

```python
from _data_sources import HybridBloombergDataSource
from _preprocessing import _preprocess_xlsx
from _models import _build_model
from datetime import datetime

# 1. Load Bloomberg data
source = HybridBloombergDataSource(
    securities=["LF98TRUU Index", "LUACTRUU Index"],
    fields=["OAS"],
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2024, 12, 31),
    excel_fallback_path="data/bloomberg_export.xlsx"
)

df = source.load_data()

# 2. Preprocess data
preprocessor = _preprocess_xlsx(
    data=df,
    target_col="LF98TRUU_Index_OAS",
    momentum_list=["LF98TRUU_Index_OAS", "LUACTRUU_Index_OAS"],
    momentum_X_days=[5, 10, 15],
    momentum_Y_days=30
)

# 3. Build and train model
model = _build_model(
    preprocessor=preprocessor,
    model_name="XGBoost",
    estimators=1000
)

# 4. Get predictions
predictions = model._return_final_data()
print(predictions.tail())
```

---

## Supported Securities & Fields

### Common Credit Index Tickers

| Ticker | Description |
|--------|-------------|
| `LF98TRUU Index` | Bloomberg Barclays US Aggregate Bond Index |
| `LUACTRUU Index` | Bloomberg Barclays US Corporate Bond Index |
| `LF98TRHY Index` | Bloomberg Barclays US High Yield Index |
| `LD01TRUU Index` | Bloomberg Barclays US 1-3 Year Government/Credit |
| `LD05TRUU Index` | Bloomberg Barclays US 5-7 Year Government/Credit |

### Supported Fields

| Field | Description | Unit |
|-------|-------------|------|
| `OAS` | Option-Adjusted Spread | basis points |
| `PX_LAST` | Last Price | price |
| `PX_OPEN` | Open Price | price |
| `PX_HIGH` | High Price | price |
| `PX_LOW` | Low Price | price |
| `PX_VOLUME` | Volume | shares |
| `YLD_YTM_MID` | Yield to Maturity | % |
| `DTS` | Duration to Worst | years |
| `AMOUNT_OUTSTANDING` | Outstanding Amount | currency |
| `RTG_MOODY` | Moody's Rating | rating |
| `RTG_SP` | S&P Rating | rating |

### Finding Security Identifiers

In Bloomberg Terminal:
1. Type `SECF<GO>` (Security Finder)
2. Search for your security
3. Copy the ticker (e.g., "LF98TRUU Index")

---

## Troubleshooting

### Problem: ImportError - blpapi not installed

**Error:**
```
ImportError: Bloomberg API (blpapi) not installed.
```

**Solution:**
```bash
pip install --index-url=https://bcms.bloomberg.com/pip/simple blpapi
```

If still failing, check Bloomberg Terminal installation directory for blpapi package.

---

### Problem: Bloomberg Terminal Not Running

**Error:**
```
BloombergTerminalNotRunning: Failed to start Bloomberg session at localhost:8194.
```

**Solution:**
1. Launch Bloomberg Terminal
2. Wait for Terminal to fully load
3. Verify DAPI is configured: `DAPI<GO>`
4. Retry connection

---

### Problem: #N/A Errors in Excel

**Error:**
Excel contains "#N/A N/A" or "#N/A Field Not Applicable"

**Solution:**
1. Enable error handling:
```yaml
bloomberg_excel:
  handle_errors: true  # Converts #N/A to NaN
```

2. Or fix in Bloomberg Terminal:
- Verify security ticker is valid
- Check field is available for that security
- Ensure date range is valid

---

### Problem: Duplicate Dates in Excel

**Error:**
```
ValueError: Bloomberg schema validation failed: Found X duplicate dates
```

**Solution:**
1. Check Excel for duplicate date rows
2. Remove duplicates manually, or
3. Use Excel formula: `=UNIQUE(A:A)` (Excel 365)

---

### Problem: Connection Timeout

**Error:**
```
TimeoutError: Request timeout after 30000ms
```

**Solution:**
1. Increase timeout in config:
```yaml
bloomberg:
  timeout: 60000  # 60 seconds
```

2. Check network connection
3. Reduce number of securities per request:
```yaml
bloomberg:
  batch_size: 50  # Reduce from 100
```

---

### Problem: Invalid Security Error

**Error:**
```
Security error for XYZ123 Index: Invalid security
```

**Solution:**
1. Verify ticker in Bloomberg Terminal: `SECF<GO>`
2. Check security is active (not delisted)
3. Ensure correct market identifier (e.g., "Index", "Equity", "Govt")

---

## Best Practices

### 1. Use Hybrid Mode in Production

For production systems, always use hybrid mode for maximum reliability:

```yaml
data_source:
  type: "bloomberg_hybrid"
  bloomberg:
    excel_fallback: "data/bloomberg_export_backup.xlsx"
```

### 2. Cache Data Locally

Reduce Bloomberg API calls by caching data:

```python
# Save DataFrame after loading
df = source.load_data()
df.to_parquet("cache/bloomberg_data.parquet")

# Load from cache if recent enough
from datetime import datetime, timedelta
cache_file = "cache/bloomberg_data.parquet"
cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))

if cache_age < timedelta(hours=1):
    df = pd.read_parquet(cache_file)
else:
    df = source.load_data()
    df.to_parquet(cache_file)
```

### 3. Batch Large Requests

For 100+ securities, use batching:

```yaml
bloomberg:
  batch_size: 50  # Process 50 securities at a time
```

### 4. Handle Rate Limits

Bloomberg API has rate limits (~2000 requests/hour). Implement delays:

```python
import time

securities_batches = [securities[i:i+50] for i in range(0, len(securities), 50)]

for batch in securities_batches:
    source = BloombergAPIDataSource(securities=batch, ...)
    df = source.load_data()
    time.sleep(2)  # 2 second delay between batches
```

### 5. Validate Data After Loading

Always validate Bloomberg data:

```python
df = source.load_data()

# Check for missing dates
expected_dates = pd.date_range(start_date, end_date, freq='B')  # Business days
missing_dates = expected_dates.difference(df['Dates'])
if len(missing_dates) > 0:
    print(f"Warning: Missing {len(missing_dates)} business days")

# Check for outliers
for col in df.columns:
    if col != 'Dates':
        z_scores = (df[col] - df[col].mean()) / df[col].std()
        outliers = df[abs(z_scores) > 3]
        if len(outliers) > 0:
            print(f"Warning: {len(outliers)} outliers in {col}")
```

### 6. Keep Excel Backups Updated

Regularly update Excel backup files (at least weekly):

1. Set up scheduled task to export Bloomberg data
2. Save with timestamp: `bloomberg_export_2024_01_15.xlsx`
3. Keep last 4 weeks of backups

### 7. Monitor API Usage

Log all Bloomberg API calls:

```python
import logging

logger = logging.getLogger("bloomberg_api_usage")
logger.info(f"Bloomberg API call: {len(securities)} securities, {len(fields)} fields")
```

Review logs monthly to optimize API usage.

---

## Additional Resources

- [Bloomberg API Developer Guide](https://www.bloomberg.com/professional/support/api-library/)
- [Bloomberg Terminal User Guide](https://www.bloomberg.com/professional/support/documentation/)
- [BBG-Credit-Momentum Documentation](../README.md)
- [Configuration Guide](CONFIGURATION.md)

---

## Support

For issues or questions:

1. Check [Troubleshooting](#troubleshooting) section
2. Review Bloomberg Terminal documentation
3. Open issue on GitHub repository
4. Contact Bloomberg support for Terminal/API issues

---

**Document Version History:**

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-06 | Initial Bloomberg integration documentation |
