# BBG-Credit-Momentum Examples

This directory contains comprehensive examples demonstrating how to use the BBG-Credit-Momentum system with Bloomberg and crypto securities.

---

## Quick Start Guide

### Prerequisites

- Python 3.8+
- BBG-Credit-Momentum installed
- Bloomberg Terminal (for API examples) OR Excel files (for Excel examples)
- Optional: Crypto exchange access for mixed portfolio examples

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For Bloomberg API (optional)
pip install --index-url=https://bcms.bloomberg.com/pip/simple blpapi
```

---

## Available Examples

### 1. Bloomberg API Examples (`bloomberg_api_example.py`)

**Purpose:** Learn how to use Bloomberg Terminal API integration

**Examples (7 total):**
1. **Basic Connection** - Simple API connection and data fetching
2. **Multiple Securities & Fields** - Fetch multiple indices with multiple fields
3. **Batch Processing** - Handle 100+ securities efficiently
4. **Retry Logic** - Automatic retry with exponential backoff
5. **Integration with Analysis** - Complete pipeline from API to predictions
6. **Custom Configuration** - All available configuration options

**Prerequisites:**
- Bloomberg Terminal running
- DAPI configured (type `DAPI<GO>` in Terminal)
- `blpapi` library installed

**Usage:**
```bash
python examples/bloomberg_api_example.py
```

**What You'll Learn:**
- How to connect to Bloomberg Terminal API
- Fetching historical credit spread data (OAS)
- Error handling and recovery
- Batch processing for performance
- Integration with ML models

---

### 2. Bloomberg Excel Examples (`bloomberg_excel_example.py`)

**Purpose:** Learn how to use Bloomberg Excel exports

**Examples (6 total):**
1. **Basic Excel Loading** - Load Bloomberg Excel exports
2. **Error Value Handling** - Handle #N/A, #VALUE!, etc.
3. **Schema Validation** - Validate Bloomberg export format
4. **Creating Excel Template** - Set up Bloomberg formulas
5. **Integration with Analysis** - Excel to model training
6. **Excel vs API Comparison** - Verify data consistency

**Prerequisites:**
- Excel file with Bloomberg data OR
- Microsoft Excel with Bloomberg Add-in

**Usage:**
```bash
python examples/bloomberg_excel_example.py
```

**What You'll Learn:**
- Loading Bloomberg Excel exports
- Handling Bloomberg error values automatically
- Creating Bloomberg Excel templates
- Bloomberg formulas (=BDH, =BDP)
- Schema validation

---

### 3. Bloomberg Hybrid Examples (`bloomberg_hybrid_example.py`)

**Purpose:** Production-ready hybrid mode (API + Excel fallback)

**Examples (7 total):**
1. **Basic Hybrid Mode** - Automatic fallback between API and Excel
2. **API-First Mode** - Real-time data with Excel fallback
3. **Excel-First Mode** - Offline mode with API fallback
4. **Terminal Unavailability** - Graceful degradation
5. **Production Deployment** - Monitoring and alerting patterns
6. **Data Source Comparison** - Validate consistency
7. **Complete Pipeline** - End-to-end production workflow

**Prerequisites:**
- Bloomberg Terminal (optional) OR
- Excel file with Bloomberg data (optional)
- At least one data source required

**Usage:**
```bash
python examples/bloomberg_hybrid_example.py
```

**What You'll Learn:**
- Maximum reliability with automatic fallback
- Production deployment patterns
- Monitoring and health checks
- Graceful degradation strategies
- When to use API vs Excel mode

---

### 4. Mixed Portfolio Examples (`mixed_portfolio_example.py`)

**Purpose:** Analyze crypto + traditional securities together

**Examples (1 complete pipeline):**
1. **Mixed Portfolio Analysis** - Crypto + Bloomberg credit in one model

**Features Demonstrated:**
- Loading BTC, ETH + Bloomberg credit indices
- Cross-asset correlation analysis
- Risk-on/risk-off regime detection
- Unified model training
- Multi-asset visualization

**Prerequisites:**
- Access to crypto exchanges (Binance) OR historical crypto data
- Bloomberg data (API or Excel)

**Usage:**
```bash
python examples/mixed_portfolio_example.py
```

**What You'll Learn:**
- Mixing crypto and traditional securities
- Cross-asset feature engineering
- Date alignment (24/7 crypto vs weekday credit)
- Regime detection
- Portfolio-level analysis

---

## Example Difficulty Levels

### Beginner
1. `bloomberg_excel_example.py` - Example 1 (Basic Excel Loading)
2. `bloomberg_hybrid_example.py` - Example 1 (Basic Hybrid Mode)

### Intermediate
1. `bloomberg_api_example.py` - Examples 1-3
2. `bloomberg_excel_example.py` - Examples 2-4
3. `bloomberg_hybrid_example.py` - Examples 2-3

### Advanced
1. `bloomberg_api_example.py` - Examples 4-6
2. `bloomberg_hybrid_example.py` - Examples 5-7
3. `mixed_portfolio_example.py` - Complete example

---

## Common Use Cases

### Use Case 1: "I want to fetch credit spread data from Bloomberg"

**Recommended Examples:**
1. Start with `bloomberg_api_example.py` - Example 1
2. Then try `bloomberg_api_example.py` - Example 5 for full pipeline

**Alternative (if no Terminal access):**
1. Use `bloomberg_excel_example.py` - Example 1
2. Then `bloomberg_excel_example.py` - Example 5

---

### Use Case 2: "I want maximum reliability (production)"

**Recommended Examples:**
1. `bloomberg_hybrid_example.py` - Example 1 (understand basics)
2. `bloomberg_hybrid_example.py` - Example 5 (production patterns)
3. `bloomberg_hybrid_example.py` - Example 7 (complete pipeline)

**Key Takeaway:** Hybrid mode tries API first, falls back to Excel automatically.

---

### Use Case 3: "I want to analyze crypto and credit together"

**Recommended Examples:**
1. `mixed_portfolio_example.py` - Complete example
2. Review `config.unified.yaml` for configuration

**Key Takeaway:** Single model can use both crypto and traditional features.

---

### Use Case 4: "I want to validate Bloomberg Excel exports"

**Recommended Examples:**
1. `bloomberg_excel_example.py` - Example 3 (Schema Validation)
2. `bloomberg_hybrid_example.py` - Example 6 (Source Comparison)

**Key Takeaway:** Automatic validation ensures data quality.

---

## Configuration Files

Each example can be configured via:

1. **config.example.yaml** - Traditional Bloomberg config
2. **config.unified.yaml** - Mixed portfolio config
3. **.env** - Environment variables for API keys

See configuration documentation for details.

---

## Troubleshooting

### Bloomberg API Issues

**Error: "Bloomberg Terminal Not Running"**
- Ensure Bloomberg Terminal is open
- Verify DAPI is configured: Type `DAPI<GO>` in Terminal
- Check Terminal is not busy

**Error: "blpapi not installed"**
```bash
pip install --index-url=https://bcms.bloomberg.com/pip/simple blpapi
```

**Error: "Connection timeout"**
- Increase timeout in configuration
- Check network connectivity
- Verify API host/port settings

### Excel Issues

**Error: "Date column not found"**
- Ensure Excel has "Dates" column
- Or specify custom date column name

**Error: "Bloomberg errors in data"**
- Enable `handle_errors=True` (automatic in BloombergExcelDataSource)
- Errors like #N/A will be converted to NaN

### Mixed Portfolio Issues

**Error: "Failed to load all securities"**
- Check each security source individually
- Review logs for specific error details
- Ensure data sources are accessible

---

## Example Output

### Bloomberg API Example Output
```
================================================================================
Example 1: Basic Bloomberg API Connection
================================================================================

Fetching data for: ['LF98TRUU Index']
Fields: ['OAS']
Date range: 2023-01-01 to 2024-12-31

Connecting to Bloomberg Terminal...
✓ Successfully loaded 504 rows
✓ Columns: ['Dates', 'LF98TRUU_Index_OAS']

First 5 rows:
        Dates  LF98TRUU_Index_OAS
0  2023-01-01              45.23
1  2023-01-02              45.67
...
```

### Mixed Portfolio Example Output
```
================================================================================
BBG Credit Momentum - Mixed Portfolio Example
================================================================================

Loading mixed portfolio data...
Securities: 4
  - Crypto: 2
  - Credit: 2
Date range: 2023-01-01 to 2024-12-31

✓ Loaded 730 rows, 6 columns

Cross-Asset Correlation Analysis:
BTC_USDT_close vs LF98TRUU_Index_OAS: -0.234

Regime Detection:
  Risk-On:  245 days (33.6%)
  Risk-Off: 189 days (25.9%)
  Neutral:  296 days (40.5%)

✓ Model trained successfully
```

---

## Additional Resources

### Documentation
- [Bloomberg Integration Guide](../docs/BLOOMBERG_INTEGRATION.md)
- [Implementation Plan](../CRYPTO_SECURITIES_IMPLEMENTATION_PLAN.md)
- [Configuration Reference](../config.example.yaml)

### Configuration Examples
- [Traditional Credit Config](../config.example.yaml)
- [Unified Mixed Portfolio Config](../config.unified.yaml)
- [Environment Variables](../.env.example)

### Source Code
- [Bloomberg API Source](../_data_sources.py) - BloombergAPIDataSource class
- [Excel Source](../_data_sources.py) - BloombergExcelDataSource class
- [Hybrid Source](../_data_sources.py) - HybridBloombergDataSource class
- [Mixed Portfolio Source](../_data_sources.py) - MixedPortfolioDataSource class

---

## Contributing Examples

Have an interesting use case? Contribute your example!

1. Create a new file: `examples/your_example.py`
2. Follow the existing example format
3. Include clear docstrings and comments
4. Add error handling
5. Update this README

---

## Support

For issues or questions:
1. Check [Troubleshooting](#troubleshooting) section above
2. Review [Bloomberg Integration Guide](../docs/BLOOMBERG_INTEGRATION.md)
3. Open an issue on GitHub repository

---

## License

MIT License - Same as BBG-Credit-Momentum project

---

**Quick Reference:**

| Example | Focus | Prerequisites | Difficulty |
|---------|-------|---------------|------------|
| bloomberg_api_example.py | Bloomberg Terminal API | Terminal + blpapi | Intermediate |
| bloomberg_excel_example.py | Excel exports | Excel file | Beginner |
| bloomberg_hybrid_example.py | Production reliability | API OR Excel | Intermediate |
| mixed_portfolio_example.py | Crypto + Credit | Crypto + Bloomberg | Advanced |

**Total Examples:** 20+
**Total Lines of Code:** 1,500+
**Estimated Learning Time:** 2-4 hours
