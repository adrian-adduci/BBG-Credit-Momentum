# Crypto Securities Implementation Plan - BBG Credit Momentum

**Date:** 2026-01-06
**Version:** 1.0
**Status:** Planning Phase

---

## Executive Summary

This plan outlines the implementation strategy for integrating crypto securities into the BBG-Credit-Momentum analysis framework with Bloomberg data integration. The system will support dual-mode operation:

1. **Bloomberg API Mode**: Real-time data fetching from Bloomberg Terminal (when access granted)
2. **Excel Export Mode**: Pre-generated Excel files with Bloomberg Excel plugin data (current working method)

The implementation will create a unified pipeline that treats both traditional credit securities and crypto securities as first-class citizens, enabling comparative analysis and portfolio-level insights.

---

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [Implementation Goals](#2-implementation-goals)
3. [Architecture Design](#3-architecture-design)
4. [Bloomberg Integration Strategy](#4-bloomberg-integration-strategy)
5. [Crypto Securities Data Pipeline](#5-crypto-securities-data-pipeline)
6. [Implementation Phases](#6-implementation-phases)
7. [Configuration Updates](#7-configuration-updates)
8. [Testing Strategy](#8-testing-strategy)
9. [Documentation Requirements](#9-documentation-requirements)
10. [Risk Analysis & Mitigation](#10-risk-analysis--mitigation)

---

## 1. Current State Analysis

### 1.1 Existing Strengths

✅ **Crypto Data Infrastructure**
- Full CCXT integration (100+ exchanges)
- Blockchain on-chain metrics (Glassnode, CoinMetrics)
- 10+ technical indicators (RSI, MACD, Bollinger, Stochastic, etc.)
- Multi-exchange aggregation
- PostgreSQL caching layer

✅ **Traditional Credit Infrastructure**
- Excel file ingestion (Bloomberg exports)
- Momentum feature engineering
- XGBoost/CART ML models
- Feature importance analysis
- Streamlit UI + FastAPI

✅ **Modular Architecture**
- Abstract data source factory pattern
- Pluggable preprocessing pipeline
- Configurable feature engineering
- Multiple model support

### 1.2 Current Gaps

⚠️ **Bloomberg Integration**
- API integration is template only (requires implementation)
- No standardized schema for Bloomberg Excel exports
- No validation for Bloomberg-specific data formats

⚠️ **Crypto Securities Treatment**
- Crypto treated as separate use case (config.crypto.yaml)
- No unified view of traditional + crypto securities
- Limited cross-asset analysis capabilities

⚠️ **Data Harmonization**
- Different column naming conventions (traditional vs crypto)
- No unified security identifier system
- Missing metadata for security classification

---

## 2. Implementation Goals

### 2.1 Primary Objectives

1. **Unified Security Framework**
   - Treat crypto and traditional securities uniformly
   - Single configuration for mixed portfolios
   - Consistent feature engineering across asset types

2. **Flexible Bloomberg Integration**
   - Bloomberg API support (when Terminal access available)
   - Robust Excel import with schema validation
   - Automatic data format detection

3. **Enhanced Analytics**
   - Cross-asset correlation analysis
   - Portfolio-level momentum indicators
   - Regime detection (risk-on/risk-off)

4. **Production-Ready Data Pipeline**
   - Error handling for API failures
   - Fallback to Excel when API unavailable
   - Data quality validation
   - Comprehensive logging

### 2.2 Success Criteria

- [ ] Bloomberg API successfully fetches credit spreads (OAS, Z-spread)
- [ ] Excel files with Bloomberg plugins are parsed correctly
- [ ] Crypto + traditional securities analyzed in single model
- [ ] Configuration supports mixed portfolios
- [ ] All data sources have 95%+ reliability
- [ ] Feature parity between API and Excel modes

---

## 3. Architecture Design

### 3.1 Unified Security Model

Create a `Security` class to represent any tradable instrument:

```python
@dataclass
class Security:
    """Universal security representation"""
    identifier: str          # "BTC/USDT" or "LF98TRUU Index"
    security_type: str       # "crypto", "credit", "equity", "fx"
    source: str              # "binance", "bloomberg", "file"
    data_fields: List[str]   # ["close", "volume"] or ["OAS", "DTS"]
    metadata: Dict[str, Any] # Additional attributes

    def to_column_prefix(self) -> str:
        """Generate DataFrame column prefix"""
        # BTC/USDT -> BTC_USDT
        # LF98TRUU Index -> LF98TRUU_Index
        return self.identifier.replace("/", "_").replace(" ", "_")
```

### 3.2 Enhanced Data Source Factory

Extend `DataSourceFactory` to handle hybrid portfolios:

```python
class MixedPortfolioDataSource(DataSource):
    """Combines multiple data sources into unified DataFrame"""

    def __init__(self, sources: List[DataSource]):
        self.sources = sources

    def load_data(self) -> pd.DataFrame:
        """
        Load and merge data from all sources:
        1. Load each source independently
        2. Align on common date index
        3. Merge into wide DataFrame
        4. Handle missing data (forward fill)
        """
        pass
```

### 3.3 Bloomberg-Specific Components

#### 3.3.1 Bloomberg API Data Source (Complete Implementation)

```python
class BloombergAPIDataSource(DataSource):
    """
    Production-ready Bloomberg API integration
    - Session management with auto-reconnect
    - Request batching (max 100 securities per request)
    - Rate limiting (Bloomberg limits: ~2000 req/hour)
    - Error handling with retry logic
    - Field mapping (OAS, DTS, PRICE, YIELD)
    """

    SUPPORTED_FIELDS = {
        "OAS": "Option-Adjusted Spread",
        "PX_LAST": "Last Price",
        "YLD_YTM_MID": "Yield to Maturity",
        "DTS": "Duration to Worst",
        "AMOUNT_OUTSTANDING": "Outstanding Amount"
    }
```

#### 3.3.2 Bloomberg Excel Data Source (Enhanced)

```python
class BloombergExcelDataSource(ExcelDataSource):
    """
    Enhanced Excel parser with Bloomberg schema validation
    - Detects Bloomberg formula columns (=BDP, =BDH)
    - Validates required fields (Dates column)
    - Handles Bloomberg error values (#N/A N/A, #N/A Field Not Applicable)
    - Supports multiple sheets (timeseries + static data)
    """

    BLOOMBERG_ERROR_VALUES = ["#N/A N/A", "#N/A Field Not Applicable", "#VALUE!"]

    def validate_schema(self) -> Dict[str, str]:
        """
        Returns:
            {column_name: detected_field_type}
        """
        pass
```

### 3.4 Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SECURITY DEFINITIONS                          │
│  securities:                                                     │
│    - {id: "BTC/USDT", type: "crypto", source: "binance"}       │
│    - {id: "LF98TRUU Index", type: "credit", source: "bloomberg"}│
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  PARALLEL DATA FETCHING                          │
│  ┌──────────────────┬──────────────────┬──────────────────┐    │
│  │ Bloomberg API    │ Crypto Exchange  │ Blockchain APIs  │    │
│  │ (if available)   │ (CCXT)           │ (Glassnode)      │    │
│  └──────────────────┴──────────────────┴──────────────────┘    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Bloomberg Excel (fallback if API unavailable)            │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA ALIGNMENT                                │
│  - Common datetime index (business days)                         │
│  - Fill missing data (forward fill with 5-day limit)            │
│  - Outlier detection (Z-score > 3)                              │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              UNIFIED FEATURE ENGINEERING                         │
│  For ALL securities:                                             │
│    - Momentum features (5, 10, 15 day)                          │
│    - Volatility (20 day rolling std)                            │
│  For crypto only:                                                │
│    - Technical indicators (RSI, MACD, etc.)                     │
│    - On-chain metrics (if available)                            │
│  For credit only:                                                │
│    - Credit spread momentum                                      │
│    - Sector correlations                                         │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ML MODEL TRAINING                              │
│  - Single model for all securities                               │
│  - OR separate models with ensemble voting                       │
│  - Cross-asset feature importance                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Bloomberg Integration Strategy

### 4.1 Bloomberg API Implementation

#### Phase 1: Core Functionality

**File:** `_data_sources.py` (enhance existing `BloombergAPIDataSource`)

**Required Methods:**

1. **`_create_session()`**
   ```python
   def _create_session(self) -> blpapi.Session:
       """
       Create Bloomberg session with error handling
       - Try local connection (localhost:8194)
       - Fall back to Bloomberg Anywhere (if configured)
       - Raise clear error if Terminal not running
       """
   ```

2. **`_fetch_historical_data()`**
   ```python
   def _fetch_historical_data(
       self,
       securities: List[str],
       fields: List[str],
       start_date: datetime,
       end_date: datetime
   ) -> pd.DataFrame:
       """
       Fetch time series data using HistoricalDataRequest
       - Batch requests (max 100 securities)
       - Handle pagination for large date ranges
       - Parse Bloomberg response format
       """
   ```

3. **`_fetch_reference_data()`**
   ```python
   def _fetch_reference_data(
       self,
       securities: List[str],
       fields: List[str]
   ) -> pd.DataFrame:
       """
       Fetch static security data using ReferenceDataRequest
       - Useful for metadata (SECURITY_NAME, CRNCY, COUNTRY)
       """
   ```

4. **Error Handling**
   ```python
   class BloombergAPIError(Exception):
       """Base exception for Bloomberg API errors"""
       pass

   class BloombergTerminalNotRunning(BloombergAPIError):
       """Bloomberg Terminal is not running"""
       pass

   class BloombergAuthenticationError(BloombergAPIError):
       """Authentication failed"""
       pass

   class BloombergInvalidSecurity(BloombergAPIError):
       """Security identifier not found"""
       pass
   ```

#### Phase 2: Advanced Features

5. **`_subscribe_realtime()`** (Future)
   ```python
   def _subscribe_realtime(
       self,
       securities: List[str],
       fields: List[str],
       callback: Callable
   ):
       """
       Real-time data subscription for intraday analysis
       - Uses subscription API
       - Callback function for each update
       """
   ```

6. **Caching Layer**
   ```python
   class BloombergCachedDataSource(BloombergAPIDataSource):
       """
       Wraps Bloomberg API with PostgreSQL cache
       - Cache historical data (immutable once date passes)
       - Fetch only new dates via API
       - Reduce API usage and costs
       """
   ```

### 4.2 Bloomberg Excel Import Enhancement

#### Enhanced Schema Validation

**File:** `_data_sources.py` (new `BloombergExcelDataSource` class)

**Features:**

1. **Formula Detection**
   ```python
   def detect_bloomberg_formulas(self, sheet) -> Dict[str, str]:
       """
       Detect Bloomberg Excel formulas in header rows:
       - =BDH("LF98TRUU Index", "OAS", start, end)
       - =BDP("AAPL US Equity", "PX_LAST")

       Returns:
           {column_name: "BDH:LF98TRUU Index:OAS"}
       """
   ```

2. **Error Value Handling**
   ```python
   BLOOMBERG_ERROR_MAP = {
       "#N/A N/A": np.nan,
       "#N/A Field Not Applicable": np.nan,
       "#VALUE!": np.nan,
       "#REF!": np.nan
   }
   ```

3. **Multi-Sheet Support**
   ```python
   class BloombergExcelDataSource:
       def load_data(self) -> Dict[str, pd.DataFrame]:
           """
           Load multiple sheets from Bloomberg workbook:
           - "Timeseries" sheet: Historical data (BDH)
           - "Static" sheet: Security metadata (BDP)
           - "Portfolio" sheet: Holdings and weights
           """
   ```

4. **Template Validation**
   ```python
   def validate_bloomberg_template(self, df: pd.DataFrame) -> List[str]:
       """
       Check required columns and format:
       - Required: "Dates" column (or "Date")
       - At least one data column
       - Dates must be in ascending order
       - No duplicate dates

       Returns:
           List of validation errors (empty if valid)
       """
   ```

#### Standardized Excel Template

Create template file: `templates/bloomberg_credit_template.xlsx`

**Sheet 1: Timeseries Data**
```
Dates       | LF98TRUU_Index_OAS | LUACTRUU_Index_OAS | HY_Spread | IG_Spread
2024-01-01  | 123.45             | 234.56             | 345.67    | 123.45
2024-01-02  | 124.56             | 235.67             | 346.78    | 124.56
...
```

**Sheet 2: Security Metadata**
```
Ticker           | Security_Name              | Asset_Class | Currency
LF98TRUU Index   | US Aggregate Bond Index    | Credit      | USD
LUACTRUU Index   | US Corporate Bond Index    | Credit      | USD
```

**Sheet 3: Bloomberg Formulas (Reference)**
```
Column_Name          | Bloomberg_Formula
LF98TRUU_Index_OAS   | =BDH("LF98TRUU Index", "OAS", A2, A100)
LUACTRUU_Index_OAS   | =BDH("LUACTRUU Index", "OAS", A2, A100)
```

### 4.3 Hybrid Mode (API + Excel)

**Fallback Strategy:**

```python
class HybridBloombergDataSource(DataSource):
    """
    Intelligent fallback between API and Excel:
    1. Try Bloomberg API first
    2. If API unavailable, look for Excel file
    3. Cache API results to reduce future API calls
    4. Merge Excel and API data if partial availability
    """

    def load_data(self) -> pd.DataFrame:
        try:
            # Try API
            return self._api_source.load_data()
        except BloombergTerminalNotRunning:
            logger.warning("Bloomberg Terminal not running, falling back to Excel")
            return self._excel_source.load_data()
        except Exception as e:
            logger.error(f"Bloomberg API error: {e}")
            return self._excel_source.load_data()
```

---

## 5. Crypto Securities Data Pipeline

### 5.1 Crypto Security Definition

Enhance crypto configuration to align with Bloomberg structure:

**Current (crypto-specific):**
```yaml
crypto:
  exchange: "binance"
  symbols: ["BTC/USDT", "ETH/USDT"]
```

**Proposed (unified):**
```yaml
securities:
  - identifier: "BTC/USDT"
    type: "crypto_spot"
    source: "binance"
    fields: ["close", "volume", "high", "low"]
    features:
      technical_indicators: true
      blockchain_metrics: ["mvrv", "nvt"]

  - identifier: "ETH/USDT"
    type: "crypto_spot"
    source: "coinbase"
    fields: ["close", "volume"]
    features:
      technical_indicators: true
      blockchain_metrics: ["active_addresses", "transaction_count"]

  - identifier: "LF98TRUU Index"
    type: "credit_index"
    source: "bloomberg"
    fields: ["OAS", "DTS", "PRICE"]
    features:
      momentum: true
      sector_correlations: true
```

### 5.2 Crypto-Specific Enhancements

**1. Crypto Security Metadata**

```python
@dataclass
class CryptoSecurityMetadata:
    """Extended metadata for crypto securities"""
    base_currency: str      # "BTC"
    quote_currency: str     # "USDT"
    exchange: str           # "binance"
    market_cap_rank: int    # 1 for BTC
    is_stablecoin: bool     # True for USDT
    blockchain: str         # "Bitcoin", "Ethereum"
    consensus: str          # "PoW", "PoS"
```

**2. Crypto Data Quality Checks**

```python
def validate_crypto_data(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Crypto-specific validation:
    - Remove candles with zero volume (exchange errors)
    - Check for flash crashes (price drop > 30% in 1 candle)
    - Verify OHLC relationships (high >= close >= low)
    - Flag suspicious volume spikes (> 10x median)
    """
```

**3. Multi-Exchange Consensus**

```python
class CryptoConsensusDataSource(DataSource):
    """
    Fetch same symbol from multiple exchanges and create consensus:
    - VWAP across exchanges
    - Outlier removal (exclude if price > 2% from median)
    - Higher confidence for liquid markets
    """

    def load_data(self) -> pd.DataFrame:
        """
        Returns DataFrame with consensus pricing:
        - BTC_USDT_close (VWAP from all exchanges)
        - BTC_USDT_spread (max - min across exchanges)
        - BTC_USDT_confidence (0-1 score based on agreement)
        """
```

### 5.3 Cross-Asset Features

**New feature engineering for mixed portfolios:**

```python
def add_cross_asset_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Calculate features that relate crypto and traditional securities:

    1. Risk-On/Risk-Off Regime
       - When credit spreads widen AND BTC falls -> Risk-Off
       - When credit spreads tighten AND BTC rises -> Risk-On

    2. Crypto-Credit Correlation
       - Rolling 30-day correlation between BTC and HY spreads
       - Regime changes when correlation flips

    3. Flight-to-Quality Indicator
       - Measure if capital flows from crypto to safe credit
       - BTC_volume_change vs IG_spread_tightening

    4. Momentum Divergence
       - Crypto momentum vs Credit momentum
       - Divergence may signal trend reversal
    """
```

---

## 6. Implementation Phases

### Phase 1: Bloomberg API Foundation (Week 1-2)

**Deliverables:**
1. Complete `BloombergAPIDataSource.load_data()` implementation
2. Error handling and retry logic
3. Unit tests with mock Bloomberg API responses
4. Configuration updates for API credentials
5. Documentation for Bloomberg Terminal setup

**Files to Modify:**
- `_data_sources.py` (complete Bloomberg API class)
- `_config.py` (add Bloomberg API config section)
- `requirements.txt` (add `blpapi`)
- `tests/test_bloomberg_api.py` (new file)

**Success Criteria:**
- [ ] Can fetch historical OAS data for credit indices
- [ ] Handles API errors gracefully
- [ ] Falls back to Excel if API unavailable

### Phase 2: Bloomberg Excel Enhancement (Week 2)

**Deliverables:**
1. `BloombergExcelDataSource` class with schema validation
2. Standard Excel template for Bloomberg exports
3. Error value handling (#N/A, etc.)
4. Multi-sheet support
5. Template validation function

**Files to Modify:**
- `_data_sources.py` (new `BloombergExcelDataSource` class)
- `templates/bloomberg_credit_template.xlsx` (new file)
- `tests/test_bloomberg_excel.py` (new file)
- `docs/BLOOMBERG_EXCEL_GUIDE.md` (new file)

**Success Criteria:**
- [ ] Parses Excel files with Bloomberg formulas
- [ ] Validates required columns
- [ ] Handles error values correctly
- [ ] Loads multi-sheet workbooks

### Phase 3: Unified Security Framework (Week 3)

**Deliverables:**
1. `Security` dataclass for universal security representation
2. `MixedPortfolioDataSource` for combining multiple sources
3. Updated configuration schema for mixed portfolios
4. Migration guide from old config to new config

**Files to Modify:**
- `_data_sources.py` (add `Security` class, `MixedPortfolioDataSource`)
- `_config.py` (support new securities schema)
- `config.unified.yaml` (new example config)
- `docs/MIGRATION_GUIDE.md` (new file)

**Success Criteria:**
- [ ] Single config file defines both crypto and credit securities
- [ ] Data loaded from multiple sources into unified DataFrame
- [ ] Column naming consistent across sources

### Phase 4: Cross-Asset Feature Engineering (Week 4)

**Deliverables:**
1. Cross-asset correlation features
2. Regime detection (risk-on/risk-off)
3. Momentum divergence indicators
4. Updated preprocessing pipeline

**Files to Modify:**
- `_preprocessing.py` (add `add_cross_asset_features()`)
- `indicators/cross_asset.py` (new file)
- `tests/test_cross_asset_features.py` (new file)

**Success Criteria:**
- [ ] Calculates crypto-credit correlation
- [ ] Detects regime changes
- [ ] Features improve model accuracy (validation)

### Phase 5: UI/API Updates (Week 5)

**Deliverables:**
1. Streamlit UI support for mixed portfolios
2. API endpoints for crypto + credit analysis
3. Visualization updates (multi-asset charts)
4. Backtesting for mixed strategies

**Files to Modify:**
- `webapp.py` (mixed portfolio UI)
- `api.py` (new endpoints)
- `tests/test_api_mixed.py` (new file)

**Success Criteria:**
- [ ] UI displays both crypto and credit securities
- [ ] API accepts mixed portfolio configs
- [ ] Charts show cross-asset relationships

### Phase 6: Testing & Documentation (Week 6)

**Deliverables:**
1. Comprehensive test suite
2. User documentation
3. Configuration examples
4. Performance benchmarks

**Files to Create:**
- `tests/integration/test_bloomberg_integration.py`
- `tests/integration/test_mixed_portfolio.py`
- `docs/BLOOMBERG_INTEGRATION.md`
- `docs/CRYPTO_SECURITIES.md`
- `examples/mixed_portfolio_example.py`

**Success Criteria:**
- [ ] 80%+ code coverage
- [ ] All documentation complete
- [ ] Example notebooks work end-to-end

---

## 7. Configuration Updates

### 7.1 New Configuration Schema

**File:** `config.unified.yaml` (new example config)

```yaml
# BBG Credit Momentum - Unified Configuration
# Supports crypto securities + traditional credit securities

# Security Definitions
securities:
  # Crypto Securities
  - identifier: "BTC/USDT"
    type: "crypto_spot"
    source:
      type: "crypto_exchange"
      exchange: "binance"
      timeframe: "1h"
    fields: ["close", "volume", "high", "low", "open"]
    features:
      technical_indicators:
        enabled: true
        rsi: {period: 14}
        macd: {fast: 12, slow: 26, signal: 9}
        bollinger: {period: 20, std: 2}
      blockchain_metrics:
        enabled: true
        provider: "glassnode"
        metrics: ["mvrv", "nvt", "active_addresses"]

  - identifier: "ETH/USDT"
    type: "crypto_spot"
    source:
      type: "crypto_exchange"
      exchange: "binance"
      timeframe: "1h"
    fields: ["close", "volume"]
    features:
      technical_indicators:
        enabled: true
      blockchain_metrics:
        enabled: true
        provider: "glassnode"
        metrics: ["mvrv", "transaction_count"]

  # Traditional Credit Securities
  - identifier: "LF98TRUU Index"
    type: "credit_index"
    display_name: "US Aggregate Bond Index"
    source:
      type: "bloomberg"
      method: "api"  # or "excel"
      excel_fallback: "data/bloomberg_credit.xlsx"
    fields: ["OAS", "DTS", "PRICE"]
    features:
      momentum:
        enabled: true
        windows: [5, 10, 15]
        baseline: 30

  - identifier: "LUACTRUU Index"
    type: "credit_index"
    display_name: "US Corporate Bond Index"
    source:
      type: "bloomberg"
      method: "api"
      excel_fallback: "data/bloomberg_credit.xlsx"
    fields: ["OAS", "PRICE"]
    features:
      momentum:
        enabled: true

# Bloomberg Configuration
bloomberg:
  api:
    enabled: true
    host: "localhost"
    port: 8194
    timeout: 30000  # milliseconds
    retry:
      max_attempts: 3
      backoff: "exponential"
  excel:
    default_sheet: "Timeseries"
    date_column: "Dates"
    handle_errors: true  # Convert #N/A to NaN

# Cross-Asset Features
cross_asset_features:
  enabled: true
  correlations:
    - pair: ["BTC/USDT", "LF98TRUU Index"]
      window: 30
      method: "pearson"
  regime_detection:
    enabled: true
    indicators: ["BTC/USDT_momentum", "LF98TRUU_Index_OAS_momentum"]

# Model Configuration
model:
  type: "XGBoost"
  estimators: 1000
  random_state: 42
  cv_folds: 5
  target:
    security: "BTC/USDT"
    field: "close"
    forecast_horizons: [1, 3, 7, 15, 30]  # days ahead

# Data Processing
preprocessing:
  alignment:
    method: "outer"  # Keep all dates, fill missing
    fill_method: "ffill"
    fill_limit: 5  # Max 5 days forward fill
  outlier_detection:
    enabled: true
    method: "zscore"
    threshold: 3.0

# Database (optional caching)
database:
  enabled: false
  host: ${DB_HOST}
  port: ${DB_PORT}
  name: "bbg_credit_momentum"
  cache_ttl: 86400  # 1 day

# Logging
logging:
  level: "INFO"
  files:
    main: "logs/_main.log"
    bloomberg: "logs/_bloomberg.log"
    crypto: "logs/_crypto.log"
```

### 7.2 Environment Variables

**File:** `.env.example` (update)

```bash
# Bloomberg Configuration
BLOOMBERG_API_ENABLED=true
BLOOMBERG_HOST=localhost
BLOOMBERG_PORT=8194

# Crypto Exchange API Keys
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here

# Blockchain Data Providers
GLASSNODE_API_KEY=your_glassnode_key
COINMETRICS_API_KEY=your_coinmetrics_key

# Database (optional)
DB_ENABLED=false
DB_HOST=192.168.1.100
DB_PORT=5432
DB_NAME=bbg_credit_momentum
DB_USER=your_username
DB_PASSWORD=your_password

# Model Configuration
MODEL_TYPE=XGBoost
TARGET_SECURITY=BTC/USDT
TARGET_FIELD=close

# Performance
NUMEXPR_MAX_THREADS=16
```

### 7.3 Migration from Old Config

**Migration Script:** `scripts/migrate_config.py`

```python
def migrate_crypto_config_to_unified(old_config_path: str) -> dict:
    """
    Convert config.crypto.yaml to new unified format:
    - crypto.symbols -> securities list
    - crypto.exchange -> securities[].source.exchange
    - Add default features configuration
    """
    pass

def migrate_credit_config_to_unified(old_config_path: str) -> dict:
    """
    Convert config.example.yaml to new unified format:
    - bloomberg.securities -> securities list
    - features.target -> model.target
    """
    pass
```

---

## 8. Testing Strategy

### 8.1 Unit Tests

**Bloomberg API Tests** (`tests/test_bloomberg_api.py`)
```python
class TestBloombergAPIDataSource:
    def test_create_session_success(self, mock_blpapi):
        """Test session creation with running Terminal"""
        pass

    def test_create_session_terminal_not_running(self, mock_blpapi):
        """Test error handling when Terminal not running"""
        pass

    def test_fetch_historical_data_single_security(self, mock_blpapi):
        """Test fetching OAS for single credit index"""
        pass

    def test_fetch_historical_data_batch(self, mock_blpapi):
        """Test batching 100+ securities"""
        pass

    def test_invalid_security_error(self, mock_blpapi):
        """Test handling of invalid Bloomberg ticker"""
        pass
```

**Bloomberg Excel Tests** (`tests/test_bloomberg_excel.py`)
```python
class TestBloombergExcelDataSource:
    def test_load_standard_template(self):
        """Test loading standard Bloomberg export"""
        pass

    def test_handle_bloomberg_errors(self):
        """Test converting #N/A values to NaN"""
        pass

    def test_multi_sheet_loading(self):
        """Test loading timeseries + metadata sheets"""
        pass

    def test_formula_detection(self):
        """Test detecting BDH/BDP formulas"""
        pass
```

**Mixed Portfolio Tests** (`tests/test_mixed_portfolio.py`)
```python
class TestMixedPortfolioDataSource:
    def test_load_crypto_and_credit(self):
        """Test loading BTC + credit index together"""
        pass

    def test_date_alignment(self):
        """Test aligning crypto (24/7) with credit (weekdays only)"""
        pass

    def test_forward_fill_limit(self):
        """Test max 5-day forward fill for missing data"""
        pass
```

### 8.2 Integration Tests

**Bloomberg Integration** (`tests/integration/test_bloomberg_integration.py`)
```python
@pytest.mark.skipif(not bloomberg_available(), reason="Bloomberg Terminal required")
class TestBloombergIntegration:
    def test_api_fetch_real_data(self):
        """Test actual Bloomberg API call (requires Terminal)"""
        pass

    def test_excel_to_api_consistency(self):
        """Verify Excel and API return same data"""
        pass
```

**End-to-End Pipeline** (`tests/integration/test_e2e_pipeline.py`)
```python
class TestEndToEndPipeline:
    def test_mixed_portfolio_training(self):
        """
        Full pipeline test:
        1. Load BTC + credit securities
        2. Feature engineering
        3. Model training
        4. Predictions
        5. Backtesting
        """
        pass
```

### 8.3 Test Data

**Mock Bloomberg Responses** (`tests/fixtures/bloomberg_responses/`)
- `historical_data_response.xml` - Sample HistoricalDataRequest response
- `reference_data_response.xml` - Sample ReferenceDataRequest response
- `error_response.xml` - Sample error response

**Sample Excel Files** (`tests/fixtures/excel/`)
- `bloomberg_credit_valid.xlsx` - Valid Bloomberg export
- `bloomberg_credit_with_errors.xlsx` - Contains #N/A values
- `bloomberg_multi_sheet.xlsx` - Multiple sheets

**Mock Crypto Data** (`tests/fixtures/crypto/`)
- `btc_usdt_1h.csv` - Sample BTC hourly data
- `eth_usdt_1h.csv` - Sample ETH hourly data

### 8.4 Performance Benchmarks

**Benchmark Script:** `tests/benchmarks/benchmark_data_loading.py`

```python
def benchmark_bloomberg_api_loading(num_securities: int, days: int):
    """
    Measure time to load historical data:
    - 10 securities, 365 days
    - 50 securities, 365 days
    - 100 securities, 365 days
    """
    pass

def benchmark_mixed_portfolio_loading():
    """
    Measure time to load mixed portfolio:
    - 5 crypto + 10 credit securities
    - 1 year of data
    """
    pass
```

**Target Performance:**
- Bloomberg API: < 5 seconds for 10 securities, 1 year
- Excel loading: < 2 seconds for 100 columns, 1 year
- Mixed portfolio: < 10 seconds for 15 securities, 1 year

---

## 9. Documentation Requirements

### 9.1 User Documentation

**1. Bloomberg Integration Guide** (`docs/BLOOMBERG_INTEGRATION.md`)

```markdown
# Bloomberg Integration Guide

## Overview
This guide explains how to integrate Bloomberg data into BBG Credit Momentum.

## Prerequisites
- Bloomberg Terminal access (for API mode)
- Excel with Bloomberg plugins (for Excel mode)

## Setup

### Option 1: Bloomberg API (Recommended)
1. Ensure Bloomberg Terminal is running
2. Install blpapi: `pip install blpapi`
3. Configure API settings in config.yaml
4. Test connection: `python scripts/test_bloomberg_connection.py`

### Option 2: Excel Export
1. Open Bloomberg Terminal
2. Export data using provided template
3. Save as .xlsx file
4. Configure file path in config.yaml

## Supported Securities
- Credit indices (LF98TRUU Index, etc.)
- Corporate bonds (AAPL 4.5 01/15/2030)
- Sovereigns (T 2.875 05/15/2032)

## Supported Fields
- OAS: Option-Adjusted Spread
- DTS: Duration to Worst
- PRICE: Last price
- YLD_YTM_MID: Yield to Maturity
```

**2. Crypto Securities Guide** (`docs/CRYPTO_SECURITIES.md`)

```markdown
# Crypto Securities Guide

## Overview
Configure cryptocurrency securities for analysis alongside traditional credit.

## Supported Exchanges
- Binance (recommended)
- Coinbase
- Kraken
- See full list: docs/SUPPORTED_EXCHANGES.md

## Configuration Example
[Include config.unified.yaml crypto section]

## Blockchain Metrics
- MVRV: Market Value to Realized Value
- NVT: Network Value to Transactions
- Active Addresses
```

**3. Configuration Reference** (`docs/CONFIGURATION.md`)

```markdown
# Configuration Reference

Complete reference for all configuration options.

## Securities Section
## Bloomberg Section
## Cross-Asset Features Section
## Model Section
## Database Section
```

### 9.2 Developer Documentation

**1. Architecture Documentation** (`docs/ARCHITECTURE.md`)
- Data flow diagrams
- Class hierarchy
- Extension points

**2. API Documentation** (`docs/API_REFERENCE.md`)
- All public classes and methods
- Usage examples
- Return types

**3. Contributing Guide** (`docs/CONTRIBUTING.md`)
- Code style (PEP 8)
- Testing requirements
- Pull request process

---

## 10. Risk Analysis & Mitigation

### 10.1 Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Bloomberg API quota limits | High | Medium | Implement caching, batch requests, monitor usage |
| Terminal connection failures | Medium | High | Excel fallback, retry logic, clear error messages |
| Data alignment issues (crypto 24/7 vs credit weekdays) | High | Medium | Robust date alignment, forward fill with limits |
| API breaking changes | Low | Low | Version pin blpapi, monitor Bloomberg announcements |
| Performance degradation with large portfolios | Medium | Medium | Lazy loading, pagination, database caching |

### 10.2 Data Quality Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Bloomberg #N/A errors | Medium | High | Automatic error value handling, data validation |
| Exchange API downtime | High | Medium | Multi-exchange fallback, cached data |
| Stale data in Excel files | Medium | Medium | Timestamp validation, freshness warnings |
| Outliers in crypto data (flash crashes) | High | Low | Outlier detection, z-score filtering |
| Missing blockchain metrics | Low | Medium | Graceful degradation, feature flagging |

### 10.3 Configuration Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Invalid security identifiers | Medium | Medium | Validation on load, clear error messages |
| Conflicting feature configurations | Low | Low | Schema validation, config linting |
| Missing API credentials | High | High | .env.example template, startup validation |
| Backwards incompatibility | Medium | Low | Migration script, deprecation warnings |

### 10.4 User Experience Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Complex configuration | High | High | Sensible defaults, configuration wizard |
| Unclear error messages | Medium | Medium | User-friendly errors, troubleshooting guide |
| Slow data loading | Medium | Medium | Progress bars, async loading, caching |
| Unexpected results | High | Low | Data validation, sanity checks, logging |

---

## 11. Success Metrics

### 11.1 Technical Metrics

- [ ] Bloomberg API success rate: > 95%
- [ ] Data loading time: < 10 seconds for typical portfolio
- [ ] Code coverage: > 80%
- [ ] API response time: < 500ms for predictions
- [ ] Zero data loss during source switching (API ↔ Excel)

### 11.2 Feature Metrics

- [ ] All Bloomberg fields supported (OAS, DTS, PRICE, YIELD)
- [ ] All CCXT exchanges accessible
- [ ] Cross-asset features improve model accuracy by > 5%
- [ ] Configuration migration succeeds for 100% of test cases

### 11.3 User Metrics

- [ ] Configuration time: < 10 minutes for new users
- [ ] Error resolution time: < 5 minutes with documentation
- [ ] User satisfaction: > 4/5 in feedback survey

---

## 12. Next Steps

### Immediate Actions (Before Implementation)

1. **Stakeholder Review**
   - Review this plan with project stakeholders
   - Confirm Bloomberg API access availability
   - Validate use cases for mixed portfolios

2. **Environment Setup**
   - Set up Bloomberg Terminal access
   - Obtain API credentials (Glassnode, exchanges)
   - Configure development database

3. **Proof of Concept**
   - Test Bloomberg API connection
   - Load sample Excel file
   - Verify crypto + credit data alignment

### Implementation Timeline

- **Week 1-2**: Phase 1 (Bloomberg API Foundation)
- **Week 2**: Phase 2 (Bloomberg Excel Enhancement)
- **Week 3**: Phase 3 (Unified Security Framework)
- **Week 4**: Phase 4 (Cross-Asset Features)
- **Week 5**: Phase 5 (UI/API Updates)
- **Week 6**: Phase 6 (Testing & Documentation)

**Total Duration:** 6 weeks

### Post-Implementation

1. **Beta Testing**
   - Internal testing with real data
   - User acceptance testing
   - Performance optimization

2. **Production Deployment**
   - Gradual rollout
   - Monitor error rates
   - Collect user feedback

3. **Iteration**
   - Address feedback
   - Add requested features
   - Optimize performance

---

## Appendices

### Appendix A: Bloomberg API Code Examples

See: `examples/bloomberg_api_usage.py`

### Appendix B: Configuration Migration Script

See: `scripts/migrate_config.py`

### Appendix C: Excel Template

See: `templates/bloomberg_credit_template.xlsx`

### Appendix D: Glossary

- **OAS**: Option-Adjusted Spread - credit spread measure
- **DTS**: Duration to Worst - interest rate sensitivity
- **MVRV**: Market Value to Realized Value - on-chain metric
- **NVT**: Network Value to Transactions - crypto valuation metric
- **VWAP**: Volume-Weighted Average Price

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-06 | Claude | Initial plan creation |

---

**Questions or feedback?** Open an issue or discuss in the project repository.
