# BBG-Credit-Momentum

**Machine Learning Decision Support System for Bloomberg Credit Analytics**

A Streamlit application that analyzes economic data to identify momentum drivers for credit trading using XGBoost and sklearn models.

[![CI](https://github.com/adrian-adduci/BBG-Credit-Momentum/actions/workflows/ci.yml/badge.svg)](https://github.com/adrian-adduci/BBG-Credit-Momentum/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Data Sources](#data-sources)
- [Model Details](#model-details)
- [Forecasting Methodology](#forecasting-methodology)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Recent Improvements](#recent-improvements)
- [Contributing](#contributing)
- [License](#license)

---

## Features

### Core Functionality
- **Interactive Streamlit Dashboard** - Real-time model training and visualization
- **XGBoost Time Series Forecasting** - Predict credit spreads 1-30 days ahead
- **Momentum Feature Engineering** - Automated rolling average calculations
- **Feature Importance Analysis** - Identify key momentum drivers over time
- **Predictive Power Scoring** - Rank features by predictive capability
- **Model Performance Metrics** - MAE, MSE, RMSE with visualizations
- **Walk-Forward Backtesting** - Expanding-window evaluation scored against a
  random-walk baseline, so error metrics are always reported as skill relative
  to doing nothing

### Data Sources
- **Excel Files** (Bloomberg exports) - Fully supported
- **CSV Files** - Fully supported
- **Bloomberg API** - Template provided (requires completion)
- **Extensible Architecture** - Easy to add custom data sources

### Configuration
- **YAML Configuration** - Centralized settings management
- **Environment Variables** - Secure credential storage
- **Flexible Parameters** - Customize models, features, and analysis

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/adrian-adduci/BBG-Credit-Momentum.git
cd BBG-Credit-Momentum

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application
streamlit run webapp.py
```

The dashboard will open in your browser at `http://localhost:8501`.

---

## Installation

### Prerequisites

- **Python 3.11** (verified on 3.11.15). The pinned `numpy==1.26.4` /
  `pandas==2.2.3` have no wheels for Python 3.13+, so newer interpreters will
  attempt slow source builds and usually fail.
- **pip** package manager
- **Bloomberg Terminal** (optional, for API integration)

> **Known dependency conflict:** `ppscore==1.3.0` declares `pandas<2`, which
> contradicts the pinned `pandas==2.2.3`. The constraint is stale — ppscore
> works correctly against pandas 2.2.3 — but a strict resolver will refuse the
> file or silently downgrade pandas. `ppscore` also imports `pkg_resources`,
> removed in setuptools 81+. See the install steps below for the workaround.

### Step 1: Clone Repository

```bash
git clone https://github.com/adrian-adduci/BBG-Credit-Momentum.git
cd BBG-Credit-Momentum
```

### Step 2: Create Virtual Environment (Recommended)

#### Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# ppscore must be installed separately: its legacy setup.py needs
# pkg_resources, which setuptools 81+ no longer ships.
grep -v '^ppscore' requirements.txt > /tmp/req.txt
pip install -r /tmp/req.txt "setuptools<81" pytest httpx
pip install --no-build-isolation ppscore==1.3.0
```

`pytest` and `httpx` are needed to run the test suite (`httpx` backs
`fastapi.testclient`) and are not listed in `requirements.txt`.

**Dependencies installed** (all updated to latest stable versions):
- `streamlit==1.39.0` - Web interface
- `pandas==2.2.3` - Data manipulation
- `numpy==1.26.4` - Numerical computing
- `scikit-learn==1.5.2` - ML algorithms
- `xgboost==2.1.2` - Gradient boosting
- `matplotlib==3.9.2` - Plotting
- `seaborn==0.13.2` - Statistical visualizations
- `plotly==5.24.1` - Interactive charts
- `ppscore==1.3.0` - Predictive power scoring
- Plus 6 more (see [requirements.txt](requirements.txt))

### Step 4: Configure Application (Optional)

```bash
# Copy configuration templates
cp config.example.yaml config.yaml
cp .env.example .env

# Edit with your settings
nano config.yaml  # or use your preferred editor
nano .env
```

---

## Usage

### Method 1: Web Interface (Recommended)

1. **Start the application:**
   ```bash
   streamlit run webapp.py
   ```

2. **Upload data:**
   - Click "Browse files" in the sidebar
   - Select your Excel file (Bloomberg export format)
   - Example file provided: `data/Economic_Data_2020_08_01.xlsx`

3. **Configure features:**
   - **Target Feature**: Column to predict (e.g., `LF98TRUU_Index_OAS`)
   - **Momentum Parameters**: Columns for rolling averages (comma-separated)
   - **Date Range**: Filter data to specific time period

4. **Load and train:**
   - Click "Load Data" to preprocess
   - Select model (XGBoost recommended)
   - Click "Train Model" to start training

5. **View results:**
   - **Historic Data**: Visualize trends over time
   - **Feature Importance & Model Analysis**:
     - Model metrics (MAE, MSE, RMSE)
     - Forecast vs actual comparison
     - Feature importance rankings
     - Predictive power scores

### Method 2: Python API

```python
from _preprocessing import _preprocess_xlsx
from _models import _build_model

# Load and preprocess data
pipeline = _preprocess_xlsx(
    xlsx_file="data/Economic_Data_2020_08_01.xlsx",
    target_col="LF98TRUU_Index_OAS",
    momentum_list=["LF98TRUU_Index_OAS", "LUACTRUU_Index_OAS"],
    momentum_X_days=[5, 10, 15],
    momentum_Y_days=30,
    horizon=5,             # label = target at t+5; the target itself is not a feature
    target_lags=[1, 2, 5], # opt in to past values of the target
)

# Train model
model = _build_model(pipeline, model_name="XGBoost")

# Get predictions and metrics
predictions = model._return_preds()
mae, mse, rmse = model._return_mean_error_metrics()

# Analyze feature importance
model.predictive_power(forecast_range=30)
model._feature_importance(forecast_range=30)
model._feature_importance_over_time(forecast_range=30)
```

### Method 3: Using Data Source Abstraction

```python
from _data_sources import DataSourceFactory
from _config import get_config

# Load configuration
config = get_config()

# Create data source (Excel, CSV, or Bloomberg API)
source = DataSourceFactory.create(
    "excel",
    file_path="data/Economic_Data_2020_08_01.xlsx"
)
df = source.load_data()

# Use with preprocessing pipeline
preprocessing_config = config.get_preprocessing_config()
pipeline = _preprocess_xlsx(df, **preprocessing_config)
```

---

## Configuration

### YAML Configuration (config.yaml)

Create from template:
```bash
cp config.example.yaml config.yaml
```

**Example configuration:**
```yaml
# Data source
data_source:
  type: "excel"
  excel:
    file_path: "data/Economic_Data_2020_08_01.xlsx"

# Model settings
model:
  type: "XGBoost"
  estimators: 1000
  random_state: 42

# Feature engineering
features:
  target: "LF98TRUU_Index_OAS"
  momentum_columns:
    - "LF98TRUU_Index_OAS"
    - "LUACTRUU_Index_OAS"
  momentum_short_windows: [5, 10, 15]
  momentum_long_window: 30

# Analysis settings
analysis:
  importance_threshold: 0.05
  max_forecast_days: 30
```

See [config.example.yaml](config.example.yaml) for all options.

### Environment Variables (.env)

Create from template:
```bash
cp .env.example .env
```

**Example .env file:**
```bash
DATA_SOURCE_TYPE=excel
DATA_FILE_PATH=data/Economic_Data_2020_08_01.xlsx
MODEL_TYPE=XGBoost
TARGET_COLUMN=LF98TRUU_Index_OAS
NUMEXPR_MAX_THREADS=16
```

Environment variables override YAML settings.

---

## Data Sources

### Excel Files (Current Method)

**Requirements:**
- File format: `.xlsx` or `.xls`
- Must contain a `Dates` column with valid dates
- Numeric columns for economic indicators

**Bloomberg Export Instructions:**
1. Open Bloomberg Terminal
2. Use Excel plugin to export data
3. Save to `data/` folder
4. Ensure date column is named "Dates"

**Example file structure:**
```
Dates          | LF98TRUU_Index_OAS | LUACTRUU_Index_OAS | ...
2020-01-01     | 123.45             | 234.56             | ...
2020-01-02     | 125.67             | 236.78             | ...
```

### CSV Files

```python
from _data_sources import CSVDataSource

source = CSVDataSource(
    file_path="data/economic_data.csv",
    date_column="Date"
)
df = source.load_data()
```

### Bloomberg API (Template Provided)

**Setup:**
```bash
pip install blpapi
```

**Implementation:**
1. Edit `_data_sources.py`
2. Complete the `BloombergAPIDataSource.load_data()` method
3. Configure in `config.yaml`:

```yaml
data_source:
  type: "bloomberg"
  bloomberg:
    securities:
      - "LF98TRUU Index"
      - "LUACTRUU Index"
    fields:
      - "OAS"
      - "PX_LAST"
    start_date: "2020-01-01"
    end_date: "2020-12-31"
```

See [CHANGES.md](CHANGES.md) for detailed Bloomberg API integration guide.

---

## Model Details

### Supported Models

- **XGBoost** (Recommended) - Gradient boosted trees
- **CART** - Decision tree regressor
- **AdaBoost** - Adaptive boosting classifier
- **Logistic Regression** - Linear classification
- **Quadratic Regression** - Polynomial regression
- **K-Nearest Neighbors** - Instance-based learning

### Feature Engineering

**Momentum Features:**
Creates a normalised momentum ratio — the short-window average expressed as a
deviation from the long-window baseline:

```
momentum = (short_term_avg - long_term_avg) / long_term_avg
```

- **Short-term windows**: 5, 10, 15 days (configurable)
- **Long-term baseline**: 30 days (configurable)

**Example:**
- Input column: `LF98TRUU_Index_OAS`
- Generated features:
  - `LF98TRUU_Index_OAS_5day_rolling_average`
  - `LF98TRUU_Index_OAS_10day_rolling_average`
  - `LF98TRUU_Index_OAS_15day_rolling_average`

> **Naming caveat:** these columns are named `..._rolling_average` but hold the
> momentum *ratio* above, not a rolling mean. The names are kept for backward
> compatibility with existing configs and saved models.

---

## Forecasting Methodology

Time series models are unusually easy to fool. This project builds the
supervised problem through `forecasting.make_supervised()`, which enforces
three rules that earlier versions of this code violated:

1. **Labels lead features.** The label for the row observed at time *t* is the
   target at *t + horizon*, via `shift(-horizon)`. A positive shift returns the
   value from *h* days **ago**, which trains the model to predict the past.
2. **The raw target is never a feature.** Past values are legitimate predictors
   but only through explicit, strictly positive `target_lags`. A lag of zero is
   rejected.
3. **Splits are chronological.** `time_ordered_split()` offers no `shuffle`
   argument, and every walk-forward fold drops `horizon` rows from the end of
   its training window so the last training labels cannot reach into the test
   window.

### Backtesting

Error metrics mean nothing without a benchmark. `backtest.py` runs an
expanding-window walk-forward evaluation and scores every fold against the
random walk — *"tomorrow's spread is today's spread"*:

```bash
python backtest.py data/Economic_Data_2020_08_01.xlsx \
    --target LF98TRUU_Index_OAS --mode both
```

`skill = 1 - model_RMSE / naive_RMSE`. Positive means the model beat the naive
forecast; zero or negative means it did not.

**Results on the bundled dataset** (1,995 daily observations, 2012-08 to
2020-07, 5 folds, target lags 1/2/5):

| Horizon | Model RMSE | Naive RMSE | Skill | Verdict |
|--------:|-----------:|-----------:|------:|---------|
| 1d  | 0.1131 | 0.0959 | −0.180 | No skill |
| 5d  | 0.3494 | 0.2787 | −0.254 | No skill |
| 10d | 0.5197 | 0.4138 | −0.256 | No skill |
| 30d | 0.7778 | 0.7135 | −0.090 | No skill |

**On this dataset the model does not beat a random walk at any horizon.** That
is the honest result, and it is reported here rather than hidden. For contrast,
the original contemporaneous setup — same-row labels with the target left in the
feature matrix — reported a test R² of **0.955**, which measured leakage rather
than forecasting ability. The corrected 30-day setup scores **−0.033**.

### Levels vs changes

`--mode level` predicts the future level; `--mode change` (the default)
predicts the future *difference*. Gradient-boosted trees average training leaf
values and therefore cannot extrapolate a drifting level at all, so a level
model loses to the naive forecast for reasons unrelated to market
predictability. On the bundled data this framing alone moves 1-day skill from
−3.00 to −0.18. Differencing first is the standard remedy and the reason
`change` is the default.

### Evaluation Metrics

- **MAE** (Mean Absolute Error) - Average prediction error magnitude
- **MSE** (Mean Squared Error) - Squared error (penalizes large errors)
- **RMSE** (Root Mean Squared Error) - Error in original units
- **Predictive Power Score** - Feature predictive capability (0-1)
- **Feature Importance** - XGBoost tree-based importance scores
- **Cross-Validation** - 5-fold time series split validation

---

## Project Structure

```
BBG-Credit-Momentum/
├── webapp.py                  # Main Streamlit application
├── backtest.py                # Walk-forward backtest CLI vs random walk
├── forecasting.py             # Leakage-safe supervised-problem primitives
├── logging_setup.py           # Shared logger factory (creates logs/ on demand)
├── _preprocessing.py          # Data preprocessing pipeline
├── _models.py                 # Model training and analysis
├── _data_sources.py          # Data source abstraction layer
├── _config.py                # Configuration management
│
├── data/                     # Data files
│   └── Economic_Data_2020_08_01.xlsx
│
├── _img/                     # Generated visualizations
│   ├── arrow_logo.png
│   ├── predictive_power.png
│   ├── feats_importance.png
│   └── feats_importance_over_time.png
│
├── logs/                     # Application logs
│   ├── _main.log
│   ├── _model.log
│   ├── _preprocess.log
│   └── _data_sources.log
│
├── .streamlit/               # Streamlit configuration
│   └── config.toml
│
├── config.yaml               # Application configuration (create from example)
├── config.example.yaml       # Configuration template
├── .env                      # Environment variables (create from example)
├── .env.example              # Environment template
│
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── CHANGES.md                # Detailed changelog
└── LICENSE                   # MIT License
```

---

## Example Outputs

### Historical Data & Forecasts
![Historical Data](https://github.com/adrian-adduci/Bloomberg_Predictive_Modelling/blob/3c1415df764e103f68a542d6cbb434d1b9b71661/_img/example_forecast.PNG)

### Feature Importance Over Time
![Feature Importance Over Time](https://github.com/adrian-adduci/Bloomberg_Predictive_Modelling/blob/3c1415df764e103f68a542d6cbb434d1b9b71661/_img/feats_importance_over_time.png)

### Predictive Power Analysis
![Predictive Power](https://github.com/adrian-adduci/Bloomberg_Predictive_Modelling/blob/3c1415df764e103f68a542d6cbb434d1b9b71661/_img/predictive_power.png)


## Troubleshooting

### Common Issues

**"ModuleNotFoundError: No module named 'streamlit'"**
```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

**"FileNotFoundError: Excel file not found"**
```bash
# Solution: Check file path
# - Ensure file is in data/ folder
# - Use absolute path or correct relative path
# - Verify file extension (.xlsx or .xls)
```

**"ValueError: Target column not found"**
```bash
# Solution: Check column name
# - Open Excel file and verify column name
# - Ensure exact spelling (case-sensitive)
# - Check for leading/trailing spaces
```

**"Unable to open logs/_main.log"**

Fixed. `logging_setup.get_logger()` creates the directory on demand and falls
back to a null handler if the filesystem is read-only, so logging can no longer
stop the application from starting.

**Streamlit shows "Please select a file"**
```bash
# Solution: Upload file or check file_buffer
# - Use file uploader in sidebar
# - Verify file is uploaded successfully
```

### Performance Tips

- **Large datasets**: Increase `NUMEXPR_MAX_THREADS` in `.env`
- **Memory issues**: Reduce `momentum_X_days` windows
- **Slow training**: Reduce `estimators` in XGBoost config
- **Disk space**: Clear `_img/` folder periodically

---

## Testing

### Manual Testing

```bash
# Test with example data
streamlit run webapp.py
# Upload: data/Economic_Data_2020_08_01.xlsx
# Target: LF98TRUU_Index_OAS
# Momentum: LF98TRUU_Index_OAS,LUACTRUU_Index_OAS
```

### Python Testing

```python
# Test preprocessing
from _preprocessing import _preprocess_xlsx

pipeline = _preprocess_xlsx(
    "data/Economic_Data_2020_08_01.xlsx",
    "LF98TRUU_Index_OAS"
)
print(f"Loaded {len(pipeline._return_dataframe())} rows")

# Test model training
from _models import _build_model

model = _build_model(pipeline)
mae, mse, rmse = model._return_mean_error_metrics()
print(f"RMSE: {rmse:.4f}")
```

### Continuous Integration

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every push to
`master` and on every pull request. Beyond the test suite it checks four things
that were previously only ever found by hand:

| Step | Guards against |
|---|---|
| Install dependencies | `requirements.txt` not being installable as written |
| Verify the environment | a dependency set that resolves but does not import, and ppscore silently downgrading pandas |
| Import on a clean clone | modules that crash on import because `logs/` does not exist |
| Assert the tree is clean | the test suite overwriting tracked chart assets in `_img/` |
| Smoke-test the backtest CLI | a change that breaks the CLI or quietly turns the no-skill result positive |

### Unit Tests

```bash
python -m pytest tests/ -q
```

The suite covers the indicator library, the cross-asset features, the API
surface, and — most importantly — the leakage regressions described under
[Forecasting Methodology](#forecasting-methodology). `tests/test_forecasting.py`
and `tests/test_pipeline_leakage.py` exist specifically so the defects fixed
there cannot come back silently.

---

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/BBG-Credit-Momentum.git
cd BBG-Credit-Momentum

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Make changes and test
streamlit run webapp.py

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Author

**Adrian Adduci**
- Email: FAA2160@columbia.edu
- GitHub: [@adrian-adduci](https://github.com/adrian-adduci)

---

## Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [pandas Documentation](https://pandas.pydata.org/docs/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Bloomberg API Documentation](https://www.bloomberg.com/professional/support/api-library/)

