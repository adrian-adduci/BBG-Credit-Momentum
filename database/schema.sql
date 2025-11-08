--
-- PostgreSQL Database Schema for BBG-Credit-Momentum Trading System
--
-- This schema defines tables for storing:
-- - Cryptocurrency OHLCV (Open, High, Low, Close, Volume) data
-- - Blockchain on-chain metrics (MVRV, NVT, active addresses, etc.)
-- - Machine learning model predictions and performance tracking
--
-- Security features:
-- - Unique constraints to prevent duplicate data
-- - Indexes for query performance
-- - Timestamps for audit trails
-- - Numeric precision for financial data
--
-- Author: BBG-Credit-Momentum Team
-- License: MIT
--

-- =============================================================================
-- Drop existing tables (be careful in production!)
-- =============================================================================

DROP TABLE IF EXISTS model_predictions CASCADE;
DROP TABLE IF EXISTS blockchain_metrics CASCADE;
DROP TABLE IF EXISTS crypto_ohlcv CASCADE;

-- =============================================================================
-- OHLCV Data Table
-- =============================================================================

CREATE TABLE crypto_ohlcv (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,              -- Trading pair (e.g., "BTC/USDT")
    exchange VARCHAR(20) NOT NULL,            -- Exchange name (e.g., "binance")
    timestamp TIMESTAMP NOT NULL,             -- Candle timestamp (UTC)
    open DECIMAL(20, 8) NOT NULL,            -- Opening price
    high DECIMAL(20, 8) NOT NULL,            -- High price
    low DECIMAL(20, 8) NOT NULL,             -- Low price
    close DECIMAL(20, 8) NOT NULL,           -- Closing price
    volume DECIMAL(20, 8) NOT NULL,          -- Trading volume
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- Record creation time

    -- Constraints
    CONSTRAINT crypto_ohlcv_unique UNIQUE(symbol, exchange, timestamp),
    CONSTRAINT crypto_ohlcv_high_gte_low CHECK (high >= low),
    CONSTRAINT crypto_ohlcv_volume_positive CHECK (volume >= 0)
);

-- Indexes for performance (queries often filter by symbol, exchange, and timestamp)
CREATE INDEX idx_crypto_ohlcv_symbol_exchange ON crypto_ohlcv(symbol, exchange);
CREATE INDEX idx_crypto_ohlcv_timestamp ON crypto_ohlcv(timestamp);
CREATE INDEX idx_crypto_ohlcv_symbol_exchange_timestamp ON crypto_ohlcv(symbol, exchange, timestamp DESC);

COMMENT ON TABLE crypto_ohlcv IS 'OHLCV candle data from cryptocurrency exchanges';
COMMENT ON COLUMN crypto_ohlcv.symbol IS 'Trading pair symbol (e.g., BTC/USDT, ETH/USDT)';
COMMENT ON COLUMN crypto_ohlcv.exchange IS 'Exchange identifier (e.g., binance, coinbase)';
COMMENT ON COLUMN crypto_ohlcv.timestamp IS 'Candle timestamp in UTC';
COMMENT ON COLUMN crypto_ohlcv.open IS 'Opening price of the candle';
COMMENT ON COLUMN crypto_ohlcv.high IS 'Highest price during the candle period';
COMMENT ON COLUMN crypto_ohlcv.low IS 'Lowest price during the candle period';
COMMENT ON COLUMN crypto_ohlcv.close IS 'Closing price of the candle';
COMMENT ON COLUMN crypto_ohlcv.volume IS 'Total trading volume during the period';

-- =============================================================================
-- Blockchain Metrics Table
-- =============================================================================

CREATE TABLE blockchain_metrics (
    id SERIAL PRIMARY KEY,
    asset VARCHAR(20) NOT NULL,               -- Asset symbol (e.g., "BTC", "ETH")
    metric_name VARCHAR(50) NOT NULL,         -- Metric identifier (e.g., "mvrv", "nvt")
    timestamp TIMESTAMP NOT NULL,             -- Metric timestamp (UTC)
    value DECIMAL(20, 8) NOT NULL,           -- Metric value
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- Record creation time

    -- Constraints
    CONSTRAINT blockchain_metrics_unique UNIQUE(asset, metric_name, timestamp)
);

-- Indexes for performance
CREATE INDEX idx_blockchain_metrics_asset_metric ON blockchain_metrics(asset, metric_name);
CREATE INDEX idx_blockchain_metrics_timestamp ON blockchain_metrics(timestamp);
CREATE INDEX idx_blockchain_metrics_asset_metric_timestamp ON blockchain_metrics(asset, metric_name, timestamp DESC);

COMMENT ON TABLE blockchain_metrics IS 'On-chain blockchain metrics from Glassnode, CoinMetrics, etc.';
COMMENT ON COLUMN blockchain_metrics.asset IS 'Asset symbol (e.g., BTC, ETH, SOL)';
COMMENT ON COLUMN blockchain_metrics.metric_name IS 'Metric identifier (e.g., mvrv, nvt, active_addresses)';
COMMENT ON COLUMN blockchain_metrics.timestamp IS 'Metric timestamp in UTC';
COMMENT ON COLUMN blockchain_metrics.value IS 'Metric value (interpretation depends on metric_name)';

-- =============================================================================
-- Model Predictions Table
-- =============================================================================

CREATE TABLE model_predictions (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(100) NOT NULL,           -- Model identifier (e.g., "XGBoost_20250105_143022")
    symbol VARCHAR(20) NOT NULL,              -- Trading pair (e.g., "BTC/USDT")
    prediction_timestamp TIMESTAMP NOT NULL,  -- When the prediction was made
    forecast_timestamp TIMESTAMP NOT NULL,    -- What time the prediction is for
    predicted_value DECIMAL(20, 8) NOT NULL, -- Predicted value
    actual_value DECIMAL(20, 8),             -- Actual value (populated later)
    mae DECIMAL(20, 8),                      -- Mean Absolute Error (populated later)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP  -- Record creation time
);

-- Indexes for performance
CREATE INDEX idx_model_predictions_model_id ON model_predictions(model_id);
CREATE INDEX idx_model_predictions_symbol ON model_predictions(symbol);
CREATE INDEX idx_model_predictions_forecast_timestamp ON model_predictions(forecast_timestamp);
CREATE INDEX idx_model_predictions_model_symbol ON model_predictions(model_id, symbol, forecast_timestamp DESC);

COMMENT ON TABLE model_predictions IS 'Machine learning model predictions and actual values for performance tracking';
COMMENT ON COLUMN model_predictions.model_id IS 'Unique model identifier (e.g., XGBoost_20250105_143022)';
COMMENT ON COLUMN model_predictions.symbol IS 'Trading pair symbol the prediction is for';
COMMENT ON COLUMN model_predictions.prediction_timestamp IS 'When the prediction was generated';
COMMENT ON COLUMN model_predictions.forecast_timestamp IS 'What future time the prediction is for';
COMMENT ON COLUMN model_predictions.predicted_value IS 'Value predicted by the model';
COMMENT ON COLUMN model_predictions.actual_value IS 'Actual observed value (NULL until data is available)';
COMMENT ON COLUMN model_predictions.mae IS 'Mean absolute error of prediction (NULL until actual value known)';

-- =============================================================================
-- Example Queries
-- =============================================================================

-- Get OHLCV data for BTC/USDT on Binance for the last 30 days
-- SELECT * FROM crypto_ohlcv
-- WHERE symbol = 'BTC/USDT'
--   AND exchange = 'binance'
--   AND timestamp >= CURRENT_TIMESTAMP - INTERVAL '30 days'
-- ORDER BY timestamp DESC;

-- Get MVRV ratio for Bitcoin
-- SELECT timestamp, value as mvrv
-- FROM blockchain_metrics
-- WHERE asset = 'BTC' AND metric_name = 'mvrv'
-- ORDER BY timestamp DESC
-- LIMIT 100;

-- Get model performance (predictions vs actuals)
-- SELECT
--     model_id,
--     symbol,
--     COUNT(*) as total_predictions,
--     AVG(ABS(predicted_value - actual_value)) as avg_mae,
--     AVG(ABS(predicted_value - actual_value) / actual_value * 100) as avg_mape_pct
-- FROM model_predictions
-- WHERE actual_value IS NOT NULL
-- GROUP BY model_id, symbol
-- ORDER BY avg_mae ASC;

-- Get most recent predictions that need actual values filled in
-- SELECT * FROM model_predictions
-- WHERE actual_value IS NULL
--   AND forecast_timestamp < CURRENT_TIMESTAMP
-- ORDER BY forecast_timestamp DESC;
