--
-- Initial Database Migration
--
-- This migration creates the initial schema for the BBG-Credit-Momentum
-- trading system database. Run this script on a fresh PostgreSQL database.
--
-- Usage:
--   psql -U your_user -d crypto_trading -f 001_initial.sql
--
-- Or using Python:
--   import psycopg2
--   conn = psycopg2.connect(...)
--   with open('001_initial.sql', 'r') as f:
--       conn.cursor().execute(f.read())
--   conn.commit()
--
-- Author: BBG-Credit-Momentum Team
-- Date: 2025-01-05
-- Version: 1.0.0
--

BEGIN;

-- =============================================================================
-- Create tables
-- =============================================================================

CREATE TABLE IF NOT EXISTS crypto_ohlcv (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    exchange VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    open DECIMAL(20, 8) NOT NULL,
    high DECIMAL(20, 8) NOT NULL,
    low DECIMAL(20, 8) NOT NULL,
    close DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(20, 8) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT crypto_ohlcv_unique UNIQUE(symbol, exchange, timestamp),
    CONSTRAINT crypto_ohlcv_high_gte_low CHECK (high >= low),
    CONSTRAINT crypto_ohlcv_volume_positive CHECK (volume >= 0)
);

CREATE TABLE IF NOT EXISTS blockchain_metrics (
    id SERIAL PRIMARY KEY,
    asset VARCHAR(20) NOT NULL,
    metric_name VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    value DECIMAL(20, 8) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT blockchain_metrics_unique UNIQUE(asset, metric_name, timestamp)
);

CREATE TABLE IF NOT EXISTS model_predictions (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(100) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    prediction_timestamp TIMESTAMP NOT NULL,
    forecast_timestamp TIMESTAMP NOT NULL,
    predicted_value DECIMAL(20, 8) NOT NULL,
    actual_value DECIMAL(20, 8),
    mae DECIMAL(20, 8),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- Create indexes
-- =============================================================================

-- crypto_ohlcv indexes
CREATE INDEX IF NOT EXISTS idx_crypto_ohlcv_symbol_exchange
    ON crypto_ohlcv(symbol, exchange);
CREATE INDEX IF NOT EXISTS idx_crypto_ohlcv_timestamp
    ON crypto_ohlcv(timestamp);
CREATE INDEX IF NOT EXISTS idx_crypto_ohlcv_symbol_exchange_timestamp
    ON crypto_ohlcv(symbol, exchange, timestamp DESC);

-- blockchain_metrics indexes
CREATE INDEX IF NOT EXISTS idx_blockchain_metrics_asset_metric
    ON blockchain_metrics(asset, metric_name);
CREATE INDEX IF NOT EXISTS idx_blockchain_metrics_timestamp
    ON blockchain_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_blockchain_metrics_asset_metric_timestamp
    ON blockchain_metrics(asset, metric_name, timestamp DESC);

-- model_predictions indexes
CREATE INDEX IF NOT EXISTS idx_model_predictions_model_id
    ON model_predictions(model_id);
CREATE INDEX IF NOT EXISTS idx_model_predictions_symbol
    ON model_predictions(symbol);
CREATE INDEX IF NOT EXISTS idx_model_predictions_forecast_timestamp
    ON model_predictions(forecast_timestamp);
CREATE INDEX IF NOT EXISTS idx_model_predictions_model_symbol
    ON model_predictions(model_id, symbol, forecast_timestamp DESC);

-- =============================================================================
-- Add comments
-- =============================================================================

COMMENT ON TABLE crypto_ohlcv IS 'OHLCV candle data from cryptocurrency exchanges';
COMMENT ON TABLE blockchain_metrics IS 'On-chain blockchain metrics from Glassnode, CoinMetrics, etc.';
COMMENT ON TABLE model_predictions IS 'Machine learning model predictions and actual values for performance tracking';

-- =============================================================================
-- Create migration tracking table
-- =============================================================================

CREATE TABLE IF NOT EXISTS schema_migrations (
    id SERIAL PRIMARY KEY,
    version VARCHAR(50) NOT NULL UNIQUE,
    description TEXT,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Record this migration
INSERT INTO schema_migrations (version, description)
VALUES ('001', 'Initial schema creation with crypto_ohlcv, blockchain_metrics, and model_predictions tables')
ON CONFLICT (version) DO NOTHING;

COMMIT;

-- =============================================================================
-- Verification queries
-- =============================================================================

-- Verify tables were created
SELECT
    table_name,
    (SELECT COUNT(*) FROM information_schema.columns WHERE table_name = t.table_name) as column_count
FROM information_schema.tables t
WHERE table_schema = 'public'
  AND table_name IN ('crypto_ohlcv', 'blockchain_metrics', 'model_predictions', 'schema_migrations')
ORDER BY table_name;

-- Verify indexes were created
SELECT
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE schemaname = 'public'
  AND tablename IN ('crypto_ohlcv', 'blockchain_metrics', 'model_predictions')
ORDER BY tablename, indexname;

-- Show migration history
SELECT * FROM schema_migrations ORDER BY applied_at;
