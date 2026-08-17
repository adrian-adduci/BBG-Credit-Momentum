"""
FastAPI REST API for Cryptocurrency Trading Strategy System

This module provides a comprehensive REST API for training ML models,
generating predictions, and backtesting trading strategies.

Endpoints:
    POST /api/train - Train model with historical data
    POST /api/predict - Generate predictions
    GET /api/strategies - List available strategies
    POST /api/backtest - Run strategy backtesting
    WebSocket /ws/signals - Real-time trading signals (template)

Example:
    # Start server
    uvicorn api:app --reload --host 0.0.0.0 --port 8000

    # Access Swagger docs
    http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime
import asyncio
import logging
from pathlib import Path

# Import project modules
import preprocessing
import models
from data_sources import DataSourceFactory
from config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Crypto Trading Strategy API",
    description="ML-powered cryptocurrency trading strategy analysis and backtesting",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration (configure for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

################################################################################
# Pydantic Models for Request/Response Validation
################################################################################

class TrainRequest(BaseModel):
    """Request model for training endpoint."""
    exchange: str = Field("binance", description="Exchange ID (binance, coinbase, kraken)")
    symbols: List[str] = Field(["BTC/USDT", "ETH/USDT"], description="Trading pairs")
    timeframe: str = Field("1h", description="Candle timeframe (1m, 5m, 1h, 1d)")
    start_date: str = Field("2023-01-01", description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD, defaults to now)")
    target_column: str = Field("BTC_USDT_close", description="Target to predict")
    model_type: str = Field("XGBoost", description="Model type (XGBoost, CART, AdaBoost)")
    momentum_windows: List[int] = Field([5, 10, 15], description="Short-term momentum windows")
    momentum_baseline: int = Field(30, description="Long-term baseline window")
    crypto_features: bool = Field(True, description="Enable crypto technical indicators")

    class Config:
        schema_extra = {
            "example": {
                "exchange": "binance",
                "symbols": ["BTC/USDT", "ETH/USDT"],
                "timeframe": "1h",
                "start_date": "2024-01-01",
                "target_column": "BTC_USDT_close",
                "model_type": "XGBoost",
                "crypto_features": True
            }
        }


class TrainResponse(BaseModel):
    """Response model for training endpoint."""
    success: bool
    model_id: str
    metrics: Dict[str, float]
    feature_importance: Optional[Dict[str, float]]
    training_time: float
    data_points: int


class PredictRequest(BaseModel):
    """Request model for prediction endpoint."""
    model_id: str = Field(..., description="Model ID from training")
    features: Dict[str, float] = Field(..., description="Feature values")

    class Config:
        schema_extra = {
            "example": {
                "model_id": "XGBoost_20250105_143022",
                "features": {
                    "BTC_USDT_close": 45000,
                    "BTC_USDT_volume": 1000000,
                    "BTC_USDT_close_5day_rolling_average": 0.02
                }
            }
        }


class PredictResponse(BaseModel):
    """Response model for prediction endpoint."""
    prediction: float
    timestamp: str


class Strategy(BaseModel):
    """Trading strategy information."""
    name: str
    description: str
    parameters: List[str]


class BacktestRequest(BaseModel):
    """Request model for backtesting endpoint."""
    model_id: str = Field(..., description="Model ID from training")
    strategy_type: str = Field("momentum", description="Strategy type (momentum, mean_reversion)")
    initial_capital: float = Field(10000.0, description="Starting capital (USD)")
    position_size: float = Field(0.1, description="Position size (fraction of capital)")
    transaction_cost: float = Field(0.001, description="Transaction cost (0.1%)")

    class Config:
        schema_extra = {
            "example": {
                "model_id": "XGBoost_20250105_143022",
                "strategy_type": "momentum",
                "initial_capital": 10000.0,
                "position_size": 0.1,
                "transaction_cost": 0.001
            }
        }


class BacktestResponse(BaseModel):
    """Response model for backtesting endpoint."""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    final_capital: float
    equity_curve: List[Dict[str, Any]]


################################################################################
# Global State (in production, use Redis or database)
################################################################################

MODELS: Dict[str, Dict] = {}

################################################################################
# API Endpoints
################################################################################

@app.get("/", tags=["Health"])
async def root():
    """
    Health check endpoint.

    Returns API status and version information.
    """
    return {
        "status": "online",
        "service": "Crypto Trading Strategy API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "train": "/api/train",
            "predict": "/api/predict",
            "backtest": "/api/backtest",
            "strategies": "/api/strategies",
            "models": "/api/models"
        }
    }


# ---------------------------------------------------------------------------
# Blocking handlers are declared `def`, not `async def`.
#
# FastAPI runs a coroutine handler on the event loop thread and a plain `def`
# handler in a threadpool. Every handler below performs synchronous work --
# ccxt HTTP calls, time.sleep in the exchange retry path, Excel reads, XGBoost
# fitting -- so declaring them `async def` pinned that work to the event loop.
# The server could then serve only one request at a time, and a slow upstream
# took the whole process down: while POST /api/train was retrying an
# unreachable exchange, GET / stopped answering as well.
#
# Only the WebSocket endpoint stays async, because it genuinely awaits.
# tests/test_api_concurrency.py guards this.
# ---------------------------------------------------------------------------


@app.post("/api/train", response_model=TrainResponse, tags=["Training"])
def train_model(request: TrainRequest):
    """
    Train a machine learning model on historical cryptocurrency data.

    This endpoint:
    1. Fetches historical data from the specified exchange
    2. Preprocesses data and creates momentum features
    3. Optionally adds technical indicators (RSI, MACD, Bollinger Bands, etc.)
    4. Trains the selected ML model
    5. Returns model ID, performance metrics, and feature importance

    The trained model is stored in memory and can be used for predictions
    and backtesting via its model_id.
    """
    try:
        start_time = datetime.now()
        logger.info(f"Training request received: {request.model_type} on {request.symbols}")

        # Parse dates
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        end_date = (
            datetime.strptime(request.end_date, "%Y-%m-%d")
            if request.end_date
            else datetime.now()
        )

        # Load data from crypto exchange
        try:
            source = DataSourceFactory.create(
                "crypto",
                exchange_id=request.exchange,
                symbols=request.symbols,
                timeframe=request.timeframe,
                start_date=start_date,
                end_date=end_date
            )
            df = source.load_data()
            logger.info(f"Loaded {len(df)} rows from {request.exchange}")
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load data from {request.exchange}: {str(e)}"
            )

        # Preprocess data
        try:
            # Convert symbol format for momentum list
            momentum_list = [
                f"{symbol.replace('/', '_')}_close" for symbol in request.symbols
            ]

            pipeline = preprocessing.BloombergPreprocessor(
                xlsx_file=df,
                target_col=request.target_column,
                momentum_list=momentum_list,
                momentum_X_days=request.momentum_windows,
                momentum_Y_days=request.momentum_baseline,
                crypto_features=request.crypto_features
            )
            logger.info("Preprocessing complete")
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Preprocessing failed: {str(e)}"
            )

        # Train model
        try:
            model = models.MomentumModel(
                pipeline=pipeline,
                model_name=request.model_type
            )
            logger.info(f"{request.model_type} training complete")
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Model training failed: {str(e)}"
            )

        # Get metrics
        mae, mse, rmse = model.get_mean_error_metrics()
        logger.info(f"Metrics - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")

        # Get feature importance (optional, may be slow)
        feature_importance = None
        try:
            model.predictive_power(forecast_range=30)
            feature_importance = model.get_features_of_importance(forecast_day=30)
        except Exception as e:
            logger.warning(f"Feature importance calculation failed: {e}")

        # Generate model ID and store
        model_id = f"{request.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        MODELS[model_id] = {
            "model": model,
            "pipeline": pipeline,
            "config": request.dict(),
            "trained_at": datetime.now().isoformat()
        }

        training_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Model {model_id} stored, training time: {training_time:.2f}s")

        return TrainResponse(
            success=True,
            model_id=model_id,
            metrics={
                "mae": float(mae),
                "mse": float(mse),
                "rmse": float(rmse)
            },
            feature_importance=feature_importance,
            training_time=training_time,
            data_points=len(df)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in train_model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest):
    """
    Generate predictions using a trained model.

    Provide feature values and get a price prediction.
    Feature names must match those used during training.
    """
    try:
        # Check if model exists
        if request.model_id not in MODELS:
            available_models = list(MODELS.keys())
            raise HTTPException(
                status_code=404,
                detail=f"Model '{request.model_id}' not found. Available models: {available_models}"
            )

        model_data = MODELS[request.model_id]
        model = model_data["model"]

        # Convert features to DataFrame
        features_df = pd.DataFrame([request.features])

        # Generate prediction
        try:
            prediction = model.model.predict(features_df)[0]
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Prediction failed. Check feature names match training data. Error: {str(e)}"
            )

        return PredictResponse(
            prediction=float(prediction),
            timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in predict: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/strategies", response_model=List[Strategy], tags=["Strategies"])
async def list_strategies():
    """
    List available trading strategies.

    Returns descriptions and parameters for each strategy type.
    """
    return [
        Strategy(
            name="momentum",
            description="Buy when short-term momentum > long-term, sell otherwise",
            parameters=["momentum_threshold", "holding_period"]
        ),
        Strategy(
            name="mean_reversion",
            description="Buy when price < Bollinger lower band, sell when > upper band",
            parameters=["bollinger_window", "num_std"]
        ),
        Strategy(
            name="ml_forecast",
            description="Buy when ML model predicts price increase > threshold",
            parameters=["forecast_threshold", "confidence_level"]
        )
    ]


@app.post("/api/backtest", response_model=BacktestResponse, tags=["Backtesting"])
def backtest_strategy(request: BacktestRequest):
    """
    Backtest a trading strategy on historical data.

    Simulates trading with the specified strategy and returns performance metrics.
    """
    try:
        # Check if model exists
        if request.model_id not in MODELS:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{request.model_id}' not found"
            )

        model_data = MODELS[request.model_id]
        model = model_data["model"]
        pipeline = model_data["pipeline"]

        # Get historical data with predictions
        df = pipeline.get_dataframe()
        predictions = model.get_preds()

        # Get target column
        target_col = model_data["config"]["target_column"]

        # Simulate trading strategy
        capital = request.initial_capital
        position = 0  # Current position size
        position_price = 0  # Entry price
        equity_curve = []
        trades = []

        for i in range(len(df) - 1):
            try:
                current_price = df.iloc[i][target_col]
                prediction = predictions[i] if i < len(predictions) else current_price

                # Strategy logic
                if request.strategy_type == "momentum":
                    # Buy signal: prediction > current price
                    if prediction > current_price and position == 0:
                        # Enter long position
                        position_size = (capital * request.position_size)
                        shares = position_size / current_price
                        position += shares
                        position_price = current_price
                        capital -= position_size * (1 + request.transaction_cost)
                        trades.append({
                            "type": "buy",
                            "price": current_price,
                            "shares": shares,
                            "date": str(df.iloc[i]["Dates"])
                        })

                    # Sell signal: prediction < current price
                    elif prediction < current_price and position > 0:
                        # Exit position
                        sell_value = position * current_price
                        capital += sell_value * (1 - request.transaction_cost)
                        trades.append({
                            "type": "sell",
                            "price": current_price,
                            "shares": position,
                            "date": str(df.iloc[i]["Dates"])
                        })
                        position = 0
                        position_price = 0

                # Calculate current equity
                equity = capital + (position * current_price if position > 0 else 0)
                equity_curve.append({
                    "date": str(df.iloc[i]["Dates"]),
                    "equity": float(equity),
                    "capital": float(capital),
                    "position_value": float(position * current_price if position > 0 else 0)
                })

            except Exception as e:
                logger.warning(f"Error at index {i}: {e}")
                continue

        # Close final position if open
        if position > 0:
            final_price = df.iloc[-1][target_col]
            capital += position * final_price * (1 - request.transaction_cost)

        # Calculate performance metrics
        total_return = (capital - request.initial_capital) / request.initial_capital

        # Sharpe ratio
        if len(equity_curve) > 1:
            equity_series = pd.Series([e["equity"] for e in equity_curve])
            returns = equity_series.pct_change().dropna()
            sharpe_ratio = (
                (returns.mean() / returns.std()) * np.sqrt(252)
                if returns.std() > 0 else 0
            )
        else:
            sharpe_ratio = 0

        # Max drawdown
        if len(equity_curve) > 1:
            equity_series = pd.Series([e["equity"] for e in equity_curve])
            running_max = equity_series.expanding().max()
            drawdown = (equity_series - running_max) / running_max
            max_drawdown = float(drawdown.min())
        else:
            max_drawdown = 0

        # Win rate
        buy_trades = [t for t in trades if t["type"] == "buy"]
        sell_trades = [t for t in trades if t["type"] == "sell"]
        if len(buy_trades) > 0 and len(sell_trades) > 0:
            profitable_trades = sum(
                1 for i in range(min(len(buy_trades), len(sell_trades)))
                if sell_trades[i]["price"] > buy_trades[i]["price"]
            )
            win_rate = profitable_trades / min(len(buy_trades), len(sell_trades))
        else:
            win_rate = 0

        logger.info(
            f"Backtest complete: Return={total_return:.2%}, "
            f"Sharpe={sharpe_ratio:.2f}, Drawdown={max_drawdown:.2%}"
        )

        return BacktestResponse(
            total_return=float(total_return),
            sharpe_ratio=float(sharpe_ratio),
            max_drawdown=float(max_drawdown),
            win_rate=float(win_rate),
            num_trades=len(trades),
            final_capital=float(capital),
            equity_curve=equity_curve[:100]  # Limit to 100 points for response size
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models", tags=["Models"])
async def list_models():
    """
    List all trained models currently in memory.

    Returns model IDs, types, and training timestamps.
    """
    models_info = []
    for model_id, model_data in MODELS.items():
        models_info.append({
            "model_id": model_id,
            "model_type": model_data["config"]["model_type"],
            "trained_at": model_data["trained_at"],
            "symbols": model_data["config"]["symbols"],
            "exchange": model_data["config"]["exchange"]
        })

    return {
        "count": len(models_info),
        "models": models_info
    }


@app.delete("/api/models/{model_id}", tags=["Models"])
async def delete_model(model_id: str):
    """
    Delete a trained model from memory.
    """
    if model_id not in MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found"
        )

    del MODELS[model_id]
    logger.info(f"Model {model_id} deleted")

    return {"success": True, "message": f"Model {model_id} deleted"}


################################################################################
# Mixed Portfolio Endpoints (Crypto + Credit)
################################################################################

class MixedPortfolioTrainRequest(BaseModel):
    """Request model for training with mixed crypto + credit portfolio."""

    # Crypto securities
    crypto_exchange: Optional[str] = Field(None, description="Crypto exchange (binance, coinbase)")
    crypto_symbols: Optional[List[str]] = Field(None, description="Crypto trading pairs")
    crypto_timeframe: str = Field("1h", description="Crypto data timeframe")

    # Bloomberg/Credit securities
    bloomberg_securities: Optional[List[str]] = Field(None, description="Bloomberg securities (e.g., LF98TRUU Index)")
    bloomberg_fields: List[str] = Field(["OAS"], description="Bloomberg fields to fetch")
    bloomberg_source: str = Field("excel", description="Bloomberg data source: 'api', 'excel', or 'hybrid'")
    bloomberg_excel_path: Optional[str] = Field(None, description="Path to Bloomberg Excel export")

    # Common parameters
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")
    target_column: str = Field(..., description="Target column to predict")
    model_type: str = Field("XGBoost", description="Model type")

    # Feature engineering
    crypto_features: bool = Field(True, description="Enable crypto technical indicators")
    cross_asset_features: bool = Field(True, description="Enable cross-asset features")
    momentum_windows: List[int] = Field([5, 10, 15], description="Momentum windows")
    momentum_baseline: int = Field(30, description="Momentum baseline")

    class Config:
        schema_extra = {
            "example": {
                "crypto_exchange": "binance",
                "crypto_symbols": ["BTC/USDT", "ETH/USDT"],
                "crypto_timeframe": "1h",
                "bloomberg_securities": ["LF98TRUU Index"],
                "bloomberg_fields": ["OAS", "DTS"],
                "bloomberg_source": "excel",
                "bloomberg_excel_path": "data/bloomberg_export.xlsx",
                "start_date": "2024-01-01",
                "target_column": "BTC_USDT_close",
                "model_type": "XGBoost",
                "crypto_features": True,
                "cross_asset_features": True
            }
        }


class CrossAssetAnalysisResponse(BaseModel):
    """Response model for cross-asset analysis."""
    success: bool
    correlations: Dict[str, float]
    regime: str
    divergence_signals: Dict[str, Any]
    flight_to_quality: float
    timestamp: str


@app.post("/api/mixed/train", response_model=TrainResponse)
def train_mixed_portfolio(request: MixedPortfolioTrainRequest):
    """
    Train model with mixed crypto + credit portfolio.

    This endpoint supports unified analysis of cryptocurrency and traditional
    credit securities in a single model. It:

    1. Loads data from multiple sources (crypto exchanges, Bloomberg API/Excel)
    2. Aligns data across different market hours (24/7 crypto vs weekday credit)
    3. Adds cross-asset features (correlations, regime detection, divergences)
    4. Trains unified ML model

    Returns:
        Model ID, metrics, and feature importance including cross-asset features
    """
    try:
        start_time = datetime.now()
        logger.info(f"Mixed portfolio training: {request.crypto_symbols} + {request.bloomberg_securities}")

        from data_sources import MixedPortfolioDataSource, Security

        # Parse dates
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        end_date = (
            datetime.strptime(request.end_date, "%Y-%m-%d")
            if request.end_date
            else datetime.now()
        )

        # Build the security universe.
        #
        # MixedPortfolioDataSource takes Security *definitions* and resolves
        # each through _load_security_data(); it does not accept pre-built
        # data-source objects. This endpoint previously passed sources=[...],
        # which is not a parameter it has, so every request raised TypeError
        # and returned 500. The endpoint has never worked.
        securities = []

        if request.crypto_symbols and request.crypto_exchange:
            for symbol in request.crypto_symbols:
                securities.append(
                    Security(
                        identifier=symbol,
                        security_type="crypto_spot",
                        source=request.crypto_exchange,
                        fields=["close"],
                        metadata={"timeframe": request.crypto_timeframe},
                    )
                )
            logger.info(
                f"Added {len(request.crypto_symbols)} crypto securities "
                f"from {request.crypto_exchange}"
            )

        if request.bloomberg_securities:
            # Each bloomberg_source mode maps onto a source type that
            # _load_security_data already dispatches on.
            if request.bloomberg_source == "api":
                source_type, metadata = "bloomberg", {}
            elif request.bloomberg_source == "excel":
                if not request.bloomberg_excel_path:
                    raise HTTPException(
                        status_code=400,
                        detail="bloomberg_excel_path required when bloomberg_source='excel'"
                    )
                source_type = "bloomberg_excel"
                metadata = {"file_path": request.bloomberg_excel_path}
            elif request.bloomberg_source == "hybrid":
                if not request.bloomberg_excel_path:
                    raise HTTPException(
                        status_code=400,
                        detail="bloomberg_excel_path required for hybrid mode fallback"
                    )
                source_type = "bloomberg"
                metadata = {"excel_fallback": request.bloomberg_excel_path}
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid bloomberg_source: {request.bloomberg_source}"
                )

            for identifier in request.bloomberg_securities:
                securities.append(
                    Security(
                        identifier=identifier,
                        security_type="credit_index",
                        source=source_type,
                        fields=request.bloomberg_fields,
                        metadata=dict(metadata),
                    )
                )
            logger.info(
                f"Added {len(request.bloomberg_securities)} Bloomberg securities "
                f"({request.bloomberg_source})"
            )

        if not securities:
            raise HTTPException(
                status_code=400,
                detail="At least one data source (crypto or Bloomberg) must be specified"
            )

        # Load and merge data
        mixed_source = MixedPortfolioDataSource(
            securities=securities,
            start_date=start_date,
            end_date=end_date,
            alignment_method="outer",  # Include all dates from all sources
            fill_method="ffill",
            fill_limit=5,
        )
        df = mixed_source.load_data()
        logger.info(f"Loaded mixed portfolio data: {len(df)} rows, {len(df.columns)} columns")

        # Preprocess with cross-asset features
        momentum_list = []
        if request.crypto_symbols:
            momentum_list.extend([f"{sym.replace('/', '_')}_close" for sym in request.crypto_symbols])
        if request.bloomberg_securities:
            # Assume bloomberg securities have OAS field
            momentum_list.extend([f"{sec.replace(' ', '_')}_OAS" for sec in request.bloomberg_securities])

        pipeline = preprocessing.BloombergPreprocessor(
            xlsx_file=df,
            target_col=request.target_column,
            momentum_list=momentum_list,
            momentum_X_days=request.momentum_windows,
            momentum_Y_days=request.momentum_baseline,
            crypto_features=request.crypto_features,
            cross_asset_features=request.cross_asset_features  # Enable cross-asset features
        )
        logger.info("Preprocessing complete with cross-asset features")

        # Train model
        model = models.MomentumModel(
            pipeline=pipeline,
            model_name=request.model_type
        )
        logger.info(f"{request.model_type} training complete")

        # Get metrics
        mae, mse, rmse = model.get_mean_error_metrics()

        # Get feature importance
        feature_importance = None
        try:
            model.predictive_power(forecast_range=30)
            feature_importance = model.get_features_of_importance(forecast_day=30)
        except Exception as e:
            logger.warning(f"Feature importance calculation failed: {e}")

        # Generate model ID
        model_id = f"mixed_{request.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        MODELS[model_id] = {
            "model": model,
            "pipeline": pipeline,
            "config": request.dict(),
            "trained_at": datetime.now().isoformat(),
            "type": "mixed_portfolio"
        }

        training_time = (datetime.now() - start_time).total_seconds()

        return TrainResponse(
            success=True,
            model_id=model_id,
            metrics={"mae": mae, "mse": mse, "rmse": rmse},
            feature_importance=feature_importance,
            training_time=training_time,
            data_points=len(df)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Mixed portfolio training failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/mixed/analysis/{model_id}", response_model=CrossAssetAnalysisResponse)
def get_cross_asset_analysis(model_id: str):
    """
    Get cross-asset analysis for a trained mixed portfolio model.

    Returns real-time cross-asset indicators including:
    - Crypto-credit correlations
    - Current market regime (risk-on/risk-off)
    - Divergence signals
    - Flight-to-quality indicators

    Args:
        model_id: ID of trained mixed portfolio model

    Returns:
        Cross-asset analysis metrics
    """
    if model_id not in MODELS:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    model_info = MODELS[model_id]

    if model_info.get("type") != "mixed_portfolio":
        raise HTTPException(
            status_code=400,
            detail=f"Model {model_id} is not a mixed portfolio model"
        )

    try:
        # Get the processed dataframe
        pipeline = model_info["pipeline"]
        df = pipeline.get_dataframe()

        # Extract cross-asset features (most recent values)
        latest_data = df.iloc[-1]

        # Find correlation columns
        corr_cols = [col for col in df.columns if col.startswith('corr_')]
        correlations = {col: float(latest_data[col]) for col in corr_cols if pd.notna(latest_data[col])}

        # Find regime columns
        regime_cols = [col for col in df.columns if 'regime' in col.lower()]
        current_regime = "neutral"
        if regime_cols:
            regime_val = latest_data[regime_cols[0]]
            if regime_val > 0.5:
                current_regime = "risk-on"
            elif regime_val < -0.5:
                current_regime = "risk-off"

        # Find divergence signals
        divergence_cols = [col for col in df.columns if 'divergence_signal' in col]
        divergence_signals = {
            col: bool(latest_data[col]) for col in divergence_cols if pd.notna(latest_data[col])
        }

        # Flight to quality
        ftq = 0.0
        if 'ftq_indicator' in df.columns:
            ftq = float(latest_data['ftq_indicator']) if pd.notna(latest_data['ftq_indicator']) else 0.0

        return CrossAssetAnalysisResponse(
            success=True,
            correlations=correlations,
            regime=current_regime,
            divergence_signals=divergence_signals,
            flight_to_quality=ftq,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Cross-asset analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


################################################################################
# WebSocket for Real-Time Signals (Template)
################################################################################

class ConnectionManager:
    """Manage WebSocket connections for real-time signal streaming."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Send message to all connected clients."""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {e}")


manager = ConnectionManager()


@app.websocket("/ws/signals")
async def websocket_signals(websocket: WebSocket):
    """
    WebSocket endpoint for real-time trading signals.

    Streams live predictions as new data arrives.

    Note: This is a template implementation. Full real-time streaming
    requires integration with CryptoWebSocketDataSource.
    """
    await manager.connect(websocket)
    try:
        while True:
            # TODO: Integrate with CryptoWebSocketDataSource for live data
            # For now, simulate with periodic updates
            await asyncio.sleep(5)

            signal = {
                "timestamp": datetime.now().isoformat(),
                "symbol": "BTC/USDT",
                "signal": "buy",  # or "sell", "hold"
                "confidence": 0.75,
                "price": 45000.0,
                "indicators": {
                    "rsi": 65,
                    "macd": 0.02,
                    "momentum": 0.15
                }
            }

            await manager.broadcast(signal)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


################################################################################
# Server Configuration
################################################################################

if __name__ == "__main__":
    import uvicorn

    # Run server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes (development only)
        log_level="info"
    )
