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
import _preprocessing
import _models
from _data_sources import DataSourceFactory
from _config import get_config

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


@app.post("/api/train", response_model=TrainResponse, tags=["Training"])
async def train_model(request: TrainRequest):
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

            pipeline = _preprocessing._preprocess_xlsx(
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
            model = _models._build_model(
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
        mae, mse, rmse = model._return_mean_error_metrics()
        logger.info(f"Metrics - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")

        # Get feature importance (optional, may be slow)
        feature_importance = None
        try:
            model.predictive_power(forecast_range=30)
            feature_importance = model._return_features_of_importance(forecast_day=30)
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
async def predict(request: PredictRequest):
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
async def backtest_strategy(request: BacktestRequest):
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
        df = pipeline._return_dataframe()
        predictions = model._return_preds()

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
