"""
Technical Indicators Package

This package provides technical indicators for trading analysis.
All indicators are organized by category (stochastic, momentum) and
follow consistent interfaces with logging and error handling.

Modules:
    stochastic: Stochastic-based indicators (Stochastic Oscillator, Stochastic RSI, Williams %R)
    momentum: Momentum-based indicators (ROC, CCI, ATR, OBV)

Quick Usage:
    >>> from indicators import calculate_stochastic, calculate_roc
    >>> stoch_k, stoch_d = calculate_stochastic(high, low, close)
    >>> roc = calculate_roc(close, window=10)

Class-Based Usage:
    >>> from indicators.stochastic import StochasticIndicators
    >>> calculator = StochasticIndicators(debug=True)
    >>> stoch_k, stoch_d = calculator.stochastic_oscillator(high, low, close)

Author: BBG-Credit-Momentum Team
License: MIT
"""

from .stochastic import (
    StochasticIndicators,
    calculate_stochastic,
    calculate_stochastic_rsi,
    calculate_williams_r,
)
from .momentum import (
    MomentumIndicators,
    calculate_roc,
    calculate_cci,
    calculate_atr,
    calculate_obv,
)

__all__ = [
    # Classes
    "StochasticIndicators",
    "MomentumIndicators",
    # Stochastic functions
    "calculate_stochastic",
    "calculate_stochastic_rsi",
    "calculate_williams_r",
    # Momentum functions
    "calculate_roc",
    "calculate_cci",
    "calculate_atr",
    "calculate_obv",
]

__version__ = "1.0.0"
