"""
Data Sources Package

Blockchain and on-chain data providers for cryptocurrency analysis.

Author: BBG-Credit-Momentum Team
License: MIT
"""

from .blockchain_provider import BlockchainDataSource, GlassnodeProvider, CoinMetricsProvider

__all__ = [
    "BlockchainDataSource",
    "GlassnodeProvider",
    "CoinMetricsProvider",
]

__version__ = "1.0.0"
