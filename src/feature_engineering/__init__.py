"""
Feature Engineering Module.

Implements feature generation including:
- Technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Qlib Alpha-158 factors (to be implemented)
- Custom features
- Regime-specific features
"""

from typing import List

__all__: List[str] = [
    "TechnicalIndicators",
]

# Import main classes
from src.feature_engineering.technical_indicators import TechnicalIndicators

