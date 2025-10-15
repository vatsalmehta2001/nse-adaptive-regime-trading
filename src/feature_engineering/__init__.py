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
    "QlibAlpha158",
    "FeatureStore",
    "FactorAnalyzer",
    "RegimeFeatureEngineer",
]

# Import main classes
from src.feature_engineering.technical_indicators import TechnicalIndicators
from src.feature_engineering.qlib_factors import QlibAlpha158
from src.feature_engineering.feature_store import FeatureStore
from src.feature_engineering.factor_analysis import FactorAnalyzer
from src.feature_engineering.regime_features import RegimeFeatureEngineer

