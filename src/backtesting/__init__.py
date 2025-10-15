"""
Backtesting Module.

Historical strategy evaluation:
- Vectorized backtesting engine
- Transaction cost modeling
- Slippage simulation
- Performance analytics
- Walk-forward analysis
"""

from typing import List

__all__: List[str] = [
    "BacktestEngine",
    "PerformanceAnalyzer",
]

from src.backtesting.backtest_engine import BacktestEngine
from src.backtesting.performance_analyzer import PerformanceAnalyzer

