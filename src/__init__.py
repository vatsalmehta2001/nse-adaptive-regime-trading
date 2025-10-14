"""
NSE Adaptive Regime Trading System.

A production-grade algorithmic trading system combining:
- OpenBB Platform for market data
- Microsoft Qlib for quantitative research
- TensorTrade for reinforcement learning
- Zerodha Kite Connect for live execution
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Module imports for convenience
from src.utils.logging_config import get_logger

logger = get_logger(__name__)
logger.info(f"NSE Adaptive Regime Trading System v{__version__} initialized")

