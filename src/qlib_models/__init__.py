"""
Qlib Models Module.

Microsoft Qlib integration for quantitative investment:
- Model training and management
- Factor library
- Backtesting with Qlib
- Strategy implementation
"""

from typing import List

__all__: List[str] = [
    "QlibModelTrainer",
    "AlphaSignalGenerator",
]

from src.qlib_models.model_trainer import QlibModelTrainer
from src.qlib_models.signal_generator import AlphaSignalGenerator

