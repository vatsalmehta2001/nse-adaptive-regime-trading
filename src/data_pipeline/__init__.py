"""
Data Pipeline Module.

Handles data acquisition, validation, and storage for NSE market data:
- OpenBB Platform integration
- DuckDB storage management
- Data validation with Pandera
- Pipeline orchestration
"""

from typing import List

__all__: List[str] = [
    "OpenBBDataFetcher",
    "DataStorageManager",
    "MarketDataValidator",
    "DataPipeline",
]

# Import main classes
from src.data_pipeline.data_storage import DataStorageManager
from src.data_pipeline.data_validator import MarketDataValidator
from src.data_pipeline.openbb_client import OpenBBDataFetcher
from src.data_pipeline.pipeline import DataPipeline

