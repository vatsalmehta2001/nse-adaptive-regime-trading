"""
Data Pipeline Module.

Handles data acquisition, validation, and storage for NSE market data:
- OpenBB Platform integration
- DuckDB storage management
- Data validation with Pandera
- Pipeline orchestration
- Data quality reporting and audit trails
"""

from typing import List

__all__: List[str] = [
    "OpenBBDataFetcher",
    "DataStorageManager",
    "MarketDataValidator",
    "DataQualityConfig",
    "DataQualityReporter",
    "DataPipeline",
]

# Import main classes
from src.data_pipeline.data_storage import DataStorageManager
from src.data_pipeline.data_validator import DataQualityConfig, MarketDataValidator
from src.data_pipeline.openbb_client import OpenBBDataFetcher
from src.data_pipeline.pipeline import DataPipeline
from src.data_pipeline.quality_reporter import DataQualityReporter

