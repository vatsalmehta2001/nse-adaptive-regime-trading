"""
Data Pipeline Module.

Handles data acquisition, cleaning, and storage from multiple sources:
- OpenBB Platform
- Zerodha Kite Connect
- Alternative data providers
"""

from typing import List

__all__: List[str] = [
    "DataFetcher",
    "DataCleaner",
    "DataStorage",
]

# Import main classes when they are implemented
# from src.data_pipeline.data_fetcher import DataFetcher
# from src.data_pipeline.data_cleaner import DataCleaner
# from src.data_pipeline.data_storage import DataStorage

