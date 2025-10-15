"""
Data Pipeline Orchestrator.

Main pipeline that coordinates data fetching, validation, storage,
and feature engineering for NSE market data.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from src.data_pipeline.data_storage import DataStorageManager
from src.data_pipeline.data_validator import MarketDataValidator
from src.data_pipeline.openbb_client import OpenBBDataFetcher
from src.feature_engineering.technical_indicators import TechnicalIndicators
from src.utils.helpers import load_config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class DataPipeline:
    """
    Main data pipeline orchestrator.

    Coordinates data fetching, validation, storage, and feature engineering
    with support for parallel processing and incremental updates.
    """

    def __init__(self, config_path: str = "config/data_sources.yaml"):
        """
        Initialize data pipeline with all components.

        Args:
            config_path: Path to data sources configuration file
        """
        logger.info("Initializing Data Pipeline...")

        # Load configuration
        self.config = load_config(config_path)

        # Initialize components
        openbb_config = self.config.get("openbb", {}).get("fetch_config", {})
        self.data_fetcher = OpenBBDataFetcher(config=openbb_config)

        duckdb_config = self.config.get("duckdb", {})
        db_path = duckdb_config.get("path", "data/trading_db.duckdb")
        self.storage = DataStorageManager(db_path=db_path)

        self.validator = MarketDataValidator()
        self.indicators = TechnicalIndicators()

        # Pipeline statistics
        self.stats = {
            "total_symbols": 0,
            "successful_fetches": 0,
            "failed_fetches": 0,
            "total_rows_fetched": 0,
            "total_rows_stored": 0,
            "validation_errors": 0,
            "start_time": None,
            "end_time": None,
        }

        logger.info("Data Pipeline initialized successfully")

    def initialize_database(self) -> None:
        """Initialize database schema if not exists."""
        logger.info("Initializing database schema...")
        self.storage.create_schema()
        logger.info("Database schema initialized")

    def fetch_and_store_historical(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        batch_size: int = 10,
        max_workers: int = 3,
        validate: bool = True,
        generate_features: bool = False,
    ) -> Dict[str, Any]:
        """
        Fetch historical data and store in database.

        Args:
            symbols: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            batch_size: Number of symbols per batch
            max_workers: Number of parallel workers
            validate: Whether to validate data
            generate_features: Whether to generate technical indicators

        Returns:
            Dictionary with pipeline statistics
        """
        self.stats["start_time"] = datetime.now()
        self.stats["total_symbols"] = len(symbols)

        logger.info(
            f"Starting historical data fetch for {len(symbols)} symbols "
            f"from {start_date} to {end_date}"
        )

        # Process symbols in batches
        batches = [
            symbols[i : i + batch_size] for i in range(0, len(symbols), batch_size)
        ]

        logger.info(f"Processing {len(batches)} batches with {max_workers} workers")

        all_results = []

        with tqdm(total=len(symbols), desc="Fetching data") as pbar:
            for batch in batches:
                batch_results = self._process_batch(
                    batch=batch,
                    start_date=start_date,
                    end_date=end_date,
                    validate=validate,
                    generate_features=generate_features,
                    max_workers=max_workers,
                )

                all_results.extend(batch_results)
                pbar.update(len(batch))

                # Small delay between batches to respect rate limits
                time.sleep(1)

        # Compile statistics
        self.stats["end_time"] = datetime.now()
        self.stats["duration_seconds"] = (
            self.stats["end_time"] - self.stats["start_time"]
        ).total_seconds()

        logger.info(
            f"Pipeline complete: {self.stats['successful_fetches']}/{self.stats['total_symbols']} "
            f"successful, {self.stats['total_rows_stored']} total rows stored"
        )

        return self.stats

    def _process_batch(
        self,
        batch: List[str],
        start_date: str,
        end_date: str,
        validate: bool,
        generate_features: bool,
        max_workers: int,
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of symbols with parallel execution.

        Args:
            batch: List of symbols in batch
            start_date: Start date
            end_date: End date
            validate: Whether to validate
            generate_features: Whether to generate features
            max_workers: Number of workers

        Returns:
            List of results for each symbol
        """
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._process_single_symbol,
                    symbol,
                    start_date,
                    end_date,
                    validate,
                    generate_features,
                ): symbol
                for symbol in batch
            }

            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    result = future.result()
                    results.append(result)

                    if result["success"]:
                        self.stats["successful_fetches"] += 1
                        self.stats["total_rows_fetched"] += result["rows_fetched"]
                        self.stats["total_rows_stored"] += result["rows_stored"]
                    else:
                        self.stats["failed_fetches"] += 1

                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    self.stats["failed_fetches"] += 1
                    results.append({
                        "symbol": symbol,
                        "success": False,
                        "error": str(e),
                    })

        return results

    def _process_single_symbol(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        validate: bool,
        generate_features: bool,
    ) -> Dict[str, Any]:
        """
        Process a single symbol through the complete pipeline.

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            validate: Whether to validate
            generate_features: Whether to generate features

        Returns:
            Result dictionary with statistics
        """
        result = {
            "symbol": symbol,
            "success": False,
            "rows_fetched": 0,
            "rows_stored": 0,
            "validation_status": None,
            "error": None,
        }

        try:
            # 1. Fetch data
            df = self.data_fetcher.fetch_equity_ohlcv(
                symbols=[symbol],
                start_date=start_date,
                end_date=end_date,
                interval="1d",
            )

            if df.empty:
                result["error"] = "No data returned"
                return result

            result["rows_fetched"] = len(df)

            # 2. Validate data
            if validate:
                df, validation_report = self.validator.validate_ohlcv(
                    df, fix_errors=True
                )
                result["validation_status"] = validation_report.get(
                    "data_quality", {}
                ).get("status", "unknown")

                if validation_report["errors"]:
                    self.stats["validation_errors"] += len(validation_report["errors"])

            # 3. Generate features (optional)
            if generate_features:
                df = self.indicators.generate_all_features(
                    df, include_basic=True, include_advanced=False
                )

            # 4. Store in database
            rows_stored = self.storage.insert_ohlcv(df, if_exists="append")
            result["rows_stored"] = rows_stored

            result["success"] = True

            logger.debug(
                f"Successfully processed {symbol}: {rows_stored} rows stored"
            )

        except Exception as e:
            logger.error(f"Failed to process {symbol}: {e}")
            result["error"] = str(e)

        return result

    def update_data(
        self,
        symbols: Optional[List[str]] = None,
        lookback_days: int = 7,
        batch_size: int = 10,
    ) -> Dict[str, Any]:
        """
        Incremental update: fetch only new data since last update.

        Args:
            symbols: List of symbols (None for all symbols in DB)
            lookback_days: Days to look back from latest date
            batch_size: Symbols per batch

        Returns:
            Update statistics
        """
        logger.info("Starting incremental data update...")

        # Get symbols from database if not provided
        if symbols is None:
            symbols = self.storage.get_available_symbols()
            logger.info(f"Updating all {len(symbols)} symbols in database")

        update_stats = {
            "symbols_updated": 0,
            "new_rows": 0,
            "symbols_skipped": 0,
        }

        end_date = datetime.now().strftime("%Y-%m-%d")

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i : i + batch_size]

            for symbol in batch:
                try:
                    # Get latest date for symbol
                    latest_date = self.storage.get_latest_date(symbol)

                    if latest_date:
                        # Calculate start date (latest + 1 day - lookback for safety)
                        latest_dt = datetime.strptime(latest_date, "%Y-%m-%d")
                        start_dt = latest_dt - timedelta(days=lookback_days)
                        start_date = start_dt.strftime("%Y-%m-%d")

                        # Check if update is needed
                        if latest_dt.date() >= datetime.now().date():
                            logger.debug(f"{symbol} is up to date")
                            update_stats["symbols_skipped"] += 1
                            continue
                    else:
                        # No data exists, fetch last 30 days
                        start_date = (datetime.now() - timedelta(days=30)).strftime(
                            "%Y-%m-%d"
                        )

                    # Fetch and store new data
                    result = self._process_single_symbol(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        validate=True,
                        generate_features=False,
                    )

                    if result["success"]:
                        update_stats["symbols_updated"] += 1
                        update_stats["new_rows"] += result["rows_stored"]

                except Exception as e:
                    logger.error(f"Failed to update {symbol}: {e}")
                    continue

            # Rate limiting between batches
            time.sleep(2)

        logger.info(
            f"Update complete: {update_stats['symbols_updated']} symbols updated, "
            f"{update_stats['new_rows']} new rows added"
        )

        return update_stats

    def backfill_missing_dates(
        self,
        symbol: str,
        max_gap_days: int = 5,
    ) -> int:
        """
        Identify and backfill missing trading dates for a symbol.

        Args:
            symbol: Stock symbol
            max_gap_days: Maximum gap to consider for backfilling

        Returns:
            Number of missing dates backfilled
        """
        logger.info(f"Checking for missing dates: {symbol}")

        # Get existing data
        df = self.storage.query_ohlcv(symbols=[symbol])

        if df.empty:
            logger.warning(f"No data found for {symbol}")
            return 0

        # Find missing dates
        missing_dates = self.validator.check_missing_dates(df, symbol)

        if not missing_dates:
            logger.info(f"No missing dates found for {symbol}")
            return 0

        logger.info(f"Found {len(missing_dates)} missing dates for {symbol}")

        # Group consecutive missing dates
        missing_ranges = self._group_consecutive_dates(missing_dates, max_gap_days)

        total_backfilled = 0

        for start_date, end_date in missing_ranges:
            try:
                logger.info(f"Backfilling {symbol} from {start_date} to {end_date}")

                result = self._process_single_symbol(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    validate=True,
                    generate_features=False,
                )

                if result["success"]:
                    total_backfilled += result["rows_stored"]

            except Exception as e:
                logger.error(f"Failed to backfill {symbol} {start_date}-{end_date}: {e}")
                continue

        logger.info(f"Backfilled {total_backfilled} missing dates for {symbol}")

        return total_backfilled

    @staticmethod
    def _group_consecutive_dates(
        dates: List[str], max_gap_days: int
    ) -> List[Tuple[str, str]]:
        """
        Group consecutive dates into ranges.

        Args:
            dates: List of date strings
            max_gap_days: Maximum gap to include in same range

        Returns:
            List of (start_date, end_date) tuples
        """
        if not dates:
            return []

        dates = sorted([datetime.strptime(d, "%Y-%m-%d") for d in dates])
        ranges = []
        start = dates[0]
        end = dates[0]

        for i in range(1, len(dates)):
            gap = (dates[i] - end).days

            if gap <= max_gap_days:
                end = dates[i]
            else:
                ranges.append((start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))
                start = dates[i]
                end = dates[i]

        # Add last range
        ranges.append((start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))

        return ranges

    def health_check(self) -> Dict[str, Any]:
        """
        Check pipeline health and data freshness.

        Returns:
            Health check report
        """
        logger.info("Running pipeline health check...")

        health_report = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": {},
            "data_quality": {},
            "issues": [],
        }

        try:
            # Check database
            db_stats = self.storage.get_database_stats()
            health_report["database"] = db_stats

            if db_stats.get("unique_symbols", 0) == 0:
                health_report["issues"].append("No symbols in database")
                health_report["status"] = "warning"

            # Check data freshness
            if "max_date" in db_stats:
                max_date = datetime.strptime(db_stats["max_date"], "%Y-%m-%d")
                days_old = (datetime.now() - max_date).days

                health_report["database"]["data_age_days"] = days_old

                if days_old > 7:
                    health_report["issues"].append(f"Data is {days_old} days old")
                    health_report["status"] = "warning"

            # Check data quality
            quality_report = self.storage.get_data_quality_report()
            health_report["data_quality"] = {
                "total_symbols": len(quality_report),
                "avg_records_per_symbol": quality_report["total_records"].mean()
                if not quality_report.empty
                else 0,
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_report["status"] = "error"
            health_report["issues"].append(str(e))

        logger.info(f"Health check complete: {health_report['status']}")

        return health_report

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        Get current pipeline statistics.

        Returns:
            Pipeline statistics
        """
        return self.stats.copy()

    def clear_cache(self) -> None:
        """Clear all caches."""
        self.data_fetcher.clear_cache()
        logger.info("Pipeline caches cleared")

