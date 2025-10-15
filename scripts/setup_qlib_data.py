#!/usr/bin/env python
"""
Setup Data Pipeline Script.

Initialize the data pipeline and fetch historical data for NSE stocks.
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.pipeline import DataPipeline
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def setup_data_pipeline(
    symbols: str = "NIFTY50",
    years: int = 2,
    start_date: str = None,
    end_date: str = None,
) -> None:
    """
    Initialize data pipeline and fetch historical data.

    Args:
        symbols: 'NIFTY50' or comma-separated symbol list
        years: Number of years of historical data
        start_date: Start date (YYYY-MM-DD), overrides years
        end_date: End date (YYYY-MM-DD)
    """
    logger.info("=" * 80)
    logger.info("NSE DATA PIPELINE SETUP")
    logger.info("=" * 80)

    try:
        # Initialize pipeline
        pipeline = DataPipeline()

        # Initialize database schema
        logger.info("Creating database schema...")
        pipeline.initialize_database()

        # Get symbol list
        if symbols.upper() == "NIFTY50":
            symbol_list = pipeline.data_fetcher.get_nifty50_constituents()
            logger.info(f"Using NIFTY 50 constituents: {len(symbol_list)} symbols")
        else:
            symbol_list = [s.strip() for s in symbols.split(",")]
            logger.info(f"Using custom symbol list: {len(symbol_list)} symbols")

        # Calculate date range
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        if start_date is None:
            start_dt = datetime.now() - timedelta(days=years * 365)
            start_date = start_dt.strftime("%Y-%m-%d")

        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(f"Total symbols: {len(symbol_list)}")
        logger.info("-" * 80)

        # Fetch and store data
        logger.info("Starting data fetch...")
        stats = pipeline.fetch_and_store_historical(
            symbols=symbol_list,
            start_date=start_date,
            end_date=end_date,
            batch_size=10,
            max_workers=3,
            validate=True,
            generate_features=False,
        )

        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("DATA PIPELINE SETUP COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total symbols processed: {stats['total_symbols']}")
        logger.info(f"Successful fetches: {stats['successful_fetches']}")
        logger.info(f"Failed fetches: {stats['failed_fetches']}")
        logger.info(f"Total rows fetched: {stats['total_rows_fetched']}")
        logger.info(f"Total rows stored: {stats['total_rows_stored']}")
        logger.info(f"Duration: {stats['duration_seconds']:.1f} seconds")

        # Generate data quality report
        logger.info("\nGenerating data quality report...")
        quality_report = pipeline.storage.get_data_quality_report()

        logger.info(f"Symbols in database: {len(quality_report)}")
        logger.info(f"Average records per symbol: {quality_report['total_records'].mean():.0f}")

        # Database statistics
        db_stats = pipeline.storage.get_database_stats()
        logger.info(f"\nDatabase size: {db_stats.get('db_size_mb', 0):.2f} MB")
        logger.info(f"Date range: {db_stats.get('min_date', 'N/A')} to {db_stats.get('max_date', 'N/A')}")

        logger.info("\n" + "=" * 80)
        logger.info(" Setup complete! Data pipeline is ready.")
        logger.info("=" * 80)

    except KeyboardInterrupt:
        logger.info("\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Setup NSE data pipeline and fetch historical data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch 2 years of NIFTY 50 data
  python scripts/setup_data_pipeline.py --symbols NIFTY50 --years 2

  # Fetch custom symbols
  python scripts/setup_data_pipeline.py --symbols "RELIANCE,TCS,HDFCBANK" --years 1

  # Fetch specific date range
  python scripts/setup_data_pipeline.py --symbols NIFTY50 --start-date 2022-01-01 --end-date 2024-12-31
        """,
    )

    parser.add_argument(
        "--symbols",
        type=str,
        default="NIFTY50",
        help="'NIFTY50' or comma-separated symbol list (default: NIFTY50)",
    )

    parser.add_argument(
        "--years",
        type=int,
        default=2,
        help="Number of years of historical data (default: 2)",
    )

    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD), overrides --years",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD), defaults to today",
    )

    args = parser.parse_args()

    setup_data_pipeline(
        symbols=args.symbols,
        years=args.years,
        start_date=args.start_date,
        end_date=args.end_date,
    )


if __name__ == "__main__":
    main()

