"""
DuckDB Storage Manager for Market Data.

High-performance OLAP storage for time-series market data using DuckDB.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import duckdb
import pandas as pd

from src.utils.database import DatabaseManager
from src.utils.helpers import ensure_dir
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class DataStorageManager:
    """
    DuckDB storage manager optimized for financial time-series data.

    Provides efficient storage and retrieval of OHLCV data, fundamentals,
    and corporate actions with proper indexing and ACID compliance.
    """

    def __init__(self, db_path: str = "data/trading_db.duckdb"):
        """
        Initialize DuckDB storage manager.

        Args:
            db_path: Path to DuckDB database file
        """
        self.db_path = Path(db_path)
        ensure_dir(self.db_path.parent)

        self.db_manager = DatabaseManager()
        self.connection = self.db_manager.connection

        logger.info(f"Data Storage Manager initialized: {self.db_path}")

    def create_schema(self) -> None:
        """Create all required tables with proper schema and indexes."""
        logger.info("Creating database schema...")

        # OHLCV table with proper types and constraints
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                symbol VARCHAR NOT NULL,
                date DATE NOT NULL,
                open DECIMAL(18,4) NOT NULL,
                high DECIMAL(18,4) NOT NULL,
                low DECIMAL(18,4) NOT NULL,
                close DECIMAL(18,4) NOT NULL,
                volume BIGINT NOT NULL,
                adj_close DECIMAL(18,4),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, date)
            )
        """)

        # Create indexes for fast queries
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol ON ohlcv(symbol)"
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_ohlcv_date ON ohlcv(date)"
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_date ON ohlcv(symbol, date)"
        )

        # Fundamentals table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS fundamentals (
                symbol VARCHAR NOT NULL,
                date DATE NOT NULL,
                market_cap DECIMAL(20,2),
                pe_ratio DECIMAL(10,2),
                pb_ratio DECIMAL(10,2),
                dividend_yield DECIMAL(8,4),
                eps DECIMAL(10,2),
                book_value DECIMAL(10,2),
                sector VARCHAR,
                industry VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, date)
            )
        """)

        # Corporate actions table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS corporate_actions (
                id INTEGER PRIMARY KEY,
                symbol VARCHAR NOT NULL,
                date DATE NOT NULL,
                action_type VARCHAR NOT NULL,  -- 'split', 'dividend', 'bonus'
                ratio DECIMAL(10,4),
                amount DECIMAL(10,2),
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_corp_actions_symbol "
            "ON corporate_actions(symbol)"
        )

        # Market metadata table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS market_metadata (
                symbol VARCHAR PRIMARY KEY,
                company_name VARCHAR,
                sector VARCHAR,
                industry VARCHAR,
                exchange VARCHAR DEFAULT 'NSE',
                isin VARCHAR,
                first_data_date DATE,
                last_data_date DATE,
                total_records INTEGER,
                last_updated TIMESTAMP,
                is_active BOOLEAN DEFAULT true
            )
        """)

        # Data quality logs table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS data_quality_logs (
                id INTEGER PRIMARY KEY,
                symbol VARCHAR,
                check_date DATE,
                check_type VARCHAR,
                issue_type VARCHAR,
                issue_count INTEGER,
                details TEXT,
                severity VARCHAR,  -- 'info', 'warning', 'error'
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        logger.info("Database schema created successfully")

    def insert_ohlcv(
        self,
        df: pd.DataFrame,
        if_exists: str = "append",
        batch_size: int = 1000,
    ) -> int:
        """
        Insert OHLCV data with automatic deduplication.

        Args:
            df: DataFrame with OHLCV data
            if_exists: 'append' or 'replace'
            batch_size: Batch size for insertion

        Returns:
            Number of rows inserted

        Raises:
            ValueError: If DataFrame format is invalid
        """
        if df.empty:
            logger.warning("Empty DataFrame provided, skipping insert")
            return 0

        # Validate required columns
        required_cols = ["symbol", "date", "open", "high", "low", "close", "volume"]
        missing_cols = set(required_cols) - set(df.columns)

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Prepare data
        df = df.copy()

        # Ensure date is in proper format
        df["date"] = pd.to_datetime(df["date"]).dt.date

        # Add adj_close if missing
        if "adj_close" not in df.columns:
            df["adj_close"] = df["close"]

        # Add timestamps
        current_time = datetime.now()
        df["created_at"] = current_time
        df["updated_at"] = current_time

        # Select only columns that exist in table
        columns = [
            "symbol",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "adj_close",
            "created_at",
            "updated_at",
        ]
        df = df[columns]

        initial_rows = len(df)
        logger.info(f"Inserting {initial_rows} rows into OHLCV table...")

        if if_exists == "replace":
            # Delete existing data for these symbols and date range
            symbols = df["symbol"].unique().tolist()
            symbol_list = ", ".join([f"'{s}'" for s in symbols])

            self.connection.execute(f"""
                DELETE FROM ohlcv 
                WHERE symbol IN ({symbol_list})
                AND date >= '{df["date"].min()}'
                AND date <= '{df["date"].max()}'
            """)

        # Use DuckDB's efficient insertion via temp view
        self.connection.register("temp_ohlcv", df)

        # Insert with ON CONFLICT handling (upsert)
        self.connection.execute("""
            INSERT INTO ohlcv 
            SELECT * FROM temp_ohlcv
            ON CONFLICT (symbol, date) 
            DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                adj_close = EXCLUDED.adj_close,
                updated_at = EXCLUDED.updated_at
        """)

        self.connection.unregister("temp_ohlcv")

        # Update metadata
        self._update_metadata(df)

        logger.info(f"Successfully inserted/updated {initial_rows} rows")

        return initial_rows

    def _update_metadata(self, df: pd.DataFrame) -> None:
        """
        Update market metadata table with latest information.

        Args:
            df: DataFrame with market data
        """
        for symbol in df["symbol"].unique():
            symbol_data = df[df["symbol"] == symbol]

            # Get existing metadata
            existing = self.connection.execute(
                "SELECT * FROM market_metadata WHERE symbol = ?", (symbol,)
            ).fetchone()

            first_date = symbol_data["date"].min()
            last_date = symbol_data["date"].max()
            record_count = len(symbol_data)

            if existing:
                # Update existing metadata
                self.connection.execute(
                    """
                    UPDATE market_metadata
                    SET last_data_date = ?,
                        total_records = total_records + ?,
                        last_updated = CURRENT_TIMESTAMP
                    WHERE symbol = ?
                """,
                    (last_date, record_count, symbol),
                )
            else:
                # Insert new metadata
                self.connection.execute(
                    """
                    INSERT INTO market_metadata
                    (symbol, first_data_date, last_data_date, total_records, last_updated)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                    (symbol, first_date, last_date, record_count),
                )

    def query_ohlcv(
        self,
        symbols: Optional[Union[str, List[str]]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Query OHLCV data efficiently.

        Args:
            symbols: Symbol or list of symbols (None for all)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            limit: Maximum rows to return

        Returns:
            DataFrame with OHLCV data
        """
        query = "SELECT * FROM ohlcv WHERE 1=1"
        params = []

        # Add symbol filter
        if symbols is not None:
            if isinstance(symbols, str):
                symbols = [symbols]

            placeholders = ",".join(["?" for _ in symbols])
            query += f" AND symbol IN ({placeholders})"
            params.extend(symbols)

        # Add date filters
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        # Add ordering
        query += " ORDER BY symbol, date"

        # Add limit
        if limit:
            query += f" LIMIT {limit}"

        logger.debug(f"Executing query: {query[:100]}...")

        result = self.connection.execute(query, params)
        df = result.df()

        logger.info(f"Retrieved {len(df)} rows from OHLCV table")

        return df

    def get_latest_date(self, symbol: str) -> Optional[str]:
        """
        Get latest available date for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Latest date (YYYY-MM-DD) or None if no data
        """
        result = self.connection.execute(
            "SELECT MAX(date) as max_date FROM ohlcv WHERE symbol = ?", (symbol,)
        ).fetchone()

        if result and result[0]:
            return str(result[0])

        return None

    def get_available_symbols(self) -> List[str]:
        """
        Get list of all symbols in database.

        Returns:
            List of symbols
        """
        result = self.connection.execute(
            "SELECT DISTINCT symbol FROM ohlcv ORDER BY symbol"
        ).fetchall()

        return [row[0] for row in result]

    def get_data_coverage(self, symbol: str) -> Dict[str, Any]:
        """
        Get data coverage statistics for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with coverage stats
        """
        query = """
            SELECT 
                MIN(date) as first_date,
                MAX(date) as last_date,
                COUNT(*) as total_records,
                COUNT(DISTINCT date) as unique_dates
            FROM ohlcv
            WHERE symbol = ?
        """

        result = self.connection.execute(query, (symbol,)).fetchone()

        if not result or not result[0]:
            return {"symbol": symbol, "has_data": False}

        return {
            "symbol": symbol,
            "has_data": True,
            "first_date": str(result[0]),
            "last_date": str(result[1]),
            "total_records": result[2],
            "unique_dates": result[3],
        }

    def get_data_quality_report(self) -> pd.DataFrame:
        """
        Generate comprehensive data quality metrics.

        Returns:
            DataFrame with quality metrics per symbol
        """
        query = """
            SELECT 
                symbol,
                COUNT(*) as total_records,
                MIN(date) as first_date,
                MAX(date) as last_date,
                SUM(CASE WHEN volume = 0 THEN 1 ELSE 0 END) as zero_volume_days,
                SUM(CASE WHEN high = low THEN 1 ELSE 0 END) as flat_days,
                AVG(volume) as avg_volume,
                AVG(close) as avg_close
            FROM ohlcv
            GROUP BY symbol
            ORDER BY symbol
        """

        df = self.connection.execute(query).df()

        logger.info(f"Generated quality report for {len(df)} symbols")

        return df

    def log_data_quality_issue(
        self,
        symbol: str,
        check_type: str,
        issue_type: str,
        issue_count: int,
        details: str,
        severity: str = "warning",
    ) -> None:
        """
        Log data quality issue.

        Args:
            symbol: Stock symbol
            check_type: Type of check performed
            issue_type: Type of issue found
            issue_count: Number of issues
            details: Detailed description
            severity: Severity level (info, warning, error)
        """
        self.connection.execute(
            """
            INSERT INTO data_quality_logs
            (symbol, check_date, check_type, issue_type, issue_count, details, severity)
            VALUES (?, CURRENT_DATE, ?, ?, ?, ?, ?)
        """,
            (symbol, check_type, issue_type, issue_count, details, severity),
        )

    def delete_symbol_data(
        self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> int:
        """
        Delete data for a symbol.

        Args:
            symbol: Stock symbol
            start_date: Start date (optional)
            end_date: End date (optional)

        Returns:
            Number of rows deleted
        """
        query = "DELETE FROM ohlcv WHERE symbol = ?"
        params = [symbol]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        self.connection.execute(query, params)

        # Get count before committing
        count = self.connection.execute(
            "SELECT changes()"
        ).fetchone()[0]

        logger.info(f"Deleted {count} rows for {symbol}")

        return count

    def vacuum(self) -> None:
        """Optimize database by running VACUUM."""
        logger.info("Running VACUUM on database...")
        self.connection.execute("VACUUM")
        logger.info("VACUUM completed")

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary with database stats
        """
        stats = {}

        # Table sizes
        for table in ["ohlcv", "fundamentals", "corporate_actions", "market_metadata"]:
            count = self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            stats[f"{table}_count"] = count

        # Database file size
        if self.db_path.exists():
            stats["db_size_mb"] = self.db_path.stat().st_size / (1024 * 1024)

        # Date range
        date_range = self.connection.execute(
            "SELECT MIN(date) as min_date, MAX(date) as max_date FROM ohlcv"
        ).fetchone()

        if date_range and date_range[0]:
            stats["min_date"] = str(date_range[0])
            stats["max_date"] = str(date_range[1])

        # Symbol count
        symbol_count = self.connection.execute(
            "SELECT COUNT(DISTINCT symbol) FROM ohlcv"
        ).fetchone()[0]
        stats["unique_symbols"] = symbol_count

        return stats

    def close(self) -> None:
        """Close database connection."""
        self.db_manager.close()
        logger.info("Database connection closed")

