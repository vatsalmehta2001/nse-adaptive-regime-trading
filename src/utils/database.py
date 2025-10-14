"""
Database Management Module.

Manages DuckDB connections and provides database utilities for storing
market data, trades, and analytics.
"""

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

import duckdb
import pandas as pd
from pydantic import BaseModel

from src.utils.helpers import ensure_dir
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class DatabaseConfig(BaseModel):
    """Database configuration model."""

    path: str = "data/trading_db.duckdb"
    memory_limit: str = "4GB"
    threads: int = 4
    read_only: bool = False


class DatabaseManager:
    """
    Manages DuckDB database connections and operations.

    DuckDB is chosen for its excellent OLAP performance with time-series data.
    """

    def __init__(self, config: Optional[DatabaseConfig] = None) -> None:
        """
        Initialize database manager.

        Args:
            config: Database configuration (uses defaults if None)
        """
        if config is None:
            db_path = os.getenv("DUCKDB_PATH", "data/trading_db.duckdb")
            config = DatabaseConfig(path=db_path)

        self.config = config
        self.db_path = Path(config.path)

        # Ensure database directory exists
        ensure_dir(self.db_path.parent)

        # Connection pool (DuckDB recommends single connection per process)
        self._connection: Optional[duckdb.DuckDBPyConnection] = None

        logger.info(f"Database manager initialized: {self.db_path}")

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """
        Get or create database connection.

        Returns:
            DuckDB connection
        """
        if self._connection is None:
            self._connection = duckdb.connect(
                str(self.db_path),
                read_only=self.config.read_only,
            )

            # Configure connection
            self._connection.execute(f"SET memory_limit='{self.config.memory_limit}'")
            self._connection.execute(f"SET threads={self.config.threads}")

            logger.debug(f"Database connection established: {self.db_path}")

        return self._connection

    def close(self) -> None:
        """Close database connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None
            logger.debug("Database connection closed")

    @contextmanager
    def get_connection(self) -> Generator[duckdb.DuckDBPyConnection, None, None]:
        """
        Context manager for database connection.

        Yields:
            DuckDB connection
        """
        try:
            yield self.connection
        finally:
            # DuckDB handles transactions automatically
            pass

    def execute(self, query: str, parameters: Optional[tuple] = None) -> duckdb.DuckDBPyConnection:
        """
        Execute SQL query.

        Args:
            query: SQL query to execute
            parameters: Query parameters (optional)

        Returns:
            Query result
        """
        if parameters:
            return self.connection.execute(query, parameters)
        return self.connection.execute(query)

    def execute_many(self, query: str, parameters: List[tuple]) -> None:
        """
        Execute SQL query with multiple parameter sets.

        Args:
            query: SQL query to execute
            parameters: List of parameter tuples
        """
        self.connection.executemany(query, parameters)

    def query_df(self, query: str, parameters: Optional[tuple] = None) -> pd.DataFrame:
        """
        Execute query and return results as DataFrame.

        Args:
            query: SQL query to execute
            parameters: Query parameters (optional)

        Returns:
            Query results as DataFrame
        """
        result = self.execute(query, parameters)
        return result.df()

    def insert_df(
        self,
        df: pd.DataFrame,
        table: str,
        if_exists: str = "append",
    ) -> None:
        """
        Insert DataFrame into table.

        Args:
            df: DataFrame to insert
            table: Table name
            if_exists: Action if table exists ('append', 'replace', 'fail')
        """
        if if_exists == "replace":
            self.execute(f"DROP TABLE IF EXISTS {table}")

        # Use DuckDB's efficient DataFrame insertion
        self.connection.register("temp_df", df)
        self.execute(f"CREATE TABLE IF NOT EXISTS {table} AS SELECT * FROM temp_df")

        if if_exists == "append" and len(df) > 0:
            self.execute(f"INSERT INTO {table} SELECT * FROM temp_df")

        self.connection.unregister("temp_df")
        logger.debug(f"Inserted {len(df)} rows into {table}")

    def table_exists(self, table: str) -> bool:
        """
        Check if table exists.

        Args:
            table: Table name

        Returns:
            True if table exists, False otherwise
        """
        result = self.query_df(
            "SELECT COUNT(*) as count FROM information_schema.tables WHERE table_name = ?",
            (table,),
        )
        return result["count"].iloc[0] > 0

    def create_market_data_tables(self) -> None:
        """Create tables for market data storage."""
        # OHLCV data table
        self.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                symbol VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                open DOUBLE NOT NULL,
                high DOUBLE NOT NULL,
                low DOUBLE NOT NULL,
                close DOUBLE NOT NULL,
                volume BIGINT NOT NULL,
                timeframe VARCHAR NOT NULL,
                PRIMARY KEY (symbol, timestamp, timeframe)
            )
        """)

        # Create index for faster queries
        self.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol ON ohlcv(symbol)")
        self.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_timestamp ON ohlcv(timestamp)")

        logger.info("Market data tables created")

    def create_trading_tables(self) -> None:
        """Create tables for trading data."""
        # Orders table
        self.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id VARCHAR PRIMARY KEY,
                symbol VARCHAR NOT NULL,
                order_type VARCHAR NOT NULL,
                side VARCHAR NOT NULL,
                quantity INTEGER NOT NULL,
                price DOUBLE,
                status VARCHAR NOT NULL,
                strategy VARCHAR,
                timestamp TIMESTAMP NOT NULL,
                filled_quantity INTEGER DEFAULT 0,
                average_price DOUBLE,
                metadata JSON
            )
        """)

        # Trades table
        self.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                trade_id VARCHAR PRIMARY KEY,
                order_id VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                side VARCHAR NOT NULL,
                quantity INTEGER NOT NULL,
                price DOUBLE NOT NULL,
                commission DOUBLE DEFAULT 0,
                timestamp TIMESTAMP NOT NULL,
                strategy VARCHAR,
                pnl DOUBLE,
                metadata JSON
            )
        """)

        # Positions table
        self.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                position_id VARCHAR PRIMARY KEY,
                symbol VARCHAR NOT NULL,
                quantity INTEGER NOT NULL,
                average_price DOUBLE NOT NULL,
                current_price DOUBLE,
                unrealized_pnl DOUBLE,
                realized_pnl DOUBLE DEFAULT 0,
                opened_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                strategy VARCHAR
            )
        """)

        logger.info("Trading tables created")

    def create_performance_tables(self) -> None:
        """Create tables for performance tracking."""
        self.execute("""
            CREATE TABLE IF NOT EXISTS daily_performance (
                date DATE PRIMARY KEY,
                portfolio_value DOUBLE NOT NULL,
                cash DOUBLE NOT NULL,
                positions_value DOUBLE NOT NULL,
                daily_return DOUBLE,
                cumulative_return DOUBLE,
                sharpe_ratio DOUBLE,
                max_drawdown DOUBLE,
                num_trades INTEGER,
                win_rate DOUBLE
            )
        """)

        logger.info("Performance tables created")

    def initialize_schema(self) -> None:
        """Initialize all database tables."""
        self.create_market_data_tables()
        self.create_trading_tables()
        self.create_performance_tables()
        logger.info("Database schema initialized")

    def vacuum(self) -> None:
        """Optimize database by vacuuming."""
        self.execute("VACUUM")
        logger.info("Database vacuumed")

    def get_table_info(self, table: str) -> pd.DataFrame:
        """
        Get table schema information.

        Args:
            table: Table name

        Returns:
            DataFrame with table schema
        """
        return self.query_df(f"PRAGMA table_info('{table}')")

    def get_row_count(self, table: str) -> int:
        """
        Get number of rows in table.

        Args:
            table: Table name

        Returns:
            Number of rows
        """
        result = self.query_df(f"SELECT COUNT(*) as count FROM {table}")
        return int(result["count"].iloc[0])

    def __enter__(self) -> "DatabaseManager":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()


# Global database instance (lazy initialization)
_db_instance: Optional[DatabaseManager] = None


def get_database() -> DatabaseManager:
    """
    Get global database instance.

    Returns:
        Database manager instance
    """
    global _db_instance

    if _db_instance is None:
        _db_instance = DatabaseManager()

    return _db_instance

