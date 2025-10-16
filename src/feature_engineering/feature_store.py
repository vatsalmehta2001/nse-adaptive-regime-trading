"""
Feature Store for Alpha Factors.

Manages storage and retrieval of Qlib Alpha-158 factors in DuckDB.
Provides incremental updates and efficient querying.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import pandas as pd
from loguru import logger


class FeatureStore:
    """
    Manage factor storage in DuckDB.

    Features:
    - Store/retrieve 158 factors efficiently
    - Incremental updates (don't recompute existing)
    - Fast queries with proper indexing
    - Coverage tracking
    """

    def __init__(self, db_path: str = "data/trading_db.duckdb"):
        """
        Connect to existing trading database.

        Args:
            db_path: Path to DuckDB database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.connection = duckdb.connect(str(self.db_path))
        logger.info(f"FeatureStore connected to {self.db_path}")

    def create_schema(self):
        """
        Create factor tables.

        Creates:
        - alpha158_factors: Stores all 158 factors
        - market_regimes: Stores regime labels and probabilities
        """
        logger.info("Creating feature store schema...")

        # Alpha-158 factors table
        factor_cols = ', '.join([
            f"factor_{i:03d} DOUBLE" for i in range(1, 159)
        ])

        self.connection.execute(f"""
            CREATE TABLE IF NOT EXISTS alpha158_factors (
                symbol VARCHAR,
                date DATE,
                {factor_cols},
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, date)
            )
        """)

        # Create indexes
        self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_alpha_symbol
            ON alpha158_factors(symbol)
        """)

        self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_alpha_date
            ON alpha158_factors(date)
        """)

        # Market regimes table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS market_regimes (
                date DATE PRIMARY KEY,
                regime_label INTEGER,
                regime_name VARCHAR,
                method VARCHAR,
                bull_prob DOUBLE,
                bear_prob DOUBLE,
                highvol_prob DOUBLE,
                crash_prob DOUBLE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_regime_date
            ON market_regimes(date)
        """)

        logger.info(" Feature store schema created")

    def store_factors(
        self,
        df: pd.DataFrame,
        symbol: str,
        overwrite: bool = False
    ) -> int:
        """
        Store factors for a symbol.

        Uses UPSERT logic to avoid duplicates.

        Args:
            df: DataFrame with factors (columns: factor_001 to factor_158)
            symbol: Stock symbol
            overwrite: Whether to overwrite existing data

        Returns:
            Number of rows inserted/updated
        """
        if df.empty:
            logger.warning(f"Empty DataFrame for {symbol}, skipping")
            return 0

        # Prepare data
        df = df.copy()
        df['symbol'] = symbol

        # Ensure date column
        if 'date' not in df.columns:
            df = df.reset_index()
            if 'date' not in df.columns:
                df['date'] = df.index

        # Select only factor columns + symbol + date
        factor_cols = [c for c in df.columns if c.startswith('factor_')]
        cols_to_store = ['symbol', 'date'] + factor_cols
        df_to_store = df[cols_to_store].copy()

        # Convert date to proper format
        df_to_store['date'] = pd.to_datetime(df_to_store['date']).dt.date

        if overwrite:
            # Delete existing data for this symbol
            self.connection.execute(
                "DELETE FROM alpha158_factors WHERE symbol = ?",
                [symbol]
            )

        # Insert data (ON CONFLICT DO UPDATE for DuckDB)
        # DuckDB uses INSERT OR REPLACE for upsert
        rows_before = self.connection.execute(
            "SELECT COUNT(*) FROM alpha158_factors WHERE symbol = ?",
            [symbol]
        ).fetchone()[0]

        # Register dataframe and insert
        self.connection.register('temp_factors', df_to_store)

        try:
            # Build column list explicitly (exclude created_at which has DEFAULT)
            cols = ', '.join(cols_to_store)
            placeholders = ', '.join(['?' for _ in cols_to_store])

            # Use pandas to_sql equivalent for DuckDB
            self.connection.execute(f"""
                INSERT OR REPLACE INTO alpha158_factors ({cols})
                SELECT {cols} FROM temp_factors
            """)

            rows_after = self.connection.execute(
                "SELECT COUNT(*) FROM alpha158_factors WHERE symbol = ?",
                [symbol]
            ).fetchone()[0]

            rows_inserted = rows_after - rows_before if not overwrite else rows_after

            logger.info(f"Stored {rows_inserted} factor rows for {symbol}")

            return rows_inserted

        finally:
            self.connection.unregister('temp_factors')

    def store_regime_labels(
        self,
        df: pd.DataFrame,
        method: str = "wasserstein"
    ) -> int:
        """
        Store regime labels (date, regime, probabilities).

        Args:
            df: DataFrame with columns: date, regime_label, regime_name, probabilities
            method: Detection method ('wasserstein' or 'hmm')

        Returns:
            Number of rows stored
        """
        if df.empty:
            logger.warning("Empty regime DataFrame, skipping")
            return 0

        df = df.copy()
        df['method'] = method

        # Ensure required columns exist
        if 'date' not in df.columns:
            df = df.reset_index()

        # Convert date
        df['date'] = pd.to_datetime(df['date']).dt.date

        # Add probability columns if not present
        prob_cols = ['bull_prob', 'bear_prob', 'highvol_prob', 'crash_prob']
        for col in prob_cols:
            if col not in df.columns:
                df[col] = 0.0

        # Select columns
        cols_to_store = ['date', 'regime_label', 'regime_name', 'method'] + prob_cols
        df_to_store = df[[c for c in cols_to_store if c in df.columns]].copy()

        # Register and insert
        self.connection.register('temp_regimes', df_to_store)

        try:
            self.connection.execute("""
                INSERT OR REPLACE INTO market_regimes
                SELECT * FROM temp_regimes
            """)

            rows = len(df_to_store)
            logger.info(f"Stored {rows} regime labels using {method} method")

            return rows

        finally:
            self.connection.unregister('temp_regimes')

    def get_factors(
        self,
        symbols: List[str],
        start_date: str = None,
        end_date: str = None,
        include_regimes: bool = True
    ) -> pd.DataFrame:
        """
        Retrieve factors with optional regime join.

        Args:
            symbols: List of symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            include_regimes: Whether to join regime data

        Returns:
            DataFrame with multi-index (symbol, date)
        """
        # Build query
        query = "SELECT * FROM alpha158_factors WHERE symbol IN ({})".format(
            ','.join(['?' for _ in symbols])
        )
        params = symbols

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        # Execute query
        df = self.connection.execute(query, params).fetchdf()

        if df.empty:
            return df

        # Join with regimes if requested
        if include_regimes and not df.empty:
            try:
                regimes = self.connection.execute("""
                    SELECT date, regime_label, regime_name
                    FROM market_regimes
                """).fetchdf()

                if not regimes.empty:
                    df = df.merge(regimes, on='date', how='left')
            except Exception as e:
                logger.warning(f"Could not join regimes: {e}")

        # Set index
        if not df.empty and 'symbol' in df.columns and 'date' in df.columns:
            df = df.set_index(['symbol', 'date']).sort_index()

        logger.info(f"Retrieved {len(df)} factor rows for {len(symbols)} symbols")

        return df

    def check_coverage(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Check which symbols/dates have factors computed.

        Args:
            symbols: List of symbols to check
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with coverage statistics
        """
        coverage = []

        for symbol in symbols:
            result = self.connection.execute("""
                SELECT
                    symbol,
                    COUNT(*) as total_rows,
                    MIN(date) as first_date,
                    MAX(date) as last_date
                FROM alpha158_factors
                WHERE symbol = ?
                  AND date >= ?
                  AND date <= ?
                GROUP BY symbol
            """, [symbol, start_date, end_date]).fetchone()

            if result:
                coverage.append({
                    'symbol': result[0],
                    'total_rows': result[1],
                    'first_date': result[2],
                    'last_date': result[3],
                    'has_data': True
                })
            else:
                coverage.append({
                    'symbol': symbol,
                    'total_rows': 0,
                    'first_date': None,
                    'last_date': None,
                    'has_data': False
                })

        return pd.DataFrame(coverage)

    def incremental_update(
        self,
        symbols: List[str] = None
    ) -> Dict[str, int]:
        """
        Update factors only for new dates.

        Args:
            symbols: Symbols to update (None = all)

        Returns:
            Dict with update statistics
        """
        if symbols is None:
            # Get all symbols from database
            symbols = self.connection.execute("""
                SELECT DISTINCT symbol FROM alpha158_factors
            """).fetchdf()['symbol'].tolist()

        stats = {
            'symbols_updated': 0,
            'rows_added': 0
        }

        for symbol in symbols:
            # Get latest date for this symbol
            latest = self.connection.execute("""
                SELECT MAX(date) as latest_date
                FROM alpha158_factors
                WHERE symbol = ?
            """, [symbol]).fetchone()[0]

            if latest:
                logger.info(f"{symbol}: Latest date is {latest}")
                # In production, would fetch and compute factors for dates after 'latest'
                # For now, just log

            stats['symbols_updated'] += 1

        return stats

    def get_database_stats(self) -> Dict[str, any]:
        """Get database statistics."""
        stats = {}

        # Factor table stats
        result = self.connection.execute("""
            SELECT
                COUNT(DISTINCT symbol) as total_symbols,
                COUNT(*) as total_rows,
                MIN(date) as first_date,
                MAX(date) as last_date
            FROM alpha158_factors
        """).fetchone()

        if result:
            stats['factors'] = {
                'total_symbols': result[0],
                'total_rows': result[1],
                'first_date': str(result[2]) if result[2] else None,
                'last_date': str(result[3]) if result[3] else None
            }

        # Regime table stats
        result = self.connection.execute("""
            SELECT
                COUNT(*) as total_rows,
                MIN(date) as first_date,
                MAX(date) as last_date
            FROM market_regimes
        """).fetchone()

        if result:
            stats['regimes'] = {
                'total_rows': result[0],
                'first_date': str(result[1]) if result[1] else None,
                'last_date': str(result[2]) if result[2] else None
            }

        return stats

    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            logger.info("FeatureStore connection closed")

