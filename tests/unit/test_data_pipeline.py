"""
Unit tests for data pipeline components.

Tests OpenBB fetcher, storage, validation, and pipeline orchestration.
"""

import pandas as pd
import pytest
from datetime import datetime, timedelta

from src.data_pipeline.data_storage import DataStorageManager
from src.data_pipeline.data_validator import MarketDataValidator
from src.data_pipeline.openbb_client import OpenBBDataFetcher
from src.feature_engineering.technical_indicators import TechnicalIndicators


class TestOpenBBDataFetcher:
    """Test OpenBB data fetching functionality."""

    def test_normalize_nse_symbol(self):
        """Test NSE symbol normalization."""
        fetcher = OpenBBDataFetcher()

        assert fetcher._normalize_nse_symbol("RELIANCE") == "RELIANCE.NS"
        assert fetcher._normalize_nse_symbol("RELIANCE.NS") == "RELIANCE.NS"
        assert fetcher._normalize_nse_symbol("tcs") == "TCS.NS"

    def test_get_nifty50_constituents(self):
        """Test getting NIFTY 50 constituent list."""
        fetcher = OpenBBDataFetcher()

        constituents = fetcher.get_nifty50_constituents()

        assert isinstance(constituents, list)
        assert len(constituents) > 40  # Should have around 50 stocks
        assert "RELIANCE" in constituents
        assert "TCS" in constituents

    def test_cache_key_generation(self):
        """Test cache key generation."""
        fetcher = OpenBBDataFetcher()

        key1 = fetcher._get_cache_key("RELIANCE", "2024-01-01", "2024-12-31")
        key2 = fetcher._get_cache_key("RELIANCE", "2024-01-01", "2024-12-31")
        key3 = fetcher._get_cache_key("TCS", "2024-01-01", "2024-12-31")

        assert key1 == key2  # Same parameters should give same key
        assert key1 != key3  # Different parameters should give different key

    def test_rate_limiter(self):
        """Test rate limiting functionality."""
        from src.data_pipeline.openbb_client import RateLimiter
        import time

        limiter = RateLimiter(requests_per_minute=60)

        start = time.time()
        for _ in range(3):
            limiter.wait_if_needed()
        duration = time.time() - start

        # Should take at least 2 seconds for 3 requests at 60 RPM
        assert duration >= 2.0


class TestDataStorageManager:
    """Test DuckDB storage functionality."""

    def test_create_schema(self, tmp_path):
        """Test database schema creation."""
        db_path = tmp_path / "test.duckdb"
        storage = DataStorageManager(db_path=str(db_path))

        storage.create_schema()

        # Check that tables exist
        tables = storage.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()

        table_names = [t[0] for t in tables]
        assert "ohlcv" in table_names
        assert "market_metadata" in table_names

    def test_insert_and_query_ohlcv(self, tmp_path, sample_ohlcv_data):
        """Test OHLCV data insertion and querying."""
        db_path = tmp_path / "test.duckdb"
        storage = DataStorageManager(db_path=str(db_path))

        storage.create_schema()

        # Insert data
        rows_inserted = storage.insert_ohlcv(sample_ohlcv_data)
        assert rows_inserted > 0

        # Query data back
        result = storage.query_ohlcv(
            symbols=["TEST"],
            start_date="2024-01-01",
            end_date="2024-12-31",
        )

        assert not result.empty
        assert len(result) > 0
        assert "symbol" in result.columns
        assert "close" in result.columns

    def test_get_latest_date(self, tmp_path, sample_ohlcv_data):
        """Test getting latest date for a symbol."""
        db_path = tmp_path / "test.duckdb"
        storage = DataStorageManager(db_path=str(db_path))

        storage.create_schema()
        storage.insert_ohlcv(sample_ohlcv_data)

        latest_date = storage.get_latest_date("TEST")

        assert latest_date is not None
        assert isinstance(latest_date, str)

    def test_get_data_coverage(self, tmp_path, sample_ohlcv_data):
        """Test data coverage statistics."""
        db_path = tmp_path / "test.duckdb"
        storage = DataStorageManager(db_path=str(db_path))

        storage.create_schema()
        storage.insert_ohlcv(sample_ohlcv_data)

        coverage = storage.get_data_coverage("TEST")

        assert coverage["has_data"] is True
        assert "first_date" in coverage
        assert "last_date" in coverage
        assert coverage["total_records"] > 0

    def test_deduplication(self, tmp_path, sample_ohlcv_data):
        """Test that duplicate entries are handled correctly."""
        db_path = tmp_path / "test.duckdb"
        storage = DataStorageManager(db_path=str(db_path))

        storage.create_schema()

        # Insert same data twice
        storage.insert_ohlcv(sample_ohlcv_data)
        storage.insert_ohlcv(sample_ohlcv_data)

        # Query to check count
        result = storage.query_ohlcv(symbols=["TEST"])

        # Should only have unique records (no duplicates)
        unique_dates = result.groupby(["symbol", "date"]).size()
        assert (unique_dates == 1).all()


class TestMarketDataValidator:
    """Test data validation functionality."""

    def test_validate_ohlcv_schema(self, sample_ohlcv_data):
        """Test OHLCV schema validation."""
        validator = MarketDataValidator()

        validated_df, report = validator.validate_ohlcv(sample_ohlcv_data)

        assert not validated_df.empty
        assert report["valid_rows"] > 0
        assert "errors" in report

    def test_detect_ohlc_violations(self):
        """Test OHLC relationship validation."""
        validator = MarketDataValidator()

        # Create data with violations
        df = pd.DataFrame({
            "symbol": ["TEST"] * 3,
            "date": pd.date_range("2024-01-01", periods=3),
            "open": [100, 105, 110],
            "high": [95, 110, 115],  # First row: high < open (violation)
            "low": [90, 100, 105],
            "close": [105, 108, 112],
            "volume": [1000, 1100, 1200],
        })

        violations = validator._validate_ohlc_relationships(df)

        assert len(violations) > 0  # Should detect the violation

    def test_detect_outliers(self, sample_ohlcv_data):
        """Test outlier detection."""
        validator = MarketDataValidator()

        df_with_outliers = validator.detect_outliers(
            sample_ohlcv_data,
            method="iqr",
            threshold=3.0,
        )

        assert "is_outlier" in df_with_outliers.columns
        assert df_with_outliers["is_outlier"].dtype == bool

    def test_generate_quality_report(self, sample_ohlcv_data):
        """Test quality report generation."""
        validator = MarketDataValidator()

        report = validator.generate_quality_report(sample_ohlcv_data)

        assert "total_rows" in report
        assert "date_range" in report
        assert "symbols" in report
        assert "data_quality" in report
        assert report["data_quality"]["status"] in [
            "excellent",
            "good",
            "acceptable",
            "poor",
        ]


class TestTechnicalIndicators:
    """Test technical indicator calculations."""

    def test_calculate_returns(self, sample_ohlcv_data):
        """Test return calculations."""
        indicators = TechnicalIndicators()

        df = indicators.calculate_returns(sample_ohlcv_data, periods=[1, 5])

        assert "return_1d" in df.columns
        assert "return_5d" in df.columns
        assert "log_return_1d" in df.columns

    def test_calculate_moving_averages(self, sample_ohlcv_data):
        """Test moving average calculations."""
        indicators = TechnicalIndicators()

        df = indicators.calculate_moving_averages(
            sample_ohlcv_data,
            windows=[5, 10],
        )

        assert "sma_5" in df.columns
        assert "ema_5" in df.columns
        assert "dist_sma_5" in df.columns

    def test_calculate_rsi(self, sample_ohlcv_data):
        """Test RSI calculation."""
        indicators = TechnicalIndicators()

        df = indicators.calculate_rsi(sample_ohlcv_data, period=14)

        assert "rsi_14" in df.columns

        # RSI should be between 0 and 100
        rsi_values = df["rsi_14"].dropna()
        if len(rsi_values) > 0:
            assert (rsi_values >= 0).all()
            assert (rsi_values <= 100).all()

    def test_calculate_bollinger_bands(self, sample_ohlcv_data):
        """Test Bollinger Bands calculation."""
        indicators = TechnicalIndicators()

        df = indicators.calculate_bollinger_bands(sample_ohlcv_data, period=20)

        assert "bb_upper_20" in df.columns
        assert "bb_middle_20" in df.columns
        assert "bb_lower_20" in df.columns
        assert "bb_percent_20" in df.columns

    def test_calculate_macd(self, sample_ohlcv_data):
        """Test MACD calculation."""
        indicators = TechnicalIndicators()

        df = indicators.calculate_macd(sample_ohlcv_data)

        assert "macd" in df.columns
        assert "macd_signal" in df.columns
        assert "macd_hist" in df.columns

    def test_generate_all_features(self, sample_ohlcv_data):
        """Test complete feature generation."""
        indicators = TechnicalIndicators()

        df = indicators.generate_all_features(
            sample_ohlcv_data,
            include_basic=True,
            include_advanced=True,
        )

        # Should have many new columns
        original_cols = len(sample_ohlcv_data.columns)
        new_cols = len(df.columns)

        assert new_cols > original_cols + 10  # At least 10 new features


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    np_random = __import__("numpy").random

    np_random.seed(42)
    close_prices = 100 * __import__("numpy").exp(
        __import__("numpy").cumsum(np_random.randn(100) * 0.02)
    )

    df = pd.DataFrame({
        "symbol": ["TEST"] * 100,
        "date": dates,
        "open": close_prices * (1 + np_random.uniform(-0.01, 0.01, 100)),
        "high": close_prices * (1 + np_random.uniform(0, 0.02, 100)),
        "low": close_prices * (1 - np_random.uniform(0, 0.02, 100)),
        "close": close_prices,
        "volume": np_random.randint(100000, 1000000, 100),
    })

    # Ensure OHLC consistency
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)

    return df

