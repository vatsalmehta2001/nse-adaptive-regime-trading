"""
Pytest Configuration and Fixtures.

Provides shared fixtures and configuration for all tests.
"""

import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    """
    Get project root directory.

    Returns:
        Path to project root
    """
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir(project_root: Path) -> Path:
    """
    Get test data directory.

    Args:
        project_root: Project root path

    Returns:
        Path to test data directory
    """
    test_dir = project_root / "tests" / "data"
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """
    Create temporary directory for tests.

    Yields:
        Path to temporary directory
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Set up mock environment variables for testing.

    Args:
        monkeypatch: Pytest monkeypatch fixture
    """
    env_vars = {
        "TRADING_MODE": "paper",
        "LOG_LEVEL": "DEBUG",
        "DUCKDB_PATH": ":memory:",  # In-memory database for tests
        "ZERODHA_API_KEY": "test_api_key",
        "ZERODHA_API_SECRET": "test_api_secret",
        "QLIB_DATA_PATH": "/tmp/test_qlib_data",
        "RL_MODELS_PATH": "/tmp/test_rl_models",
    }

    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """
    Generate sample OHLCV data for testing.

    Returns:
        DataFrame with OHLCV data
    """
    dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")
    n_periods = len(dates)

    # Generate synthetic price data
    np.random.seed(42)
    close_prices = 100 * np.exp(np.cumsum(np.random.randn(n_periods) * 0.02))

    data = pd.DataFrame(
        {
            "timestamp": dates,
            "open": close_prices * (1 + np.random.uniform(-0.01, 0.01, n_periods)),
            "high": close_prices * (1 + np.random.uniform(0, 0.02, n_periods)),
            "low": close_prices * (1 - np.random.uniform(0, 0.02, n_periods)),
            "close": close_prices,
            "volume": np.random.randint(100000, 1000000, n_periods),
        }
    )

    # Ensure OHLC consistency
    data["high"] = data[["open", "high", "close"]].max(axis=1)
    data["low"] = data[["open", "low", "close"]].min(axis=1)

    return data


@pytest.fixture
def sample_multi_symbol_data() -> pd.DataFrame:
    """
    Generate sample multi-symbol OHLCV data.

    Returns:
        DataFrame with multi-symbol OHLCV data
    """
    symbols = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]
    dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")

    all_data = []

    for symbol in symbols:
        np.random.seed(hash(symbol) % 2**32)
        n_periods = len(dates)

        close_prices = 100 * np.exp(np.cumsum(np.random.randn(n_periods) * 0.02))

        symbol_data = pd.DataFrame(
            {
                "symbol": symbol,
                "timestamp": dates,
                "open": close_prices * (1 + np.random.uniform(-0.01, 0.01, n_periods)),
                "high": close_prices * (1 + np.random.uniform(0, 0.02, n_periods)),
                "low": close_prices * (1 - np.random.uniform(0, 0.02, n_periods)),
                "close": close_prices,
                "volume": np.random.randint(100000, 1000000, n_periods),
            }
        )

        # Ensure OHLC consistency
        symbol_data["high"] = symbol_data[["open", "high", "close"]].max(axis=1)
        symbol_data["low"] = symbol_data[["open", "low", "close"]].min(axis=1)

        all_data.append(symbol_data)

    return pd.concat(all_data, ignore_index=True)


@pytest.fixture
def sample_trades() -> pd.DataFrame:
    """
    Generate sample trade data.

    Returns:
        DataFrame with trade data
    """
    dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="W")
    n_trades = len(dates)

    np.random.seed(42)

    trades = pd.DataFrame(
        {
            "trade_id": [f"T{i:04d}" for i in range(n_trades)],
            "symbol": np.random.choice(["RELIANCE", "TCS", "HDFCBANK"], n_trades),
            "side": np.random.choice(["BUY", "SELL"], n_trades),
            "quantity": np.random.randint(10, 100, n_trades),
            "price": np.random.uniform(100, 1000, n_trades),
            "timestamp": dates,
            "strategy": "test_strategy",
        }
    )

    trades["value"] = trades["quantity"] * trades["price"]

    return trades


@pytest.fixture
def sample_portfolio() -> dict:
    """
    Generate sample portfolio data.

    Returns:
        Dictionary with portfolio information
    """
    return {
        "cash": 500000.0,
        "positions": {
            "RELIANCE": {"quantity": 100, "average_price": 2500.0},
            "TCS": {"quantity": 50, "average_price": 3500.0},
            "HDFCBANK": {"quantity": 200, "average_price": 1600.0},
        },
        "initial_capital": 1000000.0,
        "currency": "INR",
    }


@pytest.fixture
def mock_config(temp_dir: Path) -> dict:
    """
    Generate mock configuration for testing.

    Args:
        temp_dir: Temporary directory path

    Returns:
        Configuration dictionary
    """
    return {
        "data_sources": {
            "openbb": {"enabled": False},
            "zerodha": {"enabled": False},
        },
        "qlib": {
            "data_path": str(temp_dir / "qlib_data"),
            "provider_uri": "~/.qlib/qlib_data/test",
        },
        "rl": {
            "models_path": str(temp_dir / "rl_models"),
            "training": {"total_timesteps": 1000},
        },
        "risk_management": {
            "max_position_size": 0.20,
            "max_drawdown": 0.15,
        },
    }


# Markers for test categorization
def pytest_configure(config: pytest.Config) -> None:
    """
    Configure pytest with custom markers.

    Args:
        config: Pytest configuration
    """
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "performance: marks tests as performance benchmarks")
    config.addinivalue_line("markers", "live: marks tests that require live market data")


# Test collection hooks
def pytest_collection_modifyitems(config: pytest.Config, items: list) -> None:
    """
    Modify test items during collection.

    Args:
        config: Pytest configuration
        items: List of test items
    """
    # Skip live tests by default
    skip_live = pytest.mark.skip(reason="Live tests require real market data and API access")

    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


@pytest.fixture(autouse=True)
def reset_singletons() -> Generator[None, None, None]:
    """
    Reset singleton instances between tests.

    Yields:
        None
    """
    yield

    # Reset any global singletons here
    # Example: DatabaseManager._instance = None


@pytest.fixture
def mock_datetime():
    """
    Mock datetime for deterministic testing.

    Yields:
        Mock datetime
    """

    class MockDatetime:
        """Mock datetime class."""

        @staticmethod
        def now():
            """Return fixed datetime."""
            return datetime(2024, 6, 15, 10, 30, 0)  # Saturday 10:30 AM

    return MockDatetime


# Performance test configuration
@pytest.fixture
def benchmark_config() -> dict:
    """
    Configuration for benchmark tests.

    Returns:
        Benchmark configuration
    """
    return {
        "min_rounds": 5,
        "max_time": 1.0,
        "calibration_precision": 10,
        "warmup": True,
    }

