"""
Performance benchmark tests.

Tests for latency-critical components to ensure they meet performance requirements.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.mark.performance
def test_data_processing_benchmark(benchmark, sample_ohlcv_data: pd.DataFrame) -> None:
    """
    Benchmark data processing performance.

    Args:
        benchmark: Pytest benchmark fixture
        sample_ohlcv_data: Sample OHLCV data
    """

    def process_data():
        """Process OHLCV data."""
        df = sample_ohlcv_data.copy()
        # Simulate feature calculation
        df["returns"] = df["close"].pct_change()
        df["sma_20"] = df["close"].rolling(20).mean()
        df["volatility"] = df["returns"].rolling(20).std()
        return df

    result = benchmark(process_data)
    assert len(result) == len(sample_ohlcv_data)


@pytest.mark.performance
def test_matrix_operations_benchmark(benchmark) -> None:
    """
    Benchmark matrix operations for portfolio optimization.

    Args:
        benchmark: Pytest benchmark fixture
    """

    def matrix_operations():
        """Perform matrix operations."""
        n = 100
        matrix = np.random.randn(n, n)
        cov_matrix = np.cov(matrix.T)
        inv_cov = np.linalg.inv(cov_matrix)
        return inv_cov

    result = benchmark(matrix_operations)
    assert result.shape == (100, 100)


@pytest.mark.performance
@pytest.mark.slow
def test_large_dataset_processing(benchmark) -> None:
    """
    Benchmark processing large datasets.

    Args:
        benchmark: Pytest benchmark fixture
    """

    def process_large_dataset():
        """Process large dataset."""
        # Simulate 1 year of 1-minute data for 50 stocks
        n_rows = 252 * 390 * 50  # trading days * minutes per day * stocks
        df = pd.DataFrame(
            {
                "symbol": np.random.choice(["STOCK" + str(i) for i in range(50)], n_rows),
                "timestamp": pd.date_range("2024-01-01", periods=n_rows, freq="1min"),
                "close": np.random.randn(n_rows) * 100 + 1000,
            }
        )

        # Group by symbol and calculate rolling metrics
        result = df.groupby("symbol")["close"].rolling(window=20).mean()
        return result

    # This test is marked as slow
    result = benchmark(process_large_dataset)
    assert len(result) > 0

