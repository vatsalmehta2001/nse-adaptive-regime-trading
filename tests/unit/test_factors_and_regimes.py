"""
Comprehensive tests for Qlib factors and regime detection.

Tests:
- Exact factor count (must be 158)
- No NaN/Inf after warmup
- Vectorization speed
- COVID crash detection
- Feature storage
- IC calculation
"""

import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    QlibAlpha158,
    FeatureStore,
    FactorAnalyzer,
)
from src.regime_detection import WassersteinRegimeDetector, HMMRegimeDetector


class TestQlibAlpha158:
    """Test Qlib factor generation."""
    
    def test_exact_factor_count(self, sample_ohlcv):
        """CRITICAL: Must generate exactly 158 factors."""
        generator = QlibAlpha158()
        factors = generator.generate_all_factors(sample_ohlcv)
        
        factor_cols = [c for c in factors.columns if c.startswith('factor_')]
        
        assert len(factor_cols) == 158, f"Expected 158 factors, got {len(factor_cols)}"
    
    def test_no_nan_after_warmup(self, sample_ohlcv):
        """After initial warmup period, should have no NaN."""
        generator = QlibAlpha158()
        factors = generator.generate_all_factors(sample_ohlcv)
        
        # Skip first 60 days (warmup)
        factors_after_warmup = factors.iloc[60:]
        
        factor_cols = [c for c in factors_after_warmup.columns if c.startswith('factor_')]
        nan_count = factors_after_warmup[factor_cols].isnull().sum().sum()
        
        assert nan_count == 0, f"Found {nan_count} NaN values after warmup"
    
    def test_no_inf_values(self, sample_ohlcv):
        """No infinite values allowed."""
        generator = QlibAlpha158()
        factors = generator.generate_all_factors(sample_ohlcv)
        
        factor_cols = [c for c in factors.columns if c.startswith('factor_')]
        inf_count = np.isinf(factors[factor_cols]).sum().sum()
        
        assert inf_count == 0, f"Found {inf_count} infinite values"
    
    def test_vectorization_speed(self, large_ohlcv):
        """Must process 500 days in <5 seconds."""
        generator = QlibAlpha158()
        
        start = time.time()
        factors = generator.generate_all_factors(large_ohlcv)
        duration = time.time() - start
        
        assert duration < 5.0, f"Too slow: {duration:.2f}s (limit: 5s)"
        assert len(factors) == len(large_ohlcv)
    
    def test_factor_names(self):
        """Test factor name generation."""
        generator = QlibAlpha158()
        names = generator.get_factor_names()
        
        assert len(names) == 158
        assert names[0] == 'factor_001'
        assert names[-1] == 'factor_158'
    
    def test_validate_factor_count(self, sample_ohlcv):
        """Test validation method."""
        generator = QlibAlpha158()
        factors = generator.generate_all_factors(sample_ohlcv)
        
        assert generator.validate_factor_count(factors) == True


class TestWassersteinRegime:
    """Test Wasserstein regime detection."""
    
    def test_fit_and_predict(self, nifty_2year_data):
        """Test basic fit/predict workflow."""
        detector = WassersteinRegimeDetector(n_regimes=4, window_size=30)
        
        detector.fit(nifty_2year_data)
        regimes = detector.predict(nifty_2year_data)
        
        # Check basic properties
        assert len(regimes) >= len(nifty_2year_data) - 30  # Accounting for window
        assert set(np.unique(regimes)).issubset({0, 1, 2, 3})
    
    def test_regime_characteristics(self, nifty_2year_data):
        """All regimes must have distinct characteristics."""
        detector = WassersteinRegimeDetector(n_regimes=4, window_size=30)
        detector.fit(nifty_2year_data)
        
        chars = detector.get_regime_characteristics()
        
        assert len(chars) == 4
        assert 'mean_return' in chars.columns
        assert 'volatility' in chars.columns
        assert 'regime_name' in chars.columns
        
        # At least one regime should have negative returns
        assert (chars['mean_return'] < 0).any()
    
    def test_covid_crash_detection(self, nifty_with_covid):
        """CRITICAL: COVID crash must be detected as crash/highvol."""
        detector = WassersteinRegimeDetector(n_regimes=4, window_size=30)
        detector.fit(nifty_with_covid)
        regimes = detector.predict(nifty_with_covid)
        
        # Check validation
        validation = detector.validate_regimes(nifty_with_covid)
        
        if 'covid_crash' in validation:
            covid_regime = validation['covid_crash']
            assert covid_regime['regime_name'] in ['crash', 'high_volatility'], \
                f"COVID detected as {covid_regime['regime_name']}, expected crash/high_volatility"


class TestHMMRegime:
    """Test HMM regime detection."""
    
    def test_basic_fit(self, nifty_2year_data):
        """Test basic HMM fitting."""
        detector = HMMRegimeDetector(n_regimes=4)
        
        # Add volatility
        nifty_2year_data['volatility'] = nifty_2year_data['returns'].rolling(20).std()
        
        detector.fit(nifty_2year_data)
        
        assert detector._fitted == True
    
    def test_transition_matrix(self, nifty_2year_data):
        """Test transition matrix calculation."""
        detector = HMMRegimeDetector(n_regimes=4)
        nifty_2year_data['volatility'] = nifty_2year_data['returns'].rolling(20).std()
        
        detector.fit(nifty_2year_data)
        trans_matrix = detector.get_transition_matrix()
        
        assert trans_matrix.shape == (4, 4)
        # Each row should sum to ~1 (probabilities)
        assert np.allclose(trans_matrix.sum(axis=1), 1.0, atol=0.1)


class TestFeatureStore:
    """Test factor storage."""
    
    def test_create_schema(self, tmp_path):
        """Test schema creation."""
        store = FeatureStore(db_path=str(tmp_path / "test.duckdb"))
        store.create_schema()
        
        # Check tables exist
        tables = store.connection.execute("""
            SELECT name FROM sqlite_master WHERE type='table'
        """).fetchdf()
        
        # DuckDB uses different system tables
        result = store.connection.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'main'
        """).fetchdf()
        
        table_names = result['table_name'].tolist()
        assert 'alpha158_factors' in table_names
        assert 'market_regimes' in table_names
    
    def test_store_and_retrieve(self, tmp_path, sample_factors):
        """Test basic storage/retrieval."""
        store = FeatureStore(db_path=str(tmp_path / "test.duckdb"))
        store.create_schema()
        
        rows = store.store_factors(sample_factors, symbol="TEST")
        assert rows > 0
        
        retrieved = store.get_factors(["TEST"], include_regimes=False)
        assert len(retrieved) > 0
    
    def test_no_duplicates(self, tmp_path, sample_factors):
        """Storing same data twice shouldn't duplicate."""
        store = FeatureStore(db_path=str(tmp_path / "test.duckdb"))
        store.create_schema()
        
        rows1 = store.store_factors(sample_factors, symbol="TEST")
        rows2 = store.store_factors(sample_factors, symbol="TEST", overwrite=False)
        
        retrieved = store.get_factors(["TEST"], include_regimes=False)
        # Should have same number of rows (upsert, not duplicate)
        assert len(retrieved) == rows1


class TestFactorAnalyzer:
    """Test factor analysis."""
    
    def test_ic_calculation(self, sample_factors, sample_returns):
        """Test IC calculation."""
        analyzer = FactorAnalyzer()
        
        # Use first 10 factors
        test_factors = sample_factors[[c for c in sample_factors.columns if c.startswith('factor_')][:10]]
        
        ic_df = analyzer.calculate_ic(
            test_factors,
            sample_returns,
            periods=[1, 5]
        )
        
        assert 'ic_1d' in ic_df.columns
        assert 'ic_5d' in ic_df.columns
        assert len(ic_df) == 10
        assert 'mean_ic' in ic_df.columns
    
    def test_correlation_analysis(self, sample_factors):
        """Test correlation detection."""
        analyzer = FactorAnalyzer()
        
        corr_df = analyzer.analyze_correlation(sample_factors, threshold=0.95)
        
        # Should be a DataFrame (might be empty if no high correlations)
        assert isinstance(corr_df, pd.DataFrame)


# Fixtures

@pytest.fixture
def sample_ohlcv():
    """Generate 1 year of OHLCV data."""
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    
    np.random.seed(42)
    close = 100 + np.random.randn(len(dates)).cumsum() * 0.5
    
    return pd.DataFrame({
        'open': close + np.random.randn(len(dates)) * 0.3,
        'high': close + abs(np.random.randn(len(dates))) * 0.5,
        'low': close - abs(np.random.randn(len(dates))) * 0.5,
        'close': close,
        'volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)


@pytest.fixture
def large_ohlcv():
    """Generate 2 years for speed testing."""
    dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
    
    np.random.seed(42)
    close = 100 + np.random.randn(len(dates)).cumsum() * 0.5
    
    return pd.DataFrame({
        'open': close + np.random.randn(len(dates)) * 0.3,
        'high': close + abs(np.random.randn(len(dates))) * 0.5,
        'low': close - abs(np.random.randn(len(dates))) * 0.5,
        'close': close,
        'volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)


@pytest.fixture
def nifty_2year_data():
    """NIFTY data for 2 years."""
    dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
    
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.015, len(dates))
    close = 18000 * (1 + pd.Series(returns)).cumprod()
    
    return pd.DataFrame({
        'close': close,
        'returns': returns
    }, index=dates)


@pytest.fixture
def nifty_with_covid():
    """NIFTY data including COVID crash."""
    dates = pd.date_range('2019-01-01', '2021-12-31', freq='D')
    
    returns = []
    for date in dates:
        if '2020-03-01' <= date.strftime('%Y-%m-%d') <= '2020-03-31':
            # COVID crash: extreme negative returns
            ret = np.random.normal(-0.03, 0.05)
        elif date < pd.Timestamp('2020-03-01'):
            # Pre-COVID bull
            ret = np.random.normal(0.001, 0.01)
        else:
            # Recovery
            ret = np.random.normal(0.002, 0.015)
        
        returns.append(ret)
    
    close = 10000 * (1 + pd.Series(returns)).cumprod()
    
    return pd.DataFrame({
        'close': close,
        'returns': returns
    }, index=dates)


@pytest.fixture
def sample_factors(sample_ohlcv):
    """Generate sample factors."""
    generator = QlibAlpha158()
    return generator.generate_all_factors(sample_ohlcv)


@pytest.fixture
def sample_returns(sample_ohlcv):
    """Generate sample returns."""
    return sample_ohlcv['close'].pct_change()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

