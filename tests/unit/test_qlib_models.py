"""
Unit Tests for Qlib Models and Backtesting.

Tests all new components:
- QlibModelTrainer
- AlphaSignalGenerator
- PortfolioOptimizer
- BacktestEngine
- PerformanceAnalyzer
- RegimeAdaptiveStrategy
"""

import numpy as np
import pandas as pd
import pytest

from src.backtesting.backtest_engine import BacktestEngine
from src.backtesting.performance_analyzer import PerformanceAnalyzer
from src.portfolio.optimizer import PortfolioOptimizer
from src.qlib_models.model_trainer import QlibModelTrainer
from src.qlib_models.signal_generator import AlphaSignalGenerator
from src.regime_detection.wasserstein_regime import WassersteinRegimeDetector
from src.strategies.regime_adaptive_strategy import RegimeAdaptiveStrategy


class TestQlibModelTrainer:
    """Test QlibModelTrainer."""
    
    @pytest.fixture
    def sample_factors(self):
        """Create sample factor data."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        symbols = ['RELIANCE', 'TCS', 'INFY']
        
        data = []
        for symbol in symbols:
            for date in dates:
                row = {'symbol': symbol, 'date': date, 'close': 100 + np.random.randn() * 5}
                for i in range(1, 159):
                    row[f'factor_{i:03d}'] = np.random.randn()
                data.append(row)
        
        df = pd.DataFrame(data)
        df = df.set_index(['symbol', 'date'])
        return df
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        trainer = QlibModelTrainer(model_type='lightgbm')
        assert trainer.model_type == 'lightgbm'
        assert trainer.config is not None
        
        with pytest.raises(ValueError):
            QlibModelTrainer(model_type='invalid')
    
    def test_data_preparation(self, sample_factors):
        """Test data preparation with time-series split."""
        trainer = QlibModelTrainer(model_type='lightgbm')
        
        X_train, y_train, X_valid, y_valid, X_test, y_test = trainer.prepare_data(
            sample_factors,
            forward_horizon=5,
            train_ratio=0.6,
            valid_ratio=0.2
        )
        
        # Check sizes
        total_samples = len(X_train) + len(X_valid) + len(X_test)
        assert len(X_train) == pytest.approx(total_samples * 0.6, rel=0.1)
        assert len(X_valid) == pytest.approx(total_samples * 0.2, rel=0.1)
        
        # Check no NaN
        assert not X_train.isna().any().any()
        assert not y_train.isna().any()
        
        # Check feature columns
        assert all(col.startswith('factor_') for col in X_train.columns)
    
    def test_no_lookahead_bias(self, sample_factors):
        """Test that there's no lookahead bias in data preparation."""
        trainer = QlibModelTrainer(model_type='lightgbm')
        
        X_train, y_train, X_valid, y_valid, X_test, y_test = trainer.prepare_data(
            sample_factors,
            forward_horizon=5,
            train_ratio=0.6,
            valid_ratio=0.2
        )
        
        # Check chronological split
        if isinstance(X_train.index, pd.MultiIndex):
            train_dates = X_train.index.get_level_values('date')
            valid_dates = X_valid.index.get_level_values('date')
            test_dates = X_test.index.get_level_values('date')
            
            assert train_dates.max() <= valid_dates.min()
            assert valid_dates.max() <= test_dates.min()
    
    def test_model_training(self, sample_factors):
        """Test model training."""
        trainer = QlibModelTrainer(model_type='lightgbm')
        
        X_train, y_train, X_valid, y_valid, X_test, y_test = trainer.prepare_data(
            sample_factors,
            forward_horizon=5
        )
        
        model = trainer.train(X_train, y_train, X_valid, y_valid, early_stopping_rounds=10)
        
        assert model is not None
        assert hasattr(model, 'predict')
    
    def test_model_evaluation(self, sample_factors):
        """Test model evaluation."""
        trainer = QlibModelTrainer(model_type='lightgbm')
        
        X_train, y_train, X_valid, y_valid, X_test, y_test = trainer.prepare_data(
            sample_factors,
            forward_horizon=5
        )
        
        model = trainer.train(X_train, y_train, X_valid, y_valid, early_stopping_rounds=10)
        metrics = trainer.evaluate(X_test, y_test, model)
        
        assert 'ic' in metrics
        assert 'mse' in metrics
        assert 'r2' in metrics
        assert 'direction_accuracy' in metrics
        
        # IC should be between -1 and 1
        assert -1 <= metrics['ic'] <= 1


class TestAlphaSignalGenerator:
    """Test AlphaSignalGenerator."""
    
    @pytest.fixture
    def sample_model(self, sample_factors):
        """Create a simple trained model."""
        trainer = QlibModelTrainer(model_type='lightgbm')
        X_train, y_train, X_valid, y_valid, X_test, y_test = trainer.prepare_data(
            sample_factors,
            forward_horizon=5
        )
        model = trainer.train(X_train, y_train, X_valid, y_valid, early_stopping_rounds=10)
        return {'model': model, 'metadata': {'model_type': 'lightgbm'}}
    
    @pytest.fixture
    def sample_factors(self):
        """Create sample factors."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        symbols = ['RELIANCE', 'TCS', 'INFY']
        
        data = []
        for symbol in symbols:
            for date in dates:
                row = {'symbol': symbol, 'date': date, 'close': 100 + np.random.randn() * 5}
                for i in range(1, 159):
                    row[f'factor_{i:03d}'] = np.random.randn()
                data.append(row)
        
        df = pd.DataFrame(data).set_index(['symbol', 'date'])
        return df
    
    def test_signal_generation(self, sample_model, sample_factors):
        """Test signal generation."""
        generator = AlphaSignalGenerator(models={'default': sample_model})
        
        predictions = generator.generate_predictions(sample_factors, model_name='default')
        
        assert len(predictions) > 0
        assert predictions.dtype in [np.float64, np.float32]
    
    def test_rank_signals(self, sample_model, sample_factors):
        """Test rank signal generation."""
        generator = AlphaSignalGenerator(models={'default': sample_model})
        
        predictions = generator.generate_predictions(sample_factors, model_name='default')
        rank_signals = generator.generate_rank_signals(predictions, method='percentile')
        
        # Percentile ranks should be between 0 and 1
        assert rank_signals.min() >= 0
        assert rank_signals.max() <= 1
    
    def test_long_short_signals(self, sample_model, sample_factors):
        """Test long/short signal generation."""
        generator = AlphaSignalGenerator(models={'default': sample_model})
        
        predictions = generator.generate_predictions(sample_factors, model_name='default')
        signals = generator.generate_long_short_signals(
            predictions,
            long_pct=0.3,
            short_pct=0.3
        )
        
        assert 'signal' in signals.columns
        assert 'weight' in signals.columns
        
        # Check signal values
        assert set(signals['signal'].unique()).issubset({-1, 0, 1})
        
        # Check weights sum close to 0 (long-short)
        assert abs(signals['weight'].sum()) < 0.1


class TestPortfolioOptimizer:
    """Test PortfolioOptimizer."""
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample expected returns and covariance."""
        np.random.seed(42)
        symbols = ['A', 'B', 'C', 'D', 'E']
        
        expected_returns = pd.Series(
            np.random.randn(5) * 0.001,
            index=symbols,
            name='expected_return'
        )
        
        # Create positive semi-definite covariance matrix
        random_matrix = np.random.randn(5, 5)
        cov_matrix = pd.DataFrame(
            random_matrix @ random_matrix.T * 0.0001,
            index=symbols,
            columns=symbols
        )
        
        return expected_returns, cov_matrix
    
    def test_mean_variance_optimization(self, sample_returns):
        """Test mean-variance optimization."""
        expected_returns, cov_matrix = sample_returns
        
        optimizer = PortfolioOptimizer(risk_free_rate=0.05)
        weights = optimizer.optimize_mean_variance(
            expected_returns,
            cov_matrix,
            risk_aversion=1.0
        )
        
        # Weights should sum to 1
        assert abs(weights.sum() - 1.0) < 1e-4
        
        # Check index matches
        assert list(weights.index) == list(expected_returns.index)
    
    def test_max_sharpe_optimization(self, sample_returns):
        """Test maximum Sharpe optimization."""
        expected_returns, cov_matrix = sample_returns
        
        optimizer = PortfolioOptimizer(risk_free_rate=0.05)
        weights = optimizer.optimize_max_sharpe(
            expected_returns,
            cov_matrix
        )
        
        assert abs(weights.sum() - 1.0) < 1e-4
    
    def test_constraints(self, sample_returns):
        """Test constraint handling."""
        expected_returns, cov_matrix = sample_returns
        
        optimizer = PortfolioOptimizer(risk_free_rate=0.05)
        
        # Long-only constraint
        weights = optimizer.optimize_mean_variance(
            expected_returns,
            cov_matrix,
            constraints={'long_only': True}
        )
        
        assert (weights >= -1e-6).all()  # All non-negative (with tolerance)
        
        # Max position constraint
        weights = optimizer.optimize_mean_variance(
            expected_returns,
            cov_matrix,
            constraints={'max_position': 0.3}
        )
        
        assert (weights <= 0.3 + 1e-6).all()


class TestBacktestEngine:
    """Test BacktestEngine."""
    
    @pytest.fixture
    def sample_signals(self):
        """Create sample signals."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        signals = []
        
        for date in dates:
            signals.append({
                'date': date,
                'symbol': 'RELIANCE',
                'signal': 1,
                'weight': 0.5
            })
            signals.append({
                'date': date,
                'symbol': 'TCS',
                'signal': -1,
                'weight': -0.5
            })
        
        return pd.DataFrame(signals)
    
    @pytest.fixture
    def sample_prices(self):
        """Create sample prices."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        prices = pd.DataFrame({
            'date': dates.repeat(2),
            'symbol': ['RELIANCE', 'TCS'] * 10,
            'close': np.random.uniform(90, 110, 20)
        })
        return prices
    
    def test_backtest_execution(self, sample_signals, sample_prices):
        """Test backtest execution."""
        engine = BacktestEngine(
            initial_capital=1000000,
            commission=0.001,
            slippage=0.0005
        )
        
        results = engine.run_backtest(
            signals=sample_signals,
            prices=sample_prices,
            rebalance_freq='daily'
        )
        
        assert len(results) > 0
        assert 'portfolio_value' in results.columns
        assert 'daily_return' in results.columns
        assert 'cumulative_return' in results.columns
    
    def test_transaction_costs(self, sample_signals, sample_prices):
        """Test that transaction costs are applied."""
        engine_no_cost = BacktestEngine(
            initial_capital=1000000,
            commission=0.0,
            slippage=0.0
        )
        
        engine_with_cost = BacktestEngine(
            initial_capital=1000000,
            commission=0.001,
            slippage=0.001
        )
        
        results_no_cost = engine_no_cost.run_backtest(
            sample_signals, sample_prices, rebalance_freq='daily'
        )
        
        results_with_cost = engine_with_cost.run_backtest(
            sample_signals, sample_prices, rebalance_freq='daily'
        )
        
        # With costs should have lower final value
        assert results_with_cost['portfolio_value'].iloc[-1] <= results_no_cost['portfolio_value'].iloc[-1]


class TestPerformanceAnalyzer:
    """Test PerformanceAnalyzer."""
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample returns."""
        np.random.seed(42)
        returns = pd.Series(
            np.random.randn(252) * 0.01,
            index=pd.date_range('2023-01-01', periods=252, freq='D'),
            name='returns'
        )
        return returns
    
    def test_return_metrics(self, sample_returns):
        """Test return metrics calculation."""
        analyzer = PerformanceAnalyzer(risk_free_rate=0.05)
        
        metrics = analyzer.calculate_returns_metrics(sample_returns)
        
        assert 'total_return' in metrics
        assert 'annualized_return' in metrics
        assert 'cagr' in metrics
        assert 'best_day' in metrics
        assert 'worst_day' in metrics
    
    def test_risk_metrics(self, sample_returns):
        """Test risk metrics calculation."""
        analyzer = PerformanceAnalyzer(risk_free_rate=0.05)
        
        metrics = analyzer.calculate_risk_metrics(sample_returns)
        
        assert 'volatility' in metrics
        assert 'max_drawdown' in metrics
        assert 'var_95' in metrics
        
        # Volatility should be positive
        assert metrics['volatility'] > 0
        
        # Max drawdown should be negative
        assert metrics['max_drawdown'] <= 0
    
    def test_risk_adjusted_metrics(self, sample_returns):
        """Test risk-adjusted metrics."""
        analyzer = PerformanceAnalyzer(risk_free_rate=0.05)
        
        metrics = analyzer.calculate_risk_adjusted_metrics(sample_returns)
        
        assert 'sharpe_ratio' in metrics
        assert 'sortino_ratio' in metrics
        assert 'calmar_ratio' in metrics
    
    def test_drawdown_calculation(self, sample_returns):
        """Test drawdown period calculation."""
        analyzer = PerformanceAnalyzer(risk_free_rate=0.05)
        
        equity_curve = (1 + sample_returns).cumprod()
        drawdowns = analyzer.calculate_drawdowns(equity_curve)
        
        if len(drawdowns) > 0:
            assert 'start_date' in drawdowns.columns
            assert 'end_date' in drawdowns.columns
            assert 'depth' in drawdowns.columns
            
            # Drawdown depth should be negative
            assert (drawdowns['depth'] <= 0).all()


class TestRegimeAdaptiveStrategy:
    """Test RegimeAdaptiveStrategy."""
    
    @pytest.fixture
    def sample_regime_models(self):
        """Create sample regime models."""
        # Create simple mock models
        models = {}
        for regime in range(4):
            models[f'regime_{regime}'] = {
                'model': None,  # Mock model
                'metadata': {'regime': regime, 'model_type': 'lightgbm'}
            }
        return models
    
    def test_strategy_initialization(self, sample_regime_models):
        """Test strategy initialization."""
        strategy = RegimeAdaptiveStrategy(
            regime_models=sample_regime_models,
            regime_configs=None
        )
        
        assert len(strategy.regime_models) == 4
        assert len(strategy.regime_configs) > 0
    
    def test_regime_config_retrieval(self, sample_regime_models):
        """Test regime configuration retrieval."""
        strategy = RegimeAdaptiveStrategy(
            regime_models=sample_regime_models
        )
        
        for regime in range(4):
            config = strategy.get_regime_config(regime)
            assert 'name' in config
            assert 'long_pct' in config
            assert 'short_pct' in config
    
    def test_position_limits(self, sample_regime_models):
        """Test position limit retrieval."""
        strategy = RegimeAdaptiveStrategy(
            regime_models=sample_regime_models
        )
        
        limits = strategy.get_position_limits(regime=0)
        
        assert 'max_position' in limits
        assert limits['max_position'] > 0
    
    def test_regime_summary(self, sample_regime_models):
        """Test regime strategy summary."""
        strategy = RegimeAdaptiveStrategy(
            regime_models=sample_regime_models
        )
        
        summary = strategy.summarize_regime_strategy()
        
        assert len(summary) == 4  # 4 regimes
        assert 'regime' in summary.columns
        assert 'regime_name' in summary.columns

