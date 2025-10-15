# Qlib Model Training and Backtesting Implementation

Complete implementation of machine learning-based alpha generation, portfolio optimization, and backtesting infrastructure for NSE markets.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Components](#components)
4. [Quick Start](#quick-start)
5. [Usage Examples](#usage-examples)
6. [Configuration](#configuration)
7. [Testing](#testing)
8. [Performance Validation](#performance-validation)

---

## Overview

This implementation provides a production-ready quantitative trading framework built on:

- **158 Alpha-158 Factors**: Comprehensive factor library for NSE stocks
- **Machine Learning Models**: LightGBM and XGBoost for return prediction
- **Portfolio Optimization**: Mean-variance, max Sharpe, risk parity, min variance
- **Regime Adaptation**: Dynamic strategy switching based on market conditions
- **Vectorized Backtesting**: Fast, accurate backtesting with transaction costs
- **Performance Analytics**: Comprehensive metrics including Sharpe, Sortino, Calmar ratios

### Key Features

- Time-series aware data splitting (no lookahead bias)
- Regime-specific model training (4 market regimes)
- Convex portfolio optimization with constraints
- Transaction cost modeling (commission + slippage)
- Comprehensive performance attribution

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Data Layer                              │
│  FeatureStore (DuckDB) → 158 Factors + Regime Labels         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   Model Training Layer                        │
│  QlibModelTrainer → LightGBM/XGBoost on Alpha Factors        │
│  Regime-specific models (Bull/Bear/HighVol/Crash)            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   Signal Generation Layer                     │
│  AlphaSignalGenerator → Predictions → Rank Signals           │
│  RegimeAdaptiveStrategy → Dynamic parameter switching        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                Portfolio Optimization Layer                   │
│  PortfolioOptimizer → Optimal weights (cvxpy)                │
│  Methods: MeanVariance, MaxSharpe, RiskParity, MinVariance   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Backtesting Layer                          │
│  BacktestEngine → Vectorized execution + transaction costs   │
│  PerformanceAnalyzer → Sharpe, Sortino, Drawdowns           │
└─────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. QlibModelTrainer

**File:** `src/qlib_models/model_trainer.py`

Trains ML models on 158 alpha factors to predict forward returns.

**Key Methods:**
- `prepare_data()`: Chronological train/valid/test split
- `train()`: Train with early stopping
- `evaluate()`: IC, R2, direction accuracy
- `train_regime_models()`: Train separate model per regime
- `save_model()` / `load_model()`: Model persistence

**Usage:**
```python
from src.qlib_models import QlibModelTrainer

trainer = QlibModelTrainer(model_type='lightgbm')
X_train, y_train, X_valid, y_valid, X_test, y_test = trainer.prepare_data(
    factors, forward_horizon=5, train_ratio=0.6, valid_ratio=0.2
)
model = trainer.train(X_train, y_train, X_valid, y_valid)
metrics = trainer.evaluate(X_test, y_test, model)
```

### 2. AlphaSignalGenerator

**File:** `src/qlib_models/signal_generator.py`

Converts model predictions to actionable trading signals.

**Key Methods:**
- `generate_predictions()`: Raw return predictions
- `generate_rank_signals()`: Percentile/zscore ranking
- `generate_long_short_signals()`: Top/bottom quantile positions
- `generate_regime_adaptive_signals()`: Regime-specific signals

**Usage:**
```python
from src.qlib_models import AlphaSignalGenerator

generator = AlphaSignalGenerator(models={'default': model})
predictions = generator.generate_predictions(factors)
signals = generator.generate_long_short_signals(
    predictions, long_pct=0.2, short_pct=0.2
)
```

### 3. PortfolioOptimizer

**File:** `src/portfolio/optimizer.py`

Portfolio weight optimization using convex optimization (cvxpy).

**Methods:**
- `optimize_mean_variance()`: Markowitz optimization
- `optimize_max_sharpe()`: Maximum Sharpe ratio
- `optimize_risk_parity()`: Equal risk contribution
- `optimize_min_variance()`: Minimum variance (defensive)

**Constraints:**
- Long-only or long-short
- Position limits (max/min weight per stock)
- Leverage limits
- Turnover constraints

**Usage:**
```python
from src.portfolio import PortfolioOptimizer

optimizer = PortfolioOptimizer(risk_free_rate=0.05)
weights = optimizer.optimize_mean_variance(
    expected_returns, 
    cov_matrix, 
    risk_aversion=1.0,
    constraints={'max_position': 0.1, 'long_only': False}
)
```

### 4. BacktestEngine

**File:** `src/backtesting/backtest_engine.py`

Vectorized backtesting with realistic transaction costs.

**Features:**
- Fully vectorized (fast)
- Commission + slippage modeling
- Daily/weekly/monthly rebalancing
- Position tracking
- Regime-aware execution

**Usage:**
```python
from src.backtesting import BacktestEngine

engine = BacktestEngine(
    initial_capital=1000000,
    commission=0.001,  # 0.1%
    slippage=0.0005    # 0.05%
)
results = engine.run_backtest(
    signals=signals,
    prices=prices,
    rebalance_freq='daily'
)
```

### 5. PerformanceAnalyzer

**File:** `src/backtesting/performance_analyzer.py`

Comprehensive performance metrics calculation.

**Metrics:**
- Returns: Total, annualized, CAGR
- Risk: Volatility, downside deviation, VaR, CVaR
- Risk-adjusted: Sharpe, Sortino, Calmar ratios
- Drawdowns: Max, average, duration
- Regime-specific breakdown

**Usage:**
```python
from src.backtesting import PerformanceAnalyzer

analyzer = PerformanceAnalyzer(risk_free_rate=0.05)
report = analyzer.generate_report(
    returns=results['daily_return'],
    regime_labels=regime_labels
)
analyzer.print_summary(report)
```

### 6. RegimeAdaptiveStrategy

**File:** `src/strategies/regime_adaptive_strategy.py`

Strategy that switches models and parameters based on market regime.

**Regime Configurations:**
- **Bull**: Aggressive (30% long, 10% short, max Sharpe)
- **Bear**: Defensive (10% long, 30% short, min variance)
- **High Vol**: Reduced exposure (15% each, risk parity)
- **Crash**: Minimal exposure (5% each, min variance)

**Usage:**
```python
from src.strategies import RegimeAdaptiveStrategy

strategy = RegimeAdaptiveStrategy(
    regime_models=regime_models,
    regime_detector=detector
)
signals = strategy.generate_adaptive_signals_history(
    factors=factors,
    regime_history=regime_labels
)
```

---

## Quick Start

### 1. Train Models

Train a single LightGBM model:

```bash
python scripts/train_models.py \
    --symbols NIFTY50 \
    --model lightgbm \
    --forward-horizon 5
```

Train regime-adaptive models:

```bash
python scripts/train_models.py \
    --regime-adaptive \
    --forward-horizon 5
```

### 2. Run Backtest

Simple backtest:

```bash
python scripts/run_backtest.py \
    --start 2023-01-01 \
    --end 2024-12-31 \
    --symbols NIFTY50
```

Regime-adaptive backtest:

```bash
python scripts/run_backtest.py \
    --start 2023-01-01 \
    --end 2024-12-31 \
    --regime-adaptive \
    --rebalance weekly
```

### 3. Run Tests

```bash
pytest tests/unit/test_qlib_models.py -v
```

---

## Usage Examples

### Example 1: Train and Evaluate Model

```python
from src.feature_engineering import FeatureStore
from src.qlib_models import QlibModelTrainer

# Load factors
store = FeatureStore(db_path='data/trading_db.duckdb')
factors = store.get_factors(
    symbols=['RELIANCE', 'TCS', 'INFY'],
    start_date='2023-01-01',
    end_date='2024-12-31'
)

# Train model
trainer = QlibModelTrainer(model_type='lightgbm')
X_train, y_train, X_valid, y_valid, X_test, y_test = trainer.prepare_data(
    factors, forward_horizon=5
)
model = trainer.train(X_train, y_train, X_valid, y_valid)

# Evaluate
metrics = trainer.evaluate(X_test, y_test, model)
print(f"Test IC: {metrics['ic']:.4f}")
print(f"Direction Accuracy: {metrics['direction_accuracy']:.2%}")

# Save model
trainer.save_model(model, 'models/my_model.pkl', metadata=metrics)
```

### Example 2: Generate Signals and Backtest

```python
from src.qlib_models import AlphaSignalGenerator
from src.backtesting import BacktestEngine, PerformanceAnalyzer

# Generate signals
generator = AlphaSignalGenerator(models={'default': model})
predictions = generator.generate_predictions(factors)
signals = generator.generate_long_short_signals(predictions, long_pct=0.2, short_pct=0.2)

# Backtest
engine = BacktestEngine(initial_capital=1000000, commission=0.001, slippage=0.0005)
results = engine.run_backtest(signals, prices, rebalance_freq='daily')

# Analyze
analyzer = PerformanceAnalyzer(risk_free_rate=0.05)
report = analyzer.generate_report(results['daily_return'])
analyzer.print_summary(report)
```

### Example 3: Portfolio Optimization

```python
from src.portfolio import PortfolioOptimizer
import pandas as pd

# Calculate expected returns and covariance
expected_returns = predictions.groupby(level='symbol').mean()
returns_matrix = factors.pivot_table(
    index='date', columns='symbol', values='close'
).pct_change()
cov_matrix = returns_matrix.cov() * 252  # Annualized

# Optimize
optimizer = PortfolioOptimizer(risk_free_rate=0.05)
weights = optimizer.optimize_mean_variance(
    expected_returns, 
    cov_matrix,
    risk_aversion=1.0,
    constraints={'max_position': 0.1}
)
print(weights)
```

### Example 4: Regime-Adaptive Strategy

```python
from src.strategies import RegimeAdaptiveStrategy
from src.regime_detection import WassersteinRegimeDetector

# Detect regimes
detector = WassersteinRegimeDetector(n_regimes=4)
detector.fit(market_data)

# Train regime models
regime_models = trainer.train_regime_models(factors, detector._regime_labels)

# Create strategy
strategy = RegimeAdaptiveStrategy(
    regime_models=regime_models,
    regime_detector=detector
)

# Generate adaptive signals
signals = strategy.generate_adaptive_signals_history(
    factors=factors,
    regime_history=regime_labels
)

# Backtest
results = engine.run_backtest(signals, prices, regime_labels=regime_labels)
```

---

## Configuration

All settings are in `config/qlib_model_config.yaml`:

### Key Sections:

1. **Training Parameters**: Model type, horizon, split ratios
2. **Model Hyperparameters**: LightGBM and XGBoost configs
3. **Signal Generation**: Ranking method, long/short percentiles
4. **Portfolio Optimization**: Method, risk aversion, constraints
5. **Backtesting**: Capital, costs, rebalancing frequency
6. **Regime-Specific**: Configs for each market regime

Example configuration:

```yaml
training:
  model_type: lightgbm
  forward_horizon: 5
  train_ratio: 0.6
  valid_ratio: 0.2

portfolio:
  optimization_method: mean_variance
  risk_aversion: 1.0
  constraints:
    max_position: 0.1
    long_only: false

backtesting:
  initial_capital: 1000000
  commission: 0.001
  slippage: 0.0005
  rebalance_freq: daily
```

---

## Testing

### Run All Tests

```bash
pytest tests/unit/test_qlib_models.py -v
```

### Test Coverage

The test suite covers:

1. **QlibModelTrainer**:
   - Time-series split validation
   - No lookahead bias
   - Model training and evaluation
   - Feature importance extraction

2. **AlphaSignalGenerator**:
   - Prediction generation
   - Rank signal calculation
   - Long/short portfolio construction

3. **PortfolioOptimizer**:
   - Mean-variance optimization
   - Max Sharpe optimization
   - Constraint handling

4. **BacktestEngine**:
   - Backtest execution
   - Transaction cost application
   - Position tracking

5. **PerformanceAnalyzer**:
   - Return metrics
   - Risk metrics
   - Risk-adjusted metrics
   - Drawdown calculation

6. **RegimeAdaptiveStrategy**:
   - Strategy initialization
   - Regime configuration
   - Position limits

### Key Test Cases

```python
# Test time-series integrity
def test_no_lookahead_bias():
    # Ensures train dates < valid dates < test dates
    assert train_dates.max() <= valid_dates.min()
    assert valid_dates.max() <= test_dates.min()

# Test transaction costs
def test_transaction_costs():
    # Verifies costs reduce returns
    assert results_with_cost <= results_without_cost

# Test optimization constraints
def test_constraints():
    # Ensures weights respect constraints
    assert (weights >= -1e-6).all()  # Long-only
    assert (weights <= 0.3).all()    # Max position
```

---

## Performance Validation

### Expected Metrics

Based on typical quantitative strategies on NSE:

```
TRAINING:
  Test IC:              0.030 - 0.080 (acceptable range)
  Test R2:              0.05 - 0.15
  Direction Accuracy:   52% - 58%

BACKTESTING:
  Sharpe Ratio:         > 1.0 (good), > 1.5 (excellent)
  Sortino Ratio:        > 1.3 (good), > 2.0 (excellent)
  Calmar Ratio:         > 0.8 (good), > 1.2 (excellent)
  Max Drawdown:         < -20% (acceptable), < -15% (good)
  Win Rate:             52% - 60%
```

### Validation Commands

```bash
# 1. Train models
python scripts/train_models.py --symbols NIFTY50 --model lightgbm

# 2. Run backtest
python scripts/run_backtest.py --start 2023-01-01 --end 2024-12-31

# 3. Validate results
python -c "
from src.backtesting import PerformanceAnalyzer
import pandas as pd

results = pd.read_csv('reports/backtest_results.csv', index_col=0, parse_dates=True)
analyzer = PerformanceAnalyzer()
metrics = analyzer.calculate_risk_adjusted_metrics(results['daily_return'])

print(f'Sharpe Ratio: {metrics[\"sharpe_ratio\"]:.2f}')
print(f'Sortino Ratio: {metrics[\"sortino_ratio\"]:.2f}')
print(f'Calmar Ratio: {metrics[\"calmar_ratio\"]:.2f}')

assert metrics['sharpe_ratio'] > 0, 'Sharpe should be positive'
print('All validation checks passed')
"
```

### Success Criteria

The implementation is successful if:

1. **Training completes without errors**
   - Models train in < 60 seconds on 500 days of data
   - Test IC in acceptable range (0.03 - 0.08)
   - No lookahead bias detected

2. **Backtesting produces valid results**
   - Backtest runs in < 30 seconds for 1 year
   - Sharpe ratio > 0.5
   - Transaction costs properly applied
   - No unrealistic returns (> 100% annual)

3. **Tests pass**
   - All unit tests pass
   - No linter errors
   - Code coverage > 80%

4. **Documentation complete**
   - All functions have docstrings
   - Usage examples work
   - Configuration documented

---

## File Structure

```
src/
├── qlib_models/
│   ├── __init__.py
│   ├── model_trainer.py        # ML model training
│   └── signal_generator.py     # Signal generation
├── portfolio/
│   ├── __init__.py
│   └── optimizer.py            # Portfolio optimization
├── backtesting/
│   ├── __init__.py
│   ├── backtest_engine.py      # Backtesting engine
│   └── performance_analyzer.py # Performance metrics
└── strategies/
    ├── __init__.py
    └── regime_adaptive_strategy.py  # Regime strategy

scripts/
├── train_models.py             # Training pipeline
└── run_backtest.py             # Backtesting pipeline

config/
└── qlib_model_config.yaml      # Configuration

tests/
└── unit/
    └── test_qlib_models.py     # Comprehensive tests

models/                          # Saved models
reports/                         # Backtest results
```

---

## Next Steps

1. **Run Training**: Train models on your factor data
2. **Validate Performance**: Run backtests and verify metrics
3. **Tune Parameters**: Adjust hyperparameters in config
4. **Integrate with RL**: Feed signals to RL execution layer
5. **Deploy Live**: Connect to Zerodha for live trading

---

## Support

For issues or questions:
- Check logs in `logs/` directory
- Review test failures for debugging
- Verify data quality in feature store
- Ensure all dependencies installed

## License

Proprietary - NSE Adaptive Regime Trading System

