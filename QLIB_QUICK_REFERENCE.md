# Qlib Models - Quick Reference

Essential commands and code snippets for model training and backtesting.

## Quick Commands

### 1. Train Single Model

```bash
# LightGBM (default)
python scripts/train_models.py --symbols NIFTY50 --model lightgbm --forward-horizon 5

# XGBoost
python scripts/train_models.py --symbols NIFTY50 --model xgboost --forward-horizon 5

# Custom date range
python scripts/train_models.py --symbols NIFTY50 --start-date 2023-01-01 --end-date 2024-12-31
```

### 2. Train Regime-Adaptive Models

```bash
# Train separate model for each regime
python scripts/train_models.py --regime-adaptive --forward-horizon 5

# With custom horizon
python scripts/train_models.py --regime-adaptive --forward-horizon 10
```

### 3. Run Backtest

```bash
# Simple backtest
python scripts/run_backtest.py --start 2023-01-01 --end 2024-12-31

# Regime-adaptive backtest
python scripts/run_backtest.py --start 2023-01-01 --end 2024-12-31 --regime-adaptive

# Weekly rebalancing
python scripts/run_backtest.py --start 2023-01-01 --end 2024-12-31 --rebalance weekly

# Custom parameters
python scripts/run_backtest.py \
    --start 2023-01-01 \
    --end 2024-12-31 \
    --initial-capital 2000000 \
    --commission 0.002 \
    --slippage 0.001 \
    --long-pct 0.3 \
    --short-pct 0.3
```

### 4. Run Tests

```bash
# All tests
pytest tests/unit/test_qlib_models.py -v

# Specific test class
pytest tests/unit/test_qlib_models.py::TestQlibModelTrainer -v

# With coverage
pytest tests/unit/test_qlib_models.py --cov=src --cov-report=html
```

---

## Code Snippets

### Train Model

```python
from src.feature_engineering import FeatureStore
from src.qlib_models import QlibModelTrainer

# Load data
store = FeatureStore()
factors = store.get_factors(
    symbols=['RELIANCE', 'TCS', 'INFY'],
    start_date='2023-01-01',
    end_date='2024-12-31'
)

# Train
trainer = QlibModelTrainer(model_type='lightgbm')
X_train, y_train, X_valid, y_valid, X_test, y_test = trainer.prepare_data(
    factors, forward_horizon=5, train_ratio=0.6, valid_ratio=0.2
)
model = trainer.train(X_train, y_train, X_valid, y_valid)

# Evaluate
metrics = trainer.evaluate(X_test, y_test, model)
print(f"IC: {metrics['ic']:.4f}, R2: {metrics['r2']:.4f}")

# Save
trainer.save_model(model, 'models/my_model.pkl', metadata=metrics)
```

### Generate Signals

```python
from src.qlib_models import AlphaSignalGenerator

# Load model
with open('models/my_model.pkl', 'rb') as f:
    model_dict = pickle.load(f)

# Generate signals
generator = AlphaSignalGenerator(models={'default': model_dict})
predictions = generator.generate_predictions(factors)
signals = generator.generate_long_short_signals(
    predictions, 
    long_pct=0.2,  # Top 20%
    short_pct=0.2  # Bottom 20%
)
```

### Optimize Portfolio

```python
from src.portfolio import PortfolioOptimizer

optimizer = PortfolioOptimizer(risk_free_rate=0.05)

# Mean-variance
weights = optimizer.optimize_mean_variance(
    expected_returns, 
    cov_matrix,
    risk_aversion=1.0,
    constraints={'max_position': 0.1, 'long_only': False}
)

# Max Sharpe
weights = optimizer.optimize_max_sharpe(
    expected_returns, 
    cov_matrix,
    constraints={'max_position': 0.1}
)

# Risk Parity
weights = optimizer.optimize_risk_parity(cov_matrix)

# Min Variance
weights = optimizer.optimize_min_variance(cov_matrix)
```

### Run Backtest

```python
from src.backtesting import BacktestEngine, PerformanceAnalyzer

# Backtest
engine = BacktestEngine(
    initial_capital=1000000,
    commission=0.001,
    slippage=0.0005
)
results = engine.run_backtest(
    signals=signals,
    prices=prices,
    rebalance_freq='daily'
)

# Analyze
analyzer = PerformanceAnalyzer(risk_free_rate=0.05)
report = analyzer.generate_report(results['daily_return'])
analyzer.print_summary(report)

# Save
results.to_csv('reports/backtest_results.csv')
```

### Regime-Adaptive Strategy

```python
from src.strategies import RegimeAdaptiveStrategy

# Create strategy
strategy = RegimeAdaptiveStrategy(
    regime_models=regime_models,
    regime_detector=detector
)

# Generate signals
signals = strategy.generate_adaptive_signals_history(
    factors=factors,
    regime_history=regime_labels
)

# Backtest with regime awareness
results = engine.run_backtest(
    signals=signals,
    prices=prices,
    regime_labels=regime_labels,
    rebalance_freq='daily'
)
```

---

## Configuration

Edit `config/qlib_model_config.yaml`:

```yaml
# Model training
training:
  model_type: lightgbm
  forward_horizon: 5
  train_ratio: 0.6
  valid_ratio: 0.2

# Signal generation
signals:
  long_pct: 0.2
  short_pct: 0.2

# Portfolio optimization
portfolio:
  optimization_method: mean_variance
  risk_aversion: 1.0
  constraints:
    max_position: 0.1
    long_only: false

# Backtesting
backtesting:
  initial_capital: 1000000
  commission: 0.001
  slippage: 0.0005
  rebalance_freq: daily
```

---

## Performance Metrics

### From Python

```python
from src.backtesting import PerformanceAnalyzer
import pandas as pd

# Load results
results = pd.read_csv('reports/backtest_results.csv', index_col=0, parse_dates=True)

# Calculate metrics
analyzer = PerformanceAnalyzer()

# Returns
returns_metrics = analyzer.calculate_returns_metrics(results['daily_return'])
print(f"Total Return: {returns_metrics['total_return']:.2%}")
print(f"CAGR: {returns_metrics['cagr']:.2%}")

# Risk
risk_metrics = analyzer.calculate_risk_metrics(results['daily_return'])
print(f"Volatility: {risk_metrics['volatility']:.2%}")
print(f"Max Drawdown: {risk_metrics['max_drawdown']:.2%}")

# Risk-adjusted
risk_adj = analyzer.calculate_risk_adjusted_metrics(results['daily_return'])
print(f"Sharpe: {risk_adj['sharpe_ratio']:.3f}")
print(f"Sortino: {risk_adj['sortino_ratio']:.3f}")
print(f"Calmar: {risk_adj['calmar_ratio']:.3f}")
```

### From Command Line

```bash
# Quick validation
python -c "
from src.backtesting import PerformanceAnalyzer
import pandas as pd

results = pd.read_csv('reports/backtest_results.csv', index_col=0, parse_dates=True)
analyzer = PerformanceAnalyzer()
metrics = analyzer.calculate_risk_adjusted_metrics(results['daily_return'])

print(f'Sharpe: {metrics[\"sharpe_ratio\"]:.2f}')
print(f'Sortino: {metrics[\"sortino_ratio\"]:.2f}')
"
```

---

## Common Tasks

### Load Latest Model

```python
from pathlib import Path
import pickle

# Find latest model
models_dir = Path('models')
latest_model = max(models_dir.glob('lightgbm_*.pkl'), key=lambda p: p.stat().st_mtime)

# Load
with open(latest_model, 'rb') as f:
    model_dict = pickle.load(f)
    model = model_dict['model']
    metadata = model_dict['metadata']

print(f"Loaded: {latest_model}")
print(f"IC: {metadata['metrics']['ic']:.4f}")
```

### Feature Importance

```python
from src.qlib_models import QlibModelTrainer

trainer = QlibModelTrainer(model_type='lightgbm')
importance_df = trainer.get_feature_importance(
    model, 
    feature_names=X_train.columns.tolist(),
    top_n=20
)

print(importance_df)
importance_df.to_csv('reports/feature_importance.csv', index=False)
```

### Regime Summary

```python
from src.strategies import RegimeAdaptiveStrategy

strategy = RegimeAdaptiveStrategy(regime_models=models)
summary = strategy.summarize_regime_strategy()
print(summary)
```

### Export Results

```python
# Export to CSV
results.to_csv('reports/backtest_results.csv')

# Export performance report to JSON
import json
with open('reports/performance_report.json', 'w') as f:
    json.dump(report, f, indent=2, default=str)

# Export signals
signals.to_csv('reports/signals.csv', index=False)
```

---

## Troubleshooting

### No factor data found
```bash
# Check database
python -c "
from src.feature_engineering import FeatureStore
store = FeatureStore()
stats = store.get_database_stats()
print(stats)
"
```

### Low IC values
- Increase forward horizon (5 → 10 days)
- Try different model (lightgbm → xgboost)
- Check for data quality issues
- Ensure regime labels are aligned

### Optimization fails
- Check covariance matrix is positive definite
- Reduce number of assets
- Relax constraints
- Try different optimization method

### Backtest errors
- Ensure signals have 'date' column
- Check price data alignment
- Verify no missing data in critical periods

---

## Expected Performance Ranges

```
Model Training:
  IC:                  0.030 - 0.080
  R2:                  0.05 - 0.15
  Direction Accuracy:  52% - 58%

Backtesting:
  Sharpe Ratio:        1.0 - 2.0
  Sortino Ratio:       1.3 - 2.5
  Max Drawdown:        -10% to -20%
  Win Rate:            52% - 60%
  Annual Return:       10% - 25%
```

---

## Workflow

1. **Train models**: `python scripts/train_models.py --regime-adaptive`
2. **Run backtest**: `python scripts/run_backtest.py --regime-adaptive --start 2023-01-01 --end 2024-12-31`
3. **Analyze results**: Check `reports/` directory
4. **Iterate**: Adjust config, retrain, retest
5. **Deploy**: Use best model for live trading

