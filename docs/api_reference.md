# API Reference

## Overview

This document provides API reference for the main modules and classes in the NSE Adaptive Regime Trading System.

## Data Pipeline

### DataFetcher

Fetches market data from various sources.

```python
from src.data_pipeline.data_fetcher import DataFetcher

fetcher = DataFetcher(config)
data = fetcher.fetch_ohlcv(
    symbols=["RELIANCE", "TCS"],
    start_date="2024-01-01",
    end_date="2024-12-31",
    timeframe="1d"
)
```

### DataCleaner

Cleans and validates market data.

```python
from src.data_pipeline.data_cleaner import DataCleaner

cleaner = DataCleaner()
cleaned_data = cleaner.clean(raw_data)
```

## Feature Engineering

### Alpha158Features

Generates Qlib Alpha-158 factors.

```python
from src.feature_engineering.alpha158 import Alpha158Features

features = Alpha158Features()
alpha_features = features.generate(ohlcv_data)
```

## Regime Detection

### RegimeDetector

Detects market regimes using multiple methods.

```python
from src.regime_detection.regime_detector import RegimeDetector

detector = RegimeDetector(method="wasserstein")
regime = detector.detect(market_data)
# Returns: "bull", "bear", "sideways", or "high_volatility"
```

## Qlib Models

### QlibModelTrainer

Trains Qlib quantitative models.

```python
from src.qlib_models.trainer import QlibModelTrainer

trainer = QlibModelTrainer(config)
model = trainer.train(
    train_data,
    model_type="lightgbm"
)
```

### QlibPredictor

Makes predictions using trained Qlib models.

```python
from src.qlib_models.predictor import QlibPredictor

predictor = QlibPredictor(model_path)
predictions = predictor.predict(features)
```

## RL Strategy

### TradingEnvironment

RL environment for training trading agents.

```python
from src.rl_strategy.environment import TradingEnvironment

env = TradingEnvironment(
    data=historical_data,
    initial_balance=1000000
)
```

### RLAgent

Reinforcement learning agent for trading.

```python
from src.rl_strategy.agent import RLAgent

agent = RLAgent(
    algorithm="ppo",
    env=env
)
agent.train(total_timesteps=1000000)
action = agent.predict(observation)
```

## Risk Management

### RiskManager

Manages portfolio risk and constraints.

```python
from src.risk_management.risk_manager import RiskManager

risk_mgr = RiskManager(config)
position_size = risk_mgr.calculate_position_size(
    symbol="RELIANCE",
    current_price=2500,
    stop_loss=2450
)
```

### VaRCalculator

Calculates Value-at-Risk metrics.

```python
from src.risk_management.var_calculator import VaRCalculator

var_calc = VaRCalculator()
portfolio_var = var_calc.calculate(
    positions,
    confidence=0.95
)
```

## Execution

### OrderManager

Manages order lifecycle.

```python
from src.execution.order_manager import OrderManager

order_mgr = OrderManager(kite_client)
order = order_mgr.place_order(
    symbol="RELIANCE",
    side="BUY",
    quantity=10,
    order_type="LIMIT",
    price=2500
)
```

### ExecutionEngine

High-level execution interface.

```python
from src.execution.execution_engine import ExecutionEngine

engine = ExecutionEngine(config)
engine.execute_strategy_signals(signals)
```

## Utilities

### Database

Database connection and operations.

```python
from src.utils.database import DatabaseManager

db = DatabaseManager()
df = db.query_df("SELECT * FROM ohlcv WHERE symbol = ?", ("RELIANCE",))
```

### Market Calendar

NSE market calendar utilities.

```python
from src.utils.market_calendar import get_market_calendar

calendar = get_market_calendar()
is_trading_day = calendar.is_trading_day(date)
is_market_open = calendar.is_market_open()
```

### Logging

Structured logging utilities.

```python
from src.utils.logging_config import get_logger, log_trade

logger = get_logger(__name__)
logger.info("Starting strategy")

log_trade(
    symbol="RELIANCE",
    action="BUY",
    quantity=10,
    price=2500,
    strategy="adaptive_regime"
)
```

## Configuration

### Loading Configuration

```python
from src.utils.helpers import load_config

config = load_config("config/qlib_config.yaml")
```

### Accessing Environment Variables

```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("ZERODHA_API_KEY")
```

## Error Handling

All major functions raise appropriate exceptions:

```python
from src.exceptions import (
    DataFetchError,
    InsufficientDataError,
    ModelTrainingError,
    OrderExecutionError,
    RiskLimitExceededError
)

try:
    data = fetcher.fetch_data(symbol)
except DataFetchError as e:
    logger.error(f"Failed to fetch data: {e}")
```

## Type Hints

All functions use Python type hints:

```python
def calculate_returns(
    prices: pd.Series,
    method: str = "simple"
) -> pd.Series:
    """
    Calculate returns from price series.
    
    Args:
        prices: Price series
        method: Return calculation method
        
    Returns:
        Returns series
    """
    pass
```

## Testing

### Unit Tests

```python
import pytest
from src.utils.helpers import format_currency

def test_format_currency():
    assert format_currency(1000000) == "₹1,000,000.00"
```

### Integration Tests

```python
@pytest.mark.integration
def test_end_to_end_backtest():
    # Test complete backtest workflow
    pass
```

## More Information

For detailed implementation examples, see:
- `tests/` directory for usage examples
- `notebooks/` directory for interactive examples
- `scripts/` directory for command-line tools

