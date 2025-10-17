# Execution Layer

Production-grade broker abstraction for the NSE Adaptive Regime Trading System.

## Quick Start

```python
from src.execution import BrokerFactory, Order, OrderSide, OrderType

# Create paper broker
broker = BrokerFactory.create("PAPER", initial_capital=1000000)

# Place order
order = Order(
    symbol="RELIANCE",
    side=OrderSide.BUY,
    quantity=10,
    order_type=OrderType.MARKET
)

order_id = broker.place_order(order)
print(f"Order placed: {order_id}")
```

## Components

- **broker_interface.py** - Abstract base class defining broker contract
- **dhan_broker.py** - DhanHQ implementation (sandbox + live)
- **kite_broker.py** - Kite Connect skeleton (future)
- **paper_broker.py** - Paper trading with live data simulation
- **broker_factory.py** - Factory for easy broker switching
- **order_manager.py** - Order lifecycle management and persistence

## Switching Brokers

Change one line to switch between brokers:

```python
# Paper trading
broker = BrokerFactory.create("PAPER", initial_capital=1000000)

# DhanHQ sandbox
broker = BrokerFactory.create("DHAN", mode="SANDBOX")

# DhanHQ live
broker = BrokerFactory.create("DHAN", mode="LIVE")
```

## Risk Management

Pre-trade risk checks included:

```python
from src.risk_management import RiskController

risk = RiskController()

is_valid, reason = risk.validate_order(
    order, portfolio_value, positions, current_price
)

if is_valid:
    broker.place_order(order)
else:
    print(f"Order rejected: {reason}")
```

## Full Documentation

See [`docs/EXECUTION_LAYER.md`](../../docs/EXECUTION_LAYER.md) for comprehensive documentation.

## Scripts

- `scripts/setup_broker.py` - Interactive broker setup
- `scripts/run_paper_trading.py` - Run paper trading
- `scripts/run_live_trading.py` - Run live trading

## Configuration

- `config/broker_config.yaml` - Broker settings
- `config/risk_config.yaml` - Risk limits

## Tests

```bash
# Unit tests
pytest tests/unit/test_execution.py -v

# Integration tests
pytest tests/integration/test_broker_flow.py -v
```

