"""
Execution layer for the NSE Adaptive Regime Trading System.

This module provides broker abstraction and order execution capabilities.
It supports multiple brokers (DhanHQ, Kite Connect) through a unified interface
and includes paper trading for risk-free strategy testing.

Components:
    - BrokerInterface: Abstract base class defining broker contract
    - DhanBroker: DhanHQ implementation (sandbox + live)
    - KiteBroker: Kite Connect implementation (future)
    - PaperTradingBroker: Custom paper trading with live data
    - BrokerFactory: Factory for easy broker switching
    - OrderManager: Order lifecycle management

Example:
    >>> from src.execution import BrokerFactory
    >>> broker = BrokerFactory.create("PAPER", initial_capital=1000000)
    >>> order = Order(symbol="RELIANCE", side=OrderSide.BUY, quantity=10,
    ...               order_type=OrderType.MARKET)
    >>> order_id = broker.place_order(order)
"""

from src.execution.broker_interface import (
    BrokerInterface,
    Order,
    Position,
    AccountBalance,
    OrderType,
    OrderSide,
    OrderStatus,
)
from src.execution.broker_factory import BrokerFactory

__all__ = [
    "BrokerInterface",
    "Order",
    "Position",
    "AccountBalance",
    "OrderType",
    "OrderSide",
    "OrderStatus",
    "BrokerFactory",
]
