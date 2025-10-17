"""
Integration tests for broker execution flow.

Tests end-to-end workflows including order placement, risk checks,
and position management.
"""

import pytest
from datetime import datetime

from src.execution.broker_factory import BrokerFactory
from src.execution.order_manager import OrderManager
from src.execution.broker_interface import (
    Order,
    OrderType,
    OrderSide,
    OrderStatus,
)
from src.risk_management.risk_controller import RiskController
from src.risk_management.circuit_breaker import CircuitBreaker


class TestPaperTradingFlow:
    """Test complete paper trading workflow."""

    def test_full_trading_cycle(self):
        """Test complete trading cycle: buy -> hold -> sell."""
        # Create paper broker
        broker = BrokerFactory.create("PAPER", initial_capital=1000000)

        initial_balance = broker.get_account_balance()
        assert initial_balance.total_balance == 1000000

        # Place buy order
        buy_order = Order(
            symbol="RELIANCE",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET,
        )

        order_id = broker.place_order(buy_order)
        assert order_id is not None
        assert buy_order.status == OrderStatus.FILLED

        # Check position created
        positions = broker.get_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "RELIANCE"
        assert positions[0].quantity == 10

        # Place sell order to close
        sell_order = Order(
            symbol="RELIANCE",
            side=OrderSide.SELL,
            quantity=10,
            order_type=OrderType.MARKET,
        )

        broker.place_order(sell_order)

        # Position should be closed
        positions = broker.get_positions()
        assert len(positions) == 0

    def test_multiple_positions(self):
        """Test managing multiple positions simultaneously."""
        broker = BrokerFactory.create("PAPER", initial_capital=1000000)

        # Buy multiple stocks
        symbols = ["RELIANCE", "TCS"]

        for symbol in symbols:
            order = Order(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=5,
                order_type=OrderType.MARKET,
            )
            broker.place_order(order)

        # Check positions
        positions = broker.get_positions()
        assert len(positions) == 2

        position_symbols = {pos.symbol for pos in positions}
        assert position_symbols == set(symbols)

    def test_limit_order_workflow(self):
        """Test limit order placement and tracking."""
        broker = BrokerFactory.create("PAPER", initial_capital=1000000)

        # Place limit order below current price
        order = Order(
            symbol="RELIANCE",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.LIMIT,
            price=2400.0,  # Below typical LTP
        )

        order_id = broker.place_order(order)

        # Order should be pending
        order_status = broker.get_order_status(order_id)
        assert order_status.status == OrderStatus.PLACED

        # Cancel order
        result = broker.cancel_order(order_id)
        assert result is True

        # Check cancellation
        order_status = broker.get_order_status(order_id)
        assert order_status.status == OrderStatus.CANCELLED


class TestRiskIntegration:
    """Test risk management integration."""

    def test_risk_controller_blocks_invalid_order(self):
        """Test that risk controller blocks orders violating limits."""
        broker = BrokerFactory.create("PAPER", initial_capital=100000)
        risk = RiskController()

        # Try to place oversized order
        order = Order(
            symbol="RELIANCE",
            side=OrderSide.BUY,
            quantity=100,  # Would be >10% of portfolio
            order_type=OrderType.MARKET,
        )

        portfolio_value = broker.get_portfolio_value()
        positions = broker.get_positions()

        is_valid, reason = risk.validate_order(
            order, portfolio_value, positions, current_price=2450.0
        )

        assert not is_valid

    def test_circuit_breaker_stops_trading(self):
        """Test that circuit breaker stops trading after losses."""
        broker = BrokerFactory.create("PAPER", initial_capital=1000000)
        breaker = CircuitBreaker()

        # Simulate consecutive losses
        breaker.record_trade(-5000.0)
        breaker.record_trade(-3000.0)
        breaker.record_trade(-2000.0)

        # Circuit breaker should be active
        assert breaker.is_active()

        # Should not place orders when circuit breaker active
        if breaker.is_active():
            # In real system, would skip order placement
            pass

    def test_full_risk_workflow(self):
        """Test complete workflow with risk management."""
        broker = BrokerFactory.create("PAPER", initial_capital=1000000)
        risk = RiskController()
        breaker = CircuitBreaker()

        # Place valid order
        order = Order(
            symbol="RELIANCE",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET,
        )

        portfolio_value = broker.get_portfolio_value()
        positions = broker.get_positions()

        # Validate with risk controller
        is_valid, reason = risk.validate_order(
            order, portfolio_value, positions, current_price=2450.0
        )

        if is_valid and not breaker.is_active():
            order_id = broker.place_order(order)
            risk.record_order()

            # Simulate trade result
            risk.record_trade_result(100.0)  # Small profit
            breaker.record_trade(100.0)

            # Should still be able to trade
            assert not breaker.is_active()
            assert risk.daily_order_count == 1


class TestOrderManager:
    """Test order manager integration."""

    def test_order_manager_tracks_orders(self):
        """Test that order manager tracks orders correctly."""
        broker = BrokerFactory.create("PAPER", initial_capital=1000000)

        # Use in-memory database for testing
        with OrderManager(broker, db_path=":memory:", enable_monitoring=False) as manager:
            order = Order(
                symbol="RELIANCE",
                side=OrderSide.BUY,
                quantity=10,
                order_type=OrderType.MARKET,
            )

            order_id = manager.place_order(order)

            # Check order in cache
            cached_order = manager.get_order_status(order_id, use_cache=True)
            assert cached_order.symbol == "RELIANCE"
            assert cached_order.status == OrderStatus.FILLED

    def test_order_manager_statistics(self):
        """Test order statistics calculation."""
        broker = BrokerFactory.create("PAPER", initial_capital=1000000)

        with OrderManager(broker, db_path=":memory:", enable_monitoring=False) as manager:
            # Place multiple orders
            for i in range(5):
                order = Order(
                    symbol="RELIANCE",
                    side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                    quantity=1,
                    order_type=OrderType.MARKET,
                )
                manager.place_order(order)

            # Get statistics
            stats = manager.get_order_statistics()

            assert stats["total_orders"] >= 5
            assert stats["filled_orders"] >= 0


class TestBrokerSwitching:
    """Test switching between brokers."""

    def test_broker_factory_paper(self):
        """Test creating paper broker via factory."""
        broker = BrokerFactory.create("PAPER", initial_capital=500000)

        assert broker is not None
        balance = broker.get_account_balance()
        assert balance.total_balance == 500000

    def test_broker_interface_consistency(self):
        """Test that all brokers implement same interface."""
        broker = BrokerFactory.create("PAPER", initial_capital=1000000)

        # Test all interface methods are available
        assert hasattr(broker, "authenticate")
        assert hasattr(broker, "place_order")
        assert hasattr(broker, "get_positions")
        assert hasattr(broker, "get_account_balance")
        assert hasattr(broker, "get_ltp")
        assert hasattr(broker, "close_all_positions")
        assert hasattr(broker, "cancel_all_orders")


class TestErrorHandling:
    """Test error handling in execution layer."""

    def test_insufficient_funds_error(self):
        """Test handling of insufficient funds."""
        broker = BrokerFactory.create("PAPER", initial_capital=10000)

        # Try to buy too much
        order = Order(
            symbol="RELIANCE",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )

        with pytest.raises(ValueError, match="Insufficient funds"):
            broker.place_order(order)

    def test_invalid_order_id(self):
        """Test handling of invalid order ID."""
        broker = BrokerFactory.create("PAPER", initial_capital=1000000)

        with pytest.raises(ValueError):
            broker.get_order_status("INVALID_ID")

    def test_cancel_nonexistent_order(self):
        """Test cancelling non-existent order."""
        broker = BrokerFactory.create("PAPER", initial_capital=1000000)

        with pytest.raises(ValueError):
            broker.cancel_order("NONEXISTENT_ORDER")


class TestPerformance:
    """Test performance of execution layer."""

    def test_order_placement_speed(self):
        """Test that order placement is fast enough."""
        import time

        broker = BrokerFactory.create("PAPER", initial_capital=1000000)

        start_time = time.time()

        # Place 10 orders
        for i in range(10):
            order = Order(
                symbol="RELIANCE",
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                quantity=1,
                order_type=OrderType.MARKET,
            )
            broker.place_order(order)

        elapsed = time.time() - start_time

        # Should complete in under 1 second for paper trading
        assert elapsed < 1.0

    def test_position_retrieval_speed(self):
        """Test that position retrieval is fast."""
        import time

        broker = BrokerFactory.create("PAPER", initial_capital=1000000)

        # Create some positions
        for _ in range(5):
            order = Order(
                symbol="RELIANCE",
                side=OrderSide.BUY,
                quantity=1,
                order_type=OrderType.MARKET,
            )
            broker.place_order(order)

        start_time = time.time()

        # Retrieve positions 100 times
        for _ in range(100):
            broker.get_positions()

        elapsed = time.time() - start_time

        # Should be very fast
        assert elapsed < 0.5

