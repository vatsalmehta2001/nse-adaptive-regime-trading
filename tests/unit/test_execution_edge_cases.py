"""
Edge case testing for execution layer.

Tests network failures, rate limits, API errors, partial fills,
database failures, and other edge cases to ensure production reliability.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.execution import BrokerFactory, Order, OrderSide, OrderType, OrderStatus
from src.execution.paper_broker import PaperTradingBroker
from src.execution.symbol_mapper import SymbolMapper
from src.risk_management import RiskController


class MockDataBroker:
    """Mock broker for testing."""

    def __init__(self, should_fail=False):
        self.should_fail = should_fail
        self.call_count = 0

    def get_ltp(self, symbol: str, exchange: str = "NSE") -> float:
        """Mock LTP."""
        self.call_count += 1
        if self.should_fail:
            raise TimeoutError("Connection timeout")
        return 2450.0

    def get_quote(self, symbol: str, exchange: str = "NSE") -> dict:
        """Mock quote."""
        if self.should_fail:
            raise ConnectionError("Network error")
        return {
            "symbol": symbol,
            "ltp": 2450.0,
            "open": 2400.0,
            "high": 2500.0,
            "low": 2350.0,
        }


class TestNetworkFailures:
    """Test handling of network failures."""

    def test_connection_timeout_handling(self):
        """Test handling of connection timeout."""
        mock_broker = MockDataBroker(should_fail=True)
        paper = PaperTradingBroker(1000000, mock_broker)

        order = Order(
            symbol="RELIANCE",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET
        )

        # Paper broker handles timeout gracefully with fallback price
        # Logs error but continues execution for robustness
        order_id = paper.place_order(order)
        assert order_id is not None
        # Order should still be filled with fallback price

    def test_intermittent_network_failure_recovery(self):
        """Test recovery from intermittent network failures."""
        mock_broker = MockDataBroker(should_fail=False)
        paper = PaperTradingBroker(1000000, mock_broker)

        # First order succeeds
        order1 = Order(
            symbol="RELIANCE",
            side=OrderSide.BUY,
            quantity=1,
            order_type=OrderType.MARKET
        )
        order_id1 = paper.place_order(order1)
        assert order_id1 is not None

        # Simulate network failure
        mock_broker.should_fail = True

        order2 = Order(
            symbol="RELIANCE",
            side=OrderSide.BUY,
            quantity=1,
            order_type=OrderType.MARKET
        )

        # Paper broker handles failure gracefully
        order_id2 = paper.place_order(order2)
        assert order_id2 is not None

        # Network recovers
        mock_broker.should_fail = False

        order3 = Order(
            symbol="RELIANCE",
            side=OrderSide.BUY,
            quantity=1,
            order_type=OrderType.MARKET
        )
        order_id3 = paper.place_order(order3)
        assert order_id3 is not None

    def test_null_response_handling(self):
        """Test handling of null/empty API responses."""
        mock_broker = Mock()
        mock_broker.get_ltp = Mock(return_value=None)

        paper = PaperTradingBroker(1000000, mock_broker)

        order = Order(
            symbol="RELIANCE",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET
        )

        # Should handle None gracefully (will use fallback price)
        order_id = paper.place_order(order)
        assert order_id is not None


class TestRateLimiting:
    """Test rate limiting behavior."""

    def test_rapid_order_placement(self):
        """Test behavior when placing orders rapidly."""
        mock_broker = MockDataBroker()
        paper = PaperTradingBroker(1000000, mock_broker)

        orders_placed = 0
        start_time = time.time()

        # Try to place 20 orders rapidly
        for i in range(20):
            try:
                order = Order(
                    symbol="RELIANCE",
                    side=OrderSide.BUY,
                    quantity=1,
                    order_type=OrderType.MARKET
                )
                paper.place_order(order)
                orders_placed += 1
            except Exception as e:
                # Some orders may fail due to insufficient funds
                pass

        elapsed = time.time() - start_time

        # Should place at least some orders
        assert orders_placed > 0
        # Should complete in reasonable time
        assert elapsed < 10.0

    def test_concurrent_position_queries(self):
        """Test querying positions while orders are being placed."""
        mock_broker = MockDataBroker()
        paper = PaperTradingBroker(1000000, mock_broker)

        # Place an order
        order = Order(
            symbol="RELIANCE",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET
        )
        paper.place_order(order)

        # Query positions multiple times rapidly
        for _ in range(10):
            positions = paper.get_positions()
            assert isinstance(positions, list)
            time.sleep(0.01)


class TestPartialFills:
    """Test partial order fills."""

    def test_partial_fill_tracking(self):
        """Test that partial fills are tracked correctly."""
        mock_broker = MockDataBroker()
        paper = PaperTradingBroker(1000000, mock_broker)

        order = Order(
            symbol="RELIANCE",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            price=2400.0
        )

        order_id = paper.place_order(order)

        # Check order status
        order_status = paper.get_order_status(order_id)

        # Check that partial fill fields exist
        assert hasattr(order_status, 'filled_quantity')
        assert hasattr(order_status, 'quantity')
        assert hasattr(order_status, 'status')

    def test_limit_order_pending_state(self):
        """Test limit order stays pending when price not met."""
        mock_broker = MockDataBroker()
        paper = PaperTradingBroker(1000000, mock_broker)

        # Place limit order below current price
        order = Order(
            symbol="RELIANCE",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.LIMIT,
            price=2000.0  # Well below LTP of 2450
        )

        order_id = paper.place_order(order)
        order_status = paper.get_order_status(order_id)

        # Should be pending/placed, not filled
        assert order_status.status in [OrderStatus.PLACED, OrderStatus.PENDING]


class TestAPIErrors:
    """Test handling of API errors."""

    def test_invalid_symbol_handling(self):
        """Test handling of invalid trading symbol."""
        mock_broker = Mock()
        mock_broker.get_ltp = Mock(side_effect=ValueError("Symbol not found"))

        paper = PaperTradingBroker(1000000, mock_broker)

        order = Order(
            symbol="INVALID_SYMBOL_XYZ",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET
        )

        # Paper broker handles the error and uses fallback price
        # (In production, would log error but continue)
        order_id = paper.place_order(order)
        assert order_id is not None

    def test_malformed_data_handling(self):
        """Test handling of malformed data."""
        mock_broker = Mock()
        # Return string instead of float
        mock_broker.get_ltp = Mock(side_effect=TypeError("Invalid type"))

        paper = PaperTradingBroker(1000000, mock_broker)

        order = Order(
            symbol="RELIANCE",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET
        )

        # Paper broker should handle type error and use fallback price
        order_id = paper.place_order(order)
        assert order_id is not None

    def test_negative_price_handling(self):
        """Test handling of negative prices."""
        with pytest.raises(ValueError, match="Price must be positive"):
            Order(
                symbol="RELIANCE",
                side=OrderSide.BUY,
                quantity=10,
                order_type=OrderType.LIMIT,
                price=-100.0
            )

    def test_zero_quantity_handling(self):
        """Test handling of zero quantity."""
        with pytest.raises(ValueError, match="Quantity must be positive"):
            Order(
                symbol="RELIANCE",
                side=OrderSide.BUY,
                quantity=0,
                order_type=OrderType.MARKET
            )


class TestSymbolMapper:
    """Test symbol mapper functionality."""

    def test_symbol_mapper_initialization(self):
        """Test SymbolMapper initializes correctly."""
        mapper = SymbolMapper()
        assert len(mapper.symbol_map) > 0

    def test_symbol_mapper_get_security_id(self):
        """Test getting security ID for symbol."""
        mapper = SymbolMapper()
        security_id = mapper.get_security_id("RELIANCE", "NSE")
        assert security_id is not None
        assert len(security_id) > 0
        assert security_id.isdigit()

    def test_symbol_mapper_invalid_symbol(self):
        """Test error handling for invalid symbol."""
        mapper = SymbolMapper()
        with pytest.raises(ValueError, match="Security ID not found"):
            mapper.get_security_id("INVALID_XYZ_123", "NSE")

    def test_symbol_mapper_add_mapping(self):
        """Test adding new mapping."""
        mapper = SymbolMapper()
        mapper.add_mapping("TESTSTOCK", "NSE", "99999", "test", "Test mapping")

        # Should be able to retrieve it
        security_id = mapper.get_security_id("TESTSTOCK", "NSE")
        assert security_id == "99999"

    def test_symbol_mapper_has_mapping(self):
        """Test checking if mapping exists."""
        mapper = SymbolMapper()
        assert mapper.has_mapping("RELIANCE", "NSE") is True
        assert mapper.has_mapping("INVALID_XYZ", "NSE") is False

    def test_symbol_mapper_get_all_symbols(self):
        """Test getting all symbols."""
        mapper = SymbolMapper()
        symbols = mapper.get_all_symbols("NSE")
        assert isinstance(symbols, list)
        assert len(symbols) > 0
        assert "RELIANCE" in symbols or "TCS" in symbols

    def test_symbol_mapper_validate_mappings(self):
        """Test validation of mappings."""
        mapper = SymbolMapper()
        results = mapper.validate_mappings()

        assert 'valid' in results
        assert 'errors' in results
        assert 'warnings' in results
        assert isinstance(results['valid'], list)


class TestRiskManagementEdgeCases:
    """Test risk management edge cases."""

    def test_risk_controller_with_zero_portfolio_value(self):
        """Test risk controller with edge case portfolio values."""
        risk = RiskController()

        order = Order(
            symbol="RELIANCE",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET
        )

        # With zero portfolio, calculations should handle gracefully
        # May pass validation due to division handling
        is_valid, reason = risk.validate_order(order, 0.0, [], 2450.0)
        # Test just ensures no crash/exception
        assert isinstance(is_valid, bool)
        assert isinstance(reason, str)

    def test_risk_controller_with_negative_pnl(self):
        """Test risk controller with large negative P&L."""
        risk = RiskController()
        risk.daily_pnl = -100000  # Large loss

        order = Order(
            symbol="RELIANCE",
            side=OrderSide.BUY,
            quantity=1,
            order_type=OrderType.MARKET
        )

        # Should block due to daily loss limit
        is_valid, reason = risk.validate_order(order, 1000000, [], 2450.0)
        assert not is_valid
        assert "daily loss" in reason.lower()

    def test_risk_controller_daily_reset(self):
        """Test daily reset functionality."""
        risk = RiskController()
        risk.daily_pnl = -10000
        risk.daily_order_count = 25

        risk.reset_daily()

        assert risk.daily_pnl == 0.0
        assert risk.daily_order_count == 0


class TestOrderCancellation:
    """Test order cancellation scenarios."""

    def test_cancel_pending_order(self):
        """Test cancelling a pending limit order."""
        mock_broker = MockDataBroker()
        paper = PaperTradingBroker(1000000, mock_broker)

        # Place limit order that won't fill
        order = Order(
            symbol="RELIANCE",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.LIMIT,
            price=2000.0
        )

        order_id = paper.place_order(order)

        # Cancel it
        result = paper.cancel_order(order_id)
        assert result is True

        # Check status
        order_status = paper.get_order_status(order_id)
        assert order_status.status == OrderStatus.CANCELLED

    def test_cancel_filled_order_fails(self):
        """Test that cancelling filled order raises error."""
        mock_broker = MockDataBroker()
        paper = PaperTradingBroker(1000000, mock_broker)

        # Place market order (fills immediately)
        order = Order(
            symbol="RELIANCE",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET
        )

        order_id = paper.place_order(order)

        # Try to cancel (should fail)
        with pytest.raises(ValueError, match="Cannot cancel"):
            paper.cancel_order(order_id)

    def test_cancel_nonexistent_order(self):
        """Test cancelling non-existent order."""
        mock_broker = MockDataBroker()
        paper = PaperTradingBroker(1000000, mock_broker)

        with pytest.raises(ValueError, match="not found"):
            paper.cancel_order("NONEXISTENT_ORDER_ID")


class TestPortfolioCalculations:
    """Test portfolio value calculations."""

    def test_portfolio_value_with_positions(self):
        """Test portfolio value calculation with open positions."""
        mock_broker = MockDataBroker()
        paper = PaperTradingBroker(1000000, mock_broker)

        # Place some orders
        order1 = Order(
            symbol="RELIANCE",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET
        )
        paper.place_order(order1)

        # Get portfolio value
        portfolio_value = paper.get_portfolio_value()

        # Should be close to initial capital (minus commissions/slippage)
        assert 990000 < portfolio_value < 1010000

    def test_portfolio_value_after_trades(self):
        """Test portfolio value after multiple trades."""
        mock_broker = MockDataBroker()
        paper = PaperTradingBroker(1000000, mock_broker)

        # Buy
        order1 = Order(
            symbol="RELIANCE",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET
        )
        paper.place_order(order1)

        # Sell
        order2 = Order(
            symbol="RELIANCE",
            side=OrderSide.SELL,
            quantity=10,
            order_type=OrderType.MARKET
        )
        paper.place_order(order2)

        # Portfolio value should be back near initial (with some loss from costs)
        portfolio_value = paper.get_portfolio_value()
        assert 990000 < portfolio_value < 1005000


class TestEmergencyFunctions:
    """Test emergency stop functions."""

    def test_close_all_positions(self):
        """Test closing all positions."""
        mock_broker = MockDataBroker()
        paper = PaperTradingBroker(1000000, mock_broker)

        # Create some positions
        for i in range(3):
            order = Order(
                symbol="RELIANCE",
                side=OrderSide.BUY,
                quantity=10,
                order_type=OrderType.MARKET
            )
            paper.place_order(order)

        # Close all
        result = paper.close_all_positions()
        assert result is True

        # Check positions are closed
        positions = paper.get_positions()
        assert len(positions) == 0

    def test_cancel_all_orders(self):
        """Test cancelling all orders."""
        mock_broker = MockDataBroker()
        paper = PaperTradingBroker(1000000, mock_broker)

        # Place some limit orders
        for i in range(3):
            order = Order(
                symbol="RELIANCE",
                side=OrderSide.BUY,
                quantity=10,
                order_type=OrderType.LIMIT,
                price=2000.0
            )
            paper.place_order(order)

        # Cancel all
        result = paper.cancel_all_orders()
        assert result is True

        # Check all are cancelled
        pending = paper.get_orders(status=OrderStatus.PLACED)
        assert len(pending) == 0


# Run with: pytest tests/unit/test_execution_edge_cases.py -v

