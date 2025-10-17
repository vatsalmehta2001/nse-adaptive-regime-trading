"""
Unit tests for execution layer.

Tests broker interface, paper trading, risk management, and order lifecycle.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch

from src.execution.broker_interface import (
    Order,
    Position,
    AccountBalance,
    OrderType,
    OrderSide,
    OrderStatus,
)
from src.execution.paper_broker import PaperTradingBroker
from src.execution.broker_factory import BrokerFactory
from src.risk_management.risk_controller import RiskController, RiskLimits
from src.risk_management.circuit_breaker import CircuitBreaker


class TestOrderDataclass:
    """Test Order dataclass and validation."""

    def test_order_creation(self):
        """Test basic order creation."""
        order = Order(
            symbol="RELIANCE",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.LIMIT,
            price=2450.0,
        )

        assert order.symbol == "RELIANCE"
        assert order.side == OrderSide.BUY
        assert order.quantity == 10
        assert order.order_type == OrderType.LIMIT
        assert order.price == 2450.0
        assert order.product == "MIS"
        assert order.exchange == "NSE"

    def test_market_order_creation(self):
        """Test market order without price."""
        order = Order(
            symbol="TCS",
            side=OrderSide.SELL,
            quantity=5,
            order_type=OrderType.MARKET,
        )

        assert order.price is None
        assert order.order_type == OrderType.MARKET

    def test_order_validation_negative_quantity(self):
        """Test that negative quantity raises error."""
        with pytest.raises(ValueError, match="Quantity must be positive"):
            Order(
                symbol="RELIANCE",
                side=OrderSide.BUY,
                quantity=-10,
                order_type=OrderType.MARKET,
            )

    def test_limit_order_requires_price(self):
        """Test that limit order without price raises error."""
        with pytest.raises(ValueError, match="LIMIT order requires price"):
            Order(
                symbol="RELIANCE",
                side=OrderSide.BUY,
                quantity=10,
                order_type=OrderType.LIMIT,
            )

    def test_stop_loss_requires_trigger_price(self):
        """Test that stop loss order requires trigger price."""
        with pytest.raises(ValueError, match="Stop loss order requires trigger_price"):
            Order(
                symbol="RELIANCE",
                side=OrderSide.SELL,
                quantity=10,
                order_type=OrderType.STOP_LOSS_MARKET,
            )


class TestPositionDataclass:
    """Test Position dataclass."""

    def test_position_creation(self):
        """Test position creation."""
        pos = Position(
            symbol="RELIANCE",
            quantity=10,
            average_price=2450.0,
            last_price=2460.0,
            pnl=100.0,
            day_pnl=100.0,
        )

        assert pos.symbol == "RELIANCE"
        assert pos.quantity == 10

    def test_position_market_value(self):
        """Test market value calculation."""
        pos = Position(
            symbol="RELIANCE",
            quantity=10,
            average_price=2450.0,
            last_price=2500.0,
            pnl=500.0,
            day_pnl=500.0,
        )

        assert pos.market_value == 25000.0  # 10 * 2500

    def test_position_unrealized_pnl(self):
        """Test unrealized P&L calculation."""
        pos = Position(
            symbol="RELIANCE",
            quantity=10,
            average_price=2450.0,
            last_price=2500.0,
            pnl=0.0,
            day_pnl=0.0,
        )

        assert pos.unrealized_pnl == 500.0  # 10 * (2500 - 2450)


class MockBroker:
    """Mock broker for testing."""

    def __init__(self):
        self.ltp_prices = {"RELIANCE": 2450.0, "TCS": 3500.0}

    def get_ltp(self, symbol: str, exchange: str = "NSE") -> float:
        """Return mock LTP."""
        return self.ltp_prices.get(symbol, 100.0)

    def get_quote(self, symbol: str, exchange: str = "NSE") -> dict:
        """Return mock quote."""
        ltp = self.get_ltp(symbol, exchange)
        return {
            "symbol": symbol,
            "ltp": ltp,
            "open": ltp,
            "high": ltp * 1.02,
            "low": ltp * 0.98,
            "close": ltp,
            "volume": 1000000,
        }


class TestPaperTradingBroker:
    """Test paper trading broker."""

    def test_initialization(self):
        """Test paper broker initialization."""
        mock_broker = MockBroker()
        paper = PaperTradingBroker(1000000, mock_broker)

        assert paper.initial_capital == 1000000
        assert paper.cash == 1000000
        assert len(paper.positions) == 0
        assert len(paper.orders) == 0

    def test_market_order_buy(self):
        """Test market order buy execution."""
        mock_broker = MockBroker()
        paper = PaperTradingBroker(1000000, mock_broker)

        order = Order(
            symbol="RELIANCE",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET,
        )

        order_id = paper.place_order(order)

        assert order_id is not None
        assert order_id.startswith("PAPER_")
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 10

        # Check position created
        positions = paper.get_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "RELIANCE"
        assert positions[0].quantity == 10

        # Check cash reduced
        assert paper.cash < 1000000

    def test_market_order_sell(self):
        """Test market order sell execution."""
        mock_broker = MockBroker()
        paper = PaperTradingBroker(1000000, mock_broker)

        # First buy
        buy_order = Order(
            symbol="RELIANCE",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET,
        )
        paper.place_order(buy_order)

        # Then sell
        sell_order = Order(
            symbol="RELIANCE",
            side=OrderSide.SELL,
            quantity=10,
            order_type=OrderType.MARKET,
        )
        paper.place_order(sell_order)

        # Position should be closed
        positions = paper.get_positions()
        assert len(positions) == 0

    def test_insufficient_funds(self):
        """Test insufficient funds error."""
        mock_broker = MockBroker()
        paper = PaperTradingBroker(10000, mock_broker)  # Small capital

        # Try to buy expensive stock
        order = Order(
            symbol="RELIANCE",
            side=OrderSide.BUY,
            quantity=100,  # Too many
            order_type=OrderType.MARKET,
        )

        with pytest.raises(ValueError, match="Insufficient funds"):
            paper.place_order(order)

    def test_limit_order_immediate_fill(self):
        """Test limit order that can fill immediately."""
        mock_broker = MockBroker()
        paper = PaperTradingBroker(1000000, mock_broker)

        # Buy limit order above current price (should fill immediately)
        order = Order(
            symbol="RELIANCE",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.LIMIT,
            price=2500.0,  # Above LTP of 2450
        )

        order_id = paper.place_order(order)

        # Should fill immediately
        assert order.status == OrderStatus.FILLED

    def test_limit_order_pending(self):
        """Test limit order that stays pending."""
        mock_broker = MockBroker()
        paper = PaperTradingBroker(1000000, mock_broker)

        # Buy limit order below current price (should stay pending)
        order = Order(
            symbol="RELIANCE",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.LIMIT,
            price=2400.0,  # Below LTP of 2450
        )

        paper.place_order(order)

        # Should be pending
        assert order.status == OrderStatus.PLACED

    def test_order_cancellation(self):
        """Test order cancellation."""
        mock_broker = MockBroker()
        paper = PaperTradingBroker(1000000, mock_broker)

        # Place limit order
        order = Order(
            symbol="RELIANCE",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.LIMIT,
            price=2400.0,
        )

        order_id = paper.place_order(order)

        # Cancel order
        result = paper.cancel_order(order_id)

        assert result is True
        assert order.status == OrderStatus.CANCELLED

    def test_get_portfolio_value(self):
        """Test portfolio value calculation."""
        mock_broker = MockBroker()
        paper = PaperTradingBroker(1000000, mock_broker)

        # Buy some stock
        order = Order(
            symbol="RELIANCE",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET,
        )
        paper.place_order(order)

        # Portfolio value should be close to initial (minus slippage/commission)
        portfolio_value = paper.get_portfolio_value()
        assert 990000 < portfolio_value < 1010000  # Allow for slippage

    def test_get_account_balance(self):
        """Test account balance retrieval."""
        mock_broker = MockBroker()
        paper = PaperTradingBroker(1000000, mock_broker)

        balance = paper.get_account_balance()

        assert balance.available_cash == 1000000
        assert balance.total_balance == 1000000


class TestRiskController:
    """Test risk controller."""

    def test_initialization(self):
        """Test risk controller initialization."""
        risk = RiskController()

        assert risk.daily_pnl == 0.0
        assert risk.daily_order_count == 0
        assert isinstance(risk.limits, RiskLimits)

    def test_position_size_limit(self):
        """Test position size limit enforcement."""
        risk = RiskController()

        # Create order for 20% of portfolio (exceeds 10% limit)
        order = Order(
            symbol="RELIANCE",
            side=OrderSide.BUY,
            quantity=1000,
            order_type=OrderType.MARKET,
        )

        is_valid, reason = risk.validate_order(
            order, portfolio_value=1000000, current_positions=[], current_price=2450.0
        )

        assert not is_valid
        assert "position size" in reason.lower()

    def test_daily_loss_limit(self):
        """Test daily loss limit triggers."""
        risk = RiskController()

        # Simulate large loss
        risk.daily_pnl = -25000  # -2.5% of portfolio

        order = Order(
            symbol="TCS",
            side=OrderSide.BUY,
            quantity=1,
            order_type=OrderType.MARKET,
        )

        is_valid, reason = risk.validate_order(
            order, portfolio_value=1000000, current_positions=[], current_price=3500.0
        )

        assert not is_valid
        assert "daily loss" in reason.lower()

    def test_trade_value_limit(self):
        """Test trade value limit."""
        risk = RiskController()

        # Create order exceeding trade value limit (₹50k)
        order = Order(
            symbol="RELIANCE",
            side=OrderSide.BUY,
            quantity=30,  # 30 * 2450 = ₹73,500
            order_type=OrderType.MARKET,
        )

        is_valid, reason = risk.validate_order(
            order, portfolio_value=1000000, current_positions=[], current_price=2450.0
        )

        assert not is_valid
        assert "trade value" in reason.lower()

    def test_valid_order(self):
        """Test that valid order passes all checks."""
        risk = RiskController()

        order = Order(
            symbol="RELIANCE",
            side=OrderSide.BUY,
            quantity=10,  # Small position
            order_type=OrderType.MARKET,
        )

        is_valid, reason = risk.validate_order(
            order, portfolio_value=1000000, current_positions=[], current_price=2450.0
        )

        assert is_valid
        assert reason == "OK"

    def test_record_trade_result(self):
        """Test recording trade results."""
        risk = RiskController()

        risk.record_trade_result(500.0)
        assert risk.daily_pnl == 500.0

        risk.record_trade_result(-200.0)
        assert risk.daily_pnl == 300.0

    def test_daily_reset(self):
        """Test daily reset."""
        risk = RiskController()

        risk.daily_pnl = 1000.0
        risk.daily_order_count = 10
        risk.violations = [{"type": "test"}]

        risk.reset_daily()

        assert risk.daily_pnl == 0.0
        assert risk.daily_order_count == 0
        assert len(risk.violations) == 0


class TestCircuitBreaker:
    """Test circuit breaker."""

    def test_initialization(self):
        """Test circuit breaker initialization."""
        breaker = CircuitBreaker()

        assert breaker.active is False
        assert breaker.consecutive_losses == 0

    def test_consecutive_losses_trigger(self):
        """Test consecutive losses trigger circuit breaker."""
        breaker = CircuitBreaker()

        # Record 3 consecutive losses
        breaker.record_trade(-1000.0)
        assert breaker.consecutive_losses == 1
        assert not breaker.is_active()

        breaker.record_trade(-1500.0)
        assert breaker.consecutive_losses == 2
        assert not breaker.is_active()

        breaker.record_trade(-2000.0)
        assert breaker.consecutive_losses == 3
        assert breaker.is_active()

    def test_win_resets_consecutive_losses(self):
        """Test that winning trade resets consecutive losses."""
        breaker = CircuitBreaker()

        breaker.record_trade(-1000.0)
        breaker.record_trade(-1500.0)
        assert breaker.consecutive_losses == 2

        # Winning trade
        breaker.record_trade(500.0)
        assert breaker.consecutive_losses == 0

    def test_drawdown_trigger(self):
        """Test drawdown triggers circuit breaker."""
        breaker = CircuitBreaker()

        # Set peak value
        breaker.peak_portfolio_value = 1000000

        # Drop by 11% (exceeds 10% limit)
        breaker.record_trade(-110000, portfolio_value=890000)

        assert breaker.is_active()

    def test_force_activate(self):
        """Test manual circuit breaker activation."""
        breaker = CircuitBreaker()

        breaker.force_activate("Manual test")

        assert breaker.is_active()
        assert breaker.activation_reason == "Manual test"

    def test_reset(self):
        """Test circuit breaker reset."""
        breaker = CircuitBreaker()

        breaker.record_trade(-1000.0)
        breaker.record_trade(-1000.0)
        breaker.record_trade(-1000.0)

        assert breaker.is_active()

        breaker.reset()

        assert not breaker.is_active()
        assert breaker.consecutive_losses == 0


class TestBrokerFactory:
    """Test broker factory."""

    def test_create_paper_broker(self):
        """Test creating paper broker."""
        # Mock DhanHQ creation
        with patch("src.execution.broker_factory.DhanBroker") as mock_dhan:
            mock_dhan.return_value = MockBroker()

            broker = BrokerFactory.create("PAPER", initial_capital=500000)

            assert isinstance(broker, PaperTradingBroker)
            assert broker.initial_capital == 500000

    def test_unknown_broker_type(self):
        """Test that unknown broker type raises error."""
        with pytest.raises(ValueError, match="Unknown broker type"):
            BrokerFactory.create("UNKNOWN")

    def test_list_available_brokers(self):
        """Test listing available brokers."""
        brokers = BrokerFactory.list_available_brokers()

        assert "DHAN" in brokers
        assert "PAPER" in brokers
        assert "KITE" in brokers

        assert brokers["DHAN"]["implemented"] is True
        assert brokers["PAPER"]["implemented"] is True
        assert brokers["KITE"]["implemented"] is False

