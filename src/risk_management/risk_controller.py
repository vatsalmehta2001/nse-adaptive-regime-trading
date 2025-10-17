"""
Risk controller for pre-trade risk management.

This module provides comprehensive pre-trade risk checks to ensure
all orders comply with defined risk limits. It validates position sizes,
portfolio exposure, daily loss limits, and other risk parameters before
orders are sent to the broker.

Classes:
    RiskLimits: Risk limit configuration
    RiskController: Pre-trade risk management system

Example:
    >>> from src.risk_management import RiskController
    >>> from src.execution import Order, OrderSide, OrderType
    >>> 
    >>> risk = RiskController()
    >>> order = Order(symbol="RELIANCE", side=OrderSide.BUY, quantity=100,
    ...               order_type=OrderType.MARKET)
    >>> is_valid, reason = risk.validate_order(order, 1000000, [])
    >>> if is_valid:
    ...     # Safe to place order
    ...     pass
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime

from src.execution.broker_interface import Order, Position, OrderSide
from src.utils.helpers import load_config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class RiskLimits:
    """
    Risk limit configuration.

    Attributes:
        max_position_size_pct: Maximum position size as % of portfolio
        max_portfolio_exposure: Maximum total exposure as % of capital
        max_daily_loss_pct: Maximum daily loss as % before stopping
        max_trade_value: Maximum rupee value per trade
        min_capital_pct: Minimum cash % to maintain
        max_orders_per_day: Maximum orders allowed per day
        allow_shorting: Whether short selling is allowed
        intraday_only: Whether overnight positions are allowed
    """

    max_position_size_pct: float = 10.0
    max_portfolio_exposure: float = 100.0
    max_daily_loss_pct: float = 2.0
    max_trade_value: float = 50000.0
    min_capital_pct: float = 10.0
    max_orders_per_day: int = 50
    allow_shorting: bool = False
    intraday_only: bool = True


class RiskController:
    """
    Pre-trade risk management system.

    Validates all orders against risk limits before execution.
    Tracks daily P&L, order counts, and portfolio metrics.

    Attributes:
        limits: Risk limit configuration
        daily_pnl: Today's realized P&L
        daily_order_count: Number of orders placed today
        violations: List of risk violations
    """

    def __init__(self, config_path: str = "config/risk_config.yaml"):
        """
        Initialize risk controller.

        Args:
            config_path: Path to risk configuration file
        """
        try:
            config = load_config(config_path)
            limits_config = config.get("risk_limits", {})

            self.limits = RiskLimits(
                max_position_size_pct=limits_config.get("max_position_size_pct", 10.0),
                max_portfolio_exposure=limits_config.get("max_portfolio_exposure", 100.0),
                max_daily_loss_pct=limits_config.get("max_daily_loss_pct", 2.0),
                max_trade_value=limits_config.get("max_trade_value", 50000.0),
                min_capital_pct=limits_config.get("min_capital_pct", 10.0),
                max_orders_per_day=limits_config.get("max_orders_per_day", 50),
                allow_shorting=config.get("restrictions", {}).get("no_shorting", True) is False,
                intraday_only=config.get("restrictions", {}).get("intraday_only", True),
            )

        except Exception as e:
            logger.warning(f"Could not load risk config: {e}. Using defaults.")
            self.limits = RiskLimits()

        self.daily_pnl = 0.0
        self.daily_order_count = 0
        self.violations: List[Dict] = []
        self.last_reset_date = datetime.now().date()

        logger.info("Risk controller initialized")
        logger.info(f"Max position size: {self.limits.max_position_size_pct}%")
        logger.info(f"Max daily loss: {self.limits.max_daily_loss_pct}%")

    def validate_order(
        self,
        order: Order,
        portfolio_value: float,
        current_positions: List[Position],
        current_price: Optional[float] = None,
    ) -> tuple[bool, str]:
        """
        Validate order against all risk limits.

        Args:
            order: Order to validate
            portfolio_value: Current portfolio value
            current_positions: List of current positions
            current_price: Current market price (optional, estimated if not provided)

        Returns:
            Tuple of (is_valid, reason)
            - is_valid: True if order passes all checks
            - reason: Rejection reason if invalid, "OK" if valid
        """
        # Auto-reset daily counters if new day
        self._check_daily_reset()

        # Check 1: Daily order limit
        if self.daily_order_count >= self.limits.max_orders_per_day:
            self._log_violation(
                "daily_order_limit",
                f"Daily order limit {self.limits.max_orders_per_day} exceeded",
            )
            return False, f"Daily order limit ({self.limits.max_orders_per_day}) exceeded"

        # Check 2: Daily loss limit
        if self.daily_pnl < 0:
            loss_pct = abs(self.daily_pnl / portfolio_value) * 100
            if loss_pct >= self.limits.max_daily_loss_pct:
                self._log_violation(
                    "daily_loss_limit",
                    f"Day loss {loss_pct:.2f}% exceeds limit {self.limits.max_daily_loss_pct}%",
                )
                return (
                    False,
                    f"Daily loss limit exceeded ({loss_pct:.2f}% vs {self.limits.max_daily_loss_pct}%)",
                )

        # Check 3: Position size limit
        order_value = self._estimate_order_value(order, current_price)
        
        # Handle zero or negative portfolio value
        if portfolio_value <= 0:
            self._log_violation(
                "invalid_portfolio_value",
                f"Portfolio value is {portfolio_value}, cannot validate order"
            )
            return False, f"Invalid portfolio value: {portfolio_value}"
        
        position_size_pct = (order_value / portfolio_value) * 100

        if position_size_pct > self.limits.max_position_size_pct:
            self._log_violation(
                "position_size_limit",
                f"Position {position_size_pct:.2f}% > limit {self.limits.max_position_size_pct}%",
            )
            return (
                False,
                f"Position size {position_size_pct:.2f}% exceeds limit {self.limits.max_position_size_pct}%",
            )

        # Check 4: Trade value limit
        if order_value > self.limits.max_trade_value:
            self._log_violation(
                "trade_value_limit",
                f"Trade ₹{order_value:,.0f} > limit ₹{self.limits.max_trade_value:,.0f}",
            )
            return (
                False,
                f"Trade value ₹{order_value:,.0f} exceeds limit ₹{self.limits.max_trade_value:,.0f}",
            )

        # Check 5: Portfolio exposure
        total_exposure = self._calculate_exposure(current_positions, portfolio_value)

        # Adjust exposure for new order
        if order.side == OrderSide.BUY:
            new_exposure = total_exposure + position_size_pct
        else:
            # Selling reduces exposure (unless short selling)
            existing_position = self._find_position(order.symbol, current_positions)
            if existing_position and existing_position.quantity > 0:
                # Closing long position
                new_exposure = total_exposure - min(position_size_pct, existing_position.market_value / portfolio_value * 100)
            else:
                # Short selling
                if not self.limits.allow_shorting:
                    self._log_violation("shorting_not_allowed", f"Short selling not allowed for {order.symbol}")
                    return False, "Short selling not allowed"
                new_exposure = total_exposure + position_size_pct

        if new_exposure > self.limits.max_portfolio_exposure:
            self._log_violation(
                "exposure_limit",
                f"Exposure {new_exposure:.2f}% > limit {self.limits.max_portfolio_exposure}%",
            )
            return (
                False,
                f"Portfolio exposure {new_exposure:.2f}% exceeds limit {self.limits.max_portfolio_exposure}%",
            )

        # Check 6: Minimum capital requirement
        if order.side == OrderSide.BUY:
            # Calculate remaining cash after order
            remaining_cash_pct = ((portfolio_value - order_value) / portfolio_value) * 100
            if remaining_cash_pct < self.limits.min_capital_pct:
                self._log_violation(
                    "min_capital",
                    f"Remaining cash {remaining_cash_pct:.2f}% < minimum {self.limits.min_capital_pct}%",
                )
                return (
                    False,
                    f"Order would leave only {remaining_cash_pct:.2f}% cash (minimum: {self.limits.min_capital_pct}%)",
                )

        # Check 7: Intraday only restriction
        if self.limits.intraday_only and order.product != "MIS":
            self._log_violation(
                "intraday_only",
                f"Only MIS (intraday) orders allowed, got {order.product}",
            )
            return False, f"Only intraday (MIS) orders allowed"

        # All checks passed
        logger.debug(f"Order validation passed for {order.symbol}")
        return True, "OK"

    def record_order(self) -> None:
        """Record that an order was placed (increment daily count)."""
        self.daily_order_count += 1
        logger.debug(f"Daily order count: {self.daily_order_count}")

    def record_trade_result(self, pnl: float) -> None:
        """
        Record trade result for P&L tracking.

        Args:
            pnl: Trade P&L (positive or negative)
        """
        self.daily_pnl += pnl
        logger.info(f"Daily P&L: ₹{self.daily_pnl:,.2f}")

    def reset_daily(self) -> None:
        """Reset daily counters (called at market close or new day)."""
        logger.info(
            f"Daily reset. Final P&L: ₹{self.daily_pnl:,.2f}, "
            f"Orders: {self.daily_order_count}"
        )
        self.daily_pnl = 0.0
        self.daily_order_count = 0
        self.violations.clear()
        self.last_reset_date = datetime.now().date()

    def get_risk_metrics(self, portfolio_value: float) -> Dict:
        """
        Get current risk metrics.

        Args:
            portfolio_value: Current portfolio value

        Returns:
            Dictionary with risk metrics
        """
        daily_loss_pct = (
            (self.daily_pnl / portfolio_value) * 100 if portfolio_value > 0 else 0
        )

        return {
            "daily_pnl": self.daily_pnl,
            "daily_loss_pct": daily_loss_pct,
            "daily_order_count": self.daily_order_count,
            "max_daily_loss_pct": self.limits.max_daily_loss_pct,
            "max_orders_per_day": self.limits.max_orders_per_day,
            "loss_limit_remaining_pct": self.limits.max_daily_loss_pct + daily_loss_pct,
            "order_limit_remaining": self.limits.max_orders_per_day - self.daily_order_count,
            "violations_count": len(self.violations),
        }

    def get_violations(self) -> List[Dict]:
        """
        Get list of risk violations.

        Returns:
            List of violation dictionaries
        """
        return self.violations.copy()

    def _check_daily_reset(self) -> None:
        """Check if we need to reset daily counters (new trading day)."""
        current_date = datetime.now().date()
        if current_date > self.last_reset_date:
            self.reset_daily()

    def _estimate_order_value(self, order: Order, current_price: Optional[float] = None) -> float:
        """
        Estimate order value for risk checks.

        Args:
            order: Order to estimate
            current_price: Current market price (if known)

        Returns:
            Estimated order value
        """
        # Use limit price if available, otherwise use current price
        if order.price is not None:
            price = order.price
        elif current_price is not None:
            price = current_price
        else:
            # Fallback: use trigger price or assume a reasonable price
            price = order.trigger_price if order.trigger_price else 100.0
            logger.warning(
                f"No price information for {order.symbol}, using {price} for risk check"
            )

        return price * order.quantity

    def _calculate_exposure(
        self, positions: List[Position], portfolio_value: float
    ) -> float:
        """
        Calculate current portfolio exposure percentage.

        Args:
            positions: Current positions
            portfolio_value: Portfolio value

        Returns:
            Exposure as percentage
        """
        if portfolio_value <= 0:
            return 0.0

        total_exposure = sum(abs(pos.market_value) for pos in positions)
        return (total_exposure / portfolio_value) * 100

    def _find_position(self, symbol: str, positions: List[Position]) -> Optional[Position]:
        """
        Find position for a symbol.

        Args:
            symbol: Trading symbol
            positions: List of positions

        Returns:
            Position if found, None otherwise
        """
        for pos in positions:
            if pos.symbol == symbol:
                return pos
        return None

    def _log_violation(self, violation_type: str, details: str) -> None:
        """
        Log risk violation.

        Args:
            violation_type: Type of violation
            details: Violation details
        """
        violation = {
            "timestamp": datetime.now(),
            "type": violation_type,
            "details": details,
        }
        self.violations.append(violation)
        logger.warning(f"Risk violation: {violation_type} - {details}")

