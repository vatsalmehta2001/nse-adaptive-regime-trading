"""
Circuit breaker for emergency trading stops.

This module implements a circuit breaker mechanism that automatically
stops trading when certain risk thresholds are breached. It tracks
consecutive losses, maximum drawdown, and other critical metrics to
protect against catastrophic losses.

Classes:
    CircuitBreaker: Emergency stop mechanism

Example:
    >>> from src.risk_management import CircuitBreaker
    >>> breaker = CircuitBreaker()
    >>> 
    >>> # Record trade results
    >>> breaker.record_trade(-5000)  # Loss
    >>> breaker.record_trade(-3000)  # Another loss
    >>> breaker.record_trade(-2000)  # Third loss
    >>> 
    >>> if breaker.is_active():
    ...     print("Circuit breaker activated! Stop trading.")
"""

from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime, timedelta

from src.utils.helpers import load_config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class CircuitBreakerConfig:
    """
    Circuit breaker configuration.

    Attributes:
        enabled: Whether circuit breaker is enabled
        consecutive_losses: Trigger after N consecutive losses
        cooldown_minutes: Minutes to wait before resuming
        max_drawdown_pct: Maximum drawdown before triggering
        loss_velocity_threshold: Maximum loss per hour
    """

    enabled: bool = True
    consecutive_losses: int = 3
    cooldown_minutes: int = 30
    max_drawdown_pct: float = 10.0
    loss_velocity_threshold: float = 50000.0  # ₹50k per hour


class CircuitBreaker:
    """
    Emergency stop mechanism for trading.

    Monitors trading activity and automatically stops trading when
    risk thresholds are breached. Tracks consecutive losses, drawdown,
    and loss velocity to protect against catastrophic losses.

    Attributes:
        config: Circuit breaker configuration
        active: Whether circuit breaker is currently active
        consecutive_losses: Current consecutive loss count
        peak_portfolio_value: Peak portfolio value for drawdown calculation
        activation_time: When circuit breaker was activated
        trade_history: Recent trade history
    """

    def __init__(self, config_path: str = "config/risk_config.yaml"):
        """
        Initialize circuit breaker.

        Args:
            config_path: Path to risk configuration file
        """
        try:
            config = load_config(config_path)
            cb_config = config.get("risk_limits", {}).get("circuit_breaker", {})

            self.config = CircuitBreakerConfig(
                enabled=cb_config.get("enabled", True),
                consecutive_losses=cb_config.get("consecutive_losses", 3),
                cooldown_minutes=cb_config.get("cooldown_minutes", 30),
                max_drawdown_pct=cb_config.get("max_drawdown_pct", 10.0),
                loss_velocity_threshold=cb_config.get("loss_velocity_threshold", 50000.0),
            )

        except Exception as e:
            logger.warning(f"Could not load circuit breaker config: {e}. Using defaults.")
            self.config = CircuitBreakerConfig()

        self.active = False
        self.consecutive_losses = 0
        self.peak_portfolio_value = 0.0
        self.activation_time: Optional[datetime] = None
        self.trade_history: List[dict] = []
        self.activation_reason = ""

        logger.info("Circuit breaker initialized")
        logger.info(f"Consecutive loss trigger: {self.config.consecutive_losses}")
        logger.info(f"Max drawdown trigger: {self.config.max_drawdown_pct}%")

    def record_trade(
        self, pnl: float, portfolio_value: Optional[float] = None
    ) -> None:
        """
        Record trade result and check circuit breaker conditions.

        Args:
            pnl: Trade P&L (positive or negative)
            portfolio_value: Current portfolio value (optional)
        """
        if not self.config.enabled:
            return

        # Record trade
        trade_record = {
            "timestamp": datetime.now(),
            "pnl": pnl,
            "portfolio_value": portfolio_value,
        }
        self.trade_history.append(trade_record)

        # Clean old trades (keep last 24 hours)
        self._clean_old_trades()

        # Update peak portfolio value
        if portfolio_value is not None and portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio_value

        # Track consecutive losses
        if pnl < 0:
            self.consecutive_losses += 1
            logger.warning(f"Consecutive losses: {self.consecutive_losses}")

            # Check consecutive loss trigger
            if self.consecutive_losses >= self.config.consecutive_losses:
                self._activate(
                    f"Consecutive losses: {self.consecutive_losses} >= {self.config.consecutive_losses}"
                )
                return

        else:
            # Reset on win
            if self.consecutive_losses > 0:
                logger.info("Consecutive loss streak broken")
            self.consecutive_losses = 0

        # Check drawdown
        if portfolio_value is not None and self.peak_portfolio_value > 0:
            drawdown_pct = (
                (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value * 100
            )

            if drawdown_pct >= self.config.max_drawdown_pct:
                self._activate(
                    f"Drawdown {drawdown_pct:.2f}% >= {self.config.max_drawdown_pct}%"
                )
                return

        # Check loss velocity
        hourly_loss = self._calculate_hourly_loss()
        if hourly_loss < -self.config.loss_velocity_threshold:
            self._activate(
                f"Loss velocity ₹{abs(hourly_loss):,.0f}/hr exceeds "
                f"₹{self.config.loss_velocity_threshold:,.0f}/hr"
            )
            return

    def is_active(self) -> bool:
        """
        Check if circuit breaker is currently active.

        Returns:
            True if active (trading should stop)
        """
        if not self.active:
            return False

        # Check if cooldown period has elapsed
        if self.activation_time is not None:
            cooldown_end = self.activation_time + timedelta(
                minutes=self.config.cooldown_minutes
            )
            if datetime.now() >= cooldown_end:
                self._deactivate()
                return False

        return True

    def force_activate(self, reason: str) -> None:
        """
        Manually activate circuit breaker.

        Args:
            reason: Reason for activation
        """
        self._activate(reason)

    def force_deactivate(self) -> None:
        """Manually deactivate circuit breaker."""
        self._deactivate()

    def reset(self) -> None:
        """Reset circuit breaker (new trading day)."""
        logger.info("Circuit breaker reset for new trading day")
        self.active = False
        self.consecutive_losses = 0
        self.peak_portfolio_value = 0.0
        self.activation_time = None
        self.trade_history.clear()
        self.activation_reason = ""

    def get_status(self) -> dict:
        """
        Get circuit breaker status.

        Returns:
            Dictionary with status information
        """
        status = {
            "active": self.is_active(),
            "consecutive_losses": self.consecutive_losses,
            "activation_reason": self.activation_reason,
            "activation_time": self.activation_time,
        }

        if self.activation_time is not None:
            elapsed = datetime.now() - self.activation_time
            remaining = timedelta(minutes=self.config.cooldown_minutes) - elapsed
            status["cooldown_remaining_minutes"] = max(0, remaining.total_seconds() / 60)

        return status

    def get_metrics(self, current_portfolio_value: Optional[float] = None) -> dict:
        """
        Get circuit breaker metrics.

        Args:
            current_portfolio_value: Current portfolio value

        Returns:
            Dictionary with metrics
        """
        drawdown = 0.0
        if current_portfolio_value and self.peak_portfolio_value > 0:
            drawdown = (
                (self.peak_portfolio_value - current_portfolio_value)
                / self.peak_portfolio_value
                * 100
            )

        hourly_loss = self._calculate_hourly_loss()

        return {
            "consecutive_losses": self.consecutive_losses,
            "max_consecutive_losses": self.config.consecutive_losses,
            "current_drawdown_pct": drawdown,
            "max_drawdown_pct": self.config.max_drawdown_pct,
            "hourly_loss_rate": hourly_loss,
            "max_hourly_loss": -self.config.loss_velocity_threshold,
            "recent_trades_count": len(self.trade_history),
        }

    def _activate(self, reason: str) -> None:
        """
        Activate circuit breaker.

        Args:
            reason: Reason for activation
        """
        if self.active:
            return  # Already active

        self.active = True
        self.activation_time = datetime.now()
        self.activation_reason = reason

        logger.error(f"CIRCUIT BREAKER ACTIVATED: {reason}")
        logger.error(
            f"Trading stopped for {self.config.cooldown_minutes} minutes"
        )

    def _deactivate(self) -> None:
        """Deactivate circuit breaker after cooldown."""
        if not self.active:
            return

        logger.warning("Circuit breaker deactivated after cooldown period")
        logger.warning("Trading resumed - monitor carefully")

        self.active = False
        self.activation_time = None
        self.activation_reason = ""
        self.consecutive_losses = 0  # Reset on deactivation

    def _calculate_hourly_loss(self) -> float:
        """
        Calculate loss rate per hour from recent trades.

        Returns:
            Loss per hour (negative if losing)
        """
        if not self.trade_history:
            return 0.0

        # Get trades from last hour
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_trades = [
            t for t in self.trade_history
            if t["timestamp"] >= one_hour_ago
        ]

        if not recent_trades:
            return 0.0

        # Sum P&L from last hour
        total_pnl = sum(t["pnl"] for t in recent_trades)

        # Calculate time span
        oldest_trade = min(recent_trades, key=lambda t: t["timestamp"])
        time_span_hours = (
            (datetime.now() - oldest_trade["timestamp"]).total_seconds() / 3600
        )

        if time_span_hours == 0:
            return 0.0

        # Extrapolate to hourly rate
        hourly_rate = total_pnl / time_span_hours

        return hourly_rate

    def _clean_old_trades(self) -> None:
        """Remove trades older than 24 hours."""
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.trade_history = [
            t for t in self.trade_history
            if t["timestamp"] >= cutoff_time
        ]

