"""
Risk management module for the NSE trading system.

This module provides comprehensive risk management capabilities including
pre-trade risk checks, position limits, loss limits, and circuit breaker
functionality for emergency stops.

Components:
    - RiskController: Pre-trade risk validation
    - CircuitBreaker: Emergency stop mechanism

Example:
    >>> from src.risk_management import RiskController
    >>> risk = RiskController()
    >>> is_valid, reason = risk.validate_order(order, portfolio_value, positions)
    >>> if not is_valid:
    ...     print(f"Order rejected: {reason}")
"""

from src.risk_management.risk_controller import RiskController, RiskLimits
from src.risk_management.circuit_breaker import CircuitBreaker

__all__ = [
    "RiskController",
    "RiskLimits",
    "CircuitBreaker",
]
