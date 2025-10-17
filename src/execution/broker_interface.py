"""
Abstract broker interface for the NSE trading system.

This module defines the contract that all broker implementations must follow.
It ensures consistent API across different brokers (DhanHQ, Kite Connect, etc.)
and enables easy switching between brokers with minimal code changes.

Classes:
    OrderType: Enumeration of supported order types
    OrderSide: Enumeration of order sides (buy/sell)
    OrderStatus: Enumeration of order status lifecycle
    Order: Universal order representation
    Position: Universal position representation
    AccountBalance: Account balance and margin information
    BrokerInterface: Abstract base class for all brokers
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum


class OrderType(Enum):
    """Order types supported by brokers."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LOSS_MARKET = "STOP_LOSS_MARKET"


class OrderSide(Enum):
    """Order side (buy/sell)."""

    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Order status lifecycle."""

    PENDING = "PENDING"
    PLACED = "PLACED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Order:
    """
    Universal order representation.

    This dataclass provides a broker-agnostic order structure that works
    across all broker implementations. Each broker translates this to its
    native format.

    Attributes:
        symbol: Trading symbol (e.g., "RELIANCE")
        side: BUY or SELL
        quantity: Number of shares
        order_type: MARKET, LIMIT, STOP_LOSS, or STOP_LOSS_MARKET
        price: Limit price for LIMIT orders (None for market orders)
        trigger_price: Stop loss trigger price (None if not stop loss)
        product: MIS (intraday), CNC (delivery), or NRML (F&O)
        validity: DAY or IOC (Immediate or Cancel)
        order_id: Broker-assigned order ID (set after placement)
        status: Current order status
        filled_quantity: Quantity filled so far
        average_price: Average fill price
        timestamp: Order creation time
        exchange: NSE or BSE
        tag: Optional tag for order identification
    """

    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    price: Optional[float] = None
    trigger_price: Optional[float] = None
    product: str = "MIS"
    validity: str = "DAY"
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    average_price: Optional[float] = None
    timestamp: Optional[datetime] = field(default_factory=datetime.now)
    exchange: str = "NSE"
    tag: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate order parameters."""
        if self.quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {self.quantity}")

        if self.order_type == OrderType.LIMIT and self.price is None:
            raise ValueError("LIMIT order requires price")

        if self.order_type in [OrderType.STOP_LOSS, OrderType.STOP_LOSS_MARKET]:
            if self.trigger_price is None:
                raise ValueError("Stop loss order requires trigger_price")

        if self.price is not None and self.price <= 0:
            raise ValueError(f"Price must be positive, got {self.price}")

        if self.trigger_price is not None and self.trigger_price <= 0:
            raise ValueError(f"Trigger price must be positive, got {self.trigger_price}")


@dataclass
class Position:
    """
    Universal position representation.

    Attributes:
        symbol: Trading symbol
        quantity: Position quantity (positive=long, negative=short)
        average_price: Average entry price
        last_price: Current market price
        pnl: Realized + unrealized P&L
        day_pnl: Today's P&L
        exchange: NSE or BSE
        product: MIS, CNC, or NRML
    """

    symbol: str
    quantity: int
    average_price: float
    last_price: float
    pnl: float
    day_pnl: float
    exchange: str = "NSE"
    product: str = "MIS"

    @property
    def market_value(self) -> float:
        """Calculate current market value of position."""
        return self.quantity * self.last_price

    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L."""
        return self.quantity * (self.last_price - self.average_price)


@dataclass
class AccountBalance:
    """
    Account balance and margin information.

    Attributes:
        available_cash: Cash available for trading
        used_margin: Margin currently used
        total_balance: Total account value
        collateral: Collateral available
    """

    available_cash: float
    used_margin: float
    total_balance: float
    collateral: float = 0.0

    @property
    def free_margin(self) -> float:
        """Calculate free margin."""
        return self.available_cash - self.used_margin


class BrokerInterface(ABC):
    """
    Abstract base class for broker implementations.

    All broker implementations must inherit from this class and implement
    all abstract methods. This ensures consistent interface across different
    brokers (DhanHQ, Kite Connect, etc.) and enables easy broker switching.

    The interface defines essential trading operations:
    - Authentication
    - Order placement, modification, and cancellation
    - Position and balance retrieval
    - Market data queries
    - Emergency functions (close all, cancel all)
    """

    @abstractmethod
    def authenticate(self) -> bool:
        """
        Authenticate with broker API.

        Returns:
            True if authentication successful, False otherwise

        Raises:
            ConnectionError: If unable to connect to broker
            ValueError: If credentials invalid
        """
        pass

    @abstractmethod
    def place_order(self, order: Order) -> str:
        """
        Place a new order with the broker.

        Args:
            order: Order object with all details

        Returns:
            Broker-assigned order ID

        Raises:
            ValueError: If order parameters invalid
            RuntimeError: If order placement fails
        """
        pass

    @abstractmethod
    def modify_order(
        self,
        order_id: str,
        quantity: Optional[int] = None,
        price: Optional[float] = None,
        trigger_price: Optional[float] = None,
    ) -> bool:
        """
        Modify an existing pending order.

        Args:
            order_id: Broker order ID to modify
            quantity: New quantity (if changing)
            price: New price (if changing)
            trigger_price: New trigger price (if changing)

        Returns:
            True if modification successful

        Raises:
            ValueError: If order_id not found or invalid parameters
            RuntimeError: If modification fails
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order.

        Args:
            order_id: Broker order ID to cancel

        Returns:
            True if cancellation successful

        Raises:
            ValueError: If order_id not found
            RuntimeError: If cancellation fails
        """
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> Order:
        """
        Get current status of an order.

        Args:
            order_id: Broker order ID

        Returns:
            Order object with current status

        Raises:
            ValueError: If order ID not found
        """
        pass

    @abstractmethod
    def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        """
        Get all orders (optionally filtered by status).

        Args:
            status: Filter by order status (None = all orders)

        Returns:
            List of Order objects
        """
        pass

    @abstractmethod
    def get_positions(self) -> List[Position]:
        """
        Get all current positions.

        Returns:
            List of Position objects

        Raises:
            ConnectionError: If unable to fetch positions
        """
        pass

    @abstractmethod
    def get_account_balance(self) -> AccountBalance:
        """
        Get account balance and margin details.

        Returns:
            AccountBalance object

        Raises:
            ConnectionError: If unable to fetch balance
        """
        pass

    @abstractmethod
    def get_ltp(self, symbol: str, exchange: str = "NSE") -> float:
        """
        Get last traded price for a symbol.

        Args:
            symbol: Trading symbol (e.g., "RELIANCE")
            exchange: Exchange (NSE or BSE)

        Returns:
            Last traded price

        Raises:
            ValueError: If symbol not found
        """
        pass

    @abstractmethod
    def get_quote(self, symbol: str, exchange: str = "NSE") -> Dict:
        """
        Get full market quote for a symbol.

        Args:
            symbol: Trading symbol
            exchange: Exchange (NSE or BSE)

        Returns:
            Dictionary with OHLC, bid, ask, volume, etc.

        Raises:
            ValueError: If symbol not found
        """
        pass

    @abstractmethod
    def close_all_positions(self) -> bool:
        """
        Emergency function to close all open positions.

        Places market orders to close all positions immediately.
        Use with caution.

        Returns:
            True if all positions closed successfully

        Raises:
            RuntimeError: If position closing fails
        """
        pass

    @abstractmethod
    def cancel_all_orders(self) -> bool:
        """
        Emergency function to cancel all pending orders.

        Returns:
            True if all orders cancelled

        Raises:
            RuntimeError: If cancellation fails
        """
        pass

