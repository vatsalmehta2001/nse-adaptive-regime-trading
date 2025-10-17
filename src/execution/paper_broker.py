"""
Paper trading broker implementation with realistic simulation.

This module provides a paper trading broker that simulates order execution
using live market data. It's perfect for strategy testing without risking
real money. The simulation includes realistic features like slippage,
partial fills, and commission.

Classes:
    PaperTradingBroker: Paper trading implementation

Example:
    >>> from src.execution.paper_broker import PaperTradingBroker
    >>> from src.execution.dhan_broker import DhanBroker
    >>> 
    >>> # Use DhanHQ sandbox for live data
    >>> data_broker = DhanBroker("client_id", "token", "SANDBOX")
    >>> paper = PaperTradingBroker(1000000, data_broker)
    >>> 
    >>> order = Order(symbol="RELIANCE", side=OrderSide.BUY, quantity=10,
    ...               order_type=OrderType.MARKET)
    >>> order_id = paper.place_order(order)
"""

from typing import Dict, List, Optional
from datetime import datetime
import random
import uuid

from src.execution.broker_interface import (
    BrokerInterface,
    Order,
    Position,
    AccountBalance,
    OrderType,
    OrderSide,
    OrderStatus,
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class PaperTradingBroker(BrokerInterface):
    """
    Paper trading implementation with realistic simulation.

    Uses live market data but simulates order execution locally.
    No real money involved - perfect for strategy testing.

    Attributes:
        initial_capital: Starting virtual capital
        data_broker: Broker to fetch live market data
        cash: Available cash
        positions: Current virtual positions
        orders: All orders (pending + filled)
        trades: Trade history
        commission_pct: Commission percentage
        slippage_pct: Slippage percentage
        fill_probability: Limit order fill probability
    """

    def __init__(
        self,
        initial_capital: float,
        data_broker: BrokerInterface,
        commission_pct: float = 0.03,
        slippage_pct: float = 0.10,
        fill_probability: float = 0.95,
    ):
        """
        Initialize paper trading.

        Args:
            initial_capital: Starting capital (e.g., 1000000)
            data_broker: Broker instance for live data (e.g., DhanBroker)
            commission_pct: Commission as percentage (default 0.03%)
            slippage_pct: Slippage as percentage (default 0.10%)
            fill_probability: Probability of limit order fill (default 95%)
        """
        self.initial_capital = initial_capital
        self.data_broker = data_broker
        self.cash = initial_capital
        self.commission_pct = commission_pct / 100.0  # Convert to decimal
        self.slippage_pct = slippage_pct / 100.0
        self.fill_probability = fill_probability

        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.trades: List[Dict] = []
        self.order_counter = 0

        logger.info(f"Paper trading initialized with ₹{initial_capital:,.0f}")

    def authenticate(self) -> bool:
        """
        Authenticate paper trading (always succeeds).

        Returns:
            True always
        """
        logger.info("Paper trading authentication (no-op)")
        return True

    def place_order(self, order: Order) -> str:
        """
        Simulate order placement.

        For market orders: Fill immediately at current price + slippage
        For limit orders: Mark as pending, check fill on each update
        For stop loss: Monitor trigger price

        Args:
            order: Order object

        Returns:
            Paper order ID

        Raises:
            ValueError: If insufficient funds
        """
        try:
            # Generate paper order ID
            self.order_counter += 1
            order.order_id = f"PAPER_{self.order_counter:06d}_{uuid.uuid4().hex[:8]}"
            order.timestamp = datetime.now()

            # Get current market price
            try:
                ltp = self.data_broker.get_ltp(order.symbol, order.exchange)
                # Handle None or invalid LTP
                if ltp is None or not isinstance(ltp, (int, float)) or ltp <= 0:
                    raise ValueError(f"Invalid LTP: {ltp}")
            except Exception as e:
                logger.error(f"Failed to get LTP for {order.symbol}: {e}")
                # Use dummy price if data fetch fails
                ltp = order.price if order.price else 100.0
                logger.warning(f"Using fallback price {ltp} for {order.symbol}")

            # Check if we have enough cash for buy orders
            if order.side == OrderSide.BUY:
                estimated_value = ltp * order.quantity
                required_cash = estimated_value * (1 + self.commission_pct + self.slippage_pct)
                if required_cash > self.cash:
                    raise ValueError(
                        f"Insufficient funds: need ₹{required_cash:,.2f}, "
                        f"have ₹{self.cash:,.2f}"
                    )

            # Handle different order types
            if order.order_type == OrderType.MARKET:
                # Market order: fill immediately with slippage
                fill_price = self._simulate_market_fill(ltp, order.side, order.quantity)
                self._execute_fill(order, fill_price, order.quantity)
                logger.info(
                    f"Paper market order filled: {order.symbol} "
                    f"{order.quantity} @ ₹{fill_price:.2f}"
                )

            elif order.order_type == OrderType.LIMIT:
                # Limit order: check if can fill immediately
                if self._can_fill_limit_order(order, ltp):
                    fill_price = order.price
                    self._execute_fill(order, fill_price, order.quantity)
                    logger.info(
                        f"Paper limit order filled immediately: {order.symbol} "
                        f"{order.quantity} @ ₹{fill_price:.2f}"
                    )
                else:
                    # Mark as pending
                    order.status = OrderStatus.PLACED
                    self.orders.append(order)
                    logger.info(
                        f"Paper limit order placed: {order.symbol} "
                        f"{order.quantity} @ ₹{order.price:.2f}"
                    )

            elif order.order_type in [OrderType.STOP_LOSS, OrderType.STOP_LOSS_MARKET]:
                # Stop loss: mark as pending, will trigger when price hits
                order.status = OrderStatus.PLACED
                self.orders.append(order)
                logger.info(
                    f"Paper stop loss order placed: {order.symbol} "
                    f"trigger @ ₹{order.trigger_price:.2f}"
                )

            return order.order_id

        except Exception as e:
            logger.error(f"Error placing paper order: {e}")
            raise

    def modify_order(
        self,
        order_id: str,
        quantity: Optional[int] = None,
        price: Optional[float] = None,
        trigger_price: Optional[float] = None,
    ) -> bool:
        """
        Modify a pending paper order.

        Args:
            order_id: Paper order ID
            quantity: New quantity
            price: New price
            trigger_price: New trigger price

        Returns:
            True if modification successful

        Raises:
            ValueError: If order not found or already filled
        """
        try:
            # Find the order
            order = None
            for o in self.orders:
                if o.order_id == order_id:
                    order = o
                    break

            if not order:
                raise ValueError(f"Order {order_id} not found")

            if order.status not in [OrderStatus.PLACED, OrderStatus.PENDING]:
                raise ValueError(
                    f"Cannot modify order in status {order.status.value}"
                )

            # Modify parameters
            if quantity is not None:
                order.quantity = quantity
            if price is not None:
                order.price = price
            if trigger_price is not None:
                order.trigger_price = trigger_price

            logger.info(f"Paper order {order_id} modified")
            return True

        except Exception as e:
            logger.error(f"Error modifying paper order {order_id}: {e}")
            raise

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending paper order.

        Args:
            order_id: Paper order ID

        Returns:
            True if cancellation successful

        Raises:
            ValueError: If order not found
        """
        try:
            # Find the order
            for order in self.orders:
                if order.order_id == order_id:
                    if order.status in [OrderStatus.PLACED, OrderStatus.PENDING]:
                        order.status = OrderStatus.CANCELLED
                        logger.info(f"Paper order {order_id} cancelled")
                        return True
                    else:
                        raise ValueError(
                            f"Cannot cancel order in status {order.status.value}"
                        )

            raise ValueError(f"Order {order_id} not found")

        except Exception as e:
            logger.error(f"Error cancelling paper order {order_id}: {e}")
            raise

    def get_order_status(self, order_id: str) -> Order:
        """
        Get status of a paper order.

        Args:
            order_id: Paper order ID

        Returns:
            Order object

        Raises:
            ValueError: If order not found
        """
        for order in self.orders:
            if order.order_id == order_id:
                return order

        raise ValueError(f"Order {order_id} not found")

    def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        """
        Get all paper orders.

        Args:
            status: Filter by status

        Returns:
            List of Order objects
        """
        if status is None:
            return self.orders.copy()
        else:
            return [o for o in self.orders if o.status == status]

    def get_positions(self) -> List[Position]:
        """
        Get all paper positions.

        Returns:
            List of Position objects
        """
        positions = []
        for symbol, pos in self.positions.items():
            if pos.quantity != 0:
                # Update last price from market data
                try:
                    current_price = self.data_broker.get_ltp(symbol, pos.exchange)
                    pos.last_price = current_price

                    # Recalculate P&L
                    unrealized = pos.quantity * (current_price - pos.average_price)
                    pos.pnl = unrealized  # For paper trading, all P&L is unrealized until close

                except Exception as e:
                    logger.warning(f"Failed to update price for {symbol}: {e}")

                positions.append(pos)

        return positions

    def get_account_balance(self) -> AccountBalance:
        """
        Get paper account balance.

        Returns:
            AccountBalance object
        """
        # Calculate total portfolio value
        positions_value = 0.0
        for pos in self.get_positions():
            positions_value += pos.quantity * pos.last_price

        total_value = self.cash + positions_value
        used_margin = positions_value

        return AccountBalance(
            available_cash=self.cash,
            used_margin=used_margin,
            total_balance=total_value,
            collateral=0.0,
        )

    def get_ltp(self, symbol: str, exchange: str = "NSE") -> float:
        """
        Get last traded price (delegates to data broker).

        Args:
            symbol: Trading symbol
            exchange: Exchange

        Returns:
            Last traded price

        Raises:
            ValueError: If symbol not found
        """
        return self.data_broker.get_ltp(symbol, exchange)

    def get_quote(self, symbol: str, exchange: str = "NSE") -> Dict:
        """
        Get market quote (delegates to data broker).

        Args:
            symbol: Trading symbol
            exchange: Exchange

        Returns:
            Quote dictionary

        Raises:
            ValueError: If symbol not found
        """
        return self.data_broker.get_quote(symbol, exchange)

    def close_all_positions(self) -> bool:
        """
        Close all paper positions.

        Returns:
            True if all positions closed
        """
        try:
            positions = self.get_positions()

            if not positions:
                logger.info("No positions to close")
                return True

            logger.warning(f"Closing all paper positions ({len(positions)} positions)")

            for position in positions:
                if position.quantity == 0:
                    continue

                # Create closing order
                close_side = OrderSide.SELL if position.quantity > 0 else OrderSide.BUY
                close_order = Order(
                    symbol=position.symbol,
                    side=close_side,
                    quantity=abs(position.quantity),
                    order_type=OrderType.MARKET,
                    product=position.product,
                    exchange=position.exchange,
                    tag="CLOSE_ALL",
                )

                self.place_order(close_order)

            logger.info("All paper positions closed")
            return True

        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
            return False

    def cancel_all_orders(self) -> bool:
        """
        Cancel all pending paper orders.

        Returns:
            True if all orders cancelled
        """
        try:
            pending_orders = [
                o for o in self.orders
                if o.status in [OrderStatus.PLACED, OrderStatus.PENDING]
            ]

            if not pending_orders:
                logger.info("No pending orders to cancel")
                return True

            logger.warning(f"Cancelling all paper orders ({len(pending_orders)} orders)")

            for order in pending_orders:
                order.status = OrderStatus.CANCELLED

            logger.info("All paper orders cancelled")
            return True

        except Exception as e:
            logger.error(f"Error cancelling all orders: {e}")
            return False

    def get_portfolio_value(self) -> float:
        """
        Calculate total portfolio value.

        Returns:
            Cash + market value of all positions
        """
        balance = self.get_account_balance()
        return balance.total_balance

    def get_trades(self) -> List[Dict]:
        """
        Get all executed trades.

        Returns:
            List of trade dictionaries
        """
        return self.trades.copy()

    def update_pending_orders(self) -> None:
        """
        Update pending orders (check for fills).

        This should be called periodically to check if limit orders
        can be filled at current market prices.
        """
        for order in self.orders:
            if order.status != OrderStatus.PLACED:
                continue

            try:
                ltp = self.data_broker.get_ltp(order.symbol, order.exchange)

                # Check limit orders
                if order.order_type == OrderType.LIMIT:
                    if self._can_fill_limit_order(order, ltp):
                        # Simulate partial or full fill
                        if random.random() < self.fill_probability:
                            self._execute_fill(order, order.price, order.quantity)
                            logger.info(
                                f"Paper limit order filled: {order.symbol} "
                                f"{order.quantity} @ ₹{order.price:.2f}"
                            )

                # Check stop loss orders
                elif order.order_type in [OrderType.STOP_LOSS, OrderType.STOP_LOSS_MARKET]:
                    if self._should_trigger_stop_loss(order, ltp):
                        # Trigger stop loss
                        if order.order_type == OrderType.STOP_LOSS:
                            fill_price = order.price
                        else:
                            fill_price = self._simulate_market_fill(
                                ltp, order.side, order.quantity
                            )
                        self._execute_fill(order, fill_price, order.quantity)
                        logger.info(
                            f"Paper stop loss triggered: {order.symbol} "
                            f"{order.quantity} @ ₹{fill_price:.2f}"
                        )

            except Exception as e:
                logger.warning(f"Error updating order {order.order_id}: {e}")

    def _simulate_market_fill(
        self, ltp: float, side: OrderSide, quantity: int
    ) -> float:
        """
        Simulate realistic market order fill with slippage.

        Args:
            ltp: Last traded price
            side: BUY or SELL
            quantity: Order quantity

        Returns:
            Realistic fill price
        """
        # Base slippage
        base_slippage = self.slippage_pct

        # Additional slippage for larger orders
        quantity_impact = (quantity / 1000) * 0.0005  # 0.05% per 1000 shares
        total_slippage = base_slippage + quantity_impact

        if side == OrderSide.BUY:
            # Pay more when buying (bid-ask spread + slippage)
            fill_price = ltp * (1 + total_slippage)
        else:
            # Get less when selling
            fill_price = ltp * (1 - total_slippage)

        return round(fill_price, 2)

    def _execute_fill(self, order: Order, fill_price: float, fill_quantity: int) -> None:
        """
        Execute order fill and update portfolio.

        Args:
            order: Order being filled
            fill_price: Execution price
            fill_quantity: Quantity filled
        """
        # Update order status
        order.filled_quantity += fill_quantity
        order.average_price = fill_price

        if order.filled_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIAL

        # Add to orders list if not already there
        if order not in self.orders:
            self.orders.append(order)

        # Calculate trade value and commission
        trade_value = fill_price * fill_quantity
        commission = trade_value * self.commission_pct

        # Update cash
        if order.side == OrderSide.BUY:
            self.cash -= (trade_value + commission)
        else:
            self.cash += (trade_value - commission)

        # Update position
        self._update_position(
            order.symbol,
            order.side,
            fill_quantity,
            fill_price,
            order.exchange,
            order.product,
        )

        # Log trade
        trade = {
            "timestamp": datetime.now(),
            "symbol": order.symbol,
            "side": order.side.value,
            "quantity": fill_quantity,
            "price": fill_price,
            "commission": commission,
            "order_id": order.order_id,
            "trade_value": trade_value,
        }
        self.trades.append(trade)

        logger.debug(
            f"Paper trade executed: {order.symbol} {order.side.value} "
            f"{fill_quantity} @ ₹{fill_price:.2f} (commission: ₹{commission:.2f})"
        )

    def _update_position(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        price: float,
        exchange: str,
        product: str,
    ) -> None:
        """
        Update position after fill.

        Args:
            symbol: Trading symbol
            side: BUY or SELL
            quantity: Fill quantity
            price: Fill price
            exchange: Exchange
            product: Product type
        """
        if symbol not in self.positions:
            # New position
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity if side == OrderSide.BUY else -quantity,
                average_price=price,
                last_price=price,
                pnl=0.0,
                day_pnl=0.0,
                exchange=exchange,
                product=product,
            )
        else:
            # Existing position
            pos = self.positions[symbol]

            if side == OrderSide.BUY:
                # Buying: average up or reverse short
                if pos.quantity >= 0:
                    # Long or flat: average up
                    total_qty = pos.quantity + quantity
                    pos.average_price = (
                        (pos.average_price * pos.quantity + price * quantity) / total_qty
                    )
                    pos.quantity = total_qty
                else:
                    # Short: cover position
                    realized_pnl = quantity * (pos.average_price - price)
                    pos.pnl += realized_pnl
                    pos.quantity += quantity
                    if pos.quantity > 0:
                        # Reversed to long
                        pos.average_price = price

            else:  # SELL
                # Selling: reduce long or increase short
                if pos.quantity > 0:
                    # Long: reduce or close
                    realized_pnl = quantity * (price - pos.average_price)
                    pos.pnl += realized_pnl
                    pos.quantity -= quantity
                    if pos.quantity < 0:
                        # Reversed to short
                        pos.average_price = price
                else:
                    # Short or flat: average down
                    total_qty = abs(pos.quantity) + quantity
                    pos.average_price = (
                        (pos.average_price * abs(pos.quantity) + price * quantity) / total_qty
                    )
                    pos.quantity = -total_qty

            pos.last_price = price

    def _can_fill_limit_order(self, order: Order, ltp: float) -> bool:
        """
        Check if limit order can be filled at current price.

        Args:
            order: Limit order
            ltp: Current last traded price

        Returns:
            True if order can fill
        """
        if order.price is None:
            return False

        if order.side == OrderSide.BUY:
            # Buy limit: fill if market price <= limit price
            return ltp <= order.price
        else:
            # Sell limit: fill if market price >= limit price
            return ltp >= order.price

    def _should_trigger_stop_loss(self, order: Order, ltp: float) -> bool:
        """
        Check if stop loss should trigger.

        Args:
            order: Stop loss order
            ltp: Current last traded price

        Returns:
            True if stop loss should trigger
        """
        if order.trigger_price is None:
            return False

        if order.side == OrderSide.BUY:
            # Buy stop: trigger if price >= trigger
            return ltp >= order.trigger_price
        else:
            # Sell stop: trigger if price <= trigger
            return ltp <= order.trigger_price

