"""
Order lifecycle management for the NSE trading system.

This module provides order management capabilities including order tracking,
status monitoring, and order history persistence. It acts as a wrapper around
the broker interface to provide additional features like order logging,
status caching, and order analytics.

Classes:
    OrderManager: Order lifecycle management

Example:
    >>> from src.execution.order_manager import OrderManager
    >>> from src.execution.broker_factory import BrokerFactory
    >>> 
    >>> broker = BrokerFactory.create("PAPER", initial_capital=1000000)
    >>> manager = OrderManager(broker)
    >>> 
    >>> order = Order(symbol="RELIANCE", side=OrderSide.BUY, quantity=10,
    ...               order_type=OrderType.MARKET)
    >>> order_id = manager.place_order(order)
    >>> status = manager.get_order_status(order_id)
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import threading
import time

from src.execution.broker_interface import (
    BrokerInterface,
    Order,
    OrderStatus,
)
from src.utils.logging_config import get_logger
from src.utils.database import DatabaseManager

logger = get_logger(__name__)


class OrderManager:
    """
    Order lifecycle management.

    Provides order tracking, status monitoring, and persistence.
    Acts as a middleware between strategy code and broker.

    Attributes:
        broker: Underlying broker instance
        db: Database manager for order persistence
        order_cache: In-memory cache of recent orders
        monitoring: Whether order monitoring is enabled
    """

    def __init__(
        self,
        broker: BrokerInterface,
        db_path: str = "data/trading_db.duckdb",
        enable_monitoring: bool = True,
        cache_duration_hours: int = 24,
    ):
        """
        Initialize order manager.

        Args:
            broker: Broker instance to manage
            db_path: Path to database for order persistence
            enable_monitoring: Enable background order status monitoring
            cache_duration_hours: Hours to keep orders in cache
        """
        self.broker = broker
        self.db = DatabaseManager(db_path)
        self.order_cache: Dict[str, Order] = {}
        self.cache_duration = timedelta(hours=cache_duration_hours)
        self.monitoring = enable_monitoring
        self._monitor_thread = None
        self._stop_monitoring = threading.Event()

        # Create orders table if not exists
        self._create_orders_table()

        if enable_monitoring:
            self._start_monitoring()

        logger.info("Order manager initialized")

    def place_order(self, order: Order, validate: bool = True) -> str:
        """
        Place order and log to database.

        Args:
            order: Order to place
            validate: Validate order before placing

        Returns:
            Order ID

        Raises:
            ValueError: If order validation fails
            RuntimeError: If order placement fails
        """
        try:
            # Place order with broker
            order_id = self.broker.place_order(order)

            # Add to cache
            self.order_cache[order_id] = order

            # Log to database
            self._log_order_to_db(order)

            logger.info(f"Order placed and logged: {order_id}")

            return order_id

        except Exception as e:
            logger.error(f"Error placing order: {e}")
            raise

    def modify_order(
        self,
        order_id: str,
        quantity: Optional[int] = None,
        price: Optional[float] = None,
        trigger_price: Optional[float] = None,
    ) -> bool:
        """
        Modify existing order.

        Args:
            order_id: Order ID to modify
            quantity: New quantity
            price: New price
            trigger_price: New trigger price

        Returns:
            True if modification successful

        Raises:
            ValueError: If order not found
            RuntimeError: If modification fails
        """
        try:
            success = self.broker.modify_order(order_id, quantity, price, trigger_price)

            if success:
                # Update cache
                if order_id in self.order_cache:
                    order = self.order_cache[order_id]
                    if quantity is not None:
                        order.quantity = quantity
                    if price is not None:
                        order.price = price
                    if trigger_price is not None:
                        order.trigger_price = trigger_price

                # Log modification
                self._log_order_modification_to_db(
                    order_id, quantity, price, trigger_price
                )

                logger.info(f"Order modified: {order_id}")

            return success

        except Exception as e:
            logger.error(f"Error modifying order {order_id}: {e}")
            raise

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancellation successful

        Raises:
            ValueError: If order not found
            RuntimeError: If cancellation fails
        """
        try:
            success = self.broker.cancel_order(order_id)

            if success:
                # Update cache
                if order_id in self.order_cache:
                    self.order_cache[order_id].status = OrderStatus.CANCELLED

                # Log cancellation
                self._log_order_cancellation_to_db(order_id)

                logger.info(f"Order cancelled: {order_id}")

            return success

        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            raise

    def get_order_status(self, order_id: str, use_cache: bool = True) -> Order:
        """
        Get order status.

        Args:
            order_id: Order ID
            use_cache: Use cached status if available

        Returns:
            Order object with current status

        Raises:
            ValueError: If order not found
        """
        # Check cache first
        if use_cache and order_id in self.order_cache:
            return self.order_cache[order_id]

        # Fetch from broker
        order = self.broker.get_order_status(order_id)

        # Update cache
        self.order_cache[order_id] = order

        return order

    def get_orders(
        self,
        status: Optional[OrderStatus] = None,
        symbol: Optional[str] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
    ) -> List[Order]:
        """
        Get orders with filtering.

        Args:
            status: Filter by order status
            symbol: Filter by symbol
            from_date: Filter by start date
            to_date: Filter by end date

        Returns:
            List of Order objects
        """
        # Get orders from broker
        orders = self.broker.get_orders(status)

        # Apply additional filters
        if symbol is not None:
            orders = [o for o in orders if o.symbol == symbol]

        if from_date is not None:
            orders = [o for o in orders if o.timestamp and o.timestamp >= from_date]

        if to_date is not None:
            orders = [o for o in orders if o.timestamp and o.timestamp <= to_date]

        return orders

    def get_order_history(
        self,
        symbol: Optional[str] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get order history from database.

        Args:
            symbol: Filter by symbol
            from_date: Filter by start date
            to_date: Filter by end date
            limit: Maximum number of orders to return

        Returns:
            List of order dictionaries
        """
        try:
            query = "SELECT * FROM orders WHERE 1=1"
            params = []

            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)

            if from_date:
                query += " AND timestamp >= ?"
                params.append(from_date.isoformat())

            if to_date:
                query += " AND timestamp <= ?"
                params.append(to_date.isoformat())

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            result = self.db.execute_query(query, params)

            return result.fetchall() if result else []

        except Exception as e:
            logger.error(f"Error fetching order history: {e}")
            return []

    def get_order_statistics(
        self,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Get order statistics.

        Args:
            from_date: Start date for statistics
            to_date: End date for statistics

        Returns:
            Dictionary with order statistics
        """
        try:
            orders = self.get_order_history(
                from_date=from_date,
                to_date=to_date,
                limit=10000,
            )

            if not orders:
                return {
                    "total_orders": 0,
                    "filled_orders": 0,
                    "cancelled_orders": 0,
                    "rejected_orders": 0,
                    "pending_orders": 0,
                    "fill_rate": 0.0,
                }

            status_counts = {}
            for order in orders:
                status = order.get("status", "UNKNOWN")
                status_counts[status] = status_counts.get(status, 0) + 1

            total = len(orders)
            filled = status_counts.get("FILLED", 0)

            return {
                "total_orders": total,
                "filled_orders": filled,
                "cancelled_orders": status_counts.get("CANCELLED", 0),
                "rejected_orders": status_counts.get("REJECTED", 0),
                "pending_orders": status_counts.get("PENDING", 0) + status_counts.get("PLACED", 0),
                "fill_rate": (filled / total * 100) if total > 0 else 0.0,
            }

        except Exception as e:
            logger.error(f"Error calculating order statistics: {e}")
            return {}

    def cleanup_cache(self) -> None:
        """Clean up old orders from cache."""
        cutoff_time = datetime.now() - self.cache_duration
        orders_to_remove = []

        for order_id, order in self.order_cache.items():
            if order.timestamp and order.timestamp < cutoff_time:
                orders_to_remove.append(order_id)

        for order_id in orders_to_remove:
            del self.order_cache[order_id]

        if orders_to_remove:
            logger.debug(f"Cleaned up {len(orders_to_remove)} orders from cache")

    def stop(self) -> None:
        """Stop order manager and monitoring."""
        if self.monitoring and self._monitor_thread:
            logger.info("Stopping order monitoring")
            self._stop_monitoring.set()
            self._monitor_thread.join(timeout=5)

        logger.info("Order manager stopped")

    def _start_monitoring(self) -> None:
        """Start background order status monitoring."""
        self._monitor_thread = threading.Thread(
            target=self._monitor_orders,
            daemon=True,
        )
        self._monitor_thread.start()
        logger.info("Order monitoring started")

    def _monitor_orders(self) -> None:
        """Background thread to monitor order status."""
        while not self._stop_monitoring.is_set():
            try:
                # Get pending orders
                pending_orders = [
                    o for o in self.order_cache.values()
                    if o.status in [OrderStatus.PENDING, OrderStatus.PLACED, OrderStatus.PARTIAL]
                ]

                # Update status
                for order in pending_orders:
                    if order.order_id:
                        try:
                            updated_order = self.broker.get_order_status(order.order_id)
                            self.order_cache[order.order_id] = updated_order

                            # Log status change
                            if updated_order.status != order.status:
                                logger.info(
                                    f"Order status changed: {order.order_id} "
                                    f"{order.status.value} -> {updated_order.status.value}"
                                )
                                self._log_order_status_change_to_db(
                                    order.order_id,
                                    order.status.value,
                                    updated_order.status.value,
                                )

                        except Exception as e:
                            logger.warning(
                                f"Error updating order status for {order.order_id}: {e}"
                            )

                # Cleanup cache periodically
                self.cleanup_cache()

                # Sleep before next check
                time.sleep(5)  # Check every 5 seconds

            except Exception as e:
                logger.error(f"Error in order monitoring: {e}")
                time.sleep(5)

    def _create_orders_table(self) -> None:
        """Create orders table in database if not exists."""
        try:
            create_table_query = """
                CREATE TABLE IF NOT EXISTS orders (
                    order_id VARCHAR PRIMARY KEY,
                    symbol VARCHAR NOT NULL,
                    side VARCHAR NOT NULL,
                    quantity INTEGER NOT NULL,
                    order_type VARCHAR NOT NULL,
                    price DOUBLE,
                    trigger_price DOUBLE,
                    product VARCHAR,
                    validity VARCHAR,
                    status VARCHAR NOT NULL,
                    filled_quantity INTEGER,
                    average_price DOUBLE,
                    timestamp TIMESTAMP NOT NULL,
                    exchange VARCHAR,
                    tag VARCHAR
                )
            """
            self.db.execute_query(create_table_query)

            # Create order modifications table
            create_mods_query = """
                CREATE TABLE IF NOT EXISTS order_modifications (
                    id INTEGER PRIMARY KEY,
                    order_id VARCHAR NOT NULL,
                    modification_type VARCHAR NOT NULL,
                    old_value VARCHAR,
                    new_value VARCHAR,
                    timestamp TIMESTAMP NOT NULL
                )
            """
            self.db.execute_query(create_mods_query)

            logger.debug("Orders tables created/verified")

        except Exception as e:
            logger.error(f"Error creating orders table: {e}")

    def _log_order_to_db(self, order: Order) -> None:
        """Log order to database."""
        try:
            insert_query = """
                INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            self.db.execute_query(
                insert_query,
                (
                    order.order_id,
                    order.symbol,
                    order.side.value,
                    order.quantity,
                    order.order_type.value,
                    order.price,
                    order.trigger_price,
                    order.product,
                    order.validity,
                    order.status.value,
                    order.filled_quantity,
                    order.average_price,
                    order.timestamp.isoformat() if order.timestamp else datetime.now().isoformat(),
                    order.exchange,
                    order.tag,
                ),
            )
        except Exception as e:
            logger.error(f"Error logging order to database: {e}")

    def _log_order_modification_to_db(
        self,
        order_id: str,
        quantity: Optional[int],
        price: Optional[float],
        trigger_price: Optional[float],
    ) -> None:
        """Log order modification to database."""
        try:
            timestamp = datetime.now().isoformat()

            if quantity is not None:
                self.db.execute_query(
                    "INSERT INTO order_modifications VALUES (NULL, ?, ?, ?, ?, ?)",
                    (order_id, "QUANTITY", None, str(quantity), timestamp),
                )

            if price is not None:
                self.db.execute_query(
                    "INSERT INTO order_modifications VALUES (NULL, ?, ?, ?, ?, ?)",
                    (order_id, "PRICE", None, str(price), timestamp),
                )

            if trigger_price is not None:
                self.db.execute_query(
                    "INSERT INTO order_modifications VALUES (NULL, ?, ?, ?, ?, ?)",
                    (order_id, "TRIGGER_PRICE", None, str(trigger_price), timestamp),
                )

        except Exception as e:
            logger.error(f"Error logging order modification: {e}")

    def _log_order_cancellation_to_db(self, order_id: str) -> None:
        """Log order cancellation to database."""
        try:
            self.db.execute_query(
                "UPDATE orders SET status = ? WHERE order_id = ?",
                ("CANCELLED", order_id),
            )
        except Exception as e:
            logger.error(f"Error logging order cancellation: {e}")

    def _log_order_status_change_to_db(
        self, order_id: str, old_status: str, new_status: str
    ) -> None:
        """Log order status change to database."""
        try:
            # Update order status
            self.db.execute_query(
                "UPDATE orders SET status = ? WHERE order_id = ?",
                (new_status, order_id),
            )

            # Log modification
            self.db.execute_query(
                "INSERT INTO order_modifications VALUES (NULL, ?, ?, ?, ?, ?)",
                (
                    order_id,
                    "STATUS",
                    old_status,
                    new_status,
                    datetime.now().isoformat(),
                ),
            )

        except Exception as e:
            logger.error(f"Error logging order status change: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

