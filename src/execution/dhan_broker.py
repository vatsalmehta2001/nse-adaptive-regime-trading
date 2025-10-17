"""
DhanHQ broker implementation for the NSE trading system.

This module implements the BrokerInterface for DhanHQ API, supporting both
sandbox (testing) and live trading modes. It handles all DhanHQ-specific
API interactions and translates between universal Order objects and
DhanHQ's native format.

Classes:
    DhanBroker: DhanHQ broker implementation

Example:
    >>> from src.execution.dhan_broker import DhanBroker
    >>> broker = DhanBroker(
    ...     client_id="1234567890",
    ...     access_token="your_token",
    ...     mode="SANDBOX"
    ... )
    >>> broker.authenticate()
    >>> order = Order(symbol="RELIANCE", side=OrderSide.BUY, quantity=1,
    ...               order_type=OrderType.MARKET)
    >>> order_id = broker.place_order(order)
"""

from typing import Dict, List, Optional
import time
from datetime import datetime

try:
    from dhanhq import dhanhq, DhanContext
except ImportError:
    dhanhq = None
    DhanContext = None

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


class DhanBroker(BrokerInterface):
    """
    DhanHQ broker implementation.

    Supports both sandbox (testing) and live trading modes. The sandbox
    provides virtual capital for testing without risking real money.

    Attributes:
        client_id: DhanHQ client ID
        access_token: DhanHQ access token
        mode: 'SANDBOX' or 'LIVE'
        dhan: DhanHQ API client instance
        max_retries: Maximum API retry attempts
        retry_delay: Delay between retries in seconds
    """

    # Exchange segment mapping
    EXCHANGE_MAP = {"NSE": "NSE_EQ", "BSE": "BSE_EQ"}

    # Order type mapping (universal -> DhanHQ)
    ORDER_TYPE_MAP = {
        OrderType.MARKET: "MARKET",
        OrderType.LIMIT: "LIMIT",
        OrderType.STOP_LOSS: "STOP_LOSS",
        OrderType.STOP_LOSS_MARKET: "STOP_LOSS_MARKET",
    }

    # Product type mapping (universal -> DhanHQ)
    PRODUCT_MAP = {
        "MIS": "INTRADAY",  # DhanHQ uses INTRADAY instead of MIS
        "CNC": "CNC",
        "NRML": "MARGIN",
    }

    # Reverse mappings for response parsing
    REVERSE_ORDER_STATUS_MAP = {
        "PENDING": OrderStatus.PENDING,
        "TRANSIT": OrderStatus.PLACED,
        "TRADED": OrderStatus.FILLED,
        "PARTIALLY_FILLED": OrderStatus.PARTIAL,
        "CANCELLED": OrderStatus.CANCELLED,
        "REJECTED": OrderStatus.REJECTED,
    }

    def __init__(
        self,
        client_id: str,
        access_token: str,
        mode: str = "SANDBOX",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize DhanHQ broker.

        Args:
            client_id: DhanHQ client ID
            access_token: DhanHQ access token
            mode: 'SANDBOX' or 'LIVE'
            max_retries: Maximum API retry attempts
            retry_delay: Delay between retries in seconds

        Raises:
            ValueError: If mode is invalid
            ImportError: If dhanhq library not installed
        """
        if dhanhq is None or DhanContext is None:
            raise ImportError(
                "dhanhq library not installed. Install with: pip install dhanhq==2.0.2"
            )

        if mode not in ["SANDBOX", "LIVE"]:
            raise ValueError("Mode must be 'SANDBOX' or 'LIVE'")

        self.client_id = client_id
        self.access_token = access_token
        self.mode = mode
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Initialize DhanHQ client with DhanContext (v2.0.2+)
        try:
            # Try v2.x API with DhanContext
            dhan_context = DhanContext(client_id, access_token)
            self.dhan = dhanhq(dhan_context)
            logger.info(f"DhanHQ broker initialized in {mode} mode (API v2.0.2)")
        except Exception as e:
            # Fallback to v1.x API if DhanContext fails
            logger.warning(f"Failed to initialize with DhanContext: {e}")
            try:
                self.dhan = dhanhq(client_id, access_token)
                logger.info(f"DhanHQ broker initialized in {mode} mode (API v1.x fallback)")
            except Exception as e2:
                logger.error(f"Failed to initialize DhanHQ client: {e2}")
                raise

    def authenticate(self) -> bool:
        """
        Authenticate with DhanHQ.

        Tests authentication by attempting to fetch fund limits.

        Returns:
            True if authentication successful, False otherwise
        """
        try:
            response = self._api_call(self.dhan.get_fund_limits)

            if response and response.get("status") == "success":
                logger.info("DhanHQ authentication successful")
                return True
            else:
                logger.error(f"DhanHQ authentication failed: {response}")
                return False

        except Exception as e:
            logger.error(f"DhanHQ authentication error: {e}")
            return False

    def place_order(self, order: Order) -> str:
        """
        Place order with DhanHQ.

        Args:
            order: Order object with all details

        Returns:
            Broker-assigned order ID

        Raises:
            ValueError: If order parameters invalid
            RuntimeError: If order placement fails
        """
        try:
            # Get security ID for symbol
            security_id = self._get_security_id(order.symbol, order.exchange)

            # Build DhanHQ order request
            dhan_order = {
                "security_id": security_id,
                "exchange_segment": self.EXCHANGE_MAP[order.exchange],
                "transaction_type": order.side.value,
                "quantity": order.quantity,
                "order_type": self.ORDER_TYPE_MAP[order.order_type],
                "product_type": self.PRODUCT_MAP.get(order.product, "INTRADAY"),
                "validity": order.validity,
            }

            # Add price for limit orders
            if order.order_type == OrderType.LIMIT:
                if order.price is None:
                    raise ValueError("Limit order requires price")
                dhan_order["price"] = order.price

            # Add trigger price for stop loss orders
            if order.order_type in [OrderType.STOP_LOSS, OrderType.STOP_LOSS_MARKET]:
                if order.trigger_price is None:
                    raise ValueError("Stop loss order requires trigger_price")
                dhan_order["trigger_price"] = order.trigger_price
                if order.order_type == OrderType.STOP_LOSS and order.price:
                    dhan_order["price"] = order.price

            # Add tag if provided
            if order.tag:
                dhan_order["tag"] = order.tag

            # Place order with retry logic
            logger.info(
                f"Placing order: {order.symbol} {order.side.value} "
                f"{order.quantity} @ {order.order_type.value}"
            )

            response = self._api_call(self.dhan.place_order, **dhan_order)

            if response and response.get("status") == "success":
                order_id = str(response["data"]["orderId"])
                order.order_id = order_id
                order.status = OrderStatus.PLACED
                order.timestamp = datetime.now()
                logger.info(f"Order placed successfully: {order_id}")
                return order_id
            else:
                error_msg = response.get("remarks", "Unknown error") if response else "No response"
                logger.error(f"Order placement failed: {error_msg}")
                raise RuntimeError(f"Order placement failed: {error_msg}")

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
        Modify an existing pending order.

        Args:
            order_id: DhanHQ order ID
            quantity: New quantity (if changing)
            price: New price (if changing)
            trigger_price: New trigger price (if changing)

        Returns:
            True if modification successful

        Raises:
            ValueError: If order_id not found or invalid parameters
            RuntimeError: If modification fails
        """
        try:
            # Build modification request
            modify_params = {"order_id": order_id}

            if quantity is not None:
                modify_params["quantity"] = quantity
            if price is not None:
                modify_params["price"] = price
            if trigger_price is not None:
                modify_params["trigger_price"] = trigger_price

            logger.info(f"Modifying order {order_id}")
            response = self._api_call(self.dhan.modify_order, **modify_params)

            if response and response.get("status") == "success":
                logger.info(f"Order {order_id} modified successfully")
                return True
            else:
                error_msg = response.get("remarks", "Unknown error") if response else "No response"
                logger.error(f"Order modification failed: {error_msg}")
                raise RuntimeError(f"Order modification failed: {error_msg}")

        except Exception as e:
            logger.error(f"Error modifying order {order_id}: {e}")
            raise

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order.

        Args:
            order_id: DhanHQ order ID

        Returns:
            True if cancellation successful

        Raises:
            ValueError: If order_id not found
            RuntimeError: If cancellation fails
        """
        try:
            logger.info(f"Cancelling order {order_id}")
            response = self._api_call(self.dhan.cancel_order, order_id=order_id)

            if response and response.get("status") == "success":
                logger.info(f"Order {order_id} cancelled successfully")
                return True
            else:
                error_msg = response.get("remarks", "Unknown error") if response else "No response"
                logger.error(f"Order cancellation failed: {error_msg}")
                raise RuntimeError(f"Order cancellation failed: {error_msg}")

        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            raise

    def get_order_status(self, order_id: str) -> Order:
        """
        Get current status of an order.

        Args:
            order_id: DhanHQ order ID

        Returns:
            Order object with current status

        Raises:
            ValueError: If order ID not found
        """
        try:
            response = self._api_call(self.dhan.get_order_by_id, order_id=order_id)

            if response and response.get("status") == "success":
                order_data = response["data"]
                return self._parse_order_response(order_data)
            else:
                raise ValueError(f"Order {order_id} not found")

        except Exception as e:
            logger.error(f"Error fetching order status for {order_id}: {e}")
            raise

    def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        """
        Get all orders (optionally filtered by status).

        Args:
            status: Filter by order status (None = all orders)

        Returns:
            List of Order objects
        """
        try:
            response = self._api_call(self.dhan.get_order_list)

            if not response or response.get("status") != "success":
                logger.warning("No orders or API error")
                return []

            orders = []
            for order_data in response.get("data", []):
                order = self._parse_order_response(order_data)
                if status is None or order.status == status:
                    orders.append(order)

            return orders

        except Exception as e:
            logger.error(f"Error fetching orders: {e}")
            return []

    def get_positions(self) -> List[Position]:
        """
        Get all current positions.

        Returns:
            List of Position objects
        """
        try:
            response = self._api_call(self.dhan.get_positions)

            if not response or response.get("status") != "success":
                logger.warning("No positions or API error")
                return []

            positions = []
            for pos_data in response.get("data", []):
                position = Position(
                    symbol=pos_data.get("tradingSymbol", ""),
                    quantity=pos_data.get("netQty", 0),
                    average_price=pos_data.get("avgPrice", 0.0),
                    last_price=pos_data.get("lastPrice", 0.0),
                    pnl=pos_data.get("realizedProfit", 0.0) + pos_data.get("unrealizedProfit", 0.0),
                    day_pnl=pos_data.get("dayPnl", 0.0),
                    exchange=self._parse_exchange(pos_data.get("exchangeSegment", "NSE_EQ")),
                    product=self._parse_product(pos_data.get("productType", "INTRADAY")),
                )
                positions.append(position)

            return positions

        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return []

    def get_account_balance(self) -> AccountBalance:
        """
        Get account balance and margin details.

        Returns:
            AccountBalance object

        Raises:
            ConnectionError: If unable to fetch balance
        """
        try:
            response = self._api_call(self.dhan.get_fund_limits)

            if not response or response.get("status") != "success":
                raise ConnectionError("Failed to fetch account balance")

            data = response.get("data", {})

            balance = AccountBalance(
                available_cash=data.get("availabelBalance", 0.0),
                used_margin=data.get("utilizedAmount", 0.0),
                total_balance=data.get("sodLimit", 0.0),
                collateral=data.get("collateralAmount", 0.0),
            )

            return balance

        except Exception as e:
            logger.error(f"Error fetching account balance: {e}")
            raise ConnectionError(f"Failed to fetch account balance: {e}")

    def get_ltp(self, symbol: str, exchange: str = "NSE") -> float:
        """
        Get last traded price for a symbol.

        Args:
            symbol: Trading symbol
            exchange: Exchange (NSE or BSE)

        Returns:
            Last traded price

        Raises:
            ValueError: If symbol not found
        """
        try:
            security_id = self._get_security_id(symbol, exchange)
            exchange_segment = self.EXCHANGE_MAP[exchange]

            response = self._api_call(
                self.dhan.get_ltp,
                security_id=security_id,
                exchange_segment=exchange_segment,
            )

            if response and response.get("status") == "success":
                ltp = response["data"].get("last_price", 0.0)
                return float(ltp)
            else:
                raise ValueError(f"Failed to fetch LTP for {symbol}")

        except Exception as e:
            logger.error(f"Error fetching LTP for {symbol}: {e}")
            raise

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
        try:
            security_id = self._get_security_id(symbol, exchange)
            exchange_segment = self.EXCHANGE_MAP[exchange]

            response = self._api_call(
                self.dhan.get_quote,
                security_id=security_id,
                exchange_segment=exchange_segment,
            )

            if response and response.get("status") == "success":
                data = response["data"]
                return {
                    "symbol": symbol,
                    "ltp": data.get("last_price", 0.0),
                    "open": data.get("open", 0.0),
                    "high": data.get("high", 0.0),
                    "low": data.get("low", 0.0),
                    "close": data.get("prev_close", 0.0),
                    "volume": data.get("volume", 0),
                    "bid": data.get("bid_price", 0.0),
                    "ask": data.get("ask_price", 0.0),
                    "bid_qty": data.get("bid_qty", 0),
                    "ask_qty": data.get("ask_qty", 0),
                }
            else:
                raise ValueError(f"Failed to fetch quote for {symbol}")

        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            raise

    def close_all_positions(self) -> bool:
        """
        Emergency function to close all open positions.

        Returns:
            True if all positions closed successfully

        Raises:
            RuntimeError: If position closing fails
        """
        try:
            positions = self.get_positions()

            if not positions:
                logger.info("No positions to close")
                return True

            logger.warning(f"Closing all positions ({len(positions)} positions)")

            success_count = 0
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
                    tag="EMERGENCY_CLOSE",
                )

                try:
                    order_id = self.place_order(close_order)
                    logger.info(f"Closed position {position.symbol}: {order_id}")
                    success_count += 1
                except Exception as e:
                    logger.error(f"Failed to close position {position.symbol}: {e}")

            if success_count == len([p for p in positions if p.quantity != 0]):
                logger.info("All positions closed successfully")
                return True
            else:
                logger.warning(f"Closed {success_count}/{len(positions)} positions")
                return False

        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
            raise RuntimeError(f"Failed to close all positions: {e}")

    def cancel_all_orders(self) -> bool:
        """
        Emergency function to cancel all pending orders.

        Returns:
            True if all orders cancelled

        Raises:
            RuntimeError: If cancellation fails
        """
        try:
            orders = self.get_orders(status=OrderStatus.PLACED)

            if not orders:
                logger.info("No pending orders to cancel")
                return True

            logger.warning(f"Cancelling all orders ({len(orders)} orders)")

            success_count = 0
            for order in orders:
                try:
                    if order.order_id:
                        self.cancel_order(order.order_id)
                        success_count += 1
                except Exception as e:
                    logger.error(f"Failed to cancel order {order.order_id}: {e}")

            if success_count == len(orders):
                logger.info("All orders cancelled successfully")
                return True
            else:
                logger.warning(f"Cancelled {success_count}/{len(orders)} orders")
                return False

        except Exception as e:
            logger.error(f"Error cancelling all orders: {e}")
            raise RuntimeError(f"Failed to cancel all orders: {e}")

    def _api_call(self, func, **kwargs):
        """
        Make API call with retry logic.

        Args:
            func: API function to call
            **kwargs: Arguments to pass to function

        Returns:
            API response

        Raises:
            RuntimeError: If all retries fail
        """
        for attempt in range(self.max_retries):
            try:
                response = func(**kwargs)
                return response
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"API call failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                    )
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"API call failed after {self.max_retries} attempts: {e}")
                    raise RuntimeError(f"API call failed: {e}")

    def _get_security_id(self, symbol: str, exchange: str) -> str:
        """
        Get DhanHQ security ID for symbol.

        Note: This is a simplified implementation. For production use,
        maintain a proper symbol-to-security_id mapping database.

        Args:
            symbol: Trading symbol
            exchange: Exchange

        Returns:
            Security ID string

        Raises:
            NotImplementedError: For live mode without proper mapping
        """
        if self.mode == "SANDBOX":
            # Sandbox uses dummy security IDs
            # Common sandbox IDs for testing
            sandbox_ids = {
                "RELIANCE": "1333",
                "TCS": "11536",
                "INFY": "1594",
                "HDFCBANK": "1333",
            }
            security_id = sandbox_ids.get(symbol, "1333")
            logger.debug(f"Using sandbox security ID {security_id} for {symbol}")
            return security_id
        else:
            # For live mode, need proper security ID lookup
            raise NotImplementedError(
                f"Security ID lookup not implemented for LIVE mode. "
                f"Please implement symbol-to-security_id mapping for {symbol}"
            )

    def _parse_order_response(self, order_data: Dict) -> Order:
        """
        Parse DhanHQ order response to universal Order object.

        Args:
            order_data: Order data from DhanHQ API

        Returns:
            Order object
        """
        # Parse order side
        side_str = order_data.get("transactionType", "BUY")
        side = OrderSide.BUY if side_str == "BUY" else OrderSide.SELL

        # Parse order type
        order_type_str = order_data.get("orderType", "MARKET")
        order_type = OrderType.MARKET
        for ot, dhan_str in self.ORDER_TYPE_MAP.items():
            if dhan_str == order_type_str:
                order_type = ot
                break

        # Parse status
        status_str = order_data.get("orderStatus", "PENDING")
        status = self.REVERSE_ORDER_STATUS_MAP.get(status_str, OrderStatus.PENDING)

        # Parse product
        product_str = order_data.get("productType", "INTRADAY")
        product = "MIS"
        for prod, dhan_str in self.PRODUCT_MAP.items():
            if dhan_str == product_str:
                product = prod
                break

        return Order(
            symbol=order_data.get("tradingSymbol", ""),
            side=side,
            quantity=order_data.get("quantity", 0),
            order_type=order_type,
            price=order_data.get("price"),
            trigger_price=order_data.get("triggerPrice"),
            product=product,
            validity=order_data.get("validity", "DAY"),
            order_id=str(order_data.get("orderId", "")),
            status=status,
            filled_quantity=order_data.get("filledQty", 0),
            average_price=order_data.get("avgPrice"),
            exchange=self._parse_exchange(order_data.get("exchangeSegment", "NSE_EQ")),
            tag=order_data.get("tag"),
        )

    def _parse_exchange(self, exchange_segment: str) -> str:
        """Parse DhanHQ exchange segment to universal exchange name."""
        if "NSE" in exchange_segment:
            return "NSE"
        elif "BSE" in exchange_segment:
            return "BSE"
        return "NSE"

    def _parse_product(self, product_type: str) -> str:
        """Parse DhanHQ product type to universal product."""
        for prod, dhan_str in self.PRODUCT_MAP.items():
            if dhan_str == product_type:
                return prod
        return "MIS"

