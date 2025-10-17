"""
Kite Connect broker implementation for the NSE trading system.

This module implements the BrokerInterface for Zerodha's Kite Connect API.
It supports live trading with Zerodha Kite and provides a complete
implementation of all broker operations.

Classes:
    KiteBroker: Kite Connect broker implementation

Example:
    >>> from src.execution.kite_broker import KiteBroker
    >>> broker = KiteBroker(
    ...     api_key="your_api_key",
    ...     api_secret="your_api_secret",
    ...     access_token="your_access_token"
    ... )
    >>> broker.authenticate()
    >>> order = Order(symbol="RELIANCE", side=OrderSide.BUY, quantity=1,
    ...               order_type=OrderType.MARKET)
    >>> order_id = broker.place_order(order)

Note:
    Kite Connect requires a login flow to generate access token.
    See: https://kite.trade/docs/connect/v3/
"""

from typing import Dict, List, Optional
import time
from datetime import datetime

try:
    from kiteconnect import KiteConnect
    KITE_AVAILABLE = True
except ImportError:
    KiteConnect = None
    KITE_AVAILABLE = False

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


class KiteBroker(BrokerInterface):
    """
    Kite Connect broker implementation.

    Implements the BrokerInterface for Zerodha's Kite Connect API.
    Supports live trading with full order management and market data.

    Attributes:
        api_key: Kite Connect API key
        api_secret: Kite Connect API secret
        access_token: Kite Connect access token
        kite: KiteConnect instance
        max_retries: Maximum API retry attempts
        retry_delay: Delay between retries in seconds
    """

    # Exchange mapping
    EXCHANGE_MAP = {
        "NSE": "NSE",
        "BSE": "BSE",
    }

    # Order type mapping (universal -> Kite)
    ORDER_TYPE_MAP = {
        OrderType.MARKET: "MARKET",
        OrderType.LIMIT: "LIMIT",
        OrderType.STOP_LOSS: "SL",
        OrderType.STOP_LOSS_MARKET: "SL-M",
    }

    # Product type mapping (universal -> Kite)
    PRODUCT_MAP = {
        "MIS": "MIS",  # Margin Intraday Square-off
        "CNC": "CNC",  # Cash and Carry (delivery)
        "NRML": "NRML",  # Normal (F&O)
    }

    # Reverse mappings for response parsing
    REVERSE_ORDER_STATUS_MAP = {
        "OPEN": OrderStatus.PLACED,
        "COMPLETE": OrderStatus.FILLED,
        "CANCELLED": OrderStatus.CANCELLED,
        "REJECTED": OrderStatus.REJECTED,
        "PENDING": OrderStatus.PENDING,
        "TRIGGER PENDING": OrderStatus.PLACED,
    }

    def __init__(
        self,
        api_key: str,
        api_secret: Optional[str] = None,
        access_token: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize Kite Connect broker.

        Args:
            api_key: Kite Connect API key
            api_secret: Kite Connect API secret (optional, for login flow)
            access_token: Kite Connect access token (required for trading)
            max_retries: Maximum API retry attempts
            retry_delay: Delay between retries in seconds

        Raises:
            ImportError: If kiteconnect library not installed
            ValueError: If access_token not provided
        """
        if not KITE_AVAILABLE or KiteConnect is None:
            raise ImportError(
                "kiteconnect library not installed. "
                "Install with: pip install kiteconnect==5.0.1"
            )

        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Initialize KiteConnect
        self.kite = KiteConnect(api_key=api_key)

        # Set access token if provided
        if access_token:
            self.kite.set_access_token(access_token)
            logger.info("Kite Connect broker initialized with access token")
        else:
            logger.warning(
                "Kite Connect initialized without access token. "
                "Call set_access_token() before trading."
            )

    def set_access_token(self, access_token: str) -> None:
        """
        Set access token for API authentication.

        Args:
            access_token: Kite Connect access token
        """
        self.access_token = access_token
        self.kite.set_access_token(access_token)
        logger.info("Access token set for Kite Connect")

    def authenticate(self) -> bool:
        """
        Authenticate with Kite Connect.

        Tests authentication by fetching user profile.

        Returns:
            True if authentication successful, False otherwise
        """
        try:
            if not self.access_token:
                logger.error("No access token provided")
                return False

            # Test authentication by fetching profile
            profile = self._api_call(self.kite.profile)

            if profile:
                logger.info(f"Kite Connect authentication successful: {profile.get('user_name')}")
                return True
            else:
                logger.error("Kite Connect authentication failed")
                return False

        except Exception as e:
            logger.error(f"Kite Connect authentication error: {e}")
            return False

    def place_order(self, order: Order) -> str:
        """
        Place order with Kite Connect.

        Args:
            order: Order object with all details

        Returns:
            Broker-assigned order ID

        Raises:
            ValueError: If order parameters invalid
            RuntimeError: If order placement fails
        """
        try:
            # Build Kite order request
            kite_order = {
                "tradingsymbol": order.symbol,
                "exchange": self.EXCHANGE_MAP[order.exchange],
                "transaction_type": order.side.value,
                "quantity": order.quantity,
                "order_type": self.ORDER_TYPE_MAP[order.order_type],
                "product": self.PRODUCT_MAP.get(order.product, "MIS"),
                "validity": order.validity,
            }

            # Add price for limit orders
            if order.order_type == OrderType.LIMIT:
                if order.price is None:
                    raise ValueError("Limit order requires price")
                kite_order["price"] = order.price

            # Add trigger price for stop loss orders
            if order.order_type in [OrderType.STOP_LOSS, OrderType.STOP_LOSS_MARKET]:
                if order.trigger_price is None:
                    raise ValueError("Stop loss order requires trigger_price")
                kite_order["trigger_price"] = order.trigger_price
                if order.order_type == OrderType.STOP_LOSS and order.price:
                    kite_order["price"] = order.price

            # Add tag if provided
            if order.tag:
                kite_order["tag"] = order.tag

            # Place order with retry logic
            logger.info(
                f"Placing order: {order.symbol} {order.side.value} "
                f"{order.quantity} @ {order.order_type.value}"
            )

            order_id = self._api_call(self.kite.place_order, variety="regular", **kite_order)

            if order_id:
                order.order_id = str(order_id)
                order.status = OrderStatus.PLACED
                order.timestamp = datetime.now()
                logger.info(f"Order placed successfully: {order_id}")
                return str(order_id)
            else:
                raise RuntimeError("Order placement failed: No order ID returned")

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
            order_id: Kite order ID
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
            modify_params = {}

            if quantity is not None:
                modify_params["quantity"] = quantity
            if price is not None:
                modify_params["price"] = price
            if trigger_price is not None:
                modify_params["trigger_price"] = trigger_price

            logger.info(f"Modifying order {order_id}")
            result = self._api_call(
                self.kite.modify_order,
                variety="regular",
                order_id=order_id,
                **modify_params
            )

            if result:
                logger.info(f"Order {order_id} modified successfully")
                return True
            else:
                raise RuntimeError("Order modification failed")

        except Exception as e:
            logger.error(f"Error modifying order {order_id}: {e}")
            raise

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order.

        Args:
            order_id: Kite order ID

        Returns:
            True if cancellation successful

        Raises:
            ValueError: If order_id not found
            RuntimeError: If cancellation fails
        """
        try:
            logger.info(f"Cancelling order {order_id}")
            result = self._api_call(
                self.kite.cancel_order,
                variety="regular",
                order_id=order_id
            )

            if result:
                logger.info(f"Order {order_id} cancelled successfully")
                return True
            else:
                raise RuntimeError("Order cancellation failed")

        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            raise

    def get_order_status(self, order_id: str) -> Order:
        """
        Get current status of an order.

        Args:
            order_id: Kite order ID

        Returns:
            Order object with current status

        Raises:
            ValueError: If order ID not found
        """
        try:
            orders = self._api_call(self.kite.orders)

            for order_data in orders:
                if str(order_data.get("order_id")) == str(order_id):
                    return self._parse_order_response(order_data)

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
            orders_data = self._api_call(self.kite.orders)

            if not orders_data:
                logger.warning("No orders returned from API")
                return []

            orders = []
            for order_data in orders_data:
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
            positions_data = self._api_call(self.kite.positions)

            if not positions_data:
                logger.warning("No positions or API error")
                return []

            positions = []
            # Kite returns 'net' and 'day' positions
            net_positions = positions_data.get("net", [])

            for pos_data in net_positions:
                if pos_data.get("quantity", 0) == 0:
                    continue  # Skip closed positions

                position = Position(
                    symbol=pos_data.get("tradingsymbol", ""),
                    quantity=pos_data.get("quantity", 0),
                    average_price=pos_data.get("average_price", 0.0),
                    last_price=pos_data.get("last_price", 0.0),
                    pnl=pos_data.get("pnl", 0.0),
                    day_pnl=pos_data.get("day_pnl", 0.0),
                    exchange=pos_data.get("exchange", "NSE"),
                    product=self._parse_product(pos_data.get("product", "MIS")),
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
            margins = self._api_call(self.kite.margins)

            if not margins:
                raise ConnectionError("Failed to fetch account balance")

            # Kite returns margins for different segments
            equity_margins = margins.get("equity", {})

            balance = AccountBalance(
                available_cash=equity_margins.get("available", {}).get("cash", 0.0),
                used_margin=equity_margins.get("utilised", {}).get("debits", 0.0),
                total_balance=equity_margins.get("net", 0.0),
                collateral=equity_margins.get("available", {}).get("collateral", 0.0),
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
            instrument = f"{exchange}:{symbol}"
            ltp_data = self._api_call(self.kite.ltp, [instrument])

            if ltp_data and instrument in ltp_data:
                ltp = ltp_data[instrument].get("last_price", 0.0)
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
            instrument = f"{exchange}:{symbol}"
            quote_data = self._api_call(self.kite.quote, [instrument])

            if quote_data and instrument in quote_data:
                data = quote_data[instrument]
                ohlc = data.get("ohlc", {})
                depth = data.get("depth", {})
                buy = depth.get("buy", [{}])[0] if depth.get("buy") else {}
                sell = depth.get("sell", [{}])[0] if depth.get("sell") else {}

                return {
                    "symbol": symbol,
                    "ltp": data.get("last_price", 0.0),
                    "open": ohlc.get("open", 0.0),
                    "high": ohlc.get("high", 0.0),
                    "low": ohlc.get("low", 0.0),
                    "close": ohlc.get("close", 0.0),
                    "volume": data.get("volume", 0),
                    "bid": buy.get("price", 0.0),
                    "ask": sell.get("price", 0.0),
                    "bid_qty": buy.get("quantity", 0),
                    "ask_qty": sell.get("quantity", 0),
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

    def _parse_order_response(self, order_data: Dict) -> Order:
        """
        Parse Kite order response to universal Order object.

        Args:
            order_data: Order data from Kite API

        Returns:
            Order object
        """
        # Parse order side
        side_str = order_data.get("transaction_type", "BUY")
        side = OrderSide.BUY if side_str == "BUY" else OrderSide.SELL

        # Parse order type
        order_type_str = order_data.get("order_type", "MARKET")
        order_type = OrderType.MARKET
        for ot, kite_str in self.ORDER_TYPE_MAP.items():
            if kite_str == order_type_str:
                order_type = ot
                break

        # Parse status
        status_str = order_data.get("status", "PENDING")
        status = self.REVERSE_ORDER_STATUS_MAP.get(status_str, OrderStatus.PENDING)

        # Parse product
        product_str = order_data.get("product", "MIS")
        product = self._parse_product(product_str)

        return Order(
            symbol=order_data.get("tradingsymbol", ""),
            side=side,
            quantity=order_data.get("quantity", 0),
            order_type=order_type,
            price=order_data.get("price"),
            trigger_price=order_data.get("trigger_price"),
            product=product,
            validity=order_data.get("validity", "DAY"),
            order_id=str(order_data.get("order_id", "")),
            status=status,
            filled_quantity=order_data.get("filled_quantity", 0),
            average_price=order_data.get("average_price"),
            exchange=order_data.get("exchange", "NSE"),
            tag=order_data.get("tag"),
        )

    def _parse_product(self, product_type: str) -> str:
        """Parse Kite product type to universal product."""
        for prod, kite_str in self.PRODUCT_MAP.items():
            if kite_str == product_type:
                return prod
        return "MIS"
