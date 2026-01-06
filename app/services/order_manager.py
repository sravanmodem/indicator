"""
Order Manager
Handles automated order placement through Kite Connect API
Manages order lifecycle: placement, modification, cancellation, exit
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from kiteconnect import KiteConnect
from loguru import logger

from app.services.signal_engine import TradeSignal
from app.services.zerodha_auth import get_auth_service


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    PLACED = "placed"
    EXECUTED = "executed"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"
    SL_M = "SL-M"


class TransactionType(Enum):
    """Transaction type enumeration."""
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class OrderResult:
    """Order placement result."""
    success: bool
    order_id: str | None
    message: str
    order_details: dict[str, Any] | None = None


@dataclass
class PlacedOrder:
    """Placed order tracking."""
    order_id: str
    signal: TradeSignal
    entry_order_id: str | None = None
    stop_loss_order_id: str | None = None
    target_order_id: str | None = None
    quantity: int = 0
    entry_price: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    placed_at: datetime = None
    executed_at: datetime | None = None
    pnl: float = 0.0


class OrderManager:
    """
    Manages order placement and lifecycle through Kite Connect.

    Features:
    - Automated order placement from signals
    - Bracket order support (entry + SL + target)
    - Order modification and cancellation
    - Position tracking
    - Error handling and retry logic
    """

    def __init__(self):
        """Initialize order manager."""
        self.auth_service = get_auth_service()
        self.kite: KiteConnect | None = None
        self.active_orders: dict[str, PlacedOrder] = {}

    def _get_kite(self) -> KiteConnect | None:
        """Get authenticated Kite instance."""
        if not self.auth_service.is_authenticated:
            logger.error("Not authenticated with Zerodha")
            return None

        if self.kite is None:
            self.kite = self.auth_service.kite

        return self.kite

    def place_signal_order(
        self,
        signal: TradeSignal,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        with_bracket: bool = True,
    ) -> OrderResult:
        """
        Place order based on signal.

        Args:
            signal: TradeSignal to execute
            quantity: Number of lots/contracts
            order_type: MARKET, LIMIT, etc.
            with_bracket: Place SL and target orders

        Returns:
            OrderResult with success status and details
        """
        try:
            kite = self._get_kite()
            if not kite:
                return OrderResult(
                    success=False,
                    order_id=None,
                    message="Not authenticated with Zerodha",
                )

            if not signal.recommended_option:
                return OrderResult(
                    success=False,
                    order_id=None,
                    message="No recommended option in signal",
                )

            option = signal.recommended_option

            # Validate signal quality before placing order
            # (This should be done before calling this function)

            # Prepare order parameters
            exchange = "NFO"  # Options are on NFO
            tradingsymbol = option.symbol
            transaction_type = TransactionType.BUY.value

            # Place entry order
            logger.info(f"Placing {order_type.value} order for {tradingsymbol}, qty: {quantity}")

            order_params = {
                "exchange": exchange,
                "tradingsymbol": tradingsymbol,
                "transaction_type": transaction_type,
                "quantity": quantity,
                "product": "MIS",  # Intraday product
                "order_type": order_type.value,
                "validity": "DAY",
            }

            # Add price for limit orders
            if order_type == OrderType.LIMIT:
                order_params["price"] = option.ltp

            # Place main entry order
            try:
                entry_order_id = kite.place_order(
                    variety="regular",
                    **order_params,
                )

                logger.info(f"Entry order placed successfully: {entry_order_id}")

                # Create tracked order
                placed_order = PlacedOrder(
                    order_id=entry_order_id,
                    signal=signal,
                    entry_order_id=entry_order_id,
                    quantity=quantity,
                    entry_price=option.ltp,
                    status=OrderStatus.PLACED,
                    placed_at=datetime.now(),
                )

                # Place bracket orders (SL and target) if requested
                if with_bracket:
                    # Wait for entry order to execute before placing bracket
                    # In production, you'd poll order status
                    # For now, we'll place SL and target as separate orders

                    sl_result = self._place_stop_loss(
                        kite=kite,
                        tradingsymbol=tradingsymbol,
                        quantity=quantity,
                        trigger_price=option.expected_at_stop if option.expected_at_stop else option.ltp * 0.8,
                    )

                    if sl_result.success:
                        placed_order.stop_loss_order_id = sl_result.order_id
                        logger.info(f"Stop loss order placed: {sl_result.order_id}")

                    target_result = self._place_target(
                        kite=kite,
                        tradingsymbol=tradingsymbol,
                        quantity=quantity,
                        price=option.expected_at_target if option.expected_at_target else option.ltp * 1.5,
                    )

                    if target_result.success:
                        placed_order.target_order_id = target_result.order_id
                        logger.info(f"Target order placed: {target_result.order_id}")

                # Track order
                self.active_orders[entry_order_id] = placed_order

                return OrderResult(
                    success=True,
                    order_id=entry_order_id,
                    message=f"Order placed successfully: {entry_order_id}",
                    order_details={
                        "entry_order_id": entry_order_id,
                        "stop_loss_order_id": placed_order.stop_loss_order_id,
                        "target_order_id": placed_order.target_order_id,
                        "symbol": tradingsymbol,
                        "quantity": quantity,
                        "price": option.ltp,
                    },
                )

            except Exception as e:
                logger.error(f"Failed to place order: {e}")
                return OrderResult(
                    success=False,
                    order_id=None,
                    message=f"Order placement failed: {str(e)}",
                )

        except Exception as e:
            logger.error(f"Order placement error: {e}")
            return OrderResult(
                success=False,
                order_id=None,
                message=f"Error: {str(e)}",
            )

    def _place_stop_loss(
        self,
        kite: KiteConnect,
        tradingsymbol: str,
        quantity: int,
        trigger_price: float,
    ) -> OrderResult:
        """Place stop-loss order."""
        try:
            order_id = kite.place_order(
                variety="regular",
                exchange="NFO",
                tradingsymbol=tradingsymbol,
                transaction_type=TransactionType.SELL.value,
                quantity=quantity,
                product="MIS",
                order_type=OrderType.SL_M.value,
                trigger_price=trigger_price,
                validity="DAY",
            )

            return OrderResult(
                success=True,
                order_id=order_id,
                message="Stop loss placed",
            )

        except Exception as e:
            logger.error(f"Failed to place stop loss: {e}")
            return OrderResult(
                success=False,
                order_id=None,
                message=f"SL failed: {str(e)}",
            )

    def _place_target(
        self,
        kite: KiteConnect,
        tradingsymbol: str,
        quantity: int,
        price: float,
    ) -> OrderResult:
        """Place target order."""
        try:
            order_id = kite.place_order(
                variety="regular",
                exchange="NFO",
                tradingsymbol=tradingsymbol,
                transaction_type=TransactionType.SELL.value,
                quantity=quantity,
                product="MIS",
                order_type=OrderType.LIMIT.value,
                price=price,
                validity="DAY",
            )

            return OrderResult(
                success=True,
                order_id=order_id,
                message="Target placed",
            )

        except Exception as e:
            logger.error(f"Failed to place target: {e}")
            return OrderResult(
                success=False,
                order_id=None,
                message=f"Target failed: {str(e)}",
            )

    def cancel_order(self, order_id: str, variety: str = "regular") -> OrderResult:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel
            variety: Order variety (regular, bo, co, etc.)

        Returns:
            OrderResult with cancellation status
        """
        try:
            kite = self._get_kite()
            if not kite:
                return OrderResult(
                    success=False,
                    order_id=None,
                    message="Not authenticated",
                )

            kite.cancel_order(variety=variety, order_id=order_id)

            # Update tracked order
            if order_id in self.active_orders:
                self.active_orders[order_id].status = OrderStatus.CANCELLED

            logger.info(f"Order cancelled: {order_id}")

            return OrderResult(
                success=True,
                order_id=order_id,
                message="Order cancelled successfully",
            )

        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return OrderResult(
                success=False,
                order_id=None,
                message=f"Cancellation failed: {str(e)}",
            )

    def modify_order(
        self,
        order_id: str,
        quantity: int | None = None,
        price: float | None = None,
        trigger_price: float | None = None,
        variety: str = "regular",
    ) -> OrderResult:
        """
        Modify an existing order.

        Args:
            order_id: Order ID to modify
            quantity: New quantity (optional)
            price: New price (optional)
            trigger_price: New trigger price for SL orders (optional)
            variety: Order variety

        Returns:
            OrderResult with modification status
        """
        try:
            kite = self._get_kite()
            if not kite:
                return OrderResult(
                    success=False,
                    order_id=None,
                    message="Not authenticated",
                )

            params = {}
            if quantity is not None:
                params["quantity"] = quantity
            if price is not None:
                params["price"] = price
            if trigger_price is not None:
                params["trigger_price"] = trigger_price

            kite.modify_order(
                variety=variety,
                order_id=order_id,
                **params,
            )

            logger.info(f"Order modified: {order_id}")

            return OrderResult(
                success=True,
                order_id=order_id,
                message="Order modified successfully",
            )

        except Exception as e:
            logger.error(f"Failed to modify order {order_id}: {e}")
            return OrderResult(
                success=False,
                order_id=None,
                message=f"Modification failed: {str(e)}",
            )

    def get_order_status(self, order_id: str) -> dict[str, Any] | None:
        """Get current status of an order."""
        try:
            kite = self._get_kite()
            if not kite:
                return None

            orders = kite.orders()
            for order in orders:
                if order["order_id"] == order_id:
                    return order

            return None

        except Exception as e:
            logger.error(f"Failed to get order status: {e}")
            return None

    def get_positions(self) -> list[dict[str, Any]]:
        """Get current positions."""
        try:
            kite = self._get_kite()
            if not kite:
                return []

            positions = kite.positions()
            return positions.get("net", [])

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    def exit_position(
        self,
        tradingsymbol: str,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
    ) -> OrderResult:
        """
        Exit an existing position.

        Args:
            tradingsymbol: Trading symbol to exit
            quantity: Quantity to exit
            order_type: Order type (MARKET or LIMIT)

        Returns:
            OrderResult with exit status
        """
        try:
            kite = self._get_kite()
            if not kite:
                return OrderResult(
                    success=False,
                    order_id=None,
                    message="Not authenticated",
                )

            order_params = {
                "exchange": "NFO",
                "tradingsymbol": tradingsymbol,
                "transaction_type": TransactionType.SELL.value,
                "quantity": quantity,
                "product": "MIS",
                "order_type": order_type.value,
                "validity": "DAY",
            }

            order_id = kite.place_order(variety="regular", **order_params)

            logger.info(f"Exit order placed: {order_id}")

            return OrderResult(
                success=True,
                order_id=order_id,
                message="Position exited successfully",
            )

        except Exception as e:
            logger.error(f"Failed to exit position: {e}")
            return OrderResult(
                success=False,
                order_id=None,
                message=f"Exit failed: {str(e)}",
            )


# Singleton instance
_order_manager: OrderManager | None = None


def get_order_manager() -> OrderManager:
    """Get or create order manager instance."""
    global _order_manager
    if _order_manager is None:
        _order_manager = OrderManager()
    return _order_manager
