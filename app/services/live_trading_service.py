"""
Live Trading Service
Real order execution via Zerodha Kite API

Features:
- LIMIT orders with 0.5% slippage buffer
- Dynamic position sizing based on available CAPITAL (not margin)
- Real position tracking from kite.positions()
- Auto-execution when enabled during market hours
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from loguru import logger

from app.core.config import get_settings, NIFTY_LOT_SIZE, BANKNIFTY_LOT_SIZE, SENSEX_LOT_SIZE


class LiveOrderStatus(Enum):
    """Order status from Zerodha."""
    PENDING = "PENDING"
    OPEN = "OPEN"
    COMPLETE = "COMPLETE"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    TRIGGER_PENDING = "TRIGGER PENDING"


@dataclass
class LiveMargin:
    """Margin details from Zerodha."""
    available_cash: float = 0.0
    available_margin: float = 0.0
    used_margin: float = 0.0
    total_margin: float = 0.0
    collateral: float = 0.0


@dataclass
class LivePosition:
    """Real position from Zerodha."""
    tradingsymbol: str
    exchange: str
    instrument_token: int
    quantity: int
    average_price: float
    last_price: float
    pnl: float
    pnl_percent: float
    product: str
    overnight_quantity: int = 0
    multiplier: int = 1
    buy_quantity: int = 0
    sell_quantity: int = 0
    buy_price: float = 0.0
    sell_price: float = 0.0


@dataclass
class LiveOrder:
    """Real order from Zerodha."""
    order_id: str
    exchange_order_id: str
    tradingsymbol: str
    exchange: str
    transaction_type: str  # BUY or SELL
    quantity: int
    price: float
    trigger_price: float
    status: LiveOrderStatus
    filled_quantity: int
    average_price: float
    order_timestamp: datetime
    variety: str = "regular"
    product: str = "NRML"
    order_type: str = "LIMIT"
    status_message: str = ""


@dataclass
class OrderResult:
    """Result of order placement."""
    success: bool
    order_id: str = ""
    error: str = ""
    tradingsymbol: str = ""
    quantity: int = 0
    price: float = 0.0
    transaction_type: str = ""
    is_paper: bool = False


class LiveTradingService:
    """
    Live Trading Service for real order execution via Zerodha Kite.

    Features:
    - LIMIT orders with configurable slippage buffer
    - Dynamic position sizing based on available CAPITAL (cash), not leveraged margin
    - Real-time position and order tracking
    - Auto-execution when enabled during market hours
    """

    SLIPPAGE_BUFFER = 0.5  # 0.5% slippage for limit orders
    MAX_LOTS_PER_ORDER = 25  # Exchange limit

    LOT_SIZES = {
        "NIFTY": NIFTY_LOT_SIZE,
        "BANKNIFTY": BANKNIFTY_LOT_SIZE,
        "SENSEX": SENSEX_LOT_SIZE,
    }

    def __init__(self):
        self.settings = get_settings()
        self._is_live_mode = False
        self._kite: Optional[Any] = None

    def set_kite_client(self, kite: Any):
        """Set the Kite client from auth service."""
        self._kite = kite
        logger.info("Live trading service: Kite client connected")

    @property
    def kite(self) -> Any:
        """Get Kite client, lazy load from auth service if needed."""
        if self._kite is None:
            try:
                from app.services.zerodha_auth import get_auth_service
                auth_service = get_auth_service()
                if auth_service.is_authenticated:
                    self._kite = auth_service.kite
            except Exception as e:
                logger.error(f"Failed to get Kite client: {e}")
        return self._kite

    @property
    def is_authenticated(self) -> bool:
        """Check if Zerodha is authenticated."""
        return self.kite is not None

    def enable_live_mode(self, enabled: bool = True):
        """Toggle live trading mode."""
        if enabled and not self.is_authenticated:
            logger.error("Cannot enable live mode - Zerodha not authenticated")
            return False

        self._is_live_mode = enabled
        if enabled:
            logger.warning("LIVE TRADING MODE ENABLED - Real money at risk!")
        else:
            logger.info("Live trading mode disabled - Paper trading active")
        return True

    @property
    def is_live_mode(self) -> bool:
        """Check if live trading is enabled."""
        return self._is_live_mode and self.is_authenticated

    async def get_available_margin(self) -> LiveMargin:
        """
        Get available margin from Zerodha.
        Returns all margin components for position sizing.
        """
        if not self.is_authenticated:
            logger.warning("Cannot fetch margin - not authenticated")
            return LiveMargin()

        try:
            margins = await asyncio.to_thread(self.kite.margins, "equity")

            available = margins.get("available", {})
            utilised = margins.get("utilised", {})

            return LiveMargin(
                available_cash=float(available.get("cash", 0)),
                available_margin=float(available.get("live_balance", 0)),
                used_margin=float(utilised.get("exposure", 0)) + float(utilised.get("span", 0)),
                total_margin=float(margins.get("net", 0)),
                collateral=float(available.get("collateral", 0)),
            )
        except Exception as e:
            logger.error(f"Failed to fetch margins: {e}")
            return LiveMargin()

    async def calculate_position_size(
        self,
        option_ltp: float,
        index: str,
        capital_percent: float = 100,  # Use 100% of available capital by default
    ) -> tuple[int, int, float]:
        """
        Calculate position size based on available CAPITAL (cash only, not including profits).

        Args:
            option_ltp: Current option price
            index: NIFTY, BANKNIFTY, or SENSEX
            capital_percent: Percentage of capital to use (default 100%)

        Returns:
            (lots, quantity, capital_used)
        """
        margin = await self.get_available_margin()

        # Use available_cash instead of available_margin
        # This ensures position sizing is based on actual capital, not leveraged margin
        if margin.available_cash <= 0:
            logger.warning("No available capital for trading")
            return 0, 0, 0.0

        lot_size = self.LOT_SIZES.get(index, 25)
        available = margin.available_cash * (capital_percent / 100)

        # For options, cost = premium * quantity
        cost_per_lot = option_ltp * lot_size

        if cost_per_lot <= 0:
            logger.warning(f"Invalid option LTP: {option_ltp}")
            return 0, 0, 0.0

        max_lots = int(available / cost_per_lot)

        # Cap at exchange limit
        max_lots = min(max_lots, self.MAX_LOTS_PER_ORDER)

        if max_lots <= 0:
            logger.warning(f"Insufficient capital: available={available:.0f}, cost_per_lot={cost_per_lot:.0f}")
            return 0, 0, 0.0

        quantity = max_lots * lot_size
        capital_used = quantity * option_ltp

        logger.info(
            f"Position size calculated: {max_lots} lots ({quantity} qty) "
            f"@ Rs.{option_ltp:.2f} = Rs.{capital_used:,.0f} "
            f"(using {capital_percent}% of Rs.{margin.available_cash:,.0f} capital)"
        )

        return max_lots, quantity, capital_used

    async def place_entry_order(
        self,
        tradingsymbol: str,
        exchange: str,  # NFO or BFO
        quantity: int,
        price: float,
        transaction_type: str = "BUY",
    ) -> OrderResult:
        """
        Place entry order with LIMIT + slippage buffer.

        Args:
            tradingsymbol: Option symbol (e.g., NIFTY24JAN25500CE)
            exchange: NFO for NIFTY/BANKNIFTY, BFO for SENSEX
            quantity: Number of shares (lots * lot_size)
            price: Current LTP (slippage will be added)
            transaction_type: BUY or SELL

        Returns:
            OrderResult with order_id or error
        """
        if not self.is_live_mode:
            return OrderResult(
                success=False,
                error="Live trading not enabled",
                is_paper=True,
            )

        try:
            # Add slippage buffer for limit orders
            if transaction_type == "BUY":
                limit_price = price * (1 + self.SLIPPAGE_BUFFER / 100)
            else:
                limit_price = price * (1 - self.SLIPPAGE_BUFFER / 100)

            # Round to tick size (0.05 for options)
            limit_price = round(limit_price * 20) / 20

            logger.info(
                f"Placing {transaction_type} order: {tradingsymbol} "
                f"x{quantity} @ Rs.{limit_price:.2f} (LTP: Rs.{price:.2f}, "
                f"slippage: {self.SLIPPAGE_BUFFER}%)"
            )

            order_id = await asyncio.to_thread(
                self.kite.place_order,
                variety="regular",
                tradingsymbol=tradingsymbol,
                exchange=exchange,
                transaction_type=transaction_type,
                quantity=quantity,
                order_type="LIMIT",
                price=limit_price,
                product="NRML",  # Normal for options
                validity="DAY",
            )

            logger.info(f"Order placed successfully: {order_id}")

            return OrderResult(
                success=True,
                order_id=str(order_id),
                tradingsymbol=tradingsymbol,
                quantity=quantity,
                price=limit_price,
                transaction_type=transaction_type,
            )

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Order placement failed: {error_msg}")
            return OrderResult(
                success=False,
                error=error_msg,
                tradingsymbol=tradingsymbol,
            )

    async def place_exit_order(
        self,
        tradingsymbol: str,
        exchange: str,
        quantity: int,
        price: float,
    ) -> OrderResult:
        """
        Place exit (SELL) order.
        Uses LIMIT with negative slippage to ensure fill.
        """
        return await self.place_entry_order(
            tradingsymbol=tradingsymbol,
            exchange=exchange,
            quantity=quantity,
            price=price,
            transaction_type="SELL",
        )

    async def get_positions(self) -> list[LivePosition]:
        """Fetch all current positions from Zerodha."""
        if not self.is_authenticated:
            return []

        try:
            positions = await asyncio.to_thread(self.kite.positions)

            net_positions = positions.get("net", [])

            result = []
            for pos in net_positions:
                qty = pos.get("quantity", 0)
                if qty == 0:  # Skip closed positions
                    continue

                avg_price = float(pos.get("average_price", 0))
                last_price = float(pos.get("last_price", 0))
                pnl = float(pos.get("pnl", 0))
                pnl_percent = ((last_price - avg_price) / avg_price * 100) if avg_price > 0 else 0

                result.append(LivePosition(
                    tradingsymbol=pos.get("tradingsymbol", ""),
                    exchange=pos.get("exchange", ""),
                    instrument_token=pos.get("instrument_token", 0),
                    quantity=qty,
                    average_price=avg_price,
                    last_price=last_price,
                    pnl=pnl,
                    pnl_percent=pnl_percent,
                    product=pos.get("product", ""),
                    overnight_quantity=pos.get("overnight_quantity", 0),
                    multiplier=pos.get("multiplier", 1),
                    buy_quantity=pos.get("buy_quantity", 0),
                    sell_quantity=pos.get("sell_quantity", 0),
                    buy_price=float(pos.get("buy_price", 0)),
                    sell_price=float(pos.get("sell_price", 0)),
                ))

            return result

        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            return []

    async def get_orders(self, only_open: bool = False) -> list[LiveOrder]:
        """Fetch all orders for today."""
        if not self.is_authenticated:
            return []

        try:
            orders = await asyncio.to_thread(self.kite.orders)

            result = []
            for order in orders:
                status_str = order.get("status", "PENDING").upper()
                try:
                    status = LiveOrderStatus(status_str)
                except ValueError:
                    status = LiveOrderStatus.PENDING

                if only_open and status not in [LiveOrderStatus.PENDING, LiveOrderStatus.OPEN, LiveOrderStatus.TRIGGER_PENDING]:
                    continue

                result.append(LiveOrder(
                    order_id=order.get("order_id", ""),
                    exchange_order_id=order.get("exchange_order_id", "") or "",
                    tradingsymbol=order.get("tradingsymbol", ""),
                    exchange=order.get("exchange", ""),
                    transaction_type=order.get("transaction_type", ""),
                    quantity=order.get("quantity", 0),
                    price=float(order.get("price", 0)),
                    trigger_price=float(order.get("trigger_price", 0)),
                    status=status,
                    filled_quantity=order.get("filled_quantity", 0),
                    average_price=float(order.get("average_price", 0)),
                    order_timestamp=order.get("order_timestamp", datetime.now()),
                    variety=order.get("variety", "regular"),
                    product=order.get("product", "NRML"),
                    order_type=order.get("order_type", "LIMIT"),
                    status_message=order.get("status_message", "") or "",
                ))

            return result

        except Exception as e:
            logger.error(f"Failed to fetch orders: {e}")
            return []

    async def get_order_status(self, order_id: str) -> Optional[LiveOrder]:
        """Get status of a specific order."""
        orders = await self.get_orders()
        for order in orders:
            if order.order_id == order_id:
                return order
        return None

    async def cancel_order(self, order_id: str) -> OrderResult:
        """Cancel an open order."""
        if not self.is_authenticated:
            return OrderResult(success=False, error="Not authenticated")

        try:
            await asyncio.to_thread(
                self.kite.cancel_order,
                variety="regular",
                order_id=order_id,
            )
            logger.info(f"Order cancelled: {order_id}")
            return OrderResult(success=True, order_id=order_id)
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return OrderResult(success=False, error=str(e), order_id=order_id)

    async def modify_order(
        self,
        order_id: str,
        quantity: Optional[int] = None,
        price: Optional[float] = None,
    ) -> OrderResult:
        """Modify an open order."""
        if not self.is_authenticated:
            return OrderResult(success=False, error="Not authenticated")

        try:
            params = {"variety": "regular", "order_id": order_id}
            if quantity is not None:
                params["quantity"] = quantity
            if price is not None:
                params["price"] = round(price * 20) / 20  # Round to tick

            await asyncio.to_thread(self.kite.modify_order, **params)
            logger.info(f"Order modified: {order_id}")
            return OrderResult(success=True, order_id=order_id)
        except Exception as e:
            logger.error(f"Failed to modify order {order_id}: {e}")
            return OrderResult(success=False, error=str(e), order_id=order_id)

    def get_exchange_for_index(self, index: str) -> str:
        """Get exchange code for index."""
        if index == "SENSEX":
            return "BFO"  # BSE F&O
        return "NFO"  # NSE F&O


# Singleton instance
_live_service: Optional[LiveTradingService] = None


def get_live_trading_service() -> LiveTradingService:
    """Get or create singleton live trading service."""
    global _live_service
    if _live_service is None:
        _live_service = LiveTradingService()
    return _live_service
