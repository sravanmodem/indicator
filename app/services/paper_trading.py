"""
Paper Trading Service
Automatic trading based on signals with capital management
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import Any
import json
from pathlib import Path

from loguru import logger

from app.core.config import get_settings
from app.services.signal_engine import TradingStyle, SignalType


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    EXECUTED = "executed"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderType(Enum):
    """Order type enumeration."""
    BUY = "buy"
    SELL = "sell"


class PositionStatus(Enum):
    """Position status enumeration."""
    OPEN = "open"
    CLOSED = "closed"
    PARTIAL_CLOSED = "partial_closed"


@dataclass
class PaperOrder:
    """Paper trading order."""
    order_id: str
    timestamp: datetime
    index: str  # NIFTY, BANKNIFTY, SENSEX
    symbol: str  # e.g., NIFTY26200CE
    strike: float
    option_type: str  # CE or PE
    order_type: OrderType
    quantity: int  # Total quantity
    lots: int
    price: float
    status: OrderStatus = OrderStatus.PENDING
    executed_quantity: int = 0
    executed_price: float = 0.0
    split_orders: list[dict] = field(default_factory=list)
    reason: str = ""
    signal_confidence: float = 0.0


@dataclass
class PaperPosition:
    """Paper trading position."""
    position_id: str
    index: str
    symbol: str
    strike: float
    option_type: str
    entry_price: float
    quantity: int
    lots: int
    entry_time: datetime
    current_price: float = 0.0
    pnl: float = 0.0
    pnl_percent: float = 0.0
    status: PositionStatus = PositionStatus.OPEN
    exit_price: float = 0.0
    exit_time: datetime | None = None
    exit_reason: str = ""
    stop_loss: float = 0.0
    target: float = 0.0
    # Price tracking for order history
    max_price: float = 0.0  # Highest price reached during trade
    min_price: float = 0.0  # Lowest price reached during trade
    max_price_time: datetime | None = None
    min_price_time: datetime | None = None


@dataclass
class OrderHistoryEntry:
    """Detailed order history entry with performance tracking."""
    order_id: str
    position_id: str
    timestamp: datetime
    index: str
    symbol: str
    strike: float
    option_type: str  # CE or PE
    direction: str  # BUY or SELL
    quantity: int
    lots: int
    # Price data
    entry_price: float
    exit_price: float = 0.0
    # Performance tracking
    max_price: float = 0.0
    min_price: float = 0.0
    max_price_time: datetime | None = None
    min_price_time: datetime | None = None
    # Calculated metrics
    pnl: float = 0.0
    pnl_percent: float = 0.0
    max_profit_percent: float = 0.0  # % from entry to max
    max_loss_percent: float = 0.0    # % from entry to min
    captured_move_percent: float = 0.0  # How much of max move was captured
    # Metadata
    signal_confidence: float = 0.0
    exit_reason: str = ""
    entry_time: datetime | None = None
    exit_time: datetime | None = None
    duration_minutes: int = 0


@dataclass
class DailyStats:
    """Daily trading statistics."""
    date: date
    starting_capital: float
    current_capital: float
    total_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    max_drawdown: float
    max_drawdown_percent: float
    daily_loss_percent: float
    is_trading_halted: bool = False
    halt_reason: str = ""


@dataclass
class ExpiryInfo:
    """Index expiry information."""
    index: str
    expiry_date: date
    days_to_expiry: int
    is_expiry_day: bool
    lot_size: int
    max_lots_per_order: int


class PaperTradingService:
    """
    Paper Trading Service for automatic signal-based trading.

    Features:
    - Auto-detect nearest expiry index
    - Priority order: NIFTY > SENSEX > BANKNIFTY (if same day expiry)
    - Capital management with daily loss limits
    - Order splitting for large orders
    - Position tracking and P&L calculation
    """

    # Configuration
    CAPITAL = 500000  # ₹5,00,000
    MAX_CAPITAL_USE = 1.0  # 100%
    MAX_DAILY_LOSS_PERCENT = 0.20  # 20%
    MAX_LOTS_PER_ORDER = 25

    # Lot sizes
    LOT_SIZES = {
        "NIFTY": 25,
        "BANKNIFTY": 15,
        "SENSEX": 10,
    }

    def __init__(self):
        self.settings = get_settings()
        self._data_fetcher = None  # Lazy loaded to avoid circular imports

        # State
        self.orders: list[PaperOrder] = []
        self.positions: list[PaperPosition] = []
        self.order_history: list[OrderHistoryEntry] = []  # Detailed order history
        self.daily_stats: DailyStats | None = None
        self.is_auto_trade = True  # Auto trade ON by default
        self._order_counter = 0
        self._position_counter = 0
        self._last_signal_id: str | None = None  # Track last signal to avoid duplicates

        # Expiry cache
        self._expiry_cache: dict[str, ExpiryInfo] = {}
        self._expiry_cache_time: datetime | None = None

        # Data file path
        self.data_file = Path(self.settings.data_dir) / "paper_trading.json"

        # Initialize daily stats
        self._initialize_daily_stats()

        # Load saved state
        self._load_state()

    @property
    def data_fetcher(self):
        """Lazy load data fetcher to avoid circular imports."""
        if self._data_fetcher is None:
            from app.services.data_fetcher import get_data_fetcher
            self._data_fetcher = get_data_fetcher()
        return self._data_fetcher

    def _initialize_daily_stats(self):
        """Initialize daily statistics."""
        today = date.today()

        if self.daily_stats is None or self.daily_stats.date != today:
            # Load previous capital or use default
            previous_capital = self.CAPITAL
            if self.daily_stats:
                previous_capital = self.daily_stats.current_capital

            self.daily_stats = DailyStats(
                date=today,
                starting_capital=previous_capital,
                current_capital=previous_capital,
                total_pnl=0,
                realized_pnl=0,
                unrealized_pnl=0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                max_drawdown=0,
                max_drawdown_percent=0,
                daily_loss_percent=0,
                is_trading_halted=False,
            )

    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        self._order_counter += 1
        return f"PO{datetime.now().strftime('%Y%m%d%H%M%S')}{self._order_counter:04d}"

    def _generate_position_id(self) -> str:
        """Generate unique position ID."""
        self._position_counter += 1
        return f"PP{datetime.now().strftime('%Y%m%d%H%M%S')}{self._position_counter:04d}"

    def get_next_expiry(self, index: str) -> ExpiryInfo:
        """
        Get next expiry date for an index.
        Uses cached data if available and less than 5 minutes old.

        Args:
            index: NIFTY, BANKNIFTY, or SENSEX

        Returns:
            ExpiryInfo with expiry details
        """
        index = index.upper()

        # Check cache first (valid for 5 minutes)
        if (
            self._expiry_cache_time
            and index in self._expiry_cache
            and (datetime.now() - self._expiry_cache_time).seconds < 300
        ):
            return self._expiry_cache[index]

        # Return fallback expiry info (will be updated async)
        today = datetime.now().date()

        # Fallback weekday mapping if Kite data not available
        fallback_days = {
            "NIFTY": 3,      # Thursday
            "SENSEX": 4,     # Friday
            "BANKNIFTY": 2,  # Wednesday
        }

        expiry_day = fallback_days.get(index, 3)
        current_weekday = today.weekday()
        days_ahead = expiry_day - current_weekday

        if days_ahead < 0:
            days_ahead += 7
        elif days_ahead == 0:
            now = datetime.now()
            if now.hour >= 15 and now.minute >= 30:
                days_ahead = 7

        expiry_date = today + timedelta(days=days_ahead)

        return ExpiryInfo(
            index=index,
            expiry_date=expiry_date,
            days_to_expiry=days_ahead,
            is_expiry_day=(days_ahead == 0),
            lot_size=self.LOT_SIZES.get(index, 25),
            max_lots_per_order=self.MAX_LOTS_PER_ORDER,
        )

    async def get_next_expiry_async(self, index: str) -> ExpiryInfo:
        """
        Get next expiry date for an index from Kite instruments (async).

        Args:
            index: NIFTY, BANKNIFTY, or SENSEX

        Returns:
            ExpiryInfo with expiry details
        """
        index = index.upper()

        try:
            # Fetch from Kite
            expiry_data = await self.data_fetcher.get_next_expiry(index)

            if "error" in expiry_data:
                logger.warning(f"Kite expiry fetch failed: {expiry_data['error']}, using fallback")
                return self.get_next_expiry(index)

            expiry_info = ExpiryInfo(
                index=index,
                expiry_date=expiry_data["expiry_date"],
                days_to_expiry=expiry_data["days_to_expiry"],
                is_expiry_day=expiry_data["is_expiry_day"],
                lot_size=self.LOT_SIZES.get(index, 25),
                max_lots_per_order=self.MAX_LOTS_PER_ORDER,
            )

            # Cache the result
            self._expiry_cache[index] = expiry_info
            self._expiry_cache_time = datetime.now()

            logger.info(f"Fetched expiry from Kite: {index} -> {expiry_data['expiry_date']} ({expiry_data['expiry_weekday']})")

            return expiry_info

        except Exception as e:
            logger.error(f"Error fetching expiry from Kite: {e}")
            return self.get_next_expiry(index)

    async def refresh_expiry_cache(self):
        """Refresh expiry cache for all indices from Kite."""
        for index in ["NIFTY", "BANKNIFTY", "SENSEX"]:
            await self.get_next_expiry_async(index)

    def get_trading_index(self) -> ExpiryInfo:
        """
        Get the index to trade based on nearest expiry (sync version).
        Uses cached data if available.

        Priority when same-day expiry: NIFTY > SENSEX > BANKNIFTY

        Returns:
            ExpiryInfo for the selected index
        """
        expiries = []
        for index in ["NIFTY", "SENSEX", "BANKNIFTY"]:
            expiry = self.get_next_expiry(index)
            expiries.append(expiry)

        # Sort by days to expiry
        expiries.sort(key=lambda x: x.days_to_expiry)

        # Get all indices with minimum days to expiry
        min_days = expiries[0].days_to_expiry
        same_day_expiries = [e for e in expiries if e.days_to_expiry == min_days]

        # If multiple same-day expiries, follow priority
        priority = {"NIFTY": 1, "SENSEX": 2, "BANKNIFTY": 3}
        same_day_expiries.sort(key=lambda x: priority.get(x.index, 99))

        selected = same_day_expiries[0]
        logger.info(f"Selected trading index: {selected.index} (Expiry: {selected.expiry_date}, Days: {selected.days_to_expiry})")

        return selected

    async def get_trading_index_async(self) -> ExpiryInfo:
        """
        Get the index to trade based on nearest expiry (async version).
        Fetches fresh data from Kite.

        Priority when same-day expiry: NIFTY > SENSEX > BANKNIFTY

        Returns:
            ExpiryInfo for the selected index
        """
        # First refresh the cache
        await self.refresh_expiry_cache()

        expiries = []
        for index in ["NIFTY", "SENSEX", "BANKNIFTY"]:
            expiry = self.get_next_expiry(index)  # Now uses cached data
            expiries.append(expiry)

        # Sort by days to expiry
        expiries.sort(key=lambda x: x.days_to_expiry)

        # Get all indices with minimum days to expiry
        min_days = expiries[0].days_to_expiry
        same_day_expiries = [e for e in expiries if e.days_to_expiry == min_days]

        # If multiple same-day expiries, follow priority
        priority = {"NIFTY": 1, "SENSEX": 2, "BANKNIFTY": 3}
        same_day_expiries.sort(key=lambda x: priority.get(x.index, 99))

        selected = same_day_expiries[0]
        logger.info(f"Selected trading index (from Kite): {selected.index} (Expiry: {selected.expiry_date}, Days: {selected.days_to_expiry})")

        return selected

    def calculate_order_size(
        self,
        price: float,
        lot_size: int,
        available_capital: float | None = None,
    ) -> tuple[int, int, list[dict]]:
        """
        Calculate order size and split if needed.

        Args:
            price: Option premium price
            lot_size: Lot size for the index
            available_capital: Capital to use (default: full available)

        Returns:
            Tuple of (total_lots, total_quantity, split_orders)
        """
        if available_capital is None:
            available_capital = self.daily_stats.current_capital * self.MAX_CAPITAL_USE

        # Calculate max affordable lots
        cost_per_lot = price * lot_size
        max_lots = int(available_capital / cost_per_lot) if cost_per_lot > 0 else 0

        if max_lots == 0:
            return 0, 0, []

        total_quantity = max_lots * lot_size

        # Split orders if exceeds max lots per order
        split_orders = []
        remaining_lots = max_lots
        order_num = 1

        while remaining_lots > 0:
            order_lots = min(remaining_lots, self.MAX_LOTS_PER_ORDER)
            order_qty = order_lots * lot_size

            split_orders.append({
                "order_num": order_num,
                "lots": order_lots,
                "quantity": order_qty,
                "value": order_qty * price,
            })

            remaining_lots -= order_lots
            order_num += 1

        logger.info(f"Order size: {max_lots} lots ({total_quantity} qty) split into {len(split_orders)} orders")

        return max_lots, total_quantity, split_orders

    def check_daily_loss_limit(self) -> bool:
        """
        Check if daily loss limit has been reached.

        Returns:
            True if trading should be halted
        """
        if self.daily_stats.is_trading_halted:
            return True

        daily_loss_percent = abs(self.daily_stats.total_pnl) / self.daily_stats.starting_capital

        if self.daily_stats.total_pnl < 0 and daily_loss_percent >= self.MAX_DAILY_LOSS_PERCENT:
            self.daily_stats.is_trading_halted = True
            self.daily_stats.halt_reason = f"Daily loss limit reached: {daily_loss_percent*100:.1f}% (Max: {self.MAX_DAILY_LOSS_PERCENT*100}%)"
            logger.warning(self.daily_stats.halt_reason)
            return True

        return False

    def is_trading_hours(self, trading_index: "ExpiryInfo | None" = None) -> tuple[bool, str]:
        """
        Check if current time is within trading hours.

        Trading Hours:
        - Normal Day: 9:30 AM to 2:00 PM
        - Expiry Day: 9:30 AM to 1:00 PM

        Args:
            trading_index: ExpiryInfo to check if it's expiry day

        Returns:
            Tuple of (is_within_hours, reason_if_not)
        """
        now = datetime.now()
        current_hour = now.hour
        current_minute = now.minute

        # Market open time: 9:30 AM
        market_open_hour = 9
        market_open_minute = 30

        # Before market open
        if current_hour < market_open_hour or (current_hour == market_open_hour and current_minute < market_open_minute):
            return False, "Market not yet open (Opens at 9:30 AM)"

        # Determine close time based on expiry
        is_expiry = trading_index.is_expiry_day if trading_index else False

        if is_expiry:
            # Expiry day: Close at 1:00 PM
            close_hour = 13
            close_minute = 0
            close_time_str = "1:00 PM"
        else:
            # Normal day: Close at 2:00 PM
            close_hour = 14
            close_minute = 0
            close_time_str = "2:00 PM"

        # After trading close time
        if current_hour > close_hour or (current_hour == close_hour and current_minute >= close_minute):
            return False, f"Trading closed for the day (Closes at {close_time_str})"

        return True, ""

    def get_trading_time_info(self, trading_index: "ExpiryInfo | None" = None) -> dict:
        """
        Get trading time information.

        Returns:
            Dictionary with trading time details
        """
        is_open, reason = self.is_trading_hours(trading_index)
        is_expiry = trading_index.is_expiry_day if trading_index else False

        return {
            "is_trading_open": is_open,
            "reason": reason,
            "is_expiry_day": is_expiry,
            "open_time": "09:30",
            "close_time": "13:00" if is_expiry else "14:00",
            "close_time_display": "1:00 PM" if is_expiry else "2:00 PM",
        }

    def has_open_position(self) -> bool:
        """Check if there are any open positions."""
        return len(self.get_open_positions()) > 0

    def find_similar_position(self, option_type: str) -> PaperPosition | None:
        """
        Find an open position with the same option type (CE/PE).

        Args:
            option_type: CE or PE

        Returns:
            PaperPosition if found, None otherwise
        """
        for position in self.positions:
            if position.status == PositionStatus.OPEN and position.option_type == option_type:
                return position
        return None

    def update_position_target(self, position: PaperPosition, new_target: float, new_stop_loss: float) -> bool:
        """
        Update target and stop loss for an existing position.

        Args:
            position: Position to update
            new_target: New target price
            new_stop_loss: New stop loss price

        Returns:
            True if updated, False otherwise
        """
        if position.status != PositionStatus.OPEN:
            return False

        position.target = new_target
        position.stop_loss = new_stop_loss
        self._save_state()
        logger.info(f"Updated position {position.position_id} - Target: {new_target}, SL: {new_stop_loss}")
        return True

    async def execute_signal_trade(
        self,
        signal: Any,
        trading_index: ExpiryInfo,
    ) -> PaperOrder | None:
        """
        Execute a trade based on signal.

        Args:
            signal: TradeSignal from signal engine
            trading_index: ExpiryInfo for the trading index

        Returns:
            PaperOrder if executed, None otherwise
        """
        # Check trading hours first
        is_trading, reason = self.is_trading_hours(trading_index)
        if not is_trading:
            logger.info(f"Outside trading hours: {reason}")
            return None

        # Check if trading is halted
        if self.check_daily_loss_limit():
            logger.warning("Trading halted due to daily loss limit")
            return None

        # Check signal direction
        if signal.direction not in ["CE", "PE"]:
            logger.info("No clear signal direction")
            return None

        # Check confidence threshold (at least 55%)
        if signal.confidence < 55:
            logger.info(f"Signal confidence too low: {signal.confidence}%")
            return None

        # Get recommended option
        if not signal.recommended_option:
            logger.warning("No recommended option in signal")
            return None

        opt = signal.recommended_option

        # Check if there's already an open position
        if self.has_open_position():
            # Check if signal matches existing position - update target if same direction
            existing_position = self.find_similar_position(signal.direction)
            if existing_position:
                # Update target and stop loss for the existing position
                self.update_position_target(
                    existing_position,
                    new_target=signal.target_1,
                    new_stop_loss=signal.stop_loss,
                )
                logger.info(f"Position already open. Updated target for {existing_position.symbol}")
            else:
                logger.info("Position already open with different direction. Skipping new order.")
            return None

        # Calculate order size
        lots, quantity, split_orders = self.calculate_order_size(
            price=opt.ltp,
            lot_size=trading_index.lot_size,
        )

        if lots == 0:
            logger.warning("Insufficient capital for trade")
            return None

        # Create order
        order = PaperOrder(
            order_id=self._generate_order_id(),
            timestamp=datetime.now(),
            index=trading_index.index,
            symbol=opt.symbol,
            strike=opt.strike,
            option_type=signal.direction,
            order_type=OrderType.BUY,
            quantity=quantity,
            lots=lots,
            price=opt.ltp,
            status=OrderStatus.EXECUTED,
            executed_quantity=quantity,
            executed_price=opt.ltp,
            split_orders=split_orders,
            reason=f"Signal: {signal.signal_type.value}, Confidence: {signal.confidence}%",
            signal_confidence=signal.confidence,
        )

        # Create position with max/min tracking initialized to entry price
        now = datetime.now()
        position = PaperPosition(
            position_id=self._generate_position_id(),
            index=trading_index.index,
            symbol=opt.symbol,
            strike=opt.strike,
            option_type=signal.direction,
            entry_price=opt.ltp,
            quantity=quantity,
            lots=lots,
            entry_time=now,
            current_price=opt.ltp,
            status=PositionStatus.OPEN,
            stop_loss=signal.stop_loss,
            target=signal.target_1,
            max_price=opt.ltp,  # Initialize to entry price
            min_price=opt.ltp,  # Initialize to entry price
            max_price_time=now,
            min_price_time=now,
        )

        # Update capital
        trade_value = quantity * opt.ltp
        self.daily_stats.current_capital -= trade_value
        self.daily_stats.total_trades += 1

        # Add to lists
        self.orders.append(order)
        self.positions.append(position)

        # Save state
        self._save_state()

        logger.info(f"Executed BUY order: {order.order_id} - {order.symbol} x {order.lots} lots @ {order.price}")

        return order

    async def update_positions(self) -> list[PaperPosition]:
        """
        Update all open positions with current prices.

        Returns:
            List of updated positions
        """
        updated = []

        for position in self.positions:
            if position.status != PositionStatus.OPEN:
                continue

            try:
                # Fetch current option price
                chain_data = await self.data_fetcher.get_option_chain(
                    index=position.index,
                    strike_count=5,
                )

                if "error" in chain_data:
                    continue

                # Find option in chain
                for opt in chain_data.get("chain", []):
                    if opt["strike"] == position.strike:
                        opt_data = opt.get(position.option_type.lower(), {})
                        if opt_data:
                            position.current_price = opt_data.get("ltp", position.current_price)
                            break

                # Update max/min price tracking
                now = datetime.now()
                if position.current_price > position.max_price:
                    position.max_price = position.current_price
                    position.max_price_time = now
                if position.current_price < position.min_price:
                    position.min_price = position.current_price
                    position.min_price_time = now

                # Calculate P&L
                position.pnl = (position.current_price - position.entry_price) * position.quantity
                position.pnl_percent = ((position.current_price - position.entry_price) / position.entry_price) * 100

                # Check stop loss / target
                spot_price = chain_data.get("spot_price", 0)

                should_exit = False
                exit_reason = ""

                if position.option_type == "CE":
                    if spot_price <= position.stop_loss:
                        should_exit = True
                        exit_reason = "Stop Loss Hit"
                    elif spot_price >= position.target:
                        should_exit = True
                        exit_reason = "Target Hit"
                else:  # PE
                    if spot_price >= position.stop_loss:
                        should_exit = True
                        exit_reason = "Stop Loss Hit"
                    elif spot_price <= position.target:
                        should_exit = True
                        exit_reason = "Target Hit"

                if should_exit:
                    await self.close_position(position, exit_reason)

                updated.append(position)

            except Exception as e:
                logger.error(f"Error updating position {position.position_id}: {e}")

        # Update unrealized P&L
        self.daily_stats.unrealized_pnl = sum(
            p.pnl for p in self.positions if p.status == PositionStatus.OPEN
        )
        self.daily_stats.total_pnl = self.daily_stats.realized_pnl + self.daily_stats.unrealized_pnl

        # Update max drawdown
        if self.daily_stats.total_pnl < 0:
            drawdown = abs(self.daily_stats.total_pnl)
            if drawdown > self.daily_stats.max_drawdown:
                self.daily_stats.max_drawdown = drawdown
                self.daily_stats.max_drawdown_percent = drawdown / self.daily_stats.starting_capital * 100

        # Save state
        self._save_state()

        return updated

    async def close_position(
        self,
        position: PaperPosition,
        reason: str = "Manual Close",
    ) -> PaperOrder | None:
        """
        Close an open position.

        Args:
            position: Position to close
            reason: Exit reason

        Returns:
            PaperOrder for the exit
        """
        if position.status != PositionStatus.OPEN:
            return None

        # Create sell order
        order = PaperOrder(
            order_id=self._generate_order_id(),
            timestamp=datetime.now(),
            index=position.index,
            symbol=position.symbol,
            strike=position.strike,
            option_type=position.option_type,
            order_type=OrderType.SELL,
            quantity=position.quantity,
            lots=position.lots,
            price=position.current_price,
            status=OrderStatus.EXECUTED,
            executed_quantity=position.quantity,
            executed_price=position.current_price,
            reason=reason,
        )

        # Update position
        now = datetime.now()
        position.status = PositionStatus.CLOSED
        position.exit_price = position.current_price
        position.exit_time = now
        position.exit_reason = reason
        position.pnl = (position.exit_price - position.entry_price) * position.quantity
        position.pnl_percent = ((position.exit_price - position.entry_price) / position.entry_price) * 100

        # Calculate performance metrics for order history
        max_profit_percent = ((position.max_price - position.entry_price) / position.entry_price) * 100 if position.entry_price > 0 else 0
        max_loss_percent = ((position.min_price - position.entry_price) / position.entry_price) * 100 if position.entry_price > 0 else 0

        # How much of max move was captured (0-100%)
        max_move = position.max_price - position.entry_price
        actual_profit = position.exit_price - position.entry_price
        captured_move_percent = (actual_profit / max_move * 100) if max_move > 0 else 0

        # Duration in minutes
        duration = (now - position.entry_time).total_seconds() / 60

        # Create order history entry
        history_entry = OrderHistoryEntry(
            order_id=order.order_id,
            position_id=position.position_id,
            timestamp=now,
            index=position.index,
            symbol=position.symbol,
            strike=position.strike,
            option_type=position.option_type,
            direction="SELL",
            quantity=position.quantity,
            lots=position.lots,
            entry_price=position.entry_price,
            exit_price=position.exit_price,
            max_price=position.max_price,
            min_price=position.min_price,
            max_price_time=position.max_price_time,
            min_price_time=position.min_price_time,
            pnl=position.pnl,
            pnl_percent=position.pnl_percent,
            max_profit_percent=max_profit_percent,
            max_loss_percent=max_loss_percent,
            captured_move_percent=captured_move_percent,
            signal_confidence=0,  # Will be populated from entry order if available
            exit_reason=reason,
            entry_time=position.entry_time,
            exit_time=now,
            duration_minutes=int(duration),
        )

        # Find original buy order to get signal confidence
        for orig_order in self.orders:
            if orig_order.symbol == position.symbol and orig_order.order_type == OrderType.BUY:
                if orig_order.timestamp.date() == position.entry_time.date():
                    history_entry.signal_confidence = orig_order.signal_confidence
                    break

        self.order_history.append(history_entry)

        # Update daily stats
        self.daily_stats.realized_pnl += position.pnl
        self.daily_stats.current_capital += position.quantity * position.exit_price

        if position.pnl > 0:
            self.daily_stats.winning_trades += 1
        else:
            self.daily_stats.losing_trades += 1

        # Add order
        self.orders.append(order)

        # Check daily loss limit
        self.check_daily_loss_limit()

        # Save state
        self._save_state()

        logger.info(f"Closed position: {position.position_id} - P&L: ₹{position.pnl:.2f}")

        return order

    async def close_all_positions(self, reason: str = "Close All") -> list[PaperOrder]:
        """Close all open positions."""
        orders = []
        for position in self.positions:
            if position.status == PositionStatus.OPEN:
                order = await self.close_position(position, reason)
                if order:
                    orders.append(order)
        return orders

    def get_open_positions(self) -> list[PaperPosition]:
        """Get all open positions."""
        return [p for p in self.positions if p.status == PositionStatus.OPEN]

    def get_closed_positions(self) -> list[PaperPosition]:
        """Get all closed positions."""
        return [p for p in self.positions if p.status == PositionStatus.CLOSED]

    def get_today_orders(self) -> list[PaperOrder]:
        """Get today's orders."""
        today = date.today()
        return [o for o in self.orders if o.timestamp.date() == today]

    def get_order_history(self, days: int = 7) -> list[OrderHistoryEntry]:
        """
        Get order history with detailed performance metrics.

        Args:
            days: Number of days to look back (default 7)

        Returns:
            List of OrderHistoryEntry sorted by timestamp descending
        """
        cutoff = datetime.now() - timedelta(days=days)
        history = [h for h in self.order_history if h.timestamp >= cutoff]
        return sorted(history, key=lambda x: x.timestamp, reverse=True)

    def get_order_history_summary(self) -> dict:
        """Get summary statistics from order history."""
        history = self.order_history
        if not history:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "avg_profit_percent": 0,
                "avg_loss_percent": 0,
                "avg_captured_move": 0,
                "best_trade_pnl": 0,
                "worst_trade_pnl": 0,
                "avg_duration_minutes": 0,
                "avg_max_profit_percent": 0,
                "avg_max_loss_percent": 0,
            }

        winning = [h for h in history if h.pnl > 0]
        losing = [h for h in history if h.pnl <= 0]

        return {
            "total_trades": len(history),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate": len(winning) / len(history) * 100 if history else 0,
            "avg_profit_percent": sum(h.pnl_percent for h in winning) / len(winning) if winning else 0,
            "avg_loss_percent": sum(h.pnl_percent for h in losing) / len(losing) if losing else 0,
            "avg_captured_move": sum(h.captured_move_percent for h in history) / len(history) if history else 0,
            "best_trade_pnl": max((h.pnl for h in history), default=0),
            "worst_trade_pnl": min((h.pnl for h in history), default=0),
            "avg_duration_minutes": sum(h.duration_minutes for h in history) / len(history) if history else 0,
            "avg_max_profit_percent": sum(h.max_profit_percent for h in history) / len(history) if history else 0,
            "avg_max_loss_percent": sum(h.max_loss_percent for h in history) / len(history) if history else 0,
        }

    def get_stats(self) -> dict:
        """Get trading statistics."""
        open_positions = self.get_open_positions()

        return {
            "capital": {
                "initial": self.CAPITAL,
                "starting_today": self.daily_stats.starting_capital,
                "current": self.daily_stats.current_capital,
                "available": self.daily_stats.current_capital,
            },
            "pnl": {
                "total": self.daily_stats.total_pnl,
                "realized": self.daily_stats.realized_pnl,
                "unrealized": self.daily_stats.unrealized_pnl,
                "percent": (self.daily_stats.total_pnl / self.daily_stats.starting_capital) * 100,
            },
            "trades": {
                "total": self.daily_stats.total_trades,
                "winning": self.daily_stats.winning_trades,
                "losing": self.daily_stats.losing_trades,
                "win_rate": (self.daily_stats.winning_trades / self.daily_stats.total_trades * 100) if self.daily_stats.total_trades > 0 else 0,
            },
            "risk": {
                "max_drawdown": self.daily_stats.max_drawdown,
                "max_drawdown_percent": self.daily_stats.max_drawdown_percent,
                "daily_loss_limit": self.MAX_DAILY_LOSS_PERCENT * 100,
                "daily_loss_used": abs(min(0, self.daily_stats.total_pnl)) / self.daily_stats.starting_capital * 100,
            },
            "status": {
                "is_halted": self.daily_stats.is_trading_halted,
                "halt_reason": self.daily_stats.halt_reason,
                "open_positions": len(open_positions),
                "is_auto_trade": self.is_auto_trade,
            },
        }

    def toggle_auto_trade(self, enabled: bool | None = None) -> bool:
        """
        Toggle or set auto trade status.

        Args:
            enabled: If None, toggles current state. If bool, sets to that value.

        Returns:
            New auto trade status
        """
        if enabled is None:
            self.is_auto_trade = not self.is_auto_trade
        else:
            self.is_auto_trade = enabled

        logger.info(f"Auto trade {'enabled' if self.is_auto_trade else 'disabled'}")
        return self.is_auto_trade

    def reset_daily(self):
        """Reset daily statistics (call at start of new trading day)."""
        self._initialize_daily_stats()
        self._save_state()

    def reset_all(self):
        """Reset all trading data."""
        self.orders = []
        self.positions = []
        self.order_history = []
        self._order_counter = 0
        self._position_counter = 0

        self.daily_stats = DailyStats(
            date=date.today(),
            starting_capital=self.CAPITAL,
            current_capital=self.CAPITAL,
            total_pnl=0,
            realized_pnl=0,
            unrealized_pnl=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            max_drawdown=0,
            max_drawdown_percent=0,
            daily_loss_percent=0,
        )

        self._save_state()
        logger.info("Paper trading reset to initial state")

    def _save_state(self):
        """Save state to file."""
        try:
            state = {
                "orders": [
                    {
                        "order_id": o.order_id,
                        "timestamp": o.timestamp.isoformat(),
                        "index": o.index,
                        "symbol": o.symbol,
                        "strike": o.strike,
                        "option_type": o.option_type,
                        "order_type": o.order_type.value,
                        "quantity": o.quantity,
                        "lots": o.lots,
                        "price": o.price,
                        "status": o.status.value,
                        "executed_quantity": o.executed_quantity,
                        "executed_price": o.executed_price,
                        "split_orders": o.split_orders,
                        "reason": o.reason,
                        "signal_confidence": o.signal_confidence,
                    }
                    for o in self.orders[-100:]  # Keep last 100 orders
                ],
                "positions": [
                    {
                        "position_id": p.position_id,
                        "index": p.index,
                        "symbol": p.symbol,
                        "strike": p.strike,
                        "option_type": p.option_type,
                        "entry_price": p.entry_price,
                        "quantity": p.quantity,
                        "lots": p.lots,
                        "entry_time": p.entry_time.isoformat(),
                        "current_price": p.current_price,
                        "pnl": p.pnl,
                        "pnl_percent": p.pnl_percent,
                        "status": p.status.value,
                        "exit_price": p.exit_price,
                        "exit_time": p.exit_time.isoformat() if p.exit_time else None,
                        "exit_reason": p.exit_reason,
                        "stop_loss": p.stop_loss,
                        "target": p.target,
                        "max_price": p.max_price,
                        "min_price": p.min_price,
                        "max_price_time": p.max_price_time.isoformat() if p.max_price_time else None,
                        "min_price_time": p.min_price_time.isoformat() if p.min_price_time else None,
                    }
                    for p in self.positions[-50:]  # Keep last 50 positions
                ],
                "daily_stats": {
                    "date": self.daily_stats.date.isoformat(),
                    "starting_capital": self.daily_stats.starting_capital,
                    "current_capital": self.daily_stats.current_capital,
                    "total_pnl": self.daily_stats.total_pnl,
                    "realized_pnl": self.daily_stats.realized_pnl,
                    "unrealized_pnl": self.daily_stats.unrealized_pnl,
                    "total_trades": self.daily_stats.total_trades,
                    "winning_trades": self.daily_stats.winning_trades,
                    "losing_trades": self.daily_stats.losing_trades,
                    "max_drawdown": self.daily_stats.max_drawdown,
                    "max_drawdown_percent": self.daily_stats.max_drawdown_percent,
                    "daily_loss_percent": self.daily_stats.daily_loss_percent,
                    "is_trading_halted": self.daily_stats.is_trading_halted,
                    "halt_reason": self.daily_stats.halt_reason,
                },
                "counters": {
                    "order": self._order_counter,
                    "position": self._position_counter,
                },
                "order_history": [
                    {
                        "order_id": h.order_id,
                        "position_id": h.position_id,
                        "timestamp": h.timestamp.isoformat(),
                        "index": h.index,
                        "symbol": h.symbol,
                        "strike": h.strike,
                        "option_type": h.option_type,
                        "direction": h.direction,
                        "quantity": h.quantity,
                        "lots": h.lots,
                        "entry_price": h.entry_price,
                        "exit_price": h.exit_price,
                        "max_price": h.max_price,
                        "min_price": h.min_price,
                        "max_price_time": h.max_price_time.isoformat() if h.max_price_time else None,
                        "min_price_time": h.min_price_time.isoformat() if h.min_price_time else None,
                        "pnl": h.pnl,
                        "pnl_percent": h.pnl_percent,
                        "max_profit_percent": h.max_profit_percent,
                        "max_loss_percent": h.max_loss_percent,
                        "captured_move_percent": h.captured_move_percent,
                        "signal_confidence": h.signal_confidence,
                        "exit_reason": h.exit_reason,
                        "entry_time": h.entry_time.isoformat() if h.entry_time else None,
                        "exit_time": h.exit_time.isoformat() if h.exit_time else None,
                        "duration_minutes": h.duration_minutes,
                    }
                    for h in self.order_history[-200:]  # Keep last 200 history entries
                ],
                "is_auto_trade": self.is_auto_trade,
            }

            self.data_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.data_file, "w") as f:
                json.dump(state, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving paper trading state: {e}")

    def _load_state(self):
        """Load state from file."""
        try:
            if not self.data_file.exists():
                return

            with open(self.data_file, "r") as f:
                state = json.load(f)

            # Load counters
            counters = state.get("counters", {})
            self._order_counter = counters.get("order", 0)
            self._position_counter = counters.get("position", 0)

            # Load daily stats
            stats_data = state.get("daily_stats", {})
            stats_date = date.fromisoformat(stats_data.get("date", date.today().isoformat()))

            if stats_date == date.today():
                self.daily_stats = DailyStats(
                    date=stats_date,
                    starting_capital=stats_data.get("starting_capital", self.CAPITAL),
                    current_capital=stats_data.get("current_capital", self.CAPITAL),
                    total_pnl=stats_data.get("total_pnl", 0),
                    realized_pnl=stats_data.get("realized_pnl", 0),
                    unrealized_pnl=stats_data.get("unrealized_pnl", 0),
                    total_trades=stats_data.get("total_trades", 0),
                    winning_trades=stats_data.get("winning_trades", 0),
                    losing_trades=stats_data.get("losing_trades", 0),
                    max_drawdown=stats_data.get("max_drawdown", 0),
                    max_drawdown_percent=stats_data.get("max_drawdown_percent", 0),
                    daily_loss_percent=stats_data.get("daily_loss_percent", 0),
                    is_trading_halted=stats_data.get("is_trading_halted", False),
                    halt_reason=stats_data.get("halt_reason", ""),
                )

            # Load orders (today's only)
            for o_data in state.get("orders", []):
                timestamp = datetime.fromisoformat(o_data["timestamp"])
                if timestamp.date() == date.today():
                    self.orders.append(PaperOrder(
                        order_id=o_data["order_id"],
                        timestamp=timestamp,
                        index=o_data["index"],
                        symbol=o_data["symbol"],
                        strike=o_data["strike"],
                        option_type=o_data["option_type"],
                        order_type=OrderType(o_data["order_type"]),
                        quantity=o_data["quantity"],
                        lots=o_data["lots"],
                        price=o_data["price"],
                        status=OrderStatus(o_data["status"]),
                        executed_quantity=o_data["executed_quantity"],
                        executed_price=o_data["executed_price"],
                        split_orders=o_data["split_orders"],
                        reason=o_data["reason"],
                        signal_confidence=o_data.get("signal_confidence", 0),
                    ))

            # Load positions (open ones)
            for p_data in state.get("positions", []):
                status = PositionStatus(p_data["status"])
                if status == PositionStatus.OPEN:
                    entry_time = datetime.fromisoformat(p_data["entry_time"])
                    self.positions.append(PaperPosition(
                        position_id=p_data["position_id"],
                        index=p_data["index"],
                        symbol=p_data["symbol"],
                        strike=p_data["strike"],
                        option_type=p_data["option_type"],
                        entry_price=p_data["entry_price"],
                        quantity=p_data["quantity"],
                        lots=p_data["lots"],
                        entry_time=entry_time,
                        current_price=p_data["current_price"],
                        pnl=p_data["pnl"],
                        pnl_percent=p_data["pnl_percent"],
                        status=status,
                        exit_price=p_data["exit_price"],
                        exit_time=datetime.fromisoformat(p_data["exit_time"]) if p_data["exit_time"] else None,
                        exit_reason=p_data["exit_reason"],
                        stop_loss=p_data["stop_loss"],
                        target=p_data["target"],
                        max_price=p_data.get("max_price", p_data["entry_price"]),
                        min_price=p_data.get("min_price", p_data["entry_price"]),
                        max_price_time=datetime.fromisoformat(p_data["max_price_time"]) if p_data.get("max_price_time") else entry_time,
                        min_price_time=datetime.fromisoformat(p_data["min_price_time"]) if p_data.get("min_price_time") else entry_time,
                    ))

            # Load order history
            for h_data in state.get("order_history", []):
                self.order_history.append(OrderHistoryEntry(
                    order_id=h_data["order_id"],
                    position_id=h_data["position_id"],
                    timestamp=datetime.fromisoformat(h_data["timestamp"]),
                    index=h_data["index"],
                    symbol=h_data["symbol"],
                    strike=h_data["strike"],
                    option_type=h_data["option_type"],
                    direction=h_data["direction"],
                    quantity=h_data["quantity"],
                    lots=h_data["lots"],
                    entry_price=h_data["entry_price"],
                    exit_price=h_data.get("exit_price", 0),
                    max_price=h_data.get("max_price", 0),
                    min_price=h_data.get("min_price", 0),
                    max_price_time=datetime.fromisoformat(h_data["max_price_time"]) if h_data.get("max_price_time") else None,
                    min_price_time=datetime.fromisoformat(h_data["min_price_time"]) if h_data.get("min_price_time") else None,
                    pnl=h_data.get("pnl", 0),
                    pnl_percent=h_data.get("pnl_percent", 0),
                    max_profit_percent=h_data.get("max_profit_percent", 0),
                    max_loss_percent=h_data.get("max_loss_percent", 0),
                    captured_move_percent=h_data.get("captured_move_percent", 0),
                    signal_confidence=h_data.get("signal_confidence", 0),
                    exit_reason=h_data.get("exit_reason", ""),
                    entry_time=datetime.fromisoformat(h_data["entry_time"]) if h_data.get("entry_time") else None,
                    exit_time=datetime.fromisoformat(h_data["exit_time"]) if h_data.get("exit_time") else None,
                    duration_minutes=h_data.get("duration_minutes", 0),
                ))

            # Load auto trade setting
            self.is_auto_trade = state.get("is_auto_trade", True)

            logger.info(f"Loaded paper trading state: {len(self.orders)} orders, {len(self.positions)} positions, {len(self.order_history)} history entries")

        except Exception as e:
            logger.error(f"Error loading paper trading state: {e}")


# Singleton instance
_paper_trading_service: PaperTradingService | None = None


def get_paper_trading_service() -> PaperTradingService:
    """Get or create paper trading service instance."""
    global _paper_trading_service
    if _paper_trading_service is None:
        _paper_trading_service = PaperTradingService()
    return _paper_trading_service
