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
from app.services.signal_engine import TradingStyle, SignalType, get_signal_engine


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
    # Trailing stop loss tracking
    initial_stop_loss: float = 0.0  # Original SL from signal
    initial_target: float = 0.0  # Original target from signal
    trailing_sl_active: bool = False  # True when trailing SL is active
    target_achieved: bool = False  # True when target price was reached
    profit_locked_percent: float = 0.0  # Current locked profit %
    sl_trail_count: int = 0  # Number of times SL was trailed up


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
    pnl: float = 0.0  # Gross P&L before charges
    pnl_percent: float = 0.0
    max_profit_percent: float = 0.0  # % from entry to max
    max_loss_percent: float = 0.0    # % from entry to min
    captured_move_percent: float = 0.0  # How much of max move was captured
    # Broker charges
    broker_charges: float = 0.0  # Total broker charges
    net_pnl: float = 0.0  # Net P&L after charges (pnl - broker_charges)
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


# Global expiry cache shared across all PaperTradingService instances
_global_expiry_cache: dict[str, "ExpiryInfo"] = {}
_global_cache_time: datetime | None = None


class PaperTradingService:
    """
    Paper Trading Service for automatic signal-based trading.

    Features:
    - Auto-detect nearest expiry index
    - Priority order: NIFTY > SENSEX > BANKNIFTY (if same day expiry)
    - Capital management with daily loss limits
    - Order splitting for large orders
    - Position tracking and P&L calculation
    - Broker charges calculation (STT, transaction charges, GST, SEBI, stamp duty)
    """

    # Configuration
    CAPITAL = 100000  # ₹1,00,000
    MAX_CAPITAL_USE = 1.0  # 100%
    MAX_DAILY_LOSS_PERCENT = 0.20  # 20% daily loss limit - halt trading if reached
    MAX_LOTS_PER_ORDER = 25

    # Lot sizes
    LOT_SIZES = {
        "NIFTY": 25,
        "BANKNIFTY": 15,
        "SENSEX": 10,
    }

    # Broker Charges (standard rates for options)
    # All rates are as per typical discount broker rates (Zerodha-like)
    BROKERAGE_PER_ORDER = 20  # Flat ₹20 per executed order (buy + sell = ₹40)
    STT_RATE = 0.000625  # 0.0625% on sell side (intrinsic value for options)
    TRANSACTION_CHARGES_NSE = 0.00053  # 0.053% for options (NSE)
    TRANSACTION_CHARGES_BSE = 0.000375  # 0.0375% for options (BSE/SENSEX)
    GST_RATE = 0.18  # 18% on brokerage + transaction charges
    SEBI_CHARGES = 0.000001  # ₹10 per crore = 0.0001%
    STAMP_DUTY_BUY = 0.00003  # 0.003% on buy side only

    def __init__(self, strategy: str = "default"):
        self.settings = get_settings()
        self.strategy = strategy  # Strategy type: default, fixed_20_percent, trailing_stoploss, profit_100_halt
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

        # Data file path - EACH STRATEGY HAS ITS OWN FILE
        # This ensures orders, positions, and history are separate per strategy
        if strategy == "default":
            self.data_file = Path(self.settings.data_dir) / "paper_trading.json"
        else:
            self.data_file = Path(self.settings.data_dir) / f"paper_trading_{strategy}.json"

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
            # Always use base CAPITAL (no compounding of profits)
            # Profits are tracked separately but not added to trading capital

            self.daily_stats = DailyStats(
                date=today,
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
        Uses cached data from Kite if available and less than 5 minutes old.
        Falls back to global singleton cache if instance cache is empty.

        IMPORTANT: Cache must be populated first using refresh_expiry_cache().
        This is done automatically on app startup.

        Args:
            index: NIFTY, BANKNIFTY, or SENSEX

        Returns:
            ExpiryInfo with expiry details

        Raises:
            RuntimeError: If cache is empty (Kite data never fetched)
        """
        index = index.upper()

        # Check instance cache first (valid for 5 minutes)
        if (
            self._expiry_cache_time
            and index in self._expiry_cache
            and (datetime.now() - self._expiry_cache_time).seconds < 300
        ):
            return self._expiry_cache[index]

        # Fallback to global singleton cache if available
        from app.services.paper_trading import _global_expiry_cache, _global_cache_time
        if (
            _global_cache_time
            and index in _global_expiry_cache
            and (datetime.now() - _global_cache_time).seconds < 300
        ):
            # Copy to instance cache for future use
            self._expiry_cache[index] = _global_expiry_cache[index]
            self._expiry_cache_time = _global_cache_time
            logger.debug(f"Using global expiry cache for {index}")
            return self._expiry_cache[index]

        # Cache miss - this means refresh_expiry_cache() was never called
        logger.error(f"Expiry cache miss for {index}. Cache must be populated using refresh_expiry_cache() first.")
        raise RuntimeError(
            f"Expiry data not available for {index}. "
            "Please call refresh_expiry_cache() first or wait for app startup to complete."
        )

    async def get_next_expiry_async(self, index: str) -> ExpiryInfo:
        """
        Get next expiry date for an index from Kite instruments (async).
        This method always fetches fresh data from Kite and updates the cache.

        Args:
            index: NIFTY, BANKNIFTY, or SENSEX

        Returns:
            ExpiryInfo with expiry details

        Raises:
            RuntimeError: If unable to fetch from Kite
        """
        index = index.upper()

        try:
            # Fetch from Kite
            expiry_data = await self.data_fetcher.get_next_expiry(index)

            if "error" in expiry_data:
                error_msg = f"Kite expiry fetch failed for {index}: {expiry_data['error']}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            expiry_info = ExpiryInfo(
                index=index,
                expiry_date=expiry_data["expiry_date"],
                days_to_expiry=expiry_data["days_to_expiry"],
                is_expiry_day=expiry_data["is_expiry_day"],
                lot_size=self.LOT_SIZES.get(index, 25),
                max_lots_per_order=self.MAX_LOTS_PER_ORDER,
            )

            # Cache the result (both instance and global)
            self._expiry_cache[index] = expiry_info
            self._expiry_cache_time = datetime.now()

            # Update global cache so other instances can use it
            global _global_expiry_cache, _global_cache_time
            _global_expiry_cache[index] = expiry_info
            _global_cache_time = datetime.now()

            logger.info(f"Fetched expiry from Kite: {index} -> {expiry_data['expiry_date']} ({expiry_data['expiry_weekday']})")

            return expiry_info

        except Exception as e:
            logger.error(f"Error fetching expiry from Kite for {index}: {e}")
            raise RuntimeError(f"Unable to fetch expiry data for {index} from Kite: {e}")

    async def refresh_expiry_cache(self):
        """
        Refresh expiry cache for all indices from Kite.

        This should be called:
        - On app startup (done automatically)
        - Before trading operations to ensure fresh data
        - When cache expires (every 5 minutes)

        Raises:
            RuntimeError: If unable to fetch data for any index
        """
        logger.info("Refreshing expiry cache from Kite for all indices...")
        success_count = 0
        errors = []

        for index in ["NIFTY", "BANKNIFTY", "SENSEX"]:
            try:
                await self.get_next_expiry_async(index)
                success_count += 1
            except Exception as e:
                error_msg = f"{index}: {str(e)}"
                errors.append(error_msg)
                logger.error(f"Failed to refresh expiry for {index}: {e}")

        if success_count == 0:
            raise RuntimeError(f"Failed to fetch expiry data for all indices. Errors: {'; '.join(errors)}")
        elif errors:
            logger.warning(f"Partial success: {success_count}/3 indices updated. Errors: {'; '.join(errors)}")
        else:
            logger.info(f"Successfully refreshed expiry cache for all {success_count} indices")

    def get_trading_index(self) -> ExpiryInfo:
        """
        Get the index to trade based on nearest expiry (sync version).
        Uses cached data if available, with fallback to default NIFTY.

        Priority when same-day expiry: NIFTY > SENSEX > BANKNIFTY

        Returns:
            ExpiryInfo for the selected index
        """
        expiries = []
        for index in ["NIFTY", "SENSEX", "BANKNIFTY"]:
            try:
                expiry = self.get_next_expiry(index)
                expiries.append(expiry)
            except RuntimeError as e:
                logger.debug(f"Cache miss for {index}: {e}")
                continue

        # If no cached data, return default NIFTY with estimated expiry
        if not expiries:
            logger.warning("No expiry cache available - using default NIFTY settings")
            from datetime import timedelta
            today = date.today()
            # Estimate next Thursday (NIFTY weekly expiry)
            days_until_thursday = (3 - today.weekday()) % 7
            if days_until_thursday == 0 and datetime.now().hour >= 15:
                days_until_thursday = 7  # Next week if today's expiry is over
            next_expiry = today + timedelta(days=days_until_thursday)

            return ExpiryInfo(
                index="NIFTY",
                expiry_date=next_expiry,
                days_to_expiry=days_until_thursday,
                is_expiry_day=(days_until_thursday == 0),
                lot_size=75,  # Default NIFTY lot size
            )

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
            # Use starting capital (fixed), not current capital (which changes with trades)
            available_capital = self.daily_stats.starting_capital * self.MAX_CAPITAL_USE

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

    def calculate_broker_charges(
        self,
        buy_value: float,
        sell_value: float,
        index: str = "NIFTY",
    ) -> dict:
        """
        Calculate all broker charges for a complete trade (buy + sell).

        Args:
            buy_value: Total value of buy order (price * quantity)
            sell_value: Total value of sell order (price * quantity)
            index: Index name (NIFTY/BANKNIFTY/SENSEX) for exchange selection

        Returns:
            Dictionary with breakdown of all charges and total
        """
        # 1. Brokerage: ₹20 per order (buy + sell = ₹40)
        brokerage = self.BROKERAGE_PER_ORDER * 2

        # 2. STT (Securities Transaction Tax): 0.0625% on sell side only (on premium)
        stt = sell_value * self.STT_RATE

        # 3. Transaction charges: Different for NSE vs BSE
        if index.upper() == "SENSEX":
            transaction_rate = self.TRANSACTION_CHARGES_BSE
        else:
            transaction_rate = self.TRANSACTION_CHARGES_NSE
        transaction_charges = (buy_value + sell_value) * transaction_rate

        # 4. GST: 18% on (brokerage + transaction charges)
        gst = (brokerage + transaction_charges) * self.GST_RATE

        # 5. SEBI charges: ₹10 per crore (0.0001%)
        sebi_charges = (buy_value + sell_value) * self.SEBI_CHARGES

        # 6. Stamp duty: 0.003% on buy side only
        stamp_duty = buy_value * self.STAMP_DUTY_BUY

        # Total charges
        total_charges = brokerage + stt + transaction_charges + gst + sebi_charges + stamp_duty

        return {
            "brokerage": round(brokerage, 2),
            "stt": round(stt, 2),
            "transaction_charges": round(transaction_charges, 2),
            "gst": round(gst, 2),
            "sebi_charges": round(sebi_charges, 2),
            "stamp_duty": round(stamp_duty, 2),
            "total": round(total_charges, 2),
        }

    def analyze_swing_entry(
        self,
        df: "pd.DataFrame",
        signal_direction: str,
        ltp: float,
    ) -> dict:
        """
        Analyze 30-minute data to find swing entry point.

        Strategy:
        - Don't enter when price is moving opposite to signal
        - Wait for reversal/pullback confirmation
        - Use swing high/low for entry confirmation

        Args:
            df: DataFrame with last 30+ minutes of data
            signal_direction: CE or PE
            ltp: Current LTP

        Returns:
            Dict with entry_allowed, entry_price, reason
        """
        if df is None or len(df) < 6:  # Need at least 30 min (6 x 5min candles)
            return {
                "entry_allowed": True,
                "entry_price": ltp,
                "reason": "Insufficient data - using LTP"
            }

        # Get last 30 minutes of data
        recent_df = df.tail(6)  # 6 candles * 5 min = 30 min

        # Calculate swing points
        highs = recent_df["High"].values
        lows = recent_df["Low"].values
        closes = recent_df["Close"].values

        current_close = closes[-1]
        prev_close = closes[-2] if len(closes) > 1 else current_close

        # Calculate price movement direction
        price_change = current_close - prev_close
        price_change_pct = (price_change / prev_close) * 100 if prev_close > 0 else 0

        # Calculate swing high and swing low from last 30 min
        swing_high = max(highs)
        swing_low = min(lows)
        swing_range = swing_high - swing_low

        # Calculate position in swing range (0-100%)
        if swing_range > 0:
            position_in_range = ((current_close - swing_low) / swing_range) * 100
        else:
            position_in_range = 50

        # Determine if price is moving in signal direction
        moving_with_signal = False
        if signal_direction == "CE" and price_change > 0:
            moving_with_signal = True
        elif signal_direction == "PE" and price_change < 0:
            moving_with_signal = True

        # ENTRY LOGIC based on swing trading
        entry_allowed = False
        entry_price = ltp
        reason = ""

        if signal_direction == "CE":
            # For CE signal:
            # - Don't enter if price is falling (opposite to signal)
            # - Wait for price to bounce from swing low or break swing high
            # - Best entry: price near swing low and starting to rise

            if position_in_range < 30 and price_change > 0:
                # Near swing low and rising - BEST ENTRY
                entry_allowed = True
                entry_price = ltp
                reason = f"CE Entry: Near swing low ({position_in_range:.0f}%) + Rising (+{price_change_pct:.1f}%)"
                logger.info(f"SWING ENTRY CONFIRMED: {reason}")

            elif position_in_range < 50 and moving_with_signal:
                # In lower half and moving up - GOOD ENTRY
                entry_allowed = True
                entry_price = ltp
                reason = f"CE Entry: Lower range ({position_in_range:.0f}%) + Upward momentum"
                logger.info(f"SWING ENTRY: {reason}")

            elif current_close > swing_high * 0.99:  # Breaking out
                # Breaking above swing high - BREAKOUT ENTRY
                entry_allowed = True
                entry_price = ltp
                reason = f"CE Entry: Breakout above swing high ({swing_high:.2f})"
                logger.info(f"BREAKOUT ENTRY: {reason}")

            elif price_change < 0:
                # Price falling - WAIT for reversal
                entry_allowed = False
                reason = f"CE Wait: Price falling ({price_change_pct:.1f}%) - Wait for reversal"
                logger.info(f"SWING WAIT: {reason}")

            else:
                # Default - enter with caution
                entry_allowed = True
                entry_price = ltp
                reason = f"CE Entry: Standard entry at {position_in_range:.0f}% of range"

        else:  # PE signal
            # For PE signal:
            # - Don't enter if price is rising (opposite to signal)
            # - Wait for price to drop from swing high or break swing low
            # - Best entry: price near swing high and starting to fall

            if position_in_range > 70 and price_change < 0:
                # Near swing high and falling - BEST ENTRY
                entry_allowed = True
                entry_price = ltp
                reason = f"PE Entry: Near swing high ({position_in_range:.0f}%) + Falling ({price_change_pct:.1f}%)"
                logger.info(f"SWING ENTRY CONFIRMED: {reason}")

            elif position_in_range > 50 and moving_with_signal:
                # In upper half and moving down - GOOD ENTRY
                entry_allowed = True
                entry_price = ltp
                reason = f"PE Entry: Upper range ({position_in_range:.0f}%) + Downward momentum"
                logger.info(f"SWING ENTRY: {reason}")

            elif current_close < swing_low * 1.01:  # Breaking down
                # Breaking below swing low - BREAKDOWN ENTRY
                entry_allowed = True
                entry_price = ltp
                reason = f"PE Entry: Breakdown below swing low ({swing_low:.2f})"
                logger.info(f"BREAKDOWN ENTRY: {reason}")

            elif price_change > 0:
                # Price rising - WAIT for reversal
                entry_allowed = False
                reason = f"PE Wait: Price rising (+{price_change_pct:.1f}%) - Wait for reversal"
                logger.info(f"SWING WAIT: {reason}")

            else:
                # Default - enter with caution
                entry_allowed = True
                entry_price = ltp
                reason = f"PE Entry: Standard entry at {position_in_range:.0f}% of range"

        return {
            "entry_allowed": entry_allowed,
            "entry_price": entry_price,
            "reason": reason,
            "swing_high": swing_high,
            "swing_low": swing_low,
            "position_in_range": position_in_range,
            "price_change_pct": price_change_pct,
        }

    def find_best_entry_option(
        self,
        option_chain: list[dict] | None,
        signal_direction: str,
        spot_price: float,
    ) -> tuple[dict | None, str]:
        """
        Find the BEST option for entry with premium between 40-60.

        Advanced Selection Criteria:
        1. Premium must be between 40-60 (strict)
        2. Best bid-ask spread (lower is better)
        3. High OI (liquidity)
        4. Volume confirmation
        5. Near-optimal delta (0.4-0.6 for best movement)

        Returns:
            Tuple of (best_option_data, selection_reason)
        """
        if not option_chain:
            return None, "No option chain available"

        option_type = "ce" if signal_direction == "CE" else "pe"
        candidates = []

        PREMIUM_MIN = 40
        PREMIUM_MAX = 60
        MIN_OI = 10000

        for opt in option_chain:
            opt_data = opt.get(option_type, {})
            if not opt_data:
                continue

            ltp = opt_data.get("ltp", 0)
            bid = opt_data.get("bid", 0)
            ask = opt_data.get("ask", 0)
            oi = opt_data.get("oi", 0)
            volume = opt_data.get("volume", 0)
            strike = opt.get("strike", 0)

            # Skip if LTP not in optimal range
            if ltp < PREMIUM_MIN or ltp > PREMIUM_MAX:
                continue

            # Skip illiquid options
            if oi < MIN_OI:
                continue

            # Calculate spread
            spread = ask - bid if bid > 0 and ask > 0 else float('inf')
            spread_pct = (spread / ltp * 100) if ltp > 0 else 100

            # Calculate moneyness (how far from ATM)
            if signal_direction == "CE":
                moneyness = (spot_price - strike) / spot_price * 100  # Positive = ITM
            else:
                moneyness = (strike - spot_price) / spot_price * 100  # Positive = ITM

            # Estimate delta based on moneyness
            if moneyness > 2:  # ITM
                estimated_delta = 0.6 + min(moneyness / 10, 0.3)
            elif moneyness > -2:  # ATM
                estimated_delta = 0.5
            else:  # OTM
                estimated_delta = 0.5 - min(abs(moneyness) / 10, 0.3)

            candidates.append({
                "opt": opt,
                "opt_data": opt_data,
                "ltp": ltp,
                "bid": bid,
                "ask": ask,
                "oi": oi,
                "volume": volume,
                "strike": strike,
                "spread": spread,
                "spread_pct": spread_pct,
                "moneyness": moneyness,
                "estimated_delta": estimated_delta,
            })

        if not candidates:
            logger.warning(f"No options found in {PREMIUM_MIN}-{PREMIUM_MAX} premium range")
            return None, f"No options in {PREMIUM_MIN}-{PREMIUM_MAX} range"

        # Score candidates
        # Best: Low spread, good OI, optimal delta (0.4-0.6), premium near 50
        for c in candidates:
            # Spread score (lower is better) - weight: 30%
            spread_score = max(0, 1 - c["spread_pct"] / 5)  # 5% spread = 0 score

            # OI score (higher is better) - weight: 25%
            oi_score = min(c["oi"] / 500000, 1)

            # Delta score (closer to 0.5 is better) - weight: 25%
            delta_score = 1 - abs(c["estimated_delta"] - 0.5) * 2

            # Premium score (closer to 50 is better) - weight: 20%
            premium_score = 1 - abs(c["ltp"] - 50) / 10

            c["score"] = (
                spread_score * 0.30 +
                oi_score * 0.25 +
                delta_score * 0.25 +
                premium_score * 0.20
            )

        # Sort by score descending
        candidates.sort(key=lambda x: x["score"], reverse=True)
        best = candidates[0]

        reason = (
            f"Best entry: Strike {best['strike']}, Premium ₹{best['ltp']:.1f}, "
            f"Spread {best['spread_pct']:.1f}%, OI {best['oi']:,}, "
            f"Delta ~{best['estimated_delta']:.2f}, Score {best['score']:.2f}"
        )

        logger.info(f"SMART OPTION SELECTION: {reason}")

        return best, reason

    def analyze_price_trend_for_entry(
        self,
        option_chain: list[dict] | None,
        signal_direction: str,
        spot_price: float,
        index_df: "pd.DataFrame | None" = None,
    ) -> dict:
        """
        Analyze if option premium is likely to come down for better entry.

        INTELLIGENT PRICE WAITING LOGIC:
        - If option at 70, check if index movement suggests it will come to 45
        - For CE: Wait if index is falling (premium will drop)
        - For PE: Wait if index is rising (premium will drop)

        Returns:
            {
                "should_wait": bool,
                "wait_reason": str,
                "expected_entry_price": float,
                "current_premium": float,
                "price_direction": str (up/down/sideways),
                "wait_time_estimate": str
            }
        """
        result = {
            "should_wait": False,
            "wait_reason": "",
            "expected_entry_price": 0,
            "current_premium": 0,
            "price_direction": "sideways",
            "wait_time_estimate": "",
        }

        if not option_chain or index_df is None or len(index_df) < 10:
            return result

        option_type = "ce" if signal_direction == "CE" else "pe"

        # Find the best option that's currently too expensive (above 60)
        expensive_option = None
        for opt in option_chain:
            opt_data = opt.get(option_type, {})
            if not opt_data:
                continue

            ltp = opt_data.get("ltp", 0)
            oi = opt_data.get("oi", 0)

            # Looking for options above 60 that might come down
            if 60 < ltp <= 100 and oi >= 10000:
                if expensive_option is None or ltp < expensive_option["ltp"]:
                    expensive_option = {
                        "opt": opt,
                        "opt_data": opt_data,
                        "ltp": ltp,
                        "strike": opt.get("strike", 0),
                    }

        if not expensive_option:
            return result

        result["current_premium"] = expensive_option["ltp"]

        # Analyze index price trend
        closes = index_df["Close"].values
        highs = index_df["High"].values
        lows = index_df["Low"].values

        # Calculate momentum over last 5 and 10 candles
        momentum_5 = (closes[-1] - closes[-6]) / closes[-6] * 100 if len(closes) >= 6 else 0
        momentum_10 = (closes[-1] - closes[-11]) / closes[-11] * 100 if len(closes) >= 11 else 0

        # Calculate average candle range (volatility)
        avg_range = (highs[-10:] - lows[-10:]).mean()
        current_range = highs[-1] - lows[-1]

        # Determine price direction
        if momentum_5 > 0.1:
            result["price_direction"] = "up"
        elif momentum_5 < -0.1:
            result["price_direction"] = "down"
        else:
            result["price_direction"] = "sideways"

        # SMART WAITING LOGIC
        current_premium = expensive_option["ltp"]
        target_premium = 45  # Ideal entry price

        # For CE: Wait if index is falling (CE premium will drop)
        if signal_direction == "CE" and result["price_direction"] == "down":
            # Estimate how much premium might drop
            # Rough delta estimate: 0.5 for ATM options
            estimated_delta = 0.5
            expected_index_move = avg_range * 2  # 2 candles worth of movement
            expected_premium_drop = expected_index_move * estimated_delta

            expected_premium = current_premium - expected_premium_drop

            if expected_premium <= 60:  # Will come into range
                result["should_wait"] = True
                result["wait_reason"] = (
                    f"Index falling (momentum {momentum_5:.2f}%), CE premium at ₹{current_premium:.0f} "
                    f"expected to drop to ₹{expected_premium:.0f}. Wait for better entry."
                )
                result["expected_entry_price"] = max(target_premium, expected_premium)
                result["wait_time_estimate"] = "2-3 candles (~10-15 min)"

        # For PE: Wait if index is rising (PE premium will drop)
        elif signal_direction == "PE" and result["price_direction"] == "up":
            estimated_delta = 0.5
            expected_index_move = avg_range * 2
            expected_premium_drop = expected_index_move * estimated_delta

            expected_premium = current_premium - expected_premium_drop

            if expected_premium <= 60:
                result["should_wait"] = True
                result["wait_reason"] = (
                    f"Index rising (momentum +{momentum_5:.2f}%), PE premium at ₹{current_premium:.0f} "
                    f"expected to drop to ₹{expected_premium:.0f}. Wait for better entry."
                )
                result["expected_entry_price"] = max(target_premium, expected_premium)
                result["wait_time_estimate"] = "2-3 candles (~10-15 min)"

        # If price is moving WITH the signal direction, premium will increase - don't wait
        if signal_direction == "CE" and result["price_direction"] == "up":
            result["should_wait"] = False
            result["wait_reason"] = "Index rising - CE premium will increase. Enter now at best available."

        if signal_direction == "PE" and result["price_direction"] == "down":
            result["should_wait"] = False
            result["wait_reason"] = "Index falling - PE premium will increase. Enter now at best available."

        logger.info(
            f"PRICE TREND ANALYSIS: Direction={signal_direction}, "
            f"IndexTrend={result['price_direction']}, Premium=₹{current_premium:.0f}, "
            f"ShouldWait={result['should_wait']}, Reason={result['wait_reason']}"
        )

        return result

    def find_optimal_entry_with_waiting(
        self,
        option_chain: list[dict] | None,
        signal_direction: str,
        spot_price: float,
        index_df: "pd.DataFrame | None" = None,
    ) -> tuple[dict | None, bool, str]:
        """
        Find optimal entry option with intelligent price waiting.

        MAIN ENTRY LOGIC:
        1. First check if any option is in 40-60 range - if yes, use it
        2. If option is above 60, analyze if it will come down
        3. If likely to come down, signal to WAIT
        4. If unlikely to come down, find the nearest option to 60

        Returns:
            Tuple of (best_option, should_wait, reason)
        """
        PREMIUM_MIN = 40
        PREMIUM_MAX = 60

        # Step 1: Try to find option in optimal range
        best_in_range, range_reason = self.find_best_entry_option(
            option_chain, signal_direction, spot_price
        )

        if best_in_range:
            # Found option in 40-60 range - no need to wait
            return best_in_range, False, f"✓ {range_reason}"

        # Step 2: No option in range - analyze if we should wait
        price_analysis = self.analyze_price_trend_for_entry(
            option_chain, signal_direction, spot_price, index_df
        )

        if price_analysis["should_wait"]:
            # Premium likely to come down - wait
            return None, True, f"⏳ WAIT: {price_analysis['wait_reason']}"

        # Step 3: Not waiting - find the nearest option to our range
        # Look for option closest to 60 (just above our range)
        option_type = "ce" if signal_direction == "CE" else "pe"
        nearest_option = None
        nearest_diff = float('inf')

        for opt in option_chain or []:
            opt_data = opt.get(option_type, {})
            if not opt_data:
                continue

            ltp = opt_data.get("ltp", 0)
            oi = opt_data.get("oi", 0)

            if oi < 10000:  # Skip illiquid
                continue

            # Find option closest to 60 (preferring slightly above)
            if PREMIUM_MAX < ltp <= 80:  # Within acceptable extended range
                diff = ltp - PREMIUM_MAX
                if diff < nearest_diff:
                    nearest_diff = diff
                    nearest_option = {
                        "opt": opt,
                        "opt_data": opt_data,
                        "ltp": ltp,
                        "bid": opt_data.get("bid", ltp),
                        "ask": opt_data.get("ask", ltp),
                        "oi": oi,
                        "volume": opt_data.get("volume", 0),
                        "strike": opt.get("strike", 0),
                        "spread_pct": 0,
                        "estimated_delta": 0.5,
                        "score": 0.5,
                    }

        if nearest_option:
            reason = (
                f"⚡ Entry at ₹{nearest_option['ltp']:.0f} (above optimal range). "
                f"No wait - {price_analysis.get('wait_reason', 'momentum with signal')}"
            )
            return nearest_option, False, reason

        # No suitable option found
        return None, False, "❌ No suitable option found in any range"

    def calculate_optimal_entry_price(
        self,
        opt_data: dict,
        signal_direction: str,
        index_df: "pd.DataFrame | None" = None,
    ) -> tuple[float, str]:
        """
        Calculate the OPTIMAL entry price for an option.

        Strategy:
        1. Use bid-ask spread analysis
        2. Check momentum direction for better timing
        3. For CE: Enter on pullbacks (bid side), For PE: Enter on rallies (ask side)

        Returns:
            Tuple of (optimal_entry_price, entry_reason)
        """
        ltp = opt_data.get("ltp", 0)
        bid = opt_data.get("bid", 0)
        ask = opt_data.get("ask", 0)

        if bid <= 0 or ask <= 0:
            return ltp, "Using LTP (no bid/ask)"

        spread = ask - bid
        spread_pct = (spread / ltp * 100) if ltp > 0 else 0
        mid_price = (bid + ask) / 2

        # Check recent price momentum from index
        momentum = "neutral"
        if index_df is not None and len(index_df) >= 3:
            recent_closes = index_df["Close"].tail(3).values
            price_change = recent_closes[-1] - recent_closes[0]
            if price_change > 0:
                momentum = "up"
            elif price_change < 0:
                momentum = "down"

        # SMART ENTRY LOGIC
        if spread_pct < 1:
            # Very tight spread - use mid price
            entry_price = mid_price
            reason = f"Tight spread ({spread_pct:.1f}%), using mid {mid_price:.2f}"
        elif spread_pct < 3:
            # Normal spread - use optimal point
            if signal_direction == "CE":
                if momentum == "down":
                    # Price falling - wait for lower bid, use bid + small offset
                    entry_price = bid + (spread * 0.2)
                    reason = f"CE on pullback, entry near bid {entry_price:.2f}"
                else:
                    # Price rising - may need to pay more
                    entry_price = bid + (spread * 0.4)
                    reason = f"CE with momentum, entry at {entry_price:.2f}"
            else:  # PE
                if momentum == "up":
                    # Price rising - wait for lower bid, use bid + small offset
                    entry_price = bid + (spread * 0.2)
                    reason = f"PE on rally, entry near bid {entry_price:.2f}"
                else:
                    # Price falling - may need to pay more
                    entry_price = bid + (spread * 0.4)
                    reason = f"PE with momentum, entry at {entry_price:.2f}"
        else:
            # Wide spread - be very careful, use bid + small offset
            entry_price = bid + (spread * 0.15)
            reason = f"Wide spread ({spread_pct:.1f}%), cautious entry at {entry_price:.2f}"

        # Ensure entry is within bid-ask
        entry_price = max(bid, min(ask, entry_price))

        logger.info(f"OPTIMAL ENTRY: LTP={ltp:.2f}, Bid={bid:.2f}, Ask={ask:.2f}, Entry={entry_price:.2f} | {reason}")

        return round(entry_price, 2), reason

    def calculate_smart_entry_price(
        self,
        ltp: float,
        signal_direction: str,
        option_chain: list[dict] | None = None,
        index_df: "pd.DataFrame | None" = None,
    ) -> tuple[float, float, float, bool, str]:
        """
        Calculate smart entry price based on swing trading analysis.

        Strategy:
        1. Analyze last 30 minutes of index price movement
        2. For default strategy: wait for reversal confirmation
        3. For aggressive strategies (5%, 15min, 20%, 100%): skip swing wait, always allow entry
        4. Use bid-ask spread and delta for precise entry/SL/target

        Args:
            ltp: Current last traded price
            signal_direction: CE or PE
            option_chain: Option chain data with bid/ask/greeks
            index_df: DataFrame with index price data for swing analysis

        Returns:
            Tuple of (entry_price, stop_loss, target, entry_allowed, reason)
        """
        entry_price = ltp
        entry_allowed = True
        entry_reason = ""

        # Step 1: Swing Trading Analysis (only for default strategy - conservative)
        if index_df is not None and len(index_df) >= 6 and self.strategy == "default":
            swing_analysis = self.analyze_swing_entry(index_df, signal_direction, ltp)
            entry_allowed = swing_analysis["entry_allowed"]
            entry_reason = swing_analysis["reason"]

            if not entry_allowed:
                # Return early - don't enter yet (only for default strategy)
                return ltp, 0, 0, False, entry_reason
        elif index_df is not None and len(index_df) >= 6 and self.strategy != "default":
            # For aggressive strategies, ignore swing wait - just log the analysis
            swing_analysis = self.analyze_swing_entry(index_df, signal_direction, ltp)
            entry_reason = swing_analysis["reason"] + " [Aggressive: ignoring swing wait]"
            logger.debug(f"Swing analysis for {self.strategy}: {swing_analysis['reason']} - proceeding anyway")

        # Step 2: Calculate entry from option chain data
        if option_chain:
            # Find the option in the chain
            opt_data = None
            for opt in option_chain:
                ce_data = opt.get("ce", {})
                pe_data = opt.get("pe", {})

                if signal_direction == "CE" and abs(ce_data.get("ltp", 0) - ltp) < 1:
                    opt_data = ce_data
                    break
                elif signal_direction == "PE" and abs(pe_data.get("ltp", 0) - ltp) < 1:
                    opt_data = pe_data
                    break

            if opt_data:
                # Use OPTIMAL entry price calculation
                entry_price, price_reason = self.calculate_optimal_entry_price(
                    opt_data, signal_direction, index_df
                )
                entry_reason = f"{entry_reason} | {price_reason}" if entry_reason else price_reason

                # Get delta for SL calculation
                delta = abs(opt_data.get("delta", 0.5))

                # Dynamic SL based on delta
                if delta >= 0.6:
                    sl_pct = 0.12  # 12% for high delta (ITM)
                elif delta >= 0.4:
                    sl_pct = 0.15  # 15% for medium delta (ATM)
                else:
                    sl_pct = 0.20  # 20% for low delta (OTM)

                stop_loss = entry_price * (1 - sl_pct)

                # Target: 1:2.5 risk/reward
                risk = entry_price - stop_loss
                target = entry_price + (risk * 2.5)

                logger.info(
                    f"Smart Entry: LTP={ltp:.2f}, Entry={entry_price:.2f}, "
                    f"SL={stop_loss:.2f} ({sl_pct*100:.0f}%), Target={target:.2f}, "
                    f"Delta={delta:.2f}"
                )

                return round(entry_price, 2), round(stop_loss, 2), round(target, 2), True, entry_reason

        # Step 3: Fallback calculation based on price level
        OPTIMAL_MIN = 40
        OPTIMAL_MAX = 60

        if OPTIMAL_MIN <= ltp <= OPTIMAL_MAX:
            sl_pct = 0.15
            target_pct = 0.30
        elif ltp < OPTIMAL_MIN:
            sl_pct = 0.20
            target_pct = 0.50
        else:
            sl_pct = 0.12
            target_pct = 0.25

        stop_loss = entry_price * (1 - sl_pct)
        target = entry_price * (1 + target_pct)

        return round(entry_price, 2), round(stop_loss, 2), round(target, 2), True, entry_reason or "Fallback entry"

    def calculate_trailing_stop_loss(
        self,
        position: "PaperPosition",
        current_price: float,
    ) -> tuple[float, str]:
        """
        Calculate trailing stop loss based on profit movement.

        Rules:
        - Initial SL from signal is baseline
        - When profit reaches 50%, move SL to entry (breakeven)
        - For every 10% profit move after that, trail SL by 10%
        - Example: 60% profit -> SL at 50%, 70% profit -> SL at 60%

        Args:
            position: Current position
            current_price: Current market price

        Returns:
            Tuple of (new_stop_loss, reason_for_change)
        """
        entry_price = position.entry_price
        current_sl = position.stop_loss

        # Calculate current profit %
        profit_percent = ((current_price - entry_price) / entry_price) * 100

        # If in loss, keep original SL
        if profit_percent <= 0:
            return current_sl, ""

        # Calculate new SL based on profit level
        new_sl = current_sl
        reason = ""

        # Rule 1: At 50% profit, move SL to entry (breakeven)
        if profit_percent >= 50 and not position.trailing_sl_active:
            new_sl = entry_price
            reason = f"BREAKEVEN: +{profit_percent:.0f}% profit, SL moved to entry {entry_price:.2f}"
            position.trailing_sl_active = True
            position.profit_locked_percent = 0
            logger.info(reason)

        # Rule 2: For every 10% above 50%, trail SL by 10%
        elif profit_percent >= 50 and position.trailing_sl_active:
            # Calculate how many 10% increments above 50%
            increments_above_50 = int((profit_percent - 50) / 10)
            locked_profit_percent = increments_above_50 * 10  # 0, 10, 20, 30...

            if locked_profit_percent > position.profit_locked_percent:
                # Calculate new SL price
                # SL should be at (locked_profit_percent)% above entry
                new_sl = entry_price * (1 + locked_profit_percent / 100)
                reason = f"TRAIL SL: +{profit_percent:.0f}% profit, SL moved to +{locked_profit_percent}% ({new_sl:.2f})"
                position.profit_locked_percent = locked_profit_percent
                position.sl_trail_count += 1
                logger.info(reason)

        # Rule 3: When target is achieved, set SL at target and wait
        if not position.target_achieved and current_price >= position.target:
            position.target_achieved = True
            new_sl = max(new_sl, position.target)  # SL at least at target
            reason = f"TARGET HIT: Price {current_price:.2f} >= Target {position.target:.2f}, SL set at {new_sl:.2f}"
            logger.info(reason)

        return round(new_sl, 2), reason

    def is_trading_hours(self, trading_index: "ExpiryInfo | None" = None) -> tuple[bool, str]:
        """
        Check if current time is within trading hours.

        Trading Hours:
        - Normal Day: 9:30 AM to 3:00 PM
        - Expiry Day: 9:30 AM to 3:00 PM (same, we handle exit separately)

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

        # Trading close time: 3:00 PM
        close_hour = 15
        close_minute = 0
        close_time_str = "3:00 PM"

        # After trading close time
        if current_hour > close_hour or (current_hour == close_hour and current_minute >= close_minute):
            return False, f"Trading closed for the day (Closes at {close_time_str})"

        return True, ""

    def is_expiry_day_power_hour(self, trading_index: "ExpiryInfo | None" = None) -> tuple[bool, str]:
        """
        Check if we are in the EXPIRY DAY POWER HOUR (1 PM - 3 PM).

        This is the "DO OR DIE" period for maximum profit trades.
        On expiry day, between 1-3 PM:
        - Options decay rapidly (theta crush)
        - Directional moves are amplified
        - Best time for aggressive trades

        Returns:
            Tuple of (is_power_hour, description)
        """
        if trading_index is None or not trading_index.is_expiry_day:
            return False, "Not expiry day"

        now = datetime.now()
        current_hour = now.hour
        current_minute = now.minute

        # Power hour: 1 PM to 3 PM (13:00 to 15:00)
        if 13 <= current_hour < 15:
            time_remaining = (15 - current_hour) * 60 - current_minute
            return True, f"🔥 EXPIRY POWER HOUR: {time_remaining} min to close"

        # Before power hour
        if current_hour < 13:
            mins_to_power = (13 - current_hour) * 60 - current_minute
            return False, f"Power hour starts in {mins_to_power} min"

        # After power hour
        return False, "Power hour ended"

    def get_expiry_day_strategy(
        self,
        trading_index: "ExpiryInfo | None",
        signal_direction: str,
        current_premium: float,
    ) -> dict:
        """
        Get AGGRESSIVE expiry day strategy for maximum profit.

        EXPIRY DAY "DO OR DIE" LOGIC:
        - Between 1-3 PM on expiry day
        - Options decay to zero by 3:30 PM
        - Take high conviction directional bets
        - Tight SL (5-8%), Large Target (50-100%)
        - Exit at 3:00 PM regardless of P&L

        Returns:
            {
                "is_active": bool,
                "sl_percent": float,
                "target_percent": float,
                "max_hold_time": int (minutes),
                "strategy_name": str,
                "risk_warning": str
            }
        """
        result = {
            "is_active": False,
            "sl_percent": 0.15,  # Default 15%
            "target_percent": 1.0,  # Default 100%
            "max_hold_time": 120,  # Default 2 hours
            "strategy_name": "Normal",
            "risk_warning": "",
        }

        is_power_hour, power_desc = self.is_expiry_day_power_hour(trading_index)

        if not is_power_hour:
            return result

        now = datetime.now()
        mins_to_close = (15 - now.hour) * 60 - now.minute

        # ========================================
        # EXPIRY DAY POWER HOUR STRATEGY
        # ========================================

        result["is_active"] = True
        result["strategy_name"] = "🔥 EXPIRY POWER HOUR"

        # Time-based aggressive settings
        if mins_to_close > 90:
            # 1:00 PM - 1:30 PM: Moderate aggressive
            result["sl_percent"] = 0.10  # 10% SL
            result["target_percent"] = 0.50  # 50% target
            result["max_hold_time"] = 60
            result["risk_warning"] = "Early power hour - moderate aggression"

        elif mins_to_close > 60:
            # 1:30 PM - 2:00 PM: High aggressive
            result["sl_percent"] = 0.08  # 8% SL
            result["target_percent"] = 0.40  # 40% target
            result["max_hold_time"] = 45
            result["risk_warning"] = "Mid power hour - high aggression"

        elif mins_to_close > 30:
            # 2:00 PM - 2:30 PM: Very aggressive
            result["sl_percent"] = 0.06  # 6% SL (tight)
            result["target_percent"] = 0.30  # 30% target
            result["max_hold_time"] = 25
            result["risk_warning"] = "⚠️ Late power hour - very aggressive, theta decay accelerating"

        else:
            # 2:30 PM - 3:00 PM: ULTRA aggressive (final 30 mins)
            result["sl_percent"] = 0.05  # 5% SL (very tight)
            result["target_percent"] = 0.20  # 20% target (quick scalp)
            result["max_hold_time"] = 15
            result["risk_warning"] = "🚨 FINAL 30 MIN - Ultra aggressive, quick scalps only"

        # Premium-based adjustment
        if current_premium < 20:
            # Very cheap premium - can be more aggressive
            result["sl_percent"] = result["sl_percent"] * 1.5  # Slightly wider SL for cheap options
            result["target_percent"] = result["target_percent"] * 2  # Higher target for cheap options
            result["risk_warning"] += f" | Cheap premium (₹{current_premium:.0f}) - high R:R"

        elif current_premium > 40:
            # Expensive premium - be careful
            result["sl_percent"] = result["sl_percent"] * 0.8  # Tighter SL
            result["risk_warning"] += f" | Premium at ₹{current_premium:.0f} - tighter SL"

        logger.info(
            f"EXPIRY STRATEGY: {result['strategy_name']} | "
            f"SL={result['sl_percent']*100:.0f}%, Target={result['target_percent']*100:.0f}%, "
            f"MaxHold={result['max_hold_time']}min | {result['risk_warning']}"
        )

        return result

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

    async def dynamic_sl_target_update(
        self,
        position: PaperPosition,
        current_premium: float,
        index_df: "pd.DataFrame | None" = None,
    ) -> tuple[bool, str]:
        """
        DYNAMIC SL/TARGET UPDATE based on market direction.

        LOGIC:
        - If market continues in signal direction, move SL to breakeven and let target run
        - If market shows reversal signs, tighten SL to lock profits
        - Check every position update if signal direction has strengthened

        Returns:
            Tuple of (was_updated, update_reason)
        """
        if position.status != PositionStatus.OPEN:
            return False, "Position not open"

        profit_pct = ((current_premium - position.entry_price) / position.entry_price) * 100
        was_updated = False
        update_reasons = []

        # ========================================
        # RULE 1: LOCK PROFIT AT MILESTONES
        # ========================================

        # At 20% profit - move SL to breakeven
        if profit_pct >= 20 and position.stop_loss < position.entry_price:
            old_sl = position.stop_loss
            position.stop_loss = position.entry_price
            was_updated = True
            update_reasons.append(f"20% milestone: SL moved to breakeven ({old_sl:.2f} → {position.entry_price:.2f})")
            logger.info(f"🔒 BREAKEVEN LOCK: {position.symbol} @ +{profit_pct:.1f}% profit")

        # At 30% profit - lock 10% profit
        if profit_pct >= 30 and position.stop_loss < position.entry_price * 1.10:
            old_sl = position.stop_loss
            position.stop_loss = position.entry_price * 1.10
            was_updated = True
            update_reasons.append(f"30% milestone: SL locked at +10% ({old_sl:.2f} → {position.stop_loss:.2f})")
            logger.info(f"🔒 PROFIT LOCK +10%: {position.symbol}")

        # At 50% profit - lock 25% profit
        if profit_pct >= 50 and position.stop_loss < position.entry_price * 1.25:
            old_sl = position.stop_loss
            position.stop_loss = position.entry_price * 1.25
            was_updated = True
            update_reasons.append(f"50% milestone: SL locked at +25% ({old_sl:.2f} → {position.stop_loss:.2f})")
            logger.info(f"🔒 PROFIT LOCK +25%: {position.symbol}")

        # At 75% profit - lock 50% profit
        if profit_pct >= 75 and position.stop_loss < position.entry_price * 1.50:
            old_sl = position.stop_loss
            position.stop_loss = position.entry_price * 1.50
            was_updated = True
            update_reasons.append(f"75% milestone: SL locked at +50% ({old_sl:.2f} → {position.stop_loss:.2f})")
            logger.info(f"🔒 PROFIT LOCK +50%: {position.symbol}")

        # At 100% profit - lock 75% profit
        if profit_pct >= 100 and position.stop_loss < position.entry_price * 1.75:
            old_sl = position.stop_loss
            position.stop_loss = position.entry_price * 1.75
            was_updated = True
            update_reasons.append(f"100% milestone: SL locked at +75% ({old_sl:.2f} → {position.stop_loss:.2f})")
            logger.info(f"🔒 PROFIT LOCK +75%: {position.symbol}")

        # ========================================
        # RULE 2: EXTEND TARGET IF MOMENTUM CONTINUES
        # ========================================
        if index_df is not None and len(index_df) >= 5:
            closes = index_df["Close"].values
            momentum = closes[-1] - closes[-5]  # 5-candle momentum

            # For CE: Upward momentum = extend target
            if position.option_type == "CE" and momentum > 0:
                # Market continuing up - extend target
                if profit_pct >= 30 and not position.target_achieved:
                    new_target = current_premium * 1.30  # 30% more from current
                    if new_target > position.target:
                        old_target = position.target
                        position.target = new_target
                        was_updated = True
                        update_reasons.append(f"CE momentum continuing: Target extended ({old_target:.2f} → {new_target:.2f})")
                        logger.info(f"📈 TARGET EXTENDED: {position.symbol} - Market momentum continuing UP")

            # For PE: Downward momentum = extend target
            if position.option_type == "PE" and momentum < 0:
                # Market continuing down - extend target
                if profit_pct >= 30 and not position.target_achieved:
                    new_target = current_premium * 1.30
                    if new_target > position.target:
                        old_target = position.target
                        position.target = new_target
                        was_updated = True
                        update_reasons.append(f"PE momentum continuing: Target extended ({old_target:.2f} → {new_target:.2f})")
                        logger.info(f"📈 TARGET EXTENDED: {position.symbol} - Market momentum continuing DOWN")

        # ========================================
        # RULE 3: TIGHTEN SL IF REVERSAL DETECTED
        # ========================================
        if index_df is not None and len(index_df) >= 3:
            closes = index_df["Close"].values
            short_momentum = closes[-1] - closes[-3]  # 3-candle momentum

            # For CE: Downward short-term momentum = potential reversal
            if position.option_type == "CE" and short_momentum < 0 and profit_pct > 10:
                # Tighten SL to lock current profit
                min_profit_lock = profit_pct * 0.5  # Lock 50% of current profit
                target_sl = position.entry_price * (1 + min_profit_lock / 100)

                if target_sl > position.stop_loss:
                    old_sl = position.stop_loss
                    position.stop_loss = target_sl
                    was_updated = True
                    update_reasons.append(f"CE reversal detected: SL tightened to lock {min_profit_lock:.0f}% ({old_sl:.2f} → {target_sl:.2f})")
                    logger.warning(f"⚠️ REVERSAL DETECTED: {position.symbol} - Tightening SL")

            # For PE: Upward short-term momentum = potential reversal
            if position.option_type == "PE" and short_momentum > 0 and profit_pct > 10:
                min_profit_lock = profit_pct * 0.5
                target_sl = position.entry_price * (1 + min_profit_lock / 100)

                if target_sl > position.stop_loss:
                    old_sl = position.stop_loss
                    position.stop_loss = target_sl
                    was_updated = True
                    update_reasons.append(f"PE reversal detected: SL tightened to lock {min_profit_lock:.0f}% ({old_sl:.2f} → {target_sl:.2f})")
                    logger.warning(f"⚠️ REVERSAL DETECTED: {position.symbol} - Tightening SL")

        if was_updated:
            self._save_state()

        reason = " | ".join(update_reasons) if update_reasons else "No update needed"
        return was_updated, reason

    async def execute_signal_trade(
        self,
        signal: Any,
        trading_index: ExpiryInfo,
        index_df: Any = None,
    ) -> PaperOrder | None:
        """
        Execute a trade based on signal with swing entry confirmation.

        Args:
            signal: TradeSignal from signal engine
            trading_index: ExpiryInfo for the trading index
            index_df: DataFrame with index price data for swing analysis

        Returns:
            PaperOrder if executed, None otherwise
        """
        # Check trading hours first
        # Check trading hours (can be bypassed for testing)
        from app.core.config import get_settings
        settings = get_settings()
        bypass_hours = getattr(settings, 'bypass_market_hours', False)

        is_trading, reason = self.is_trading_hours(trading_index)
        logger.info(f"[{self.strategy}] Paper trading hours check: allowed={is_trading}, reason={reason}, bypass={bypass_hours}")

        if not is_trading and not bypass_hours:
            logger.warning(f"[{self.strategy}] TRADE BLOCKED: Outside trading hours: {reason}")
            return None
        elif not is_trading and bypass_hours:
            logger.warning(f"[{self.strategy}] Trading hours check would fail ({reason}), but bypassing for testing")

        # Check if trading is halted
        if self.check_daily_loss_limit():
            logger.warning(f"[{self.strategy}] TRADE BLOCKED: Trading halted due to daily loss limit")
            return None

        # Check signal direction
        if signal.direction not in ["CE", "PE"]:
            logger.warning(f"[{self.strategy}] TRADE BLOCKED: No clear signal direction (got: {signal.direction})")
            return None

        logger.info(f"[{self.strategy}] Signal direction check passed: {signal.direction}")

        # Check confidence threshold (at least 60%)
        if signal.confidence < 60:
            logger.warning(f"[{self.strategy}] TRADE BLOCKED: Signal confidence too low: {signal.confidence}% (need 60%+)")
            return None

        logger.info(f"[{self.strategy}] Confidence check passed: {signal.confidence}%")

        # Check if this is a pre-market signal (direction only, no trade)
        if hasattr(signal, 'is_pre_market') and signal.is_pre_market:
            logger.info(f"[{self.strategy}] PRE-MARKET SIGNAL: Direction={signal.direction}, Confidence={signal.confidence}%")
            logger.info(f"[{self.strategy}] {signal.market_direction_note if hasattr(signal, 'market_direction_note') else 'Wait for 9:30 AM to trade'}")
            logger.warning(f"[{self.strategy}] TRADE BLOCKED: Pre-market signal - no trades before 9:30 AM")
            return None

        # Get recommended option
        if not signal.recommended_option:
            logger.warning(f"[{self.strategy}] TRADE BLOCKED: No recommended option in signal")
            return None

        opt = signal.recommended_option
        logger.info(f"[{self.strategy}] Signal option: {opt.strike} {signal.direction} @ Rs.{opt.ltp:.2f}")

        # Check if there's already an open position
        if self.has_open_position():
            # FIXED SL/TARGET: Don't update existing position targets
            # Once signal generates SL/Target, they are LOCKED
            existing_position = self.find_similar_position(signal.direction)
            if existing_position:
                logger.warning(f"[{self.strategy}] TRADE BLOCKED: Position already open: {existing_position.symbol} - SL/Target LOCKED (no updates)")
            else:
                logger.warning(f"[{self.strategy}] TRADE BLOCKED: Position already open with different direction. Skipping new order.")
            return None

        logger.info(f"[{self.strategy}] All pre-checks passed. Proceeding with order execution...")

        # Check for reversal signal - for default strategy only (conservative)
        # For aggressive strategies (5%, 15min, 20%, 100%), skip reversal check entirely
        if signal.is_reversal_signal and self.strategy == "default":
            logger.info(f"REVERSAL SIGNAL DETECTED: {signal.reversal_reason}")
            logger.info("Waiting for reversal confirmation before entry...")
            return None

        # Log if reversal detected on aggressive strategy (for debugging)
        if signal.is_reversal_signal and self.strategy != "default":
            logger.debug(f"Reversal detected on {self.strategy} but proceeding (aggressive strategy ignores reversals)")

        # Get option chain for smart entry calculation
        chain_data = await self.data_fetcher.get_option_chain(index=trading_index.index)
        option_chain = chain_data.get("chain", []) if "error" not in chain_data else None

        # Get current spot price
        spot_price = index_df["Close"].iloc[-1] if index_df is not None and len(index_df) > 0 else 0

        # ========================================
        # SMART ENTRY WITH INTELLIGENT WAITING
        # ========================================
        best_option, should_wait, selection_reason = self.find_optimal_entry_with_waiting(
            option_chain=option_chain,
            signal_direction=signal.direction,
            spot_price=spot_price,
            index_df=index_df,
        )

        if should_wait:
            logger.info(f"[{self.strategy}] {selection_reason}")
            return None

        # Use the best option found (either from our search or the signal's recommended option)
        if best_option:
            # Override with our optimally selected option
            opt_ltp = best_option["ltp"]
            opt_bid = best_option.get("bid", opt_ltp)
            opt_ask = best_option.get("ask", opt_ltp)
            opt_strike = best_option["strike"]
            opt_symbol = best_option["opt_data"].get("symbol", f"{opt_strike}{signal.direction}")

            logger.info(f"[{self.strategy}] SMART SELECTION: {selection_reason}")
        else:
            # Fallback to signal's recommended option
            opt_ltp = opt.ltp
            opt_bid = opt.bid
            opt_ask = opt.ask
            opt_strike = opt.strike
            opt_symbol = opt.symbol
            selection_reason = "Using signal's recommended option"

        # Calculate smart entry price with swing analysis
        entry_price, smart_sl, smart_target, entry_allowed, entry_reason = self.calculate_smart_entry_price(
            ltp=opt_ltp,
            signal_direction=signal.direction,
            option_chain=option_chain,
            index_df=index_df,
        )

        if not entry_allowed:
            logger.info(f"SWING WAIT: {entry_reason}")
            return None

        # Use smart entry values if valid, otherwise use signal values
        if smart_sl > 0 and smart_target > 0:
            stop_loss = smart_sl
            target = smart_target
            logger.info(f"Smart Entry: Price={entry_price:.2f}, SL={stop_loss:.2f}, Target={target:.2f}")
        else:
            stop_loss = signal.stop_loss
            target = signal.target_1
            logger.info(f"Signal Entry: Price={entry_price:.2f}, SL={stop_loss:.2f}, Target={target:.2f}")

        # ========================================
        # EXPIRY DAY POWER HOUR (1-3 PM) - DO OR DIE
        # Override SL/Target for aggressive expiry trades
        # ========================================
        expiry_strategy = self.get_expiry_day_strategy(
            trading_index=trading_index,
            signal_direction=signal.direction,
            current_premium=entry_price,
        )

        if expiry_strategy["is_active"]:
            # Apply aggressive expiry day settings
            stop_loss = entry_price * (1 - expiry_strategy["sl_percent"])
            target = entry_price * (1 + expiry_strategy["target_percent"])

            logger.info(
                f"🔥 EXPIRY POWER HOUR TRADE: "
                f"Entry={entry_price:.2f}, SL={stop_loss:.2f} ({expiry_strategy['sl_percent']*100:.0f}%), "
                f"Target={target:.2f} ({expiry_strategy['target_percent']*100:.0f}%) | "
                f"MaxHold={expiry_strategy['max_hold_time']}min | {expiry_strategy['risk_warning']}"
            )
            entry_reason = f"{entry_reason} | {expiry_strategy['strategy_name']}"

        logger.info(f"Entry Reason: {entry_reason}")

        # Calculate order size using entry price
        lots, quantity, split_orders = self.calculate_order_size(
            price=entry_price,
            lot_size=trading_index.lot_size,
        )

        if lots == 0:
            logger.warning("Insufficient capital for trade")
            return None

        # Create order with smart entry price (using smart selected option)
        order = PaperOrder(
            order_id=self._generate_order_id(),
            timestamp=datetime.now(),
            index=trading_index.index,
            symbol=opt_symbol,  # Use smart selected symbol
            strike=opt_strike,  # Use smart selected strike
            option_type=signal.direction,
            order_type=OrderType.BUY,
            quantity=quantity,
            lots=lots,
            price=entry_price,
            status=OrderStatus.EXECUTED,
            executed_quantity=quantity,
            executed_price=entry_price,
            split_orders=split_orders,
            reason=f"Signal: {signal.signal_type.value}, Confidence: {signal.confidence}% | {selection_reason} | {entry_reason}",
            signal_confidence=signal.confidence,
        )

        # Create position with smart entry values
        now = datetime.now()
        position = PaperPosition(
            position_id=self._generate_position_id(),
            index=trading_index.index,
            symbol=opt_symbol,  # Use smart selected symbol
            strike=opt_strike,  # Use smart selected strike
            option_type=signal.direction,
            entry_price=entry_price,
            quantity=quantity,
            lots=lots,
            entry_time=now,
            current_price=entry_price,
            status=PositionStatus.OPEN,
            stop_loss=stop_loss,
            target=target,
            max_price=entry_price,  # Initialize to entry price
            min_price=entry_price,  # Initialize to entry price
            max_price_time=now,
            min_price_time=now,
            # Store initial values for trailing SL
            initial_stop_loss=stop_loss,
            initial_target=target,
        )

        # Keep current_capital fixed at starting_capital (no profit reinvestment)
        # P&L is tracked separately in realized_pnl and total_pnl
        self.daily_stats.total_trades += 1

        # Add to lists
        self.orders.append(order)
        self.positions.append(position)

        # Save state
        self._save_state()

        logger.info(f"Executed BUY order: {order.order_id} - {order.symbol} x {order.lots} lots @ {order.price}")

        return order

    async def _check_next_signal_and_update(self, position: PaperPosition, exit_reason: str) -> bool:
        """
        Check if next signal is in same direction as current position.
        If yes, update SL/target instead of exiting.

        Args:
            position: Current position about to exit
            exit_reason: Reason for exit

        Returns:
            True if position was updated (don't exit), False if should exit normally
        """
        try:
            from app.services.signal_engine import get_signal_engine, TradingStyle
            from app.core.config import NIFTY_INDEX_TOKEN, BANKNIFTY_INDEX_TOKEN, SENSEX_INDEX_TOKEN

            logger.info(f"Checking next signal before exiting {position.symbol}...")

            # Get token for index
            tokens = {
                "NIFTY": NIFTY_INDEX_TOKEN,
                "BANKNIFTY": BANKNIFTY_INDEX_TOKEN,
                "SENSEX": SENSEX_INDEX_TOKEN,
            }
            token = tokens.get(position.index, NIFTY_INDEX_TOKEN)

            # Fetch historical data for signal generation
            df = await self.data_fetcher.fetch_historical_data(
                instrument_token=token,
                timeframe="5minute",
                days=3,
            )

            if df.empty:
                logger.warning("No historical data for signal generation, proceeding with exit")
                return False

            # Get option chain
            chain_data = await self.data_fetcher.get_option_chain(index=position.index)
            option_chain = chain_data.get("chain", []) if "error" not in chain_data else None

            # Generate new signal
            engine = get_signal_engine(TradingStyle.INTRADAY)
            signal = engine.analyze(df=df, option_chain=option_chain)

            if not signal:
                logger.info("No new signal generated, proceeding with exit")
                return False

            # Check if signal confidence is above 60%
            if signal.confidence < 60:
                logger.info(f"New signal confidence {signal.confidence}% < 60%, proceeding with exit")
                return False

            # Check if signal direction matches current position
            if signal.direction != position.option_type:
                logger.info(f"New signal direction {signal.direction} != position {position.option_type}, proceeding with exit")
                return False

            # Same direction and high confidence! Update SL/target instead of exiting
            logger.info(f"✓ SAME DIRECTION signal found! {signal.direction} @ {signal.confidence}% confidence")
            logger.info(f"Updating position SL/Target: Old SL={position.stop_loss:.2f}, Target={position.target:.2f}")

            # Update stop loss and target from new signal
            old_sl = position.stop_loss
            old_target = position.target

            position.stop_loss = signal.stop_loss
            position.target = signal.target_1

            logger.info(f"Updated SL/Target: New SL={position.stop_loss:.2f}, Target={position.target:.2f}")
            logger.info(f"Position {position.symbol} will CONTINUE with updated levels (not exiting)")

            # Save state
            self._save_state()

            return True  # Don't exit, we updated instead

        except Exception as e:
            logger.error(f"Error checking next signal: {e}, proceeding with exit")
            return False

    async def update_positions(self) -> tuple[list[PaperPosition], list[dict]]:
        """
        Update all open positions with current prices.
        Checks for exit signals and closes positions automatically.

        Returns:
            Tuple of (updated positions, closed positions info)
        """
        updated = []
        closed_positions = []

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

                current_premium = position.current_price
                should_exit = False
                exit_reason = ""

                # ========================================
                # NEW EXIT LOGIC: ALL EXITS THROUGH STOP LOSS ONLY
                # - When target is hit, SL is moved to target price
                # - Trailing SL: at 50% profit -> SL to breakeven
                # - Every 10% above 50% -> trail SL by 10%
                # ========================================

                # Store initial SL and target if not set
                if position.initial_stop_loss == 0 and position.stop_loss > 0:
                    position.initial_stop_loss = position.stop_loss
                if position.initial_target == 0 and position.target > 0:
                    position.initial_target = position.target

                # ========================================
                # DYNAMIC SL/TARGET UPDATE BASED ON MARKET DIRECTION
                # ========================================
                try:
                    from app.core.config import NIFTY_INDEX_TOKEN, BANKNIFTY_INDEX_TOKEN, SENSEX_INDEX_TOKEN
                    tokens = {
                        "NIFTY": NIFTY_INDEX_TOKEN,
                        "BANKNIFTY": BANKNIFTY_INDEX_TOKEN,
                        "SENSEX": SENSEX_INDEX_TOKEN,
                    }
                    token = tokens.get(position.index, NIFTY_INDEX_TOKEN)

                    # Fetch recent index data for momentum analysis
                    index_df = await self.data_fetcher.fetch_historical_data(
                        instrument_token=token,
                        timeframe="5minute",
                        days=1,
                    )

                    # Apply dynamic SL/target updates based on market direction
                    dynamic_updated, dynamic_reason = await self.dynamic_sl_target_update(
                        position=position,
                        current_premium=current_premium,
                        index_df=index_df,
                    )

                    if dynamic_updated:
                        logger.info(f"DYNAMIC UPDATE for {position.symbol}: {dynamic_reason}")

                except Exception as e:
                    logger.debug(f"Could not apply dynamic update: {e}")

                # Apply trailing stop loss logic (modifies position.stop_loss based on profit)
                new_sl, sl_reason = self.calculate_trailing_stop_loss(position, current_premium)
                if new_sl > position.stop_loss:
                    old_sl = position.stop_loss
                    position.stop_loss = new_sl
                    logger.info(f"SL Updated for {position.symbol}: {old_sl:.2f} -> {new_sl:.2f} ({sl_reason})")

                # ONLY EXIT CONDITION: Stop loss hit
                # ALL strategies now exit ONLY through stop loss
                if position.stop_loss > 0 and current_premium <= position.stop_loss:
                    should_exit = True
                    sl_type = "TRAILING SL" if position.trailing_sl_active else "STOP LOSS"
                    if position.target_achieved:
                        sl_type = "TARGET LOCKED SL"
                    exit_reason = f"{sl_type} HIT: Rs.{current_premium:.2f} <= SL Rs.{position.stop_loss:.2f} | P&L: {position.pnl_percent:+.1f}%"
                    if position.profit_locked_percent > 0:
                        exit_reason += f" | Locked +{position.profit_locked_percent:.0f}%"
                    logger.warning(f"SL triggered for {position.symbol}: {exit_reason}")

                # 3. STRATEGY-SPECIFIC PROFIT TARGETS (Exit immediately at target)
                # When target is reached, close position and take profit
                if not should_exit:
                    target_percent = 0
                    if self.strategy == "fixed_20_percent":
                        target_percent = 20.0
                    elif self.strategy == "profit_100_halt":
                        target_percent = 100.0
                    elif self.strategy in ["5_percent_daily", "5_percent_15min"]:
                        target_percent = 5.0
                    else:
                        # Default strategy: 100% profit target
                        target_percent = 100.0

                    # When target % is reached, EXIT immediately to take profit
                    if target_percent > 0 and position.pnl_percent >= target_percent:
                        if not position.target_achieved:
                            position.target_achieved = True
                            should_exit = True
                            exit_reason = f"TARGET EXIT +{target_percent:.0f}% profit taken | Entry: {position.entry_price:.2f}, Exit: {position.current_price:.2f}"
                            logger.info(f"TARGET ACHIEVED: {exit_reason}")

                    # Note: trailing_stoploss strategy uses the new universal trailing SL logic above
                    # (calculate_trailing_stop_loss method handles all trailing at 50% and every 10% above)

                # 4. Market close force exit
                # All strategies: Exit at 3:00 PM (market close)
                now = datetime.now()
                close_hour, close_minute = 15, 0  # 3:00 PM

                if not should_exit and now.hour == close_hour and now.minute >= close_minute:
                    should_exit = True
                    exit_reason = f"MARKET CLOSE EXIT @ {now.strftime('%H:%M')} | P&L: {position.pnl_percent:+.1f}%"
                    logger.info(f"Market close exit for {position.symbol}")

                if should_exit:
                    # For TARGET-BASED exits: Always exit, don't update (to allow new trades)
                    # For other exits (SL, market close): Check if same signal exists (smart update)
                    should_update_instead = False
                    is_target_exit = "TARGET EXIT" in exit_reason

                    if not is_target_exit:
                        # Only check for smart exit update if NOT a target-based exit
                        should_update_instead = await self._check_next_signal_and_update(position, exit_reason)

                    if not should_update_instead:
                        # Exit normally
                        await self.close_position(position, exit_reason)
                        closed_positions.append({
                            "position_id": position.position_id,
                            "symbol": position.symbol,
                            "exit_reason": exit_reason,
                            "pnl": position.pnl,
                            "pnl_percent": position.pnl_percent,
                        })
                    else:
                        logger.info(f"Position {position.symbol} updated with new signal instead of exiting")

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

        return updated, closed_positions

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

        # Calculate broker charges
        buy_value = position.entry_price * position.quantity
        sell_value = position.exit_price * position.quantity
        charges = self.calculate_broker_charges(buy_value, sell_value, position.index)
        broker_charges = charges["total"]

        # Calculate net P&L after broker charges
        gross_pnl = position.pnl
        net_pnl = gross_pnl - broker_charges

        logger.info(f"Trade closed: Gross P&L: ₹{gross_pnl:.2f}, Charges: ₹{broker_charges:.2f}, Net P&L: ₹{net_pnl:.2f}")

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
            pnl=gross_pnl,  # Gross P&L before charges
            pnl_percent=position.pnl_percent,
            max_profit_percent=max_profit_percent,
            max_loss_percent=max_loss_percent,
            captured_move_percent=captured_move_percent,
            broker_charges=broker_charges,
            net_pnl=net_pnl,  # Net P&L after charges
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

        # Update daily stats with NET P&L (after broker charges)
        # Capital logic: Deduct losses, restore to starting capital on profits
        self.daily_stats.realized_pnl += net_pnl

        # Capital adjustment logic:
        # - Loss: Deduct from capital (reduces position size)
        # - Profit: Restore capital to starting amount (resets position size)
        if net_pnl < 0:
            # Loss: Reduce capital
            self.daily_stats.current_capital += net_pnl  # Deduct loss (net_pnl is negative)
            logger.info(f"Loss deducted from capital: ₹{net_pnl:.2f} | Remaining: ₹{self.daily_stats.current_capital:.2f}")
        elif net_pnl > 0:
            # Profit: Restore capital to starting amount
            self.daily_stats.current_capital = self.daily_stats.starting_capital
            logger.info(f"Profit realized: ₹{net_pnl:.2f} | Capital restored to: ₹{self.daily_stats.current_capital:.2f}")

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
                        # Trailing SL fields
                        "initial_stop_loss": p.initial_stop_loss,
                        "initial_target": p.initial_target,
                        "trailing_sl_active": p.trailing_sl_active,
                        "target_achieved": p.target_achieved,
                        "profit_locked_percent": p.profit_locked_percent,
                        "sl_trail_count": p.sl_trail_count,
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
                        "broker_charges": h.broker_charges,
                        "net_pnl": h.net_pnl,
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

            # Write to temp file first, then rename (atomic operation)
            import tempfile
            temp_file = self.data_file.with_suffix('.tmp')

            # Retry logic for permission errors
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    with open(temp_file, "w") as f:
                        json.dump(state, f, indent=2)

                    # Rename temp file to actual file (atomic on most systems)
                    import shutil
                    shutil.move(str(temp_file), str(self.data_file))
                    break
                except PermissionError as pe:
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(0.1)  # Wait 100ms and retry
                        continue
                    else:
                        logger.warning(f"Could not save state after {max_retries} attempts: {pe}")
                except Exception as e:
                    logger.warning(f"Error during save attempt {attempt + 1}: {e}")
                    if attempt == max_retries - 1:
                        raise

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
                        # Trailing SL fields
                        initial_stop_loss=p_data.get("initial_stop_loss", p_data.get("stop_loss", 0)),
                        initial_target=p_data.get("initial_target", p_data.get("target", 0)),
                        trailing_sl_active=p_data.get("trailing_sl_active", False),
                        target_achieved=p_data.get("target_achieved", False),
                        profit_locked_percent=p_data.get("profit_locked_percent", 0),
                        sl_trail_count=p_data.get("sl_trail_count", 0),
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
                    broker_charges=h_data.get("broker_charges", 0),
                    net_pnl=h_data.get("net_pnl", h_data.get("pnl", 0)),  # Fallback to pnl if net_pnl not available
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
