"""
Auto Trading Service
Background task that monitors signals and auto-executes trades via AI

Features:
- Runs continuously during market hours
- Generates signals and sends to AI for approval
- Auto-executes trades when AI says BUY
- Monitors positions and sends to AI for exit decisions
- Respects trading hours (9:15 AM - 3:30 PM)
"""

import asyncio
from datetime import datetime, time, date
from typing import Optional

from loguru import logger

from app.core.config import get_settings, NIFTY_INDEX_TOKEN, BANKNIFTY_INDEX_TOKEN, SENSEX_INDEX_TOKEN


# Trading window
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)

# Known market holidays for 2024-2026 (NSE)
INDIAN_MARKET_HOLIDAYS = {
    date(2024, 1, 26), date(2024, 3, 8), date(2024, 3, 25), date(2024, 3, 29),
    date(2024, 4, 11), date(2024, 4, 14), date(2024, 4, 17), date(2024, 4, 21),
    date(2024, 5, 1), date(2024, 5, 23), date(2024, 6, 17), date(2024, 7, 17),
    date(2024, 8, 15), date(2024, 9, 16), date(2024, 10, 2), date(2024, 10, 12),
    date(2024, 11, 1), date(2024, 11, 15), date(2024, 12, 25),
    date(2025, 1, 26), date(2025, 2, 26), date(2025, 3, 14), date(2025, 3, 31),
    date(2025, 4, 10), date(2025, 4, 14), date(2025, 4, 18), date(2025, 5, 1),
    date(2025, 8, 15), date(2025, 8, 27), date(2025, 10, 2), date(2025, 10, 21),
    date(2025, 11, 5), date(2025, 12, 25),
    date(2026, 1, 26), date(2026, 2, 17), date(2026, 3, 3), date(2026, 3, 20),
    date(2026, 4, 3), date(2026, 4, 14), date(2026, 5, 1), date(2026, 5, 12),
    date(2026, 6, 5), date(2026, 7, 6), date(2026, 8, 15), date(2026, 8, 17),
    date(2026, 9, 4), date(2026, 10, 2), date(2026, 10, 20), date(2026, 11, 9),
    date(2026, 11, 24), date(2026, 12, 25),
}
SIGNAL_CHECK_INTERVAL = 30  # Check for signals every 30 seconds
POSITION_CHECK_INTERVAL = 15  # Check positions every 15 seconds


class AutoTrader:
    """
    Background auto-trading service.

    Flow:
    1. Check if market is open
    2. Generate signal for selected index
    3. If signal exists, send to AI for BUY/SKIP
    4. If AI says BUY, execute trade automatically
    5. Monitor positions and send to AI for EXIT/HOLD
    6. If AI says EXIT, close position
    """

    def __init__(self):
        self.settings = get_settings()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_signal_time: Optional[datetime] = None
        self._last_signal_id: Optional[str] = None
        self._consecutive_errors = 0
        self._max_errors = 10

    @property
    def is_running(self) -> bool:
        return self._running

    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        now = datetime.now()
        current_time = now.time()
        current_date = now.date()
        weekday = now.weekday()

        # Weekend check
        if weekday >= 5:
            return False

        # Holiday check
        if current_date in INDIAN_MARKET_HOLIDAYS:
            return False

        # Time check
        if current_time < MARKET_OPEN or current_time >= MARKET_CLOSE:
            return False

        return True

    async def start(self):
        """Start the auto-trading background task."""
        if self._running:
            logger.warning("Auto trader already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._trading_loop())
        logger.info("Auto Trader STARTED - Will auto-execute AI-approved trades")

    async def stop(self):
        """Stop the auto-trading background task."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Auto Trader STOPPED")

    async def _trading_loop(self):
        """Main trading loop that runs continuously."""
        logger.info("Auto trading loop started")

        while self._running:
            try:
                # Check market hours
                if not self.is_market_open():
                    await asyncio.sleep(60)  # Sleep 1 minute when market closed
                    continue

                # Run signal check and position monitoring in parallel
                await asyncio.gather(
                    self._check_and_execute_signals(),
                    self._monitor_positions(),
                    return_exceptions=True
                )

                # Reset error counter on success
                self._consecutive_errors = 0

                # Wait before next check
                await asyncio.sleep(SIGNAL_CHECK_INTERVAL)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._consecutive_errors += 1
                logger.error(f"Auto trader error ({self._consecutive_errors}/{self._max_errors}): {e}")

                if self._consecutive_errors >= self._max_errors:
                    logger.error("Too many consecutive errors, stopping auto trader")
                    self._running = False
                    break

                await asyncio.sleep(10)  # Brief pause on error

        logger.info("Auto trading loop ended")

    async def _check_and_execute_signals(self):
        """Check for signals and execute if AI approves."""
        try:
            from app.services.paper_trading import get_paper_trading_service
            from app.services.signal_engine import get_signal_engine, TradingStyle
            from app.services.data_fetcher import get_data_fetcher
            from app.services.ai_trading_service import get_ai_trading_service
            from app.services.zerodha_auth import get_auth_service

            # Check authentication
            auth = get_auth_service()
            if not auth.is_authenticated:
                return

            paper = get_paper_trading_service()
            fetcher = get_data_fetcher()
            ai_service = get_ai_trading_service()

            # Skip if auto-trade is disabled
            if not paper.is_auto_trade:
                return

            # Skip if already have open position
            if paper.has_open_position():
                return

            # Get trading index
            trading_index = paper.get_trading_index()
            tokens = {
                "NIFTY": NIFTY_INDEX_TOKEN,
                "BANKNIFTY": BANKNIFTY_INDEX_TOKEN,
                "SENSEX": SENSEX_INDEX_TOKEN,
            }
            token = tokens.get(trading_index.index, NIFTY_INDEX_TOKEN)

            # Fetch historical data for signal generation (5-min for indicators)
            df = await fetcher.fetch_historical_data(
                instrument_token=token,
                timeframe="5minute",
                days=3,
            )

            if df.empty:
                return

            # Fetch 1-minute data for last 15 minutes (short-term momentum)
            df_1min = await fetcher.fetch_historical_data(
                instrument_token=token,
                timeframe="minute",
                days=1,  # Just 1 day is enough for 15 minutes
            )

            # Fetch 10-minute data for last 1 hour (longer-term trend)
            df_10min = await fetcher.fetch_historical_data(
                instrument_token=token,
                timeframe="10minute",
                days=1,  # 1 day is enough for 1 hour
            )

            # Get option chain
            chain_data = await fetcher.get_option_chain(index=trading_index.index)
            option_chain = chain_data.get("chain", []) if "error" not in chain_data else None

            # Generate signal
            engine = get_signal_engine(TradingStyle.INTRADAY)
            signal = engine.analyze(df=df, option_chain=option_chain)

            # Prepare OHLCV data for AI
            # Last 15 candles of 1-minute data
            ohlcv_1min = []
            if not df_1min.empty:
                recent_1min = df_1min.tail(15)
                for idx, row in recent_1min.iterrows():
                    ohlcv_1min.append({
                        "time": idx.strftime("%H:%M") if hasattr(idx, 'strftime') else str(idx),
                        "open": float(row.get("open", 0)),
                        "high": float(row.get("high", 0)),
                        "low": float(row.get("low", 0)),
                        "close": float(row.get("close", 0)),
                        "volume": int(row.get("volume", 0)),
                    })

            # Last 6 candles of 10-minute data (1 hour)
            ohlcv_10min = []
            if not df_10min.empty:
                recent_10min = df_10min.tail(6)
                for idx, row in recent_10min.iterrows():
                    ohlcv_10min.append({
                        "time": idx.strftime("%H:%M") if hasattr(idx, 'strftime') else str(idx),
                        "open": float(row.get("open", 0)),
                        "high": float(row.get("high", 0)),
                        "low": float(row.get("low", 0)),
                        "close": float(row.get("close", 0)),
                        "volume": int(row.get("volume", 0)),
                    })

            # Store in signal for AI context
            if signal:
                signal.ohlcv_1min = ohlcv_1min
                signal.ohlcv_10min = ohlcv_10min

            if not signal or not signal.recommended_option:
                return

            # Skip if same signal as last time (avoid duplicate trades)
            signal_id = f"{signal.direction}_{signal.recommended_option.strike}_{signal.recommended_option.ltp:.0f}"
            if signal_id == self._last_signal_id:
                # Same signal, skip unless 5 minutes passed
                if self._last_signal_time and (datetime.now() - self._last_signal_time).seconds < 300:
                    return

            # Log signal
            logger.info(f"AUTO-TRADE Signal: {signal.direction} @ {signal.recommended_option.strike} | LTP: {signal.recommended_option.ltp:.2f} | Confidence: {signal.confidence:.0f}%")

            # Execute trade (AI decision happens inside execute_signal_trade)
            order = await paper.execute_signal_trade(signal, trading_index)

            if order:
                self._last_signal_id = signal_id
                self._last_signal_time = datetime.now()
                logger.info(f"AUTO-TRADE Executed: {order.symbol} | Qty: {order.quantity} | Price: {order.price:.2f}")
            else:
                logger.debug("AUTO-TRADE: Signal rejected by AI or other condition")

        except Exception as e:
            logger.error(f"Auto trade signal check error: {e}")

    async def _monitor_positions(self):
        """Monitor open positions and trigger AI exit analysis."""
        try:
            from app.services.paper_trading import get_paper_trading_service
            from app.services.zerodha_auth import get_auth_service

            auth = get_auth_service()
            if not auth.is_authenticated:
                return

            paper = get_paper_trading_service()

            # Update positions (this internally calls AI for exit decisions)
            updated_positions, closed_positions = await paper.update_positions()

            if closed_positions:
                for pos in closed_positions:
                    logger.info(f"AUTO-TRADE Position Closed: {pos.get('symbol', 'unknown')} | Reason: {pos.get('exit_reason', 'unknown')} | P&L: {pos.get('pnl', 0):.0f}")

        except Exception as e:
            logger.error(f"Auto trade position monitor error: {e}")


# Singleton instance
_auto_trader: Optional[AutoTrader] = None


def get_auto_trader() -> AutoTrader:
    """Get or create singleton auto trader instance."""
    global _auto_trader
    if _auto_trader is None:
        _auto_trader = AutoTrader()
    return _auto_trader
