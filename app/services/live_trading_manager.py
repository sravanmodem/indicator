"""
Live Trading Manager
Signal-based live trading with same logic as paper_strategy_fixed_20

Features:
- 80% minimum confidence filter
- Smart exit: Check next signal before closing
- Fixed 20% profit strategy
- Same signals as paper trading
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, time
from typing import Any, Optional

from loguru import logger

from app.core.config import NIFTY_INDEX_TOKEN, BANKNIFTY_INDEX_TOKEN, SENSEX_INDEX_TOKEN
from app.services.live_trading_service import get_live_trading_service, LivePosition
from app.services.data_fetcher import get_data_fetcher
from app.services.signal_engine import get_signal_engine, TradingStyle, TradeSignal


@dataclass
class LiveTradePosition:
    """Tracked position for signal-based live trading."""
    tradingsymbol: str
    index: str
    exchange: str
    option_type: str  # CE or PE
    strike: float
    entry_price: float
    quantity: int
    lots: int
    entry_time: datetime
    stop_loss: float
    target: float
    signal_confidence: float
    is_open: bool = True
    exit_price: float = 0.0
    exit_time: Optional[datetime] = None
    exit_reason: str = ""
    max_pnl_percent: float = 0.0  # Track max profit reached
    trading_halted: bool = False  # For 20% profit halt


class LiveTradingManager:
    """
    Manages live trading with signal-based entry/exit logic.
    Mirrors paper_trading_service with fixed_20_percent strategy.
    """

    CONFIDENCE_THRESHOLD = 80.0  # Minimum confidence for entry
    PROFIT_TARGET = 20.0  # Exit at 20% profit
    LOSS_LIMIT = 20.0  # Halt trading at 20% daily loss
    MARKET_CLOSE_HOUR = 15
    MARKET_CLOSE_MINUTE = 0

    def __init__(self):
        self.live_service = get_live_trading_service()
        self.data_fetcher = get_data_fetcher()
        self.tracked_positions: list[LiveTradePosition] = []
        self.is_trading_halted = False
        self.halt_reason = ""
        self.starting_capital = 0.0  # Will be set from margin on first trade
        self.daily_pnl = 0.0  # Track total P&L for the day

    @property
    def is_live_mode(self) -> bool:
        """Check if live trading is enabled."""
        return self.live_service.is_live_mode

    def is_trading_hours(self) -> tuple[bool, str]:
        """
        Check if we're in trading hours.

        Returns:
            (is_allowed, reason)
        """
        now = datetime.now()
        current_time = now.time()
        weekday = now.weekday()

        # Weekend check
        if weekday >= 5:
            return False, "Weekend"

        # Market hours: 9:30 AM to 3:00 PM
        start_time = time(9, 30)
        end_time = time(15, 0)

        if current_time < start_time:
            return False, "Pre-market"
        if current_time >= end_time:
            return False, "Post-market"

        return True, "OK"

    def has_open_position(self) -> bool:
        """Check if there's an open tracked position."""
        return any(pos.is_open for pos in self.tracked_positions)

    def check_daily_loss_limit(self) -> bool:
        """
        Check if 20% daily loss limit has been reached.

        Returns:
            True if trading should be halted
        """
        if self.is_trading_halted:
            return True

        if self.starting_capital <= 0:
            return False

        loss_percent = abs(self.daily_pnl) / self.starting_capital * 100

        if self.daily_pnl < 0 and loss_percent >= self.LOSS_LIMIT:
            self.is_trading_halted = True
            self.halt_reason = f"Daily loss limit reached: -{loss_percent:.1f}% (Max: -{self.LOSS_LIMIT}%)"
            logger.warning(f"LIVE TRADING HALTED: {self.halt_reason}")
            return True

        return False

    async def execute_signal_trade(
        self,
        signal: TradeSignal,
        index: str,
    ) -> bool:
        """
        Execute a trade based on signal (same logic as paper trading).

        Args:
            signal: TradeSignal from signal engine
            index: Trading index (NIFTY, BANKNIFTY, SENSEX)

        Returns:
            True if executed, False otherwise
        """
        # Check if live mode enabled
        if not self.is_live_mode:
            logger.warning("TRADE BLOCKED: Live trading not enabled")
            return False

        # Check trading hours
        is_trading, reason = self.is_trading_hours()
        if not is_trading:
            logger.info(f"Outside trading hours: {reason}")
            return False

        # Check if trading is halted (20% profit or 20% loss reached)
        if self.check_daily_loss_limit():
            logger.warning(f"TRADE BLOCKED: {self.halt_reason}")
            return False

        # Check signal direction
        if signal.direction not in ["CE", "PE"]:
            logger.warning(f"TRADE BLOCKED: No clear signal direction (got: {signal.direction})")
            return False

        # Check confidence threshold (80%)
        if signal.confidence < self.CONFIDENCE_THRESHOLD:
            logger.warning(
                f"TRADE BLOCKED: Signal confidence too low: {signal.confidence}% "
                f"(need {self.CONFIDENCE_THRESHOLD}%+)"
            )
            return False

        # Get recommended option
        if not signal.recommended_option:
            logger.warning("TRADE BLOCKED: No recommended option in signal")
            return False

        # Check if already have open position
        if self.has_open_position():
            logger.warning("TRADE BLOCKED: Position already open")
            return False

        opt = signal.recommended_option
        logger.info(
            f"Signal: {signal.direction} @ {opt.strike} | "
            f"LTP: Rs.{opt.ltp:.2f} | Confidence: {signal.confidence}%"
        )

        # Calculate position size from available margin
        lots, quantity, capital = await self.live_service.calculate_position_size(
            option_ltp=opt.ltp,
            index=index,
            margin_percent=100,  # Use 100% margin
        )

        if lots == 0:
            logger.warning("Insufficient margin for trade")
            return False

        # Get exchange
        exchange = self.live_service.get_exchange_for_index(index)

        # Place order
        logger.info(
            f"Placing LIVE order: {opt.symbol} x {lots} lots ({quantity} qty) "
            f"@ Rs.{opt.ltp:.2f}"
        )

        result = await self.live_service.place_entry_order(
            tradingsymbol=opt.symbol,
            exchange=exchange,
            quantity=quantity,
            price=opt.ltp,
            transaction_type="BUY",
        )

        if not result.success:
            logger.error(f"Order placement failed: {result.error}")
            return False

        logger.info(f"✓ Order placed successfully: {result.order_id}")

        # Set starting capital on first trade
        if self.starting_capital == 0:
            margin = await self.live_service.get_available_margin()
            self.starting_capital = margin.available_margin
            logger.info(f"Starting capital set: Rs.{self.starting_capital:,.0f}")

        # Track position
        tracked_position = LiveTradePosition(
            tradingsymbol=opt.symbol,
            index=index,
            exchange=exchange,
            option_type=signal.direction,
            strike=opt.strike,
            entry_price=opt.ltp,
            quantity=quantity,
            lots=lots,
            entry_time=datetime.now(),
            stop_loss=signal.stop_loss,
            target=signal.target_1,
            signal_confidence=signal.confidence,
        )

        self.tracked_positions.append(tracked_position)

        logger.info(
            f"Position tracked: {opt.symbol} | Entry: Rs.{opt.ltp:.2f} | "
            f"SL: Rs.{signal.stop_loss:.2f} | Target: Rs.{signal.target_1:.2f}"
        )

        return True

    async def _check_next_signal_and_update(
        self,
        position: LiveTradePosition,
        exit_reason: str
    ) -> bool:
        """
        Check if next signal is in same direction.
        If yes, update SL/target instead of exiting.

        Returns:
            True if position was updated (don't exit), False if should exit
        """
        try:
            logger.info(f"Checking next signal before exiting {position.tradingsymbol}...")

            # Get token for index
            tokens = {
                "NIFTY": NIFTY_INDEX_TOKEN,
                "BANKNIFTY": BANKNIFTY_INDEX_TOKEN,
                "SENSEX": SENSEX_INDEX_TOKEN,
            }
            token = tokens.get(position.index, NIFTY_INDEX_TOKEN)

            # Fetch historical data
            df = await self.data_fetcher.fetch_historical_data(
                instrument_token=token,
                timeframe="5minute",
                days=3,
            )

            if df.empty:
                logger.warning("No historical data, proceeding with exit")
                return False

            # Get option chain
            chain_data = await self.data_fetcher.get_option_chain(index=position.index)
            option_chain = chain_data.get("chain", []) if "error" not in chain_data else None

            # Generate new signal
            engine = get_signal_engine(TradingStyle.INTRADAY)
            signal = engine.analyze(df=df, option_chain=option_chain)

            if not signal:
                logger.info("No new signal, proceeding with exit")
                return False

            # Check confidence
            if signal.confidence < self.CONFIDENCE_THRESHOLD:
                logger.info(
                    f"New signal confidence {signal.confidence}% < {self.CONFIDENCE_THRESHOLD}%, "
                    "proceeding with exit"
                )
                return False

            # Check direction match
            if signal.direction != position.option_type:
                logger.info(
                    f"New signal direction {signal.direction} != position {position.option_type}, "
                    "proceeding with exit"
                )
                return False

            # Same direction! Update SL/target
            logger.info(f"✓ SAME DIRECTION signal found! {signal.direction} @ {signal.confidence}%")
            logger.info(
                f"Updating SL/Target: Old SL={position.stop_loss:.2f}, Target={position.target:.2f}"
            )

            position.stop_loss = signal.stop_loss
            position.target = signal.target_1

            logger.info(
                f"Updated SL/Target: New SL={position.stop_loss:.2f}, Target={position.target:.2f}"
            )
            logger.info(f"Position {position.tradingsymbol} will CONTINUE")

            return True  # Don't exit

        except Exception as e:
            logger.error(f"Error checking next signal: {e}, proceeding with exit")
            return False

    async def update_positions(self) -> list[dict]:
        """
        Update all tracked positions and check for exits.
        Similar to paper trading update_positions.

        Returns:
            List of closed position info
        """
        if not self.is_live_mode:
            return []

        closed_positions = []

        # Get real positions from Kite
        live_positions = await self.live_service.get_positions()

        for tracked_pos in self.tracked_positions:
            if not tracked_pos.is_open:
                continue

            try:
                # Find matching live position
                live_pos = None
                for lp in live_positions:
                    if lp.tradingsymbol == tracked_pos.tradingsymbol:
                        live_pos = lp
                        break

                if not live_pos:
                    # Position might have been closed or not found
                    logger.warning(f"Position {tracked_pos.tradingsymbol} not found in live positions")
                    continue

                current_price = live_pos.last_price
                pnl_percent = live_pos.pnl_percent

                # Track max profit
                if pnl_percent > tracked_pos.max_pnl_percent:
                    tracked_pos.max_pnl_percent = pnl_percent

                should_exit = False
                exit_reason = ""

                # Exit conditions (same as paper fixed_20_percent)

                # 1. Stop loss hit
                if tracked_pos.stop_loss > 0 and current_price <= tracked_pos.stop_loss:
                    should_exit = True
                    exit_reason = (
                        f"STOP LOSS HIT: Rs.{current_price:.2f} <= SL Rs.{tracked_pos.stop_loss:.2f} | "
                        f"P&L: {pnl_percent:+.1f}%"
                    )

                # 2. Target hit
                if not should_exit and tracked_pos.target > 0 and current_price >= tracked_pos.target:
                    should_exit = True
                    exit_reason = (
                        f"TARGET HIT: Rs.{current_price:.2f} >= Target Rs.{tracked_pos.target:.2f} | "
                        f"P&L: {pnl_percent:+.1f}%"
                    )

                # 3. Fixed 20% profit (halt trading)
                if not should_exit and pnl_percent >= self.PROFIT_TARGET:
                    should_exit = True
                    exit_reason = f"FIXED 20% PROFIT: {pnl_percent:+.1f}% reached"
                    # Mark for halt
                    tracked_pos.trading_halted = True

                # 4. Market close force exit
                now = datetime.now()
                if not should_exit and now.hour == self.MARKET_CLOSE_HOUR and now.minute >= self.MARKET_CLOSE_MINUTE:
                    should_exit = True
                    exit_reason = f"MARKET CLOSE EXIT @ {now.strftime('%H:%M')} | P&L: {pnl_percent:+.1f}%"

                if should_exit:
                    # SMART EXIT: Check next signal
                    should_update_instead = await self._check_next_signal_and_update(
                        tracked_pos,
                        exit_reason
                    )

                    if not should_update_instead:
                        # Exit position
                        result = await self.live_service.place_exit_order(
                            tradingsymbol=tracked_pos.tradingsymbol,
                            exchange=tracked_pos.exchange,
                            quantity=tracked_pos.quantity,
                            price=current_price,
                        )

                        if result.success:
                            tracked_pos.is_open = False
                            tracked_pos.exit_price = current_price
                            tracked_pos.exit_time = datetime.now()
                            tracked_pos.exit_reason = exit_reason

                            logger.info(f"✓ Position closed: {tracked_pos.tradingsymbol} | {exit_reason}")

                            # Update daily P&L
                            self.daily_pnl += live_pos.pnl
                            logger.info(f"Daily P&L updated: Rs.{self.daily_pnl:,.0f}")

                            closed_positions.append({
                                "symbol": tracked_pos.tradingsymbol,
                                "exit_reason": exit_reason,
                                "pnl_percent": pnl_percent,
                                "pnl": live_pos.pnl,
                            })

                            # Check and halt if 20% loss reached
                            self.check_daily_loss_limit()

                            # Halt trading if 20% profit reached
                            if tracked_pos.trading_halted:
                                self.is_trading_halted = True
                                self.halt_reason = "20% profit target reached - No more trades today"
                                logger.warning(f"TRADING HALTED: {self.halt_reason}")
                        else:
                            logger.error(f"Failed to exit position: {result.error}")
                    else:
                        logger.info(f"Position {tracked_pos.tradingsymbol} updated instead of exiting")

            except Exception as e:
                logger.error(f"Error updating position {tracked_pos.tradingsymbol}: {e}")

        return closed_positions

    def reset_daily_state(self):
        """Reset daily state (call this at start of day)."""
        self.is_trading_halted = False
        self.halt_reason = ""
        # Clear closed positions from previous days
        self.tracked_positions = [p for p in self.tracked_positions if p.is_open]
        logger.info("Live trading manager: Daily state reset")


# Singleton
_live_manager: Optional[LiveTradingManager] = None


def get_live_trading_manager() -> LiveTradingManager:
    """Get or create singleton live trading manager."""
    global _live_manager
    if _live_manager is None:
        _live_manager = LiveTradingManager()
    return _live_manager
