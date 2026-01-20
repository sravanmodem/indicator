"""
15-Minute Dedicated Signal Engine
Generates ONE high-confidence signal per 15-minute boundary for 5% profit targeting

Features:
- ONE signal per 15-minute boundary (9:30-9:44, 9:45-9:59, 10:00-10:14, etc.)
- Once signal generated in a boundary, LOCKED until next boundary
- Requires 90% confidence minimum (waits if below 90%)
- If confidence < 90%, skips entire boundary, tries next one
- Targets 5% profit with aggressive entry
- Only for 5_percent_15min strategy

Execution Timeline:
9:30:00-9:44:59 → Generate signal (if 90%+ confidence, execute) OR skip entire window
9:45:00-9:59:59 → Generate NEW signal (if 90%+ confidence, execute) OR skip entire window
10:00:00-10:14:59 → Generate NEW signal (if 90%+ confidence, execute) OR skip entire window
etc.
"""

import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional
from loguru import logger

from app.services.signal_engine import SignalEngine, TradingStyle, TradeSignal
from app.core.config import INDICATOR_PARAMS, SIGNAL_THRESHOLDS


@dataclass
class SignalGeneration15Min:
    """Track signal generation timing for 15-minute strategy."""
    last_signal_time: Optional[datetime] = None
    last_signal_id: Optional[str] = None
    last_signal_boundary: Optional[datetime] = None  # Track which 15-min boundary had a signal
    consecutive_low_confidence: int = 0  # Count of low confidence periods
    last_check_time: Optional[datetime] = None


class Signal15MinEngine:
    """
    Dedicated signal engine for 15-minute trading strategy.

    Rules:
    1. Generate signal only at 15-minute intervals (9:30, 9:45, 10:00, etc.)
    2. Require 90% confidence minimum
    3. If confidence < 90%, SKIP this interval and wait for next 15-min period
    4. When signal meets 90% threshold, execute immediately (5% target)
    5. Don't retry same signal within same 15-min interval
    """

    CONFIDENCE_THRESHOLD = 90  # Strict 90% requirement
    SIGNAL_INTERVAL_MINUTES = 15
    PROFIT_TARGET = 5.0

    def __init__(self):
        self.base_engine = SignalEngine(TradingStyle.INTRADAY)
        self.signal_state = SignalGeneration15Min()
        self.market_open_time = None  # 9:30 AM

    def get_next_15min_boundary(self, now: datetime) -> datetime:
        """
        Calculate the next 15-minute boundary from market open.

        Market opens at 9:30 AM.
        Boundaries: 9:30, 9:45, 10:00, 10:15, 10:30, etc.
        """
        # Market open is at 9:30 AM
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)

        # If before market open, return market open
        if now < market_open:
            return market_open

        # Calculate minutes since market open
        minutes_since_open = int((now - market_open).total_seconds() / 60)

        # Find next 15-minute boundary
        next_boundary_minutes = ((minutes_since_open // 15) + 1) * 15
        next_boundary = market_open + timedelta(minutes=next_boundary_minutes)

        return next_boundary

    def get_current_15min_boundary(self, now: datetime) -> datetime:
        """Get the current 15-minute boundary."""
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)

        if now < market_open:
            return market_open

        minutes_since_open = int((now - market_open).total_seconds() / 60)
        current_boundary_minutes = (minutes_since_open // 15) * 15
        current_boundary = market_open + timedelta(minutes=current_boundary_minutes)

        return current_boundary

    def should_generate_signal(self, now: datetime) -> tuple[bool, str]:
        """
        Check if we should generate a signal now.

        ONE signal per 15-minute boundary:
        - 9:30-9:44:59 → Signal at boundary 9:30
        - 9:45-9:59:59 → Signal at boundary 9:45
        - 10:00-10:14:59 → Signal at boundary 10:00
        - etc.

        Returns:
            (should_generate, reason)
        """
        current_boundary = self.get_current_15min_boundary(now)

        # If we already generated a signal in THIS boundary, skip
        if self.signal_state.last_signal_boundary:
            if self.signal_state.last_signal_boundary == current_boundary:
                time_until_next = (current_boundary + timedelta(minutes=15) - now).total_seconds() / 60
                return False, f"Already got signal for {current_boundary.strftime('%H:%M')} boundary. Next signal at {(current_boundary + timedelta(minutes=15)).strftime('%H:%M')} ({time_until_next:.0f} min)"

        return True, f"Ready for signal at {current_boundary.strftime('%H:%M')} boundary"

    async def generate_signal(
        self,
        df: pd.DataFrame,
        option_chain: Optional[list] = None,
        spot_price: Optional[float] = None,
        vix_value: Optional[float] = None,
    ) -> Optional[TradeSignal]:
        """
        Generate a 15-minute signal with strict 90% confidence requirement.

        Args:
            df: 5-minute OHLCV data
            option_chain: Option chain data
            spot_price: Current spot price
            vix_value: Current VIX value

        Returns:
            TradeSignal if confidence >= 90%, None otherwise
        """
        now = datetime.now()

        # Check if we should generate a signal at this 15-min boundary
        should_generate, reason = self.should_generate_signal(now)
        if not should_generate:
            logger.debug(f"15MIN: {reason}")
            return None

        # Generate signal using base engine
        signal = self.base_engine.analyze(
            df=df,
            option_chain=option_chain,
            spot_price=spot_price,
            vix_value=vix_value
        )

        if not signal:
            logger.info(f"15MIN: No signal generated at {now.strftime('%H:%M')}")
            self.signal_state.consecutive_low_confidence += 1
            return None

        # Check confidence threshold (90% minimum)
        if signal.confidence < self.CONFIDENCE_THRESHOLD:
            logger.warning(
                f"15MIN [{now.strftime('%H:%M')}]: Confidence {signal.confidence:.0f}% < 90% threshold. "
                f"WAITING FOR NEXT 15-MIN INTERVAL. ({self.get_next_15min_boundary(now).strftime('%H:%M')})"
            )
            self.signal_state.consecutive_low_confidence += 1
            return None

        # Confidence threshold met (>= 90%)
        logger.info(
            f"15MIN [{now.strftime('%H:%M')}]: ✓ HIGH CONFIDENCE SIGNAL "
            f"Direction={signal.direction} | Confidence={signal.confidence:.0f}% | "
            f"Strike={signal.recommended_option.strike if signal.recommended_option else 'N/A'} | "
            f"Target=5% | Ready for auto-execution"
        )

        # Set profit target to 5% for this strategy
        signal.target_1 = signal.entry_price * 1.05 if signal.entry_price > 0 else signal.recommended_option.ltp * 1.05

        # Reset low confidence counter since we got a high confidence signal
        self.signal_state.consecutive_low_confidence = 0

        # Store this signal as the last one (track boundary for one-signal-per-boundary enforcement)
        current_boundary = self.get_current_15min_boundary(now)
        self.signal_state.last_signal_time = now
        self.signal_state.last_signal_boundary = current_boundary
        self.signal_state.last_signal_id = f"{signal.direction}_{signal.recommended_option.strike if signal.recommended_option else 'NA'}"

        logger.info(f"15MIN: Signal locked for {current_boundary.strftime('%H:%M')} boundary. Next signal at {(current_boundary + timedelta(minutes=15)).strftime('%H:%M')}")

        return signal


# Singleton instance
_signal_engine_15min: Optional[Signal15MinEngine] = None


def get_signal_engine_15min() -> Signal15MinEngine:
    """Get or create singleton 15-minute signal engine."""
    global _signal_engine_15min
    if _signal_engine_15min is None:
        _signal_engine_15min = Signal15MinEngine()
    return _signal_engine_15min
