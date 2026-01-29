"""
Expiry Day Signal Engine
Aggressive high-frequency trading for expiry day with 1:00 PM - 3:00 PM trading window

TRADING LOGIC (Aggressive Scalping):
=====================================
1. TRADING WINDOW ONLY
   - 1:00 PM to 3:00 PM on expiry days
   - Skip if outside this window
   - Lock in profits quickly (10%+ target)

2. AGGRESSIVE MOMENTUM (Lower Thresholds)
   - RSI > 60 (CE) or RSI < 40 (PE) - more lenient
   - MACD momentum signals
   - Price acceleration detection

3. SIMPLIFIED TREND
   - SuperTrend direction match (more important than EMA)
   - Quick entry on trend confirmation
   - Less waiting for perfect setups

4. RISK MANAGEMENT
   - Tight stop loss: 0.75% (half of normal 1.5%)
   - Aggressive position sizing
   - Quick exit on SL or 10%+ profit

5. CONFIDENCE SCORING (50% Minimum - AGGRESSIVE)
   - Simplified: Momentum (30%) + Trend (40%) + Options (30%)
   - 50%+ = TRADE | 40-49% = MONITOR | <40% = SKIP
   - Lower thresholds suit expiry day volatility

6. SIGNAL QUALITY FILTERS
   - Skip if VIX > 50 (extreme spike)
   - Skip if < 30 min to market close
   - Allow higher frequency (every 1-2 min vs 15-min windows)

Execution: Whenever conditions are met during 1-3 PM window
"""

import pandas as pd
from datetime import datetime, time
from dataclasses import dataclass, field
from typing import Optional, List
from loguru import logger

from app.services.signal_engine import SignalEngine, TradingStyle, TradeSignal
from app.core.config import INDICATOR_PARAMS, SIGNAL_THRESHOLDS


@dataclass
class ExpiryDayMetrics:
    """Simplified metrics for expiry day aggressive trading."""
    momentum_score: float = 0.0  # 0-30
    trend_score: float = 0.0     # 0-40
    options_score: float = 0.0   # 0-30
    final_confidence: float = 0.0  # 0-100

    def calculate_total(self) -> float:
        """Calculate total confidence from component scores."""
        total = self.momentum_score + self.trend_score + self.options_score
        return min(100.0, total)


@dataclass
class ExpiryDaySignalState:
    """Track signal generation state for expiry day trading."""
    last_signal_time: Optional[datetime] = None
    last_signal_id: Optional[str] = None
    consecutive_low_confidence: int = 0
    last_check_time: Optional[datetime] = None
    signals_today: int = 0
    max_signals_per_day: int = 50  # Allow many signals during 2-hour window


class SignalExpiryDayEngine:
    """
    Aggressive expiry day signal engine with simplified, fast logic.

    Key Features:
    - 1:00 PM - 3:00 PM trading window only
    - Simplified momentum + trend + options scoring
    - 50% minimum confidence threshold (aggressive)
    - Quick exits at 10%+ profit
    - Tight 0.75% stop loss
    """

    # Confidence thresholds (more aggressive)
    CONFIDENCE_THRESHOLD_TRADE = 50       # Execute immediately (very lenient)
    CONFIDENCE_THRESHOLD_MONITOR = 40     # Monitor for entry
    CONFIDENCE_THRESHOLD_SKIP = 30        # Skip entirely

    # Trading window
    TRADING_START = time(13, 0)            # 1:00 PM
    TRADING_END = time(15, 0)              # 3:00 PM (market close)
    MIN_MINUTES_TO_CLOSE = 0               # Trade until market close (no minimum buffer)

    # Risk Management (more aggressive)
    STOP_LOSS_PERCENT = 0.75               # Tight stop at 0.75% (vs 1.5% normal)
    POSITION_RISK_PERCENT = 1.0            # Risk 1% per trade
    TARGET_PROFIT_PERCENT = 10.0           # Aim for 10%+ profit

    # Volatility filters
    MAX_VIX_FOR_TRADING = 50               # Allow higher VIX on expiry
    MIN_ATR_PERCENT = 0.2                  # Lower ATR threshold (more opportunities)

    def __init__(self):
        self.base_engine = SignalEngine(TradingStyle.INTRADAY)
        self.signal_state = ExpiryDaySignalState()
        self.last_processed_time: Optional[datetime] = None

    def is_in_trading_window(self) -> tuple[bool, str]:
        """
        Check if current time is within expiry day trading window (1-3 PM).

        Returns:
            Tuple of (is_valid, reason)
        """
        now = datetime.now()
        current_time = now.time()

        # Check if within 1 PM - 3 PM window
        if current_time < self.TRADING_START:
            return False, f"Too early: {current_time.strftime('%H:%M')} < 1:00 PM"

        if current_time >= self.TRADING_END:
            return False, f"Too late: {current_time.strftime('%H:%M')} >= 3:00 PM"

        # Check if enough time remaining
        minutes_to_close = int((self.TRADING_END.hour * 60 + self.TRADING_END.minute -
                               current_time.hour * 60 - current_time.minute))
        if minutes_to_close < self.MIN_MINUTES_TO_CLOSE:
            return False, f"Too close to close: only {minutes_to_close} min remaining"

        return True, "Within trading window (1-3 PM)"

    def calculate_momentum_score_expiry(self, signal) -> float:
        """
        Calculate momentum score for expiry day (aggressive, 0-30 points).
        Lower thresholds: RSI > 60 instead of 70, MACD positive.
        """
        score = 0.0

        try:
            if not signal.indicators:
                return 0.0

            # Look for momentum indicators
            rsi_value = None
            macd_positive = False

            for indicator in signal.indicators:
                if indicator.name and "RSI" in indicator.name.upper():
                    rsi_value = indicator.value
                elif indicator.name and "MACD" in indicator.name.upper():
                    if signal.direction == "ce" and indicator.signal == "ce":
                        macd_positive = True
                    elif signal.direction == "pe" and indicator.signal == "pe":
                        macd_positive = True

            # RSI check (0-15 points) - AGGRESSIVE THRESHOLDS
            if rsi_value is not None:
                if signal.direction == "ce":
                    if rsi_value > 70:
                        score += 15  # Strong momentum
                    elif rsi_value > 60:
                        score += 10  # Moderate momentum (lower threshold)
                    elif rsi_value > 50:
                        score += 5   # Weak momentum
                else:  # PE
                    if rsi_value < 30:
                        score += 15  # Strong momentum
                    elif rsi_value < 40:
                        score += 10  # Moderate momentum (lower threshold)
                    elif rsi_value > 50:
                        score += 5   # Weak momentum

            # MACD check (0-15 points)
            if macd_positive:
                score += 15

        except Exception as e:
            logger.debug(f"Error calculating expiry day momentum score: {e}")

        return min(score, 30.0)

    def calculate_trend_score_expiry(self, signal) -> float:
        """
        Calculate trend score for expiry day (aggressive, 0-40 points).
        Heavy weight on SuperTrend direction match.
        """
        score = 0.0

        try:
            if not signal.indicators:
                return 0.0

            # Look for trend indicators
            supertrend_match = False
            ema_match = False

            for indicator in signal.indicators:
                if indicator.name and "SUPERTREND" in indicator.name.upper():
                    if signal.direction == "ce" and indicator.signal == "ce":
                        supertrend_match = True
                        score += 25  # Heavy weight on SuperTrend
                    elif signal.direction == "pe" and indicator.signal == "pe":
                        supertrend_match = True
                        score += 25

                elif indicator.name and "EMA" in indicator.name.upper():
                    if signal.direction == "ce" and indicator.signal == "ce":
                        ema_match = True
                        score += 15
                    elif signal.direction == "pe" and indicator.signal == "pe":
                        ema_match = True
                        score += 15

        except Exception as e:
            logger.debug(f"Error calculating expiry day trend score: {e}")

        return min(score, 40.0)

    def calculate_options_score_expiry(self, signal) -> float:
        """
        Calculate options market score for expiry day (0-30 points).
        Focus on OI and volume for liquid options.
        """
        score = 0.0

        try:
            if not signal.recommended_option:
                return 0.0

            opt = signal.recommended_option

            # Open Interest check (0-15 points)
            if opt.oi:
                if opt.oi > 500000:
                    score += 15
                elif opt.oi > 200000:
                    score += 10
                elif opt.oi > 100000:
                    score += 5

            # Volume check (0-15 points)
            if opt.volume:
                if opt.volume > 50000:
                    score += 15
                elif opt.volume > 10000:
                    score += 10
                elif opt.volume > 1000:
                    score += 5

        except Exception as e:
            logger.debug(f"Error calculating expiry day options score: {e}")

        return min(score, 30.0)

    def check_market_filters(self) -> tuple[bool, str]:
        """Check if market conditions are suitable for trading."""
        try:
            from app.services.data_fetcher import get_data_fetcher

            fetcher = get_data_fetcher()
            vix_data = fetcher.get_vix_value()
            vix_value = float(vix_data.get("vix", 0)) if vix_data else 0

            # VIX filter (more lenient for expiry)
            if vix_value > self.MAX_VIX_FOR_TRADING:
                return False, f"VIX too high: {vix_value:.1f} > {self.MAX_VIX_FOR_TRADING}"

        except Exception as e:
            logger.debug(f"Error checking market filters: {e}")

        return True, "Market conditions OK"

    async def generate_signal(self, df: pd.DataFrame, option_chain: list = None) -> Optional[TradeSignal]:
        """
        Generate aggressive signal for expiry day trading.

        Args:
            df: DataFrame with 5-minute OHLCV data
            option_chain: List of options in current chain

        Returns:
            TradeSignal if conditions met, None otherwise
        """
        now = datetime.now()

        # ========================================
        # PHASE 1: TRADING WINDOW CHECK
        # ========================================
        in_window, window_reason = self.is_in_trading_window()
        if not in_window:
            logger.debug(f"EXPIRY [{now.strftime('%H:%M')}] Outside window: {window_reason}")
            return None

        # ========================================
        # PHASE 2: MARKET CONDITIONS CHECK
        # ========================================
        market_ok, market_reason = self.check_market_filters()
        if not market_ok:
            logger.warning(f"EXPIRY [{now.strftime('%H:%M')}] Market not suitable: {market_reason}")
            return None

        # ========================================
        # PHASE 3: GENERATE BASE SIGNAL (BYPASS TIME CHECKS)
        # ========================================
        try:
            # Call base engine's analyze with bypass for expiry day trading window
            signal = self.base_engine.analyze(df=df, option_chain=option_chain, bypass_time_check=True)

            if not signal or not signal.recommended_option:
                logger.debug(f"EXPIRY [{now.strftime('%H:%M')}] No base signal generated")
                return None

            # ========================================
            # PHASE 4: CALCULATE EXPIRY DAY METRICS
            # ========================================
            metrics = ExpiryDayMetrics()

            # Calculate component scores (simplified & aggressive)
            metrics.momentum_score = self.calculate_momentum_score_expiry(signal)
            metrics.trend_score = self.calculate_trend_score_expiry(signal)
            metrics.options_score = self.calculate_options_score_expiry(signal)

            # Total confidence
            metrics.final_confidence = metrics.calculate_total()

            logger.info(
                f"EXPIRY [{now.strftime('%H:%M')}] SIGNAL QUALITY SCORE\n"
                f"  Momentum: {metrics.momentum_score:.0f}/30 | "
                f"Trend: {metrics.trend_score:.0f}/40 | "
                f"Options: {metrics.options_score:.0f}/30\n"
                f"  ► FINAL CONFIDENCE: {metrics.final_confidence:.0f}%"
            )

            # ========================================
            # PHASE 5: BLENDED CONFIDENCE CHECK
            # ========================================

            # Use BLENDED confidence: 50% base engine + 50% component score
            # Keeps some institutional quality while being aggressive
            blended_confidence = (signal.confidence * 0.5) + (metrics.final_confidence * 0.5)

            if blended_confidence >= self.CONFIDENCE_THRESHOLD_TRADE:
                logger.info(
                    f"EXPIRY [{now.strftime('%H:%M')}] ✓ AGGRESSIVE SIGNAL - EXECUTING ✓\n"
                    f"  Direction: {signal.direction.upper()} | "
                    f"Strike: {signal.recommended_option.strike} | "
                    f"Blended Confidence: {blended_confidence:.0f}% (>= {self.CONFIDENCE_THRESHOLD_TRADE}%) | "
                    f"OI: {signal.recommended_option.oi/1000000:.1f}M | "
                    f"Volume: {signal.recommended_option.volume/1000:.0f}K\n"
                    f"  Confidence: Base Engine {signal.confidence:.0f}% + Component Score {metrics.final_confidence:.0f}% = Blended {blended_confidence:.0f}%\n"
                    f"  → EXPIRY DAY: Tight 0.75% SL, 10%+ profit target"
                )

                # Set aggressive targets for expiry day
                signal.stop_loss = signal.entry_price * (0.9925 if signal.direction == "ce" else 1.0075)
                signal.target_1 = signal.entry_price * 1.10  # 10% target

                # Update state
                self.signal_state.consecutive_low_confidence = 0
                self.signal_state.last_signal_time = now
                self.signal_state.signals_today += 1
                self.signal_state.last_signal_id = f"{signal.direction}_{signal.recommended_option.strike}"

                logger.info(
                    f"EXPIRY TRADE DETAILS:\n"
                    f"  Entry: {signal.entry_price:.2f} | "
                    f"SL: {signal.stop_loss:.2f} (0.75%) | "
                    f"Target: {signal.target_1:.2f} (10%)\n"
                    f"  Signals generated today: {self.signal_state.signals_today}"
                )

                return signal

            elif metrics.final_confidence >= self.CONFIDENCE_THRESHOLD_MONITOR:
                logger.warning(
                    f"EXPIRY [{now.strftime('%H:%M')}] ⚠️ MONITOR - Blended Confidence {blended_confidence:.0f}% "
                    f"(< {self.CONFIDENCE_THRESHOLD_TRADE}% threshold). Waiting for stronger signal."
                )
                self.signal_state.consecutive_low_confidence += 1
                return None

            else:
                logger.debug(
                    f"EXPIRY [{now.strftime('%H:%M')}] SKIP - Confidence {metrics.final_confidence:.0f}% "
                    f"too low. Wait for next opportunity."
                )
                return None

        except Exception as e:
            logger.error(f"Error generating expiry day signal: {e}", exc_info=True)
            return None


# Singleton instance
_signal_engine_expiry_day: Optional[SignalExpiryDayEngine] = None


def get_signal_engine_expiry_day() -> SignalExpiryDayEngine:
    """Get or create singleton expiry day signal engine."""
    global _signal_engine_expiry_day
    if _signal_engine_expiry_day is None:
        _signal_engine_expiry_day = SignalExpiryDayEngine()
    return _signal_engine_expiry_day
