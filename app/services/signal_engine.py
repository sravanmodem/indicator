"""
Signal Generation Engine
Combines all indicators to generate CE/PE trading signals
"""

import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from loguru import logger

from app.indicators.trend import (
    calculate_supertrend,
    calculate_ema,
    calculate_vwap,
    calculate_adx,
)
from app.indicators.momentum import (
    calculate_rsi,
    calculate_macd,
    calculate_stochastic,
)
from app.indicators.volatility import (
    calculate_bollinger_bands,
    calculate_atr,
)
from app.indicators.volume import calculate_obv
from app.indicators.options import (
    calculate_pcr,
    calculate_max_pain,
    analyze_oi_change,
    analyze_vix,
)
from app.indicators.pivots import (
    calculate_pivot_points,
    calculate_cpr,
    calculate_camarilla,
    get_nearest_level,
)
from app.indicators.greeks import (
    calculate_greeks,
    calculate_expected_prices,
    estimate_implied_volatility,
    get_days_to_expiry,
)
from app.core.config import INDICATOR_PARAMS, SIGNAL_THRESHOLDS, OPTION_FILTERS, PREMIUM_TARGETS


class SignalType(Enum):
    """Signal type enumeration."""
    STRONG_CE = "strong_ce"
    CE = "ce"
    WEAK_CE = "weak_ce"
    NEUTRAL = "neutral"
    WEAK_PE = "weak_pe"
    PE = "pe"
    STRONG_PE = "strong_pe"


class TradingStyle(Enum):
    """Trading style enumeration."""
    SCALPING = "scalping"
    INTRADAY = "intraday"
    SWING = "swing"


@dataclass
class IndicatorSignal:
    """Individual indicator signal."""
    name: str
    value: Any
    signal: str  # ce, pe, neutral
    strength: float  # 0-1
    reason: str


@dataclass
class RecommendedOption:
    """Recommended option to buy."""
    symbol: str
    strike: float
    option_type: str  # CE or PE
    ltp: float
    oi: int
    volume: int
    bid: float
    ask: float
    reason: str

    # Greeks
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    delta_percent: float | None = None

    # Expected prices
    expected_at_target: float | None = None
    expected_at_stop: float | None = None
    expected_profit: float | None = None
    expected_loss: float | None = None
    price_tomorrow: float | None = None

    # Interpretations
    delta_interpretation: str | None = None
    theta_interpretation: str | None = None


@dataclass
class TradeSignal:
    """Complete trade signal."""
    timestamp: datetime
    instrument: str
    signal_type: SignalType
    direction: str  # CE or PE
    confidence: float  # 0-100
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float | None
    indicators: list[IndicatorSignal]
    supporting_factors: list[str]
    warning_factors: list[str]
    risk_reward: float
    trading_style: TradingStyle
    recommended_option: RecommendedOption | None = None


class SignalEngine:
    """
    Generates CE/PE signals by combining multiple indicators.

    Signal Logic:
    1. Trend indicators (SuperTrend, EMA, VWAP, ADX) - Primary direction
    2. Momentum indicators (RSI, MACD, Stochastic) - Entry timing
    3. Options indicators (OI, PCR, Max Pain) - Institutional flow
    4. Support/Resistance (Pivots, CPR) - Price levels
    5. Volatility (BB, ATR, VIX) - Risk management
    """

    def __init__(self, trading_style: TradingStyle = TradingStyle.INTRADAY):
        self.trading_style = trading_style
        self.params = INDICATOR_PARAMS
        self.thresholds = SIGNAL_THRESHOLDS

    def analyze(
        self,
        df: pd.DataFrame,
        option_chain: list[dict] | None = None,
        spot_price: float | None = None,
        vix_value: float | None = None,
        prev_day_ohlc: dict | None = None,
    ) -> TradeSignal | None:
        """
        Analyze market data and generate trade signal.

        Args:
            df: OHLCV DataFrame
            option_chain: Option chain data
            spot_price: Current spot price
            vix_value: Current VIX value
            prev_day_ohlc: Previous day OHLC for pivots

        Returns:
            TradeSignal if conditions met, None otherwise
        """
        if df.empty or len(df) < 50:
            logger.warning("Insufficient data for analysis")
            return None

        indicators: list[IndicatorSignal] = []
        supporting = []
        warnings = []

        current_price = df["Close"].iloc[-1]
        spot_price = spot_price or current_price

        # === TREND INDICATORS ===

        # 1. SuperTrend
        st = calculate_supertrend(
            df,
            period=self.params["supertrend"]["period"],
            multiplier=self.params["supertrend"]["multiplier"],
        )
        st_signal = "ce" if st.direction.iloc[-1] == 1 else "pe"
        st_strength = 0.8 if st_signal == ("ce" if df["Close"].iloc[-1] > st.supertrend.iloc[-1] else "pe") else 0.3

        indicators.append(IndicatorSignal(
            name="SuperTrend",
            value=st.supertrend.iloc[-1],
            signal=st_signal,
            strength=st_strength,
            reason=f"Price {'above' if st_signal == 'ce' else 'below'} SuperTrend ({st.supertrend.iloc[-1]:.2f})"
        ))

        if st.direction.iloc[-1] != st.direction.iloc[-2]:
            supporting.append(f"Fresh SuperTrend flip to {'bullish' if st_signal == 'ce' else 'bearish'}")

        # 2. EMA System
        ema = calculate_ema(
            df,
            fast_period=self.params["ema_fast"],
            slow_period=self.params["ema_slow"],
            trend_period=self.params["ema_trend"],
        )
        ema_signal = "ce" if ema.ema_fast.iloc[-1] > ema.ema_slow.iloc[-1] else "pe"

        indicators.append(IndicatorSignal(
            name="EMA Crossover",
            value={"fast": ema.ema_fast.iloc[-1], "slow": ema.ema_slow.iloc[-1]},
            signal=ema_signal,
            strength=0.7 if ema.trend_aligned.iloc[-1] else 0.4,
            reason=f"{self.params['ema_fast']} EMA {'>' if ema_signal == 'ce' else '<'} {self.params['ema_slow']} EMA"
        ))

        if ema.crossover.iloc[-1] != 0:
            supporting.append(f"EMA crossover {'bullish' if ema.crossover.iloc[-1] == 1 else 'bearish'}")

        # 3. VWAP
        vwap = calculate_vwap(df)
        vwap_signal = "ce" if current_price > vwap.vwap.iloc[-1] else "pe"
        vwap_extreme = abs(vwap.deviation.iloc[-1]) > 1.5

        indicators.append(IndicatorSignal(
            name="VWAP",
            value=vwap.vwap.iloc[-1],
            signal=vwap_signal,
            strength=0.6 if not vwap_extreme else 0.3,
            reason=f"Price {'above' if vwap_signal == 'ce' else 'below'} VWAP ({vwap.vwap.iloc[-1]:.2f})"
        ))

        if vwap_extreme:
            warnings.append(f"Price extended from VWAP ({vwap.deviation.iloc[-1]:.1f}%)")

        # 4. ADX
        adx = calculate_adx(df, period=self.params["adx_period"])
        adx_value = adx.adx.iloc[-1]
        adx_signal = "ce" if adx.plus_di.iloc[-1] > adx.minus_di.iloc[-1] else "pe"

        indicators.append(IndicatorSignal(
            name="ADX",
            value=adx_value,
            signal=adx_signal if adx_value > self.thresholds["adx_trend"] else "neutral",
            strength=min(adx_value / 50, 1.0),
            reason=f"ADX {adx_value:.1f} - {adx.trend_strength.iloc[-1]}"
        ))

        if adx_value < self.thresholds["adx_trend"]:
            warnings.append(f"Weak trend (ADX {adx_value:.1f} < 25)")

        # === MOMENTUM INDICATORS ===

        # 5. RSI
        rsi = calculate_rsi(df, period=self.params["rsi_period"])
        rsi_value = rsi.rsi.iloc[-1]

        if rsi_value > self.thresholds["rsi_neutral"]:
            rsi_signal = "ce"
        elif rsi_value < self.thresholds["rsi_neutral"]:
            rsi_signal = "pe"
        else:
            rsi_signal = "neutral"

        indicators.append(IndicatorSignal(
            name="RSI",
            value=rsi_value,
            signal=rsi_signal,
            strength=abs(rsi_value - 50) / 50,
            reason=f"RSI {rsi_value:.1f}"
        ))

        if rsi.divergence.iloc[-1] == 1:
            supporting.append("Bullish RSI divergence detected")
        elif rsi.divergence.iloc[-1] == -1:
            supporting.append("Bearish RSI divergence detected")

        # 6. MACD
        macd = calculate_macd(
            df,
            fast_period=self.params["macd_fast"],
            slow_period=self.params["macd_slow"],
            signal_period=self.params["macd_signal"],
        )
        macd_signal = "ce" if macd.macd_line.iloc[-1] > macd.signal_line.iloc[-1] else "pe"

        indicators.append(IndicatorSignal(
            name="MACD",
            value=macd.histogram.iloc[-1],
            signal=macd_signal,
            strength=0.6 if macd.zero_cross.iloc[-1] != 0 else 0.4,
            reason=f"MACD {'>' if macd_signal == 'ce' else '<'} Signal, Histogram: {macd.histogram.iloc[-1]:.2f}"
        ))

        if macd.crossover.iloc[-1] != 0:
            supporting.append(f"MACD crossover {'bullish' if macd.crossover.iloc[-1] == 1 else 'bearish'}")

        # 7. Stochastic
        stoch = calculate_stochastic(df)
        stoch_signal = "neutral"
        if stoch.crossover.iloc[-1] == 1:
            stoch_signal = "ce"
            supporting.append("Stochastic bullish crossover in oversold")
        elif stoch.crossover.iloc[-1] == -1:
            stoch_signal = "pe"
            supporting.append("Stochastic bearish crossover in overbought")

        indicators.append(IndicatorSignal(
            name="Stochastic",
            value={"k": stoch.k.iloc[-1], "d": stoch.d.iloc[-1]},
            signal=stoch_signal,
            strength=0.7 if stoch.crossover.iloc[-1] != 0 else 0.3,
            reason=f"%K: {stoch.k.iloc[-1]:.1f}, %D: {stoch.d.iloc[-1]:.1f}"
        ))

        # === VOLATILITY ===

        # 8. Bollinger Bands
        bb = calculate_bollinger_bands(df)

        if bb.squeeze.iloc[-1]:
            supporting.append("Bollinger Squeeze detected - Big move expected")

        if bb.lower_touch.iloc[-1]:
            supporting.append("Price at lower Bollinger Band")
        elif bb.upper_touch.iloc[-1]:
            supporting.append("Price at upper Bollinger Band")

        # 9. ATR for stops
        atr = calculate_atr(df)
        atr_value = atr.atr.iloc[-1]

        # === OPTIONS INDICATORS ===

        if option_chain:
            # 10. PCR
            total_ce_oi = sum(opt.get("ce", {}).get("oi", 0) for opt in option_chain)
            total_pe_oi = sum(opt.get("pe", {}).get("oi", 0) for opt in option_chain)
            pcr = calculate_pcr(total_pe_oi, total_ce_oi)

            indicators.append(IndicatorSignal(
                name="PCR",
                value=pcr.pcr_oi,
                signal=pcr.signal,
                strength=0.6 if pcr.sentiment in ["extreme_fear", "extreme_greed"] else 0.3,
                reason=f"PCR {pcr.pcr_oi:.2f} - {pcr.sentiment}"
            ))

            # 11. Max Pain
            max_pain = calculate_max_pain(option_chain, spot_price)

            indicators.append(IndicatorSignal(
                name="Max Pain",
                value=max_pain.max_pain_strike,
                signal="ce" if max_pain.magnet_direction == "up" else "pe" if max_pain.magnet_direction == "down" else "neutral",
                strength=0.5 if abs(max_pain.distance_from_spot) > 0.5 else 0.2,
                reason=f"Max Pain at {max_pain.max_pain_strike}, Spot {max_pain.distance_from_spot:+.1f}% away"
            ))

            # 12. OI Analysis
            oi_analysis = analyze_oi_change(
                price_change=((current_price / df["Close"].iloc[-2]) - 1) * 100,
                oi_change=0,  # Would need actual OI change data
                option_chain=option_chain,
            )

            if oi_analysis.call_wall:
                supporting.append(f"Call wall (resistance) at {oi_analysis.call_wall}")
            if oi_analysis.put_wall:
                supporting.append(f"Put wall (support) at {oi_analysis.put_wall}")

        # === VIX ANALYSIS ===

        if vix_value:
            vix_analysis = analyze_vix(vix_value)

            indicators.append(IndicatorSignal(
                name="VIX",
                value=vix_value,
                signal="neutral",
                strength=0.3,
                reason=f"VIX {vix_value:.1f} - {vix_analysis.option_premium_state}"
            ))

            if vix_analysis.level in ["high", "extreme"]:
                warnings.append(f"High VIX ({vix_value:.1f}) - Options expensive")
            elif vix_analysis.level == "very_low":
                supporting.append(f"Low VIX ({vix_value:.1f}) - Options cheap")

            if vix_analysis.spike_detected:
                supporting.append("VIX spike detected - Fear extreme")

        # === PIVOT LEVELS ===

        if prev_day_ohlc:
            pivots = calculate_pivot_points(
                prev_day_ohlc["high"],
                prev_day_ohlc["low"],
                prev_day_ohlc["close"],
            )
            cpr = calculate_cpr(
                prev_day_ohlc["high"],
                prev_day_ohlc["low"],
                prev_day_ohlc["close"],
            )
            camarilla = calculate_camarilla(
                prev_day_ohlc["high"],
                prev_day_ohlc["low"],
                prev_day_ohlc["close"],
                current_price,
            )

            levels = get_nearest_level(current_price, pivots, cpr, camarilla)

            if levels["nearest_support"]:
                supporting.append(f"Nearest support: {levels['nearest_support']:.2f} ({levels['support_distance']:.0f} pts)")
            if levels["nearest_resistance"]:
                supporting.append(f"Nearest resistance: {levels['nearest_resistance']:.2f} ({levels['resistance_distance']:.0f} pts)")

            if current_price > pivots.pivot:
                supporting.append(f"Price above daily pivot ({pivots.pivot:.2f})")
            else:
                supporting.append(f"Price below daily pivot ({pivots.pivot:.2f})")

        # === SIGNAL AGGREGATION ===

        ce_score = 0
        pe_score = 0
        total_weight = 0

        for ind in indicators:
            weight = ind.strength
            total_weight += weight

            if ind.signal == "ce":
                ce_score += weight
            elif ind.signal == "pe":
                pe_score += weight

        # Normalize scores
        if total_weight > 0:
            ce_confidence = (ce_score / total_weight) * 100
            pe_confidence = (pe_score / total_weight) * 100
        else:
            ce_confidence = pe_confidence = 0

        # Determine signal
        confidence_diff = abs(ce_confidence - pe_confidence)

        if confidence_diff < 10:
            signal_type = SignalType.NEUTRAL
            direction = "NEUTRAL"
        elif ce_confidence > pe_confidence:
            direction = "CE"
            if confidence_diff > 40:
                signal_type = SignalType.STRONG_CE
            elif confidence_diff > 20:
                signal_type = SignalType.CE
            else:
                signal_type = SignalType.WEAK_CE
        else:
            direction = "PE"
            if confidence_diff > 40:
                signal_type = SignalType.STRONG_PE
            elif confidence_diff > 20:
                signal_type = SignalType.PE
            else:
                signal_type = SignalType.WEAK_PE

        # Set confidence based on direction
        if direction == "CE":
            confidence = ce_confidence
        elif direction == "PE":
            confidence = pe_confidence
        else:
            confidence = 50

        # Find best option to buy FIRST (before calculating targets)
        recommended_option = None
        if option_chain and direction in ["CE", "PE"]:
            recommended_option = self._find_best_option(
                option_chain=option_chain,
                spot_price=spot_price,
                direction=direction,
                trading_style=self.trading_style,
            )

        # Calculate PREMIUM-BASED targets using Delta
        # Entry = Option LTP, Target/SL = % of premium based on Delta
        if recommended_option and recommended_option.ltp:
            entry_premium = recommended_option.ltp
            delta = abs(recommended_option.delta or 0.5)  # Default to 0.5 if no delta

            # Determine target/SL percentages based on Delta level
            if delta >= 0.6:
                # High Delta (0.6-0.9): 40% profit, 20% SL
                target_pct = PREMIUM_TARGETS["high_delta"]["target"]
                sl_pct = PREMIUM_TARGETS["high_delta"]["sl"]
            elif delta >= 0.4:
                # Medium Delta (0.4-0.6): 50% profit, 25% SL
                target_pct = PREMIUM_TARGETS["medium_delta"]["target"]
                sl_pct = PREMIUM_TARGETS["medium_delta"]["sl"]
            else:
                # Low Delta (0.2-0.4): 60% profit, 30% SL
                target_pct = PREMIUM_TARGETS["low_delta"]["target"]
                sl_pct = PREMIUM_TARGETS["low_delta"]["sl"]

            # Calculate premium-based levels
            # Target = Entry × (1 + target_pct), e.g., ₹50 × 1.50 = ₹75
            # SL = Entry × (1 - sl_pct), e.g., ₹50 × 0.75 = ₹37.50
            target_1 = entry_premium * (1 + target_pct)
            target_2 = entry_premium * (1 + target_pct * 1.5)  # 1.5x the target %
            stop_loss = entry_premium * (1 - sl_pct)

            # Risk/Reward based on premium
            risk = entry_premium - stop_loss  # e.g., ₹50 - ₹37.50 = ₹12.50
            reward = target_1 - entry_premium  # e.g., ₹75 - ₹50 = ₹25
            risk_reward = reward / risk if risk > 0 else 0

            # Use premium as entry price (not spot price)
            entry_price = entry_premium

            logger.info(
                f"Premium-based targets: Entry=₹{entry_premium:.1f}, "
                f"Delta={delta:.2f}, Target%={target_pct*100:.0f}%, SL%={sl_pct*100:.0f}% | "
                f"TGT=₹{target_1:.1f}, SL=₹{stop_loss:.1f}, R:R=1:{risk_reward:.1f}"
            )
        else:
            # Fallback to ATR-based spot levels if no option found
            atr_mult = 1.5 if self.trading_style == TradingStyle.SCALPING else 2.0
            entry_price = current_price

            if direction == "CE":
                stop_loss = current_price - (atr_value * atr_mult)
                target_1 = current_price + (atr_value * atr_mult * 1.5)
                target_2 = current_price + (atr_value * atr_mult * 2.5)
            elif direction == "PE":
                stop_loss = current_price + (atr_value * atr_mult)
                target_1 = current_price - (atr_value * atr_mult * 1.5)
                target_2 = current_price - (atr_value * atr_mult * 2.5)
            else:
                stop_loss = current_price
                target_1 = current_price
                target_2 = None

            risk = abs(current_price - stop_loss)
            reward = abs(target_1 - current_price)
            risk_reward = reward / risk if risk > 0 else 0

        return TradeSignal(
            timestamp=datetime.now(),
            instrument="NIFTY",
            signal_type=signal_type,
            direction=direction,
            confidence=round(confidence, 1),
            entry_price=round(entry_price, 2),
            stop_loss=round(stop_loss, 2),
            target_1=round(target_1, 2),
            target_2=round(target_2, 2) if target_2 else None,
            indicators=indicators,
            supporting_factors=supporting,
            warning_factors=warnings,
            risk_reward=round(risk_reward, 2),
            trading_style=self.trading_style,
            recommended_option=recommended_option,
        )

    def _find_best_option(
        self,
        option_chain: list[dict],
        spot_price: float,
        direction: str,
        trading_style: TradingStyle,
    ) -> RecommendedOption | None:
        """
        Find the best option to buy based on signal direction and premium range.
        Primary filter: Premium must be between 45-55 (configurable).
        Includes Greeks calculation for Delta-based target/SL.

        Selection criteria:
        1. Premium in target range (45-55 by default)
        2. Good OI (liquidity)
        3. Reasonable bid-ask spread
        4. Sufficient volume
        """
        try:
            option_type = "ce" if direction == "CE" else "pe"

            # Get premium filter settings
            premium_min = OPTION_FILTERS.get("premium_min", 45)
            premium_max = OPTION_FILTERS.get("premium_max", 50)
            min_oi = OPTION_FILTERS.get("min_oi", 10000)

            # Find ATM strike (for reference)
            atm_strike = None
            min_diff = float("inf")
            for opt in option_chain:
                diff = abs(opt["strike"] - spot_price)
                if diff < min_diff:
                    min_diff = diff
                    atm_strike = opt["strike"]

            if not atm_strike:
                return None

            # Find best option with premium in 45-50 range
            best_option = None
            best_score = -1
            candidates = []

            for opt in option_chain:
                opt_data = opt.get(option_type, {})
                if not opt_data or not opt_data.get("ltp"):
                    continue

                ltp = opt_data.get("ltp", 0)
                oi = opt_data.get("oi", 0)
                volume = opt_data.get("volume", 0)
                bid = opt_data.get("bid", 0)
                ask = opt_data.get("ask", 0)

                # PRIMARY FILTER: Premium must be in 45-50 range
                if ltp < premium_min or ltp > premium_max:
                    continue

                # Skip illiquid options
                if oi < min_oi:
                    continue

                candidates.append({
                    "opt": opt,
                    "ltp": ltp,
                    "oi": oi,
                    "volume": volume,
                    "bid": bid,
                    "ask": ask,
                })

            # If no options in 45-50 range, expand search slightly
            if not candidates:
                logger.warning(f"No options in {premium_min}-{premium_max} range, expanding search...")
                for opt in option_chain:
                    opt_data = opt.get(option_type, {})
                    if not opt_data or not opt_data.get("ltp"):
                        continue

                    ltp = opt_data.get("ltp", 0)
                    oi = opt_data.get("oi", 0)
                    volume = opt_data.get("volume", 0)
                    bid = opt_data.get("bid", 0)
                    ask = opt_data.get("ask", 0)

                    # Expanded range: 40-60
                    if ltp < 40 or ltp > 60:
                        continue

                    if oi < min_oi:
                        continue

                    candidates.append({
                        "opt": opt,
                        "ltp": ltp,
                        "oi": oi,
                        "volume": volume,
                        "bid": bid,
                        "ask": ask,
                    })

            # Score candidates
            for candidate in candidates:
                opt = candidate["opt"]
                ltp = candidate["ltp"]
                oi = candidate["oi"]
                volume = candidate["volume"]
                bid = candidate["bid"]
                ask = candidate["ask"]
                opt_data = opt.get(option_type, {})

                # Calculate score
                spread = (ask - bid) / ltp if ltp > 0 else 1
                spread_score = max(0, 1 - spread * 10)  # Lower spread = higher score

                oi_score = min(oi / 500000, 1)  # Cap at 500k OI
                volume_score = min(volume / 50000, 1)  # Cap at 50k volume

                # Premium score - closer to ideal (47.5) is better
                ideal_premium = (premium_min + premium_max) / 2  # 47.5
                premium_diff = abs(ltp - ideal_premium)
                premium_score = max(0, 1 - (premium_diff / 10))  # Score based on distance from ideal

                # ATM gets small bonus
                atm_bonus = 0.1 if opt["strike"] == atm_strike else 0

                # Calculate total score (premium score is most important)
                score = (premium_score * 0.35 + spread_score * 0.25 + oi_score * 0.25 + volume_score * 0.1 + atm_bonus * 0.05)

                if score > best_score:
                    best_score = score
                    strike_type = "ATM" if opt["strike"] == atm_strike else (
                        "ITM" if (direction == "CE" and opt["strike"] < spot_price) or
                                 (direction == "PE" and opt["strike"] > spot_price)
                        else "OTM"
                    )

                    # Calculate Greeks
                    days_to_expiry = get_days_to_expiry()
                    time_to_expiry = days_to_expiry / 365
                    iv = estimate_implied_volatility(option_chain, spot_price)

                    greeks = calculate_greeks(
                        spot_price=spot_price,
                        strike_price=opt["strike"],
                        time_to_expiry=time_to_expiry,
                        risk_free_rate=0.065,  # 6.5% India risk-free rate
                        volatility=iv,
                        option_type=direction,
                        current_premium=ltp,
                    )

                    # Calculate premium-based targets using Delta
                    delta_abs = abs(greeks.delta) if greeks.delta else 0.5
                    if delta_abs >= 0.6:
                        target_pct = PREMIUM_TARGETS["high_delta"]["target"]
                        sl_pct = PREMIUM_TARGETS["high_delta"]["sl"]
                    elif delta_abs >= 0.4:
                        target_pct = PREMIUM_TARGETS["medium_delta"]["target"]
                        sl_pct = PREMIUM_TARGETS["medium_delta"]["sl"]
                    else:
                        target_pct = PREMIUM_TARGETS["low_delta"]["target"]
                        sl_pct = PREMIUM_TARGETS["low_delta"]["sl"]

                    # Premium-based target/SL
                    premium_target = ltp * (1 + target_pct)
                    premium_sl = ltp * (1 - sl_pct)
                    premium_profit = premium_target - ltp
                    premium_loss = ltp - premium_sl

                    # Calculate theta decay for tomorrow
                    theta_decay = abs(greeks.theta) if greeks.theta else 0
                    price_tomorrow = ltp - theta_decay

                    best_option = RecommendedOption(
                        symbol=opt_data.get("symbol", f"{opt['strike']}{direction}"),
                        strike=opt["strike"],
                        option_type=direction,
                        ltp=ltp,
                        oi=oi,
                        volume=volume,
                        bid=bid,
                        ask=ask,
                        reason=f"{strike_type} strike, Premium ₹{ltp:.0f}, Delta {delta_abs:.2f} → TGT {target_pct*100:.0f}%/SL {sl_pct*100:.0f}%",
                        # Greeks
                        delta=greeks.delta,
                        gamma=greeks.gamma,
                        theta=greeks.theta,
                        vega=greeks.vega,
                        delta_percent=greeks.delta_percent,
                        # Premium-based expected prices (not spot-based)
                        expected_at_target=premium_target,
                        expected_at_stop=premium_sl,
                        expected_profit=premium_profit,
                        expected_loss=premium_loss,
                        price_tomorrow=price_tomorrow,
                        # Interpretations
                        delta_interpretation=greeks.delta_interpretation,
                        theta_interpretation=greeks.theta_interpretation,
                    )

            return best_option

        except Exception as e:
            logger.error(f"Error finding best option: {e}")
            return None

    def check_exit_signal(
        self,
        df: pd.DataFrame,
        position_type: str,  # "CE" or "PE"
        entry_price: float,
        current_ltp: float,
    ) -> dict | None:
        """
        Check if an exit signal should be generated for an open position.

        Exit conditions:
        1. Signal reversal (CE position gets PE signal or vice versa)
        2. Momentum exhaustion (RSI overbought/oversold reversal)
        3. Trend reversal (SuperTrend flip)
        4. Profit target reached (based on signal)
        5. Stop loss hit

        Args:
            df: OHLCV DataFrame
            position_type: "CE" or "PE" - the current position type
            entry_price: Entry price of the position
            current_ltp: Current LTP of the option

        Returns:
            Exit signal dict with reason, or None if no exit
        """
        try:
            if df.empty or len(df) < 20:
                return None

            # Normalize column names to Title case (High, Low, Close, Open, Volume)
            df = df.copy()
            column_map = {
                'open': 'Open', 'high': 'High', 'low': 'Low',
                'close': 'Close', 'volume': 'Volume',
                'OPEN': 'Open', 'HIGH': 'High', 'LOW': 'Low',
                'CLOSE': 'Close', 'VOLUME': 'Volume',
            }
            df.rename(columns={k: v for k, v in column_map.items() if k in df.columns}, inplace=True)

            # Calculate indicators - each returns a result object
            st_result = calculate_supertrend(df)
            rsi_result = calculate_rsi(df)
            macd_result = calculate_macd(df)
            ema_result = calculate_ema(df, fast_period=9, slow_period=21, trend_period=None)

            exit_reasons = []
            exit_score = 0

            # Get latest and previous values
            idx = len(df) - 1
            prev_idx = idx - 1 if idx > 0 else idx

            # 1. SuperTrend Reversal - Strong exit signal
            supertrend_dir = st_result.direction.iloc[idx]
            prev_supertrend_dir = st_result.direction.iloc[prev_idx]

            if position_type == "CE" and supertrend_dir == -1:
                exit_reasons.append("SuperTrend turned bearish")
                exit_score += 40
            elif position_type == "PE" and supertrend_dir == 1:
                exit_reasons.append("SuperTrend turned bullish")
                exit_score += 40

            # SuperTrend just flipped - immediate exit
            if supertrend_dir != prev_supertrend_dir and supertrend_dir != 0:
                exit_reasons.append("SuperTrend FLIP - Trend reversal")
                exit_score += 30

            # 2. RSI Exhaustion
            rsi = rsi_result.rsi.iloc[idx] if not pd.isna(rsi_result.rsi.iloc[idx]) else 50
            if position_type == "CE" and rsi > 75:
                exit_reasons.append(f"RSI overbought ({rsi:.0f}) - Momentum exhaustion")
                exit_score += 25
            elif position_type == "PE" and rsi < 25:
                exit_reasons.append(f"RSI oversold ({rsi:.0f}) - Momentum exhaustion")
                exit_score += 25

            # 3. MACD Crossover (Signal reversal)
            macd = macd_result.macd_line.iloc[idx] if not pd.isna(macd_result.macd_line.iloc[idx]) else 0
            macd_sig = macd_result.signal_line.iloc[idx] if not pd.isna(macd_result.signal_line.iloc[idx]) else 0
            prev_macd = macd_result.macd_line.iloc[prev_idx] if not pd.isna(macd_result.macd_line.iloc[prev_idx]) else 0
            prev_macd_sig = macd_result.signal_line.iloc[prev_idx] if not pd.isna(macd_result.signal_line.iloc[prev_idx]) else 0

            # MACD crossed below signal (bearish for CE)
            if position_type == "CE" and prev_macd > prev_macd_sig and macd < macd_sig:
                exit_reasons.append("MACD bearish crossover")
                exit_score += 20
            # MACD crossed above signal (bullish for PE)
            elif position_type == "PE" and prev_macd < prev_macd_sig and macd > macd_sig:
                exit_reasons.append("MACD bullish crossover")
                exit_score += 20

            # 4. EMA Crossover (using fast=9, slow=21)
            ema9 = ema_result.ema_fast.iloc[idx] if not pd.isna(ema_result.ema_fast.iloc[idx]) else 0
            ema21 = ema_result.ema_slow.iloc[idx] if not pd.isna(ema_result.ema_slow.iloc[idx]) else 0
            prev_ema9 = ema_result.ema_fast.iloc[prev_idx] if not pd.isna(ema_result.ema_fast.iloc[prev_idx]) else 0
            prev_ema21 = ema_result.ema_slow.iloc[prev_idx] if not pd.isna(ema_result.ema_slow.iloc[prev_idx]) else 0

            if position_type == "CE" and prev_ema9 > prev_ema21 and ema9 < ema21:
                exit_reasons.append("EMA bearish crossover (9 crossed below 21)")
                exit_score += 20
            elif position_type == "PE" and prev_ema9 < prev_ema21 and ema9 > ema21:
                exit_reasons.append("EMA bullish crossover (9 crossed above 21)")
                exit_score += 20

            # 5. Profit/Loss based exit
            pnl_percent = ((current_ltp - entry_price) / entry_price) * 100

            # STOP LOSS AT -10% - Exit immediately if loss exceeds 10%
            if pnl_percent <= -10:
                return {
                    "should_exit": True,
                    "exit_score": 100,
                    "reasons": [f"Stop Loss Hit ({pnl_percent:.1f}%)"],
                    "pnl_percent": pnl_percent,
                    "current_ltp": current_ltp,
                    "entry_price": entry_price,
                }

            # NO FIXED PROFIT TARGET - Let profits run
            # Only exit on indicator-based signals (trend reversal, etc.)
            if exit_score >= 40:
                return {
                    "should_exit": True,
                    "exit_score": exit_score,
                    "reasons": exit_reasons,
                    "pnl_percent": pnl_percent,
                    "current_ltp": current_ltp,
                    "entry_price": entry_price,
                }

            return None

        except Exception as e:
            logger.error(f"Error checking exit signal: {e}")
            return None


# Singleton instance
_signal_engine: SignalEngine | None = None


def get_signal_engine(style: TradingStyle = TradingStyle.INTRADAY) -> SignalEngine:
    """Get or create signal engine instance."""
    global _signal_engine
    if _signal_engine is None or _signal_engine.trading_style != style:
        _signal_engine = SignalEngine(style)
    return _signal_engine
