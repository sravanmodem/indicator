"""
Options-Specific Indicators
PCR, Max Pain, OI Analysis, GEX calculations
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class PCRResult:
    """Put-Call Ratio result."""
    pcr_oi: float          # PCR based on OI
    pcr_volume: float      # PCR based on volume
    sentiment: str         # extreme_fear, fear, neutral, greed, extreme_greed
    signal: str           # CE, PE, or neutral
    pcr_change: float     # Change from previous


def calculate_pcr(
    total_put_oi: int,
    total_call_oi: int,
    put_volume: int = 0,
    call_volume: int = 0,
    previous_pcr: float | None = None,
) -> PCRResult:
    """
    Calculate Put-Call Ratio.

    PCR is a contrarian sentiment indicator:
    - PCR > 1.3 = Extreme fear, potential bottom (CE opportunity)
    - PCR < 0.7 = Extreme greed, potential top (PE opportunity)

    Args:
        total_put_oi: Total put open interest
        total_call_oi: Total call open interest
        put_volume: Put trading volume
        call_volume: Call trading volume
        previous_pcr: Previous PCR for change calculation

    Returns:
        PCRResult with ratio and signals
    """
    pcr_oi = total_put_oi / total_call_oi if total_call_oi > 0 else 0
    pcr_volume = put_volume / call_volume if call_volume > 0 else 0

    # Determine sentiment
    if pcr_oi > 1.5:
        sentiment = "extreme_fear"
        signal = "CE"  # Contrarian bullish
    elif pcr_oi > 1.2:
        sentiment = "fear"
        signal = "CE"
    elif pcr_oi < 0.5:
        sentiment = "extreme_greed"
        signal = "PE"  # Contrarian bearish
    elif pcr_oi < 0.7:
        sentiment = "greed"
        signal = "PE"
    else:
        sentiment = "neutral"
        signal = "neutral"

    pcr_change = pcr_oi - previous_pcr if previous_pcr else 0

    return PCRResult(
        pcr_oi=round(pcr_oi, 3),
        pcr_volume=round(pcr_volume, 3),
        sentiment=sentiment,
        signal=signal,
        pcr_change=round(pcr_change, 3),
    )


@dataclass
class MaxPainResult:
    """Max Pain calculation result."""
    max_pain_strike: int
    max_pain_value: float    # Total pain at max pain strike
    call_pain: dict[int, float]
    put_pain: dict[int, float]
    total_pain: dict[int, float]
    distance_from_spot: float  # Percentage
    magnet_direction: str    # up, down, or at_pain


def calculate_max_pain(
    option_chain: list[dict],
    spot_price: float,
) -> MaxPainResult:
    """
    Calculate Max Pain strike.

    Max Pain is where option buyers lose maximum money:
    - Spot below Max Pain = Bullish magnet (CE)
    - Spot above Max Pain = Bearish magnet (PE)
    - Most effective near expiry

    Args:
        option_chain: List of dicts with strike, ce_oi, pe_oi
        spot_price: Current spot price

    Returns:
        MaxPainResult with max pain strike and analysis
    """
    call_pain: dict[int, float] = {}
    put_pain: dict[int, float] = {}
    total_pain: dict[int, float] = {}

    strikes = [opt["strike"] for opt in option_chain]

    for strike in strikes:
        call_pain[strike] = 0
        put_pain[strike] = 0

        # Calculate pain for call buyers if price expires at this strike
        for opt in option_chain:
            opt_strike = opt["strike"]
            ce_oi = opt.get("ce_oi", opt.get("ce", {}).get("oi", 0))
            pe_oi = opt.get("pe_oi", opt.get("pe", {}).get("oi", 0))

            # Call buyer pain: max(0, opt_strike - expiry_price) * OI
            if opt_strike < strike:
                call_pain[strike] += (strike - opt_strike) * ce_oi

            # Put buyer pain: max(0, expiry_price - opt_strike) * OI
            if opt_strike > strike:
                put_pain[strike] += (opt_strike - strike) * pe_oi

        total_pain[strike] = call_pain[strike] + put_pain[strike]

    # Find max pain (strike with highest total pain)
    max_pain_strike = max(total_pain, key=total_pain.get)
    max_pain_value = total_pain[max_pain_strike]

    # Distance from spot
    distance_pct = ((max_pain_strike - spot_price) / spot_price) * 100

    # Magnet direction
    if distance_pct > 0.5:
        magnet_direction = "up"  # Spot should move up toward max pain
    elif distance_pct < -0.5:
        magnet_direction = "down"  # Spot should move down toward max pain
    else:
        magnet_direction = "at_pain"

    return MaxPainResult(
        max_pain_strike=max_pain_strike,
        max_pain_value=max_pain_value,
        call_pain=call_pain,
        put_pain=put_pain,
        total_pain=total_pain,
        distance_from_spot=round(distance_pct, 2),
        magnet_direction=magnet_direction,
    )


@dataclass
class OIAnalysisResult:
    """OI Analysis result."""
    interpretation: str      # long_buildup, short_buildup, long_unwinding, short_covering
    signal: str             # CE, PE, book_ce, book_pe
    strength: str           # weak, moderate, strong
    call_wall: int | None   # Strike with highest call OI (resistance)
    put_wall: int | None    # Strike with highest put OI (support)
    oi_change_trend: str    # increasing, decreasing, stable


def analyze_oi_change(
    price_change: float,
    oi_change: float,
    option_chain: list[dict] | None = None,
) -> OIAnalysisResult:
    """
    Analyze Open Interest changes.

    OI interpretation:
    - Price up + OI up = Long buildup (CE)
    - Price up + OI down = Short covering (Book CE)
    - Price down + OI up = Short buildup (PE)
    - Price down + OI down = Long unwinding (Book PE)

    Args:
        price_change: Percentage price change
        oi_change: Percentage OI change
        option_chain: Optional chain data for wall detection

    Returns:
        OIAnalysisResult with interpretation and signals
    """
    # Determine interpretation
    if price_change > 0 and oi_change > 0:
        interpretation = "long_buildup"
        signal = "CE"
    elif price_change > 0 and oi_change < 0:
        interpretation = "short_covering"
        signal = "book_ce"  # Weak rally
    elif price_change < 0 and oi_change > 0:
        interpretation = "short_buildup"
        signal = "PE"
    else:  # price_change < 0 and oi_change < 0
        interpretation = "long_unwinding"
        signal = "book_pe"  # Weak decline

    # Strength based on magnitude
    oi_magnitude = abs(oi_change)
    if oi_magnitude > 10:
        strength = "strong"
    elif oi_magnitude > 5:
        strength = "moderate"
    else:
        strength = "weak"

    # OI trend
    if oi_change > 2:
        oi_change_trend = "increasing"
    elif oi_change < -2:
        oi_change_trend = "decreasing"
    else:
        oi_change_trend = "stable"

    # Find walls if chain provided
    call_wall = None
    put_wall = None

    if option_chain:
        max_call_oi = 0
        max_put_oi = 0

        for opt in option_chain:
            ce_oi = opt.get("ce_oi", opt.get("ce", {}).get("oi", 0))
            pe_oi = opt.get("pe_oi", opt.get("pe", {}).get("oi", 0))

            if ce_oi > max_call_oi:
                max_call_oi = ce_oi
                call_wall = opt["strike"]

            if pe_oi > max_put_oi:
                max_put_oi = pe_oi
                put_wall = opt["strike"]

    return OIAnalysisResult(
        interpretation=interpretation,
        signal=signal,
        strength=strength,
        call_wall=call_wall,
        put_wall=put_wall,
        oi_change_trend=oi_change_trend,
    )


@dataclass
class GEXResult:
    """Gamma Exposure result."""
    total_gex: float
    gex_by_strike: dict[int, float]
    flip_point: int | None    # Price where GEX flips sign
    environment: str          # positive_gamma, negative_gamma, neutral
    expected_behavior: str    # mean_reversion, trending, volatile


def calculate_gex(
    option_chain: list[dict],
    spot_price: float,
) -> GEXResult:
    """
    Calculate Gamma Exposure (GEX).

    GEX shows market maker hedging behavior:
    - Positive GEX = MM hedge against moves (mean reversion)
    - Negative GEX = MM amplify moves (trending/volatile)

    Args:
        option_chain: List with strike, ce_oi, pe_oi, ce_gamma, pe_gamma
        spot_price: Current spot price

    Returns:
        GEXResult with GEX analysis
    """
    gex_by_strike: dict[int, float] = {}
    total_gex = 0

    for opt in option_chain:
        strike = opt["strike"]

        # Get gamma values (simplified if not provided)
        ce_gamma = opt.get("ce_gamma", 0.01 if abs(strike - spot_price) < 100 else 0.005)
        pe_gamma = opt.get("pe_gamma", 0.01 if abs(strike - spot_price) < 100 else 0.005)

        ce_oi = opt.get("ce_oi", opt.get("ce", {}).get("oi", 0))
        pe_oi = opt.get("pe_oi", opt.get("pe", {}).get("oi", 0))

        # Contract size (NIFTY = 25)
        contract_size = 25

        # GEX = (Call Gamma * Call OI - Put Gamma * Put OI) * Contract Size * Spot^2 / 100
        strike_gex = (ce_gamma * ce_oi - pe_gamma * pe_oi) * contract_size * spot_price ** 2 / 100

        gex_by_strike[strike] = strike_gex
        total_gex += strike_gex

    # Find flip point (where cumulative GEX changes sign)
    cumulative = 0
    flip_point = None
    sorted_strikes = sorted(gex_by_strike.keys())

    for strike in sorted_strikes:
        prev_cumulative = cumulative
        cumulative += gex_by_strike[strike]

        if prev_cumulative * cumulative < 0:  # Sign changed
            flip_point = strike
            break

    # Environment determination
    if total_gex > 1e9:
        environment = "positive_gamma"
        expected_behavior = "mean_reversion"
    elif total_gex < -1e9:
        environment = "negative_gamma"
        expected_behavior = "trending"
    else:
        environment = "neutral"
        expected_behavior = "volatile"

    return GEXResult(
        total_gex=total_gex,
        gex_by_strike=gex_by_strike,
        flip_point=flip_point,
        environment=environment,
        expected_behavior=expected_behavior,
    )


@dataclass
class VIXAnalysis:
    """VIX analysis result."""
    level: str              # very_low, low, normal, elevated, high, extreme
    option_premium_state: str  # cheap, fair, expensive, very_expensive
    strategy_suggestion: str
    spike_detected: bool
    contango: bool          # Near-term < Long-term (normal)


def analyze_vix(
    vix_value: float,
    vix_change_pct: float = 0,
    vix_52w_low: float = 10,
    vix_52w_high: float = 40,
) -> VIXAnalysis:
    """
    Analyze India VIX for option trading.

    VIX implications:
    - Low VIX = Options cheap, good for buying
    - High VIX = Options expensive, avoid buying
    - VIX spike = Fear, potential contrarian CE opportunity

    Args:
        vix_value: Current VIX value
        vix_change_pct: VIX change percentage
        vix_52w_low: 52-week low
        vix_52w_high: 52-week high

    Returns:
        VIXAnalysis with trading implications
    """
    # Level classification
    if vix_value < 12:
        level = "very_low"
        option_premium_state = "cheap"
        strategy = "Options are cheap. Good for buying CE/PE on breakouts."
    elif vix_value < 15:
        level = "low"
        option_premium_state = "fair"
        strategy = "Normal conditions. Standard directional trades work."
    elif vix_value < 20:
        level = "normal"
        option_premium_state = "fair"
        strategy = "Balanced environment. Use technical signals."
    elif vix_value < 25:
        level = "elevated"
        option_premium_state = "expensive"
        strategy = "Premiums elevated. Be selective with option buying."
    elif vix_value < 35:
        level = "high"
        option_premium_state = "expensive"
        strategy = "High fear. Option buying risky due to IV crush potential."
    else:
        level = "extreme"
        option_premium_state = "very_expensive"
        strategy = "Extreme fear. Potential contrarian CE opportunity if reversal confirms."

    # Spike detection
    spike_detected = vix_change_pct > 15

    # Simple contango check (normally VIX should be below its recent range midpoint)
    midpoint = (vix_52w_low + vix_52w_high) / 2
    contango = vix_value < midpoint

    return VIXAnalysis(
        level=level,
        option_premium_state=option_premium_state,
        strategy_suggestion=strategy,
        spike_detected=spike_detected,
        contango=contango,
    )
