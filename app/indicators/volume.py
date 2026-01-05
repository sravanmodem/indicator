"""
Volume-Based Indicators
OBV, Volume Profile calculations
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any


@dataclass
class OBVResult:
    """OBV calculation result."""
    obv: pd.Series
    obv_ema: pd.Series
    divergence: pd.Series     # 1 = bullish, -1 = bearish
    trend: pd.Series          # up, down, flat
    breakout: pd.Series       # OBV breaking trendline


def calculate_obv(
    df: pd.DataFrame,
    ema_period: int = 20,
) -> OBVResult:
    """
    Calculate OBV (On Balance Volume).

    OBV shows volume flow:
    - OBV rising + Price rising = Confirmed uptrend (CE)
    - OBV falling + Price falling = Confirmed downtrend (PE)
    - Divergence = Smart money positioning

    Args:
        df: DataFrame with Close and Volume columns
        ema_period: EMA period for OBV smoothing

    Returns:
        OBVResult with OBV values and signals
    """
    close = df["Close"]
    volume = df["Volume"]

    # Calculate OBV
    obv = pd.Series(0, index=df.index, dtype=float)

    for i in range(1, len(df)):
        if close.iloc[i] > close.iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i - 1]

    # OBV EMA for trend
    obv_ema = obv.ewm(span=ema_period, adjust=False).mean()

    # OBV trend
    trend = pd.Series("flat", index=df.index)
    trend[obv > obv_ema] = "up"
    trend[obv < obv_ema] = "down"

    # Divergence detection
    divergence = pd.Series(0, index=df.index)
    lookback = 10

    for i in range(lookback, len(df)):
        price_window = close.iloc[i - lookback:i + 1]
        obv_window = obv.iloc[i - lookback:i + 1]

        # Bullish divergence: price lower low, OBV higher low
        price_ll = price_window.iloc[-1] < price_window.min() * 1.001
        obv_hl = obv_window.iloc[-1] > obv_window.min()

        if price_ll and obv_hl:
            divergence.iloc[i] = 1

        # Bearish divergence: price higher high, OBV lower high
        price_hh = price_window.iloc[-1] > price_window.max() * 0.999
        obv_lh = obv_window.iloc[-1] < obv_window.max()

        if price_hh and obv_lh:
            divergence.iloc[i] = -1

    # OBV breakout (simplified - crosses above recent high)
    obv_high = obv.rolling(window=20).max()
    obv_low = obv.rolling(window=20).min()

    breakout = pd.Series(0, index=df.index)
    breakout[obv > obv_high.shift(1)] = 1   # Bullish breakout
    breakout[obv < obv_low.shift(1)] = -1   # Bearish breakout

    return OBVResult(
        obv=obv,
        obv_ema=obv_ema,
        divergence=divergence,
        trend=trend,
        breakout=breakout,
    )


@dataclass
class VolumeProfileResult:
    """Volume Profile calculation result."""
    poc: float                # Point of Control price
    vah: float                # Value Area High
    val: float                # Value Area Low
    profile: dict[float, int]  # Price -> Volume mapping
    hvn: list[float]          # High Volume Nodes
    lvn: list[float]          # Low Volume Nodes


def calculate_volume_profile(
    df: pd.DataFrame,
    num_bins: int = 50,
    value_area_pct: float = 0.70,
) -> VolumeProfileResult:
    """
    Calculate Volume Profile.

    Volume Profile shows volume at each price level:
    - POC (Point of Control) = Price with most volume (strong S/R)
    - VAH/VAL = Value area boundaries
    - HVN = High volume nodes (price consolidation)
    - LVN = Low volume nodes (price moves fast through)

    Args:
        df: DataFrame with OHLCV data
        num_bins: Number of price bins
        value_area_pct: Percentage of volume for value area

    Returns:
        VolumeProfileResult with profile data
    """
    high = df["High"].max()
    low = df["Low"].min()
    volume = df["Volume"]

    # Create price bins
    price_range = high - low
    bin_size = price_range / num_bins
    bins = [low + (i * bin_size) for i in range(num_bins + 1)]

    # Distribute volume to bins
    profile: dict[float, int] = {bins[i]: 0 for i in range(num_bins)}

    for idx in df.index:
        row_high = df.loc[idx, "High"]
        row_low = df.loc[idx, "Low"]
        row_volume = df.loc[idx, "Volume"]

        # Find bins this candle spans
        for i in range(num_bins):
            bin_low = bins[i]
            bin_high = bins[i + 1]

            # Check overlap
            if row_low <= bin_high and row_high >= bin_low:
                # Proportional volume distribution
                overlap_low = max(row_low, bin_low)
                overlap_high = min(row_high, bin_high)
                candle_range = row_high - row_low if row_high != row_low else 1

                overlap_pct = (overlap_high - overlap_low) / candle_range
                profile[bin_low] += int(row_volume * overlap_pct)

    # Find POC (price with highest volume)
    poc_price = max(profile, key=profile.get)
    poc = poc_price + (bin_size / 2)  # Center of bin

    # Calculate Value Area
    total_volume = sum(profile.values())
    target_volume = total_volume * value_area_pct

    # Start from POC and expand
    sorted_bins = sorted(profile.items(), key=lambda x: x[1], reverse=True)
    value_area_volume = 0
    value_area_prices = []

    for price, vol in sorted_bins:
        if value_area_volume >= target_volume:
            break
        value_area_prices.append(price)
        value_area_volume += vol

    vah = max(value_area_prices) + bin_size
    val = min(value_area_prices)

    # Find HVN and LVN
    avg_volume = total_volume / num_bins
    hvn = [p + (bin_size / 2) for p, v in profile.items() if v > avg_volume * 1.5]
    lvn = [p + (bin_size / 2) for p, v in profile.items() if v < avg_volume * 0.5 and v > 0]

    return VolumeProfileResult(
        poc=poc,
        vah=vah,
        val=val,
        profile=profile,
        hvn=sorted(hvn),
        lvn=sorted(lvn),
    )


def analyze_volume_bar(
    df: pd.DataFrame,
    current_idx: int,
    avg_period: int = 20,
) -> dict[str, Any]:
    """
    Analyze current volume bar characteristics.

    Volume Spread Analysis (VSA) patterns:
    - High volume + narrow spread = Absorption
    - Low volume + wide spread = No demand/supply
    - High volume + close at high = Strong demand

    Args:
        df: DataFrame with OHLCV data
        current_idx: Index of bar to analyze
        avg_period: Period for average volume

    Returns:
        Dictionary with volume analysis
    """
    if current_idx < avg_period:
        return {"valid": False}

    row = df.iloc[current_idx]
    avg_volume = df["Volume"].iloc[current_idx - avg_period:current_idx].mean()
    avg_range = (df["High"] - df["Low"]).iloc[current_idx - avg_period:current_idx].mean()

    current_volume = row["Volume"]
    current_range = row["High"] - row["Low"]
    close_position = (row["Close"] - row["Low"]) / current_range if current_range > 0 else 0.5

    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
    range_ratio = current_range / avg_range if avg_range > 0 else 1

    # Determine pattern
    pattern = "neutral"

    if volume_ratio > 1.5 and range_ratio < 0.7:
        # High volume, narrow range = Absorption
        if close_position > 0.6:
            pattern = "no_supply"  # Bullish
        elif close_position < 0.4:
            pattern = "no_demand"  # Bearish

    elif volume_ratio > 2 and close_position < 0.3:
        pattern = "stopping_volume"  # Potential reversal up

    elif volume_ratio > 2 and close_position > 0.7 and row["Close"] > row["Open"]:
        pattern = "upthrust"  # Potential reversal down (if at resistance)

    return {
        "valid": True,
        "volume_ratio": round(volume_ratio, 2),
        "range_ratio": round(range_ratio, 2),
        "close_position": round(close_position, 2),
        "pattern": pattern,
        "is_high_volume": volume_ratio > 1.5,
        "is_wide_range": range_ratio > 1.3,
    }
