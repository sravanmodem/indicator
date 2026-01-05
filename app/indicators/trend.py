"""
Trend Indicators
SuperTrend, EMA, VWAP, ADX calculations
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Literal


@dataclass
class SuperTrendResult:
    """SuperTrend calculation result."""
    supertrend: pd.Series
    direction: pd.Series  # 1 = bullish, -1 = bearish
    upper_band: pd.Series
    lower_band: pd.Series


def calculate_supertrend(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0,
) -> SuperTrendResult:
    """
    Calculate SuperTrend indicator.

    SuperTrend is an ATR-based trend-following indicator.
    - Price above SuperTrend = Bullish (CE signal)
    - Price below SuperTrend = Bearish (PE signal)

    Args:
        df: DataFrame with OHLC data (columns: Open, High, Low, Close)
        period: ATR period
        multiplier: ATR multiplier

    Returns:
        SuperTrendResult with indicator values
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    # Calculate ATR
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    # Calculate basic bands
    hl2 = (high + low) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)

    # Initialize SuperTrend
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    for i in range(period, len(df)):
        # Previous values
        prev_supertrend = supertrend.iloc[i - 1] if i > period else lower_band.iloc[i]
        prev_upper = upper_band.iloc[i - 1] if i > period else upper_band.iloc[i]
        prev_lower = lower_band.iloc[i - 1] if i > period else lower_band.iloc[i]

        # Current bands
        curr_upper = upper_band.iloc[i]
        curr_lower = lower_band.iloc[i]

        # Upper band logic
        if curr_upper < prev_upper or close.iloc[i - 1] > prev_upper:
            upper_band.iloc[i] = curr_upper
        else:
            upper_band.iloc[i] = prev_upper

        # Lower band logic
        if curr_lower > prev_lower or close.iloc[i - 1] < prev_lower:
            lower_band.iloc[i] = curr_lower
        else:
            lower_band.iloc[i] = prev_lower

        # SuperTrend logic
        if pd.isna(prev_supertrend):
            supertrend.iloc[i] = lower_band.iloc[i]
            direction.iloc[i] = 1
        elif prev_supertrend == prev_upper:
            if close.iloc[i] > upper_band.iloc[i]:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            else:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
        else:
            if close.iloc[i] < lower_band.iloc[i]:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1

    return SuperTrendResult(
        supertrend=supertrend,
        direction=direction,
        upper_band=upper_band,
        lower_band=lower_band,
    )


@dataclass
class EMAResult:
    """EMA calculation result."""
    ema_fast: pd.Series
    ema_slow: pd.Series
    ema_trend: pd.Series | None
    crossover: pd.Series  # 1 = bullish cross, -1 = bearish cross, 0 = no cross
    trend_aligned: pd.Series  # True if all EMAs aligned


def calculate_ema(
    df: pd.DataFrame,
    fast_period: int = 9,
    slow_period: int = 21,
    trend_period: int | None = 50,
) -> EMAResult:
    """
    Calculate EMA crossover system.

    Triple EMA system for trend identification:
    - Fast > Slow > Trend = Strong bullish (CE)
    - Fast < Slow < Trend = Strong bearish (PE)

    Args:
        df: DataFrame with Close column
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        trend_period: Trend EMA period (optional)

    Returns:
        EMAResult with EMA values and crossover signals
    """
    close = df["Close"]

    ema_fast = close.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=slow_period, adjust=False).mean()
    ema_trend = close.ewm(span=trend_period, adjust=False).mean() if trend_period else None

    # Crossover detection
    prev_diff = (ema_fast.shift(1) - ema_slow.shift(1))
    curr_diff = (ema_fast - ema_slow)

    crossover = pd.Series(0, index=df.index)
    crossover[(prev_diff <= 0) & (curr_diff > 0)] = 1   # Bullish crossover
    crossover[(prev_diff >= 0) & (curr_diff < 0)] = -1  # Bearish crossover

    # Trend alignment
    if ema_trend is not None:
        trend_aligned = (
            ((ema_fast > ema_slow) & (ema_slow > ema_trend)) |
            ((ema_fast < ema_slow) & (ema_slow < ema_trend))
        )
    else:
        trend_aligned = pd.Series(False, index=df.index)

    return EMAResult(
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        ema_trend=ema_trend,
        crossover=crossover,
        trend_aligned=trend_aligned,
    )


@dataclass
class VWAPResult:
    """VWAP calculation result."""
    vwap: pd.Series
    upper_band_1: pd.Series
    upper_band_2: pd.Series
    lower_band_1: pd.Series
    lower_band_2: pd.Series
    deviation: pd.Series


def calculate_vwap(
    df: pd.DataFrame,
    anchor: Literal["day", "week", "month"] = "day",
    std_bands: list[float] = [1, 2],
) -> VWAPResult:
    """
    Calculate VWAP with standard deviation bands.

    VWAP = Volume Weighted Average Price
    - Price above VWAP = Bullish bias (CE)
    - Price below VWAP = Bearish bias (PE)
    - Bands indicate overbought/oversold

    Args:
        df: DataFrame with OHLCV data
        anchor: Reset period (day, week, month)
        std_bands: Standard deviation multipliers for bands

    Returns:
        VWAPResult with VWAP and bands
    """
    # Typical price
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    volume = df["Volume"]

    # For intraday, assume daily reset
    # In production, group by date for proper anchoring
    cumulative_tp_vol = (typical_price * volume).cumsum()
    cumulative_vol = volume.cumsum()

    vwap = cumulative_tp_vol / cumulative_vol

    # Standard deviation bands
    squared_diff = ((typical_price - vwap) ** 2 * volume).cumsum()
    variance = squared_diff / cumulative_vol
    std_dev = np.sqrt(variance)

    upper_band_1 = vwap + (std_bands[0] * std_dev)
    upper_band_2 = vwap + (std_bands[1] * std_dev) if len(std_bands) > 1 else upper_band_1
    lower_band_1 = vwap - (std_bands[0] * std_dev)
    lower_band_2 = vwap - (std_bands[1] * std_dev) if len(std_bands) > 1 else lower_band_1

    # Price deviation from VWAP
    deviation = (df["Close"] - vwap) / vwap * 100

    return VWAPResult(
        vwap=vwap,
        upper_band_1=upper_band_1,
        upper_band_2=upper_band_2,
        lower_band_1=lower_band_1,
        lower_band_2=lower_band_2,
        deviation=deviation,
    )


@dataclass
class ADXResult:
    """ADX calculation result."""
    adx: pd.Series
    plus_di: pd.Series
    minus_di: pd.Series
    trend_strength: pd.Series  # weak, developing, strong, very_strong, extreme


def calculate_adx(
    df: pd.DataFrame,
    period: int = 14,
) -> ADXResult:
    """
    Calculate ADX (Average Directional Index) with +DI/-DI.

    ADX measures trend STRENGTH (not direction):
    - ADX > 25 + (+DI > -DI) = Bullish trend (CE)
    - ADX > 25 + (-DI > +DI) = Bearish trend (PE)
    - ADX < 20 = No trend (avoid trading)

    Args:
        df: DataFrame with OHLC data
        period: ADX period

    Returns:
        ADXResult with ADX and DI values
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)

    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move

    # Smoothed values
    atr = tr.rolling(window=period).mean()
    plus_dm_smooth = plus_dm.rolling(window=period).mean()
    minus_dm_smooth = minus_dm.rolling(window=period).mean()

    # Directional Indicators
    plus_di = 100 * (plus_dm_smooth / atr)
    minus_di = 100 * (minus_dm_smooth / atr)

    # DX and ADX
    di_diff = abs(plus_di - minus_di)
    di_sum = plus_di + minus_di
    dx = 100 * (di_diff / di_sum)
    adx = dx.rolling(window=period).mean()

    # Trend strength classification
    trend_strength = pd.Series("weak", index=df.index)
    trend_strength[adx >= 20] = "developing"
    trend_strength[adx >= 25] = "strong"
    trend_strength[adx >= 40] = "very_strong"
    trend_strength[adx >= 50] = "extreme"

    return ADXResult(
        adx=adx,
        plus_di=plus_di,
        minus_di=minus_di,
        trend_strength=trend_strength,
    )
