"""
Momentum Indicators
RSI, MACD, Stochastic, CCI calculations
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Literal


@dataclass
class RSIResult:
    """RSI calculation result."""
    rsi: pd.Series
    oversold: pd.Series      # RSI < 30
    overbought: pd.Series    # RSI > 70
    bullish_zone: pd.Series  # RSI 40-80 (uptrend behavior)
    bearish_zone: pd.Series  # RSI 20-60 (downtrend behavior)
    divergence: pd.Series    # 1 = bullish, -1 = bearish, 0 = none


def calculate_rsi(
    df: pd.DataFrame,
    period: int = 14,
    oversold: float = 30,
    overbought: float = 70,
) -> RSIResult:
    """
    Calculate RSI (Relative Strength Index).

    RSI measures momentum on 0-100 scale:
    - RSI > 50 = Bullish momentum (CE bias)
    - RSI < 50 = Bearish momentum (PE bias)
    - Divergence = High probability reversal

    Args:
        df: DataFrame with Close column
        period: RSI period
        oversold: Oversold threshold
        overbought: Overbought threshold

    Returns:
        RSIResult with RSI values and signals
    """
    close = df["Close"]
    delta = close.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    # Use Wilder's smoothing
    for i in range(period, len(df)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]) / period

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Zones
    oversold_zone = rsi < oversold
    overbought_zone = rsi > overbought
    bullish_zone = (rsi >= 40) & (rsi <= 80)
    bearish_zone = (rsi >= 20) & (rsi <= 60)

    # Simple divergence detection
    divergence = pd.Series(0, index=df.index)

    # Look back 10 periods for divergence
    lookback = 10
    for i in range(lookback, len(df)):
        price_window = close.iloc[i - lookback:i + 1]
        rsi_window = rsi.iloc[i - lookback:i + 1]

        # Skip if rsi_window has NaN values
        if rsi_window.isna().any():
            continue

        # Bullish divergence: price lower low, RSI higher low
        rsi_min_val = rsi_window.min()
        if (price_window.iloc[-1] < price_window.min() * 1.001 and
            rsi_window.iloc[-1] > rsi_min_val and
            rsi.iloc[i] < 40):
            divergence.iloc[i] = 1

        # Bearish divergence: price higher high, RSI lower high
        rsi_max_val = rsi_window.max()
        if (price_window.iloc[-1] > price_window.max() * 0.999 and
              rsi_window.iloc[-1] < rsi_max_val and
              rsi.iloc[i] > 60):
            divergence.iloc[i] = -1

    return RSIResult(
        rsi=rsi,
        oversold=oversold_zone,
        overbought=overbought_zone,
        bullish_zone=bullish_zone,
        bearish_zone=bearish_zone,
        divergence=divergence,
    )


@dataclass
class MACDResult:
    """MACD calculation result."""
    macd_line: pd.Series
    signal_line: pd.Series
    histogram: pd.Series
    crossover: pd.Series     # 1 = bullish, -1 = bearish
    zero_cross: pd.Series    # 1 = crossed above, -1 = crossed below
    histogram_trend: pd.Series  # growing/shrinking


def calculate_macd(
    df: pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> MACDResult:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    MACD shows momentum and trend:
    - MACD above signal = Bullish (CE)
    - MACD below signal = Bearish (PE)
    - Zero line cross = Strong trend signal

    Args:
        df: DataFrame with Close column
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line period

    Returns:
        MACDResult with MACD values and signals
    """
    close = df["Close"]

    # Calculate EMAs
    ema_fast = close.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=slow_period, adjust=False).mean()

    # MACD line and signal
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    # Crossover signals
    prev_diff = macd_line.shift(1) - signal_line.shift(1)
    curr_diff = macd_line - signal_line

    crossover = pd.Series(0, index=df.index)
    crossover[(prev_diff <= 0) & (curr_diff > 0)] = 1   # Bullish
    crossover[(prev_diff >= 0) & (curr_diff < 0)] = -1  # Bearish

    # Zero line cross
    zero_cross = pd.Series(0, index=df.index)
    zero_cross[(macd_line.shift(1) <= 0) & (macd_line > 0)] = 1
    zero_cross[(macd_line.shift(1) >= 0) & (macd_line < 0)] = -1

    # Histogram trend
    histogram_trend = pd.Series("neutral", index=df.index)
    histogram_trend[(histogram > histogram.shift(1)) & (histogram > 0)] = "growing_bullish"
    histogram_trend[(histogram < histogram.shift(1)) & (histogram > 0)] = "shrinking_bullish"
    histogram_trend[(histogram < histogram.shift(1)) & (histogram < 0)] = "growing_bearish"
    histogram_trend[(histogram > histogram.shift(1)) & (histogram < 0)] = "shrinking_bearish"

    return MACDResult(
        macd_line=macd_line,
        signal_line=signal_line,
        histogram=histogram,
        crossover=crossover,
        zero_cross=zero_cross,
        histogram_trend=histogram_trend,
    )


@dataclass
class StochasticResult:
    """Stochastic calculation result."""
    k: pd.Series
    d: pd.Series
    oversold: pd.Series
    overbought: pd.Series
    crossover: pd.Series     # 1 = bullish in OS, -1 = bearish in OB
    divergence: pd.Series


def calculate_stochastic(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
    smooth: int = 3,
    oversold: float = 20,
    overbought: float = 80,
) -> StochasticResult:
    """
    Calculate Stochastic Oscillator.

    Stochastic shows price position within range:
    - %K crosses above %D in oversold = Strong CE signal
    - %K crosses below %D in overbought = Strong PE signal

    Args:
        df: DataFrame with OHLC data
        k_period: %K period
        d_period: %D period
        smooth: Smoothing factor
        oversold: Oversold threshold
        overbought: Overbought threshold

    Returns:
        StochasticResult with K, D values and signals
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    # Lowest low and highest high
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()

    # %K
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)

    # Smooth %K if needed
    if smooth > 1:
        k = k.rolling(window=smooth).mean()

    # %D is SMA of %K
    d = k.rolling(window=d_period).mean()

    # Zones
    oversold_zone = k < oversold
    overbought_zone = k > overbought

    # Crossover in zones (high probability signals)
    prev_diff = k.shift(1) - d.shift(1)
    curr_diff = k - d

    crossover = pd.Series(0, index=df.index)
    # Bullish: K crosses above D while oversold
    crossover[(prev_diff <= 0) & (curr_diff > 0) & oversold_zone] = 1
    # Bearish: K crosses below D while overbought
    crossover[(prev_diff >= 0) & (curr_diff < 0) & overbought_zone] = -1

    # Divergence detection
    divergence = pd.Series(0, index=df.index)
    lookback = 10

    for i in range(lookback, len(df)):
        price_window = close.iloc[i - lookback:i + 1]
        k_window = k.iloc[i - lookback:i + 1]

        # Bullish divergence in oversold
        if (price_window.iloc[-1] < price_window.min() * 1.001 and
            k_window.iloc[-1] > k_window.min() * 1.05 and
            k.iloc[i] < 30):
            divergence.iloc[i] = 1

        # Bearish divergence in overbought
        elif (price_window.iloc[-1] > price_window.max() * 0.999 and
              k_window.iloc[-1] < k_window.max() * 0.95 and
              k.iloc[i] > 70):
            divergence.iloc[i] = -1

    return StochasticResult(
        k=k,
        d=d,
        oversold=oversold_zone,
        overbought=overbought_zone,
        crossover=crossover,
        divergence=divergence,
    )


@dataclass
class CCIResult:
    """CCI calculation result."""
    cci: pd.Series
    breakout_up: pd.Series    # CCI crosses above +100
    breakout_down: pd.Series  # CCI crosses below -100
    extreme_up: pd.Series     # CCI > +200
    extreme_down: pd.Series   # CCI < -200
    zero_cross: pd.Series


def calculate_cci(
    df: pd.DataFrame,
    period: int = 20,
    constant: float = 0.015,
) -> CCIResult:
    """
    Calculate CCI (Commodity Channel Index).

    CCI measures price deviation from mean:
    - CCI crosses above +100 = Breakout, BUY CE
    - CCI crosses below -100 = Breakdown, BUY PE
    - Reversal: CCI returns from extreme = Fade the move

    Args:
        df: DataFrame with OHLC data
        period: CCI period
        constant: Constant multiplier

    Returns:
        CCIResult with CCI values and signals
    """
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    sma = typical_price.rolling(window=period).mean()
    mean_deviation = typical_price.rolling(window=period).apply(
        lambda x: np.abs(x - x.mean()).mean()
    )

    cci = (typical_price - sma) / (constant * mean_deviation)

    # Breakout signals
    breakout_up = (cci.shift(1) <= 100) & (cci > 100)
    breakout_down = (cci.shift(1) >= -100) & (cci < -100)

    # Extreme zones
    extreme_up = cci > 200
    extreme_down = cci < -200

    # Zero line cross
    zero_cross = pd.Series(0, index=df.index)
    zero_cross[(cci.shift(1) <= 0) & (cci > 0)] = 1
    zero_cross[(cci.shift(1) >= 0) & (cci < 0)] = -1

    return CCIResult(
        cci=cci,
        breakout_up=breakout_up,
        breakout_down=breakout_down,
        extreme_up=extreme_up,
        extreme_down=extreme_down,
        zero_cross=zero_cross,
    )
