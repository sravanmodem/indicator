"""
Volatility Indicators
Bollinger Bands, ATR calculations
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class BollingerResult:
    """Bollinger Bands calculation result."""
    middle: pd.Series
    upper: pd.Series
    lower: pd.Series
    bandwidth: pd.Series
    percent_b: pd.Series
    squeeze: pd.Series      # True when bandwidth is low
    upper_touch: pd.Series  # Price touching upper band
    lower_touch: pd.Series  # Price touching lower band


def calculate_bollinger_bands(
    df: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0,
    squeeze_threshold: float = 0.05,
) -> BollingerResult:
    """
    Calculate Bollinger Bands.

    Bollinger Bands show volatility and potential reversal zones:
    - Price at upper band = Overbought (PE consideration)
    - Price at lower band = Oversold (CE consideration)
    - Squeeze (narrow bands) = Big move coming

    Args:
        df: DataFrame with Close column
        period: SMA period
        std_dev: Standard deviation multiplier
        squeeze_threshold: Bandwidth threshold for squeeze

    Returns:
        BollingerResult with bands and signals
    """
    close = df["Close"]

    # Middle band (SMA)
    middle = close.rolling(window=period).mean()

    # Standard deviation
    std = close.rolling(window=period).std()

    # Upper and lower bands
    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)

    # Bandwidth: (Upper - Lower) / Middle
    bandwidth = (upper - lower) / middle

    # %B: (Price - Lower) / (Upper - Lower)
    percent_b = (close - lower) / (upper - lower)

    # Squeeze detection
    bandwidth_percentile = bandwidth.rolling(window=period * 6).rank(pct=True)
    squeeze = bandwidth_percentile < squeeze_threshold

    # Band touches
    upper_touch = close >= upper
    lower_touch = close <= lower

    return BollingerResult(
        middle=middle,
        upper=upper,
        lower=lower,
        bandwidth=bandwidth,
        percent_b=percent_b,
        squeeze=squeeze,
        upper_touch=upper_touch,
        lower_touch=lower_touch,
    )


@dataclass
class ATRResult:
    """ATR calculation result."""
    atr: pd.Series
    atr_percent: pd.Series     # ATR as percentage of price
    expanding: pd.Series       # ATR increasing
    contracting: pd.Series     # ATR decreasing
    volatility_state: pd.Series  # low, normal, high, extreme


def calculate_atr(
    df: pd.DataFrame,
    period: int = 14,
) -> ATRResult:
    """
    Calculate ATR (Average True Range).

    ATR measures volatility:
    - High ATR = Large price swings, wider stops needed
    - Low ATR = Small moves, tighter stops possible
    - Rising ATR = Trend accelerating
    - Falling ATR = Trend weakening

    Args:
        df: DataFrame with OHLC data
        period: ATR period

    Returns:
        ATRResult with ATR values and volatility state
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    # True Range components
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    # True Range is max of the three
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR is smoothed TR
    atr = tr.rolling(window=period).mean()

    # ATR as percentage of close
    atr_percent = (atr / close) * 100

    # Trend in ATR
    atr_sma = atr.rolling(window=5).mean()
    expanding = atr > atr_sma
    contracting = atr < atr_sma

    # Volatility state based on percentile
    atr_percentile = atr.rolling(window=period * 10).rank(pct=True)

    volatility_state = pd.Series("normal", index=df.index)
    volatility_state[atr_percentile < 0.2] = "low"
    volatility_state[atr_percentile > 0.8] = "high"
    volatility_state[atr_percentile > 0.95] = "extreme"

    return ATRResult(
        atr=atr,
        atr_percent=atr_percent,
        expanding=expanding,
        contracting=contracting,
        volatility_state=volatility_state,
    )


def calculate_atr_stop(
    entry_price: float,
    atr_value: float,
    direction: str,
    multiplier: float = 2.0,
) -> float:
    """
    Calculate ATR-based stop loss.

    Args:
        entry_price: Entry price
        atr_value: Current ATR value
        direction: 'long' or 'short'
        multiplier: ATR multiplier for stop distance

    Returns:
        Stop loss price
    """
    stop_distance = atr_value * multiplier

    if direction.lower() == "long":
        return entry_price - stop_distance
    else:
        return entry_price + stop_distance


def calculate_atr_target(
    entry_price: float,
    atr_value: float,
    direction: str,
    risk_reward: float = 2.0,
    multiplier: float = 2.0,
) -> float:
    """
    Calculate ATR-based target.

    Args:
        entry_price: Entry price
        atr_value: Current ATR value
        direction: 'long' or 'short'
        risk_reward: Risk-reward ratio
        multiplier: ATR multiplier for stop distance

    Returns:
        Target price
    """
    stop_distance = atr_value * multiplier
    target_distance = stop_distance * risk_reward

    if direction.lower() == "long":
        return entry_price + target_distance
    else:
        return entry_price - target_distance
