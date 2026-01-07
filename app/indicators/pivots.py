"""
Pivot Point Indicators
Standard Pivots, CPR, Camarilla calculations
"""

import pandas as pd
from dataclasses import dataclass
from typing import Literal


@dataclass
class PivotResult:
    """Pivot Points calculation result."""
    pivot: float
    r1: float
    r2: float
    r3: float
    s1: float
    s2: float
    s3: float
    bias: str  # bullish, bearish, neutral


def calculate_pivot_points(
    high: float,
    low: float,
    close: float,
    pivot_type: Literal["standard", "fibonacci", "woodie"] = "standard",
) -> PivotResult:
    """
    Calculate Pivot Points.

    Pivot Points are key S/R levels:
    - Price above Pivot = Bullish bias (CE)
    - Price below Pivot = Bearish bias (PE)
    - R1/R2/R3 = Resistance levels
    - S1/S2/S3 = Support levels

    Args:
        high: Previous day high
        low: Previous day low
        close: Previous day close
        pivot_type: Type of pivot calculation

    Returns:
        PivotResult with pivot levels
    """
    if pivot_type == "standard":
        pivot = (high + low + close) / 3
        r1 = (2 * pivot) - low
        s1 = (2 * pivot) - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        r3 = high + 2 * (pivot - low)
        s3 = low - 2 * (high - pivot)

    elif pivot_type == "fibonacci":
        pivot = (high + low + close) / 3
        range_hl = high - low
        r1 = pivot + (0.382 * range_hl)
        r2 = pivot + (0.618 * range_hl)
        r3 = pivot + (1.000 * range_hl)
        s1 = pivot - (0.382 * range_hl)
        s2 = pivot - (0.618 * range_hl)
        s3 = pivot - (1.000 * range_hl)

    elif pivot_type == "woodie":
        pivot = (high + low + 2 * close) / 4
        r1 = (2 * pivot) - low
        r2 = pivot + (high - low)
        r3 = r1 + (high - low)
        s1 = (2 * pivot) - high
        s2 = pivot - (high - low)
        s3 = s1 - (high - low)

    else:
        raise ValueError(f"Unknown pivot type: {pivot_type}")

    # Determine bias based on close position
    if close > pivot:
        bias = "bullish"
    elif close < pivot:
        bias = "bearish"
    else:
        bias = "neutral"

    return PivotResult(
        pivot=round(pivot, 2),
        r1=round(r1, 2),
        r2=round(r2, 2),
        r3=round(r3, 2),
        s1=round(s1, 2),
        s2=round(s2, 2),
        s3=round(s3, 2),
        bias=bias,
    )


@dataclass
class CPRResult:
    """Central Pivot Range result."""
    pivot: float
    tc: float           # Top CPR
    bc: float           # Bottom CPR
    width: float        # CPR width in points
    width_pct: float    # CPR width as percentage
    expected_day: str   # trending, ranging, big_move
    virgin_cpr: bool    # Was previous day's CPR untouched


def calculate_cpr(
    high: float,
    low: float,
    close: float,
    prev_high: float | None = None,
    prev_low: float | None = None,
) -> CPRResult:
    """
    Calculate Central Pivot Range.

    CPR shows intraday bias:
    - Narrow CPR (<20 pts) = Trending day expected
    - Wide CPR (>50 pts) = Ranging day expected
    - Price above CPR = Bullish (CE on dips)
    - Price below CPR = Bearish (PE on rallies)

    Args:
        high: Previous day high
        low: Previous day low
        close: Previous day close
        prev_high: Day before previous high (for virgin CPR)
        prev_low: Day before previous low (for virgin CPR)

    Returns:
        CPRResult with CPR levels and analysis
    """
    pivot = (high + low + close) / 3
    bc = (high + low) / 2
    tc = (pivot - bc) + pivot

    # Ensure TC > BC
    if tc < bc:
        tc, bc = bc, tc

    width = tc - bc
    width_pct = (width / pivot) * 100

    # Expected day type
    if width < 20:
        expected_day = "trending"
    elif width > 50:
        expected_day = "ranging"
    elif width < 10:
        expected_day = "big_move"  # Very narrow = explosive move
    else:
        expected_day = "normal"

    # Virgin CPR check (if previous day's price range doesn't touch today's CPR)
    virgin_cpr = False
    if prev_high and prev_low:
        if prev_low > tc or prev_high < bc:
            virgin_cpr = True

    return CPRResult(
        pivot=round(pivot, 2),
        tc=round(tc, 2),
        bc=round(bc, 2),
        width=round(width, 2),
        width_pct=round(width_pct, 3),
        expected_day=expected_day,
        virgin_cpr=virgin_cpr,
    )


@dataclass
class CamarillaResult:
    """Camarilla Pivot result."""
    h4: float  # Breakout level up
    h3: float  # Reversal level (resistance)
    h2: float
    h1: float
    l1: float
    l2: float
    l3: float  # Reversal level (support)
    l4: float  # Breakout level down
    range_type: str  # inside, outside, breakout_up, breakout_down


def calculate_camarilla(
    high: float,
    low: float,
    close: float,
    current_price: float | None = None,
) -> CamarillaResult:
    """
    Calculate Camarilla Pivot Points.

    Camarilla levels for intraday trading:
    - H3/L3 = Reversal levels (fade moves)
    - H4/L4 = Breakout levels (trade breakouts)

    Strategy:
    - Open between H3-L3: Range day, fade at H3/L3
    - Break H4: Strong bullish, buy CE
    - Break L4: Strong bearish, buy PE

    Args:
        high: Previous day high
        low: Previous day low
        close: Previous day close
        current_price: Current price for range determination

    Returns:
        CamarillaResult with levels and range type
    """
    range_hl = high - low

    h4 = close + (range_hl * 1.1 / 2)
    h3 = close + (range_hl * 1.1 / 4)
    h2 = close + (range_hl * 1.1 / 6)
    h1 = close + (range_hl * 1.1 / 12)

    l1 = close - (range_hl * 1.1 / 12)
    l2 = close - (range_hl * 1.1 / 6)
    l3 = close - (range_hl * 1.1 / 4)
    l4 = close - (range_hl * 1.1 / 2)

    # Determine range type based on current price
    range_type = "inside"
    if current_price:
        if current_price > h4:
            range_type = "breakout_up"
        elif current_price < l4:
            range_type = "breakout_down"
        elif current_price > h3 or current_price < l3:
            range_type = "outside"

    return CamarillaResult(
        h4=round(h4, 2),
        h3=round(h3, 2),
        h2=round(h2, 2),
        h1=round(h1, 2),
        l1=round(l1, 2),
        l2=round(l2, 2),
        l3=round(l3, 2),
        l4=round(l4, 2),
        range_type=range_type,
    )


@dataclass
class StandardPivotResult:
    """Standard Pivot Points result for template."""
    pivot: float
    r1: float
    r2: float
    r3: float
    s1: float
    s2: float
    s3: float


@dataclass
class CamarillaPivotResult:
    """Camarilla Pivot Points result for template."""
    r1: float
    r2: float
    r3: float
    r4: float
    s1: float
    s2: float
    s3: float
    s4: float


def calculate_standard_pivots(high: float, low: float, close: float) -> StandardPivotResult:
    """Calculate standard pivot points for template display."""
    pivot = (high + low + close) / 3
    r1 = (2 * pivot) - low
    s1 = (2 * pivot) - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)

    return StandardPivotResult(
        pivot=round(pivot, 2),
        r1=round(r1, 2),
        r2=round(r2, 2),
        r3=round(r3, 2),
        s1=round(s1, 2),
        s2=round(s2, 2),
        s3=round(s3, 2),
    )


def calculate_camarilla_pivots(high: float, low: float, close: float) -> CamarillaPivotResult:
    """Calculate camarilla pivot points for template display."""
    range_hl = high - low

    r4 = close + (range_hl * 1.1 / 2)
    r3 = close + (range_hl * 1.1 / 4)
    r2 = close + (range_hl * 1.1 / 6)
    r1 = close + (range_hl * 1.1 / 12)

    s1 = close - (range_hl * 1.1 / 12)
    s2 = close - (range_hl * 1.1 / 6)
    s3 = close - (range_hl * 1.1 / 4)
    s4 = close - (range_hl * 1.1 / 2)

    return CamarillaPivotResult(
        r1=round(r1, 2),
        r2=round(r2, 2),
        r3=round(r3, 2),
        r4=round(r4, 2),
        s1=round(s1, 2),
        s2=round(s2, 2),
        s3=round(s3, 2),
        s4=round(s4, 2),
    )


def get_nearest_level(
    price: float,
    pivot_result: PivotResult | None = None,
    cpr_result: CPRResult | None = None,
    camarilla_result: CamarillaResult | None = None,
) -> dict:
    """
    Find nearest support and resistance levels.

    Args:
        price: Current price
        pivot_result: Standard pivot result
        cpr_result: CPR result
        camarilla_result: Camarilla result

    Returns:
        Dictionary with nearest support and resistance
    """
    supports = []
    resistances = []

    if pivot_result:
        supports.extend([pivot_result.s1, pivot_result.s2, pivot_result.s3])
        resistances.extend([pivot_result.r1, pivot_result.r2, pivot_result.r3])
        supports.append(pivot_result.pivot)
        resistances.append(pivot_result.pivot)

    if cpr_result:
        supports.append(cpr_result.bc)
        resistances.append(cpr_result.tc)

    if camarilla_result:
        supports.extend([camarilla_result.l1, camarilla_result.l2,
                        camarilla_result.l3, camarilla_result.l4])
        resistances.extend([camarilla_result.h1, camarilla_result.h2,
                          camarilla_result.h3, camarilla_result.h4])

    # Filter and find nearest
    supports_below = [s for s in supports if s < price]
    resistances_above = [r for r in resistances if r > price]

    nearest_support = max(supports_below) if supports_below else None
    nearest_resistance = min(resistances_above) if resistances_above else None

    return {
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
        "support_distance": round(price - nearest_support, 2) if nearest_support else None,
        "resistance_distance": round(nearest_resistance - price, 2) if nearest_resistance else None,
    }
