"""
Option Greeks Calculations
Delta, Gamma, Theta, Vega, Rho using Black-Scholes-Merton model
"""

import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy.stats import norm
from typing import Literal


@dataclass
class GreeksResult:
    """Option Greeks calculation result."""
    delta: float          # Price sensitivity (0-1 for calls, 0--1 for puts)
    gamma: float          # Delta sensitivity
    theta: float          # Time decay (per day)
    vega: float           # Volatility sensitivity (per 1% IV change)
    rho: float            # Interest rate sensitivity

    # Additional metrics
    delta_percent: float  # Delta as percentage
    theta_per_day: float  # Theta rupees per day
    vega_per_pct: float   # Vega rupees per 1% IV change

    # Interpretation
    delta_interpretation: str
    gamma_interpretation: str
    theta_interpretation: str
    vega_interpretation: str


@dataclass
class ExpectedPriceResult:
    """Expected option price calculations."""
    current_price: float
    expected_1pct_move: float  # If spot moves 1%
    expected_2pct_move: float  # If spot moves 2%
    expected_target: float     # At signal target
    expected_stop: float       # At signal stop loss

    # Time decay impact
    price_tomorrow: float      # After 1 day (theta decay)
    price_3days: float         # After 3 days

    # IV scenarios
    price_if_iv_up_5pct: float   # If IV increases 5%
    price_if_iv_down_5pct: float # If IV decreases 5%

    # Profit/Loss scenarios
    profit_at_target: float
    loss_at_stop: float
    breakeven_spot: float


def calculate_greeks(
    spot_price: float,
    strike_price: float,
    time_to_expiry: float,  # In years
    risk_free_rate: float,  # Annual rate (e.g., 0.065 for 6.5%)
    volatility: float,      # Implied volatility (e.g., 0.15 for 15%)
    option_type: Literal["CE", "PE"],
    current_premium: float | None = None,
) -> GreeksResult:
    """
    Calculate Option Greeks using Black-Scholes-Merton model.

    Args:
        spot_price: Current underlying price
        strike_price: Strike price of option
        time_to_expiry: Time to expiry in years (days/365)
        risk_free_rate: Risk-free interest rate (annual)
        volatility: Implied volatility (annual)
        option_type: "CE" for call, "PE" for put
        current_premium: Current option premium (for validation)

    Returns:
        GreeksResult with all Greeks and interpretations
    """
    # Avoid division by zero
    if time_to_expiry <= 0:
        time_to_expiry = 1/365  # 1 day minimum

    # Calculate d1 and d2 for Black-Scholes
    d1 = (np.log(spot_price / strike_price) +
          (risk_free_rate + 0.5 * volatility ** 2) * time_to_expiry) / \
         (volatility * np.sqrt(time_to_expiry))

    d2 = d1 - volatility * np.sqrt(time_to_expiry)

    # Standard normal CDF and PDF
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    n_d1 = norm.pdf(d1)  # PDF for gamma and vega

    # DELTA
    if option_type == "CE":
        delta = N_d1
    else:  # PE
        delta = N_d1 - 1

    # GAMMA (same for calls and puts)
    gamma = n_d1 / (spot_price * volatility * np.sqrt(time_to_expiry))

    # THETA (per year, convert to per day)
    if option_type == "CE":
        theta_year = -(spot_price * n_d1 * volatility) / (2 * np.sqrt(time_to_expiry)) - \
                     risk_free_rate * strike_price * np.exp(-risk_free_rate * time_to_expiry) * N_d2
    else:  # PE
        theta_year = -(spot_price * n_d1 * volatility) / (2 * np.sqrt(time_to_expiry)) + \
                     risk_free_rate * strike_price * np.exp(-risk_free_rate * time_to_expiry) * (1 - N_d2)

    theta = theta_year / 365  # Per day

    # VEGA (per 1% change in volatility)
    vega = spot_price * n_d1 * np.sqrt(time_to_expiry) / 100

    # RHO (per 1% change in interest rate)
    if option_type == "CE":
        rho = strike_price * time_to_expiry * np.exp(-risk_free_rate * time_to_expiry) * N_d2 / 100
    else:  # PE
        rho = -strike_price * time_to_expiry * np.exp(-risk_free_rate * time_to_expiry) * (1 - N_d2) / 100

    # Additional metrics
    delta_percent = delta * 100
    theta_per_day = theta
    vega_per_pct = vega

    # Interpretations
    if option_type == "CE":
        if abs(delta) > 0.7:
            delta_interp = "Deep ITM - High directional exposure"
        elif abs(delta) > 0.3:
            delta_interp = "ATM - Balanced directional exposure"
        else:
            delta_interp = "OTM - Low directional exposure"
    else:
        if abs(delta) > 0.7:
            delta_interp = "Deep ITM - High directional exposure"
        elif abs(delta) > 0.3:
            delta_interp = "ATM - Balanced directional exposure"
        else:
            delta_interp = "OTM - Low directional exposure"

    if gamma > 0.01:
        gamma_interp = "High gamma - Delta changes rapidly"
    elif gamma > 0.005:
        gamma_interp = "Moderate gamma - Standard delta movement"
    else:
        gamma_interp = "Low gamma - Stable delta"

    if abs(theta) > 10:
        theta_interp = f"High decay - Losing ₹{abs(theta):.2f}/day"
    elif abs(theta) > 5:
        theta_interp = f"Moderate decay - Losing ₹{abs(theta):.2f}/day"
    else:
        theta_interp = f"Low decay - Losing ₹{abs(theta):.2f}/day"

    if vega > 5:
        vega_interp = f"High IV sensitivity - ₹{vega:.2f} per 1% IV"
    elif vega > 2:
        vega_interp = f"Moderate IV sensitivity - ₹{vega:.2f} per 1% IV"
    else:
        vega_interp = f"Low IV sensitivity - ₹{vega:.2f} per 1% IV"

    return GreeksResult(
        delta=round(delta, 4),
        gamma=round(gamma, 6),
        theta=round(theta, 4),
        vega=round(vega, 4),
        rho=round(rho, 4),
        delta_percent=round(delta_percent, 2),
        theta_per_day=round(theta_per_day, 2),
        vega_per_pct=round(vega_per_pct, 2),
        delta_interpretation=delta_interp,
        gamma_interpretation=gamma_interp,
        theta_interpretation=theta_interp,
        vega_interpretation=vega_interp,
    )


def calculate_expected_prices(
    spot_price: float,
    strike_price: float,
    current_premium: float,
    greeks: GreeksResult,
    target_price: float,
    stop_loss: float,
    days_to_expiry: int,
    current_iv: float = 0.15,
) -> ExpectedPriceResult:
    """
    Calculate expected option prices under various scenarios.

    Args:
        spot_price: Current spot price
        strike_price: Strike price
        current_premium: Current option premium
        greeks: Calculated Greeks
        target_price: Signal target price
        stop_loss: Signal stop loss price
        days_to_expiry: Days until expiry
        current_iv: Current implied volatility

    Returns:
        ExpectedPriceResult with price projections
    """
    # 1% and 2% spot moves
    spot_1pct = spot_price * 0.01
    spot_2pct = spot_price * 0.02

    expected_1pct = current_premium + (greeks.delta * spot_1pct)
    expected_2pct = current_premium + (greeks.delta * spot_2pct)

    # At target and stop
    target_move = target_price - spot_price
    stop_move = stop_loss - spot_price

    expected_target = current_premium + (greeks.delta * target_move)
    expected_stop = current_premium + (greeks.delta * stop_move)

    # Time decay scenarios
    price_tomorrow = current_premium + greeks.theta_per_day
    price_3days = current_premium + (greeks.theta_per_day * 3)

    # Make sure prices don't go negative
    price_tomorrow = max(0.05, price_tomorrow)
    price_3days = max(0.05, price_3days)
    expected_target = max(0.05, expected_target)
    expected_stop = max(0.05, expected_stop)

    # IV scenarios (5% change in IV)
    iv_change_5pct = current_iv * 0.05
    price_if_iv_up = current_premium + (greeks.vega_per_pct * 5)
    price_if_iv_down = max(0.05, current_premium - (greeks.vega_per_pct * 5))

    # P&L calculations
    profit_at_target = expected_target - current_premium
    loss_at_stop = expected_stop - current_premium

    # Breakeven spot (where premium covers theta decay)
    # Simplified: spot move needed to offset 1 day theta
    if greeks.delta != 0:
        theta_offset_move = -greeks.theta_per_day / greeks.delta
        breakeven_spot = spot_price + theta_offset_move
    else:
        breakeven_spot = spot_price

    return ExpectedPriceResult(
        current_price=round(current_premium, 2),
        expected_1pct_move=round(expected_1pct, 2),
        expected_2pct_move=round(expected_2pct, 2),
        expected_target=round(expected_target, 2),
        expected_stop=round(expected_stop, 2),
        price_tomorrow=round(price_tomorrow, 2),
        price_3days=round(price_3days, 2),
        price_if_iv_up_5pct=round(price_if_iv_up, 2),
        price_if_iv_down_5pct=round(price_if_iv_down, 2),
        profit_at_target=round(profit_at_target, 2),
        loss_at_stop=round(loss_at_stop, 2),
        breakeven_spot=round(breakeven_spot, 2),
    )


def estimate_implied_volatility(
    option_chain: list[dict],
    spot_price: float,
) -> float:
    """
    Estimate implied volatility from ATM options.

    Args:
        option_chain: Option chain data
        spot_price: Current spot price

    Returns:
        Estimated IV (e.g., 0.15 for 15%)
    """
    # Find ATM strike
    atm_strike = min(
        [opt["strike"] for opt in option_chain],
        key=lambda x: abs(x - spot_price)
    )

    # Get ATM option premiums
    atm_opt = next((opt for opt in option_chain if opt["strike"] == atm_strike), None)

    if not atm_opt:
        return 0.15  # Default 15%

    ce_data = atm_opt.get("ce", {})
    pe_data = atm_opt.get("pe", {})

    ce_ltp = ce_data.get("ltp", 0)
    pe_ltp = pe_data.get("ltp", 0)

    # Use simple approximation: IV ≈ (ATM Premium / Spot) * sqrt(365/DTE)
    # Simplified - in production, use Newton-Raphson iteration
    if ce_ltp > 0:
        avg_premium = (ce_ltp + pe_ltp) / 2 if pe_ltp > 0 else ce_ltp
        estimated_iv = (avg_premium / spot_price) * 2  # Rough approximation
        return max(0.08, min(estimated_iv, 0.50))  # Clamp between 8% and 50%

    return 0.15  # Default


def get_days_to_expiry(expiry_date: datetime | None = None) -> int:
    """
    Calculate days to expiry.

    Args:
        expiry_date: Expiry datetime. If None, assumes weekly expiry (Thursday)

    Returns:
        Days to expiry
    """
    if expiry_date is None:
        # Find next Thursday (weekly NIFTY expiry)
        today = datetime.now()
        days_ahead = 3 - today.weekday()  # Thursday is 3
        if days_ahead <= 0:
            days_ahead += 7
        expiry_date = today + timedelta(days=days_ahead)

    days = (expiry_date - datetime.now()).days
    return max(1, days)  # At least 1 day
