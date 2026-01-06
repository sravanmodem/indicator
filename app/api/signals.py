"""
Trading Signals API Routes
Handles signal generation and indicator calculations
"""

from datetime import datetime
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from loguru import logger

from app.services.zerodha_auth import get_auth_service
from app.services.data_fetcher import get_data_fetcher
from app.services.signal_engine import get_signal_engine, TradingStyle, SignalType
from app.core.config import NIFTY_INDEX_TOKEN, BANKNIFTY_INDEX_TOKEN

router = APIRouter(prefix="/signals", tags=["Trading Signals"])


def require_auth():
    """Check if user is authenticated."""
    auth = get_auth_service()
    if not auth.is_authenticated:
        raise HTTPException(status_code=401, detail="Not authenticated")


@router.get("/analyze/{index}")
async def analyze_market(
    index: Literal["NIFTY", "BANKNIFTY"] = "NIFTY",
    timeframe: str = Query(default="5minute"),
    style: Literal["scalping", "intraday", "swing"] = Query(default="intraday"),
    days: int = Query(default=5),
):
    """
    Analyze market and generate CE/PE signal.

    Returns comprehensive analysis with:
    - Signal direction (CE/PE)
    - Confidence level
    - Entry, stop loss, targets
    - All indicator values
    - Supporting and warning factors
    """
    require_auth()
    fetcher = get_data_fetcher()

    # Get instrument token
    token = NIFTY_INDEX_TOKEN if index == "NIFTY" else BANKNIFTY_INDEX_TOKEN

    # Fetch historical data
    df = await fetcher.fetch_historical_data(
        instrument_token=token,
        timeframe=timeframe,
        days=days,
    )

    if df.empty:
        raise HTTPException(status_code=404, detail="Market data not found")

    # Get option chain
    chain_data = await fetcher.get_option_chain(index=index)
    option_chain = chain_data.get("chain", []) if "error" not in chain_data else None
    spot_price = chain_data.get("spot_price")

    # Get previous day OHLC for pivots
    prev_day_ohlc = None
    if len(df) >= 2:
        # Group by date and get previous day's data
        df_daily = df.resample("D").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
        }).dropna()

        if len(df_daily) >= 2:
            prev_day = df_daily.iloc[-2]
            prev_day_ohlc = {
                "high": prev_day["High"],
                "low": prev_day["Low"],
                "close": prev_day["Close"],
            }

    # Get signal engine
    style_map = {
        "scalping": TradingStyle.SCALPING,
        "intraday": TradingStyle.INTRADAY,
        "swing": TradingStyle.SWING,
    }
    engine = get_signal_engine(style_map[style])

    # Analyze
    signal = engine.analyze(
        df=df,
        option_chain=option_chain,
        spot_price=spot_price,
        prev_day_ohlc=prev_day_ohlc,
    )

    if not signal:
        raise HTTPException(status_code=500, detail="Analysis failed")

    # Format response
    return {
        "timestamp": signal.timestamp.isoformat(),
        "index": index,
        "timeframe": timeframe,
        "style": style,
        "signal": {
            "type": signal.signal_type.value,
            "direction": signal.direction,
            "confidence": signal.confidence,
        },
        "levels": {
            "entry": signal.entry_price,
            "stop_loss": signal.stop_loss,
            "target_1": signal.target_1,
            "target_2": signal.target_2,
            "risk_reward": signal.risk_reward,
        },
        "indicators": [
            {
                "name": ind.name,
                "value": ind.value if not isinstance(ind.value, dict) else ind.value,
                "signal": ind.signal,
                "strength": ind.strength,
                "reason": ind.reason,
            }
            for ind in signal.indicators
        ],
        "factors": {
            "supporting": signal.supporting_factors,
            "warnings": signal.warning_factors,
        },
        "option_chain_summary": {
            "pcr": chain_data.get("pcr"),
            "total_ce_oi": chain_data.get("total_ce_oi"),
            "total_pe_oi": chain_data.get("total_pe_oi"),
            "atm_strike": chain_data.get("atm_strike"),
        } if chain_data and "error" not in chain_data else None,
    }


@router.get("/quick/{index}")
async def quick_signal(
    index: Literal["NIFTY", "BANKNIFTY"] = "NIFTY",
):
    """
    Get quick CE/PE signal without full analysis.

    Returns simplified signal for quick decision making.
    """
    require_auth()
    fetcher = get_data_fetcher()

    token = NIFTY_INDEX_TOKEN if index == "NIFTY" else BANKNIFTY_INDEX_TOKEN

    df = await fetcher.fetch_historical_data(
        instrument_token=token,
        timeframe="5minute",
        days=2,
    )

    if df.empty:
        raise HTTPException(status_code=404, detail="Market data not found")

    engine = get_signal_engine(TradingStyle.INTRADAY)
    signal = engine.analyze(df=df)

    if not signal:
        raise HTTPException(status_code=500, detail="Analysis failed")

    return {
        "index": index,
        "signal": signal.direction,
        "confidence": signal.confidence,
        "entry": signal.entry_price,
        "stop_loss": signal.stop_loss,
        "target": signal.target_1,
        "timestamp": signal.timestamp.isoformat(),
    }


@router.get("/indicators/{index}")
async def get_indicators(
    index: Literal["NIFTY", "BANKNIFTY"] = "NIFTY",
    timeframe: str = Query(default="5minute"),
):
    """
    Get raw indicator values without signal generation.

    Useful for custom analysis or charting.
    """
    require_auth()
    fetcher = get_data_fetcher()

    token = NIFTY_INDEX_TOKEN if index == "NIFTY" else BANKNIFTY_INDEX_TOKEN

    df = await fetcher.fetch_historical_data(
        instrument_token=token,
        timeframe=timeframe,
        days=5,
    )

    if df.empty:
        raise HTTPException(status_code=404, detail="Market data not found")

    from app.indicators.trend import calculate_supertrend, calculate_ema, calculate_vwap, calculate_adx
    from app.indicators.momentum import calculate_rsi, calculate_macd, calculate_stochastic
    from app.indicators.volatility import calculate_bollinger_bands, calculate_atr

    # Calculate all indicators
    st = calculate_supertrend(df)
    ema = calculate_ema(df)
    vwap = calculate_vwap(df)
    adx = calculate_adx(df)
    rsi = calculate_rsi(df)
    macd = calculate_macd(df)
    stoch = calculate_stochastic(df)
    bb = calculate_bollinger_bands(df)
    atr = calculate_atr(df)

    # Get latest values
    return {
        "index": index,
        "timeframe": timeframe,
        "timestamp": datetime.now().isoformat(),
        "price": {
            "open": df["Open"].iloc[-1],
            "high": df["High"].iloc[-1],
            "low": df["Low"].iloc[-1],
            "close": df["Close"].iloc[-1],
            "volume": df["Volume"].iloc[-1],
        },
        "indicators": {
            "supertrend": {
                "value": st.supertrend.iloc[-1],
                "direction": "bullish" if st.direction.iloc[-1] == 1 else "bearish",
            },
            "ema": {
                "fast": ema.ema_fast.iloc[-1],
                "slow": ema.ema_slow.iloc[-1],
                "trend": ema.ema_trend.iloc[-1] if ema.ema_trend is not None else None,
                "aligned": bool(ema.trend_aligned.iloc[-1]),
            },
            "vwap": {
                "value": vwap.vwap.iloc[-1],
                "upper_1": vwap.upper_band_1.iloc[-1],
                "lower_1": vwap.lower_band_1.iloc[-1],
                "deviation": vwap.deviation.iloc[-1],
            },
            "adx": {
                "value": adx.adx.iloc[-1],
                "plus_di": adx.plus_di.iloc[-1],
                "minus_di": adx.minus_di.iloc[-1],
                "strength": adx.trend_strength.iloc[-1],
            },
            "rsi": {
                "value": rsi.rsi.iloc[-1],
                "oversold": bool(rsi.oversold.iloc[-1]),
                "overbought": bool(rsi.overbought.iloc[-1]),
            },
            "macd": {
                "line": macd.macd_line.iloc[-1],
                "signal": macd.signal_line.iloc[-1],
                "histogram": macd.histogram.iloc[-1],
            },
            "stochastic": {
                "k": stoch.k.iloc[-1],
                "d": stoch.d.iloc[-1],
                "oversold": bool(stoch.oversold.iloc[-1]),
                "overbought": bool(stoch.overbought.iloc[-1]),
            },
            "bollinger": {
                "upper": bb.upper.iloc[-1],
                "middle": bb.middle.iloc[-1],
                "lower": bb.lower.iloc[-1],
                "bandwidth": bb.bandwidth.iloc[-1],
                "squeeze": bool(bb.squeeze.iloc[-1]),
            },
            "atr": {
                "value": atr.atr.iloc[-1],
                "percent": atr.atr_percent.iloc[-1],
                "state": atr.volatility_state.iloc[-1],
            },
        },
    }


@router.get("/stats")
async def get_signal_statistics():
    """
    Get signal statistics for the history page.

    Returns summary stats for total signals, win rate, etc.
    """
    from app.services.signal_history_service import get_history_service

    try:
        history_service = get_history_service()
        stats = history_service.get_statistics(days=30)

        return {
            "total": stats.get("total_signals", 0),
            "win_rate": stats.get("win_rate", 0),
            "avg_score": stats.get("avg_quality_score", 0),
            "target_hit": stats.get("target_hit", 0),
            "sl_hit": stats.get("stop_loss_hit", 0),
            "pending": stats.get("pending", 0),
            "total_pnl": stats.get("total_pnl", 0),
        }
    except Exception as e:
        logger.error(f"Error getting signal stats: {e}")
        return {
            "total": 0,
            "win_rate": 0,
            "avg_score": 0,
            "target_hit": 0,
            "sl_hit": 0,
            "pending": 0,
            "total_pnl": 0,
        }
