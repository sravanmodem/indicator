"""
Market Data API Routes
Handles quotes, historical data, option chain
"""

from datetime import datetime, timedelta
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from loguru import logger

from app.services.zerodha_auth import get_auth_service
from app.services.data_fetcher import get_data_fetcher
from app.services.websocket_manager import get_ws_manager, TickMode
from app.core.config import NIFTY_INDEX_TOKEN, BANKNIFTY_INDEX_TOKEN

router = APIRouter(prefix="/market", tags=["Market Data"])


def require_auth():
    """Check if user is authenticated."""
    auth = get_auth_service()
    if not auth.is_authenticated:
        raise HTTPException(status_code=401, detail="Not authenticated")


@router.get("/quote/{symbol}")
async def get_quote(symbol: str):
    """
    Get current quote for a symbol.

    Example: /market/quote/NSE:NIFTY%2050
    """
    require_auth()
    fetcher = get_data_fetcher()
    quote = await fetcher.fetch_quote([symbol])

    if not quote:
        raise HTTPException(status_code=404, detail="Quote not found")

    return quote


@router.get("/ohlc/{symbol}")
async def get_ohlc(symbol: str):
    """Get OHLC for a symbol."""
    require_auth()
    fetcher = get_data_fetcher()
    ohlc = await fetcher.fetch_ohlc([symbol])

    if not ohlc:
        raise HTTPException(status_code=404, detail="OHLC not found")

    return ohlc


@router.get("/historical/{instrument_token}")
async def get_historical(
    instrument_token: int,
    timeframe: str = Query(default="5minute", description="Candle interval"),
    days: int = Query(default=5, description="Number of days"),
):
    """
    Get historical OHLCV data.

    Args:
        instrument_token: Zerodha instrument token
        timeframe: minute, 3minute, 5minute, 15minute, 30minute, 60minute, day
        days: Number of days of data
    """
    require_auth()
    fetcher = get_data_fetcher()

    df = await fetcher.fetch_historical_data(
        instrument_token=instrument_token,
        timeframe=timeframe,
        days=days,
    )

    if df.empty:
        raise HTTPException(status_code=404, detail="Historical data not found")

    # Convert to JSON-serializable format
    df_reset = df.reset_index()
    df_reset["date"] = df_reset["date"].dt.strftime("%Y-%m-%d %H:%M:%S")

    return {
        "instrument_token": instrument_token,
        "timeframe": timeframe,
        "candles": df_reset.to_dict(orient="records"),
        "count": len(df),
    }


@router.get("/option-chain/{index}")
async def get_option_chain(
    index: Literal["NIFTY", "BANKNIFTY"] = "NIFTY",
    strikes: int = Query(default=10, description="Number of strikes around ATM"),
):
    """
    Get option chain with OI data.

    Returns complete chain with CE/PE data for analysis.
    """
    require_auth()
    fetcher = get_data_fetcher()

    chain = await fetcher.get_option_chain(
        index=index,
        strike_count=strikes,
    )

    if "error" in chain:
        raise HTTPException(status_code=500, detail=chain["error"])

    return chain


@router.get("/instruments/{exchange}")
async def get_instruments(
    exchange: Literal["NSE", "NFO", "BSE", "BFO"] = "NFO",
):
    """Get instruments list for an exchange."""
    require_auth()
    fetcher = get_data_fetcher()

    df = await fetcher.fetch_instruments(exchange)

    if df.empty:
        raise HTTPException(status_code=404, detail="Instruments not found")

    return {
        "exchange": exchange,
        "count": len(df),
        "instruments": df.head(100).to_dict(orient="records"),
    }


@router.get("/nifty-options")
async def get_nifty_options(
    min_strike: int = Query(default=None),
    max_strike: int = Query(default=None),
):
    """Get NIFTY options for current expiry."""
    require_auth()
    fetcher = get_data_fetcher()

    strike_range = (min_strike, max_strike) if min_strike and max_strike else None
    options = await fetcher.get_nifty_options(strike_range=strike_range)

    if options.empty:
        raise HTTPException(status_code=404, detail="Options not found")

    return {
        "count": len(options),
        "expiry": str(options.iloc[0]["expiry"]) if not options.empty else None,
        "options": options.to_dict(orient="records"),
    }


# WebSocket Management Endpoints

@router.post("/ws/connect")
async def connect_websocket():
    """Connect to Zerodha WebSocket."""
    require_auth()
    ws = get_ws_manager()

    success = await ws.connect()
    if not success:
        raise HTTPException(status_code=500, detail="WebSocket connection failed")

    return {"success": True, "message": "WebSocket connected"}


@router.post("/ws/disconnect")
async def disconnect_websocket():
    """Disconnect WebSocket."""
    ws = get_ws_manager()
    await ws.disconnect()
    return {"success": True, "message": "WebSocket disconnected"}


@router.post("/ws/subscribe")
async def subscribe_instruments(
    tokens: list[int],
    mode: Literal["ltp", "quote", "full"] = "full",
):
    """Subscribe to instrument tokens."""
    require_auth()
    ws = get_ws_manager()

    if not ws.is_connected:
        raise HTTPException(status_code=400, detail="WebSocket not connected")

    mode_map = {"ltp": TickMode.LTP, "quote": TickMode.QUOTE, "full": TickMode.FULL}
    success = await ws.subscribe(tokens, mode_map[mode])

    return {"success": success, "subscribed": tokens}


@router.get("/ws/status")
async def websocket_status():
    """Get WebSocket connection status."""
    ws = get_ws_manager()
    return ws.get_connection_status()


@router.get("/ws/ticks")
async def get_latest_ticks():
    """Get latest tick data for all subscribed instruments."""
    ws = get_ws_manager()
    ticks = ws.get_all_ticks()

    return {
        "count": len(ticks),
        "ticks": {
            token: {
                "last_price": tick.last_price,
                "volume": tick.volume,
                "oi": tick.oi,
                "bid": tick.bid_price,
                "ask": tick.ask_price,
                "timestamp": tick.timestamp.isoformat() if tick.timestamp else None,
            }
            for token, tick in ticks.items()
        },
    }
