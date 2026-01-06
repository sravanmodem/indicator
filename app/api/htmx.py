"""
HTMX API Routes
Server-side rendered partials for real-time UI updates
"""

from datetime import datetime

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from loguru import logger

from app.services.zerodha_auth import get_auth_service
from app.services.data_fetcher import get_data_fetcher
from app.services.websocket_manager import get_ws_manager
from app.services.signal_engine import get_signal_engine, TradingStyle
from app.core.config import NIFTY_INDEX_TOKEN, BANKNIFTY_INDEX_TOKEN, SENSEX_INDEX_TOKEN

router = APIRouter(prefix="/htmx", tags=["HTMX Partials"])
templates = Jinja2Templates(directory="app/templates")


def require_auth():
    """Check if user is authenticated."""
    auth = get_auth_service()
    if not auth.is_authenticated:
        raise HTTPException(status_code=401, detail="Not authenticated")


@router.get("/auth-status", response_class=HTMLResponse)
async def auth_status_partial(request: Request):
    """Render authentication status partial."""
    auth = get_auth_service()
    status = auth.get_auth_status()

    return templates.TemplateResponse(
        "partials/auth_status.html",
        {"request": request, "status": status},
    )


@router.get("/signal-card/{index}", response_class=HTMLResponse)
async def signal_card_partial(
    request: Request,
    index: str = "NIFTY",
    style: str = "intraday",
):
    """Render signal card with CE/PE recommendation."""
    index = index.upper()
    style = style.lower()

    try:
        require_auth()
        fetcher = get_data_fetcher()

        if index == "NIFTY":
            token = NIFTY_INDEX_TOKEN
        elif index == "SENSEX":
            token = SENSEX_INDEX_TOKEN
        else:
            token = BANKNIFTY_INDEX_TOKEN

        df = await fetcher.fetch_historical_data(
            instrument_token=token,
            timeframe="5minute",
            days=3,
        )

        if df.empty:
            return templates.TemplateResponse(
                "partials/signal_card.html",
                {"request": request, "error": "Market closed - No live data available", "index": index},
            )

        chain_data = await fetcher.get_option_chain(index=index)
        option_chain = chain_data.get("chain", []) if "error" not in chain_data else None

        style_map = {
            "scalping": TradingStyle.SCALPING,
            "intraday": TradingStyle.INTRADAY,
            "swing": TradingStyle.SWING,
        }
        engine = get_signal_engine(style_map.get(style, TradingStyle.INTRADAY))
        signal = engine.analyze(df=df, option_chain=option_chain)

        return templates.TemplateResponse(
            "partials/signal_card.html",
            {
                "request": request,
                "signal": signal,
                "index": index,
                "style": style,
                "chain_data": chain_data if "error" not in chain_data else None,
            },
        )

    except HTTPException:
        return templates.TemplateResponse(
            "partials/signal_card.html",
            {"request": request, "error": "Please login first", "index": index},
        )
    except Exception as e:
        logger.error(f"Signal card error: {e}")
        return templates.TemplateResponse(
            "partials/signal_card.html",
            {"request": request, "error": f"Market closed or error: {str(e)[:50]}", "index": index},
        )


@router.get("/indicator-panel/{index}", response_class=HTMLResponse)
async def indicator_panel_partial(
    request: Request,
    index: str = "NIFTY",
):
    """Render indicator values panel."""
    index = index.upper()

    try:
        require_auth()
        fetcher = get_data_fetcher()

        if index == "NIFTY":
            token = NIFTY_INDEX_TOKEN
        elif index == "SENSEX":
            token = SENSEX_INDEX_TOKEN
        else:
            token = BANKNIFTY_INDEX_TOKEN

        df = await fetcher.fetch_historical_data(
            instrument_token=token,
            timeframe="5minute",
            days=2,
        )

        if df.empty:
            return templates.TemplateResponse(
                "partials/indicator_panel.html",
                {"request": request, "error": "Market closed - No data available", "index": index},
            )

        from app.indicators.trend import calculate_supertrend, calculate_ema, calculate_adx
        from app.indicators.momentum import calculate_rsi, calculate_macd
        from app.indicators.volatility import calculate_atr

        indicators = {
            "supertrend": calculate_supertrend(df),
            "ema": calculate_ema(df),
            "adx": calculate_adx(df),
            "rsi": calculate_rsi(df),
            "macd": calculate_macd(df),
            "atr": calculate_atr(df),
        }

        return templates.TemplateResponse(
            "partials/indicator_panel.html",
            {
                "request": request,
                "indicators": indicators,
                "index": index,
                "price": df["Close"].iloc[-1],
            },
        )

    except Exception as e:
        logger.error(f"Indicator panel error: {e}")
        return templates.TemplateResponse(
            "partials/indicator_panel.html",
            {"request": request, "error": f"Market closed or error: {str(e)[:50]}", "index": index},
        )


@router.get("/option-chain-table/{index}", response_class=HTMLResponse)
async def option_chain_table_partial(
    request: Request,
    index: str = "NIFTY",
):
    """Render option chain table."""
    index = index.upper()

    try:
        require_auth()
        fetcher = get_data_fetcher()

        chain_data = await fetcher.get_option_chain(index=index, strike_count=8)

        if "error" in chain_data:
            return templates.TemplateResponse(
                "partials/option_chain_table.html",
                {"request": request, "error": chain_data["error"], "index": index},
            )

        return templates.TemplateResponse(
            "partials/option_chain_table.html",
            {
                "request": request,
                "chain": chain_data,
                "index": index,
            },
        )

    except Exception as e:
        logger.error(f"Option chain table error: {e}")
        return templates.TemplateResponse(
            "partials/option_chain_table.html",
            {"request": request, "error": f"Market closed or error: {str(e)[:50]}", "index": index},
        )


@router.get("/market-overview", response_class=HTMLResponse)
async def market_overview_partial(request: Request):
    """Render market overview with NIFTY, BANKNIFTY, and SENSEX."""
    try:
        require_auth()
        fetcher = get_data_fetcher()

        # Fetch quotes for all indices
        quotes = await fetcher.fetch_quote(["NSE:NIFTY 50", "NSE:NIFTY BANK", "BSE:SENSEX"])

        nifty = quotes.get("NSE:NIFTY 50", {})
        banknifty = quotes.get("NSE:NIFTY BANK", {})
        sensex = quotes.get("BSE:SENSEX", {})

        return templates.TemplateResponse(
            "partials/market_overview.html",
            {
                "request": request,
                "nifty": nifty,
                "banknifty": banknifty,
                "sensex": sensex,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            },
        )

    except Exception as e:
        logger.error(f"Market overview error: {e}")
        return templates.TemplateResponse(
            "partials/market_overview.html",
            {"request": request, "error": "Market data unavailable"},
        )


@router.get("/ws-status", response_class=HTMLResponse)
async def ws_status_partial(request: Request):
    """Render WebSocket connection status."""
    ws = get_ws_manager()
    status = ws.get_connection_status()

    return templates.TemplateResponse(
        "partials/ws_status.html",
        {"request": request, "status": status},
    )


@router.get("/recommended-option/{index}", response_class=HTMLResponse)
async def recommended_option_partial(
    request: Request,
    index: str = "NIFTY",
    style: str = "intraday",
):
    """Render recommended option card with Greeks and expected prices."""
    index = index.upper()
    style = style.lower()

    try:
        require_auth()
        fetcher = get_data_fetcher()

        if index == "NIFTY":
            token = NIFTY_INDEX_TOKEN
        elif index == "SENSEX":
            token = SENSEX_INDEX_TOKEN
        else:
            token = BANKNIFTY_INDEX_TOKEN

        df = await fetcher.fetch_historical_data(
            instrument_token=token,
            timeframe="5minute",
            days=3,
        )

        if df.empty:
            return templates.TemplateResponse(
                "partials/recommended_option.html",
                {"request": request, "error": "Market closed - No live data available"},
            )

        chain_data = await fetcher.get_option_chain(index=index)
        option_chain = chain_data.get("chain", []) if "error" not in chain_data else None

        style_map = {
            "scalping": TradingStyle.SCALPING,
            "intraday": TradingStyle.INTRADAY,
            "swing": TradingStyle.SWING,
        }
        engine = get_signal_engine(style_map.get(style, TradingStyle.INTRADAY))
        signal = engine.analyze(df=df, option_chain=option_chain)

        # Log if recommended_option is missing
        if signal and not signal.recommended_option:
            logger.warning(f"{index} signal has no recommended_option. Chain data available: {option_chain is not None and len(option_chain) > 0 if option_chain else False}")
            return templates.TemplateResponse(
                "partials/recommended_option.html",
                {"request": request, "error": f"No suitable {index} option found matching criteria"},
            )

        return templates.TemplateResponse(
            "partials/recommended_option.html",
            {
                "request": request,
                "signal": signal,
                "style": style,
            },
        )

    except HTTPException:
        return templates.TemplateResponse(
            "partials/recommended_option.html",
            {"request": request, "error": "Please login first"},
        )
    except Exception as e:
        logger.error(f"Recommended option error: {e}")
        return templates.TemplateResponse(
            "partials/recommended_option.html",
            {"request": request, "error": f"Market closed or error: {str(e)[:50]}"},
        )


@router.get("/live-ticks", response_class=HTMLResponse)
async def live_ticks_partial(request: Request):
    """Render live tick data table."""
    ws = get_ws_manager()
    ticks = ws.get_all_ticks()

    return templates.TemplateResponse(
        "partials/live_ticks.html",
        {"request": request, "ticks": ticks},
    )


@router.get("/toast/{message}", response_class=HTMLResponse)
async def toast_partial(
    request: Request,
    message: str,
    type: str = "info",
):
    """Render toast notification."""
    return templates.TemplateResponse(
        "partials/toast.html",
        {"request": request, "message": message, "type": type},
    )
