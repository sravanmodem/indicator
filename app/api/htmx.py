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
from app.services.signal_quality import get_quality_analyzer
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

        # Score signal quality
        quality_score = None
        if signal:
            quality_analyzer = get_quality_analyzer()
            quality_score = quality_analyzer.analyze_quality(signal)
            logger.info(f"{index} Quality Score: {quality_score.total_score:.1f}/100 (High: {quality_score.is_high_quality})")

        return templates.TemplateResponse(
            "partials/signal_card.html",
            {
                "request": request,
                "signal": signal,
                "quality_score": quality_score,
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

        # Score signal quality
        quality_score = None
        if signal:
            quality_analyzer = get_quality_analyzer()
            quality_score = quality_analyzer.analyze_quality(signal)

        return templates.TemplateResponse(
            "partials/recommended_option.html",
            {
                "request": request,
                "signal": signal,
                "quality_score": quality_score,
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


@router.get("/history/signals", response_class=HTMLResponse)
async def history_signals_partial(
    request: Request,
    days: int = 7,
    result: str = "all",
    index: str = "all",
):
    """Render signals history table."""
    try:
        require_auth()

        from app.services.signal_history_service import get_history_service
        from app.models.signal_history import SignalResult

        history_service = get_history_service()

        # Get signals
        signals = history_service.get_recent_signals(limit=100, days=days)

        # Filter by result
        if result != "all":
            try:
                result_enum = SignalResult(result)
                signals = [s for s in signals if s.result == result_enum]
            except ValueError:
                pass

        # Filter by index
        if index != "all":
            signals = [s for s in signals if s.index_name == index.upper()]

        return templates.TemplateResponse(
            "partials/signals_history_table.html",
            {
                "request": request,
                "signals": signals,
            },
        )

    except Exception as e:
        logger.error(f"History signals error: {e}")
        return templates.TemplateResponse(
            "partials/signals_history_table.html",
            {"request": request, "signals": [], "error": str(e)},
        )


# ============================================================================
# Dynamic Data Endpoints for Real-time Updates
# ============================================================================

@router.get("/market-status", response_class=HTMLResponse)
async def market_status_partial(request: Request):
    """Render live market status bar with NIFTY, BANKNIFTY, SENSEX, VIX."""
    try:
        auth = get_auth_service()
        if not auth.is_authenticated:
            return templates.TemplateResponse(
                "partials/market_status.html",
                {"request": request, "market": None},
            )

        fetcher = get_data_fetcher()

        # Fetch quotes for all indices
        quotes = await fetcher.fetch_quote([
            "NSE:NIFTY 50",
            "NSE:NIFTY BANK",
            "BSE:SENSEX",
            "NSE:INDIA VIX"
        ])

        nifty = quotes.get("NSE:NIFTY 50", {})
        banknifty = quotes.get("NSE:NIFTY BANK", {})
        sensex = quotes.get("BSE:SENSEX", {})
        vix = quotes.get("NSE:INDIA VIX", {})

        # Check if market is open (9:15 AM to 3:30 PM IST on weekdays)
        now = datetime.now()
        is_open = (
            now.weekday() < 5 and  # Monday to Friday
            now.hour >= 9 and now.hour < 16 and
            (now.hour > 9 or now.minute >= 15) and
            (now.hour < 15 or now.minute <= 30)
        )

        market_data = {
            "is_open": is_open,
            "nifty_price": nifty.get("last_price", 0),
            "nifty_change": nifty.get("change", 0),
            "banknifty_price": banknifty.get("last_price", 0),
            "banknifty_change": banknifty.get("change", 0),
            "sensex_price": sensex.get("last_price", 0),
            "sensex_change": sensex.get("change", 0),
            "vix": vix.get("last_price", 0),
        }

        return templates.TemplateResponse(
            "partials/market_status.html",
            {"request": request, "market": market_data},
        )

    except Exception as e:
        logger.error(f"Market status error: {e}")
        return templates.TemplateResponse(
            "partials/market_status.html",
            {"request": request, "market": None},
        )


@router.get("/live-price/{index}", response_class=HTMLResponse)
async def live_price_partial(request: Request, index: str = "NIFTY"):
    """Render live price for a specific index."""
    index = index.upper()

    try:
        auth = get_auth_service()
        if not auth.is_authenticated:
            return HTMLResponse(content="<span class='text-surface-500'>--</span>")

        fetcher = get_data_fetcher()

        # Map index to quote key
        quote_map = {
            "NIFTY": "NSE:NIFTY 50",
            "BANKNIFTY": "NSE:NIFTY BANK",
            "SENSEX": "BSE:SENSEX",
        }

        quote_key = quote_map.get(index, f"NSE:{index}")
        quotes = await fetcher.fetch_quote([quote_key])
        data = quotes.get(quote_key, {})

        price = data.get("last_price", 0)
        change = data.get("change", 0)

        color_class = "text-bullish-400" if change >= 0 else "text-bearish-400"
        sign = "+" if change >= 0 else ""

        return HTMLResponse(
            content=f"""
            <span class="text-2xl font-bold text-white">{price:,.2f}</span>
            <span class="text-sm {color_class} ml-2">{sign}{change:.2f}%</span>
            """
        )

    except Exception as e:
        logger.error(f"Live price error: {e}")
        return HTMLResponse(content="<span class='text-surface-500'>--</span>")


@router.get("/index-card/{index}", response_class=HTMLResponse)
async def index_card_partial(request: Request, index: str = "NIFTY"):
    """Render index card with price and quick signal."""
    index = index.upper()

    try:
        auth = get_auth_service()
        if not auth.is_authenticated:
            return HTMLResponse(content="""
                <div class="text-center py-4 text-surface-500">
                    <p>Please login to view data</p>
                </div>
            """)

        fetcher = get_data_fetcher()

        # Map index to quote key
        quote_map = {
            "NIFTY": "NSE:NIFTY 50",
            "BANKNIFTY": "NSE:NIFTY BANK",
            "SENSEX": "BSE:SENSEX",
        }

        quote_key = quote_map.get(index, f"NSE:{index}")
        quotes = await fetcher.fetch_quote([quote_key])
        data = quotes.get(quote_key, {})

        price = data.get("last_price", 0)
        change = data.get("change", 0)
        net_change = data.get("net_change", 0)
        ohlc = data.get("ohlc", {})

        color_class = "bullish" if change >= 0 else "bearish"
        sign = "+" if change >= 0 else ""

        return HTMLResponse(
            content=f"""
            <div class="flex items-center justify-between mb-4">
                <div>
                    <p class="text-3xl font-bold text-white">{price:,.2f}</p>
                    <p class="text-sm text-{color_class}-400">{sign}{net_change:,.2f} ({sign}{change:.2f}%)</p>
                </div>
                <div class="text-right text-xs text-surface-500">
                    <p>O: {ohlc.get('open', 0):,.2f}</p>
                    <p>H: {ohlc.get('high', 0):,.2f}</p>
                    <p>L: {ohlc.get('low', 0):,.2f}</p>
                </div>
            </div>
            """
        )

    except Exception as e:
        logger.error(f"Index card error: {e}")
        return HTMLResponse(content="""
            <div class="text-center py-4 text-surface-500">
                <p>Data unavailable</p>
            </div>
        """)


@router.get("/live-signals", response_class=HTMLResponse)
async def live_signals_partial(request: Request):
    """Render live high-quality signals panel."""
    try:
        auth = get_auth_service()
        if not auth.is_authenticated:
            return templates.TemplateResponse(
                "partials/live_signals.html",
                {"request": request, "signals": []},
            )

        fetcher = get_data_fetcher()
        quality_analyzer = get_quality_analyzer()

        signals = []

        for idx, token in [
            ("NIFTY", NIFTY_INDEX_TOKEN),
            ("BANKNIFTY", BANKNIFTY_INDEX_TOKEN),
            ("SENSEX", SENSEX_INDEX_TOKEN),
        ]:
            try:
                df = await fetcher.fetch_historical_data(
                    instrument_token=token,
                    timeframe="5minute",
                    days=3,
                )

                if df.empty:
                    continue

                chain_data = await fetcher.get_option_chain(index=idx)
                option_chain = chain_data.get("chain", []) if "error" not in chain_data else None

                engine = get_signal_engine(TradingStyle.INTRADAY)
                signal = engine.analyze(df=df, option_chain=option_chain)

                if signal and signal.recommended_option:
                    quality_score = quality_analyzer.analyze_quality(signal)

                    # Only include high quality signals (70+)
                    if quality_score.total_score >= 70:
                        signals.append({
                            "index": idx,
                            "type": signal.signal_type.value,
                            "option_symbol": signal.recommended_option.symbol,
                            "entry": signal.recommended_option.entry_price,
                            "target": signal.recommended_option.target_price,
                            "stoploss": signal.recommended_option.stop_loss,
                            "risk_reward": signal.recommended_option.risk_reward,
                            "quality_score": quality_score.total_score,
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                        })

            except Exception as e:
                logger.error(f"Live signals error for {idx}: {e}")
                continue

        return templates.TemplateResponse(
            "partials/live_signals.html",
            {"request": request, "signals": signals},
        )

    except Exception as e:
        logger.error(f"Live signals error: {e}")
        return templates.TemplateResponse(
            "partials/live_signals.html",
            {"request": request, "signals": []},
        )


@router.get("/recent-signals", response_class=HTMLResponse)
async def recent_signals_partial(request: Request, index: str = "all"):
    """Render recent signals table."""
    try:
        from app.services.signal_history_service import get_history_service

        history_service = get_history_service()
        signals_db = history_service.get_recent_signals(limit=20, days=1)

        # Filter by index if specified
        if index != "all":
            signals_db = [s for s in signals_db if s.index_name == index.upper()]

        # Convert to template format
        signals = []
        for s in signals_db:
            signals.append({
                "timestamp": s.created_at.strftime("%H:%M:%S") if s.created_at else "--",
                "index": s.index_name,
                "type": s.signal_type,
                "strike": s.strike_price,
                "entry": s.entry_price,
                "target": s.target_price,
                "quality_score": s.quality_score or 0,
                "status": s.result.value if s.result else "active",
            })

        return templates.TemplateResponse(
            "partials/recent_signals.html",
            {"request": request, "signals": signals},
        )

    except Exception as e:
        logger.error(f"Recent signals error: {e}")
        return templates.TemplateResponse(
            "partials/recent_signals.html",
            {"request": request, "signals": []},
        )


@router.get("/quality-breakdown/{index}", response_class=HTMLResponse)
async def quality_breakdown_partial(
    request: Request,
    index: str = "NIFTY",
    style: str = "intraday",
):
    """Render quality score breakdown for an index."""
    index = index.upper()

    try:
        require_auth()
        fetcher = get_data_fetcher()

        token_map = {
            "NIFTY": NIFTY_INDEX_TOKEN,
            "BANKNIFTY": BANKNIFTY_INDEX_TOKEN,
            "SENSEX": SENSEX_INDEX_TOKEN,
        }

        token = token_map.get(index, NIFTY_INDEX_TOKEN)

        df = await fetcher.fetch_historical_data(
            instrument_token=token,
            timeframe="5minute",
            days=3,
        )

        if df.empty:
            return templates.TemplateResponse(
                "partials/quality_breakdown.html",
                {"request": request, "error": "No data available"},
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

        if not signal:
            return templates.TemplateResponse(
                "partials/quality_breakdown.html",
                {"request": request, "error": "No signal generated"},
            )

        quality_analyzer = get_quality_analyzer()
        quality_score = quality_analyzer.analyze_quality(signal)

        return templates.TemplateResponse(
            "partials/quality_breakdown.html",
            {
                "request": request,
                "quality_score": quality_score,
                "index": index,
            },
        )

    except HTTPException:
        return templates.TemplateResponse(
            "partials/quality_breakdown.html",
            {"request": request, "error": "Please login first"},
        )
    except Exception as e:
        logger.error(f"Quality breakdown error: {e}")
        return templates.TemplateResponse(
            "partials/quality_breakdown.html",
            {"request": request, "error": f"Error: {str(e)[:50]}"},
        )


@router.get("/server-time", response_class=HTMLResponse)
async def server_time_partial(request: Request):
    """Render server time."""
    now = datetime.now()
    return HTMLResponse(
        content=f"""<span class="text-xs text-surface-400">{now.strftime("%H:%M:%S")}</span>"""
    )


@router.get("/connection-status", response_class=HTMLResponse)
async def connection_status_partial(request: Request):
    """Render broker connection status."""
    auth = get_auth_service()
    status = auth.get_auth_status()

    if status.get("authenticated"):
        return HTMLResponse(
            content="""
            <div class="flex items-center gap-2">
                <span class="w-2 h-2 rounded-full bg-bullish-500 animate-pulse"></span>
                <span class="text-xs text-bullish-400">Connected</span>
            </div>
            """
        )
    else:
        return HTMLResponse(
            content="""
            <div class="flex items-center gap-2">
                <span class="w-2 h-2 rounded-full bg-surface-500"></span>
                <span class="text-xs text-surface-400">Disconnected</span>
            </div>
            """
        )


@router.get("/dashboard-stats", response_class=HTMLResponse)
async def dashboard_stats_partial(request: Request):
    """Render dashboard statistics cards."""
    try:
        from app.services.signal_history_service import get_history_service

        auth = get_auth_service()
        history_service = get_history_service()

        # Get today's signals
        today_signals = history_service.get_recent_signals(limit=100, days=1)

        # Calculate stats
        total_signals = len(today_signals)
        winners = sum(1 for s in today_signals if s.result and s.result.value == "target_hit")
        losers = sum(1 for s in today_signals if s.result and s.result.value == "stop_loss")
        active = sum(1 for s in today_signals if not s.result or s.result.value == "active")

        win_rate = (winners / (winners + losers) * 100) if (winners + losers) > 0 else 0

        # Calculate P&L
        total_pnl = sum(s.pnl or 0 for s in today_signals)

        pnl_bg = "bg-bullish-500/20" if total_pnl >= 0 else "bg-bearish-500/20"
        pnl_color = "text-bullish-400" if total_pnl >= 0 else "text-bearish-400"
        pnl_sign = "+" if total_pnl >= 0 else ""

        return HTMLResponse(
            content=f"""
            <div class="grid grid-cols-4 gap-4">
                <div class="stat-card">
                    <div class="flex items-center gap-3">
                        <div class="p-2 rounded-lg bg-primary-500/20">
                            <i data-lucide="activity" class="w-5 h-5 text-primary-400"></i>
                        </div>
                        <div>
                            <p class="text-xs text-surface-500 uppercase">Today's Signals</p>
                            <p class="text-2xl font-bold text-white">{total_signals}</p>
                        </div>
                    </div>
                </div>
                <div class="stat-card">
                    <div class="flex items-center gap-3">
                        <div class="p-2 rounded-lg bg-bullish-500/20">
                            <i data-lucide="target" class="w-5 h-5 text-bullish-400"></i>
                        </div>
                        <div>
                            <p class="text-xs text-surface-500 uppercase">Win Rate</p>
                            <p class="text-2xl font-bold text-bullish-400">{win_rate:.1f}%</p>
                        </div>
                    </div>
                </div>
                <div class="stat-card">
                    <div class="flex items-center gap-3">
                        <div class="p-2 rounded-lg {pnl_bg}">
                            <i data-lucide="indian-rupee" class="w-5 h-5 {pnl_color}"></i>
                        </div>
                        <div>
                            <p class="text-xs text-surface-500 uppercase">Today's P&L</p>
                            <p class="text-2xl font-bold {pnl_color}">{pnl_sign}₹{total_pnl:,.0f}</p>
                        </div>
                    </div>
                </div>
                <div class="stat-card">
                    <div class="flex items-center gap-3">
                        <div class="p-2 rounded-lg bg-amber-500/20">
                            <i data-lucide="clock" class="w-5 h-5 text-amber-400"></i>
                        </div>
                        <div>
                            <p class="text-xs text-surface-500 uppercase">Active Trades</p>
                            <p class="text-2xl font-bold text-amber-400">{active}</p>
                        </div>
                    </div>
                </div>
            </div>
            """
        )

    except Exception as e:
        logger.error(f"Dashboard stats error: {e}")
        return HTMLResponse(content="<p class='text-surface-500'>Stats unavailable</p>")


@router.get("/index-price/{index}", response_class=HTMLResponse)
async def index_price_partial(request: Request, index: str = "NIFTY"):
    """Render index price with signal for dashboard cards."""
    index = index.upper()

    try:
        auth = get_auth_service()
        if not auth.is_authenticated:
            return HTMLResponse(content="""
                <div class="space-y-2">
                    <div class="flex items-baseline gap-2">
                        <span class="text-xl font-bold text-surface-500">--</span>
                    </div>
                    <div class="flex items-center gap-2">
                        <span class="badge-neutral">Login required</span>
                    </div>
                </div>
            """)

        fetcher = get_data_fetcher()

        # Get quote
        quote_map = {
            "NIFTY": "NSE:NIFTY 50",
            "BANKNIFTY": "NSE:NIFTY BANK",
            "SENSEX": "BSE:SENSEX",
        }

        quote_key = quote_map.get(index, f"NSE:{index}")
        quotes = await fetcher.fetch_quote([quote_key])
        data = quotes.get(quote_key, {})

        price = data.get("last_price", 0)
        change = data.get("change", 0)
        net_change = data.get("net_change", 0)

        # Get signal for this index
        token_map = {
            "NIFTY": NIFTY_INDEX_TOKEN,
            "BANKNIFTY": BANKNIFTY_INDEX_TOKEN,
            "SENSEX": SENSEX_INDEX_TOKEN,
        }

        signal_type = "Neutral"
        signal_badge = "badge-neutral"
        quality_score = 0

        try:
            token = token_map.get(index, NIFTY_INDEX_TOKEN)
            df = await fetcher.fetch_historical_data(
                instrument_token=token,
                timeframe="5minute",
                days=2,
            )

            if not df.empty:
                chain_data = await fetcher.get_option_chain(index=index)
                option_chain = chain_data.get("chain", []) if "error" not in chain_data else None

                engine = get_signal_engine(TradingStyle.INTRADAY)
                signal = engine.analyze(df=df, option_chain=option_chain)

                if signal:
                    quality_analyzer = get_quality_analyzer()
                    quality_result = quality_analyzer.analyze_quality(signal)
                    quality_score = int(quality_result.total_score)

                    if signal.signal_type.value in ["CE", "STRONG_CE"]:
                        signal_type = "CE Signal"
                        signal_badge = "badge-success"
                    elif signal.signal_type.value in ["PE", "STRONG_PE"]:
                        signal_type = "PE Signal"
                        signal_badge = "badge-danger"
        except Exception as e:
            logger.debug(f"Signal fetch error for {index}: {e}")

        color_class = "text-bullish-400" if change >= 0 else "text-bearish-400"
        sign = "+" if change >= 0 else ""

        return HTMLResponse(
            content=f"""
            <div class="space-y-2">
                <div class="flex items-baseline gap-2">
                    <span class="text-xl font-bold text-white">{price:,.2f}</span>
                    <span class="text-sm {color_class}">{sign}{net_change:,.2f} ({sign}{change:.2f}%)</span>
                </div>
                <div class="flex items-center gap-2">
                    <span class="{signal_badge}">{signal_type}</span>
                    <span class="text-xs text-surface-500">Score: {quality_score}</span>
                </div>
            </div>
            """
        )

    except Exception as e:
        logger.error(f"Index price error for {index}: {e}")
        return HTMLResponse(content="""
            <div class="space-y-2">
                <div class="flex items-baseline gap-2">
                    <span class="text-xl font-bold text-surface-500">--</span>
                </div>
                <div class="flex items-center gap-2">
                    <span class="badge-neutral">Error</span>
                </div>
            </div>
        """)


@router.get("/indicator-summary/{index}", response_class=HTMLResponse)
async def indicator_summary_partial(request: Request, index: str = "NIFTY"):
    """Render technical indicator summary for an index."""
    index = index.upper()

    try:
        auth = get_auth_service()
        if not auth.is_authenticated:
            return HTMLResponse(content="""
                <div class="text-center py-4 text-surface-500">
                    <p>Login to view indicators</p>
                </div>
            """)

        fetcher = get_data_fetcher()

        token_map = {
            "NIFTY": NIFTY_INDEX_TOKEN,
            "BANKNIFTY": BANKNIFTY_INDEX_TOKEN,
            "SENSEX": SENSEX_INDEX_TOKEN,
        }

        token = token_map.get(index, NIFTY_INDEX_TOKEN)

        df = await fetcher.fetch_historical_data(
            instrument_token=token,
            timeframe="5minute",
            days=2,
        )

        if df.empty:
            return HTMLResponse(content="""
                <div class="text-center py-4 text-surface-500">
                    <p>No data available</p>
                </div>
            """)

        from app.indicators.trend import calculate_supertrend, calculate_ema, calculate_adx
        from app.indicators.momentum import calculate_rsi, calculate_macd

        # Calculate indicators
        supertrend = calculate_supertrend(df)
        ema = calculate_ema(df)
        adx = calculate_adx(df)
        rsi = calculate_rsi(df)
        macd = calculate_macd(df)

        # Determine signals
        st_signal = "BUY" if supertrend.get("signal") == "BUY" else "SELL"
        st_color = "bullish" if st_signal == "BUY" else "bearish"

        ema_signal = "BUY" if ema.get("signal") == "BUY" else "SELL"
        ema_color = "bullish" if ema_signal == "BUY" else "bearish"

        rsi_value = rsi.get("value", 50)
        rsi_color = "bullish" if rsi_value < 30 else ("bearish" if rsi_value > 70 else "surface")

        macd_signal = "BUY" if macd.get("signal") == "BUY" else "SELL"
        macd_color = "bullish" if macd_signal == "BUY" else "bearish"

        adx_value = adx.get("value", 0)
        adx_strength = "Strong" if adx_value > 25 else "Weak"

        return HTMLResponse(
            content=f"""
            <div class="flex items-center justify-between py-2 border-b border-surface-800">
                <div class="flex items-center gap-2">
                    <span class="w-2 h-2 rounded-full bg-{st_color}-500"></span>
                    <span class="text-sm text-surface-300">SuperTrend</span>
                </div>
                <span class="text-sm font-medium text-{st_color}-400">{st_signal}</span>
            </div>
            <div class="flex items-center justify-between py-2 border-b border-surface-800">
                <div class="flex items-center gap-2">
                    <span class="w-2 h-2 rounded-full bg-{ema_color}-500"></span>
                    <span class="text-sm text-surface-300">EMA Cross</span>
                </div>
                <span class="text-sm font-medium text-{ema_color}-400">{ema_signal}</span>
            </div>
            <div class="flex items-center justify-between py-2 border-b border-surface-800">
                <div class="flex items-center gap-2">
                    <span class="w-2 h-2 rounded-full bg-{rsi_color}-500"></span>
                    <span class="text-sm text-surface-300">RSI (14)</span>
                </div>
                <span class="text-sm font-medium text-surface-400">{rsi_value:.1f}</span>
            </div>
            <div class="flex items-center justify-between py-2 border-b border-surface-800">
                <div class="flex items-center gap-2">
                    <span class="w-2 h-2 rounded-full bg-{macd_color}-500"></span>
                    <span class="text-sm text-surface-300">MACD</span>
                </div>
                <span class="text-sm font-medium text-{macd_color}-400">{macd_signal}</span>
            </div>
            <div class="flex items-center justify-between py-2">
                <div class="flex items-center gap-2">
                    <span class="w-2 h-2 rounded-full bg-surface-500"></span>
                    <span class="text-sm text-surface-300">ADX</span>
                </div>
                <span class="text-sm font-medium text-surface-400">{adx_value:.1f} ({adx_strength})</span>
            </div>
            """
        )

    except Exception as e:
        logger.error(f"Indicator summary error for {index}: {e}")
        return HTMLResponse(content="""
            <div class="text-center py-4 text-surface-500">
                <p>Error loading indicators</p>
            </div>
        """)


@router.get("/market-sentiment", response_class=HTMLResponse)
async def market_sentiment_partial(request: Request):
    """Render market sentiment panel with PCR, VIX, Max Pain."""
    try:
        auth = get_auth_service()
        if not auth.is_authenticated:
            return HTMLResponse(content="""
                <div class="text-center py-4 text-surface-500">
                    <p>Login to view sentiment</p>
                </div>
            """)

        fetcher = get_data_fetcher()

        # Get VIX
        vix_quote = await fetcher.fetch_quote(["NSE:INDIA VIX"])
        vix = vix_quote.get("NSE:INDIA VIX", {}).get("last_price", 0)

        # Get NIFTY option chain for PCR
        chain_data = await fetcher.get_option_chain(index="NIFTY")

        pcr = chain_data.get("pcr", 0) if "error" not in chain_data else 0
        spot_price = chain_data.get("spot_price", 0) if "error" not in chain_data else 0
        atm_strike = chain_data.get("atm_strike", 0) if "error" not in chain_data else 0

        # Determine sentiment
        if pcr < 0.7:
            sentiment = "Bearish"
            sentiment_badge = "badge-danger"
        elif pcr < 1.0:
            sentiment = "Mildly Bullish"
            sentiment_badge = "badge-success"
        elif pcr < 1.3:
            sentiment = "Bullish"
            sentiment_badge = "badge-success"
        else:
            sentiment = "Very Bullish"
            sentiment_badge = "badge-success"

        # VIX color
        vix_color = "bullish" if vix < 15 else ("amber" if vix < 20 else "bearish")
        vix_label = "Low Volatility" if vix < 15 else ("Moderate" if vix < 20 else "High Volatility")

        # PCR progress (scale 0-2 to 0-100%)
        pcr_pct = min(100, int((pcr / 2) * 100))
        pcr_color = "bullish" if pcr > 0.7 else "bearish"

        # Max pain estimate (ATM)
        max_pain = atm_strike
        diff = spot_price - max_pain
        diff_sign = "+" if diff >= 0 else ""

        return HTMLResponse(
            content=f"""
            <!-- PCR -->
            <div>
                <div class="flex justify-between text-xs mb-1">
                    <span class="text-surface-500">Put/Call Ratio</span>
                    <span class="text-surface-300 font-medium">{pcr:.2f}</span>
                </div>
                <div class="h-2 bg-surface-800 rounded-full overflow-hidden">
                    <div class="h-full bg-gradient-to-r from-{pcr_color}-600 to-{pcr_color}-400 rounded-full" style="width: {pcr_pct}%"></div>
                </div>
                <p class="text-2xs text-surface-600 mt-1">{sentiment}</p>
            </div>

            <!-- VIX -->
            <div>
                <div class="flex justify-between text-xs mb-1">
                    <span class="text-surface-500">India VIX</span>
                    <span class="text-surface-300 font-medium">{vix:.2f}</span>
                </div>
                <div class="h-2 bg-surface-800 rounded-full overflow-hidden">
                    <div class="h-full bg-gradient-to-r from-{vix_color}-600 to-{vix_color}-400 rounded-full" style="width: {min(100, int(vix * 3))}%"></div>
                </div>
                <p class="text-2xs text-surface-600 mt-1">{vix_label}</p>
            </div>

            <!-- Max Pain -->
            <div>
                <div class="flex justify-between text-xs mb-1">
                    <span class="text-surface-500">Max Pain (NIFTY)</span>
                    <span class="text-surface-300 font-medium">{max_pain:,.0f}</span>
                </div>
                <p class="text-2xs text-surface-600">Spot at {spot_price:,.2f} ({diff_sign}{diff:,.0f})</p>
            </div>
            """
        )

    except Exception as e:
        logger.error(f"Market sentiment error: {e}")
        return HTMLResponse(content="""
            <div class="text-center py-4 text-surface-500">
                <p>Error loading sentiment</p>
            </div>
        """)


@router.get("/hero-stats", response_class=HTMLResponse)
async def hero_stats_partial(request: Request):
    """Render hero stats row (Portfolio, P&L, Active Signals, Win Rate)."""
    try:
        from app.services.signal_history_service import get_history_service

        history_service = get_history_service()
        settings = history_service.get_settings()
        stats_30d = history_service.get_statistics(days=30)
        today_signals = history_service.get_recent_signals(limit=100, days=1)

        # Portfolio value from settings
        portfolio_value = settings.total_capital if settings else 0

        # Calculate today's P&L
        today_pnl = sum(s.actual_pnl or 0 for s in today_signals if s.actual_pnl)
        pnl_pct = (today_pnl / portfolio_value * 100) if portfolio_value > 0 else 0
        pnl_color = "text-bullish-400" if today_pnl >= 0 else "text-bearish-400"
        pnl_badge = "badge-success" if today_pnl >= 0 else "badge-danger"
        pnl_sign = "+" if today_pnl >= 0 else ""
        pnl_icon = "trending-up" if today_pnl >= 0 else "trending-down"
        pnl_bg = "bg-bullish-500/10" if today_pnl >= 0 else "bg-bearish-500/10"

        # Active signals count
        active_count = sum(1 for s in today_signals if not s.result or s.result.value == "pending")
        high_quality_active = sum(1 for s in today_signals if (not s.result or s.result.value == "pending") and s.quality_score and s.quality_score >= 70)

        # Win rate from 30-day stats
        win_rate = stats_30d.get("win_rate", 0)
        target_hit = stats_30d.get("target_hit", 0)
        total_closed = stats_30d.get("target_hit", 0) + stats_30d.get("stop_loss_hit", 0)

        return HTMLResponse(content=f"""
            <!-- Portfolio Value Card -->
            <div class="card p-5">
                <div class="flex items-center justify-between mb-3">
                    <span class="data-label">Portfolio Value</span>
                    <div class="p-2 rounded-lg bg-primary-500/10">
                        <i data-lucide="wallet" class="w-4 h-4 text-primary-400"></i>
                    </div>
                </div>
                <div class="flex items-baseline gap-2">
                    <span class="text-2xl font-bold text-white">₹{portfolio_value:,.0f}</span>
                </div>
                <p class="mt-1 text-xs text-surface-500">Capital allocated for trading</p>
            </div>

            <!-- Day P&L Card -->
            <div class="card p-5">
                <div class="flex items-center justify-between mb-3">
                    <span class="data-label">Today's P&L</span>
                    <div class="p-2 rounded-lg {pnl_bg}">
                        <i data-lucide="{pnl_icon}" class="w-4 h-4 {pnl_color}"></i>
                    </div>
                </div>
                <div class="flex items-baseline gap-2">
                    <span class="text-2xl font-bold {pnl_color}">{pnl_sign}₹{abs(today_pnl):,.0f}</span>
                    <span class="{pnl_badge} text-xs">{pnl_sign}{pnl_pct:.2f}%</span>
                </div>
                <p class="mt-1 text-xs text-surface-500">Realized + Unrealized</p>
            </div>

            <!-- Active Signals Card -->
            <div class="card p-5">
                <div class="flex items-center justify-between mb-3">
                    <span class="data-label">Active Signals</span>
                    <div class="p-2 rounded-lg bg-amber-500/10">
                        <i data-lucide="zap" class="w-4 h-4 text-amber-400"></i>
                    </div>
                </div>
                <div class="flex items-baseline gap-2">
                    <span class="text-2xl font-bold text-white">{active_count}</span>
                    <span class="badge-warning text-xs">{high_quality_active} High Quality</span>
                </div>
                <p class="mt-1 text-xs text-surface-500">Signals above 70 score</p>
            </div>

            <!-- Win Rate Card -->
            <div class="card p-5">
                <div class="flex items-center justify-between mb-3">
                    <span class="data-label">Win Rate (30D)</span>
                    <div class="p-2 rounded-lg bg-bullish-500/10">
                        <i data-lucide="target" class="w-4 h-4 text-bullish-400"></i>
                    </div>
                </div>
                <div class="flex items-baseline gap-2">
                    <span class="text-2xl font-bold text-white">{win_rate:.0f}%</span>
                    <span class="badge-neutral text-xs">{target_hit}/{total_closed} trades</span>
                </div>
                <p class="mt-1 text-xs text-surface-500">Based on signal history</p>
            </div>
        """)

    except Exception as e:
        logger.error(f"Hero stats error: {e}")
        return HTMLResponse(content="""
            <div class="col-span-4 card p-5 text-center">
                <i data-lucide="alert-circle" class="w-8 h-8 mx-auto mb-2 text-surface-600"></i>
                <p class="text-surface-500">Unable to load statistics</p>
                <p class="text-xs text-surface-600 mt-1">Please refresh the page</p>
            </div>
        """)


@router.get("/trading-stats-30d", response_class=HTMLResponse)
async def trading_stats_30d_partial(request: Request):
    """Render 30-day trading statistics."""
    try:
        from app.services.signal_history_service import get_history_service

        history_service = get_history_service()
        stats = history_service.get_statistics(days=30)

        total_signals = stats.get("total_signals", 0)
        target_hit = stats.get("target_hit", 0)
        win_rate = stats.get("win_rate", 0)
        avg_quality = stats.get("avg_quality_score", 0)

        if total_signals == 0:
            return HTMLResponse(content="""
                <div class="text-center py-6">
                    <i data-lucide="bar-chart-2" class="w-10 h-10 mx-auto mb-3 text-surface-600"></i>
                    <p class="text-surface-400 text-sm">No trading data yet</p>
                    <p class="text-surface-500 text-xs mt-1">Statistics will appear after your first signals</p>
                </div>
            """)

        # Calculate high quality signals (score >= 70)
        high_quality_count = int(total_signals * (avg_quality / 100)) if avg_quality > 0 else 0
        high_quality_pct = (high_quality_count / total_signals * 100) if total_signals > 0 else 0

        # Determine best performing index (would need more data, using placeholder logic)
        best_index = "NIFTY"  # Default, would need per-index tracking

        return HTMLResponse(content=f"""
            <div class="space-y-4">
                <div class="flex justify-between items-center">
                    <span class="text-sm text-surface-400">Total Signals</span>
                    <span class="text-sm font-semibold text-white">{total_signals}</span>
                </div>
                <div class="flex justify-between items-center">
                    <span class="text-sm text-surface-400">High Quality</span>
                    <span class="text-sm font-semibold text-bullish-400">{high_quality_count} ({high_quality_pct:.0f}%)</span>
                </div>
                <div class="flex justify-between items-center">
                    <span class="text-sm text-surface-400">Target Hit</span>
                    <span class="text-sm font-semibold text-bullish-400">{win_rate:.0f}%</span>
                </div>
                <div class="flex justify-between items-center">
                    <span class="text-sm text-surface-400">Avg. Score</span>
                    <span class="text-sm font-semibold text-white">{avg_quality:.1f}</span>
                </div>
                <div class="flex justify-between items-center">
                    <span class="text-sm text-surface-400">Best Index</span>
                    <span class="text-sm font-semibold text-primary-400">{best_index}</span>
                </div>
            </div>
        """)

    except Exception as e:
        logger.error(f"30-day stats error: {e}")
        return HTMLResponse(content="""
            <div class="text-center py-4 text-surface-500">
                <p class="text-sm">Unable to load statistics</p>
            </div>
        """)
