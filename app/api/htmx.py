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

        from app.indicators.trend import calculate_supertrend, calculate_ema, calculate_adx, calculate_vwap
        from app.indicators.momentum import calculate_rsi, calculate_macd, calculate_stochastic
        from app.indicators.volume import calculate_obv
        from app.indicators.volatility import calculate_atr, calculate_bollinger_bands

        # Calculate all indicators
        supertrend = calculate_supertrend(df)
        ema = calculate_ema(df)
        adx = calculate_adx(df)
        rsi = calculate_rsi(df)
        macd = calculate_macd(df)
        atr = calculate_atr(df)

        price = df["Close"].iloc[-1]

        # Try optional indicators with fallbacks
        try:
            vwap = calculate_vwap(df)
            vwap_value = vwap.vwap.iloc[-1] if hasattr(vwap, 'vwap') else price
        except Exception:
            vwap_value = price

        try:
            stoch = calculate_stochastic(df)
            stoch_k = stoch.k.iloc[-1] if hasattr(stoch, 'k') else 50
            stoch_d = stoch.d.iloc[-1] if hasattr(stoch, 'd') else 50
        except Exception:
            stoch_k, stoch_d = 50, 50

        try:
            obv = calculate_obv(df)
            obv_trend = "UP" if obv.obv.iloc[-1] > obv.obv.iloc[-5] else "DOWN"
        except Exception:
            obv_trend = "NEUTRAL"

        try:
            bb = calculate_bollinger_bands(df)
            bb_upper = bb.upper.iloc[-1]
            bb_lower = bb.lower.iloc[-1]
            bb_middle = bb.middle.iloc[-1]
            bb_width = ((bb_upper - bb_lower) / bb_middle * 100) if bb_middle > 0 else 0
            if price > bb_upper:
                bb_position = "Above Upper"
            elif price < bb_lower:
                bb_position = "Below Lower"
            else:
                bb_position = "Middle"
        except Exception:
            bb_position, bb_width = "Middle", 2.0

        # Build indicators dict for template
        st_signal = "BUY" if supertrend.direction.iloc[-1] == 1 else "SELL"
        ema_signal = "BUY" if ema.ema_fast.iloc[-1] > ema.ema_slow.iloc[-1] else "SELL"
        macd_signal = "BUY" if macd.macd_line.iloc[-1] > macd.signal_line.iloc[-1] else "SELL"

        # Count bullish signals
        buy_count = sum([
            st_signal == "BUY",
            ema_signal == "BUY",
            macd_signal == "BUY",
            rsi.rsi.iloc[-1] < 50,
            stoch_k < 50,
        ])
        total_count = 5

        if buy_count >= 4:
            overall = "STRONG_BUY"
        elif buy_count >= 3:
            overall = "BUY"
        elif buy_count <= 1:
            overall = "STRONG_SELL"
        elif buy_count <= 2:
            overall = "SELL"
        else:
            overall = "NEUTRAL"

        indicators = {
            "supertrend": {
                "signal": st_signal,
                "value": supertrend.supertrend.iloc[-1],
            },
            "ema": {
                "signal": ema_signal,
                "fast": ema.ema_fast.iloc[-1],
                "slow": ema.ema_slow.iloc[-1],
            },
            "vwap": {
                "value": vwap_value,
                "above": price > vwap_value,
            },
            "adx": {
                "value": adx.adx.iloc[-1],
            },
            "rsi": {
                "value": rsi.rsi.iloc[-1],
            },
            "macd": {
                "signal": macd_signal,
                "histogram": macd.histogram.iloc[-1],
            },
            "stoch": {
                "k": stoch_k,
                "d": stoch_d,
            },
            "obv": {
                "trend": obv_trend,
            },
            "bb": {
                "position": bb_position,
                "width": bb_width,
            },
            "atr": {
                "value": atr.atr.iloc[-1],
                "percent": (atr.atr.iloc[-1] / price * 100) if price > 0 else 0,
            },
            "overall": overall,
            "buy_count": buy_count,
            "total_count": total_count,
        }

        return templates.TemplateResponse(
            "partials/indicator_panel.html",
            {
                "request": request,
                "indicators": indicators,
                "index": index,
                "price": price,
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

        # Calculate percentage change if not provided correctly
        if change == 0 and net_change != 0 and price != 0:
            prev_price = price - net_change
            if prev_price != 0:
                change = (net_change / prev_price) * 100

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

        # Calculate percentage change if not provided correctly
        if change == 0 and net_change != 0 and price != 0:
            # Calculate percentage from net_change
            prev_price = price - net_change
            if prev_price != 0:
                change = (net_change / prev_price) * 100

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

        # Determine signals from dataclass results
        st_signal = "BUY" if supertrend.direction.iloc[-1] == 1 else "SELL"
        st_color = "bullish" if st_signal == "BUY" else "bearish"

        ema_signal = "BUY" if ema.ema_fast.iloc[-1] > ema.ema_slow.iloc[-1] else "SELL"
        ema_color = "bullish" if ema_signal == "BUY" else "bearish"

        rsi_value = rsi.rsi.iloc[-1]
        rsi_color = "bullish" if rsi_value < 30 else ("bearish" if rsi_value > 70 else "surface")

        macd_signal = "BUY" if macd.macd_line.iloc[-1] > macd.signal_line.iloc[-1] else "SELL"
        macd_color = "bullish" if macd_signal == "BUY" else "bearish"

        adx_value = adx.adx.iloc[-1]
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

        # Portfolio value from settings (dict)
        portfolio_value = settings.get("total_capital", 250000) if settings else 250000

        # Calculate today's P&L (signals are dicts)
        today_pnl = sum(s.get("pnl", 0) or 0 for s in today_signals)
        pnl_pct = (today_pnl / portfolio_value * 100) if portfolio_value > 0 else 0
        pnl_color = "text-bullish-400" if today_pnl >= 0 else "text-bearish-400"
        pnl_badge = "badge-success" if today_pnl >= 0 else "badge-danger"
        pnl_sign = "+" if today_pnl >= 0 else ""
        pnl_icon = "trending-up" if today_pnl >= 0 else "trending-down"
        pnl_bg = "bg-bullish-500/10" if today_pnl >= 0 else "bg-bearish-500/10"

        # Active signals count (signals are dicts)
        active_count = sum(1 for s in today_signals if s.get("outcome") in (None, "pending", "open"))
        high_quality_active = sum(1 for s in today_signals if s.get("outcome") in (None, "pending", "open") and s.get("quality_score", 0) >= 70)

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

def get_instrument_token(index: str) -> int:
    """Get instrument token for an index."""
    tokens = {
        "NIFTY": NIFTY_INDEX_TOKEN,
        "BANKNIFTY": BANKNIFTY_INDEX_TOKEN,
        "SENSEX": SENSEX_INDEX_TOKEN,
    }
    return tokens.get(index.upper(), NIFTY_INDEX_TOKEN)


@router.get("/mini-chain/{index}", response_class=HTMLResponse)
async def mini_chain_partial(request: Request, index: str = "NIFTY"):
    """Render mini option chain (ATM strikes)."""
    try:
        require_auth()
        fetcher = get_data_fetcher()
        chain_data = await fetcher.get_option_chain(index=index.upper())

        if "error" in chain_data:
            return HTMLResponse(content="""
                <div class="p-4 text-center text-surface-500">
                    <p class="text-sm">Login to view option chain</p>
                </div>
            """)

        chain = chain_data.get("chain", [])
        atm_strike = chain_data.get("atm_strike", 0)

        atm_options = [opt for opt in chain if abs(opt.get("strike", 0) - atm_strike) <= 200]

        if not atm_options:
            return HTMLResponse(content="""
                <div class="p-4 text-center text-surface-500">
                    <p class="text-sm">No ATM options available</p>
                </div>
            """)

        html = ""
        for opt in sorted(atm_options, key=lambda x: x.get("strike", 0))[:5]:
            strike = opt.get("strike", 0)
            ce_ltp = opt.get("ce_ltp", 0)
            pe_ltp = opt.get("pe_ltp", 0)
            ce_oi = opt.get("ce_oi", 0)
            pe_oi = opt.get("pe_oi", 0)
            is_atm = strike == atm_strike
            atm_badge = '<span class="badge-primary text-2xs ml-1">ATM</span>' if is_atm else ""
            row_class = "bg-primary-500/5" if is_atm else ""

            html += f"""
                <div class="flex items-center justify-between p-3 {row_class}">
                    <div class="text-left">
                        <p class="text-sm font-medium text-bullish-400">Rs{ce_ltp:.1f}</p>
                        <p class="text-2xs text-surface-500">{ce_oi:,} OI</p>
                    </div>
                    <div class="text-center">
                        <p class="text-sm font-semibold text-white">{strike:,.0f}{atm_badge}</p>
                    </div>
                    <div class="text-right">
                        <p class="text-sm font-medium text-bearish-400">Rs{pe_ltp:.1f}</p>
                        <p class="text-2xs text-surface-500">{pe_oi:,} OI</p>
                    </div>
                </div>
            """

        return HTMLResponse(content=html)

    except HTTPException:
        return HTMLResponse(content="""
            <div class="p-4 text-center text-surface-500">
                <p class="text-sm">Please login first</p>
            </div>
        """)
    except Exception as e:
        logger.error(f"Mini chain error: {e}")
        return HTMLResponse(content="""
            <div class="p-4 text-center text-surface-500">
                <p class="text-sm">Error loading chain</p>
            </div>
        """)


@router.get("/trend-indicators/{index}", response_class=HTMLResponse)
async def trend_indicators_partial(request: Request, index: str = "NIFTY"):
    """Render trend indicators panel."""
    try:
        require_auth()
        fetcher = get_data_fetcher()
        token = get_instrument_token(index.upper())
        df = await fetcher.fetch_historical_data(instrument_token=token, timeframe="5minute", days=5)

        if df.empty:
            return HTMLResponse(content="""<div class="text-center py-4 text-surface-500"><p class="text-sm">No data available</p></div>""")

        from app.indicators.trend import calculate_supertrend, calculate_ema, calculate_vwap, calculate_adx

        st = calculate_supertrend(df)
        ema = calculate_ema(df)
        vwap = calculate_vwap(df)
        adx = calculate_adx(df)
        current_price = df["Close"].iloc[-1]

        st_dir = "Bullish" if st.direction.iloc[-1] == 1 else "Bearish"
        st_color = "text-bullish-400" if st_dir == "Bullish" else "text-bearish-400"
        ema_signal = "Bullish" if ema.ema_fast.iloc[-1] > ema.ema_slow.iloc[-1] else "Bearish"
        ema_color = "text-bullish-400" if ema_signal == "Bullish" else "text-bearish-400"
        vwap_signal = "Above" if current_price > vwap.vwap.iloc[-1] else "Below"
        vwap_color = "text-bullish-400" if vwap_signal == "Above" else "text-bearish-400"
        adx_value = adx.adx.iloc[-1]

        return HTMLResponse(content=f"""
            <div class="space-y-4">
                <div class="flex items-center justify-between">
                    <span class="text-sm text-surface-300">SuperTrend</span>
                    <span class="text-sm font-medium {st_color}">{st_dir}</span>
                </div>
                <div class="flex items-center justify-between">
                    <span class="text-sm text-surface-300">EMA (9/21)</span>
                    <span class="text-sm font-medium {ema_color}">{ema_signal}</span>
                </div>
                <div class="flex items-center justify-between">
                    <span class="text-sm text-surface-300">VWAP</span>
                    <span class="text-sm font-medium {vwap_color}">{vwap_signal}</span>
                </div>
                <div class="flex items-center justify-between">
                    <span class="text-sm text-surface-300">ADX</span>
                    <span class="text-sm font-medium text-amber-400">{adx_value:.1f}</span>
                </div>
            </div>
        """)
    except HTTPException:
        return HTMLResponse(content="""<div class="text-center py-4 text-surface-500"><p class="text-sm">Please login first</p></div>""")
    except Exception as e:
        logger.error(f"Trend indicators error: {e}")
        return HTMLResponse(content="""<div class="text-center py-4 text-surface-500"><p class="text-sm">Error loading indicators</p></div>""")


@router.get("/momentum-indicators/{index}", response_class=HTMLResponse)
async def momentum_indicators_partial(request: Request, index: str = "NIFTY"):
    """Render momentum indicators panel."""
    try:
        require_auth()
        fetcher = get_data_fetcher()
        token = get_instrument_token(index.upper())
        df = await fetcher.fetch_historical_data(instrument_token=token, timeframe="5minute", days=5)

        if df.empty:
            return HTMLResponse(content="""<div class="text-center py-4 text-surface-500"><p class="text-sm">No data available</p></div>""")

        from app.indicators.momentum import calculate_rsi, calculate_macd, calculate_stochastic

        rsi = calculate_rsi(df)
        macd = calculate_macd(df)
        stoch = calculate_stochastic(df)

        rsi_value = rsi.rsi.iloc[-1]
        rsi_signal = "Overbought" if rsi_value > 70 else "Oversold" if rsi_value < 30 else "Neutral"
        rsi_color = "text-bearish-400" if rsi_value > 70 else "text-bullish-400" if rsi_value < 30 else "text-surface-400"

        macd_signal = "Bullish" if macd.macd_line.iloc[-1] > macd.signal_line.iloc[-1] else "Bearish"
        macd_color = "text-bullish-400" if macd_signal == "Bullish" else "text-bearish-400"

        stoch_k = stoch.k.iloc[-1]
        stoch_signal = "Overbought" if stoch_k > 80 else "Oversold" if stoch_k < 20 else "Neutral"

        return HTMLResponse(content=f"""
            <div class="space-y-4">
                <div class="flex items-center justify-between">
                    <span class="text-sm text-surface-300">RSI (14)</span>
                    <span class="text-sm font-medium {rsi_color}">{rsi_value:.1f} - {rsi_signal}</span>
                </div>
                <div class="flex items-center justify-between">
                    <span class="text-sm text-surface-300">MACD</span>
                    <span class="text-sm font-medium {macd_color}">{macd_signal}</span>
                </div>
                <div class="flex items-center justify-between">
                    <span class="text-sm text-surface-300">Stochastic</span>
                    <span class="text-sm font-medium text-surface-300">{stoch_k:.1f} - {stoch_signal}</span>
                </div>
            </div>
        """)
    except HTTPException:
        return HTMLResponse(content="""<div class="text-center py-4 text-surface-500"><p class="text-sm">Please login first</p></div>""")
    except Exception as e:
        logger.error(f"Momentum indicators error: {e}")
        return HTMLResponse(content="""<div class="text-center py-4 text-surface-500"><p class="text-sm">Error loading indicators</p></div>""")


@router.get("/key-levels/{index}", response_class=HTMLResponse)
async def key_levels_partial(request: Request, index: str = "NIFTY"):
    """Render key support/resistance levels."""
    try:
        require_auth()
        fetcher = get_data_fetcher()
        token = get_instrument_token(index.upper())
        df = await fetcher.fetch_historical_data(instrument_token=token, timeframe="day", days=5)

        if df.empty or len(df) < 2:
            return HTMLResponse(content="""<div class="text-center py-4 text-surface-500"><p class="text-sm">No data available</p></div>""")

        from app.indicators.pivots import calculate_pivot_points

        prev_day = df.iloc[-2]
        pivots = calculate_pivot_points(prev_day["High"], prev_day["Low"], prev_day["Close"])
        current_price = df["Close"].iloc[-1]

        return HTMLResponse(content=f"""
            <div class="space-y-2 text-sm">
                <div class="flex justify-between"><span class="text-bearish-400">R2</span><span class="font-mono">{pivots.r2:,.1f}</span></div>
                <div class="flex justify-between"><span class="text-bearish-400">R1</span><span class="font-mono">{pivots.r1:,.1f}</span></div>
                <div class="flex justify-between py-1 px-2 bg-primary-500/10 rounded"><span class="text-primary-400">Pivot</span><span class="font-mono font-medium">{pivots.pivot:,.1f}</span></div>
                <div class="flex justify-between"><span class="text-bullish-400">S1</span><span class="font-mono">{pivots.s1:,.1f}</span></div>
                <div class="flex justify-between"><span class="text-bullish-400">S2</span><span class="font-mono">{pivots.s2:,.1f}</span></div>
                <div class="pt-2 border-t border-surface-700 flex justify-between"><span class="text-surface-400">Current</span><span class="font-mono font-medium text-white">{current_price:,.1f}</span></div>
            </div>
        """)
    except HTTPException:
        return HTMLResponse(content="""<div class="text-center py-4 text-surface-500"><p class="text-sm">Please login first</p></div>""")
    except Exception as e:
        logger.error(f"Key levels error: {e}")
        return HTMLResponse(content="""<div class="text-center py-4 text-surface-500"><p class="text-sm">Error loading levels</p></div>""")


@router.get("/sentiment/{index}", response_class=HTMLResponse)
async def sentiment_partial(request: Request, index: str = "NIFTY"):
    """Render market sentiment for specific index."""
    try:
        require_auth()
        fetcher = get_data_fetcher()
        chain_data = await fetcher.get_option_chain(index=index.upper())

        if "error" in chain_data:
            return HTMLResponse(content="""<div class="text-center py-4 text-surface-500"><p class="text-sm">Login to view sentiment</p></div>""")

        pcr = chain_data.get("pcr", 1.0)
        sentiment = "Bearish" if pcr < 0.7 else "Mildly Bullish" if pcr < 1.0 else "Bullish" if pcr < 1.3 else "Very Bullish"
        max_pain = chain_data.get("max_pain", 0)

        return HTMLResponse(content=f"""
            <div class="space-y-3">
                <div class="flex justify-between"><span class="text-surface-400">PCR</span><span class="font-medium">{pcr:.2f} - {sentiment}</span></div>
                <div class="flex justify-between"><span class="text-surface-400">Max Pain</span><span class="font-mono">{max_pain:,.0f}</span></div>
            </div>
        """)
    except HTTPException:
        return HTMLResponse(content="""<div class="text-center py-4 text-surface-500"><p class="text-sm">Please login first</p></div>""")
    except Exception as e:
        logger.error(f"Sentiment error: {e}")
        return HTMLResponse(content="""<div class="text-center py-4 text-surface-500"><p class="text-sm">Error loading sentiment</p></div>""")


@router.get("/recent-signals/{index}", response_class=HTMLResponse)
async def recent_signals_partial(request: Request, index: str = "NIFTY"):
    """Render recent signals table rows."""
    try:
        from app.services.signal_history_service import get_history_service
        history_service = get_history_service()
        signals = history_service.get_signals(days=7, index=index.upper())[:10]

        if not signals:
            return HTMLResponse(content="""<tr><td colspan="8" class="text-center py-6 text-surface-500">No recent signals for this index</td></tr>""")

        html = ""
        for s in signals:
            timestamp = s.get("recorded_at", datetime.now())
            time_str = timestamp.strftime("%H:%M") if timestamp else "--:--"
            signal_type = s.get("signal_type", "")
            type_badge = "badge-success" if "CE" in str(signal_type).upper() else "badge-danger"
            outcome = s.get("outcome", "pending")
            status_badge = '<span class="badge-success">Target Hit</span>' if outcome == "target_hit" else '<span class="badge-danger">SL Hit</span>' if outcome == "stop_loss_hit" else '<span class="badge-warning">Open</span>'

            html += f"""<tr>
                <td class="text-surface-300">{time_str}</td>
                <td><span class="{type_badge}">{signal_type}</span></td>
                <td class="font-mono">{s.get("strike", 0):,.0f}</td>
                <td class="font-mono">Rs{s.get("entry", 0):.1f}</td>
                <td class="font-mono text-bullish-400">Rs{s.get("target", 0):.1f}</td>
                <td class="font-mono text-bearish-400">Rs{s.get("stop_loss", 0):.1f}</td>
                <td class="font-medium">{s.get("quality_score", 0):.0f}</td>
                <td>{status_badge}</td>
            </tr>"""

        return HTMLResponse(content=html)
    except Exception as e:
        logger.error(f"Recent signals error: {e}")
        return HTMLResponse(content="""<tr><td colspan="8" class="text-center py-6 text-surface-500">Error loading signals</td></tr>""")


@router.get("/paper-closed-positions", response_class=HTMLResponse)
async def paper_closed_positions_partial(request: Request):
    """Render closed paper trading positions table rows."""
    try:
        from app.services.paper_trading_service import get_paper_trading_service
        from app.models.paper_trading import PositionStatus

        service = get_paper_trading_service()
        positions = service.get_all_positions(limit=20)

        # Filter to only closed positions
        closed = [p for p in positions if p.status == PositionStatus.CLOSED]

        if not closed:
            return HTMLResponse(content="""
                <tr>
                    <td colspan="7" class="text-center py-6 text-surface-500">
                        <i data-lucide="inbox" class="w-8 h-8 mx-auto mb-2 text-surface-700"></i>
                        <p>No closed positions yet</p>
                    </td>
                </tr>
            """)

        html = ""
        for p in closed:
            pnl_class = "text-bullish-400" if (p.pnl or 0) >= 0 else "text-bearish-400"
            pnl_sign = "+" if (p.pnl or 0) >= 0 else ""
            result_badge = "badge-success" if p.exit_reason == "target_hit" else "badge-danger" if p.exit_reason == "stop_loss_hit" else "badge-neutral"
            result_text = "Target Hit" if p.exit_reason == "target_hit" else "SL Hit" if p.exit_reason == "stop_loss_hit" else "Manual"
            type_badge = "badge-success" if p.option_type == "CE" else "badge-danger"

            html += f"""
                <tr class="hover:bg-surface-800/30">
                    <td class="px-4 py-3 text-surface-200">{p.option_symbol}</td>
                    <td class="px-4 py-3"><span class="{type_badge}">{p.option_type}</span></td>
                    <td class="px-4 py-3 text-surface-300">{p.quantity}</td>
                    <td class="px-4 py-3 text-surface-200">₹{p.entry_price:.2f}</td>
                    <td class="px-4 py-3 text-surface-200">₹{p.exit_price:.2f}</td>
                    <td class="px-4 py-3 {pnl_class}">{pnl_sign}₹{abs(p.pnl or 0):,.0f}</td>
                    <td class="px-4 py-3"><span class="{result_badge}">{result_text}</span></td>
                </tr>
            """

        return HTMLResponse(content=html)

    except Exception as e:
        logger.error(f"Paper closed positions error: {e}")
        return HTMLResponse(content="""
            <tr>
                <td colspan="7" class="text-center py-6 text-surface-500">Error loading positions</td>
            </tr>
        """)


@router.get("/quick-trade-signals", response_class=HTMLResponse)
async def quick_trade_signals_partial(request: Request):
    """Render quick trade signals for paper trading."""
    try:
        auth = get_auth_service()
        if not auth.is_authenticated:
            return HTMLResponse(content="""
                <p class="text-sm text-surface-500">Login to view signals</p>
            """)

        fetcher = get_data_fetcher()
        quality_analyzer = get_quality_analyzer()
        signals_html = ""

        for idx, token in [
            ("NIFTY", NIFTY_INDEX_TOKEN),
            ("BANKNIFTY", BANKNIFTY_INDEX_TOKEN),
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
                    quality = quality_analyzer.analyze_quality(signal)
                    opt = signal.recommended_option

                    type_class = "bullish" if signal.direction in ["CE", "STRONG_CE"] else "bearish"
                    score_class = "bullish" if quality.total_score >= 70 else "amber" if quality.total_score >= 50 else "surface"

                    signals_html += f"""
                        <button
                            onclick="populateFromSignal('{idx}', '{signal.direction}', {opt.strike}, {opt.entry_price:.2f}, {opt.target_price:.2f}, {opt.stop_loss:.2f})"
                            class="w-full p-3 bg-surface-800 hover:bg-surface-700 rounded-lg border border-surface-700 transition-colors text-left"
                        >
                            <div class="flex items-center justify-between mb-1">
                                <span class="font-medium text-white">{idx}</span>
                                <span class="text-{type_class}-400 font-semibold">{signal.direction}</span>
                            </div>
                            <div class="flex items-center justify-between text-xs">
                                <span class="text-surface-400">₹{opt.entry_price:.1f} → ₹{opt.target_price:.1f}</span>
                                <span class="text-{score_class}-400">Score: {quality.total_score:.0f}</span>
                            </div>
                        </button>
                    """

            except Exception as e:
                logger.debug(f"Quick trade signal error for {idx}: {e}")
                continue

        if not signals_html:
            return HTMLResponse(content="""
                <p class="text-sm text-surface-500 text-center py-4">No active signals available</p>
            """)

        return HTMLResponse(content=signals_html)

    except Exception as e:
        logger.error(f"Quick trade signals error: {e}")
        return HTMLResponse(content="""
            <p class="text-sm text-surface-500">Error loading signals</p>
        """)


@router.get("/pivot-levels/{index}", response_class=HTMLResponse)
async def pivot_levels_partial(request: Request, index: str = "NIFTY"):
    """Render pivot levels panel."""
    try:
        require_auth()
        fetcher = get_data_fetcher()
        token = get_instrument_token(index.upper())

        df = await fetcher.fetch_historical_data(instrument_token=token, timeframe="day", days=5)

        if df.empty:
            return HTMLResponse(content="""
                <div class="text-center py-6 text-surface-500">
                    <p>No data available</p>
                </div>
            """)

        # Calculate pivot points from previous day
        prev_high = df["High"].iloc[-2]
        prev_low = df["Low"].iloc[-2]
        prev_close = df["Close"].iloc[-2]

        pivot = (prev_high + prev_low + prev_close) / 3
        r1 = (2 * pivot) - prev_low
        r2 = pivot + (prev_high - prev_low)
        r3 = prev_high + 2 * (pivot - prev_low)
        s1 = (2 * pivot) - prev_high
        s2 = pivot - (prev_high - prev_low)
        s3 = prev_low - 2 * (prev_high - pivot)

        current_price = df["Close"].iloc[-1]

        html = f"""
            <div class="space-y-3">
                <div class="flex justify-between items-center p-2 bg-bearish-500/10 rounded">
                    <span class="text-sm text-bearish-400">R3</span>
                    <span class="font-mono text-bearish-400">{r3:,.2f}</span>
                </div>
                <div class="flex justify-between items-center p-2 bg-bearish-500/10 rounded">
                    <span class="text-sm text-bearish-400">R2</span>
                    <span class="font-mono text-bearish-400">{r2:,.2f}</span>
                </div>
                <div class="flex justify-between items-center p-2 bg-bearish-500/10 rounded">
                    <span class="text-sm text-bearish-400">R1</span>
                    <span class="font-mono text-bearish-400">{r1:,.2f}</span>
                </div>
                <div class="flex justify-between items-center p-3 bg-primary-500/20 rounded-lg border border-primary-500/30">
                    <span class="text-sm font-medium text-primary-400">Pivot</span>
                    <span class="font-mono font-bold text-primary-400">{pivot:,.2f}</span>
                </div>
                <div class="flex justify-between items-center p-2 bg-bullish-500/10 rounded">
                    <span class="text-sm text-bullish-400">S1</span>
                    <span class="font-mono text-bullish-400">{s1:,.2f}</span>
                </div>
                <div class="flex justify-between items-center p-2 bg-bullish-500/10 rounded">
                    <span class="text-sm text-bullish-400">S2</span>
                    <span class="font-mono text-bullish-400">{s2:,.2f}</span>
                </div>
                <div class="flex justify-between items-center p-2 bg-bullish-500/10 rounded">
                    <span class="text-sm text-bullish-400">S3</span>
                    <span class="font-mono text-bullish-400">{s3:,.2f}</span>
                </div>
                <div class="mt-4 p-3 bg-surface-800/50 rounded-lg">
                    <div class="flex justify-between items-center">
                        <span class="text-sm text-surface-400">Current Price</span>
                        <span class="font-mono font-bold text-white">{current_price:,.2f}</span>
                    </div>
                </div>
            </div>
        """
        return HTMLResponse(content=html)

    except Exception as e:
        logger.error(f"Pivot levels error: {e}")
        return HTMLResponse(content="""
            <div class="text-center py-6 text-surface-500">
                <p>Error loading pivot levels</p>
            </div>
        """)


@router.get("/greeks-panel/{index}", response_class=HTMLResponse)
async def greeks_panel_partial(request: Request, index: str = "NIFTY"):
    """Render options greeks panel."""
    try:
        require_auth()
        fetcher = get_data_fetcher()

        chain_data = await fetcher.get_option_chain(index=index.upper(), strike_count=3)

        if "error" in chain_data:
            return HTMLResponse(content="""
                <div class="text-center py-6 text-surface-500">
                    <p>Login to view Greeks</p>
                </div>
            """)

        chain = chain_data.get("chain", [])
        atm_strike = chain_data.get("atm_strike", 0)

        # Find ATM option
        atm_option = None
        for opt in chain:
            if opt.get("strike") == atm_strike:
                atm_option = opt
                break

        if not atm_option:
            return HTMLResponse(content="""
                <div class="text-center py-6 text-surface-500">
                    <p>No ATM option data</p>
                </div>
            """)

        ce_delta = atm_option.get("ce_delta", 0.5)
        ce_gamma = atm_option.get("ce_gamma", 0.01)
        ce_theta = atm_option.get("ce_theta", -5)
        ce_vega = atm_option.get("ce_vega", 0.1)
        ce_iv = atm_option.get("ce_iv", 15)

        pe_delta = atm_option.get("pe_delta", -0.5)
        pe_gamma = atm_option.get("pe_gamma", 0.01)
        pe_theta = atm_option.get("pe_theta", -5)
        pe_vega = atm_option.get("pe_vega", 0.1)
        pe_iv = atm_option.get("pe_iv", 15)

        html = f"""
            <div class="space-y-4">
                <div class="grid grid-cols-2 gap-4">
                    <div class="p-3 bg-bullish-500/10 rounded-lg border border-bullish-500/30">
                        <h4 class="text-xs font-semibold text-bullish-400 mb-2">CE Greeks</h4>
                        <div class="space-y-2 text-sm">
                            <div class="flex justify-between">
                                <span class="text-surface-400">Delta</span>
                                <span class="text-bullish-400">{ce_delta:.3f}</span>
                            </div>
                            <div class="flex justify-between">
                                <span class="text-surface-400">Gamma</span>
                                <span class="text-surface-300">{ce_gamma:.4f}</span>
                            </div>
                            <div class="flex justify-between">
                                <span class="text-surface-400">Theta</span>
                                <span class="text-bearish-400">{ce_theta:.2f}</span>
                            </div>
                            <div class="flex justify-between">
                                <span class="text-surface-400">Vega</span>
                                <span class="text-surface-300">{ce_vega:.3f}</span>
                            </div>
                            <div class="flex justify-between">
                                <span class="text-surface-400">IV</span>
                                <span class="text-amber-400">{ce_iv:.1f}%</span>
                            </div>
                        </div>
                    </div>
                    <div class="p-3 bg-bearish-500/10 rounded-lg border border-bearish-500/30">
                        <h4 class="text-xs font-semibold text-bearish-400 mb-2">PE Greeks</h4>
                        <div class="space-y-2 text-sm">
                            <div class="flex justify-between">
                                <span class="text-surface-400">Delta</span>
                                <span class="text-bearish-400">{pe_delta:.3f}</span>
                            </div>
                            <div class="flex justify-between">
                                <span class="text-surface-400">Gamma</span>
                                <span class="text-surface-300">{pe_gamma:.4f}</span>
                            </div>
                            <div class="flex justify-between">
                                <span class="text-surface-400">Theta</span>
                                <span class="text-bearish-400">{pe_theta:.2f}</span>
                            </div>
                            <div class="flex justify-between">
                                <span class="text-surface-400">Vega</span>
                                <span class="text-surface-300">{pe_vega:.3f}</span>
                            </div>
                            <div class="flex justify-between">
                                <span class="text-surface-400">IV</span>
                                <span class="text-amber-400">{pe_iv:.1f}%</span>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="p-3 bg-surface-800/50 rounded-lg">
                    <div class="flex justify-between items-center">
                        <span class="text-sm text-surface-400">ATM Strike</span>
                        <span class="font-mono font-bold text-white">{atm_strike:,.0f}</span>
                    </div>
                </div>
            </div>
        """
        return HTMLResponse(content=html)

    except Exception as e:
        logger.error(f"Greeks panel error: {e}")
        return HTMLResponse(content="""
            <div class="text-center py-6 text-surface-500">
                <p>Error loading Greeks</p>
            </div>
        """)


@router.get("/atm-options/{index}", response_class=HTMLResponse)
async def atm_options_partial(request: Request, index: str = "NIFTY"):
    """Render ATM options list."""
    try:
        require_auth()
        fetcher = get_data_fetcher()

        chain_data = await fetcher.get_option_chain(index=index.upper(), strike_count=5)

        if "error" in chain_data:
            return HTMLResponse(content="""
                <div class="p-4 text-center text-surface-500">
                    <p>Login to view options</p>
                </div>
            """)

        chain = chain_data.get("chain", [])
        atm_strike = chain_data.get("atm_strike", 0)

        # Get strikes near ATM
        atm_options = sorted(
            [opt for opt in chain if abs(opt.get("strike", 0) - atm_strike) <= 300],
            key=lambda x: x.get("strike", 0)
        )[:5]

        if not atm_options:
            return HTMLResponse(content="""
                <div class="p-4 text-center text-surface-500">
                    <p>No options data</p>
                </div>
            """)

        html = ""
        for opt in atm_options:
            strike = opt.get("strike", 0)
            ce_ltp = opt.get("ce_ltp", 0)
            pe_ltp = opt.get("pe_ltp", 0)
            ce_oi = opt.get("ce_oi", 0)
            pe_oi = opt.get("pe_oi", 0)
            is_atm = strike == atm_strike

            atm_badge = '<span class="text-2xs bg-primary-500/20 text-primary-400 px-1 rounded ml-1">ATM</span>' if is_atm else ""
            row_bg = "bg-primary-500/5" if is_atm else ""

            html += f"""
                <div class="flex items-center justify-between p-3 {row_bg} hover:bg-surface-800/50">
                    <div class="text-left">
                        <p class="text-sm font-medium text-bullish-400">₹{ce_ltp:.1f}</p>
                        <p class="text-2xs text-surface-500">{ce_oi:,} OI</p>
                    </div>
                    <div class="text-center">
                        <p class="text-sm font-semibold text-white">{strike:,.0f}{atm_badge}</p>
                    </div>
                    <div class="text-right">
                        <p class="text-sm font-medium text-bearish-400">₹{pe_ltp:.1f}</p>
                        <p class="text-2xs text-surface-500">{pe_oi:,} OI</p>
                    </div>
                </div>
            """

        return HTMLResponse(content=html)

    except Exception as e:
        logger.error(f"ATM options error: {e}")
        return HTMLResponse(content="""
            <div class="p-4 text-center text-surface-500">
                <p>Error loading options</p>
            </div>
        """)


@router.get("/signals-history", response_class=HTMLResponse)
async def signals_history_partial(
    request: Request,
    index: str = "",
    type: str = "",
    quality: str = "",
    status: str = "",
    period: str = "month",
    sort: str = "timestamp",
    dir: str = "desc",
    page: int = 1,
    size: int = 25,
):
    """Render signals history table rows with filtering and pagination."""
    try:
        from app.services.signal_history_service import get_history_service

        history_service = get_history_service()

        # Determine days based on period
        period_days = {
            "today": 1,
            "week": 7,
            "month": 30,
            "all": 365,
        }
        days = period_days.get(period, 30)

        # Get signals from history service
        signals_raw = history_service.get_signals(days=days)

        # Apply filters
        if index:
            signals_raw = [s for s in signals_raw if s.get("index", "").upper() == index.upper()]

        if type:
            if type == "CE":
                signals_raw = [s for s in signals_raw if s.get("signal_type", "") in ("CE", "STRONG_CE", "ce")]
            elif type == "PE":
                signals_raw = [s for s in signals_raw if s.get("signal_type", "") in ("PE", "STRONG_PE", "pe")]

        if quality:
            if quality == "high":
                signals_raw = [s for s in signals_raw if s.get("quality_score", 0) >= 70]
            elif quality == "medium":
                signals_raw = [s for s in signals_raw if 50 <= s.get("quality_score", 0) < 70]
            elif quality == "low":
                signals_raw = [s for s in signals_raw if s.get("quality_score", 0) < 50]

        if status:
            signals_raw = [s for s in signals_raw if s.get("outcome", "active") == status]

        # Sort
        reverse = dir == "desc"
        if sort == "timestamp":
            signals_raw = sorted(signals_raw, key=lambda x: x.get("recorded_at", datetime.min), reverse=reverse)
        elif sort == "score":
            signals_raw = sorted(signals_raw, key=lambda x: x.get("quality_score", 0), reverse=reverse)

        # Pagination
        total = len(signals_raw)
        total_pages = max(1, (total + size - 1) // size)
        start = (page - 1) * size
        end = start + size
        signals_page = signals_raw[start:end]

        # Format for template
        signals = []
        for s in signals_page:
            recorded_at = s.get("recorded_at")
            timestamp = recorded_at.strftime("%Y-%m-%d %H:%M") if recorded_at else "--"

            signals.append({
                "timestamp": timestamp,
                "index": s.get("index", "--"),
                "type": s.get("signal_type", "--"),
                "strike": s.get("strike", "--"),
                "entry": s.get("entry_price", 0),
                "target": s.get("target_price", 0),
                "stoploss": s.get("stop_loss", 0),
                "quality_score": s.get("quality_score", 0),
                "risk_reward": s.get("risk_reward", "--"),
                "status": s.get("outcome", "active"),
                "pnl": s.get("pnl"),
            })

        # Update result count element via HX-Trigger header
        response = templates.TemplateResponse(
            "partials/signals_history_table.html",
            {"request": request, "signals": signals},
        )
        response.headers["HX-Trigger"] = f'{{"updateResultCount": "{total} signals found"}}'

        return response

    except Exception as e:
        logger.error(f"Signals history error: {e}")
        return templates.TemplateResponse(
            "partials/signals_history_table.html",
            {"request": request, "signals": []},
        )
