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
        {
            "request": request,
            "authenticated": status.get("is_authenticated", False),
            "user_id": status.get("user_id"),
            "user_name": status.get("user_name"),
        },
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


@router.get("/market-header-cards", response_class=HTMLResponse)
async def market_header_cards_partial(request: Request):
    """Render market header cards with OHLC data for NIFTY, BANKNIFTY, and SENSEX."""
    try:
        require_auth()
        fetcher = get_data_fetcher()

        # Fetch OHLC data for all indices
        ohlc_data = await fetcher.fetch_ohlc(["NSE:NIFTY 50", "NSE:NIFTY BANK", "BSE:SENSEX"])

        # Also fetch quotes for change percentage
        quotes = await fetcher.fetch_quote(["NSE:NIFTY 50", "NSE:NIFTY BANK", "BSE:SENSEX"])

        # Merge OHLC with quote data (change percentage)
        nifty = {**ohlc_data.get("NSE:NIFTY 50", {}), **quotes.get("NSE:NIFTY 50", {})}
        banknifty = {**ohlc_data.get("NSE:NIFTY BANK", {}), **quotes.get("NSE:NIFTY BANK", {})}
        sensex = {**ohlc_data.get("BSE:SENSEX", {}), **quotes.get("BSE:SENSEX", {})}

        return templates.TemplateResponse(
            "partials/market_header_cards.html",
            {
                "request": request,
                "nifty": nifty,
                "banknifty": banknifty,
                "sensex": sensex,
            },
        )

    except Exception as e:
        logger.error(f"Market header cards error: {e}")
        return templates.TemplateResponse(
            "partials/market_header_cards.html",
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


@router.get("/option-chain-stats/{index}", response_class=HTMLResponse)
async def option_chain_stats_partial(
    request: Request,
    index: str = "NIFTY",
):
    """Render option chain statistics panel."""
    index = index.upper()

    try:
        require_auth()
        fetcher = get_data_fetcher()

        chain_data = await fetcher.get_option_chain(index=index, strike_count=7)

        if "error" in chain_data:
            return templates.TemplateResponse(
                "partials/option_chain_stats.html",
                {"request": request, "error": chain_data["error"], "index": index},
            )

        # Calculate additional stats
        from app.indicators.options import calculate_pcr

        pcr_result = calculate_pcr(
            total_put_oi=chain_data.get("total_pe_oi", 0),
            total_call_oi=chain_data.get("total_ce_oi", 0),
        )

        # Find max OI strikes
        chain = chain_data.get("chain", [])
        max_ce_oi = 0
        max_pe_oi = 0
        max_ce_strike = 0
        max_pe_strike = 0
        total_volume = 0

        for row in chain:
            ce_oi = row.get("ce", {}).get("oi", 0) or 0
            pe_oi = row.get("pe", {}).get("oi", 0) or 0
            ce_vol = row.get("ce", {}).get("volume", 0) or 0
            pe_vol = row.get("pe", {}).get("volume", 0) or 0
            total_volume += ce_vol + pe_vol

            if ce_oi > max_ce_oi:
                max_ce_oi = ce_oi
                max_ce_strike = row["strike"]
            if pe_oi > max_pe_oi:
                max_pe_oi = pe_oi
                max_pe_strike = row["strike"]

        stats = {
            "spot_price": chain_data.get("spot_price", 0),
            "atm_strike": chain_data.get("atm_strike", 0),
            "pcr": chain_data.get("pcr", 0),
            "pcr_sentiment": pcr_result.sentiment,
            "pcr_signal": pcr_result.signal,
            "total_ce_oi": chain_data.get("total_ce_oi", 0),
            "total_pe_oi": chain_data.get("total_pe_oi", 0),
            "total_volume": total_volume,
            "max_ce_strike": max_ce_strike,
            "max_pe_strike": max_pe_strike,
            "max_ce_oi": max_ce_oi,
            "max_pe_oi": max_pe_oi,
            "expiry": chain_data.get("expiry", ""),
            "index": index,
        }

        return templates.TemplateResponse(
            "partials/option_chain_stats.html",
            {"request": request, "stats": stats, "index": index},
        )

    except Exception as e:
        logger.error(f"Option chain stats error: {e}")
        return templates.TemplateResponse(
            "partials/option_chain_stats.html",
            {"request": request, "error": f"Error: {str(e)[:50]}", "index": index},
        )


@router.get("/option-chain-full/{index}", response_class=HTMLResponse)
async def option_chain_full_partial(
    request: Request,
    index: str = "NIFTY",
):
    """Render full option chain table with 15 strikes around ATM (7 above + ATM + 7 below)."""
    index = index.upper()

    try:
        require_auth()
        fetcher = get_data_fetcher()

        chain_data = await fetcher.get_option_chain(index=index, strike_count=7)

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
        logger.error(f"Option chain full error: {e}")
        return templates.TemplateResponse(
            "partials/option_chain_table.html",
            {"request": request, "error": f"Error: {str(e)[:50]}", "index": index},
        )


@router.get("/max-pain/{index}", response_class=HTMLResponse)
async def max_pain_partial(
    request: Request,
    index: str = "NIFTY",
):
    """Render Max Pain analysis panel."""
    index = index.upper()

    try:
        require_auth()
        fetcher = get_data_fetcher()

        chain_data = await fetcher.get_option_chain(index=index, strike_count=20)

        if "error" in chain_data:
            return templates.TemplateResponse(
                "partials/max_pain.html",
                {"request": request, "error": chain_data["error"], "index": index},
            )

        from app.indicators.options import calculate_max_pain

        max_pain = calculate_max_pain(
            option_chain=chain_data.get("chain", []),
            spot_price=chain_data.get("spot_price", 0),
        )

        return templates.TemplateResponse(
            "partials/max_pain.html",
            {
                "request": request,
                "max_pain": max_pain,
                "spot_price": chain_data.get("spot_price", 0),
                "index": index,
            },
        )

    except Exception as e:
        logger.error(f"Max pain error: {e}")
        return templates.TemplateResponse(
            "partials/max_pain.html",
            {"request": request, "error": f"Error: {str(e)[:50]}", "index": index},
        )


@router.get("/oi-analysis/{index}", response_class=HTMLResponse)
async def oi_analysis_partial(
    request: Request,
    index: str = "NIFTY",
):
    """Render OI Analysis panel."""
    index = index.upper()

    try:
        require_auth()
        fetcher = get_data_fetcher()

        # Get option chain
        chain_data = await fetcher.get_option_chain(index=index, strike_count=15)

        if "error" in chain_data:
            return templates.TemplateResponse(
                "partials/oi_analysis.html",
                {"request": request, "error": chain_data["error"], "index": index},
            )

        # Get historical data for price change
        if index == "NIFTY":
            token = NIFTY_INDEX_TOKEN
        elif index == "SENSEX":
            token = SENSEX_INDEX_TOKEN
        else:
            token = BANKNIFTY_INDEX_TOKEN

        df = await fetcher.fetch_historical_data(
            instrument_token=token,
            timeframe="day",
            days=5,
        )

        price_change = 0
        if not df.empty and len(df) > 1:
            price_change = ((df["Close"].iloc[-1] - df["Close"].iloc[-2]) / df["Close"].iloc[-2]) * 100

        from app.indicators.options import analyze_oi_change, calculate_gex

        # For OI change, use a simplified estimate
        oi_change = 2.5  # Placeholder - would need previous day OI data

        oi_analysis = analyze_oi_change(
            price_change=price_change,
            oi_change=oi_change,
            option_chain=chain_data.get("chain", []),
        )

        # Calculate GEX
        gex = calculate_gex(
            option_chain=chain_data.get("chain", []),
            spot_price=chain_data.get("spot_price", 0),
        )

        return templates.TemplateResponse(
            "partials/oi_analysis.html",
            {
                "request": request,
                "oi_analysis": oi_analysis,
                "gex": gex,
                "price_change": price_change,
                "spot_price": chain_data.get("spot_price", 0),
                "index": index,
            },
        )

    except Exception as e:
        logger.error(f"OI analysis error: {e}")
        return templates.TemplateResponse(
            "partials/oi_analysis.html",
            {"request": request, "error": f"Error: {str(e)[:50]}", "index": index},
        )


@router.get("/trend-indicators/{index}", response_class=HTMLResponse)
async def trend_indicators_partial(
    request: Request,
    index: str = "NIFTY",
):
    """Render detailed trend indicators."""
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
            days=3,
        )

        if df.empty:
            return templates.TemplateResponse(
                "partials/trend_indicators.html",
                {"request": request, "error": "No data available", "index": index},
            )

        from app.indicators.trend import calculate_supertrend, calculate_ema, calculate_adx, calculate_vwap

        indicators = {
            "supertrend": calculate_supertrend(df),
            "ema": calculate_ema(df),
            "adx": calculate_adx(df),
            "vwap": calculate_vwap(df) if 'Volume' in df.columns else None,
        }

        return templates.TemplateResponse(
            "partials/trend_indicators.html",
            {
                "request": request,
                "indicators": indicators,
                "price": df["Close"].iloc[-1],
                "index": index,
            },
        )

    except Exception as e:
        logger.error(f"Trend indicators error: {e}")
        return templates.TemplateResponse(
            "partials/trend_indicators.html",
            {"request": request, "error": f"Error: {str(e)[:50]}", "index": index},
        )


@router.get("/momentum-indicators/{index}", response_class=HTMLResponse)
async def momentum_indicators_partial(
    request: Request,
    index: str = "NIFTY",
):
    """Render detailed momentum indicators."""
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
            days=3,
        )

        if df.empty:
            return templates.TemplateResponse(
                "partials/momentum_indicators.html",
                {"request": request, "error": "No data available", "index": index},
            )

        from app.indicators.momentum import calculate_rsi, calculate_macd, calculate_stochastic

        indicators = {
            "rsi": calculate_rsi(df),
            "macd": calculate_macd(df),
            "stochastic": calculate_stochastic(df),
        }

        return templates.TemplateResponse(
            "partials/momentum_indicators.html",
            {
                "request": request,
                "indicators": indicators,
                "price": df["Close"].iloc[-1],
                "index": index,
            },
        )

    except Exception as e:
        logger.error(f"Momentum indicators error: {e}")
        return templates.TemplateResponse(
            "partials/momentum_indicators.html",
            {"request": request, "error": f"Error: {str(e)[:50]}", "index": index},
        )


@router.get("/volatility-indicators/{index}", response_class=HTMLResponse)
async def volatility_indicators_partial(
    request: Request,
    index: str = "NIFTY",
):
    """Render detailed volatility indicators."""
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
            days=3,
        )

        if df.empty:
            return templates.TemplateResponse(
                "partials/volatility_indicators.html",
                {"request": request, "error": "No data available", "index": index},
            )

        from app.indicators.volatility import calculate_atr, calculate_bollinger_bands

        indicators = {
            "atr": calculate_atr(df),
            "bollinger": calculate_bollinger_bands(df),
        }

        return templates.TemplateResponse(
            "partials/volatility_indicators.html",
            {
                "request": request,
                "indicators": indicators,
                "price": df["Close"].iloc[-1],
                "index": index,
            },
        )

    except Exception as e:
        logger.error(f"Volatility indicators error: {e}")
        return templates.TemplateResponse(
            "partials/volatility_indicators.html",
            {"request": request, "error": f"Error: {str(e)[:50]}", "index": index},
        )


@router.get("/pivot-indicators/{index}", response_class=HTMLResponse)
async def pivot_indicators_partial(
    request: Request,
    index: str = "NIFTY",
):
    """Render pivot point indicators."""
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
            timeframe="day",
            days=5,
        )

        if df.empty:
            return templates.TemplateResponse(
                "partials/pivot_indicators.html",
                {"request": request, "error": "No data available", "index": index},
            )

        from app.indicators.pivots import calculate_standard_pivots, calculate_camarilla_pivots

        # Use previous day's data for pivot calculation
        if len(df) >= 2:
            prev_high = df["High"].iloc[-2]
            prev_low = df["Low"].iloc[-2]
            prev_close = df["Close"].iloc[-2]
        else:
            prev_high = df["High"].iloc[-1]
            prev_low = df["Low"].iloc[-1]
            prev_close = df["Close"].iloc[-1]

        standard = calculate_standard_pivots(prev_high, prev_low, prev_close)
        camarilla = calculate_camarilla_pivots(prev_high, prev_low, prev_close)

        return templates.TemplateResponse(
            "partials/pivot_indicators.html",
            {
                "request": request,
                "standard": standard,
                "camarilla": camarilla,
                "price": df["Close"].iloc[-1],
                "index": index,
            },
        )

    except Exception as e:
        logger.error(f"Pivot indicators error: {e}")
        return templates.TemplateResponse(
            "partials/pivot_indicators.html",
            {"request": request, "error": f"Error: {str(e)[:50]}", "index": index},
        )


@router.get("/signal-card-compact/{index}", response_class=HTMLResponse)
async def signal_card_compact_partial(
    request: Request,
    index: str = "NIFTY",
    style: str = "intraday",
):
    """Render compact signal card for dashboard."""
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
                "partials/signal_card_compact.html",
                {"request": request, "error": "Market closed", "index": index},
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
            "partials/signal_card_compact.html",
            {
                "request": request,
                "signal": signal,
                "index": index,
                "style": style,
            },
        )

    except HTTPException:
        return templates.TemplateResponse(
            "partials/signal_card_compact.html",
            {"request": request, "error": "Please login first", "index": index},
        )
    except Exception as e:
        logger.error(f"Signal card compact error: {e}")
        return templates.TemplateResponse(
            "partials/signal_card_compact.html",
            {"request": request, "error": f"Error: {str(e)[:30]}", "index": index},
        )


@router.get("/indicator-panel-compact/{index}", response_class=HTMLResponse)
async def indicator_panel_compact_partial(
    request: Request,
    index: str = "NIFTY",
):
    """Render compact indicator panel for dashboard."""
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
                "partials/indicator_panel_compact.html",
                {"request": request, "error": "Market closed", "index": index},
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
            "partials/indicator_panel_compact.html",
            {
                "request": request,
                "indicators": indicators,
                "index": index,
                "price": df["Close"].iloc[-1],
            },
        )

    except Exception as e:
        logger.error(f"Indicator panel compact error: {e}")
        return templates.TemplateResponse(
            "partials/indicator_panel_compact.html",
            {"request": request, "error": f"Error: {str(e)[:30]}", "index": index},
        )


@router.get("/recommended-option-compact/{index}", response_class=HTMLResponse)
async def recommended_option_compact_partial(
    request: Request,
    index: str = "NIFTY",
    style: str = "intraday",
):
    """Render compact recommended option card for dashboard."""
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
                "partials/recommended_option_compact.html",
                {"request": request, "error": "Market closed"},
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

        if signal and not signal.recommended_option:
            return templates.TemplateResponse(
                "partials/recommended_option_compact.html",
                {"request": request, "error": f"No suitable {index} option found"},
            )

        return templates.TemplateResponse(
            "partials/recommended_option_compact.html",
            {
                "request": request,
                "signal": signal,
                "style": style,
            },
        )

    except HTTPException:
        return templates.TemplateResponse(
            "partials/recommended_option_compact.html",
            {"request": request, "error": "Please login first"},
        )
    except Exception as e:
        logger.error(f"Recommended option compact error: {e}")
        return templates.TemplateResponse(
            "partials/recommended_option_compact.html",
            {"request": request, "error": f"Error: {str(e)[:30]}"},
        )


@router.get("/sidebar-expiries", response_class=HTMLResponse)
async def sidebar_expiries_partial(request: Request):
    """Render sidebar with dynamic expiry days from Kite."""
    try:
        require_auth()
        from app.services.paper_trading import get_paper_trading_service

        paper = get_paper_trading_service()

        # Refresh expiry data from Kite
        await paper.refresh_expiry_cache()

        # Get expiry info for all indices
        expiries = {}
        for index in ["NIFTY", "BANKNIFTY", "SENSEX"]:
            exp = paper.get_next_expiry(index)
            expiries[index] = {
                "expiry_date": exp.expiry_date,
                "days_to_expiry": exp.days_to_expiry,
                "is_expiry_day": exp.is_expiry_day,
                "weekday": exp.expiry_date.strftime("%a"),
            }

        return templates.TemplateResponse(
            "partials/sidebar_expiries.html",
            {"request": request, "expiries": expiries},
        )
    except Exception as e:
        logger.error(f"Sidebar expiries error: {e}")
        # Return fallback with default expiry days
        return templates.TemplateResponse(
            "partials/sidebar_expiries.html",
            {
                "request": request,
                "expiries": {
                    "NIFTY": {"weekday": "Thu", "days_to_expiry": None, "is_expiry_day": False},
                    "BANKNIFTY": {"weekday": "Wed", "days_to_expiry": None, "is_expiry_day": False},
                    "SENSEX": {"weekday": "Fri", "days_to_expiry": None, "is_expiry_day": False},
                },
            },
        )
