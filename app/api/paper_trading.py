"""
Paper Trading API Routes
Endpoints for paper trading management
"""

import io
from datetime import datetime, time

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from loguru import logger
import pandas as pd

from app.services.zerodha_auth import get_auth_service
from app.services.paper_trading import get_paper_trading_service
from app.services.signal_engine import get_signal_engine, TradingStyle
from app.services.data_fetcher import get_data_fetcher
from app.core.config import NIFTY_INDEX_TOKEN, BANKNIFTY_INDEX_TOKEN, SENSEX_INDEX_TOKEN

# Trading time restrictions - Full market hours
MARKET_OPEN_TIME = time(9, 15)   # 9:15 AM market open
MARKET_CLOSE_TIME = time(15, 30)  # 3:30 PM market close
SQUARE_OFF_TIME = time(15, 15)    # 3:15 PM - force square off before close

# Known market holidays for 2024-2026 (NSE)
from datetime import date as date_type
INDIAN_MARKET_HOLIDAYS = {
    date_type(2024, 1, 26), date_type(2024, 3, 8), date_type(2024, 3, 25),
    date_type(2024, 3, 29), date_type(2024, 4, 11), date_type(2024, 4, 14),
    date_type(2024, 4, 17), date_type(2024, 4, 21), date_type(2024, 5, 1),
    date_type(2024, 5, 23), date_type(2024, 6, 17), date_type(2024, 7, 17),
    date_type(2024, 8, 15), date_type(2024, 9, 16), date_type(2024, 10, 2),
    date_type(2024, 10, 12), date_type(2024, 11, 1), date_type(2024, 11, 15),
    date_type(2024, 12, 25),
    date_type(2025, 1, 26), date_type(2025, 2, 26), date_type(2025, 3, 14),
    date_type(2025, 3, 31), date_type(2025, 4, 10), date_type(2025, 4, 14),
    date_type(2025, 4, 18), date_type(2025, 5, 1), date_type(2025, 8, 15),
    date_type(2025, 8, 27), date_type(2025, 10, 2), date_type(2025, 10, 21),
    date_type(2025, 11, 5), date_type(2025, 12, 25),
    # 2026 holidays (tentative - NSE)
    date_type(2026, 1, 26), date_type(2026, 2, 17), date_type(2026, 3, 3),
    date_type(2026, 3, 20), date_type(2026, 4, 3), date_type(2026, 4, 14),
    date_type(2026, 5, 1), date_type(2026, 5, 12), date_type(2026, 6, 5),
    date_type(2026, 7, 6), date_type(2026, 8, 15), date_type(2026, 8, 17),
    date_type(2026, 9, 4), date_type(2026, 10, 2), date_type(2026, 10, 20),
    date_type(2026, 11, 9), date_type(2026, 11, 24), date_type(2026, 12, 25),
}


def is_trading_allowed() -> tuple[bool, str]:
    """
    Check if trading is allowed based on:
    - Weekday (Mon-Fri)
    - Market hours (9:15 AM - 3:30 PM IST)
    - Not a market holiday
    """
    now = datetime.now()
    current_time = now.time()
    current_date = now.date()
    weekday = now.weekday()  # 0=Monday, 6=Sunday

    # Check weekend (Saturday=5, Sunday=6)
    if weekday >= 5:
        day_name = "Saturday" if weekday == 5 else "Sunday"
        return False, f"Market closed: {day_name} - Weekend"

    # Check holidays
    if current_date in INDIAN_MARKET_HOLIDAYS:
        return False, f"Market closed: Holiday on {current_date.strftime('%d %b %Y')}"

    # Check pre-market
    if current_time < MARKET_OPEN_TIME:
        return False, f"Pre-market: Opens at {MARKET_OPEN_TIME.strftime('%I:%M %p')}"

    # Check post-market
    if current_time >= MARKET_CLOSE_TIME:
        return False, f"Post-market: Closed at {MARKET_CLOSE_TIME.strftime('%I:%M %p')}"

    # Check square-off time (no new trades after 3:15 PM)
    if current_time >= SQUARE_OFF_TIME:
        return False, f"Square-off period: No new trades after {SQUARE_OFF_TIME.strftime('%I:%M %p')}"

    return True, "Trading allowed"

router = APIRouter(prefix="/paper", tags=["Paper Trading"])
templates = Jinja2Templates(directory="app/templates")


def require_auth():
    """Check if user is authenticated."""
    auth = get_auth_service()
    if not auth.is_authenticated:
        raise HTTPException(status_code=401, detail="Not authenticated")


@router.get("/", response_class=HTMLResponse)
async def paper_trading_page(request: Request):
    """Render paper trading dashboard."""
    auth = get_auth_service()

    if not auth.is_authenticated:
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/")

    paper = get_paper_trading_service()

    # Fetch fresh expiry data from Kite
    await paper.refresh_expiry_cache()
    trading_index = paper.get_trading_index()

    return templates.TemplateResponse(
        "paper_trading.html",
        {
            "request": request,
            "user": auth.user_profile,
            "trading_index": trading_index,
            "stats": paper.get_stats(),
        },
    )


@router.get("/stats")
async def get_stats():
    """Get paper trading statistics."""
    require_auth()
    paper = get_paper_trading_service()
    return paper.get_stats()


@router.get("/trading-index")
async def get_trading_index():
    """Get the current trading index based on nearest expiry."""
    require_auth()
    paper = get_paper_trading_service()
    trading_index = paper.get_trading_index()

    return {
        "index": trading_index.index,
        "expiry_date": trading_index.expiry_date.isoformat(),
        "days_to_expiry": trading_index.days_to_expiry,
        "is_expiry_day": trading_index.is_expiry_day,
        "lot_size": trading_index.lot_size,
        "max_lots_per_order": trading_index.max_lots_per_order,
    }


@router.get("/expiries")
async def get_all_expiries():
    """Get expiry info for all indices."""
    require_auth()
    paper = get_paper_trading_service()

    expiries = []
    for index in ["NIFTY", "SENSEX", "BANKNIFTY"]:
        exp = paper.get_next_expiry(index)
        expiries.append({
            "index": exp.index,
            "expiry_date": exp.expiry_date.isoformat(),
            "days_to_expiry": exp.days_to_expiry,
            "is_expiry_day": exp.is_expiry_day,
            "lot_size": exp.lot_size,
        })

    return {"expiries": expiries}


@router.post("/execute-signal")
async def execute_signal_trade():
    """Execute trade based on current signal."""
    logger.info("Execute signal endpoint called")
    require_auth()

    # Check trading time restrictions
    # Set BYPASS_MARKET_HOURS=true in .env to skip this check for testing
    from app.core.config import get_settings
    settings = get_settings()
    bypass_hours = getattr(settings, 'bypass_market_hours', False)

    allowed, reason = is_trading_allowed()
    logger.info(f"Trading allowed check: {allowed}, reason: {reason}, bypass: {bypass_hours}")

    if not allowed and not bypass_hours:
        logger.warning(f"Trade rejected - Market check failed: {reason}")
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": reason},
        )
    elif not allowed and bypass_hours:
        logger.warning(f"Market check would fail ({reason}), but bypassing for testing")

    paper = get_paper_trading_service()
    fetcher = get_data_fetcher()

    # Get trading index
    trading_index = paper.get_trading_index()

    # Get token for the index
    tokens = {
        "NIFTY": NIFTY_INDEX_TOKEN,
        "BANKNIFTY": BANKNIFTY_INDEX_TOKEN,
        "SENSEX": SENSEX_INDEX_TOKEN,
    }
    token = tokens.get(trading_index.index, NIFTY_INDEX_TOKEN)

    # Fetch historical data
    df = await fetcher.fetch_historical_data(
        instrument_token=token,
        timeframe="5minute",
        days=3,
    )

    if df.empty:
        logger.warning("Trade rejected - No historical data available (market may be closed)")
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "Market closed - No data available"},
        )

    # Get option chain
    chain_data = await fetcher.get_option_chain(index=trading_index.index)
    option_chain = chain_data.get("chain", []) if "error" not in chain_data else None
    logger.info(f"Option chain fetched: {len(option_chain) if option_chain else 0} strikes")

    # Generate signal
    engine = get_signal_engine(TradingStyle.INTRADAY)
    signal = engine.analyze(df=df, option_chain=option_chain)

    if not signal:
        logger.warning("Trade rejected - No signal generated from indicators")
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "No signal generated"},
        )

    logger.info(f"Signal generated: {signal.direction} | Confidence: {signal.confidence:.0f}%")

    # Execute trade
    order = await paper.execute_signal_trade(signal, trading_index)

    if order:
        logger.info(f"Order executed: {order.symbol} | Qty: {order.quantity} | Price: {order.price:.2f}")
        return {
            "success": True,
            "order": {
                "order_id": order.order_id,
                "symbol": order.symbol,
                "lots": order.lots,
                "quantity": order.quantity,
                "price": order.price,
                "split_orders": order.split_orders,
            },
            "signal": {
                "direction": signal.direction,
                "confidence": signal.confidence,
            },
        }
    else:
        halt_reason = paper.daily_stats.halt_reason if paper.daily_stats.is_trading_halted else "Trade not executed"
        logger.warning(f"Trade rejected - {halt_reason}")
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": halt_reason,
            },
        )


@router.get("/positions")
async def get_positions():
    """Get all positions."""
    require_auth()
    paper = get_paper_trading_service()

    open_positions = paper.get_open_positions()
    closed_positions = paper.get_closed_positions()

    return {
        "open": [
            {
                "position_id": p.position_id,
                "index": p.index,
                "symbol": p.symbol,
                "strike": p.strike,
                "option_type": p.option_type,
                "entry_price": p.entry_price,
                "current_price": p.current_price,
                "quantity": p.quantity,
                "lots": p.lots,
                "pnl": p.pnl,
                "pnl_percent": p.pnl_percent,
                "entry_time": p.entry_time.isoformat(),
                "stop_loss": p.stop_loss,
                "target": p.target,
            }
            for p in open_positions
        ],
        "closed": [
            {
                "position_id": p.position_id,
                "index": p.index,
                "symbol": p.symbol,
                "strike": p.strike,
                "option_type": p.option_type,
                "entry_price": p.entry_price,
                "exit_price": p.exit_price,
                "quantity": p.quantity,
                "lots": p.lots,
                "pnl": p.pnl,
                "pnl_percent": p.pnl_percent,
                "entry_time": p.entry_time.isoformat(),
                "exit_time": p.exit_time.isoformat() if p.exit_time else None,
                "exit_reason": p.exit_reason,
            }
            for p in closed_positions[-10:]  # Last 10
        ],
    }


@router.post("/positions/{position_id}/close")
async def close_position(position_id: str):
    """Close a specific position."""
    require_auth()
    paper = get_paper_trading_service()

    # Find position
    position = next(
        (p for p in paper.positions if p.position_id == position_id),
        None,
    )

    if not position:
        raise HTTPException(status_code=404, detail="Position not found")

    # Update position with current price first
    await paper.update_positions()

    # Close position
    order = await paper.close_position(position, "Manual Close")

    if order:
        return {
            "success": True,
            "order_id": order.order_id,
            "exit_price": order.executed_price,
            "pnl": position.pnl,
        }
    else:
        raise HTTPException(status_code=400, detail="Failed to close position")


@router.post("/positions/close-all")
async def close_all_positions():
    """Close all open positions."""
    require_auth()
    paper = get_paper_trading_service()

    # Update positions first
    await paper.update_positions()

    # Close all
    orders = await paper.close_all_positions("Close All")

    return {
        "success": True,
        "closed_count": len(orders),
        "orders": [{"order_id": o.order_id, "symbol": o.symbol} for o in orders],
    }


@router.get("/orders")
async def get_orders():
    """Get today's orders."""
    require_auth()
    paper = get_paper_trading_service()

    orders = paper.get_today_orders()

    return {
        "orders": [
            {
                "order_id": o.order_id,
                "timestamp": o.timestamp.isoformat(),
                "index": o.index,
                "symbol": o.symbol,
                "order_type": o.order_type.value,
                "quantity": o.quantity,
                "lots": o.lots,
                "price": o.price,
                "status": o.status.value,
                "split_orders": o.split_orders,
                "reason": o.reason,
            }
            for o in orders
        ],
    }


@router.post("/update-positions")
async def update_positions():
    """Update all positions with current prices and check exit signals."""
    require_auth()
    paper = get_paper_trading_service()

    updated, closed = await paper.update_positions()

    return {
        "success": True,
        "updated_count": len(updated),
        "closed": closed,  # List of positions closed by exit signals
    }


@router.post("/reset")
async def reset_paper_trading():
    """Reset all paper trading data."""
    require_auth()
    paper = get_paper_trading_service()

    paper.reset_all()

    return {"success": True, "message": "Paper trading reset to initial state"}


@router.post("/reset-daily")
async def reset_daily():
    """Reset daily statistics only."""
    require_auth()
    paper = get_paper_trading_service()

    paper.reset_daily()

    return {"success": True, "message": "Daily statistics reset"}


@router.post("/toggle-auto-trade")
async def toggle_auto_trade(enabled: bool = None):
    """Toggle auto trade on/off."""
    require_auth()
    paper = get_paper_trading_service()

    new_status = paper.toggle_auto_trade(enabled)

    return {
        "success": True,
        "is_auto_trade": new_status,
        "message": f"Auto trade {'enabled' if new_status else 'disabled'}",
    }


@router.get("/auto-trade-status")
async def get_auto_trade_status():
    """Get auto trade status."""
    require_auth()
    paper = get_paper_trading_service()

    return {
        "is_auto_trade": paper.is_auto_trade,
    }


@router.get("/expiries-live")
async def get_expiries_from_kite():
    """Get expiry info for all indices fetched from Kite."""
    require_auth()
    paper = get_paper_trading_service()

    # Refresh cache from Kite
    await paper.refresh_expiry_cache()

    expiries = []
    for index in ["NIFTY", "SENSEX", "BANKNIFTY"]:
        exp = paper.get_next_expiry(index)
        expiries.append({
            "index": exp.index,
            "expiry_date": exp.expiry_date.isoformat(),
            "expiry_weekday": exp.expiry_date.strftime("%a"),
            "days_to_expiry": exp.days_to_expiry,
            "is_expiry_day": exp.is_expiry_day,
            "lot_size": exp.lot_size,
        })

    return {"expiries": expiries, "source": "kite"}


# HTMX Partials

@router.get("/htmx/stats", response_class=HTMLResponse)
async def htmx_stats(request: Request):
    """HTMX partial for stats card."""
    try:
        require_auth()
        paper = get_paper_trading_service()
        stats = paper.get_stats()

        return templates.TemplateResponse(
            "partials/paper_stats.html",
            {"request": request, "stats": stats},
        )
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return templates.TemplateResponse(
            "partials/paper_stats.html",
            {"request": request, "error": str(e)},
        )


@router.get("/htmx/trading-index", response_class=HTMLResponse)
async def htmx_trading_index(request: Request):
    """HTMX partial for trading index display."""
    try:
        require_auth()
        paper = get_paper_trading_service()

        # Refresh expiries from Kite
        await paper.refresh_expiry_cache()

        trading_index = paper.get_trading_index()

        # Get all expiries (now from cache with fresh Kite data)
        expiries = []
        for index in ["NIFTY", "SENSEX", "BANKNIFTY"]:
            exp = paper.get_next_expiry(index)
            expiries.append(exp)

        return templates.TemplateResponse(
            "partials/paper_trading_index.html",
            {
                "request": request,
                "trading_index": trading_index,
                "expiries": expiries,
            },
        )
    except Exception as e:
        logger.error(f"Trading index error: {e}")
        return templates.TemplateResponse(
            "partials/paper_trading_index.html",
            {"request": request, "error": str(e)},
        )


@router.get("/htmx/positions", response_class=HTMLResponse)
async def htmx_positions(request: Request):
    """HTMX partial for positions table."""
    try:
        require_auth()
        paper = get_paper_trading_service()

        # Update positions
        await paper.update_positions()

        open_positions = paper.get_open_positions()
        closed_positions = paper.get_closed_positions()

        return templates.TemplateResponse(
            "partials/paper_positions.html",
            {
                "request": request,
                "open_positions": open_positions,
                "closed_positions": closed_positions[-10:],
            },
        )
    except Exception as e:
        logger.error(f"Positions error: {e}")
        return templates.TemplateResponse(
            "partials/paper_positions.html",
            {"request": request, "error": str(e)},
        )


@router.get("/htmx/orders", response_class=HTMLResponse)
async def htmx_orders(request: Request):
    """HTMX partial for orders table."""
    try:
        require_auth()
        paper = get_paper_trading_service()

        orders = paper.get_today_orders()

        return templates.TemplateResponse(
            "partials/paper_orders.html",
            {"request": request, "orders": orders},
        )
    except Exception as e:
        logger.error(f"Orders error: {e}")
        return templates.TemplateResponse(
            "partials/paper_orders.html",
            {"request": request, "error": str(e)},
        )


@router.get("/order-history", response_class=HTMLResponse)
async def order_history_page(request: Request):
    """Render order history page."""
    auth = get_auth_service()

    if not auth.is_authenticated:
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/")

    paper = get_paper_trading_service()
    history = paper.get_order_history(days=30)
    summary = paper.get_order_history_summary()

    return templates.TemplateResponse(
        "order_history.html",
        {
            "request": request,
            "user": auth.user_profile,
            "history": history,
            "summary": summary,
        },
    )


@router.get("/order-history/data")
async def get_order_history(days: int = 30):
    """Get order history as JSON."""
    require_auth()
    paper = get_paper_trading_service()

    history = paper.get_order_history(days=days)

    return {
        "history": [
            {
                "order_id": h.order_id,
                "position_id": h.position_id,
                "timestamp": h.timestamp.isoformat(),
                "index": h.index,
                "symbol": h.symbol,
                "strike": h.strike,
                "option_type": h.option_type,
                "direction": h.direction,
                "quantity": h.quantity,
                "lots": h.lots,
                "entry_price": h.entry_price,
                "exit_price": h.exit_price,
                "max_price": h.max_price,
                "min_price": h.min_price,
                "pnl": h.pnl,
                "pnl_percent": h.pnl_percent,
                "max_profit_percent": h.max_profit_percent,
                "max_loss_percent": h.max_loss_percent,
                "captured_move_percent": h.captured_move_percent,
                "signal_confidence": h.signal_confidence,
                "exit_reason": h.exit_reason,
                "entry_time": h.entry_time.isoformat() if h.entry_time else None,
                "exit_time": h.exit_time.isoformat() if h.exit_time else None,
                "duration_minutes": h.duration_minutes,
            }
            for h in history
        ],
    }


@router.get("/order-history/summary")
async def get_order_history_summary():
    """Get order history summary statistics."""
    require_auth()
    paper = get_paper_trading_service()

    return paper.get_order_history_summary()


@router.get("/order-history/download")
async def download_order_history(days: int = 30):
    """Download order history as Excel file."""
    require_auth()
    paper = get_paper_trading_service()

    history = paper.get_order_history(days=days)

    if not history:
        raise HTTPException(status_code=404, detail="No order history found")

    # Create DataFrame
    data = []
    for h in history:
        data.append({
            "Date": h.timestamp.strftime("%Y-%m-%d"),
            "Time": h.timestamp.strftime("%H:%M:%S"),
            "Index": h.index,
            "Symbol": h.symbol,
            "Strike": h.strike,
            "Type": h.option_type,
            "Direction": h.direction,
            "Lots": h.lots,
            "Quantity": h.quantity,
            "Entry Price": h.entry_price,
            "Exit Price": h.exit_price,
            "Max Price": h.max_price,
            "Min Price": h.min_price,
            "P&L": h.pnl,
            "P&L %": h.pnl_percent,
            "Max Profit %": h.max_profit_percent,
            "Max Loss %": h.max_loss_percent,
            "Captured Move %": h.captured_move_percent,
            "Duration (min)": h.duration_minutes,
            "Exit Reason": h.exit_reason,
            "Confidence": h.signal_confidence,
        })

    df = pd.DataFrame(data)

    # Create Excel file in memory
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Order History', index=False)

    output.seek(0)

    filename = f"order_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/htmx/order-history", response_class=HTMLResponse)
async def htmx_order_history(request: Request, days: int = 30):
    """HTMX partial for order history table."""
    try:
        require_auth()
        paper = get_paper_trading_service()

        history = paper.get_order_history(days=days)
        summary = paper.get_order_history_summary()

        return templates.TemplateResponse(
            "partials/paper_order_history.html",
            {
                "request": request,
                "history": history,
                "summary": summary,
            },
        )
    except Exception as e:
        logger.error(f"Order history error: {e}")
        return templates.TemplateResponse(
            "partials/paper_order_history.html",
            {"request": request, "error": str(e)},
        )


@router.get("/htmx/signal-panel", response_class=HTMLResponse)
async def htmx_signal_panel(request: Request):
    """HTMX partial for current signal panel."""
    try:
        require_auth()
        paper = get_paper_trading_service()
        fetcher = get_data_fetcher()

        trading_index = paper.get_trading_index()

        # Get token
        tokens = {
            "NIFTY": NIFTY_INDEX_TOKEN,
            "BANKNIFTY": BANKNIFTY_INDEX_TOKEN,
            "SENSEX": SENSEX_INDEX_TOKEN,
        }
        token = tokens.get(trading_index.index, NIFTY_INDEX_TOKEN)

        # Fetch data
        df = await fetcher.fetch_historical_data(
            instrument_token=token,
            timeframe="5minute",
            days=3,
        )

        if df.empty:
            return templates.TemplateResponse(
                "partials/paper_signal_panel.html",
                {"request": request, "error": "Market closed"},
            )

        # Get option chain
        chain_data = await fetcher.get_option_chain(index=trading_index.index)
        option_chain = chain_data.get("chain", []) if "error" not in chain_data else None

        # Generate signal
        engine = get_signal_engine(TradingStyle.INTRADAY)
        signal = engine.analyze(df=df, option_chain=option_chain)

        # Calculate order preview
        order_preview = None
        if signal and signal.recommended_option:
            lots, qty, splits = paper.calculate_order_size(
                price=signal.recommended_option.ltp,
                lot_size=trading_index.lot_size,
            )
            order_preview = {
                "lots": lots,
                "quantity": qty,
                "total_value": qty * signal.recommended_option.ltp,
                "split_count": len(splits),
                "splits": splits,
            }

        return templates.TemplateResponse(
            "partials/paper_signal_panel.html",
            {
                "request": request,
                "signal": signal,
                "trading_index": trading_index,
                "order_preview": order_preview,
                "chain_data": chain_data if "error" not in chain_data else None,
                "stats": paper.get_stats(),
            },
        )
    except Exception as e:
        logger.error(f"Signal panel error: {e}")
        return templates.TemplateResponse(
            "partials/paper_signal_panel.html",
            {"request": request, "error": str(e)},
        )
