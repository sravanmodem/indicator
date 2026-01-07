"""
User Dashboard API Routes
Personal trading dashboard and controls for individual users
"""

from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from typing import Optional
from pydantic import BaseModel

from app.models.user_trading import (
    get_user_trading_service,
    TradingMode, CopyTradingMode
)
from app.services.user_auth import get_user_auth_service

router = APIRouter(prefix="/user", tags=["User Dashboard"])
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))


def get_current_user(request: Request):
    """Get current logged in user."""
    user = getattr(request.state, 'user', None)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


def get_trading_user(request: Request):
    """Get trading user profile for current user."""
    user = get_current_user(request)
    service = get_user_trading_service()

    # Get trading user by username
    trading_user = service.get_user_by_username(user.get('username'))
    if not trading_user:
        # Create trading user if doesn't exist
        success, message, trading_user = service.create_user(
            username=user.get('username'),
            email=user.get('email', ''),
            password='',  # Uses existing auth
            capital=100000,
            created_by='system'
        )

    return trading_user


# ==================== Dashboard ====================

@router.get("/dashboard", response_class=HTMLResponse)
async def user_dashboard(request: Request):
    """User trading dashboard."""
    current_user = get_current_user(request)
    trading_user = get_trading_user(request)
    service = get_user_trading_service()

    stats = service.get_user_dashboard_stats(trading_user.user_id)
    trades = service.get_user_trades(trading_user.user_id, limit=20)

    # Add default values for missing stats
    stats.setdefault('capital_change', 0)
    stats.setdefault('available_margin', trading_user.capital * 0.9)
    stats.setdefault('margin_used_percent', 10)
    stats.setdefault('today_pnl', 0)
    stats.setdefault('today_trades', 0)
    stats.setdefault('open_positions', 0)
    stats.setdefault('unrealized_pnl', 0)
    stats.setdefault('win_rate', 0)
    stats.setdefault('winning_trades', 0)
    stats.setdefault('losing_trades', 0)

    return templates.TemplateResponse(
        "user/dashboard.html",
        {
            "request": request,
            "user": trading_user,
            "stats": stats,
            "trades": [t.to_dict() for t in trades],
            "current_user": current_user
        }
    )


# ==================== Trades ====================

@router.get("/trades", response_class=HTMLResponse)
async def user_trades(request: Request):
    """User trades history page."""
    current_user = get_current_user(request)
    trading_user = get_trading_user(request)
    service = get_user_trading_service()

    trades = service.get_user_trades(trading_user.user_id, limit=200)
    stats = service.get_user_dashboard_stats(trading_user.user_id)

    # Add default values
    stats.setdefault('total_trades', len(trades))
    stats.setdefault('winning_trades', 0)
    stats.setdefault('losing_trades', 0)
    stats.setdefault('win_rate', 0)
    stats.setdefault('total_pnl', 0)

    return templates.TemplateResponse(
        "user/trades.html",
        {
            "request": request,
            "user": trading_user,
            "trades": [t.to_dict() for t in trades],
            "stats": stats,
            "current_user": current_user
        }
    )


@router.get("/trades/export")
async def export_user_trades(
    request: Request,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None
):
    """Export user trades as CSV."""
    trading_user = get_trading_user(request)
    service = get_user_trading_service()

    trades = service.get_user_trades(trading_user.user_id, limit=1000)

    # Build CSV
    import csv
    import io

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        'Date', 'Time', 'Instrument', 'Type', 'Mode', 'Entry Price',
        'Exit Price', 'Quantity', 'P&L', 'Commission', 'Status'
    ])

    for trade in trades:
        t = trade.to_dict()
        writer.writerow([
            t.get('entry_time', '')[:10],
            t.get('entry_time', '')[11:16],
            t.get('instrument', ''),
            t.get('trade_type', ''),
            t.get('mode', ''),
            t.get('entry_price', ''),
            t.get('exit_price', ''),
            t.get('quantity', ''),
            t.get('pnl', ''),
            t.get('commission', ''),
            t.get('status', '')
        ])

    from fastapi.responses import StreamingResponse
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=trades_{trading_user.username}.csv"}
    )


# ==================== Settings ====================

class TradingModeUpdate(BaseModel):
    mode: str


class CopyModeUpdate(BaseModel):
    mode: str


class RiskMultiplierUpdate(BaseModel):
    multiplier: float


@router.post("/settings/trading-mode")
async def set_trading_mode(request: Request, data: TradingModeUpdate):
    """Update user trading mode."""
    trading_user = get_trading_user(request)
    service = get_user_trading_service()

    # Validate mode
    try:
        mode = TradingMode(data.mode)
    except ValueError:
        return JSONResponse({"success": False, "error": "Invalid mode"}, status_code=400)

    # Check if user can use this mode
    if mode == TradingMode.LIVE and not trading_user.restrictions.can_trade_live:
        return JSONResponse({"success": False, "error": "Live trading not enabled"}, status_code=403)
    if mode == TradingMode.PAPER and not trading_user.restrictions.can_use_paper:
        return JSONResponse({"success": False, "error": "Paper trading not enabled"}, status_code=403)

    service.update_user(trading_user.user_id, {"trading_mode": data.mode})
    return {"success": True}


@router.post("/settings/copy-mode")
async def set_copy_mode(request: Request, data: CopyModeUpdate):
    """Update user copy trading mode."""
    trading_user = get_trading_user(request)
    service = get_user_trading_service()

    # Validate mode
    try:
        mode = CopyTradingMode(data.mode)
    except ValueError:
        return JSONResponse({"success": False, "error": "Invalid mode"}, status_code=400)

    service.update_user(trading_user.user_id, {"copy_trading_mode": data.mode})
    return {"success": True}


@router.post("/settings/risk-multiplier")
async def set_risk_multiplier(request: Request, data: RiskMultiplierUpdate):
    """Update user risk multiplier."""
    trading_user = get_trading_user(request)
    service = get_user_trading_service()

    # Validate multiplier
    if data.multiplier < 0.1 or data.multiplier > 5:
        return JSONResponse({"success": False, "error": "Invalid multiplier"}, status_code=400)

    # Update restrictions
    restrictions = trading_user.restrictions
    restrictions.risk_multiplier = data.multiplier

    service.update_user(trading_user.user_id, {
        "restrictions": {
            "max_capital": restrictions.max_capital,
            "max_lots_per_trade": restrictions.max_lots_per_trade,
            "max_daily_trades": restrictions.max_daily_trades,
            "max_daily_loss": restrictions.max_daily_loss,
            "max_daily_loss_percent": restrictions.max_daily_loss_percent,
            "can_trade_live": restrictions.can_trade_live,
            "can_use_paper": restrictions.can_use_paper,
            "risk_multiplier": data.multiplier,
            "trading_start_time": restrictions.trading_start_time,
            "trading_end_time": restrictions.trading_end_time,
        }
    })
    return {"success": True}


# ==================== Zerodha Integration ====================

@router.post("/zerodha/sync")
async def sync_zerodha(request: Request):
    """Sync Zerodha account data."""
    trading_user = get_trading_user(request)

    if not trading_user.zerodha.is_connected:
        return JSONResponse({"success": False, "error": "Zerodha not connected"}, status_code=400)

    # TODO: Implement actual Zerodha sync
    # This would fetch latest positions, orders, and margin from Zerodha

    return {"success": True, "message": "Sync completed"}


@router.post("/zerodha/disconnect")
async def disconnect_zerodha(request: Request):
    """Disconnect Zerodha account."""
    trading_user = get_trading_user(request)
    service = get_user_trading_service()

    # Clear Zerodha credentials
    service.update_user(trading_user.user_id, {
        "zerodha": {
            "api_key": "",
            "api_secret": "",
            "user_id": "",
            "access_token": "",
            "refresh_token": "",
            "is_connected": False,
            "last_sync": None
        }
    })

    return {"success": True}


# ==================== HTMX Endpoints ====================

@router.get("/htmx/active-signals", response_class=HTMLResponse)
async def htmx_active_signals(request: Request):
    """Get active signals for HTMX."""
    trading_user = get_trading_user(request)
    service = get_user_trading_service()

    signals = service.get_active_signals()

    if not signals:
        return HTMLResponse("""
            <div class="p-8 text-center text-slate-500">
                <p>No active signals at the moment</p>
                <p class="text-sm mt-1">Signals will appear here when admin broadcasts them</p>
            </div>
        """)

    html = '<table class="w-full"><thead>'
    html += '<tr class="text-left text-xs font-semibold text-slate-500 uppercase tracking-wider border-b border-slate-700/50 bg-surface-800/50">'
    html += '<th class="p-4">Signal</th><th class="p-4">Index</th><th class="p-4">Type</th>'
    html += '<th class="p-4 text-right">Entry</th><th class="p-4 text-right">Target</th>'
    html += '<th class="p-4 text-right">SL</th><th class="p-4 text-center">Action</th></tr></thead><tbody>'

    for signal in signals:
        ce_class = "bg-green-500/20 text-green-400" if signal.signal_type == "CE" else "bg-red-500/20 text-red-400"
        html += f'''
            <tr class="border-b border-slate-700/30 hover:bg-surface-800/50">
                <td class="p-4 font-mono text-xs text-slate-400">{signal.signal_id[:8]}</td>
                <td class="p-4 text-white">{signal.index}</td>
                <td class="p-4"><span class="px-2 py-1 rounded text-xs font-bold {ce_class}">{signal.signal_type}</span></td>
                <td class="p-4 text-right font-mono text-white">₹{signal.entry_price:.2f}</td>
                <td class="p-4 text-right font-mono text-green-400">₹{signal.target:.2f}</td>
                <td class="p-4 text-right font-mono text-red-400">₹{signal.stop_loss:.2f}</td>
                <td class="p-4 text-center">
                    <button onclick="executeSignal('{signal.signal_id}')" class="px-3 py-1.5 rounded-lg bg-primary-600 hover:bg-primary-700 text-white text-xs font-bold transition-colors">
                        Execute
                    </button>
                </td>
            </tr>
        '''

    html += '</tbody></table>'
    return HTMLResponse(html)


@router.get("/htmx/open-positions", response_class=HTMLResponse)
async def htmx_open_positions(request: Request):
    """Get open positions for HTMX."""
    trading_user = get_trading_user(request)
    service = get_user_trading_service()

    trades = service.get_user_trades(trading_user.user_id, limit=50)
    open_trades = [t for t in trades if t.status == 'open']

    if not open_trades:
        return HTMLResponse("""
            <div class="p-8 text-center text-slate-500">
                <p>No open positions</p>
                <p class="text-sm mt-1">Your active trades will appear here</p>
            </div>
        """)

    html = '<table class="w-full"><thead>'
    html += '<tr class="text-left text-xs font-semibold text-slate-500 uppercase tracking-wider border-b border-slate-700/50 bg-surface-800/50">'
    html += '<th class="p-4">Instrument</th><th class="p-4">Type</th><th class="p-4 text-right">Entry</th>'
    html += '<th class="p-4 text-right">Current</th><th class="p-4 text-right">Qty</th>'
    html += '<th class="p-4 text-right">P&L</th><th class="p-4 text-center">Action</th></tr></thead><tbody>'

    for trade in open_trades:
        t = trade.to_dict()
        ce_class = "bg-green-500/20 text-green-400" if t.get('trade_type') == "CE" else "bg-red-500/20 text-red-400"
        pnl = t.get('unrealized_pnl', 0) or 0
        pnl_class = "text-green-400" if pnl >= 0 else "text-red-400"
        pnl_sign = "+" if pnl >= 0 else ""

        html += f'''
            <tr class="border-b border-slate-700/30 hover:bg-surface-800/50">
                <td class="p-4 text-white font-medium">{t.get('instrument', '')}</td>
                <td class="p-4"><span class="px-2 py-1 rounded text-xs font-bold {ce_class}">{t.get('trade_type', '')}</span></td>
                <td class="p-4 text-right font-mono text-white">₹{t.get('entry_price', 0):.2f}</td>
                <td class="p-4 text-right font-mono text-white">₹{t.get('current_price', t.get('entry_price', 0)):.2f}</td>
                <td class="p-4 text-right font-mono text-white">{t.get('quantity', 0)}</td>
                <td class="p-4 text-right font-mono {pnl_class}">{pnl_sign}₹{abs(pnl):.0f}</td>
                <td class="p-4 text-center">
                    <button onclick="closeTrade('{t.get('trade_id', '')}')" class="px-3 py-1 rounded bg-red-500/20 hover:bg-red-500/30 text-red-400 text-xs font-bold transition-colors">
                        Close
                    </button>
                </td>
            </tr>
        '''

    html += '</tbody></table>'
    return HTMLResponse(html)


# ==================== Trade Actions ====================

@router.post("/trade/{trade_id}/close")
async def close_trade(request: Request, trade_id: str):
    """Close a trade."""
    trading_user = get_trading_user(request)
    service = get_user_trading_service()

    # TODO: Implement actual trade close logic
    # This would execute a market order to close the position

    return {"success": True, "message": "Trade closed"}


@router.post("/signal/{signal_id}/execute")
async def execute_signal(request: Request, signal_id: str):
    """Execute a signal manually."""
    trading_user = get_trading_user(request)
    service = get_user_trading_service()

    # Get the signal
    signals = service.get_active_signals()
    signal = next((s for s in signals if s.signal_id == signal_id), None)

    if not signal:
        return JSONResponse({"success": False, "error": "Signal not found"}, status_code=404)

    # Execute trade for user
    success, message, trade = service.execute_trade_for_user(
        user_id=trading_user.user_id,
        signal=signal,
        mode=trading_user.trading_mode.value
    )

    if success:
        return {"success": True, "trade_id": trade.trade_id}
    else:
        return JSONResponse({"success": False, "error": message}, status_code=400)
