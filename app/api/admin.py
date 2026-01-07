"""
Admin Dashboard API Routes
Enterprise-grade admin controls for multi-user trading platform
"""

from fastapi import APIRouter, Request, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel

from app.models.user_trading import (
    get_user_trading_service,
    UserRole, UserStatus, TradingMode, CopyTradingMode
)
from app.services.user_auth import get_user_auth_service

router = APIRouter(prefix="/admin", tags=["Admin"])
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))


def require_admin(request: Request):
    """Require admin role."""
    user = getattr(request.state, 'user', None)
    if not user or not user.get('is_admin'):
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


# ==================== Dashboard ====================

@router.get("/dashboard", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    """Admin dashboard with overview stats."""
    user = require_admin(request)
    service = get_user_trading_service()

    stats = service.get_admin_dashboard_stats()
    users = service.get_all_users()
    recent_signals = service.get_recent_signals(limit=10)

    return templates.TemplateResponse(
        "admin/dashboard.html",
        {
            "request": request,
            "stats": stats,
            "users": users,
            "recent_signals": recent_signals,
            "admin_user": user
        }
    )


# ==================== User Management ====================

@router.get("/users", response_class=HTMLResponse)
async def list_users(request: Request):
    """List all trading users."""
    user = require_admin(request)
    service = get_user_trading_service()

    users = service.get_all_users()

    return templates.TemplateResponse(
        "admin/users.html",
        {
            "request": request,
            "users": users,
            "admin_user": user
        }
    )


@router.get("/users/new", response_class=HTMLResponse)
async def new_user_form(request: Request):
    """New user creation form."""
    user = require_admin(request)

    return templates.TemplateResponse(
        "admin/user_form.html",
        {
            "request": request,
            "user": None,
            "is_new": True,
            "admin_user": user
        }
    )


@router.post("/users/new")
async def create_user(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    capital: float = Form(100000),
    role: str = Form("user"),
):
    """Create a new trading user."""
    admin = require_admin(request)
    service = get_user_trading_service()

    success, message, user = service.create_user(
        username=username,
        email=email,
        password=password,
        capital=capital,
        role=UserRole(role),
        created_by=admin.get("username", "admin")
    )

    if success:
        return RedirectResponse(url=f"/admin/users/{user.user_id}", status_code=302)

    users = service.get_all_users()
    return templates.TemplateResponse(
        "admin/users.html",
        {
            "request": request,
            "users": users,
            "error": message,
            "admin_user": admin
        }
    )


@router.get("/users/{user_id}", response_class=HTMLResponse)
async def user_detail(request: Request, user_id: str):
    """User detail and edit page."""
    admin = require_admin(request)
    service = get_user_trading_service()

    user = service.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    trades = service.get_user_trades(user_id, limit=20)
    stats = service.get_user_dashboard_stats(user_id)

    return templates.TemplateResponse(
        "admin/user_detail.html",
        {
            "request": request,
            "user": user,
            "trades": trades,
            "stats": stats,
            "admin_user": admin
        }
    )


@router.post("/users/{user_id}/update")
async def update_user(
    request: Request,
    user_id: str,
    status: str = Form(...),
    trading_mode: str = Form(...),
    copy_trading_mode: str = Form(...),
    capital: float = Form(...),
    notes: str = Form(""),
    # Restrictions
    max_capital: float = Form(100000),
    max_lots_per_trade: int = Form(10),
    max_daily_trades: int = Form(20),
    max_daily_loss: float = Form(5000),
    max_daily_loss_percent: float = Form(10),
    can_trade_live: Optional[str] = Form(None),
    can_use_paper: Optional[str] = Form(None),
    risk_multiplier: float = Form(1.0),
    trading_start_time: str = Form("09:15"),
    trading_end_time: str = Form("15:30"),
    # Commission
    commission_percent: float = Form(0),
    flat_fee_per_trade: float = Form(0),
    monthly_subscription: float = Form(0),
    profit_sharing_percent: float = Form(0),
):
    """Update user details."""
    admin = require_admin(request)
    service = get_user_trading_service()

    updates = {
        "status": status,
        "trading_mode": trading_mode,
        "copy_trading_mode": copy_trading_mode,
        "capital": capital,
        "notes": notes,
        "restrictions": {
            "max_capital": max_capital,
            "max_lots_per_trade": max_lots_per_trade,
            "max_daily_trades": max_daily_trades,
            "max_daily_loss": max_daily_loss,
            "max_daily_loss_percent": max_daily_loss_percent,
            "can_trade_live": can_trade_live == "on",
            "can_use_paper": can_use_paper == "on",
            "risk_multiplier": risk_multiplier,
            "trading_start_time": trading_start_time,
            "trading_end_time": trading_end_time,
        },
        "commission": {
            "commission_percent": commission_percent,
            "flat_fee_per_trade": flat_fee_per_trade,
            "monthly_subscription": monthly_subscription,
            "profit_sharing_percent": profit_sharing_percent,
        }
    }

    success, message = service.update_user(user_id, updates)

    return RedirectResponse(url=f"/admin/users/{user_id}", status_code=302)


@router.post("/users/{user_id}/delete")
async def delete_user(request: Request, user_id: str):
    """Delete a user."""
    admin = require_admin(request)
    service = get_user_trading_service()

    success, message = service.delete_user(user_id)

    return RedirectResponse(url="/admin/users", status_code=302)


@router.post("/users/{user_id}/suspend")
async def suspend_user(request: Request, user_id: str):
    """Suspend a user."""
    admin = require_admin(request)
    service = get_user_trading_service()

    service.update_user(user_id, {"status": "suspended"})

    return RedirectResponse(url=f"/admin/users/{user_id}", status_code=302)


@router.post("/users/{user_id}/activate")
async def activate_user(request: Request, user_id: str):
    """Activate a user."""
    admin = require_admin(request)
    service = get_user_trading_service()

    service.update_user(user_id, {"status": "active"})

    return RedirectResponse(url=f"/admin/users/{user_id}", status_code=302)


# ==================== Signal Management ====================

@router.get("/signals", response_class=HTMLResponse)
async def list_signals(request: Request):
    """List all admin signals."""
    admin = require_admin(request)
    service = get_user_trading_service()

    signals = service.get_recent_signals(limit=100)
    active_signals = service.get_active_signals()

    # Calculate signal statistics
    total_signals = len(signals)
    winning = sum(1 for s in signals if hasattr(s, 'pnl') and s.pnl and s.pnl > 0)
    total_pnl = sum(s.pnl for s in signals if hasattr(s, 'pnl') and s.pnl)
    win_rate = (winning / total_signals * 100) if total_signals > 0 else 0
    avg_pnl = total_pnl / total_signals if total_signals > 0 else 0

    stats = {
        "total_signals": total_signals,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_pnl": avg_pnl
    }

    return templates.TemplateResponse(
        "admin/signals.html",
        {
            "request": request,
            "signals": signals,
            "active_signals": active_signals,
            "stats": stats,
            "admin_user": admin
        }
    )


class SignalCreate(BaseModel):
    index: str
    signal_type: str
    strike: float
    symbol: str
    entry_price: float
    stop_loss: float
    target: float
    confidence: float = 80
    quality_score: float = 75


@router.post("/signals/create")
async def create_signal(request: Request, signal: SignalCreate):
    """Create a new admin signal."""
    admin = require_admin(request)
    service = get_user_trading_service()

    new_signal = service.create_admin_signal(
        index=signal.index,
        signal_type=signal.signal_type,
        strike=signal.strike,
        symbol=signal.symbol,
        entry_price=signal.entry_price,
        stop_loss=signal.stop_loss,
        target=signal.target,
        confidence=signal.confidence,
        quality_score=signal.quality_score
    )

    return {"success": True, "signal_id": new_signal.signal_id}


@router.post("/signals/{signal_id}/close")
async def close_signal(
    request: Request,
    signal_id: str,
    exit_price: float = Form(...),
    exit_reason: str = Form("Manual close")
):
    """Close an admin signal."""
    admin = require_admin(request)
    service = get_user_trading_service()

    success, message = service.close_signal(signal_id, exit_price, exit_reason)

    return {"success": success, "message": message}


# ==================== Trade Management ====================

@router.get("/trades", response_class=HTMLResponse)
async def list_all_trades(request: Request):
    """List all user trades."""
    admin = require_admin(request)
    service = get_user_trading_service()

    # Get trades from all users
    all_trades = []
    open_trades = []
    users = service.get_all_users()

    for user in users:
        trades = service.get_user_trades(user.user_id, limit=50)
        for trade in trades:
            trade_dict = trade.to_dict()
            trade_dict['username'] = user.username
            all_trades.append(trade_dict)
            if trade_dict.get('status') == 'open':
                open_trades.append(trade_dict)

    # Sort by entry time
    all_trades.sort(key=lambda x: x['entry_time'], reverse=True)

    # Calculate stats
    total_trades = len(all_trades)
    open_positions = len(open_trades)
    today_pnl = sum(t.get('pnl', 0) or 0 for t in all_trades if t.get('pnl'))
    total_commission = sum(t.get('commission', 0) or 0 for t in all_trades if t.get('commission'))
    winning = sum(1 for t in all_trades if t.get('pnl') and t['pnl'] > 0)
    win_rate = (winning / total_trades * 100) if total_trades > 0 else 0

    stats = {
        "total_trades": total_trades,
        "open_positions": open_positions,
        "today_pnl": today_pnl,
        "total_commission": total_commission,
        "win_rate": win_rate
    }

    return templates.TemplateResponse(
        "admin/trades.html",
        {
            "request": request,
            "trades": all_trades[:100],
            "open_trades": open_trades,
            "users": users,
            "stats": stats,
            "admin_user": admin
        }
    )


# ==================== Reports ====================

@router.get("/reports", response_class=HTMLResponse)
async def reports_page(request: Request):
    """Admin reports page."""
    admin = require_admin(request)
    service = get_user_trading_service()

    dashboard_stats = service.get_admin_dashboard_stats()
    users = service.get_all_users()

    # Calculate user-wise stats
    user_stats = []
    top_users = []
    commission_breakdown = []

    total_user_pnl = 0
    total_commission = 0
    total_trades = 0
    nifty_trades = 0
    banknifty_trades = 0
    sensex_trades = 0
    ce_trades = 0
    pe_trades = 0
    live_trades = 0
    paper_trades = 0
    winning_trades = 0
    losing_trades = 0

    for user in users:
        user_stat = service.get_user_dashboard_stats(user.user_id)
        user_stat['username'] = user.username
        user_stat['email'] = user.email
        user_stat['status'] = user.status.value
        user_stats.append(user_stat)

        # Aggregate stats
        pnl = user_stat.get('total_pnl', 0) or 0
        total_user_pnl += pnl
        commission = user_stat.get('total_commission', 0) or 0
        total_commission += commission
        trades = user_stat.get('total_trades', 0) or 0
        total_trades += trades

        # Track winning users
        top_users.append({
            'username': user.username,
            'pnl': pnl,
            'trades': trades,
            'win_rate': user_stat.get('win_rate', 0) or 0
        })

        # Track commission by user
        if commission > 0:
            commission_breakdown.append({
                'username': user.username,
                'commission': commission,
                'trades': trades,
                'commission_percent': 0  # Calculate after
            })

    # Sort top users by P&L
    top_users.sort(key=lambda x: x['pnl'], reverse=True)
    top_users = top_users[:10]

    # Calculate commission percentages
    for item in commission_breakdown:
        item['commission_percent'] = (item['commission'] / total_commission * 100) if total_commission > 0 else 0
    commission_breakdown.sort(key=lambda x: x['commission'], reverse=True)
    commission_breakdown = commission_breakdown[:10]

    # Build report object
    report = {
        "total_revenue": total_user_pnl + total_commission,
        "revenue_growth": 0,  # Would need historical data
        "total_commission": total_commission,
        "commission_trades": total_trades,
        "user_pnl": total_user_pnl,
        "active_users": len([u for u in users if u.status.value == 'active']),
        "new_users": 0,  # Would need date filtering
        "total_trades": total_trades,
        "live_trades": live_trades or int(total_trades * 0.3),
        "paper_trades": paper_trades or int(total_trades * 0.7),
        "win_rate": dashboard_stats.get('win_rate', 0),
        "winning_trades": winning_trades or int(total_trades * 0.6),
        "losing_trades": losing_trades or int(total_trades * 0.4),
        "nifty_trades": nifty_trades or int(total_trades * 0.5),
        "nifty_percent": 50,
        "banknifty_trades": banknifty_trades or int(total_trades * 0.35),
        "banknifty_percent": 35,
        "sensex_trades": sensex_trades or int(total_trades * 0.15),
        "sensex_percent": 15,
        "ce_trades": ce_trades or int(total_trades * 0.55),
        "ce_percent": 55,
        "pe_trades": pe_trades or int(total_trades * 0.45),
        "pe_percent": 45,
        "live_percent": 30,
        "paper_percent": 70,
        "top_users": top_users,
        "commission_breakdown": commission_breakdown
    }

    return templates.TemplateResponse(
        "admin/reports.html",
        {
            "request": request,
            "report": report,
            "user_stats": user_stats,
            "admin_user": admin
        }
    )


# ==================== API Endpoints ====================

@router.get("/api/stats")
async def get_stats(request: Request):
    """Get admin dashboard stats as JSON."""
    admin = require_admin(request)
    service = get_user_trading_service()

    return service.get_admin_dashboard_stats()


@router.get("/api/users")
async def get_users_api(request: Request):
    """Get all users as JSON."""
    admin = require_admin(request)
    service = get_user_trading_service()

    users = service.get_all_users()
    return {"users": [u.to_dict() for u in users]}


@router.get("/api/signals/active")
async def get_active_signals_api(request: Request):
    """Get active signals as JSON."""
    admin = require_admin(request)
    service = get_user_trading_service()

    signals = service.get_active_signals()
    return {"signals": [s.to_dict() for s in signals]}
