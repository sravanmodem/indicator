"""
Auto Trading API Routes
Manages automated trading configuration and execution
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from loguru import logger
from pydantic import BaseModel

from app.services.zerodha_auth import get_auth_service

router = APIRouter(prefix="/api/auto-trade", tags=["Auto Trading"])
templates = Jinja2Templates(directory="app/templates")

# Settings file path
SETTINGS_FILE = Path("data/auto_trade_settings.json")
SETTINGS_FILE.parent.mkdir(exist_ok=True)


class IndexConfig(BaseModel):
    """Configuration for a single index."""
    enabled: bool = True
    capital: float = 250000  # ₹2.5 lakh
    maxDailyLoss: float = 20  # 20%
    maxLossPerTrade: float = 10  # 10%
    fundUtilization: float = 100  # 100%


class TradingSettings(BaseModel):
    """Trading rules and filters."""
    minQuality: str = "70"
    tradingStyle: str = "intraday"
    tradeTypes: list[str] = ["CE", "PE"]
    startTime: str = "09:20"
    endTime: str = "15:15"
    avoidExpiry: bool = True
    closeEOD: bool = True
    trailingSL: bool = False


class AutoTradeConfig(BaseModel):
    """Complete auto-trade configuration."""
    enabled: bool = False
    indices: dict[str, dict] = {
        "NIFTY": {
            "enabled": True,
            "capital": 250000,
            "maxDailyLoss": 20,
            "maxLossPerTrade": 10,
            "fundUtilization": 100,
        },
        "BANKNIFTY": {
            "enabled": True,
            "capital": 250000,
            "maxDailyLoss": 20,
            "maxLossPerTrade": 10,
            "fundUtilization": 100,
        },
        "SENSEX": {
            "enabled": True,
            "capital": 250000,
            "maxDailyLoss": 20,
            "maxLossPerTrade": 10,
            "fundUtilization": 100,
        },
    }
    settings: dict = {
        "minQuality": "70",
        "tradingStyle": "intraday",
        "tradeTypes": ["CE", "PE"],
        "startTime": "09:20",
        "endTime": "15:15",
        "avoidExpiry": True,
        "closeEOD": True,
        "trailingSL": False,
    }


# Runtime state
_auto_trade_state = {
    "running": False,
    "started_at": None,
    "total_pnl": 0,
    "active_trades": 0,
    "signals_today": 0,
    "indices": {
        "NIFTY": {"pnl": 0, "trades": 0, "lossUsedPct": 0},
        "BANKNIFTY": {"pnl": 0, "trades": 0, "lossUsedPct": 0},
        "SENSEX": {"pnl": 0, "trades": 0, "lossUsedPct": 0},
    },
    "trades_log": [],
}


def load_settings() -> dict[str, Any]:
    """Load settings from file."""
    if SETTINGS_FILE.exists():
        try:
            return json.loads(SETTINGS_FILE.read_text())
        except Exception as e:
            logger.error(f"Failed to load auto-trade settings: {e}")
    return AutoTradeConfig().model_dump()


def save_settings(config: dict[str, Any]) -> bool:
    """Save settings to file."""
    try:
        SETTINGS_FILE.parent.mkdir(exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps(config, indent=2))
        logger.info("Auto-trade settings saved")
        return True
    except Exception as e:
        logger.error(f"Failed to save auto-trade settings: {e}")
        return False


@router.get("/settings")
async def get_settings():
    """Get current auto-trade settings."""
    settings = load_settings()
    settings["enabled"] = _auto_trade_state["running"]
    return settings


@router.post("/settings")
async def update_settings(config: AutoTradeConfig):
    """Update auto-trade settings."""
    settings_dict = config.model_dump()

    if not save_settings(settings_dict):
        raise HTTPException(status_code=500, detail="Failed to save settings")

    logger.info(f"Auto-trade settings updated: {list(settings_dict['indices'].keys())}")
    return {"status": "success", "message": "Settings saved"}


@router.post("/start")
async def start_auto_trading():
    """Start the auto-trading system."""
    auth = get_auth_service()

    if not auth.is_authenticated:
        raise HTTPException(status_code=401, detail="Please login to Zerodha first")

    if _auto_trade_state["running"]:
        return {"status": "already_running", "message": "Auto trading is already running"}

    # Load settings
    settings = load_settings()

    # Validate settings
    total_capital = sum(
        idx["capital"] for idx in settings.get("indices", {}).values()
        if idx.get("enabled", False)
    )

    if total_capital <= 0:
        raise HTTPException(status_code=400, detail="No capital allocated for trading")

    # Start the auto-trading
    _auto_trade_state["running"] = True
    _auto_trade_state["started_at"] = datetime.now().isoformat()

    logger.info(f"Auto trading STARTED with total capital: ₹{total_capital:,.0f}")

    return {
        "status": "success",
        "message": "Auto trading started",
        "total_capital": total_capital,
        "started_at": _auto_trade_state["started_at"],
    }


@router.post("/stop")
async def stop_auto_trading():
    """Stop the auto-trading system."""
    if not _auto_trade_state["running"]:
        return {"status": "already_stopped", "message": "Auto trading is not running"}

    _auto_trade_state["running"] = False
    _auto_trade_state["started_at"] = None

    logger.info("Auto trading STOPPED")

    return {
        "status": "success",
        "message": "Auto trading stopped",
        "total_pnl": _auto_trade_state["total_pnl"],
    }


@router.get("/status")
async def get_status():
    """Get current auto-trading status."""
    return {
        "running": _auto_trade_state["running"],
        "startedAt": _auto_trade_state["started_at"],
        "totalPnl": _auto_trade_state["total_pnl"],
        "activeTrades": _auto_trade_state["active_trades"],
        "signalsToday": _auto_trade_state["signals_today"],
        "indices": _auto_trade_state["indices"],
    }


@router.get("/trades")
async def get_trades_log():
    """Get today's auto trades log."""
    return {
        "trades": _auto_trade_state["trades_log"],
        "count": len(_auto_trade_state["trades_log"]),
    }


# HTMX endpoint for trades log table
htmx_router = APIRouter(prefix="/htmx", tags=["HTMX Auto Trade"])


@htmx_router.get("/auto-trades-log", response_class=HTMLResponse)
async def auto_trades_log_partial(request: Request):
    """Render auto trades log table rows."""
    trades = _auto_trade_state["trades_log"]

    if not trades:
        return HTMLResponse(content="""
            <tr>
                <td colspan="10" class="text-center py-8 text-surface-500">
                    <i data-lucide="clock" class="w-8 h-8 mx-auto mb-2 text-surface-700"></i>
                    <p>No auto trades today</p>
                    <p class="text-xs mt-1">Trades will appear here when executed</p>
                </td>
            </tr>
        """)

    html_rows = []
    for trade in trades:
        pnl = trade.get("pnl", 0)
        pnl_color = "text-bullish-400" if pnl >= 0 else "text-bearish-400"
        pnl_sign = "+" if pnl >= 0 else ""

        status = trade.get("status", "active")
        status_badge = {
            "active": "badge-primary",
            "target_hit": "badge-success",
            "sl_hit": "badge-danger",
            "closed": "badge-neutral",
        }.get(status, "badge-neutral")

        signal_badge = "badge-success" if trade.get("signal") in ["CE", "STRONG_CE"] else "badge-danger"

        html_rows.append(f"""
            <tr class="hover:bg-surface-800/30 transition-colors">
                <td class="px-4 py-3 text-sm text-surface-400">{trade.get('time', '--')}</td>
                <td class="px-4 py-3 font-medium text-white">{trade.get('index', '--')}</td>
                <td class="px-4 py-3"><span class="{signal_badge}">{trade.get('signal', '--')}</span></td>
                <td class="px-4 py-3 text-surface-200">{trade.get('strike', '--')}</td>
                <td class="px-4 py-3 text-surface-200">{trade.get('action', '--')}</td>
                <td class="px-4 py-3 text-surface-200">{trade.get('qty', 0)}</td>
                <td class="px-4 py-3 text-surface-200">₹{trade.get('entry', 0):,.2f}</td>
                <td class="px-4 py-3 text-surface-200">₹{trade.get('ltp', 0):,.2f}</td>
                <td class="px-4 py-3 {pnl_color}">{pnl_sign}₹{abs(pnl):,.0f}</td>
                <td class="px-4 py-3"><span class="{status_badge}">{status.replace('_', ' ').title()}</span></td>
            </tr>
        """)

    return HTMLResponse(content="".join(html_rows))


def get_htmx_router():
    """Return the HTMX router for auto-trade endpoints."""
    return htmx_router
