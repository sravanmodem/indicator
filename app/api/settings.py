"""
Settings API Routes
Handles trading settings updates and management
"""

from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger

from app.services.signal_history_service import get_history_service
from app.services.zerodha_auth import get_auth_service

router = APIRouter(prefix="/api/settings", tags=["Settings"])


def require_auth():
    """Check if user is authenticated."""
    auth = get_auth_service()
    if not auth.is_authenticated:
        raise HTTPException(status_code=401, detail="Not authenticated")


@router.post("/update")
async def update_settings(
    total_capital: float = Form(...),
    daily_profit_target: float = Form(...),
    max_risk_per_trade_pct: float = Form(...),
    max_daily_loss_pct: float = Form(...),
    max_positions: int = Form(...),
    automation_mode: str = Form(...),
    trading_style: str = Form(...),
):
    """Update user trading settings."""
    try:
        require_auth()

        history_service = get_history_service()

        # Validate inputs
        if total_capital < 10000:
            return JSONResponse(
                status_code=400,
                content={"error": "Capital must be at least ₹10,000"}
            )

        if max_risk_per_trade_pct < 0.1 or max_risk_per_trade_pct > 10:
            return JSONResponse(
                status_code=400,
                content={"error": "Risk per trade must be between 0.1% and 10%"}
            )

        # Update settings
        settings = history_service.update_settings(
            total_capital=total_capital,
            daily_profit_target=daily_profit_target,
            max_risk_per_trade_pct=max_risk_per_trade_pct,
            max_daily_loss_pct=max_daily_loss_pct,
            max_positions=max_positions,
            automation_mode=automation_mode,
            trading_style=trading_style,
        )

        logger.info(f"Settings updated: Capital=₹{total_capital:,.0f}, Target=₹{daily_profit_target:,.0f}")

        return JSONResponse(
            status_code=200,
            content={
                "message": "Settings updated successfully",
                "settings": {
                    "total_capital": settings.total_capital,
                    "daily_profit_target": settings.daily_profit_target,
                }
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating settings: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@router.get("/get")
async def get_settings():
    """Get current trading settings."""
    try:
        require_auth()

        history_service = get_history_service()
        settings = history_service.get_settings()

        return {
            "total_capital": settings.total_capital,
            "daily_profit_target": settings.daily_profit_target,
            "max_risk_per_trade_pct": settings.max_risk_per_trade_pct,
            "max_daily_loss_pct": settings.max_daily_loss_pct,
            "max_positions": settings.max_positions,
            "automation_mode": settings.automation_mode,
            "trading_style": settings.trading_style,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))
