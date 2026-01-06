"""
Paper Trading API Routes
Virtual portfolio management endpoints (v2)
"""

from fastapi import APIRouter, HTTPException, Form
from fastapi.responses import JSONResponse
from loguru import logger

from app.services.paper_trading_service import get_paper_trading_service

router = APIRouter(prefix="/api/paper-trading", tags=["Paper Trading"])


@router.post("/open-position")
async def open_paper_position(
    signal_id: int = Form(None),
    index_name: str = Form(...),
    option_symbol: str = Form(...),
    strike: float = Form(...),
    option_type: str = Form(...),
    entry_price: float = Form(...),
    quantity: int = Form(...),
    stop_loss: float = Form(...),
    target: float = Form(...),
    notes: str = Form(None),
):
    """Open a new paper trading position."""
    try:
        service = get_paper_trading_service()
        position = service.open_position(
            signal_id=signal_id,
            index_name=index_name,
            option_symbol=option_symbol,
            strike=strike,
            option_type=option_type,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=stop_loss,
            target=target,
            notes=notes,
        )

        return JSONResponse({
            "success": True,
            "message": f"Paper position opened: {option_symbol}",
            "position_id": position.id,
        })

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error opening paper position: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/close-position/{position_id}")
async def close_paper_position(
    position_id: int,
    exit_price: float = Form(...),
    exit_reason: str = Form("manual_exit"),
):
    """Close a paper trading position."""
    try:
        service = get_paper_trading_service()
        position = service.close_position(
            position_id=position_id,
            exit_price=exit_price,
            exit_reason=exit_reason,
        )

        return JSONResponse({
            "success": True,
            "message": f"Position closed with P&L: ₹{position.pnl:,.0f}",
            "pnl": position.pnl,
            "pnl_percentage": position.pnl_percentage,
        })

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error closing paper position: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/account")
async def get_paper_account():
    """Get paper trading account details."""
    try:
        service = get_paper_trading_service()
        account = service.get_account()

        # Calculate win rate
        total_closed = account.winning_trades + account.losing_trades
        win_rate = (account.winning_trades / total_closed * 100) if total_closed > 0 else 0

        return {
            # For template compatibility
            "capital": account.initial_capital,
            "available": account.current_capital,
            "today_pnl": account.total_pnl,
            "win_rate": round(win_rate, 1),
            "total_trades": account.total_trades,
            # Additional details
            "initial_capital": account.initial_capital,
            "current_capital": account.current_capital,
            "total_pnl": account.total_pnl,
            "pnl_percentage": round(((account.current_capital - account.initial_capital) / account.initial_capital * 100), 2) if account.initial_capital > 0 else 0,
            "winning_trades": account.winning_trades,
            "losing_trades": account.losing_trades,
            "is_active": account.is_active,
        }

    except Exception as e:
        logger.error(f"Error getting paper account: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset-account")
async def reset_paper_account(
    initial_capital: float = Form(500000),
):
    """Reset paper trading account."""
    try:
        service = get_paper_trading_service()
        service.reset_account(initial_capital=initial_capital)

        return JSONResponse({
            "success": True,
            "message": f"Paper trading account reset to ₹{initial_capital:,.0f}",
        })

    except Exception as e:
        logger.error(f"Error resetting paper account: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/positions/open")
async def get_open_positions():
    """Get all open paper positions."""
    try:
        service = get_paper_trading_service()
        positions = service.get_open_positions()

        return [
            {
                "id": p.id,
                "index_name": p.index_name,
                "option_symbol": p.option_symbol,
                "strike": p.strike,
                "option_type": p.option_type,
                "entry_time": p.entry_time.isoformat(),
                "entry_price": p.entry_price,
                "quantity": p.quantity,
                "stop_loss": p.stop_loss,
                "target": p.target,
                "notes": p.notes,
            }
            for p in positions
        ]

    except Exception as e:
        logger.error(f"Error getting open positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/positions/all")
async def get_all_positions(limit: int = 50):
    """Get all paper positions."""
    try:
        service = get_paper_trading_service()
        positions = service.get_all_positions(limit=limit)

        return [
            {
                "id": p.id,
                "index_name": p.index_name,
                "option_symbol": p.option_symbol,
                "strike": p.strike,
                "option_type": p.option_type,
                "entry_time": p.entry_time.isoformat(),
                "entry_price": p.entry_price,
                "quantity": p.quantity,
                "stop_loss": p.stop_loss,
                "target": p.target,
                "exit_time": p.exit_time.isoformat() if p.exit_time else None,
                "exit_price": p.exit_price,
                "exit_reason": p.exit_reason,
                "pnl": p.pnl,
                "pnl_percentage": p.pnl_percentage,
                "status": p.status.value,
                "notes": p.notes,
            }
            for p in positions
        ]

    except Exception as e:
        logger.error(f"Error getting all positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics")
async def get_paper_statistics():
    """Get paper trading statistics."""
    try:
        service = get_paper_trading_service()
        stats = service.get_statistics()
        return stats

    except Exception as e:
        logger.error(f"Error getting paper statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))
