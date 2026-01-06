"""
Paper Trading Service
Virtual portfolio management for practice trading
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

from loguru import logger
from sqlalchemy import create_engine, desc
from sqlalchemy.orm import sessionmaker

from app.models.paper_trading import (
    Base,
    PaperTradingPosition,
    PaperTradingOrder,
    PaperTradingAccount,
    OrderSide,
    OrderStatus,
    PositionStatus,
)


class PaperTradingService:
    """Service for managing paper trading portfolio."""

    def __init__(self):
        """Initialize paper trading service."""
        # Create database in user's home directory
        db_path = Path.home() / ".options_indicator" / "paper_trading.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(self.engine)

        Session = sessionmaker(bind=self.engine)
        self.session = Session()

        # Initialize account if not exists
        self._init_account()

    def _init_account(self):
        """Initialize paper trading account if not exists."""
        account = self.session.query(PaperTradingAccount).filter_by(user_id="default").first()
        if not account:
            account = PaperTradingAccount(
                user_id="default",
                initial_capital=500000,
                current_capital=500000,
            )
            self.session.add(account)
            self.session.commit()
            logger.info("Initialized paper trading account with ₹5,00,000")

    def get_account(self) -> PaperTradingAccount:
        """Get paper trading account details."""
        return self.session.query(PaperTradingAccount).filter_by(user_id="default").first()

    def reset_account(self, initial_capital: float = 500000):
        """Reset paper trading account."""
        account = self.get_account()
        account.initial_capital = initial_capital
        account.current_capital = initial_capital
        account.total_pnl = 0
        account.total_trades = 0
        account.winning_trades = 0
        account.losing_trades = 0
        account.last_updated = datetime.now()
        self.session.commit()
        logger.info(f"Reset paper trading account to ₹{initial_capital:,.0f}")

    def open_position(
        self,
        signal_id: Optional[int],
        index_name: str,
        option_symbol: str,
        strike: float,
        option_type: str,
        entry_price: float,
        quantity: int,
        stop_loss: float,
        target: float,
        notes: Optional[str] = None,
    ) -> PaperTradingPosition:
        """Open a new paper trading position."""
        account = self.get_account()

        # Check if enough capital
        required_capital = entry_price * quantity
        if required_capital > account.current_capital:
            logger.warning(f"Insufficient capital. Required: ₹{required_capital:,.0f}, Available: ₹{account.current_capital:,.0f}")
            raise ValueError("Insufficient capital")

        # Create position
        position = PaperTradingPosition(
            signal_id=signal_id,
            index_name=index_name,
            option_symbol=option_symbol,
            strike=strike,
            option_type=option_type,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=stop_loss,
            target=target,
            status=PositionStatus.OPEN,
            notes=notes,
        )
        self.session.add(position)

        # Create buy order
        order = PaperTradingOrder(
            position_id=position.id,
            symbol=option_symbol,
            side=OrderSide.BUY,
            order_type="market",
            quantity=quantity,
            price=entry_price,
            filled_time=datetime.now(),
            filled_price=entry_price,
            status=OrderStatus.FILLED,
        )
        self.session.add(order)

        # Update account
        account.current_capital -= required_capital
        account.total_trades += 1
        account.last_updated = datetime.now()

        self.session.commit()
        logger.info(f"Opened paper position: {option_symbol} @ ₹{entry_price} x {quantity}")
        return position

    def close_position(
        self,
        position_id: int,
        exit_price: float,
        exit_reason: str = "manual_exit",
    ) -> PaperTradingPosition:
        """Close a paper trading position."""
        position = self.session.query(PaperTradingPosition).filter_by(id=position_id).first()
        if not position:
            raise ValueError(f"Position {position_id} not found")

        if position.status == PositionStatus.CLOSED:
            raise ValueError(f"Position {position_id} already closed")

        # Calculate P&L
        pnl = (exit_price - position.entry_price) * position.quantity
        pnl_pct = ((exit_price - position.entry_price) / position.entry_price) * 100

        # Update position
        position.exit_time = datetime.now()
        position.exit_price = exit_price
        position.exit_reason = exit_reason
        position.pnl = pnl
        position.pnl_percentage = pnl_pct
        position.status = PositionStatus.CLOSED

        # Create sell order
        order = PaperTradingOrder(
            position_id=position.id,
            symbol=position.option_symbol,
            side=OrderSide.SELL,
            order_type="market",
            quantity=position.quantity,
            price=exit_price,
            filled_time=datetime.now(),
            filled_price=exit_price,
            status=OrderStatus.FILLED,
        )
        self.session.add(order)

        # Update account
        account = self.get_account()
        account.current_capital += (position.entry_price + pnl / position.quantity) * position.quantity
        account.total_pnl += pnl

        if pnl > 0:
            account.winning_trades += 1
        else:
            account.losing_trades += 1

        account.last_updated = datetime.now()

        self.session.commit()
        logger.info(f"Closed paper position: {position.option_symbol} @ ₹{exit_price} | P&L: ₹{pnl:,.0f} ({pnl_pct:+.2f}%)")
        return position

    def get_open_positions(self) -> List[PaperTradingPosition]:
        """Get all open positions."""
        return self.session.query(PaperTradingPosition).filter_by(status=PositionStatus.OPEN).all()

    def get_all_positions(self, limit: int = 50) -> List[PaperTradingPosition]:
        """Get all positions."""
        return (
            self.session.query(PaperTradingPosition)
            .order_by(desc(PaperTradingPosition.entry_time))
            .limit(limit)
            .all()
        )

    def get_statistics(self) -> Dict:
        """Get paper trading statistics."""
        account = self.get_account()
        positions = self.get_all_positions(limit=1000)

        closed_positions = [p for p in positions if p.status == PositionStatus.CLOSED]
        winning_positions = [p for p in closed_positions if p.pnl and p.pnl > 0]
        losing_positions = [p for p in closed_positions if p.pnl and p.pnl < 0]

        total_closed = len(closed_positions)
        win_rate = (len(winning_positions) / total_closed * 100) if total_closed > 0 else 0

        avg_win = sum(p.pnl for p in winning_positions) / len(winning_positions) if winning_positions else 0
        avg_loss = sum(p.pnl for p in losing_positions) / len(losing_positions) if losing_positions else 0

        profit_factor = abs(sum(p.pnl for p in winning_positions) / sum(p.pnl for p in losing_positions)) if losing_positions and sum(p.pnl for p in losing_positions) != 0 else 0

        return {
            "initial_capital": account.initial_capital,
            "current_capital": account.current_capital,
            "total_pnl": account.total_pnl,
            "pnl_percentage": ((account.current_capital - account.initial_capital) / account.initial_capital * 100),
            "total_trades": account.total_trades,
            "winning_trades": len(winning_positions),
            "losing_trades": len(losing_positions),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "open_positions": len([p for p in positions if p.status == PositionStatus.OPEN]),
        }

    def check_stop_loss_target(self, position_id: int, current_price: float) -> Optional[str]:
        """Check if position hit stop loss or target."""
        position = self.session.query(PaperTradingPosition).filter_by(id=position_id).first()
        if not position or position.status == PositionStatus.CLOSED:
            return None

        # Check target hit
        if current_price >= position.target:
            self.close_position(position_id, position.target, "target_hit")
            return "target_hit"

        # Check stop loss hit
        if current_price <= position.stop_loss:
            self.close_position(position_id, position.stop_loss, "stop_loss_hit")
            return "stop_loss_hit"

        return None


# Global service instance
_paper_trading_service: Optional[PaperTradingService] = None


def get_paper_trading_service() -> PaperTradingService:
    """Get or create paper trading service singleton."""
    global _paper_trading_service
    if _paper_trading_service is None:
        _paper_trading_service = PaperTradingService()
    return _paper_trading_service
