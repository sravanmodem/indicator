"""
Signal History Service
Manages signal storage, tracking, and result updates
"""

from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import create_engine, desc
from sqlalchemy.orm import sessionmaker, Session
from loguru import logger

from app.models.signal_history import Base, SignalHistory, SignalResult, TradingSettings
from app.services.signal_engine import TradeSignal
from app.services.signal_quality import QualityScore
from app.core.config import get_settings


class SignalHistoryService:
    """Service for managing signal history."""

    def __init__(self):
        """Initialize signal history service."""
        settings = get_settings()
        # Use synchronous SQLite for history
        db_url = settings.database_url.replace("sqlite+aiosqlite", "sqlite")
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def _get_session(self) -> Session:
        """Get database session."""
        return self.SessionLocal()

    def save_signal(
        self,
        signal: TradeSignal,
        index_name: str,
        trading_style: str,
        quality_score: Optional[QualityScore] = None,
        quantity: Optional[int] = None,
        risk_amount: Optional[float] = None,
        potential_profit: Optional[float] = None,
        user_capital: Optional[float] = None,
    ) -> SignalHistory:
        """
        Save a new signal to history.

        Args:
            signal: TradeSignal to save
            index_name: Index name (NIFTY, BANKNIFTY, etc.)
            trading_style: Trading style (scalping, intraday, swing)
            quality_score: Optional quality score
            quantity: Position quantity
            risk_amount: Risk amount
            potential_profit: Potential profit
            user_capital: User's capital at signal time

        Returns:
            SignalHistory record
        """
        session = self._get_session()
        try:
            # Extract key indicators
            indicators_data = [
                {
                    "name": ind.name,
                    "signal": ind.signal,
                    "strength": ind.strength,
                    "value": str(ind.value) if ind.value is not None else None,
                }
                for ind in signal.indicators[:5]  # Store top 5 indicators
            ]

            # Create history record
            history = SignalHistory(
                timestamp=signal.timestamp,
                index_name=index_name.upper(),
                trading_style=trading_style.lower(),
                signal_type=signal.signal_type.value,
                direction=signal.direction,
                confidence=signal.confidence,
                quality_score=quality_score.total_score if quality_score else None,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                target_1=signal.target_1,
                target_2=signal.target_2,
                risk_reward=signal.risk_reward,
                option_symbol=signal.recommended_option.symbol if signal.recommended_option else None,
                option_strike=signal.recommended_option.strike if signal.recommended_option else None,
                option_ltp=signal.recommended_option.ltp if signal.recommended_option else None,
                option_delta=signal.recommended_option.delta if signal.recommended_option else None,
                option_theta=signal.recommended_option.theta if signal.recommended_option else None,
                quantity=quantity,
                risk_amount=risk_amount,
                potential_profit=potential_profit,
                indicators=indicators_data,
                supporting_factors=signal.supporting_factors,
                warning_factors=signal.warning_factors,
                user_capital=user_capital,
                result=SignalResult.PENDING,
            )

            session.add(history)
            session.commit()
            session.refresh(history)

            logger.info(f"Signal saved to history: ID={history.id}, {index_name} {signal.direction}")
            return history

        except Exception as e:
            session.rollback()
            logger.error(f"Error saving signal to history: {e}")
            raise
        finally:
            session.close()

    def update_execution(
        self,
        signal_id: int,
        order_id: str,
        execution_price: float,
        quantity: int,
    ) -> bool:
        """
        Update signal with execution details.

        Args:
            signal_id: Signal history ID
            order_id: Order ID
            execution_price: Execution price
            quantity: Executed quantity

        Returns:
            Success status
        """
        session = self._get_session()
        try:
            history = session.query(SignalHistory).filter_by(id=signal_id).first()
            if history:
                history.executed = True
                history.order_id = order_id
                history.execution_price = execution_price
                history.execution_time = datetime.now()
                history.quantity = quantity

                session.commit()
                logger.info(f"Signal {signal_id} execution updated: Order {order_id}")
                return True
            return False

        except Exception as e:
            session.rollback()
            logger.error(f"Error updating execution: {e}")
            return False
        finally:
            session.close()

    def update_result(
        self,
        signal_id: int,
        result: SignalResult,
        exit_price: float,
        actual_pnl: Optional[float] = None,
        exit_reason: Optional[str] = None,
    ) -> bool:
        """
        Update signal result when closed.

        Args:
            signal_id: Signal history ID
            result: Signal result (target_hit, stop_loss_hit, etc.)
            exit_price: Exit price
            actual_pnl: Actual P&L
            exit_reason: Reason for exit

        Returns:
            Success status
        """
        session = self._get_session()
        try:
            history = session.query(SignalHistory).filter_by(id=signal_id).first()
            if history:
                history.result = result
                history.exit_price = exit_price
                history.exit_time = datetime.now()
                history.actual_pnl = actual_pnl
                history.exit_reason = exit_reason

                session.commit()

                result_emoji = "✅" if result == SignalResult.TARGET_HIT else "🛑" if result == SignalResult.STOP_LOSS_HIT else "📌"
                logger.info(f"{result_emoji} Signal {signal_id} result: {result.value}, P&L: ₹{actual_pnl:,.0f}" if actual_pnl else f"Signal {signal_id} result: {result.value}")
                return True
            return False

        except Exception as e:
            session.rollback()
            logger.error(f"Error updating result: {e}")
            return False
        finally:
            session.close()

    def get_signal_by_id(self, signal_id: int) -> Optional[SignalHistory]:
        """Get signal by ID."""
        session = self._get_session()
        try:
            return session.query(SignalHistory).filter_by(id=signal_id).first()
        finally:
            session.close()

    def get_recent_signals(self, limit: int = 50, days: int = 7) -> list[SignalHistory]:
        """
        Get recent signals.

        Args:
            limit: Maximum number of signals
            days: Number of days to look back

        Returns:
            List of SignalHistory records
        """
        session = self._get_session()
        try:
            since = datetime.now() - timedelta(days=days)
            return (
                session.query(SignalHistory)
                .filter(SignalHistory.timestamp >= since)
                .order_by(desc(SignalHistory.timestamp))
                .limit(limit)
                .all()
            )
        finally:
            session.close()

    def get_pending_signals(self) -> list[SignalHistory]:
        """Get all pending signals (no result yet)."""
        session = self._get_session()
        try:
            return (
                session.query(SignalHistory)
                .filter(SignalHistory.result == SignalResult.PENDING)
                .order_by(desc(SignalHistory.timestamp))
                .all()
            )
        finally:
            session.close()

    def get_statistics(self, days: int = 30) -> dict:
        """
        Get signal statistics.

        Args:
            days: Number of days to analyze

        Returns:
            Statistics dictionary
        """
        session = self._get_session()
        try:
            since = datetime.now() - timedelta(days=days)
            all_signals = (
                session.query(SignalHistory)
                .filter(SignalHistory.timestamp >= since)
                .all()
            )

            if not all_signals:
                return {
                    "total_signals": 0,
                    "target_hit": 0,
                    "stop_loss_hit": 0,
                    "pending": 0,
                    "win_rate": 0,
                    "avg_quality_score": 0,
                    "total_pnl": 0,
                }

            target_hit = sum(1 for s in all_signals if s.result == SignalResult.TARGET_HIT)
            stop_loss_hit = sum(1 for s in all_signals if s.result == SignalResult.STOP_LOSS_HIT)
            pending = sum(1 for s in all_signals if s.result == SignalResult.PENDING)

            closed_signals = target_hit + stop_loss_hit
            win_rate = (target_hit / closed_signals * 100) if closed_signals > 0 else 0

            quality_scores = [s.quality_score for s in all_signals if s.quality_score is not None]
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0

            total_pnl = sum(s.actual_pnl for s in all_signals if s.actual_pnl is not None)

            return {
                "total_signals": len(all_signals),
                "target_hit": target_hit,
                "stop_loss_hit": stop_loss_hit,
                "pending": pending,
                "win_rate": round(win_rate, 1),
                "avg_quality_score": round(avg_quality, 1),
                "total_pnl": round(total_pnl, 2),
            }

        finally:
            session.close()

    # Trading Settings Management

    def get_settings(self, user_id: str = "default") -> TradingSettings:
        """Get user trading settings."""
        session = self._get_session()
        try:
            settings = session.query(TradingSettings).filter_by(user_id=user_id).first()
            if not settings:
                # Create default settings
                settings = TradingSettings(user_id=user_id)
                session.add(settings)
                session.commit()
                session.refresh(settings)
            return settings
        finally:
            session.close()

    def update_settings(
        self,
        total_capital: Optional[float] = None,
        daily_profit_target: Optional[float] = None,
        max_risk_per_trade_pct: Optional[float] = None,
        max_daily_loss_pct: Optional[float] = None,
        max_positions: Optional[int] = None,
        automation_mode: Optional[str] = None,
        trading_style: Optional[str] = None,
        user_id: str = "default",
    ) -> TradingSettings:
        """Update user trading settings."""
        session = self._get_session()
        try:
            settings = session.query(TradingSettings).filter_by(user_id=user_id).first()
            if not settings:
                settings = TradingSettings(user_id=user_id)
                session.add(settings)

            if total_capital is not None:
                settings.total_capital = total_capital
            if daily_profit_target is not None:
                settings.daily_profit_target = daily_profit_target
            if max_risk_per_trade_pct is not None:
                settings.max_risk_per_trade_pct = max_risk_per_trade_pct
            if max_daily_loss_pct is not None:
                settings.max_daily_loss_pct = max_daily_loss_pct
            if max_positions is not None:
                settings.max_positions = max_positions
            if automation_mode is not None:
                settings.automation_mode = automation_mode
            if trading_style is not None:
                settings.trading_style = trading_style

            settings.updated_at = datetime.now()

            session.commit()
            session.refresh(settings)

            logger.info(f"Trading settings updated: Capital=₹{settings.total_capital:,.0f}")
            return settings

        except Exception as e:
            session.rollback()
            logger.error(f"Error updating settings: {e}")
            raise
        finally:
            session.close()


# Singleton instance
_history_service: SignalHistoryService | None = None


def get_history_service() -> SignalHistoryService:
    """Get or create signal history service instance."""
    global _history_service
    if _history_service is None:
        _history_service = SignalHistoryService()
    return _history_service
