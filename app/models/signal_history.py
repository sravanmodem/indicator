"""
Signal History Database Models
Stores all generated signals with entry, SL, target and results tracking
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import Column, Integer, String, Float, DateTime, Enum as SQLEnum, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class SignalResult(str, Enum):
    """Signal result enumeration."""
    PENDING = "pending"  # Signal active, no result yet
    TARGET_HIT = "target_hit"  # Target reached
    STOP_LOSS_HIT = "stop_loss_hit"  # Stop loss triggered
    PARTIAL_TARGET = "partial_target"  # Partial target hit
    MANUAL_EXIT = "manual_exit"  # Manually closed
    EXPIRED = "expired"  # Signal expired without action
    CANCELLED = "cancelled"  # Signal cancelled


class SignalHistory(Base):
    """Signal history table."""
    __tablename__ = "signal_history"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Signal identification
    timestamp = Column(DateTime, nullable=False, default=datetime.now)
    index_name = Column(String(20), nullable=False)  # NIFTY, BANKNIFTY, SENSEX
    trading_style = Column(String(20), nullable=False)  # scalping, intraday, swing

    # Signal details
    signal_type = Column(String(20), nullable=False)  # strong_ce, ce, pe, etc.
    direction = Column(String(5), nullable=False)  # CE or PE
    confidence = Column(Float, nullable=False)  # 0-100
    quality_score = Column(Float, nullable=True)  # 0-100 quality score

    # Entry levels
    entry_price = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=False)
    target_1 = Column(Float, nullable=False)
    target_2 = Column(Float, nullable=True)
    risk_reward = Column(Float, nullable=False)

    # Option details
    option_symbol = Column(String(50), nullable=True)
    option_strike = Column(Float, nullable=True)
    option_ltp = Column(Float, nullable=True)
    option_delta = Column(Float, nullable=True)
    option_theta = Column(Float, nullable=True)

    # Position sizing
    quantity = Column(Integer, nullable=True)
    risk_amount = Column(Float, nullable=True)
    potential_profit = Column(Float, nullable=True)

    # Execution tracking
    order_id = Column(String(50), nullable=True)  # Order ID if executed
    executed = Column(Boolean, default=False)
    execution_price = Column(Float, nullable=True)
    execution_time = Column(DateTime, nullable=True)

    # Result tracking
    result = Column(SQLEnum(SignalResult), default=SignalResult.PENDING)
    exit_price = Column(Float, nullable=True)
    exit_time = Column(DateTime, nullable=True)
    actual_pnl = Column(Float, nullable=True)  # Actual profit/loss
    exit_reason = Column(String(200), nullable=True)

    # Additional metadata
    indicators = Column(JSON, nullable=True)  # Key indicators snapshot
    supporting_factors = Column(JSON, nullable=True)
    warning_factors = Column(JSON, nullable=True)

    # User tracking
    user_capital = Column(Float, nullable=True)  # Capital at signal time

    def __repr__(self):
        return f"<SignalHistory(id={self.id}, {self.index_name} {self.direction} @ {self.timestamp}, Result={self.result.value})>"


class TradingSettings(Base):
    """User trading settings table."""
    __tablename__ = "trading_settings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(50), nullable=False, default="default")  # For future multi-user support

    # Capital settings
    total_capital = Column(Float, nullable=False, default=500000)
    daily_profit_target = Column(Float, nullable=False, default=25000)

    # Risk settings
    max_risk_per_trade_pct = Column(Float, nullable=False, default=2.0)
    max_daily_loss_pct = Column(Float, nullable=False, default=3.0)
    max_positions = Column(Integer, nullable=False, default=5)

    # Trading preferences
    automation_mode = Column(String(20), default="manual")  # manual, semi_auto, full_auto
    trading_style = Column(String(20), default="intraday")
    indices_to_trade = Column(JSON, default=["NIFTY", "BANKNIFTY"])

    # Timestamps
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    def __repr__(self):
        return f"<TradingSettings(user_id={self.user_id}, capital=₹{self.total_capital:,.0f})>"
