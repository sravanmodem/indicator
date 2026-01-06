"""
Paper Trading Models
Virtual portfolio tracking for practice trading
"""

from datetime import datetime
from enum import Enum

from sqlalchemy import Column, Integer, Float, String, DateTime, Enum as SQLEnum, Boolean
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class OrderType(str, Enum):
    """Order type enum."""
    MARKET = "market"
    LIMIT = "limit"


class OrderSide(str, Enum):
    """Order side enum."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    """Order status enum."""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"


class PositionStatus(str, Enum):
    """Position status enum."""
    OPEN = "open"
    CLOSED = "closed"


class PaperTradingPosition(Base):
    """Paper trading position table."""
    __tablename__ = "paper_trading_positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, nullable=False, default=datetime.now)

    # Signal reference
    signal_id = Column(Integer, nullable=True)
    index_name = Column(String(20), nullable=False)
    option_symbol = Column(String(100), nullable=False)
    strike = Column(Float, nullable=False)
    option_type = Column(String(5), nullable=False)  # CE or PE

    # Entry
    entry_time = Column(DateTime, nullable=False, default=datetime.now)
    entry_price = Column(Float, nullable=False)
    quantity = Column(Integer, nullable=False)
    stop_loss = Column(Float, nullable=False)
    target = Column(Float, nullable=False)

    # Exit
    exit_time = Column(DateTime, nullable=True)
    exit_price = Column(Float, nullable=True)
    exit_reason = Column(String(50), nullable=True)  # target_hit, stop_loss_hit, manual_exit

    # P&L
    pnl = Column(Float, nullable=True)
    pnl_percentage = Column(Float, nullable=True)

    # Status
    status = Column(SQLEnum(PositionStatus), default=PositionStatus.OPEN)

    # Notes
    notes = Column(String(500), nullable=True)


class PaperTradingOrder(Base):
    """Paper trading order table."""
    __tablename__ = "paper_trading_orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    position_id = Column(Integer, nullable=True)

    # Order details
    order_time = Column(DateTime, nullable=False, default=datetime.now)
    symbol = Column(String(100), nullable=False)
    side = Column(SQLEnum(OrderSide), nullable=False)
    order_type = Column(SQLEnum(OrderType), nullable=False)
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)

    # Execution
    filled_time = Column(DateTime, nullable=True)
    filled_price = Column(Float, nullable=True)
    status = Column(SQLEnum(OrderStatus), default=OrderStatus.PENDING)


class PaperTradingAccount(Base):
    """Paper trading account table."""
    __tablename__ = "paper_trading_account"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(100), nullable=False, default="default")

    # Capital
    initial_capital = Column(Float, nullable=False, default=500000)
    current_capital = Column(Float, nullable=False, default=500000)
    total_pnl = Column(Float, nullable=False, default=0)

    # Statistics
    total_trades = Column(Integer, nullable=False, default=0)
    winning_trades = Column(Integer, nullable=False, default=0)
    losing_trades = Column(Integer, nullable=False, default=0)

    # Last updated
    last_updated = Column(DateTime, nullable=False, default=datetime.now)

    # Settings
    is_active = Column(Boolean, default=True)
