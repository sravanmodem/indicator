"""
User Trading Models
Multi-user trading system with admin controls
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from enum import Enum
from typing import Optional, Dict, Any, List
import json
from pathlib import Path
import secrets
import hashlib

from loguru import logger


class UserRole(Enum):
    """User roles."""
    ADMIN = "admin"
    USER = "user"


class UserStatus(Enum):
    """User account status."""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    PENDING = "pending"


class TradingMode(Enum):
    """Trading mode for users."""
    PAPER = "paper"
    LIVE = "live"
    DISABLED = "disabled"


class CopyTradingMode(Enum):
    """Copy trading mode."""
    AUTO = "auto"  # Auto-execute admin signals
    MANUAL = "manual"  # User confirms each trade
    DISABLED = "disabled"


@dataclass
class ZerodhaCredentials:
    """User's Zerodha API credentials."""
    api_key: str = ""
    api_secret: str = ""
    access_token: str = ""
    refresh_token: str = ""
    user_id: str = ""
    user_name: str = ""
    is_connected: bool = False
    last_connected: Optional[str] = None
    token_expiry: Optional[str] = None


@dataclass
class UserRestrictions:
    """Admin-defined restrictions for a user."""
    max_capital: float = 100000  # Max capital allowed
    max_lots_per_trade: int = 10  # Max lots per single trade
    max_daily_trades: int = 20  # Max trades per day
    max_daily_loss: float = 5000  # Max daily loss allowed
    max_daily_loss_percent: float = 10.0  # Max daily loss %
    allowed_indices: List[str] = field(default_factory=lambda: ["NIFTY", "BANKNIFTY", "SENSEX"])
    allowed_option_types: List[str] = field(default_factory=lambda: ["CE", "PE"])
    trading_start_time: str = "09:15"  # Trading start time
    trading_end_time: str = "15:30"  # Trading end time
    can_trade_live: bool = False  # Can trade with real money
    can_use_paper: bool = True  # Can use paper trading
    risk_multiplier: float = 1.0  # Position size multiplier (0.5 = half size)


@dataclass
class CommissionSettings:
    """Commission/charges settings for a user."""
    commission_percent: float = 0.0  # % of profit as commission
    flat_fee_per_trade: float = 0.0  # Flat fee per trade
    monthly_subscription: float = 0.0  # Monthly subscription fee
    profit_sharing_percent: float = 0.0  # % of profit shared with admin
    min_profit_for_sharing: float = 0.0  # Min profit before sharing applies


@dataclass
class UserTradingStats:
    """User's trading statistics."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    total_commission_paid: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    current_streak: int = 0
    max_win_streak: int = 0
    max_loss_streak: int = 0
    # Daily stats
    today_trades: int = 0
    today_pnl: float = 0.0
    today_commission: float = 0.0
    stats_date: str = ""


@dataclass
class TradingUser:
    """Trading system user."""
    user_id: str
    username: str
    email: str
    password_hash: str
    salt: str
    role: UserRole = UserRole.USER
    status: UserStatus = UserStatus.PENDING

    # Trading settings
    trading_mode: TradingMode = TradingMode.PAPER
    copy_trading_mode: CopyTradingMode = CopyTradingMode.MANUAL
    capital: float = 100000  # User's trading capital

    # Zerodha integration
    zerodha: ZerodhaCredentials = field(default_factory=ZerodhaCredentials)

    # Restrictions (set by admin)
    restrictions: UserRestrictions = field(default_factory=UserRestrictions)

    # Commission settings
    commission: CommissionSettings = field(default_factory=CommissionSettings)

    # Stats
    stats: UserTradingStats = field(default_factory=UserTradingStats)

    # Metadata
    created_at: str = ""
    last_login: str = ""
    created_by: str = ""  # Admin who created this user
    notes: str = ""  # Admin notes about user

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "password_hash": self.password_hash,
            "salt": self.salt,
            "role": self.role.value,
            "status": self.status.value,
            "trading_mode": self.trading_mode.value,
            "copy_trading_mode": self.copy_trading_mode.value,
            "capital": self.capital,
            "zerodha": asdict(self.zerodha),
            "restrictions": asdict(self.restrictions),
            "commission": asdict(self.commission),
            "stats": asdict(self.stats),
            "created_at": self.created_at,
            "last_login": self.last_login,
            "created_by": self.created_by,
            "notes": self.notes,
        }
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradingUser":
        """Create from dictionary."""
        return cls(
            user_id=data["user_id"],
            username=data["username"],
            email=data["email"],
            password_hash=data["password_hash"],
            salt=data["salt"],
            role=UserRole(data.get("role", "user")),
            status=UserStatus(data.get("status", "pending")),
            trading_mode=TradingMode(data.get("trading_mode", "paper")),
            copy_trading_mode=CopyTradingMode(data.get("copy_trading_mode", "manual")),
            capital=data.get("capital", 100000),
            zerodha=ZerodhaCredentials(**data.get("zerodha", {})),
            restrictions=UserRestrictions(**data.get("restrictions", {})),
            commission=CommissionSettings(**data.get("commission", {})),
            stats=UserTradingStats(**data.get("stats", {})),
            created_at=data.get("created_at", ""),
            last_login=data.get("last_login", ""),
            created_by=data.get("created_by", ""),
            notes=data.get("notes", ""),
        )


@dataclass
class UserTrade:
    """Individual trade executed for a user."""
    trade_id: str
    user_id: str
    admin_signal_id: str  # Reference to admin's signal

    # Trade details
    index: str
    symbol: str
    strike: float
    option_type: str
    lots: int
    quantity: int

    # Execution
    entry_price: float
    entry_time: str
    exit_price: float = 0.0
    exit_time: str = ""
    exit_reason: str = ""

    # P&L
    pnl: float = 0.0
    pnl_percent: float = 0.0
    commission: float = 0.0
    net_pnl: float = 0.0

    # Status
    status: str = "open"  # open, closed, cancelled
    is_paper: bool = True

    # Targets
    stop_loss: float = 0.0
    target: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserTrade":
        return cls(**data)


@dataclass
class AdminSignal:
    """Signal generated by admin for copy trading."""
    signal_id: str
    timestamp: str

    # Signal details
    index: str
    signal_type: str  # CE or PE
    strike: float
    symbol: str
    confidence: float
    quality_score: float

    # Targets
    entry_price: float
    stop_loss: float
    target: float

    # Execution tracking
    executed_by: List[str] = field(default_factory=list)  # User IDs who executed
    skipped_by: List[str] = field(default_factory=list)  # User IDs who skipped

    # Status
    status: str = "active"  # active, closed, expired
    exit_price: float = 0.0
    exit_time: str = ""
    exit_reason: str = ""
    pnl_percent: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdminSignal":
        return cls(**data)


class UserTradingService:
    """Service for managing trading users."""

    def __init__(self):
        self.data_dir = Path(__file__).parent.parent.parent / "data"
        self.data_dir.mkdir(exist_ok=True)

        self.users_file = self.data_dir / "trading_users.json"
        self.trades_file = self.data_dir / "user_trades.json"
        self.signals_file = self.data_dir / "admin_signals.json"

        self._users: Dict[str, TradingUser] = {}
        self._trades: List[UserTrade] = []
        self._signals: List[AdminSignal] = []

        self._load_data()

    def _hash_password(self, password: str, salt: Optional[str] = None) -> tuple[str, str]:
        """Hash password with salt."""
        if salt is None:
            salt = secrets.token_hex(16)
        hashed = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode(),
            salt.encode(),
            100000
        ).hex()
        return hashed, salt

    def _verify_password(self, password: str, stored_hash: str, salt: str) -> bool:
        """Verify password."""
        hashed, _ = self._hash_password(password, salt)
        return hashed == stored_hash

    def _load_data(self):
        """Load all data from files."""
        # Load users
        if self.users_file.exists():
            try:
                with open(self.users_file, 'r') as f:
                    data = json.load(f)
                    for user_data in data.get("users", []):
                        user = TradingUser.from_dict(user_data)
                        self._users[user.user_id] = user
                logger.info(f"Loaded {len(self._users)} trading users")
            except Exception as e:
                logger.error(f"Error loading users: {e}")

        # Load trades
        if self.trades_file.exists():
            try:
                with open(self.trades_file, 'r') as f:
                    data = json.load(f)
                    self._trades = [UserTrade.from_dict(t) for t in data.get("trades", [])]
            except Exception as e:
                logger.error(f"Error loading trades: {e}")

        # Load signals
        if self.signals_file.exists():
            try:
                with open(self.signals_file, 'r') as f:
                    data = json.load(f)
                    self._signals = [AdminSignal.from_dict(s) for s in data.get("signals", [])]
            except Exception as e:
                logger.error(f"Error loading signals: {e}")

    def _save_users(self):
        """Save users to file."""
        try:
            data = {"users": [u.to_dict() for u in self._users.values()]}
            with open(self.users_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving users: {e}")

    def _save_trades(self):
        """Save trades to file."""
        try:
            data = {"trades": [t.to_dict() for t in self._trades[-1000:]]}  # Keep last 1000
            with open(self.trades_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving trades: {e}")

    def _save_signals(self):
        """Save signals to file."""
        try:
            data = {"signals": [s.to_dict() for s in self._signals[-500:]]}  # Keep last 500
            with open(self.signals_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving signals: {e}")

    # ==================== User Management ====================

    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        created_by: str,
        capital: float = 100000,
        role: UserRole = UserRole.USER
    ) -> tuple[bool, str, Optional[TradingUser]]:
        """Create a new trading user."""
        # Check if username exists
        for user in self._users.values():
            if user.username.lower() == username.lower():
                return False, "Username already exists", None
            if user.email.lower() == email.lower():
                return False, "Email already exists", None

        # Create user
        user_id = secrets.token_hex(8)
        password_hash, salt = self._hash_password(password)

        user = TradingUser(
            user_id=user_id,
            username=username,
            email=email,
            password_hash=password_hash,
            salt=salt,
            role=role,
            status=UserStatus.ACTIVE if role == UserRole.ADMIN else UserStatus.PENDING,
            capital=capital,
            created_at=datetime.now().isoformat(),
            created_by=created_by,
        )

        self._users[user_id] = user
        self._save_users()

        logger.info(f"Created trading user: {username} (ID: {user_id})")
        return True, "User created successfully", user

    def authenticate_user(self, username: str, password: str) -> Optional[TradingUser]:
        """Authenticate a trading user."""
        for user in self._users.values():
            if user.username.lower() == username.lower():
                if self._verify_password(password, user.password_hash, user.salt):
                    if user.status == UserStatus.SUSPENDED:
                        return None
                    user.last_login = datetime.now().isoformat()
                    self._save_users()
                    return user
        return None

    def get_user(self, user_id: str) -> Optional[TradingUser]:
        """Get user by ID."""
        return self._users.get(user_id)

    def get_user_by_username(self, username: str) -> Optional[TradingUser]:
        """Get user by username."""
        for user in self._users.values():
            if user.username.lower() == username.lower():
                return user
        return None

    def get_all_users(self) -> List[TradingUser]:
        """Get all users."""
        return list(self._users.values())

    def update_user(self, user_id: str, updates: Dict[str, Any]) -> tuple[bool, str]:
        """Update user details."""
        user = self._users.get(user_id)
        if not user:
            return False, "User not found"

        # Update allowed fields
        if "status" in updates:
            user.status = UserStatus(updates["status"])
        if "trading_mode" in updates:
            user.trading_mode = TradingMode(updates["trading_mode"])
        if "copy_trading_mode" in updates:
            user.copy_trading_mode = CopyTradingMode(updates["copy_trading_mode"])
        if "capital" in updates:
            user.capital = float(updates["capital"])
        if "notes" in updates:
            user.notes = updates["notes"]

        # Update restrictions
        if "restrictions" in updates:
            for key, value in updates["restrictions"].items():
                if hasattr(user.restrictions, key):
                    setattr(user.restrictions, key, value)

        # Update commission
        if "commission" in updates:
            for key, value in updates["commission"].items():
                if hasattr(user.commission, key):
                    setattr(user.commission, key, value)

        self._save_users()
        return True, "User updated successfully"

    def delete_user(self, user_id: str) -> tuple[bool, str]:
        """Delete a user."""
        if user_id not in self._users:
            return False, "User not found"

        del self._users[user_id]
        self._save_users()
        return True, "User deleted successfully"

    def update_zerodha_credentials(
        self,
        user_id: str,
        api_key: str,
        api_secret: str
    ) -> tuple[bool, str]:
        """Update user's Zerodha credentials."""
        user = self._users.get(user_id)
        if not user:
            return False, "User not found"

        user.zerodha.api_key = api_key
        user.zerodha.api_secret = api_secret
        self._save_users()

        return True, "Credentials updated"

    def set_zerodha_session(
        self,
        user_id: str,
        access_token: str,
        user_zerodha_id: str,
        user_name: str
    ) -> tuple[bool, str]:
        """Set Zerodha session after OAuth."""
        user = self._users.get(user_id)
        if not user:
            return False, "User not found"

        user.zerodha.access_token = access_token
        user.zerodha.user_id = user_zerodha_id
        user.zerodha.user_name = user_name
        user.zerodha.is_connected = True
        user.zerodha.last_connected = datetime.now().isoformat()
        self._save_users()

        return True, "Zerodha connected"

    # ==================== Signal Management ====================

    def create_admin_signal(
        self,
        index: str,
        signal_type: str,
        strike: float,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        target: float,
        confidence: float,
        quality_score: float
    ) -> AdminSignal:
        """Create a new admin signal for copy trading."""
        signal = AdminSignal(
            signal_id=secrets.token_hex(8),
            timestamp=datetime.now().isoformat(),
            index=index,
            signal_type=signal_type,
            strike=strike,
            symbol=symbol,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target=target,
            confidence=confidence,
            quality_score=quality_score,
        )

        self._signals.append(signal)
        self._save_signals()

        logger.info(f"Created admin signal: {symbol} {signal_type} @ {entry_price}")
        return signal

    def get_active_signals(self) -> List[AdminSignal]:
        """Get active signals."""
        return [s for s in self._signals if s.status == "active"]

    def get_recent_signals(self, limit: int = 50) -> List[AdminSignal]:
        """Get recent signals."""
        return sorted(self._signals, key=lambda x: x.timestamp, reverse=True)[:limit]

    def close_signal(
        self,
        signal_id: str,
        exit_price: float,
        exit_reason: str
    ) -> tuple[bool, str]:
        """Close an admin signal."""
        for signal in self._signals:
            if signal.signal_id == signal_id:
                signal.status = "closed"
                signal.exit_price = exit_price
                signal.exit_time = datetime.now().isoformat()
                signal.exit_reason = exit_reason
                if signal.entry_price > 0:
                    signal.pnl_percent = ((exit_price - signal.entry_price) / signal.entry_price) * 100
                self._save_signals()
                return True, "Signal closed"
        return False, "Signal not found"

    # ==================== Trade Execution ====================

    def execute_trade_for_user(
        self,
        user_id: str,
        signal: AdminSignal,
        lots: int
    ) -> tuple[bool, str, Optional[UserTrade]]:
        """Execute a trade for a user based on admin signal."""
        user = self._users.get(user_id)
        if not user:
            return False, "User not found", None

        # Check restrictions
        if user.status != UserStatus.ACTIVE:
            return False, "User account is not active", None

        if user.trading_mode == TradingMode.DISABLED:
            return False, "Trading is disabled for this user", None

        if signal.index not in user.restrictions.allowed_indices:
            return False, f"Trading {signal.index} is not allowed", None

        if signal.signal_type not in user.restrictions.allowed_option_types:
            return False, f"Trading {signal.signal_type} is not allowed", None

        if lots > user.restrictions.max_lots_per_trade:
            return False, f"Max lots per trade is {user.restrictions.max_lots_per_trade}", None

        # Check daily limits
        today = date.today().isoformat()
        if user.stats.stats_date != today:
            user.stats.today_trades = 0
            user.stats.today_pnl = 0
            user.stats.today_commission = 0
            user.stats.stats_date = today

        if user.stats.today_trades >= user.restrictions.max_daily_trades:
            return False, "Daily trade limit reached", None

        if user.stats.today_pnl <= -user.restrictions.max_daily_loss:
            return False, "Daily loss limit reached", None

        # Calculate quantity
        lot_size = {"NIFTY": 25, "BANKNIFTY": 15, "SENSEX": 10}.get(signal.index, 25)
        quantity = lots * lot_size

        # Apply risk multiplier
        adjusted_lots = int(lots * user.restrictions.risk_multiplier)
        if adjusted_lots < 1:
            adjusted_lots = 1
        adjusted_quantity = adjusted_lots * lot_size

        # Create trade
        trade = UserTrade(
            trade_id=secrets.token_hex(8),
            user_id=user_id,
            admin_signal_id=signal.signal_id,
            index=signal.index,
            symbol=signal.symbol,
            strike=signal.strike,
            option_type=signal.signal_type,
            lots=adjusted_lots,
            quantity=adjusted_quantity,
            entry_price=signal.entry_price,
            entry_time=datetime.now().isoformat(),
            stop_loss=signal.stop_loss,
            target=signal.target,
            is_paper=user.trading_mode == TradingMode.PAPER,
        )

        self._trades.append(trade)

        # Update signal tracking
        if user_id not in signal.executed_by:
            signal.executed_by.append(user_id)

        # Update user stats
        user.stats.today_trades += 1
        user.stats.total_trades += 1

        self._save_trades()
        self._save_signals()
        self._save_users()

        logger.info(f"Executed trade for user {user.username}: {signal.symbol} x{adjusted_lots}")
        return True, "Trade executed", trade

    def close_user_trade(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str
    ) -> tuple[bool, str]:
        """Close a user's trade."""
        for trade in self._trades:
            if trade.trade_id == trade_id and trade.status == "open":
                trade.exit_price = exit_price
                trade.exit_time = datetime.now().isoformat()
                trade.exit_reason = exit_reason
                trade.status = "closed"

                # Calculate P&L
                trade.pnl = (exit_price - trade.entry_price) * trade.quantity
                if trade.entry_price > 0:
                    trade.pnl_percent = ((exit_price - trade.entry_price) / trade.entry_price) * 100

                # Apply commission
                user = self._users.get(trade.user_id)
                if user and trade.pnl > 0:
                    commission = trade.pnl * (user.commission.commission_percent / 100)
                    commission += user.commission.flat_fee_per_trade
                    trade.commission = commission
                    trade.net_pnl = trade.pnl - commission

                    # Update user stats
                    user.stats.total_pnl += trade.net_pnl
                    user.stats.total_commission_paid += commission
                    user.stats.today_pnl += trade.net_pnl
                    user.stats.today_commission += commission

                    if trade.pnl > 0:
                        user.stats.winning_trades += 1
                        user.stats.best_trade = max(user.stats.best_trade, trade.pnl)
                    else:
                        user.stats.losing_trades += 1
                        user.stats.worst_trade = min(user.stats.worst_trade, trade.pnl)

                    self._save_users()
                else:
                    trade.net_pnl = trade.pnl

                self._save_trades()
                return True, "Trade closed"

        return False, "Trade not found"

    def get_user_trades(
        self,
        user_id: str,
        status: Optional[str] = None,
        limit: int = 50
    ) -> List[UserTrade]:
        """Get trades for a user."""
        trades = [t for t in self._trades if t.user_id == user_id]
        if status:
            trades = [t for t in trades if t.status == status]
        return sorted(trades, key=lambda x: x.entry_time, reverse=True)[:limit]

    def get_user_open_trades(self, user_id: str) -> List[UserTrade]:
        """Get open trades for a user."""
        return [t for t in self._trades if t.user_id == user_id and t.status == "open"]

    # ==================== Dashboard Stats ====================

    def get_admin_dashboard_stats(self) -> Dict[str, Any]:
        """Get stats for admin dashboard."""
        total_users = len(self._users)
        active_users = len([u for u in self._users.values() if u.status == UserStatus.ACTIVE])
        live_traders = len([u for u in self._users.values() if u.trading_mode == TradingMode.LIVE])
        paper_traders = len([u for u in self._users.values() if u.trading_mode == TradingMode.PAPER])

        # Today's stats
        today = date.today().isoformat()
        today_trades = [t for t in self._trades if t.entry_time.startswith(today)]
        today_pnl = sum(t.net_pnl for t in today_trades if t.status == "closed")
        today_commission = sum(t.commission for t in today_trades if t.status == "closed")

        # Total stats
        total_commission = sum(u.stats.total_commission_paid for u in self._users.values())
        total_pnl = sum(u.stats.total_pnl for u in self._users.values())

        # Active signals
        active_signals = len(self.get_active_signals())

        return {
            "total_users": total_users,
            "active_users": active_users,
            "live_traders": live_traders,
            "paper_traders": paper_traders,
            "today_trades": len(today_trades),
            "today_pnl": today_pnl,
            "today_commission": today_commission,
            "total_commission": total_commission,
            "total_pnl": total_pnl,
            "active_signals": active_signals,
        }

    def get_user_dashboard_stats(self, user_id: str) -> Dict[str, Any]:
        """Get stats for user dashboard."""
        user = self._users.get(user_id)
        if not user:
            return {}

        open_trades = self.get_user_open_trades(user_id)
        today = date.today().isoformat()

        # Calculate unrealized P&L (would need current prices in real scenario)
        unrealized_pnl = 0  # Placeholder

        return {
            "capital": user.capital,
            "trading_mode": user.trading_mode.value,
            "copy_trading_mode": user.copy_trading_mode.value,
            "zerodha_connected": user.zerodha.is_connected,
            "open_trades": len(open_trades),
            "today_trades": user.stats.today_trades,
            "today_pnl": user.stats.today_pnl,
            "today_commission": user.stats.today_commission,
            "total_trades": user.stats.total_trades,
            "total_pnl": user.stats.total_pnl,
            "total_commission": user.stats.total_commission_paid,
            "winning_trades": user.stats.winning_trades,
            "losing_trades": user.stats.losing_trades,
            "win_rate": (user.stats.winning_trades / user.stats.total_trades * 100) if user.stats.total_trades > 0 else 0,
            "best_trade": user.stats.best_trade,
            "worst_trade": user.stats.worst_trade,
            "restrictions": asdict(user.restrictions),
            "commission_settings": asdict(user.commission),
        }


# Singleton instance
_user_trading_service: Optional[UserTradingService] = None


def get_user_trading_service() -> UserTradingService:
    """Get singleton instance."""
    global _user_trading_service
    if _user_trading_service is None:
        _user_trading_service = UserTradingService()
    return _user_trading_service
