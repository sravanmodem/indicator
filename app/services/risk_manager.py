"""
Risk Management System
Manages position sizing, risk limits, and profit targets
Ensures safe trading with defined risk parameters for 25k daily profit target
"""

from dataclasses import dataclass
from datetime import datetime, date
from typing import Any

from loguru import logger

from app.services.signal_engine import TradeSignal


@dataclass
class RiskParameters:
    """Risk management parameters."""
    # Capital settings
    total_capital: float  # Total trading capital
    daily_profit_target: float  # Target profit for the day (25,000)

    # Risk limits
    max_risk_per_trade_pct: float = 2.0  # Max 2% risk per trade
    max_daily_loss_pct: float = 3.0  # Max 3% loss per day
    max_positions: int = 5  # Max concurrent positions

    # Position sizing
    min_risk_reward: float = 1.5  # Minimum R:R ratio
    position_size_method: str = "fixed_risk"  # fixed_risk or kelly

    # Stop loss settings
    trailing_stop_enabled: bool = True
    trailing_stop_trigger: float = 1.5  # Move to BE after 1.5R profit

    # Target management
    partial_exit_at_1r: float = 0.5  # Exit 50% at 1R
    move_sl_to_be_at_1r: bool = True  # Move SL to breakeven at 1R


@dataclass
class PositionSize:
    """Calculated position size."""
    quantity: int  # Number of contracts/lots
    risk_amount: float  # Risk amount in rupees
    potential_profit: float  # Potential profit at target
    risk_reward_ratio: float  # Actual R:R ratio
    stop_loss_distance: float  # Distance to SL in rupees
    capital_allocation_pct: float  # Percentage of capital used


@dataclass
class DailyStats:
    """Daily trading statistics."""
    date: date
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_pnl: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    active_positions: int = 0
    target_reached: bool = False
    max_loss_hit: bool = False


class RiskManager:
    """
    Manages all risk aspects of trading.

    Features:
    - Position sizing based on risk parameters
    - Daily P&L tracking
    - Profit target monitoring (25k daily target)
    - Loss limit enforcement (3% max daily loss)
    - Position limit management
    - Risk/Reward validation
    """

    def __init__(self, params: RiskParameters):
        """Initialize risk manager with parameters."""
        self.params = params
        self.daily_stats: dict[date, DailyStats] = {}
        self.active_positions: list[dict[str, Any]] = []

    def calculate_position_size(
        self,
        signal: TradeSignal,
        entry_price: float,
        stop_loss: float,
    ) -> PositionSize | None:
        """
        Calculate optimal position size based on risk parameters.

        Args:
            signal: TradeSignal with targets and stops
            entry_price: Entry price for the option
            stop_loss: Stop loss price

        Returns:
            PositionSize or None if trade doesn't meet criteria
        """
        try:
            # Validate signal quality
            if signal.confidence < 60:
                logger.warning("Signal confidence too low for position sizing")
                return None

            # Check daily limits
            today = date.today()
            stats = self._get_daily_stats(today)

            if stats.max_loss_hit:
                logger.warning("Daily max loss limit hit - No new positions")
                return None

            if stats.target_reached:
                logger.info("Daily profit target reached - No new positions")
                return None

            if stats.active_positions >= self.params.max_positions:
                logger.warning(f"Max positions ({self.params.max_positions}) reached")
                return None

            # Calculate risk per contract
            stop_loss_distance = abs(entry_price - stop_loss)
            if stop_loss_distance <= 0:
                logger.error("Invalid stop loss distance")
                return None

            # Calculate maximum risk amount for this trade
            max_risk_amount = self.params.total_capital * (self.params.max_risk_per_trade_pct / 100)

            # Adjust risk based on remaining daily risk budget
            remaining_daily_risk = self._calculate_remaining_daily_risk(stats)
            if remaining_daily_risk <= 0:
                logger.warning("No remaining daily risk budget")
                return None

            # Use the lower of trade risk or remaining daily risk
            trade_risk_amount = min(max_risk_amount, remaining_daily_risk)

            # Calculate number of contracts based on risk
            # For options: risk_per_contract = stop_loss_distance
            quantity = int(trade_risk_amount / stop_loss_distance)

            # Ensure minimum position size
            if quantity < 1:
                logger.warning(f"Position size too small: {quantity}")
                return None

            # Calculate potential profit
            target_distance = abs(signal.target_1 - entry_price) if signal.target_1 else stop_loss_distance * 2
            potential_profit = quantity * target_distance

            # Calculate actual R:R
            actual_rr = target_distance / stop_loss_distance if stop_loss_distance > 0 else 0

            # Validate R:R
            if actual_rr < self.params.min_risk_reward:
                logger.warning(f"R:R {actual_rr:.2f} below minimum {self.params.min_risk_reward}")
                return None

            # Calculate capital allocation
            position_cost = quantity * entry_price
            capital_allocation_pct = (position_cost / self.params.total_capital) * 100

            # Limit capital allocation to 20% per position
            if capital_allocation_pct > 20:
                logger.warning(f"Position size would use {capital_allocation_pct:.1f}% of capital (max 20%)")
                # Adjust quantity
                max_quantity = int((self.params.total_capital * 0.20) / entry_price)
                quantity = max_quantity
                position_cost = quantity * entry_price
                capital_allocation_pct = (position_cost / self.params.total_capital) * 100
                potential_profit = quantity * target_distance
                trade_risk_amount = quantity * stop_loss_distance

            logger.info(f"Position size calculated: {quantity} contracts, Risk: ₹{trade_risk_amount:.0f}, Potential: ₹{potential_profit:.0f}")

            return PositionSize(
                quantity=quantity,
                risk_amount=trade_risk_amount,
                potential_profit=potential_profit,
                risk_reward_ratio=actual_rr,
                stop_loss_distance=stop_loss_distance,
                capital_allocation_pct=capital_allocation_pct,
            )

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return None

    def _calculate_remaining_daily_risk(self, stats: DailyStats) -> float:
        """Calculate remaining risk budget for the day."""
        max_daily_loss = self.params.total_capital * (self.params.max_daily_loss_pct / 100)

        # Current daily loss
        current_loss = abs(stats.net_pnl) if stats.net_pnl < 0 else 0

        # Remaining risk
        remaining = max_daily_loss - current_loss

        return max(remaining, 0)

    def can_take_trade(self, signal: TradeSignal) -> tuple[bool, str]:
        """
        Check if a trade can be taken based on risk limits.

        Args:
            signal: TradeSignal to validate

        Returns:
            Tuple of (can_trade: bool, reason: str)
        """
        today = date.today()
        stats = self._get_daily_stats(today)

        # Check daily loss limit
        if stats.max_loss_hit:
            return False, "Daily loss limit reached"

        # Check profit target
        if stats.target_reached:
            return False, "Daily profit target reached - No new trades"

        # Check position limit
        if stats.active_positions >= self.params.max_positions:
            return False, f"Max positions ({self.params.max_positions}) reached"

        # Check signal quality
        if signal.confidence < 60:
            return False, f"Signal confidence too low ({signal.confidence:.1f}% < 60%)"

        # Check R:R
        if signal.risk_reward < self.params.min_risk_reward:
            return False, f"R:R too low ({signal.risk_reward:.2f} < {self.params.min_risk_reward})"

        # Check if option is available
        if not signal.recommended_option:
            return False, "No recommended option available"

        return True, "Trade allowed"

    def update_trade_result(
        self,
        profit_loss: float,
        trade_type: str = "win",
    ) -> None:
        """
        Update daily statistics with trade result.

        Args:
            profit_loss: P&L amount (positive for profit, negative for loss)
            trade_type: "win" or "loss"
        """
        today = date.today()
        stats = self._get_daily_stats(today)

        stats.total_trades += 1

        if profit_loss > 0:
            stats.winning_trades += 1
            stats.gross_profit += profit_loss
            stats.largest_win = max(stats.largest_win, profit_loss)
        else:
            stats.losing_trades += 1
            stats.gross_loss += abs(profit_loss)
            stats.largest_loss = max(stats.largest_loss, abs(profit_loss))

        stats.net_pnl += profit_loss

        # Calculate derived metrics
        if stats.total_trades > 0:
            stats.win_rate = (stats.winning_trades / stats.total_trades) * 100

        if stats.gross_loss > 0:
            stats.profit_factor = stats.gross_profit / stats.gross_loss
        else:
            stats.profit_factor = stats.gross_profit if stats.gross_profit > 0 else 0

        # Check if profit target reached
        if stats.net_pnl >= self.params.daily_profit_target:
            stats.target_reached = True
            logger.info(f"🎯 Daily profit target reached! P&L: ₹{stats.net_pnl:,.0f}")

        # Check if max loss hit
        max_loss = self.params.total_capital * (self.params.max_daily_loss_pct / 100)
        if abs(stats.net_pnl) >= max_loss and stats.net_pnl < 0:
            stats.max_loss_hit = True
            logger.warning(f"⚠️ Daily loss limit hit! P&L: ₹{stats.net_pnl:,.0f}")

        self.daily_stats[today] = stats

    def add_active_position(self, position: dict[str, Any]) -> None:
        """Add a position to active tracking."""
        today = date.today()
        stats = self._get_daily_stats(today)

        self.active_positions.append(position)
        stats.active_positions = len(self.active_positions)

        logger.info(f"Active positions: {stats.active_positions}/{self.params.max_positions}")

    def remove_active_position(self, order_id: str) -> None:
        """Remove a position from active tracking."""
        today = date.today()
        stats = self._get_daily_stats(today)

        self.active_positions = [p for p in self.active_positions if p.get("order_id") != order_id]
        stats.active_positions = len(self.active_positions)

        logger.info(f"Active positions: {stats.active_positions}/{self.params.max_positions}")

    def get_daily_summary(self, trade_date: date | None = None) -> DailyStats:
        """Get daily trading summary."""
        if trade_date is None:
            trade_date = date.today()

        return self._get_daily_stats(trade_date)

    def _get_daily_stats(self, trade_date: date) -> DailyStats:
        """Get or create daily stats for a date."""
        if trade_date not in self.daily_stats:
            self.daily_stats[trade_date] = DailyStats(date=trade_date)

        return self.daily_stats[trade_date]

    def calculate_capital_required(self, target_daily_profit: float) -> float:
        """
        Calculate capital required for target daily profit.

        For 25k daily profit with 2% risk per trade and 1.5:1 R:R minimum:
        - If win rate is 50%, need ~4-5 winning trades
        - Each winning trade should make ~6k-7k
        - With 1.5:1 R:R, risk per trade = 4k-5k
        - At 2% risk, capital = 200k-250k

        Returns:
            Recommended minimum capital
        """
        # Conservative estimate with 50% win rate
        win_rate = 0.50
        avg_rr = 2.0  # Average R:R ratio

        # Expected value per trade = (win_rate * reward) - (lose_rate * risk)
        # For positive expectancy, we need multiple winners

        # Target profit per winning trade
        trades_needed = 1 / win_rate  # If 50% win rate, need 2 trades on average for 1 win
        profit_per_winning_trade = target_daily_profit / win_rate  # 25k / 0.50 = 50k total

        # With 2:1 R:R, risk per trade = profit / 2
        risk_per_trade = profit_per_winning_trade / (avg_rr * 2)

        # Capital = risk_per_trade / risk_percentage
        required_capital = risk_per_trade / (self.params.max_risk_per_trade_pct / 100)

        logger.info(f"For ₹{target_daily_profit:,.0f} daily target: Recommended capital ₹{required_capital:,.0f}")

        return required_capital


# Create default risk parameters for 25k daily profit
def create_default_risk_params(capital: float = 500000) -> RiskParameters:
    """
    Create default risk parameters for 25k daily profit target.

    Args:
        capital: Total trading capital (default 5 lakh for 25k target)

    Returns:
        RiskParameters configured for 25k daily target
    """
    return RiskParameters(
        total_capital=capital,
        daily_profit_target=25000,  # 25k target
        max_risk_per_trade_pct=2.0,  # 2% max risk per trade
        max_daily_loss_pct=3.0,  # 3% max daily loss
        max_positions=5,  # Max 5 concurrent positions
        min_risk_reward=1.5,  # Minimum 1.5:1 R:R
        position_size_method="fixed_risk",
        trailing_stop_enabled=True,
        trailing_stop_trigger=1.5,
        partial_exit_at_1r=0.5,
        move_sl_to_be_at_1r=True,
    )


# Singleton instance
_risk_manager: RiskManager | None = None


def get_risk_manager(params: RiskParameters | None = None) -> RiskManager:
    """Get or create risk manager instance."""
    global _risk_manager
    if _risk_manager is None:
        if params is None:
            params = create_default_risk_params()
        _risk_manager = RiskManager(params)
    return _risk_manager
