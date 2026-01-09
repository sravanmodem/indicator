"""
Paper Trading with Multiple Exit Strategies
Three separate trading strategies with independent histories
"""

from enum import Enum
from typing import Literal


class ExitStrategy(str, Enum):
    """Exit strategy types."""
    FIXED_20_PERCENT = "fixed_20_percent"  # Page 1: Exit at 20% profit
    TRAILING_STOPLOSS = "trailing_stoploss"  # Page 2: Trailing stop loss
    PROFIT_100_HALT = "profit_100_halt"  # Page 3: Exit at 100%, then halt


# Strategy configurations
STRATEGY_CONFIGS = {
    ExitStrategy.FIXED_20_PERCENT: {
        "name": "Fixed 20% Profit",
        "description": "Exit when profit reaches 20%, no new trades after exit",
        "exit_at_profit": 20.0,  # Exit at 20% profit
        "halt_after_exit": True,  # No new trades after hitting profit target
    },
    ExitStrategy.TRAILING_STOPLOSS: {
        "name": "Trailing Stop Loss",
        "description": "Lock 20% profit when reached, trail as profit increases",
        "profit_lock_threshold": 20.0,  # Start trailing at 20%
        "trailing_enabled": True,
    },
    ExitStrategy.PROFIT_100_HALT: {
        "name": "100% Profit & Halt",
        "description": "Exit at 100% profit, then stop trading for the day",
        "exit_at_profit": 100.0,  # Exit at 100% profit
        "halt_after_exit": True,  # No new trades after hitting target
    },
}


def get_strategy_name(strategy: ExitStrategy) -> str:
    """Get display name for strategy."""
    return STRATEGY_CONFIGS[strategy]["name"]


def get_strategy_description(strategy: ExitStrategy) -> str:
    """Get description for strategy."""
    return STRATEGY_CONFIGS[strategy]["description"]
