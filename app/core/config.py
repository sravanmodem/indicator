"""
Application Configuration
Manages all settings with secure credential handling
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with validation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Zerodha API
    kite_api_key: str = Field(default="", description="Zerodha Kite API Key")
    kite_api_secret: str = Field(default="", description="Zerodha Kite API Secret")

    # Application
    app_secret_key: str = Field(
        default="change-this-secret-key-in-production",
        description="Secret key for encryption"
    )
    app_host: str = Field(default="127.0.0.1")
    app_port: int = Field(default=8000)
    debug: bool = Field(default=False)

    # Database
    database_url: str = Field(default="sqlite+aiosqlite:///./data/trading.db")

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")
    log_file: str = Field(default="logs/app.log")

    # Trading Settings
    default_timeframe: str = Field(default="5minute")
    max_positions: int = Field(default=5)
    risk_per_trade: float = Field(default=0.01)  # 1% of capital

    # Paths
    @property
    def base_dir(self) -> Path:
        return Path(__file__).parent.parent.parent

    @property
    def data_dir(self) -> Path:
        path = self.base_dir / "data"
        path.mkdir(exist_ok=True)
        return path

    @property
    def logs_dir(self) -> Path:
        path = self.base_dir / "logs"
        path.mkdir(exist_ok=True)
        return path

    @property
    def token_file(self) -> Path:
        return self.data_dir / "tokens.enc"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Trading Constants
NIFTY_LOT_SIZE = 25
BANKNIFTY_LOT_SIZE = 15
SENSEX_LOT_SIZE = 10

# Instrument tokens (will be fetched dynamically)
NIFTY_INDEX_TOKEN = 256265
BANKNIFTY_INDEX_TOKEN = 260105
SENSEX_INDEX_TOKEN = 265  # BSE SENSEX token

# Timeframe mappings
TIMEFRAME_MAP = {
    "1minute": "minute",
    "3minute": "3minute",
    "5minute": "5minute",
    "15minute": "15minute",
    "30minute": "30minute",
    "60minute": "60minute",
    "day": "day",
}

# Indicator default parameters
INDICATOR_PARAMS = {
    "supertrend": {"period": 10, "multiplier": 3.0},
    "ema_fast": 9,
    "ema_slow": 21,
    "ema_trend": 50,
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "adx_period": 14,
    "bollinger_period": 20,
    "bollinger_std": 2.0,
    "atr_period": 14,
    "stochastic_k": 14,
    "stochastic_d": 3,
    "cci_period": 20,
    "vwap_std_bands": [1, 2],
}

# Signal thresholds
SIGNAL_THRESHOLDS = {
    "adx_trend": 25,
    "adx_strong": 40,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "rsi_neutral": 50,
    "pcr_oversold": 1.3,
    "pcr_overbought": 0.7,
    "vix_low": 12,
    "vix_normal": 20,
    "vix_high": 25,
    "vix_extreme": 35,
}
