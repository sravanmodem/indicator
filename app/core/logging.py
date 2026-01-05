"""
Logging Configuration
Professional-grade logging setup with rotation and formatting
"""

import sys
from pathlib import Path

from loguru import logger

from app.core.config import get_settings


def setup_logging() -> None:
    """Configure application logging."""
    settings = get_settings()

    # Remove default handler
    logger.remove()

    # Console handler with color
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
        backtrace=True,
        diagnose=settings.debug,
    )

    # File handler with rotation
    log_file = settings.logs_dir / "app.log"
    logger.add(
        log_file,
        level=settings.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation="10 MB",
        retention="30 days",
        compression="gz",
        backtrace=True,
        diagnose=False,
    )

    # Trading signals log (separate file)
    signals_log = settings.logs_dir / "signals.log"
    logger.add(
        signals_log,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
        rotation="1 day",
        retention="90 days",
        filter=lambda record: "signal" in record["extra"],
    )

    logger.info("Logging configured successfully")


def log_signal(
    signal_type: str,
    instrument: str,
    action: str,
    price: float,
    indicators: dict,
    confidence: float,
) -> None:
    """Log trading signal with context."""
    logger.bind(signal=True).info(
        f"{signal_type} | {instrument} | {action} | Price: {price:.2f} | "
        f"Confidence: {confidence:.1%} | Indicators: {indicators}"
    )
