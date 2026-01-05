#!/usr/bin/env python3
"""
Run the NIFTY Options Indicator application.
"""

import sys
import uvicorn
from app.core.config import get_settings

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def main():
    """Start the application."""
    settings = get_settings()

    print("""
    +===========================================================+
    |     NIFTY OPTIONS INDICATOR - Institutional Grade         |
    |                CE/PE Trading Signals                      |
    +===========================================================+
    """)

    print(f"Starting server at http://{settings.app_host}:{settings.app_port}")
    print("Press Ctrl+C to stop\n")

    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
        log_level="info" if settings.debug else "warning",
    )


if __name__ == "__main__":
    main()
