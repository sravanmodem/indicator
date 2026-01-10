"""API routes for the trading application."""

# Import all routers for easy access
from app.api import (
    auth,
    market,
    signals,
    htmx,
    paper_trading,
    user_auth,
    email,
    admin,
    user_dashboard,
    consolidated,  # NEW: Consolidated API endpoints
)

__all__ = [
    "auth",
    "market",
    "signals",
    "htmx",
    "paper_trading",
    "user_auth",
    "email",
    "admin",
    "user_dashboard",
    "consolidated",
]
