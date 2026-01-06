"""
Main FastAPI Application
Institutional-Grade NIFTY Options Indicator System
"""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger

from app.api import auth, market, signals, htmx, settings, paper_trading, auto_trade
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.services.zerodha_auth import get_auth_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    setup_logging()
    logger.info("Starting NIFTY Options Indicator System...")

    # Try to restore session from stored tokens
    auth_service = get_auth_service()
    if await auth_service.restore_session():
        logger.info("Session restored from stored tokens")
    else:
        logger.info("No valid stored session, login required")

    yield

    # Shutdown
    logger.info("Shutting down...")
    from app.services.websocket_manager import get_ws_manager
    ws = get_ws_manager()
    await ws.disconnect()


# Create FastAPI app
app = FastAPI(
    title="NIFTY Options Indicator",
    description="Institutional-Grade CE/PE Trading Signals for NIFTY Options",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = Path(__file__).parent / "static"
static_path.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Templates
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# Include routers
app.include_router(auth.router)
app.include_router(auth.zerodha_router)  # /zerodha/callback route
app.include_router(market.router)
app.include_router(signals.router)
app.include_router(htmx.router)
app.include_router(settings.router)
app.include_router(paper_trading.router)
app.include_router(auto_trade.router)
app.include_router(auto_trade.get_htmx_router())


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render home page."""
    auth_service = get_auth_service()
    status = auth_service.get_auth_status()

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "auth_status": status,
            "settings": get_settings(),
        },
    )


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Render main dashboard."""
    auth_service = get_auth_service()

    if not auth_service.is_authenticated:
        return RedirectResponse(url="/")

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "user": auth_service.user_profile,
        },
    )


@app.get("/history", response_class=HTMLResponse)
async def signal_history(request: Request):
    """Render signals history page."""
    auth_service = get_auth_service()

    if not auth_service.is_authenticated:
        return RedirectResponse(url="/")

    from app.services.signal_history_service import get_history_service

    history_service = get_history_service()
    stats = history_service.get_statistics(days=30)

    return templates.TemplateResponse(
        "history.html",
        {
            "request": request,
            "user": auth_service.user_profile,
            "stats": stats,
        },
    )


@app.get("/settings", response_class=HTMLResponse)
async def trading_settings(request: Request):
    """Render trading settings page."""
    auth_service = get_auth_service()

    if not auth_service.is_authenticated:
        return RedirectResponse(url="/")

    from app.services.signal_history_service import get_history_service

    history_service = get_history_service()
    settings = history_service.get_settings()

    return templates.TemplateResponse(
        "settings.html",
        {
            "request": request,
            "user": auth_service.user_profile,
            "settings": settings,
        },
    )


@app.get("/nifty", response_class=HTMLResponse)
async def nifty_page(request: Request):
    """Render NIFTY index page."""
    auth_service = get_auth_service()

    if not auth_service.is_authenticated:
        return RedirectResponse(url="/")

    return templates.TemplateResponse(
        "index_trading.html",
        {
            "request": request,
            "user": auth_service.user_profile,
            "index_name": "NIFTY",
            "index_color": "blue",
        },
    )


@app.get("/banknifty", response_class=HTMLResponse)
async def banknifty_page(request: Request):
    """Render BANKNIFTY index page."""
    auth_service = get_auth_service()

    if not auth_service.is_authenticated:
        return RedirectResponse(url="/")

    return templates.TemplateResponse(
        "index_trading.html",
        {
            "request": request,
            "user": auth_service.user_profile,
            "index_name": "BANKNIFTY",
            "index_color": "purple",
        },
    )


@app.get("/sensex", response_class=HTMLResponse)
async def sensex_page(request: Request):
    """Render SENSEX index page."""
    auth_service = get_auth_service()

    if not auth_service.is_authenticated:
        return RedirectResponse(url="/")

    return templates.TemplateResponse(
        "index_trading.html",
        {
            "request": request,
            "user": auth_service.user_profile,
            "index_name": "SENSEX",
            "index_color": "green",
        },
    )


@app.get("/auto-trade", response_class=HTMLResponse)
async def auto_trade_config(request: Request):
    """Render auto-trade configuration page."""
    auth_service = get_auth_service()

    if not auth_service.is_authenticated:
        return RedirectResponse(url="/")

    from app.services.signal_history_service import get_history_service

    history_service = get_history_service()
    settings = history_service.get_settings()

    return templates.TemplateResponse(
        "auto_trade.html",
        {
            "request": request,
            "user": auth_service.user_profile,
            "settings": settings,
        },
    )


@app.get("/paper-trading", response_class=HTMLResponse)
async def paper_trading_page(request: Request):
    """Render paper trading page."""
    auth_service = get_auth_service()

    if not auth_service.is_authenticated:
        return RedirectResponse(url="/")

    return templates.TemplateResponse(
        "paper_trading.html",
        {
            "request": request,
            "user": auth_service.user_profile,
        },
    )


@app.get("/terminal", response_class=HTMLResponse)
async def trading_terminal(request: Request):
    """Render trading terminal page."""
    auth_service = get_auth_service()

    if not auth_service.is_authenticated:
        return RedirectResponse(url="/")

    return templates.TemplateResponse(
        "terminal.html",
        {
            "request": request,
            "user": auth_service.user_profile,
        },
    )


@app.get("/option-chain", response_class=HTMLResponse)
async def option_chain_page(request: Request):
    """Render option chain page."""
    auth_service = get_auth_service()

    if not auth_service.is_authenticated:
        return RedirectResponse(url="/")

    return templates.TemplateResponse(
        "option_chain.html",
        {
            "request": request,
            "user": auth_service.user_profile,
        },
    )


@app.get("/signals", response_class=HTMLResponse)
async def signals_page(request: Request):
    """Render signals page."""
    auth_service = get_auth_service()

    if not auth_service.is_authenticated:
        return RedirectResponse(url="/")

    return templates.TemplateResponse(
        "history.html",
        {
            "request": request,
            "user": auth_service.user_profile,
        },
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    auth_service = get_auth_service()
    return {
        "status": "healthy",
        "authenticated": auth_service.is_authenticated,
    }


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
    )
