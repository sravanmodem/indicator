"""
Settings API Routes
Handles application settings and configuration
"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from loguru import logger

from app.core.config import get_settings

router = APIRouter(prefix="/settings", tags=["Settings"])
templates = Jinja2Templates(directory="app/templates")


@router.get("", response_class=HTMLResponse)
async def settings_page(request: Request):
    """Render settings page."""
    app_settings = get_settings()

    return templates.TemplateResponse(
        "settings.html",
        {
            "request": request,
            "settings": {
                "app_host": app_settings.app_host,
                "app_port": app_settings.app_port,
                "debug": app_settings.debug,
            },
        },
    )


@router.get("/api/current")
async def get_current_settings():
    """Get current application settings."""
    app_settings = get_settings()

    return {
        "app_host": app_settings.app_host,
        "app_port": app_settings.app_port,
        "debug": app_settings.debug,
        "zerodha_api_key": app_settings.zerodha_api_key[:8] + "..." if app_settings.zerodha_api_key else None,
    }
