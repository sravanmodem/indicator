"""
Authentication API Routes
Handles Zerodha login flow and session management
"""

from fastapi import APIRouter, Request, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from loguru import logger

from app.services.zerodha_auth import get_auth_service

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.get("/login")
async def login(request: Request):
    """
    Initiate Zerodha login.
    Redirects to Zerodha login page.
    """
    auth = get_auth_service()
    login_url = auth.get_login_url()
    return RedirectResponse(url=login_url)


@router.get("/callback")
async def auth_callback(
    request: Request,
    request_token: str = Query(..., description="Token from Zerodha redirect"),
    status: str = Query(default="success"),
):
    """
    Handle Zerodha OAuth callback.
    Exchange request_token for access_token.
    """
    if status != "success":
        logger.error(f"Auth callback failed with status: {status}")
        return RedirectResponse(url="/?error=auth_failed")

    auth = get_auth_service()
    result = await auth.authenticate_with_request_token(request_token)

    if result["success"]:
        logger.info(f"User authenticated: {result['user_name']}")
        return RedirectResponse(url="/dashboard")
    else:
        logger.error(f"Authentication failed: {result.get('error')}")
        return RedirectResponse(url=f"/?error={result.get('error', 'unknown')}")


@router.get("/status")
async def auth_status():
    """Get current authentication status."""
    auth = get_auth_service()
    return auth.get_auth_status()


@router.post("/logout")
async def logout():
    """Logout and clear session."""
    auth = get_auth_service()
    success = await auth.logout()
    return {"success": success}


@router.get("/restore")
async def restore_session():
    """Try to restore session from stored tokens."""
    auth = get_auth_service()
    success = await auth.restore_session()
    return {"success": success, "status": auth.get_auth_status()}


# Zerodha callback route - matches the redirect URL configured in Kite app
zerodha_router = APIRouter(prefix="/zerodha", tags=["Zerodha"])


@zerodha_router.get("/callback")
async def zerodha_callback(
    request: Request,
    request_token: str = Query(None, description="Token from Zerodha redirect"),
    status: str = Query(default="success"),
    action: str = Query(default=None),
):
    """
    Handle Zerodha OAuth callback at /zerodha/callback
    This is the redirect URL configured in Kite Connect app.
    """
    logger.info(f"Zerodha callback received - status: {status}, token: {request_token[:10] if request_token else 'None'}...")

    if status != "success" or not request_token:
        logger.error(f"Auth callback failed with status: {status}")
        return RedirectResponse(url="/?error=auth_failed")

    auth = get_auth_service()
    result = await auth.authenticate_with_request_token(request_token)

    if result["success"]:
        logger.info(f"User authenticated: {result['user_name']}")
        return RedirectResponse(url="/dashboard")
    else:
        logger.error(f"Authentication failed: {result.get('error')}")
        return RedirectResponse(url=f"/?error={result.get('error', 'unknown')}")
