"""
User Authentication API Routes
Login, logout, and password management endpoints
"""

from fastapi import APIRouter, Request, Response, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from app.services.user_auth import get_user_auth_service

router = APIRouter(prefix="/user", tags=["User Auth"])

templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))

SESSION_COOKIE_NAME = "session_token"


def get_current_user(request: Request):
    """Get current user from session cookie."""
    session_token = request.cookies.get(SESSION_COOKIE_NAME)
    if not session_token:
        return None

    auth_service = get_user_auth_service()
    return auth_service.validate_session(session_token)


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: str = None, message: str = None):
    """Render login page."""
    # Check if already logged in
    user = get_current_user(request)
    if user:
        return RedirectResponse(url="/", status_code=302)

    return templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "error": error,
            "message": message
        }
    )


@router.post("/login")
async def login(
    request: Request,
    response: Response,
    username: str = Form(...),
    password: str = Form(...)
):
    """Handle login form submission."""
    auth_service = get_user_auth_service()
    session_token = auth_service.authenticate(username, password)

    if not session_token:
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "error": "Invalid username or password"
            }
        )

    # Set session cookie
    redirect = RedirectResponse(url="/", status_code=302)
    redirect.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_token,
        httponly=True,
        max_age=86400,  # 24 hours
        samesite="lax"
    )

    return redirect


@router.get("/logout")
async def logout(request: Request):
    """Handle logout."""
    session_token = request.cookies.get(SESSION_COOKIE_NAME)
    if session_token:
        auth_service = get_user_auth_service()
        auth_service.logout(session_token)

    redirect = RedirectResponse(url="/user/login", status_code=302)
    redirect.delete_cookie(SESSION_COOKIE_NAME)

    return redirect


@router.get("/profile", response_class=HTMLResponse)
async def profile_page(request: Request):
    """Render user profile page for changing credentials."""
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/user/login", status_code=302)

    auth_service = get_user_auth_service()
    user_info = auth_service.get_user_info(user["username"])

    return templates.TemplateResponse(
        "profile.html",
        {
            "request": request,
            "user": user_info,
            "message": None,
            "error": None
        }
    )


@router.post("/change-password")
async def change_password(
    request: Request,
    current_password: str = Form(...),
    new_password: str = Form(...),
    confirm_password: str = Form(...)
):
    """Handle password change."""
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/user/login", status_code=302)

    auth_service = get_user_auth_service()
    user_info = auth_service.get_user_info(user["username"])

    if new_password != confirm_password:
        return templates.TemplateResponse(
            "profile.html",
            {
                "request": request,
                "user": user_info,
                "error": "New passwords do not match",
                "message": None
            }
        )

    success, message = auth_service.change_password(
        user["username"],
        current_password,
        new_password
    )

    if success:
        return templates.TemplateResponse(
            "profile.html",
            {
                "request": request,
                "user": auth_service.get_user_info(user["username"]),
                "message": message,
                "error": None
            }
        )
    else:
        return templates.TemplateResponse(
            "profile.html",
            {
                "request": request,
                "user": user_info,
                "error": message,
                "message": None
            }
        )


@router.post("/change-username")
async def change_username(
    request: Request,
    new_username: str = Form(...),
    password: str = Form(...)
):
    """Handle username change."""
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/user/login", status_code=302)

    auth_service = get_user_auth_service()

    success, message = auth_service.change_username(
        user["username"],
        new_username,
        password
    )

    user_info = auth_service.get_user_info(new_username if success else user["username"])

    if success:
        return templates.TemplateResponse(
            "profile.html",
            {
                "request": request,
                "user": user_info,
                "message": message,
                "error": None
            }
        )
    else:
        return templates.TemplateResponse(
            "profile.html",
            {
                "request": request,
                "user": user_info,
                "error": message,
                "message": None
            }
        )
