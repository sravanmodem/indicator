"""
Email API Routes
Email settings and report endpoints
"""

from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from typing import Optional

from app.services.email_service import get_email_service
from app.services.paper_trading import get_paper_trading_service

router = APIRouter(prefix="/email", tags=["Email"])

templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))


@router.get("/settings", response_class=HTMLResponse)
async def email_settings_page(request: Request):
    """Render email settings page."""
    email_service = get_email_service()
    settings = email_service.get_settings()
    report_history = email_service.get_report_history()

    return templates.TemplateResponse(
        "email_settings.html",
        {
            "request": request,
            "settings": settings,
            "report_history": report_history,
            "message": None,
            "error": None
        }
    )


@router.post("/settings")
async def update_email_settings(
    request: Request,
    enabled: Optional[str] = Form(None),
    smtp_server: str = Form(""),
    smtp_port: int = Form(587),
    smtp_username: str = Form(""),
    smtp_password: str = Form(""),
    sender_email: str = Form(""),
    recipient_email: str = Form(""),
    use_tls: Optional[str] = Form(None),
    send_daily_report: Optional[str] = Form(None),
    send_monthly_report: Optional[str] = Form(None),
    send_yearly_report: Optional[str] = Form(None),
    daily_report_time: str = Form("18:00"),
):
    """Update email settings."""
    email_service = get_email_service()

    success, message = email_service.update_settings(
        enabled=enabled == "on",
        smtp_server=smtp_server,
        smtp_port=smtp_port,
        smtp_username=smtp_username,
        smtp_password=smtp_password if smtp_password != "***" else None,
        sender_email=sender_email,
        recipient_email=recipient_email,
        use_tls=use_tls == "on",
        send_daily_report=send_daily_report == "on",
        send_monthly_report=send_monthly_report == "on",
        send_yearly_report=send_yearly_report == "on",
        daily_report_time=daily_report_time,
    )

    settings = email_service.get_settings()
    report_history = email_service.get_report_history()

    return templates.TemplateResponse(
        "email_settings.html",
        {
            "request": request,
            "settings": settings,
            "report_history": report_history,
            "message": message if success else None,
            "error": None if success else message
        }
    )


@router.post("/test-connection")
async def test_connection():
    """Test SMTP connection."""
    email_service = get_email_service()
    success, message = email_service.test_connection()
    return {"success": success, "message": message}


@router.post("/test-email")
async def send_test_email():
    """Send a test email."""
    email_service = get_email_service()
    success, message = email_service.send_test_email()
    return {"success": success, "message": message}


@router.post("/send-report/{report_type}")
async def send_report(report_type: str):
    """Send a P&L report immediately."""
    if report_type not in ['daily', 'monthly', 'yearly']:
        return {"success": False, "message": f"Invalid report type: {report_type}"}

    email_service = get_email_service()
    paper_trading_service = get_paper_trading_service()

    success, message = email_service.send_report_now(report_type, paper_trading_service)
    return {"success": success, "message": message}


@router.get("/report-history")
async def get_report_history():
    """Get history of sent reports."""
    email_service = get_email_service()
    return {"reports": email_service.get_report_history()}


@router.get("/api/settings")
async def get_email_settings_api():
    """Get email settings as JSON."""
    email_service = get_email_service()
    return email_service.get_settings()
