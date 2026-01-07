"""
Toast Notification System for Backend Integration
Allows FastAPI endpoints to send toast notifications via response headers
"""

from starlette.responses import Response
from typing import Literal


class ToastResponse(Response):
    """
    Custom response class that automatically includes toast notification headers.

    Usage in FastAPI endpoints:
        @app.post("/api/trade")
        async def place_trade(request: Request):
            # Do something...
            return ToastResponse(
                "Trade placed successfully!",
                toast_message="Trade executed for NIFTY 23000 CE",
                toast_type="success"
            )
    """

    def __init__(
        self,
        content=None,
        status_code: int = 200,
        headers: dict = None,
        toast_message: str = None,
        toast_type: Literal["success", "error", "warning", "info"] = "info",
        toast_duration: int = 5000,
        media_type: str = None,
    ):
        super().__init__(content, status_code, headers, media_type)

        # Add toast headers if message provided
        if toast_message:
            self.headers["X-Toast-Message"] = toast_message
            self.headers["X-Toast-Type"] = toast_type
            self.headers["X-Toast-Duration"] = str(toast_duration)


class HTMLToastResponse(Response):
    """
    HTML response with toast notification support.
    """

    def __init__(
        self,
        content: str,
        status_code: int = 200,
        headers: dict = None,
        toast_message: str = None,
        toast_type: Literal["success", "error", "warning", "info"] = "info",
        toast_duration: int = 5000,
    ):
        super().__init__(
            content=content,
            status_code=status_code,
            headers=headers,
            media_type="text/html"
        )

        # Add toast headers if message provided
        if toast_message:
            self.headers["X-Toast-Message"] = toast_message
            self.headers["X-Toast-Type"] = toast_type
            self.headers["X-Toast-Duration"] = str(toast_duration)


class JSONToastResponse(Response):
    """
    JSON response with toast notification support.
    """

    def __init__(
        self,
        content: dict,
        status_code: int = 200,
        headers: dict = None,
        toast_message: str = None,
        toast_type: Literal["success", "error", "warning", "info"] = "info",
        toast_duration: int = 5000,
    ):
        import json

        super().__init__(
            content=json.dumps(content),
            status_code=status_code,
            headers=headers,
            media_type="application/json"
        )

        # Add toast headers if message provided
        if toast_message:
            self.headers["X-Toast-Message"] = toast_message
            self.headers["X-Toast-Type"] = toast_type
            self.headers["X-Toast-Duration"] = str(toast_duration)


def add_toast_to_response(
    response: Response,
    message: str,
    toast_type: Literal["success", "error", "warning", "info"] = "info",
    duration: int = 5000,
) -> Response:
    """
    Add toast notification headers to an existing response.

    Usage:
        response = FileResponse("path/to/file.html")
        response = add_toast_to_response(
            response,
            message="File downloaded successfully!",
            toast_type="success"
        )
        return response
    """
    response.headers["X-Toast-Message"] = message
    response.headers["X-Toast-Type"] = toast_type
    response.headers["X-Toast-Duration"] = str(duration)
    return response
