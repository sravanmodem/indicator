"""
Zerodha Authentication Service
Handles login flow, token generation, and session management
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import urlencode

from kiteconnect import KiteConnect
from loguru import logger

from app.core.config import get_settings
from app.core.security import get_token_manager


class ZerodhaAuthService:
    """
    Manages Zerodha Kite authentication lifecycle.

    Flow:
    1. Generate login URL
    2. User logs in and gets request_token via redirect
    3. Exchange request_token for access_token
    4. Store encrypted tokens for reuse
    5. Auto-refresh on startup if tokens valid
    """

    def __init__(self):
        self.settings = get_settings()
        self.token_manager = get_token_manager()
        self._kite: KiteConnect | None = None
        self._user_profile: dict | None = None
        self._is_authenticated = False

    @property
    def kite(self) -> KiteConnect:
        """Get or create KiteConnect instance."""
        if self._kite is None:
            self._kite = KiteConnect(api_key=self.settings.kite_api_key)
        return self._kite

    @property
    def is_authenticated(self) -> bool:
        """Check if we have valid authentication."""
        return self._is_authenticated and self._kite is not None

    @property
    def user_profile(self) -> dict | None:
        """Get cached user profile."""
        return self._user_profile

    def get_login_url(self, redirect_url: str | None = None) -> str:
        """
        Generate Zerodha login URL.

        Args:
            redirect_url: URL to redirect after login (optional)

        Returns:
            Login URL string
        """
        login_url = self.kite.login_url()
        logger.info(f"Generated login URL: {login_url}")
        return login_url

    async def authenticate_with_request_token(self, request_token: str) -> dict[str, Any]:
        """
        Exchange request token for access token.

        Args:
            request_token: Token received from Zerodha redirect

        Returns:
            Dictionary with user data and status
        """
        try:
            logger.info("Exchanging request token for access token...")

            # Generate session
            data = self.kite.generate_session(
                request_token=request_token,
                api_secret=self.settings.kite_api_secret,
            )

            access_token = data["access_token"]
            user_id = data.get("user_id", "")

            # Set access token on kite instance
            self.kite.set_access_token(access_token)

            # Fetch user profile
            self._user_profile = self.kite.profile()
            self._is_authenticated = True

            # Calculate expiry (tokens valid until next trading day 6 AM)
            now = datetime.now()
            if now.hour >= 6:
                expires_at = (now + timedelta(days=1)).replace(
                    hour=6, minute=0, second=0, microsecond=0
                )
            else:
                expires_at = now.replace(hour=6, minute=0, second=0, microsecond=0)

            # Store tokens securely
            self.token_manager.save_tokens(
                access_token=access_token,
                request_token=request_token,
                user_id=user_id,
                expires_at=expires_at,
            )

            logger.info(f"Successfully authenticated user: {user_id}")
            logger.info(f"User: {self._user_profile.get('user_name', 'Unknown')}")

            return {
                "success": True,
                "user_id": user_id,
                "user_name": self._user_profile.get("user_name"),
                "email": self._user_profile.get("email"),
                "broker": self._user_profile.get("broker"),
                "exchanges": self._user_profile.get("exchanges", []),
                "products": self._user_profile.get("products", []),
            }

        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            self._is_authenticated = False
            return {
                "success": False,
                "error": str(e),
            }

    async def restore_session(self) -> bool:
        """
        Try to restore session from stored tokens.

        Returns:
            True if session restored successfully
        """
        try:
            tokens = self.token_manager.load_tokens()
            if not tokens or not tokens.get("access_token"):
                logger.info("No stored tokens available")
                return False

            access_token = tokens["access_token"]
            self.kite.set_access_token(access_token)

            # Verify token is still valid by fetching profile
            self._user_profile = self.kite.profile()
            self._is_authenticated = True

            logger.info(
                f"Session restored for user: {tokens.get('user_id')} "
                f"({self._user_profile.get('user_name')})"
            )
            return True

        except Exception as e:
            logger.warning(f"Failed to restore session: {e}")
            self.token_manager.clear_tokens()
            self._is_authenticated = False
            return False

    async def logout(self) -> bool:
        """
        Logout and clear stored tokens.

        Returns:
            True if logout successful
        """
        try:
            if self._is_authenticated:
                try:
                    self.kite.invalidate_access_token()
                except Exception:
                    pass  # Token might already be invalid

            self.token_manager.clear_tokens()
            self._kite = None
            self._user_profile = None
            self._is_authenticated = False

            logger.info("Logged out successfully")
            return True

        except Exception as e:
            logger.error(f"Logout error: {e}")
            return False

    def get_auth_status(self) -> dict[str, Any]:
        """
        Get current authentication status.

        Returns:
            Dictionary with auth status and user info
        """
        return {
            "is_authenticated": self._is_authenticated,
            "user_id": self._user_profile.get("user_id") if self._user_profile else None,
            "user_name": self._user_profile.get("user_name") if self._user_profile else None,
            "has_stored_tokens": self.token_manager.is_token_valid(),
        }


# Singleton instance
_auth_service: ZerodhaAuthService | None = None


def get_auth_service() -> ZerodhaAuthService:
    """Get or create auth service instance."""
    global _auth_service
    if _auth_service is None:
        _auth_service = ZerodhaAuthService()
    return _auth_service
