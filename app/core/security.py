"""
Security Module
Handles encryption/decryption of sensitive data like access tokens
"""

import base64
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from loguru import logger

from app.core.config import get_settings


class TokenManager:
    """Secure token storage and retrieval."""

    def __init__(self):
        self.settings = get_settings()
        self._fernet = self._create_cipher()

    def _create_cipher(self) -> Fernet:
        """Create Fernet cipher from app secret."""
        salt = b"nifty_options_trading_v1"
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480000,
        )
        key = base64.urlsafe_b64encode(
            kdf.derive(self.settings.app_secret_key.encode())
        )
        return Fernet(key)

    def save_tokens(
        self,
        access_token: str,
        request_token: str | None = None,
        user_id: str | None = None,
        expires_at: datetime | None = None,
    ) -> bool:
        """
        Securely save authentication tokens.

        Args:
            access_token: Kite access token
            request_token: Request token used for login
            user_id: Zerodha user ID
            expires_at: Token expiration datetime

        Returns:
            True if saved successfully
        """
        try:
            data = {
                "access_token": access_token,
                "request_token": request_token,
                "user_id": user_id,
                "expires_at": expires_at.isoformat() if expires_at else None,
                "saved_at": datetime.now().isoformat(),
            }

            encrypted = self._fernet.encrypt(json.dumps(data).encode())

            token_file = self.settings.token_file
            token_file.parent.mkdir(parents=True, exist_ok=True)
            token_file.write_bytes(encrypted)

            logger.info(f"Tokens saved securely for user: {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save tokens: {e}")
            return False

    def load_tokens(self) -> dict[str, Any] | None:
        """
        Load and decrypt stored tokens.

        Returns:
            Dictionary with token data or None if not found/invalid
        """
        try:
            token_file = self.settings.token_file
            if not token_file.exists():
                logger.debug("No stored tokens found")
                return None

            encrypted = token_file.read_bytes()
            decrypted = self._fernet.decrypt(encrypted)
            data = json.loads(decrypted.decode())

            # Check expiration
            if data.get("expires_at"):
                expires_at = datetime.fromisoformat(data["expires_at"])
                if datetime.now() > expires_at:
                    logger.warning("Stored tokens have expired")
                    return None

            logger.info(f"Loaded tokens for user: {data.get('user_id')}")
            return data

        except Exception as e:
            logger.error(f"Failed to load tokens: {e}")
            return None

    def clear_tokens(self) -> bool:
        """Remove stored tokens."""
        try:
            token_file = self.settings.token_file
            if token_file.exists():
                token_file.unlink()
                logger.info("Tokens cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to clear tokens: {e}")
            return False

    def is_token_valid(self) -> bool:
        """Check if we have valid, non-expired tokens."""
        tokens = self.load_tokens()
        return tokens is not None and tokens.get("access_token") is not None

    def get_access_token(self) -> str | None:
        """Get the current access token if valid."""
        tokens = self.load_tokens()
        return tokens.get("access_token") if tokens else None


# Singleton instance
_token_manager: TokenManager | None = None


def get_token_manager() -> TokenManager:
    """Get or create token manager instance."""
    global _token_manager
    if _token_manager is None:
        _token_manager = TokenManager()
    return _token_manager
