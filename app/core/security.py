"""
Security Module
Handles encryption/decryption of sensitive data like access tokens
Uses SQLite database for persistence across server restarts
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
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import get_settings


class TokenManager:
    """Secure token storage and retrieval using database."""

    def __init__(self):
        self.settings = get_settings()
        self._fernet = self._create_cipher()
        self._engine = None
        self._session = None

    def _get_db_session(self):
        """Get or create database session."""
        if self._session is None:
            from app.models.paper_trading import Base, AuthToken

            # Use same database path as paper trading service
            db_path = Path.home() / ".options_indicator" / "paper_trading.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)

            self._engine = create_engine(f"sqlite:///{db_path}")
            Base.metadata.create_all(self._engine)

            Session = sessionmaker(bind=self._engine)
            self._session = Session()
        return self._session

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

    def _encrypt(self, data: str) -> str:
        """Encrypt a string."""
        return self._fernet.encrypt(data.encode()).decode()

    def _decrypt(self, data: str) -> str:
        """Decrypt a string."""
        return self._fernet.decrypt(data.encode()).decode()

    def save_tokens(
        self,
        access_token: str,
        request_token: str | None = None,
        user_id: str | None = None,
        expires_at: datetime | None = None,
    ) -> bool:
        """
        Securely save authentication tokens to database.

        Args:
            access_token: Kite access token
            request_token: Request token used for login
            user_id: Zerodha user ID
            expires_at: Token expiration datetime

        Returns:
            True if saved successfully
        """
        try:
            from app.models.paper_trading import AuthToken

            session = self._get_db_session()

            # Encrypt tokens before storing
            encrypted_access = self._encrypt(access_token)
            encrypted_request = self._encrypt(request_token) if request_token else None

            # Check if token already exists for this user
            existing = session.query(AuthToken).filter_by(user_id=user_id or "default").first()

            if existing:
                # Update existing token
                existing.access_token = encrypted_access
                existing.request_token = encrypted_request
                existing.expires_at = expires_at
                existing.updated_at = datetime.now()
            else:
                # Create new token entry
                token = AuthToken(
                    user_id=user_id or "default",
                    access_token=encrypted_access,
                    request_token=encrypted_request,
                    expires_at=expires_at,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                )
                session.add(token)

            session.commit()
            logger.info(f"Tokens saved to database for user: {user_id}")

            # Also save to file as backup
            self._save_to_file(access_token, request_token, user_id, expires_at)

            return True

        except Exception as e:
            logger.error(f"Failed to save tokens to database: {e}")
            # Fallback to file storage
            return self._save_to_file(access_token, request_token, user_id, expires_at)

    def _save_to_file(
        self,
        access_token: str,
        request_token: str | None,
        user_id: str | None,
        expires_at: datetime | None,
    ) -> bool:
        """Fallback file-based token storage."""
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

            logger.info(f"Tokens saved to file for user: {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save tokens to file: {e}")
            return False

    def load_tokens(self) -> dict[str, Any] | None:
        """
        Load and decrypt stored tokens from database.

        Returns:
            Dictionary with token data or None if not found/invalid
        """
        # Try database first
        try:
            from app.models.paper_trading import AuthToken

            session = self._get_db_session()
            token = session.query(AuthToken).filter_by(user_id="default").first()

            if token:
                # Check expiration
                if token.expires_at and datetime.now() > token.expires_at:
                    logger.warning("Stored tokens have expired")
                    self.clear_tokens()
                    return None

                # Decrypt tokens
                access_token = self._decrypt(token.access_token)
                request_token = self._decrypt(token.request_token) if token.request_token else None

                logger.info(f"Loaded tokens from database for user: {token.user_id}")
                return {
                    "access_token": access_token,
                    "request_token": request_token,
                    "user_id": token.user_id,
                    "expires_at": token.expires_at.isoformat() if token.expires_at else None,
                }

        except Exception as e:
            logger.warning(f"Failed to load tokens from database: {e}")

        # Fallback to file storage
        return self._load_from_file()

    def _load_from_file(self) -> dict[str, Any] | None:
        """Fallback file-based token loading."""
        try:
            token_file = self.settings.token_file
            if not token_file.exists():
                logger.debug("No stored tokens found in file")
                return None

            encrypted = token_file.read_bytes()
            decrypted = self._fernet.decrypt(encrypted)
            data = json.loads(decrypted.decode())

            # Check expiration
            if data.get("expires_at"):
                expires_at = datetime.fromisoformat(data["expires_at"])
                if datetime.now() > expires_at:
                    logger.warning("Stored tokens have expired (file)")
                    return None

            logger.info(f"Loaded tokens from file for user: {data.get('user_id')}")
            return data

        except Exception as e:
            logger.error(f"Failed to load tokens from file: {e}")
            return None

    def clear_tokens(self) -> bool:
        """Remove stored tokens from database and file."""
        try:
            # Clear from database
            try:
                from app.models.paper_trading import AuthToken

                session = self._get_db_session()
                session.query(AuthToken).delete()
                session.commit()
                logger.info("Tokens cleared from database")
            except Exception as e:
                logger.warning(f"Failed to clear tokens from database: {e}")

            # Clear from file
            token_file = self.settings.token_file
            if token_file.exists():
                token_file.unlink()
                logger.info("Tokens cleared from file")

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
