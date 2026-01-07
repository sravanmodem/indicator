"""
User Authentication Service
Simple admin user authentication with password change capability
"""

import hashlib
import json
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

from loguru import logger


class UserAuthService:
    """Handles user authentication with session management."""

    DEFAULT_USERNAME = "admin"
    DEFAULT_PASSWORD = "admin123"  # Will be hashed on first run
    SESSION_TIMEOUT_HOURS = 24

    def __init__(self):
        self.data_dir = Path(__file__).parent.parent.parent / "data"
        self.data_dir.mkdir(exist_ok=True)
        self.users_file = self.data_dir / "users.json"
        self.sessions_file = self.data_dir / "sessions.json"

        self._users: Dict[str, Dict[str, Any]] = {}
        self._sessions: Dict[str, Dict[str, Any]] = {}

        self._load_users()
        self._load_sessions()

    def _hash_password(self, password: str, salt: Optional[str] = None) -> tuple[str, str]:
        """Hash password with salt."""
        if salt is None:
            salt = secrets.token_hex(16)
        hashed = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode(),
            salt.encode(),
            100000
        ).hex()
        return hashed, salt

    def _verify_password(self, password: str, stored_hash: str, salt: str) -> bool:
        """Verify password against stored hash."""
        hashed, _ = self._hash_password(password, salt)
        return hashed == stored_hash

    def _load_users(self):
        """Load users from file or create default admin."""
        if self.users_file.exists():
            try:
                with open(self.users_file, 'r') as f:
                    self._users = json.load(f)
                logger.info(f"Loaded {len(self._users)} users")
            except Exception as e:
                logger.error(f"Error loading users: {e}")
                self._create_default_admin()
        else:
            self._create_default_admin()

    def _create_default_admin(self):
        """Create default admin user."""
        hashed, salt = self._hash_password(self.DEFAULT_PASSWORD)
        self._users = {
            self.DEFAULT_USERNAME: {
                "username": self.DEFAULT_USERNAME,
                "password_hash": hashed,
                "salt": salt,
                "is_admin": True,
                "created_at": datetime.now().isoformat(),
                "last_login": None,
                "password_changed": False
            }
        }
        self._save_users()
        logger.info("Created default admin user")

    def _save_users(self):
        """Save users to file."""
        try:
            with open(self.users_file, 'w') as f:
                json.dump(self._users, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving users: {e}")

    def _load_sessions(self):
        """Load sessions from file."""
        if self.sessions_file.exists():
            try:
                with open(self.sessions_file, 'r') as f:
                    self._sessions = json.load(f)
                # Clean expired sessions
                self._cleanup_sessions()
            except Exception as e:
                logger.error(f"Error loading sessions: {e}")
                self._sessions = {}
        else:
            self._sessions = {}

    def _save_sessions(self):
        """Save sessions to file."""
        try:
            with open(self.sessions_file, 'w') as f:
                json.dump(self._sessions, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving sessions: {e}")

    def _cleanup_sessions(self):
        """Remove expired sessions."""
        now = datetime.now()
        expired = []
        for session_id, session in self._sessions.items():
            expires = datetime.fromisoformat(session["expires_at"])
            if expires < now:
                expired.append(session_id)

        for session_id in expired:
            del self._sessions[session_id]

        if expired:
            self._save_sessions()
            logger.info(f"Cleaned up {len(expired)} expired sessions")

    def authenticate(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and return session token if successful."""
        user = self._users.get(username)
        if not user:
            logger.warning(f"Login attempt for unknown user: {username}")
            return None

        if not self._verify_password(password, user["password_hash"], user["salt"]):
            logger.warning(f"Invalid password for user: {username}")
            return None

        # Create session
        session_token = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(hours=self.SESSION_TIMEOUT_HOURS)

        self._sessions[session_token] = {
            "username": username,
            "created_at": datetime.now().isoformat(),
            "expires_at": expires_at.isoformat(),
            "is_admin": user["is_admin"]
        }
        self._save_sessions()

        # Update last login
        user["last_login"] = datetime.now().isoformat()
        self._save_users()

        logger.info(f"User {username} logged in successfully")
        return session_token

    def validate_session(self, session_token: str) -> Optional[Dict[str, Any]]:
        """Validate session token and return user info if valid."""
        if not session_token:
            return None

        session = self._sessions.get(session_token)
        if not session:
            return None

        # Check expiration
        expires = datetime.fromisoformat(session["expires_at"])
        if expires < datetime.now():
            del self._sessions[session_token]
            self._save_sessions()
            return None

        username = session["username"]
        user = self._users.get(username)
        if not user:
            return None

        return {
            "username": username,
            "is_admin": user["is_admin"],
            "password_changed": user.get("password_changed", False)
        }

    def logout(self, session_token: str):
        """Invalidate session."""
        if session_token in self._sessions:
            del self._sessions[session_token]
            self._save_sessions()
            logger.info("User logged out")

    def change_password(self, username: str, old_password: str, new_password: str) -> tuple[bool, str]:
        """Change user password."""
        user = self._users.get(username)
        if not user:
            return False, "User not found"

        if not self._verify_password(old_password, user["password_hash"], user["salt"]):
            return False, "Current password is incorrect"

        if len(new_password) < 6:
            return False, "New password must be at least 6 characters"

        if new_password == old_password:
            return False, "New password must be different from current password"

        # Update password
        hashed, salt = self._hash_password(new_password)
        user["password_hash"] = hashed
        user["salt"] = salt
        user["password_changed"] = True
        user["password_updated_at"] = datetime.now().isoformat()
        self._save_users()

        logger.info(f"Password changed for user: {username}")
        return True, "Password changed successfully"

    def change_username(self, old_username: str, new_username: str, password: str) -> tuple[bool, str]:
        """Change username."""
        user = self._users.get(old_username)
        if not user:
            return False, "User not found"

        if not self._verify_password(password, user["password_hash"], user["salt"]):
            return False, "Password is incorrect"

        if new_username in self._users:
            return False, "Username already exists"

        if len(new_username) < 3:
            return False, "Username must be at least 3 characters"

        # Update username
        user["username"] = new_username
        del self._users[old_username]
        self._users[new_username] = user

        # Update sessions
        for session in self._sessions.values():
            if session["username"] == old_username:
                session["username"] = new_username

        self._save_users()
        self._save_sessions()

        logger.info(f"Username changed from {old_username} to {new_username}")
        return True, "Username changed successfully"

    def get_user_info(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user info without sensitive data."""
        user = self._users.get(username)
        if not user:
            return None

        return {
            "username": user["username"],
            "is_admin": user["is_admin"],
            "created_at": user["created_at"],
            "last_login": user["last_login"],
            "password_changed": user.get("password_changed", False)
        }


# Singleton instance
_user_auth_service: Optional[UserAuthService] = None


def get_user_auth_service() -> UserAuthService:
    """Get singleton instance of UserAuthService."""
    global _user_auth_service
    if _user_auth_service is None:
        _user_auth_service = UserAuthService()
    return _user_auth_service
