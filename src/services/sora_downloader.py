"""Internal Sora share-link parser (ported from sora-downloader)

This module integrates the sora-downloader logic into sora2api so that
the same /get-sora-link JSON API can be provided by the FastAPI app.
"""

import os
import threading
from pathlib import Path
from typing import Dict, Any, Optional

from curl_cffi.requests import Session, errors
from dotenv import load_dotenv, set_key
from ..core.config import config


class SoraAuthManager:
    """Encapsulate Sora auth, auto-refresh and persistence logic."""

    def __init__(self, dotenv_path: Optional[str] = None):
        # Default to project root .env
        if dotenv_path is None:
            project_root = Path(__file__).resolve().parents[2]
            dotenv_path = str(project_root / ".env")

        self.dotenv_path = dotenv_path
        load_dotenv(dotenv_path=self.dotenv_path)

        self.access_token = os.getenv("SORA_AUTH_TOKEN")
        self.refresh_token = os.getenv("SORA_REFRESH_TOKEN")
        self.client_id = os.getenv(
            "SORA_CLIENT_ID",
            "app_OHnYmJt5u1XEdhDUx0ig1ziv",
        )

        self.lock = threading.Lock()
        self.session = Session(impersonate="chrome110", proxies=self._get_proxies())

        # If we only have refresh_token, refresh immediately once
        if not self.access_token and self.refresh_token:
            print("Access token not found. Attempting to refresh immediately...")
            try:
                self.refresh(initial_attempt=True)
            except Exception as e:
                print(f"Initial token refresh failed: {e}")

    def _get_proxies(self) -> Dict[str, str]:
        proxy_url = os.getenv("HTTP_PROXY")
        return {"http": proxy_url, "https": proxy_url} if proxy_url else {}

    def _ensure_env_file(self) -> None:
        if not os.path.exists(self.dotenv_path):
            # Create empty .env file if missing
            with open(self.dotenv_path, "w", encoding="utf-8"):
                pass

    def _save_tokens_to_env(self) -> None:
        """Persist new tokens to .env file."""
        self._ensure_env_file()

        if self.access_token:
            set_key(self.dotenv_path, "SORA_AUTH_TOKEN", self.access_token)
        if self.refresh_token:
            set_key(self.dotenv_path, "SORA_REFRESH_TOKEN", self.refresh_token)
        print("Tokens successfully updated and saved to .env file.")

    def refresh(self, initial_attempt: bool = False) -> None:
        """Use refresh_token to obtain new access_token and refresh_token.

        Uses a thread lock to ensure only one refresh is performed at a time.
        """
        with self.lock:
            if not self.refresh_token:
                raise Exception("Refresh token is not configured.")

            print("Attempting to refresh OpenAI access token...")
            url = f"{config.sora_auth_base_url}/oauth/token"
            payload = {
                "client_id": self.client_id,
                "grant_type": "refresh_token",
                "redirect_uri": "com.openai.sora://auth.openai.com/android/com.openai.sora/callback",
                "refresh_token": self.refresh_token,
            }
            try:
                response = self.session.post(url, json=payload, timeout=20)
                response.raise_for_status()
                data = response.json()

                # Update in-memory tokens
                self.access_token = data["access_token"]
                # OpenAI returns a new refresh_token each time
                self.refresh_token = data["refresh_token"]

                print("Successfully refreshed access token.")
                # Persist to .env file
                self._save_tokens_to_env()

            except errors.RequestsError as e:
                print(
                    "Failed to refresh token. "
                    f"Status: {e.response.status_code if e.response else 'N/A'}, "
                    f"Response: {e.response.text if e.response else 'No Response'}"
                )
                raise Exception(f"Failed to refresh token: {e}") from e

    def fetch_post_detail(self, video_id: str) -> Dict[str, Any]:
        """Call Sora backend to get post detail by video_id.

        This matches the original sora-downloader make_sora_api_call logic.
        """
        if not self.access_token:
            raise Exception("SORA_AUTH_TOKEN is not configured.")

        api_url = f"{config.sora_base_url}/project_y/post/{video_id}"
        headers = {
            "User-Agent": "Sora/1.2025.308",
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "oai-package-name": "com.openai.sora",
            "authorization": f"Bearer {self.access_token}",
        }
        response = self.session.get(api_url, headers=headers, timeout=20)
        response.raise_for_status()
        return response.json()


# Global singleton for the downloader endpoints
auth_manager = SoraAuthManager()
