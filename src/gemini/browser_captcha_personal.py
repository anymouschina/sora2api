"""Personal (non-headless) browser captcha service placeholder.

Flow2API supports a long-running browser window for manually solving captcha.
In this merged repo we keep the interface, but implementation is intentionally
minimal; prefer `captcha_method = "yescaptcha"` or `"browser"` in headless mode.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from ..core.logger import debug_logger
from ..services.proxy_manager import ProxyManager


class BrowserCaptchaService:
    _instance = None
    _lock = asyncio.Lock()

    def __init__(self, proxy_manager: ProxyManager):
        self.proxy_manager = proxy_manager

    @classmethod
    async def get_instance(cls, proxy_manager: ProxyManager):
        async with cls._lock:
            if cls._instance is None:
                cls._instance = BrowserCaptchaService(proxy_manager)
            return cls._instance

    async def open_login_window(self):
        debug_logger.log_warning("[CAPTCHA personal] open_login_window not implemented in this build")

    async def get_token(self, project_id: str) -> Optional[str]:
        debug_logger.log_warning("[CAPTCHA personal] get_token not implemented in this build")
        return None

    async def close(self):
        return None

