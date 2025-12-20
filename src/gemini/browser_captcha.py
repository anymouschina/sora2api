"""Headless browser captcha service (Playwright).

This is optional and only used when `captcha_method = "browser"`.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from playwright.async_api import async_playwright

from ..core.config import config
from ..core.logger import debug_logger
from ..services.proxy_manager import ProxyManager


class BrowserCaptchaService:
    _instance = None
    _lock = asyncio.Lock()

    def __init__(self, proxy_manager: ProxyManager):
        self.proxy_manager = proxy_manager
        self.playwright = None
        self.browser = None

    @classmethod
    async def get_instance(cls, proxy_manager: ProxyManager):
        async with cls._lock:
            if cls._instance is None:
                cls._instance = BrowserCaptchaService(proxy_manager)
                await cls._instance._init()
            return cls._instance

    async def _init(self):
        self.playwright = await async_playwright().start()

        launch_kwargs = {"headless": True}
        # Optional separate proxy for browser captcha
        if config.browser_proxy_enabled and config.browser_proxy_url:
            launch_kwargs["proxy"] = {"server": config.browser_proxy_url}

        self.browser = await self.playwright.chromium.launch(**launch_kwargs)
        debug_logger.log_info("[CAPTCHA] Headless browser initialized")

    async def get_token(self, project_id: str) -> Optional[str]:
        if not self.browser:
            return None

        context = await self.browser.new_context()
        page = await context.new_page()

        try:
            url = f"https://labs.google/fx/tools/flow/project/{project_id}"
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            # Best-effort: rely on upstream page to provide a token via JS execution.
            token = await page.evaluate("() => window.__RECAPTCHA_TOKEN__ || null")
            return token
        except Exception as e:
            debug_logger.log_error(f"[CAPTCHA] get_token error: {e}")
            return None
        finally:
            await context.close()

    async def close(self):
        try:
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
        finally:
            self.browser = None
            self.playwright = None

