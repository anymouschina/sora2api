"""Load balancing for Gemini token pool."""

from __future__ import annotations

import random
from typing import Optional

from ..core.logger import debug_logger
from .models import GeminiToken
from .token_manager import GeminiTokenManager
from ..services.concurrency_manager import ConcurrencyManager


class GeminiLoadBalancer:
    def __init__(self, token_manager: GeminiTokenManager, concurrency_manager: Optional[ConcurrencyManager] = None):
        self.token_manager = token_manager
        self.concurrency_manager = concurrency_manager

    async def select_token(self, for_image_generation: bool = False, for_video_generation: bool = False) -> Optional[GeminiToken]:
        tokens = await self.token_manager.get_active_tokens()
        if not tokens:
            return None

        available = []
        for token in tokens:
            if for_image_generation and not token.image_enabled:
                continue
            if for_video_generation and not token.video_enabled:
                continue

            if self.concurrency_manager and token.id is not None:
                if for_image_generation:
                    if not await self.concurrency_manager.can_use_image(token.id):
                        continue
                if for_video_generation:
                    if not await self.concurrency_manager.can_use_video(token.id):
                        continue

            available.append(token)

        if not available:
            debug_logger.log_info("[GEMINI_LOAD_BALANCER] No available tokens after filtering")
            return None

        return random.choice(available)

