"""Gemini token manager (ST->AT, refresh, credits)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from ..core.logger import debug_logger
from ..core.config import config
from .database import GeminiDatabase
from .models import GeminiProject, GeminiToken
from .flow_client import FlowClient


class GeminiTokenManager:
    def __init__(self, db: GeminiDatabase, flow_client: FlowClient):
        self.db = db
        self.flow_client = flow_client
        self._lock = asyncio.Lock()

    async def get_all_tokens(self) -> List[GeminiToken]:
        return await self.db.get_all_tokens()

    async def get_active_tokens(self) -> List[GeminiToken]:
        return await self.db.get_active_tokens()

    async def get_token(self, token_id: int) -> Optional[GeminiToken]:
        return await self.db.get_token(token_id)

    async def delete_token(self, token_id: int):
        await self.db.delete_token(token_id)

    async def enable_token(self, token_id: int):
        await self.db.update_token(token_id, is_active=True, ban_reason=None, banned_at=None)
        await self.db.reset_consecutive_errors(token_id)

    async def disable_token(self, token_id: int):
        await self.db.update_token(token_id, is_active=False)

    async def st_to_at(self, st: str) -> Dict[str, Any]:
        return await self.flow_client.st_to_at(st)

    async def refresh_credits(self, token_id: int) -> int:
        token = await self.db.get_token(token_id)
        if not token or not token.at:
            raise ValueError("Token not found or AT missing")
        result = await self.flow_client.get_credits(token.at)
        credits = int(result.get("credits") or 0)
        user_paygate_tier = result.get("userPaygateTier")
        await self.db.update_token(token_id, credits=credits, user_paygate_tier=user_paygate_tier)
        return credits

    async def add_token(
        self,
        st: str,
        project_id: Optional[str] = None,
        project_name: Optional[str] = None,
        remark: Optional[str] = None,
        image_enabled: bool = True,
        video_enabled: bool = True,
        image_concurrency: int = -1,
        video_concurrency: int = -1,
    ) -> GeminiToken:
        existing = await self.db.get_token_by_st(st)
        if existing:
            raise ValueError(f"Token 已存在（邮箱: {existing.email}）")

        # ST -> AT
        debug_logger.log_info("[GEMINI_ADD_TOKEN] Converting ST to AT...")
        try:
            result = await self.flow_client.st_to_at(st)
        except Exception as e:
            raise ValueError(
                "ST转AT失败："
                f"{str(e)}。"
                "如果出现 curl:(28)/无法连接 labs.google 等错误，请在管理后台开启代理（Sora 管理页的代理配置对 Gemini 也生效），"
                "或使用可访问 labs.google 的网络环境。"
            )
        at = result.get("access_token")
        expires = result.get("expires")
        user_info = result.get("user", {}) or {}
        email = user_info.get("email", "") or ""
        name = user_info.get("name") or (email.split("@")[0] if email else "")

        at_expires = None
        if expires:
            try:
                at_expires = datetime.fromisoformat(str(expires).replace("Z", "+00:00"))
            except Exception:
                at_expires = None

        # credits
        credits = 0
        user_paygate_tier = None
        try:
            credits_result = await self.flow_client.get_credits(at)
            credits = int(credits_result.get("credits") or 0)
            user_paygate_tier = credits_result.get("userPaygateTier")
        except Exception:
            pass

        # project
        if project_id:
            if not project_name:
                project_name = datetime.now().strftime("%b %d - %H:%M")
        else:
            if not project_name:
                project_name = datetime.now().strftime("%b %d - %H:%M")
            try:
                project_id = await self.flow_client.create_project(st, project_name)
            except Exception as e:
                raise ValueError(
                    "创建项目失败："
                    f"{str(e)}。"
                    "通常是网络/代理不可达或 ST 权限异常导致。"
                )

        token = GeminiToken(
            st=st,
            at=at,
            at_expires=at_expires,
            email=email,
            name=name,
            remark=remark,
            is_active=True,
            credits=credits,
            user_paygate_tier=user_paygate_tier,
            current_project_id=project_id,
            current_project_name=project_name,
            image_enabled=image_enabled,
            video_enabled=video_enabled,
            image_concurrency=image_concurrency,
            video_concurrency=video_concurrency,
        )

        token_id = await self.db.add_token(token)
        token.id = token_id

        await self.db.add_project(
            GeminiProject(
                project_id=project_id,
                token_id=token_id,
                project_name=project_name,
                tool_name="PINHOLE",
            )
        )

        return token

    async def update_token(
        self,
        token_id: int,
        st: Optional[str] = None,
        project_id: Optional[str] = None,
        project_name: Optional[str] = None,
        remark: Optional[str] = None,
        image_enabled: Optional[bool] = None,
        video_enabled: Optional[bool] = None,
        image_concurrency: Optional[int] = None,
        video_concurrency: Optional[int] = None,
    ):
        update_fields: Dict[str, Any] = {}
        if st is not None:
            update_fields["st"] = st
        if project_id is not None:
            update_fields["current_project_id"] = project_id
        if project_name is not None:
            update_fields["current_project_name"] = project_name
        if remark is not None:
            update_fields["remark"] = remark
        if image_enabled is not None:
            update_fields["image_enabled"] = image_enabled
        if video_enabled is not None:
            update_fields["video_enabled"] = video_enabled
        if image_concurrency is not None:
            update_fields["image_concurrency"] = image_concurrency
        if video_concurrency is not None:
            update_fields["video_concurrency"] = video_concurrency

        if update_fields:
            await self.db.update_token(token_id, **update_fields)

    async def refresh_at(self, token_id: int) -> GeminiToken:
        async with self._lock:
            token = await self.db.get_token(token_id)
            if not token:
                raise ValueError("Token not found")
            result = await self.flow_client.st_to_at(token.st)
            at = result.get("access_token")
            expires = result.get("expires")
            at_expires = None
            if expires:
                try:
                    at_expires = datetime.fromisoformat(str(expires).replace("Z", "+00:00"))
                except Exception:
                    at_expires = None
            await self.db.update_token(token_id, at=at, at_expires=at_expires)
            token.at = at
            token.at_expires = at_expires
            return token

    async def is_at_valid(self, token_id: int) -> bool:
        token = await self.db.get_token(token_id)
        if not token:
            return False
        if not token.at:
            await self.refresh_at(token_id)
            return True
        if not token.at_expires:
            # unknown -> try refresh
            await self.refresh_at(token_id)
            return True

        now = datetime.now(timezone.utc)
        expires = token.at_expires
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        # refresh 1 hour early
        if expires - now <= timedelta(hours=1):
            await self.refresh_at(token_id)
        return True
