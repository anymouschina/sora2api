"""Gemini/Flow2API data models (kept separate from Sora models)."""

from __future__ import annotations

from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel


class GeminiToken(BaseModel):
    """Token model for Google Labs / VideoFX (Veo)."""

    id: Optional[int] = None
    st: str  # __Secure-next-auth.session-token
    at: Optional[str] = None  # access token derived from ST
    at_expires: Optional[datetime] = None

    email: str = ""
    name: Optional[str] = ""
    remark: Optional[str] = None

    is_active: bool = True
    created_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    use_count: int = 0

    credits: int = 0
    user_paygate_tier: Optional[str] = None

    current_project_id: Optional[str] = None
    current_project_name: Optional[str] = None

    image_enabled: bool = True
    video_enabled: bool = True

    image_concurrency: int = -1
    video_concurrency: int = -1

    ban_reason: Optional[str] = None
    banned_at: Optional[datetime] = None


class GeminiProject(BaseModel):
    id: Optional[int] = None
    project_id: str
    token_id: int
    project_name: str
    tool_name: str = "PINHOLE"
    is_active: bool = True
    created_at: Optional[datetime] = None


class GeminiTokenStats(BaseModel):
    token_id: int
    image_count: int = 0
    video_count: int = 0
    success_count: int = 0
    error_count: int = 0
    last_success_at: Optional[datetime] = None
    last_error_at: Optional[datetime] = None
    today_image_count: int = 0
    today_video_count: int = 0
    today_error_count: int = 0
    today_date: Optional[str] = None
    consecutive_error_count: int = 0


class GeminiTask(BaseModel):
    id: Optional[int] = None
    task_id: str
    token_id: int
    model: str
    prompt: str
    status: str = "processing"  # processing/completed/failed
    progress: int = 0
    result_urls: Optional[List[str]] = None
    error_message: Optional[str] = None
    scene_id: Optional[str] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class GeminiRequestLog(BaseModel):
    id: Optional[int] = None
    token_id: Optional[int] = None
    operation: str
    request_body: Optional[str] = None
    response_body: Optional[str] = None
    status_code: int
    duration: float
    created_at: Optional[datetime] = None


class GeminiCaptchaConfig(BaseModel):
    id: int = 1
    captcha_method: str = "browser"  # yescaptcha/browser/personal
    yescaptcha_api_key: str = ""
    yescaptcha_base_url: str = "https://api.yescaptcha.com"
    browser_proxy_enabled: bool = False
    browser_proxy_url: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

