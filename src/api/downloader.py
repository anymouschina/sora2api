"""Downloader-style endpoints (ported from sora-downloader).

Provides:
- GET /downloader: simple web UI for manual use
- POST /get-sora-link: JSON API compatible with sora-downloader
"""

import os
import re
import asyncio
from pathlib import Path
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from curl_cffi.requests import AsyncSession, errors

from ..services.sora_downloader import auth_manager
from ..services.token_manager import TokenManager
from ..services.proxy_manager import ProxyManager


router = APIRouter()

# Dependency injection (use existing token pool and proxy config)
token_manager: TokenManager = None
proxy_manager: ProxyManager = None


def set_dependencies(tm: TokenManager, pm: ProxyManager):
    """Inject shared services from main app."""
    global token_manager, proxy_manager
    token_manager = tm
    proxy_manager = pm


class GetSoraLinkRequest(BaseModel):
    url: str
    token: Optional[str] = None


APP_ACCESS_TOKEN = os.getenv("APP_ACCESS_TOKEN")


@router.get("/downloader")
async def downloader_page() -> FileResponse:
    """Serve a simple downloader UI page."""
    static_dir = Path(__file__).resolve().parent.parent / "static"
    return FileResponse(str(static_dir / "downloader.html"))


@router.post("/get-sora-link")
async def get_sora_link(request: GetSoraLinkRequest):
    """Get watermark-free video link from Sora share URL.

    This mirrors the behavior of the original Flask /get-sora-link API so that
    it can be used both by the web UI and by external callers
    (e.g. sora2api watermark-free custom parser).
    """
    # Optional access protection
    if APP_ACCESS_TOKEN:
        if request.token != APP_ACCESS_TOKEN:
            return JSONResponse(
                {"error": "无效或缺失的访问令牌。"},
                status_code=401,
            )

    sora_url = request.url
    if not sora_url:
        return JSONResponse({"error": "未提供 URL"}, status_code=400)

    match = re.search(r"sora\.chatgpt\.com/p/([a-zA-Z0-9_]+)", sora_url)
    if not match:
        return JSONResponse(
            {"error": "无效的 Sora 链接格式。请发布后复制分享链接"},
            status_code=400,
        )

    video_id = match.group(1)

    async def _fetch_post_with_token_pool():
        """Use existing token pool in sora2api (preferred)."""
        if token_manager is None:
            return None

        try:
            tokens = await token_manager.get_active_tokens()
        except Exception as e:
            return JSONResponse(
                {"error": f"获取可用账号失败: {e}"},
                status_code=500,
            )

        if not tokens:
            return None

        token_obj = tokens[0]

        # Build request
        api_url = f"https://sora.chatgpt.com/backend/project_y/post/{video_id}"
        headers = {
            "User-Agent": "Sora/1.2025.308",
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "oai-package-name": "com.openai.sora",
            "authorization": f"Bearer {token_obj.token}",
        }

        proxy_url = None
        if proxy_manager is not None:
            try:
                proxy_url = await proxy_manager.get_proxy_url()
            except Exception:
                proxy_url = None

        kwargs = {
            "headers": headers,
            "timeout": 20,
            "impersonate": "chrome110",
        }
        if proxy_url:
            kwargs["proxy"] = proxy_url

        try:
            async with AsyncSession() as session:
                response = await session.get(api_url, **kwargs)

                if response.status_code == 200:
                    return response.json()

                if response.status_code in (401, 403):
                    return JSONResponse(
                        {
                            "error": (
                                "Sora 账号 Access Token 无效或已过期，"
                                "请在管理后台更新或禁用该账号后重试。"
                            )
                        },
                        status_code=500,
                    )

                return JSONResponse(
                    {
                        "error": (
                            f"请求 OpenAI API 失败: "
                            f"{response.status_code} - {response.text[:200]}"
                        )
                    },
                    status_code=500,
                )
        except Exception as e:
            return JSONResponse(
                {"error": f"请求 OpenAI API 失败: {e}"},
                status_code=500,
            )

    # Fallback: use env-based SoraAuthManager (.env SORA_AUTH_TOKEN / SORA_REFRESH_TOKEN)
    async def _fetch_post_with_env():
        # Ensure we have at least one credential configured
        if not auth_manager.access_token and not auth_manager.refresh_token:
            return JSONResponse(
                {
                    "error": (
                        "服务器配置错误：未设置 SORA_AUTH_TOKEN 或 SORA_REFRESH_TOKEN，"
                        "且号池中也没有可用账号。"
                    )
                },
                status_code=500,
            )

        try:
            return await asyncio.to_thread(auth_manager.fetch_post_detail, video_id)
        except errors.RequestsError as e:
            # If 401/403, try refresh then retry once
            if e.response is not None and e.response.status_code in [401, 403]:
                print(
                    f"Access token expired or invalid (HTTP {e.response.status_code}). "
                    "Triggering refresh..."
                )
                try:
                    await asyncio.to_thread(auth_manager.refresh)
                    print("Retrying API call with new token...")
                    return await asyncio.to_thread(
                        auth_manager.fetch_post_detail, video_id
                    )
                except Exception as refresh_error:
                    return JSONResponse(
                        {
                            "error": (
                                "无法刷新认证令牌，请检查SORA_REFRESH_TOKEN配置。"
                                f"错误: {refresh_error}"
                            )
                        },
                        status_code=500,
                    )
            # Other network errors
            return JSONResponse(
                {"error": f"请求 OpenAI API 失败: {e}"},
                status_code=500,
            )
        except Exception as e:
            return JSONResponse(
                {"error": f"发生未知错误: {e}"},
                status_code=500,
            )

    # Prefer token pool; fallback to .env credentials
    result = await _fetch_post_with_token_pool()
    if result is None:
        result = await _fetch_post_with_env()
    if isinstance(result, JSONResponse):
        # Error path already wrapped
        return result

    # Extract download link
    try:
        attachment = result["post"]["attachments"][0]

        # 优先使用后端直接给出的 download_urls.no_watermark
        download_urls = attachment.get("download_urls") or {}
        no_watermark = download_urls.get("no_watermark")
        watermark = download_urls.get("watermark")

        if no_watermark:
            download_link = no_watermark
        elif watermark:
            # 部分场景仅提供 watermark 字段
            download_link = watermark
        else:
            # 回退到 encodings.source.path（与原逻辑一致）
            download_link = attachment["encodings"]["source"]["path"]

        return {"download_link": download_link}
    except (KeyError, IndexError):
        return JSONResponse(
            {
                "error": "无法从API响应中找到下载链接，可能是API结构已更改。"
            },
            status_code=500,
        )
