"""Gemini/Flow admin routes (token pool management).

Mounted with prefix `/gemini` to avoid clashing with Sora admin APIs.
Uses the same admin session token as Sora (`/api/login`).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, List

from ..api.admin import verify_admin_token
from ..core.config import config
from ..core.logger import debug_logger
from ..gemini.database import GeminiDatabase
from ..gemini.token_manager import GeminiTokenManager
from ..gemini.curl_import import parse_curl_or_cookie
from ..gemini.flow_client import FlowClient

router = APIRouter()

token_manager: Optional[GeminiTokenManager] = None
db: Optional[GeminiDatabase] = None


def set_dependencies(tm: GeminiTokenManager, database: GeminiDatabase):
    global token_manager, db
    token_manager = tm
    db = database


class AddTokenRequest(BaseModel):
    st: str
    project_id: Optional[str] = None
    project_name: Optional[str] = None
    remark: Optional[str] = None
    image_enabled: bool = True
    video_enabled: bool = True
    image_concurrency: int = -1
    video_concurrency: int = -1


class UpdateTokenRequest(BaseModel):
    st: str
    project_id: Optional[str] = None
    project_name: Optional[str] = None
    remark: Optional[str] = None
    image_enabled: Optional[bool] = None
    video_enabled: Optional[bool] = None
    image_concurrency: Optional[int] = None
    video_concurrency: Optional[int] = None


class ST2ATRequest(BaseModel):
    st: str


class CaptchaConfigRequest(BaseModel):
    captcha_method: str
    yescaptcha_api_key: Optional[str] = ""
    yescaptcha_base_url: Optional[str] = "https://api.yescaptcha.com"
    browser_proxy_enabled: bool = False
    browser_proxy_url: Optional[str] = None


class ImportTokenItem(BaseModel):
    email: Optional[str] = None
    access_token: Optional[str] = None
    session_token: Optional[str] = None
    is_active: bool = True
    image_enabled: bool = True
    video_enabled: bool = True
    image_concurrency: int = -1
    video_concurrency: int = -1
    remark: Optional[str] = None
    project_id: Optional[str] = None
    project_name: Optional[str] = None


class ImportTokensRequest(BaseModel):
    tokens: List[ImportTokenItem]


def _require_deps():
    if token_manager is None or db is None:
        raise HTTPException(status_code=500, detail="Gemini admin dependencies not initialized")


@router.get("/api/stats")
async def get_stats(token: str = Depends(verify_admin_token)):
    _require_deps()
    tokens = await token_manager.get_all_tokens()
    active_tokens = [t for t in tokens if t.is_active]

    total_images = total_videos = total_errors = 0
    today_images = today_videos = today_errors = 0
    for t in tokens:
        stats = await db.get_token_stats(t.id)
        if not stats:
            continue
        total_images += stats.image_count
        total_videos += stats.video_count
        total_errors += stats.error_count
        today_images += stats.today_image_count
        today_videos += stats.today_video_count
        today_errors += stats.today_error_count

    return {
        "total_tokens": len(tokens),
        "active_tokens": len(active_tokens),
        "today_images": today_images,
        "total_images": total_images,
        "today_videos": today_videos,
        "total_videos": total_videos,
        "today_errors": today_errors,
        "total_errors": total_errors,
    }


@router.get("/api/tokens")
async def get_tokens(token: str = Depends(verify_admin_token)):
    _require_deps()
    tokens = await token_manager.get_all_tokens()
    result = []
    for t in tokens:
        stats = await db.get_token_stats(t.id)
        result.append(
            {
                "id": t.id,
                "st": t.st,
                "at": t.at,
                "at_expires": t.at_expires.isoformat() if t.at_expires else None,
                "token": t.at,  # front-end compatibility
                "email": t.email,
                "name": t.name,
                "remark": t.remark,
                "is_active": t.is_active,
                "created_at": t.created_at.isoformat() if t.created_at else None,
                "last_used_at": t.last_used_at.isoformat() if t.last_used_at else None,
                "use_count": t.use_count,
                "credits": t.credits,
                "user_paygate_tier": t.user_paygate_tier,
                "current_project_id": t.current_project_id,
                "current_project_name": t.current_project_name,
                "image_enabled": t.image_enabled,
                "video_enabled": t.video_enabled,
                "image_concurrency": t.image_concurrency,
                "video_concurrency": t.video_concurrency,
                "image_count": stats.image_count if stats else 0,
                "video_count": stats.video_count if stats else 0,
                "error_count": stats.error_count if stats else 0,
            }
        )
    return result


@router.post("/api/tokens/st2at")
async def st2at(request: ST2ATRequest, token: str = Depends(verify_admin_token)):
    _require_deps()
    try:
        result = await token_manager.st_to_at(request.st)
        return {"success": True, "access_token": result.get("access_token"), "expires": result.get("expires"), "user": result.get("user")}
    except Exception as e:
        try:
            import traceback
            debug_logger.log_error(
                error_message=f"[gemini][st2at] {str(e)}",
                status_code=400,
                response_text=traceback.format_exc(),
            )
        except Exception:
            pass
        print(f"[gemini][st2at] error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/tokens")
async def add_token(request: AddTokenRequest, token: str = Depends(verify_admin_token)):
    _require_deps()
    try:
        new_token = await token_manager.add_token(
            st=FlowClient.normalize_session_token(request.st),
            project_id=request.project_id,
            project_name=request.project_name,
            remark=request.remark,
            image_enabled=request.image_enabled,
            video_enabled=request.video_enabled,
            image_concurrency=request.image_concurrency,
            video_concurrency=request.video_concurrency,
        )
        return {"success": True, "message": "Token添加成功", "token": {"id": new_token.id, "email": new_token.email, "credits": new_token.credits}}
    except Exception as e:
        try:
            import traceback
            debug_logger.log_error(
                error_message=f"[gemini][add_token] {str(e)}",
                status_code=400,
                response_text=traceback.format_exc(),
            )
        except Exception:
            pass
        print(f"[gemini][add_token] error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


class ImportCurlRequest(BaseModel):
    ecurl: str
    project_id: Optional[str] = None
    project_name: Optional[str] = None
    remark: Optional[str] = None
    image_enabled: bool = True
    video_enabled: bool = True
    image_concurrency: int = -1
    video_concurrency: int = -1


@router.post("/api/tokens/import-curl")
async def import_token_from_curl(request: ImportCurlRequest, token: str = Depends(verify_admin_token)):
    _require_deps()
    parsed = parse_curl_or_cookie(request.ecurl)
    st = parsed.st
    if not st:
        raise HTTPException(
            status_code=400,
            detail=(
                "未从 ecurl/curl 中解析到 `__Secure-next-auth.session-token` / `next-auth.session-token`。"
                "当前 Gemini/Flow 只支持 labs.google 的 next-auth 会话 Token（ST），不支持 gemini.google.com 的网页 batchexecute 接口。"
            ),
        )
    try:
        new_token = await token_manager.add_token(
            st=st,
            project_id=request.project_id,
            project_name=request.project_name,
            remark=request.remark,
            image_enabled=request.image_enabled,
            video_enabled=request.video_enabled,
            image_concurrency=request.image_concurrency,
            video_concurrency=request.video_concurrency,
        )
        return {"success": True, "message": "Token导入成功", "token": {"id": new_token.id, "email": new_token.email, "credits": new_token.credits}}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/api/tokens/{token_id}")
async def update_token(token_id: int, request: UpdateTokenRequest, token: str = Depends(verify_admin_token)):
    _require_deps()
    try:
        await token_manager.update_token(
            token_id=token_id,
            st=request.st,
            project_id=request.project_id,
            project_name=request.project_name,
            remark=request.remark,
            image_enabled=request.image_enabled,
            video_enabled=request.video_enabled,
            image_concurrency=request.image_concurrency,
            video_concurrency=request.video_concurrency,
        )
        return {"success": True, "message": "Token更新成功"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/api/tokens/{token_id}")
async def delete_token(token_id: int, token: str = Depends(verify_admin_token)):
    _require_deps()
    try:
        await token_manager.delete_token(token_id)
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/tokens/{token_id}/enable")
async def enable_token(token_id: int, token: str = Depends(verify_admin_token)):
    _require_deps()
    await token_manager.enable_token(token_id)
    return {"success": True}


@router.post("/api/tokens/{token_id}/disable")
async def disable_token(token_id: int, token: str = Depends(verify_admin_token)):
    _require_deps()
    await token_manager.disable_token(token_id)
    return {"success": True}


@router.post("/api/tokens/{token_id}/refresh-at")
async def refresh_at(token_id: int, token: str = Depends(verify_admin_token)):
    _require_deps()
    try:
        t = await token_manager.refresh_at(token_id)
        return {"success": True, "token": {"id": t.id, "at_expires": t.at_expires.isoformat() if t.at_expires else None}}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/tokens/{token_id}/refresh-credits")
async def refresh_credits(token_id: int, token: str = Depends(verify_admin_token)):
    _require_deps()
    try:
        credits = await token_manager.refresh_credits(token_id)
        return {"success": True, "credits": credits}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/tokens/import")
async def import_tokens(request: ImportTokensRequest, token: str = Depends(verify_admin_token)):
    _require_deps()
    added = 0
    updated = 0
    for item in request.tokens:
        st = item.session_token or ""
        if not st:
            continue
        existing = await db.get_token_by_st(st)
        if existing:
            await db.update_token(
                existing.id,
                email=item.email or existing.email,
                remark=item.remark if item.remark is not None else existing.remark,
                is_active=1 if item.is_active else 0,
                image_enabled=1 if item.image_enabled else 0,
                video_enabled=1 if item.video_enabled else 0,
                image_concurrency=item.image_concurrency,
                video_concurrency=item.video_concurrency,
                current_project_id=item.project_id if item.project_id else existing.current_project_id,
                current_project_name=item.project_name if item.project_name else existing.current_project_name,
            )
            updated += 1
        else:
            try:
                await token_manager.add_token(
                    st=st,
                    project_id=item.project_id,
                    project_name=item.project_name,
                    remark=item.remark,
                    image_enabled=item.image_enabled,
                    video_enabled=item.video_enabled,
                    image_concurrency=item.image_concurrency,
                    video_concurrency=item.video_concurrency,
                )
                added += 1
            except Exception:
                continue
    return {"success": True, "added": added, "updated": updated}


@router.get("/api/logs")
async def get_logs(limit: int = 100, token: str = Depends(verify_admin_token)):
    _require_deps()
    logs = await db.get_request_logs(limit=limit)
    for entry in logs:
        created_at = entry.get("created_at")
        if hasattr(created_at, "isoformat"):
            entry["created_at"] = created_at.isoformat()
    return logs


@router.get("/api/captcha/config")
async def get_captcha_config(token: str = Depends(verify_admin_token)):
    _require_deps()
    cfg = await db.get_captcha_config()
    return {
        "captcha_method": cfg.captcha_method,
        "yescaptcha_api_key": cfg.yescaptcha_api_key,
        "yescaptcha_base_url": cfg.yescaptcha_base_url,
        "browser_proxy_enabled": cfg.browser_proxy_enabled,
        "browser_proxy_url": cfg.browser_proxy_url,
    }


@router.post("/api/captcha/config")
async def update_captcha_config(request: CaptchaConfigRequest, token: str = Depends(verify_admin_token)):
    _require_deps()
    await db.update_captcha_config(
        captcha_method=request.captcha_method,
        yescaptcha_api_key=request.yescaptcha_api_key or "",
        yescaptcha_base_url=request.yescaptcha_base_url or "https://api.yescaptcha.com",
        browser_proxy_enabled=request.browser_proxy_enabled,
        browser_proxy_url=request.browser_proxy_url,
    )

    # hot reload into memory
    config.set_captcha_method(request.captcha_method)
    config.set_yescaptcha_api_key(request.yescaptcha_api_key or "")
    config.set_yescaptcha_base_url(request.yescaptcha_base_url or "https://api.yescaptcha.com")
    config.set_browser_proxy_enabled(request.browser_proxy_enabled)
    config.set_browser_proxy_url(request.browser_proxy_url or "")

    return {"success": True}
