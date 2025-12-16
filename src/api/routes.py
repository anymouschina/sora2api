"""API routes - OpenAI compatible endpoints"""
import asyncio
from fastapi import APIRouter, Depends, HTTPException, Path, UploadFile, File, Form, Header
from fastapi.responses import StreamingResponse, JSONResponse
from typing import List, Optional, Dict, Any
import json
import re
import time
from ..core.auth import verify_api_key_header
from ..core.config import config
from ..core.models import ChatCompletionRequest, Task as DbTask
from ..core.database import Database
from ..services.generation_handler import GenerationHandler, MODEL_CONFIG
from ..core.logger import debug_logger

router = APIRouter()

# Dependency injection will be set up in main.py
generation_handler: GenerationHandler = None

def set_generation_handler(handler: GenerationHandler):
    """Set generation handler instance"""
    global generation_handler
    generation_handler = handler

def _extract_remix_id(text: str) -> str:
    """Extract remix ID from text

    Supports two formats:
    1. Full URL: https://sora.chatgpt.com/p/s_68e3a06dcd888191b150971da152c1f5
    2. Short ID: s_68e3a06dcd888191b150971da152c1f5

    Args:
        text: Text to search for remix ID

    Returns:
        Remix ID (s_[a-f0-9]{32}) or empty string if not found
    """
    if not text:
        return ""

    # Match Sora share link format: s_[a-f0-9]{32}
    match = re.search(r's_[a-f0-9]{32}', text)
    if match:
        return match.group(0)

    return ""

def _normalize_orientation(body: dict) -> str:
    """Infer video orientation from explicit field or aspect ratio."""
    orientation_raw = str(body.get("orientation") or "").lower()
    aspect_ratio_raw = str(body.get("aspectRatio") or body.get("aspect_ratio") or "").lower()

    if orientation_raw in ["portrait", "vertical"]:
        return "portrait"
    if orientation_raw in ["landscape", "horizontal", "wide", "widescreen"]:
        return "landscape"

    ratio_value = None
    if aspect_ratio_raw:
        ratio_match = re.match(r"(\d+(?:\.\d+)?)[/:](\d+(?:\.\d+)?)", aspect_ratio_raw)
        if ratio_match:
            try:
                width = float(ratio_match.group(1))
                height = float(ratio_match.group(2))
                if height > 0:
                    ratio_value = width / height
            except Exception:
                ratio_value = None
        else:
            try:
                ratio_value = float(aspect_ratio_raw)
            except Exception:
                ratio_value = None

    if ratio_value is not None:
        return "portrait" if ratio_value < 1 else "landscape"

    return "landscape"

def _normalize_duration_seconds(body: dict) -> int:
    """Pick duration bucket (10s/15s) from request body."""
    duration_raw = body.get("durationSeconds")
    if duration_raw is None:
        duration_raw = body.get("duration")

    duration_val: Optional[float] = None
    if isinstance(duration_raw, (int, float)):
        duration_val = float(duration_raw)
    elif isinstance(duration_raw, str) and duration_raw.strip():
        try:
            duration_val = float(duration_raw.strip())
        except ValueError:
            duration_val = None

    # Only 10s and 15s variants are supported in MODEL_CONFIG
    if duration_val is not None and duration_val > 10:
        return 15
    return 10

def _normalize_video_model_key(model_raw: Optional[str], orientation: str, duration_seconds: int) -> Optional[str]:
    """Map incoming model name to an internal MODEL_CONFIG key."""
    duration_bucket = 15 if duration_seconds > 10 else 10
    if model_raw is None:
        normalized = ""
    elif isinstance(model_raw, str):
        normalized = model_raw.strip()
    else:
        normalized = str(model_raw).strip()

    model_key = normalized.lower()
    if model_key in MODEL_CONFIG:
        return model_key

    # Handle common aliases
    alias_orientation = None
    if model_key in {"sora", "sora-2", "sora2", "sora-video", "sora-video-s"}:
        alias_orientation = None
    elif model_key in {"sora-video-portrait", "sora-video-vertical"}:
        alias_orientation = "portrait"
    elif model_key == "sora-video-landscape":
        alias_orientation = "landscape"
    elif model_key in {"sora-video-10s", "sora-video-15s"}:
        # Orientation-less variants already exist in MODEL_CONFIG
        return model_key
    else:
        # Try to parse names like sora-video-landscape-10 or sora-video-portrait-15s
        match = re.match(r"sora-video-(portrait|landscape)-?(\d+)", model_key)
        if match:
            parsed_orientation = match.group(1)
            parsed_duration = int(match.group(2))
            duration_bucket = 15 if parsed_duration > 10 else 10
            candidate = f"sora-video-{parsed_orientation}-{duration_bucket}s"
            if candidate in MODEL_CONFIG:
                return candidate

    if alias_orientation:
        orientation = alias_orientation

    candidate = f"sora-video-{orientation}-{duration_bucket}s"
    if candidate in MODEL_CONFIG:
        return candidate

    fallback = f"sora-video-{duration_bucket}s"
    if fallback in MODEL_CONFIG:
        return fallback

    return None

def _normalize_video_task_request(body: dict) -> tuple[str, str, Optional[str]]:
    """Normalize incoming video task payload to internal parameters."""
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Invalid request body")

    prompt_raw = body.get("prompt")
    if isinstance(prompt_raw, str):
        prompt = prompt_raw.strip()
    elif prompt_raw is None:
        prompt = ""
    else:
        prompt = str(prompt_raw).strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    # Remix/post id compatibility
    remix_target_id = None
    for key in ["remixTargetId", "remix_target_id", "pid", "postId", "post_id"]:
        candidate = body.get(key)
        if isinstance(candidate, str) and candidate.strip():
            remix_target_id = _extract_remix_id(candidate) or candidate.strip()
            break
    if not remix_target_id:
        remix_target_id = _extract_remix_id(prompt)

    orientation = _normalize_orientation(body)
    duration_seconds = _normalize_duration_seconds(body)
    model = _normalize_video_model_key(body.get("model"), orientation, duration_seconds)
    if not model:
        raise HTTPException(status_code=400, detail="Invalid model or unsupported duration/orientation")

    return model, prompt, remix_target_id

def _format_character_task(task: DbTask):
    """Format character task for polling responses."""
    raw_status = (task.status or "").lower()
    if raw_status in ["completed", "succeeded"]:
        status = "succeeded"
    elif raw_status in ["failed", "error"]:
        status = "failed"
    else:
        status = "running"

    progress = int(task.progress or 0)
    progress = max(0, min(progress, 100))

    character_id = None
    username = None
    display_name = None
    if task.result_urls:
        try:
            results = json.loads(task.result_urls)
            if isinstance(results, list) and results:
                entry = results[0]
                if isinstance(entry, dict):
                    character_id = entry.get("character_id")
                    username = entry.get("username")
                    display_name = entry.get("display_name")
        except Exception:
            pass

    payload = {
        "id": task.task_id,
        "status": status,
        "progress": progress,
        "results": [],
        "error": task.error_message,
    }
    if character_id:
        payload["results"] = [{
            "character_id": character_id,
            "username": username,
            "display_name": display_name,
        }]
    return payload

def _build_failure_reason(error_message: Optional[str]) -> str:
    """Map error message to a simple failure reason code."""
    if not error_message:
        return ""
    lowered = error_message.lower()
    if "input" in lowered and "moderation" in lowered:
        return "input_moderation"
    if "output" in lowered and "moderation" in lowered:
        return "output_moderation"
    if "content blocked" in lowered:
        return "output_moderation"
    return "error"

def _to_bool(value: Any, default: bool = False) -> bool:
    """Coerce incoming payload value to boolean."""
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in ["1", "true", "yes", "on"]
    return bool(value)

def _format_sora2_video_payload(task: DbTask) -> Dict[str, Any]:
    """Format DB task into Sora2 video API response payload."""
    raw_status = (task.status or "").lower()
    if raw_status in ["completed", "succeeded"]:
        status = "succeeded"
    elif raw_status in ["failed", "error"]:
        status = "failed"
    else:
        status = "running"

    progress_raw = task.progress or 0
    progress = int(progress_raw * 100) if progress_raw <= 1 else int(progress_raw)
    progress = max(0, min(progress, 100))

    results: List[Dict[str, Any]] = []
    if task.result_urls:
        try:
            parsed = json.loads(task.result_urls)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, str):
                        url_val = item.strip()
                        if url_val:
                            results.append({
                                "url": url_val,
                                "removeWatermark": False,
                                "pid": task.post_id
                            })
                    elif isinstance(item, dict):
                        url_val = item.get("url")
                        if url_val:
                            results.append({
                                "url": url_val,
                                "removeWatermark": bool(item.get("removeWatermark", False)),
                                "pid": item.get("pid") or task.post_id
                            })
        except Exception:
            # Ignore malformed JSON, fallback to empty results
            pass

    failure_reason = _build_failure_reason(task.error_message) if status == "failed" else ""

    return {
        "id": task.task_id,
        "results": results,
        "progress": progress,
        "status": status,
        "failure_reason": failure_reason,
        "error": task.error_message or ""
    }

def _normalize_video_task_request_extended(
    raw_body: dict,
    apply_default_portrait: bool = False
) -> tuple[str, str, Optional[str], str, str]:
    """Normalize video request with optional defaults and extract reference image/size."""
    body = dict(raw_body) if isinstance(raw_body, dict) else {}

    if apply_default_portrait and (
        "aspectRatio" not in body and "aspect_ratio" not in body and "orientation" not in body
    ):
        body["aspectRatio"] = "9:16"

    model, prompt, remix_target_id = _normalize_video_task_request(body)

    reference_url_raw = body.get("url")
    reference_url = reference_url_raw.strip() if isinstance(reference_url_raw, str) else (
        str(reference_url_raw).strip() if reference_url_raw is not None else ""
    )

    size_raw = body.get("size") or "small"
    size = "large" if str(size_raw).lower() == "large" else "small"

    return model, prompt, remix_target_id, reference_url, size

async def _stream_sora2_video(task_id: str, shut_progress: bool = False):
    """Server-sent events stream for Sora2 video tasks."""
    db: Database = generation_handler.db  # type: ignore[attr-defined]
    poll_interval = max(float(config.poll_interval), 1.0)

    while True:
        task = await db.get_task(task_id)
        if not task:
            error_payload = {
                "id": task_id,
                "results": [],
                "progress": 0,
                "status": "failed",
                "failure_reason": "error",
                "error": "Task not found"
            }
            yield f"data: {json.dumps(error_payload)}\n\n"
            yield "data: [DONE]\n\n"
            return

        payload = _format_sora2_video_payload(task)
        is_final = payload["status"] in ["succeeded", "failed"]

        if not shut_progress or is_final:
            yield f"data: {json.dumps(payload)}\n\n"

        if is_final:
            yield "data: [DONE]\n\n"
            return

        await asyncio.sleep(poll_interval)

async def _post_webhook_payload(web_hook: str, payload: Dict[str, Any]):
    """Send webhook payload with basic error logging."""
    from curl_cffi.requests import AsyncSession

    try:
        async with AsyncSession() as session:
            await session.post(
                web_hook,
                json=payload,
                headers={"Content-Type": "application/json"},
                impersonate="chrome",
                timeout=20,
            )
    except Exception as exc:
        debug_logger.log_error(
            error_message=f"Failed to POST webhook to {web_hook}: {str(exc)}",
            status_code=500,
            response_text=str(exc),
        )

async def _dispatch_webhook_updates(task_id: str, web_hook: str, shut_progress: bool = False):
    """Poll task status and dispatch updates to webhook."""
    if generation_handler is None:
        return

    db: Database = generation_handler.db  # type: ignore[attr-defined]
    poll_interval = max(float(config.poll_interval), 1.0)
    last_progress: Optional[int] = None

    try:
        while True:
            task = await db.get_task(task_id)
            if not task:
                await _post_webhook_payload(web_hook, {
                    "id": task_id,
                    "results": [],
                    "progress": 0,
                    "status": "failed",
                    "failure_reason": "error",
                    "error": "Task not found"
                })
                return

            payload = _format_sora2_video_payload(task)
            is_final = payload["status"] in ["succeeded", "failed"]
            should_send = (not shut_progress or is_final)

            if should_send:
                if is_final or last_progress is None or payload["progress"] != last_progress:
                    await _post_webhook_payload(web_hook, payload)

            if is_final:
                return

            last_progress = payload["progress"]
            await asyncio.sleep(poll_interval)
    except Exception as exc:
        debug_logger.log_error(
            error_message=f"Webhook dispatcher failed for task {task_id}: {str(exc)}",
            status_code=500,
            response_text=str(exc),
        )

async def _get_sora2_task_response(task_id: str) -> Dict[str, Any]:
    """Fetch task from DB, refresh progress once, and format payload."""
    if generation_handler is None:
        raise HTTPException(status_code=500, detail="Generation handler not initialized")

    db: Database = generation_handler.db  # type: ignore[attr-defined]
    task = await db.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    try:
        refreshed_task, _ = await generation_handler.refresh_video_task_progress(task)  # type: ignore[attr-defined]
        task = refreshed_task or task
    except Exception:
        # Ignore refresh errors, fallback to current DB state
        pass

    return _format_sora2_video_payload(task)

@router.get("/v1/models")
async def list_models(api_key: str = Depends(verify_api_key_header)):
    """List available models"""
    models = []
    
    for model_id, config in MODEL_CONFIG.items():
        description = f"{config['type'].capitalize()} generation"
        if config['type'] == 'image':
            description += f" - {config['width']}x{config['height']}"
        else:
            description += f" - {config['orientation']}"
        
        models.append({
            "id": model_id,
            "object": "model",
            "owned_by": "sora2api",
            "description": description
        })
    
    return {
        "object": "list",
        "data": models
    }

@router.post("/backend/project_y/file/upload")
async def upload_file_passthrough(
    file: UploadFile = File(...),
    use_case: str = Form("profile"),
    token_id: Optional[int] = Form(None),
    authorization: Optional[str] = Header(None),
):
    """
    Passthrough for Sora-style file upload.

    Token selection priority:
    1) token_id form field (must be active)
    2) Authorization: Bearer <token> header (kept for compatibility)
    3) Auto-pick from token pool (image-enabled)

    Returns upstream JSON and status code (typically includes `file_id`, optional `asset_pointer`/`url`).
    """
    if generation_handler is None:
        raise HTTPException(status_code=500, detail="Generation handler not initialized")

    # Pick token
    token_str: Optional[str] = None
    token_obj = None

    if token_id is not None:
        token_obj = await generation_handler.token_manager.db.get_token(token_id)
        if not token_obj or not token_obj.is_active:
            raise HTTPException(status_code=400, detail="Invalid or inactive token_id")
        token_str = token_obj.token
    else:
        # Strip Bearer prefix if present
        header_token = None
        if authorization:
            header_token = authorization
            if authorization.lower().startswith("bearer "):
                header_token = authorization.split(" ", 1)[1].strip()
            header_token = header_token or None

        # If header token equals our API key, treat it as client auth, not upstream token
        if header_token and header_token == config.api_key:
            header_token = None

        if header_token:
            token_str = header_token
        else:
            token_obj = await generation_handler.load_balancer.select_token(for_image_generation=True)
            if not token_obj:
                raise HTTPException(status_code=503, detail="No available tokens for upload")
            token_str = token_obj.token

    try:
        file_bytes = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to read uploaded file")

    status_code, resp = await generation_handler.upload_file_passthrough(
        token=token_str,
        file_bytes=file_bytes,
        filename=file.filename or "image.png",
        use_case=use_case or "profile",
    )

    # Lightweight accounting if we used a pooled token
    if token_obj:
        try:
            await generation_handler.token_manager.record_usage(token_obj.id, is_video=False)
        except Exception:
            pass

    return JSONResponse(status_code=status_code, content=resp)

async def _create_video_task_common(body: dict):
    """Shared video-task creation logic (used by multiple compatible routes)."""
    if generation_handler is None:
        raise HTTPException(status_code=500, detail="Generation handler not initialized")

    model, prompt, remix_target_id, reference_url, size = _normalize_video_task_request_extended(
        body,
        apply_default_portrait=True,
    )

    try:
        task_id = await generation_handler.create_video_task(
            model=model,
            prompt=prompt,
            remix_target_id=remix_target_id,
            image=reference_url or None,
            size=size,
        )
        response = {
            "id": task_id,
            "taskId": task_id,  # grsai-compatible field
            "status": "running",
            "progress": 0,
        }
        if remix_target_id:
            response["pid"] = remix_target_id
            response["remixTargetId"] = remix_target_id
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/v1/video/tasks")
@router.post("/v1/video/sora")
@router.post("/client/v1/video/sora")
@router.post("/client/video/sora")
async def create_video_task(
    body: dict,
    api_key: str = Depends(verify_api_key_header),
):
    """
    Create a video generation task (non-streaming).

    This handler is lenient with model/duration/orientation aliases to match
    different sora2api/grsai client expectations.
    """
    return await _create_video_task_common(body)

@router.post("/v1/video/sora-video")
@router.post("/client/v1/video/sora-video")
@router.post("/client/video/sora-video")
async def create_sora2_video_task(
    body: dict,
    api_key: str = Depends(verify_api_key_header),
):
    """Sora2 video endpoint with stream/webhook support."""
    if generation_handler is None:
        raise HTTPException(status_code=500, detail="Generation handler not initialized")

    model, prompt, remix_target_id, reference_url, size = _normalize_video_task_request_extended(
        body,
        apply_default_portrait=True,
    )

    # Payload fields
    web_hook_raw = body.get("webHook")
    web_hook = web_hook_raw.strip() if isinstance(web_hook_raw, str) else (str(web_hook_raw).strip() if web_hook_raw is not None else "")
    shut_progress = _to_bool(body.get("shutProgress"), False)
    stream = _to_bool(body.get("stream"), True)

    try:
        task_id = await generation_handler.create_video_task(
            model=model,
            prompt=prompt,
            remix_target_id=remix_target_id,
            image=reference_url or None,
            size=size,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Webhook mode (ack then push updates)
    if web_hook:
        if web_hook != "-1":
            asyncio.create_task(_dispatch_webhook_updates(task_id, web_hook, shut_progress))
        return {
            "code": 0,
            "msg": "success",
            "data": {
                "id": task_id
            }
        }

    # Polling mode (immediate payload)
    if not stream:
        db: Database = generation_handler.db  # type: ignore[attr-defined]
        task = await db.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        return _format_sora2_video_payload(task)

    async def event_stream():
        async for chunk in _stream_sora2_video(task_id, shut_progress):
            yield chunk

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@router.get("/v1/video/sora-video/{task_id}")
async def get_sora2_video_task(
    task_id: str = Path(..., description="Sora video task id"),
    api_key: str = Depends(verify_api_key_header),
):
    """Get Sora2 video task progress/result in unified format."""
    return await _get_sora2_task_response(task_id)

@router.post("/v1/video/sora-video/result")
async def get_sora2_video_result(
    body: dict,
    api_key: str = Depends(verify_api_key_header),
):
    """Polling endpoint for Sora2 video tasks (body: {\"id\": \"...\"})."""
    task_id_raw = body.get("id") or body.get("taskId") or body.get("task_id") or ""
    task_id = str(task_id_raw).strip()
    if not task_id:
        raise HTTPException(status_code=400, detail="id is required")
    return await _get_sora2_task_response(task_id)

@router.post("/v1/video/sora-upload-character")
@router.post("/client/v1/video/sora-upload-character")
@router.post("/client/video/sora-upload-character")
async def upload_character(
    body: dict,
    api_key: str = Depends(verify_api_key_header),
):
    """Trigger character creation from a video URL (non-blocking by default).

    Optional: pass wait=true to block until completion.
    """
    if generation_handler is None:
        raise HTTPException(status_code=500, detail="Generation handler not initialized")

    video_url = (body.get("url") or body.get("video") or "").strip()
    if not video_url:
        raise HTTPException(status_code=400, detail="url is required")

    wait_for_result = _to_bool(body.get("wait"), False)

    try:
        task_id = await generation_handler.create_character_task(video_url, is_pid=False)

        if not wait_for_result:
            return {
                "id": task_id,
                "taskId": task_id,
                "status": "running",
                "progress": 0,
            }

        # Wait for completion (poll DB)
        db: Database = generation_handler.db  # type: ignore[attr-defined]
        timeout = max(getattr(config, "video_timeout", 600), 60)
        start = time.time()
        poll_interval = 2.0

        while True:
            task = await db.get_task(task_id)
            if task:
                status_lower = (task.status or "").lower()
                if status_lower in ["completed", "succeeded"]:
                    return _format_character_task(task)
                if status_lower in ["failed", "error"]:
                    raise HTTPException(status_code=500, detail=task.error_message or "Character creation failed")

            if time.time() - start > timeout:
                raise HTTPException(status_code=504, detail="Character creation timed out")

            await asyncio.sleep(poll_interval)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/v1/video/sora-create-character")
@router.post("/client/v1/video/sora-create-character")
@router.post("/client/video/sora-create-character")
@router.post("/v1/video/characters/create")
@router.post("/client/v1/video/characters/create")
@router.post("/client/video/characters/create")
async def create_character_from_pid(
    body: dict,
    api_key: str = Depends(verify_api_key_header),
):
    """Create character using a published pid (non-stream)."""
    if generation_handler is None:
        raise HTTPException(status_code=500, detail="Generation handler not initialized")

    pid_raw = (
        body.get("pid")
        or body.get("postId")
        or body.get("post_id")
        or body.get("id")
        or ""
    )
    pid = _extract_remix_id(pid_raw) if isinstance(pid_raw, str) else None
    if not pid:
        raise HTTPException(status_code=400, detail="pid is required")

    try:
        task_id = await generation_handler.create_character_task(pid, is_pid=True)
        return {
            "id": task_id,
            "taskId": task_id,
            "pid": pid,
            "status": "running",
            "progress": 0,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/v1/video/tasks/{task_id}")
async def get_video_task(
    task_id: str = Path(..., description="Sora video task id"),
    api_key: str = Depends(verify_api_key_header),
):
    """Query video task status and result by task id."""
    if generation_handler is None:
        raise HTTPException(status_code=500, detail="Generation handler not initialized")

    db: Database = generation_handler.db  # type: ignore[attr-defined]
    task: Optional[DbTask] = await db.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # 优先拉取上游最新进度并同步至 DB（完成/失败状态仍由后台轮询决定）
    upstream = None
    upstream_raw = None
    try:
        task, pending = await generation_handler.refresh_video_task_progress(task)  # type: ignore[arg-type]
        if pending:
            # 只挑选一部分关键字段作为上游可见信息
            upstream = {
                "status": pending.get("status"),
                "progress_pct": pending.get("progress_pct"),
                "progress_pos_in_queue": pending.get("progress_pos_in_queue"),
                "estimated_queue_wait_time": pending.get("estimated_queue_wait_time"),
                "queue_status_message": pending.get("queue_status_message"),
                "priority": pending.get("priority"),
            }
            # 同时保留完整的 pending 原始数据，便于排查问题
            upstream_raw = pending
    except Exception:
        # 刷新失败时不影响原有逻辑，继续使用 DB 中的状态
        pass

    # Map internal status to external
    raw_status = (task.status or "").lower()
    if raw_status in ["completed", "succeeded"]:
        status = "succeeded"
    elif raw_status in ["failed", "error"]:
        status = "failed"
    elif raw_status in ["processing", "running"]:
        status = "running"
    else:
        status = "running"

    progress = int((task.progress or 0) * 100) if task.progress is not None and task.progress <= 1 else int(task.progress or 0)
    progress = max(0, min(progress, 100))

    video_url = None
    thumbnail_url = None
    content = None
    if task.result_urls:
        try:
            urls = json.loads(task.result_urls)
            if isinstance(urls, list) and urls:
                video_url = urls[0]
        except Exception:
            pass

    return {
        "id": task.task_id,
        "post_id": task.post_id,
        "status": status,
        "progress": progress,
        "video_url": video_url,
        "thumbnail_url": thumbnail_url,
        "content": content,
        "error": task.error_message,
        "upstream": upstream,
        "upstream_raw": upstream_raw,
    }

@router.post("/v1/draw/result")
async def get_draw_result(
    body: dict,
    api_key: str = Depends(verify_api_key_header),
):
    """
    grsai-style polling endpoint: accepts {"id": "<task_id>"} and returns the
    same payload as GET /v1/video/tasks/{id}.
    """
    task_id_raw = body.get("id") or body.get("taskId") or body.get("task_id") or ""
    if not isinstance(task_id_raw, str):
        task_id_raw = str(task_id_raw)
    task_id = task_id_raw.strip()
    if not task_id:
        raise HTTPException(status_code=400, detail="id is required")
    db: Database = generation_handler.db  # type: ignore[attr-defined]
    task: Optional[DbTask] = await db.get_task(task_id)
    if task and (task.model or "").startswith("sora-character"):
        return _format_character_task(task)
    return await get_video_task(task_id=task_id, api_key=api_key)

@router.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    api_key: str = Depends(verify_api_key_header)
):
    """Create chat completion (unified endpoint for image and video generation)"""
    try:
        # Extract prompt from messages
        if not request.messages:
            raise HTTPException(status_code=400, detail="Messages cannot be empty")

        last_message = request.messages[-1]
        content = last_message.content

        # Handle both string and array format (OpenAI multimodal)
        prompt = ""
        image_data = request.image  # Default to request.image if provided
        video_data = request.video  # Video parameter
        remix_target_id = request.remix_target_id  # Remix target ID

        if isinstance(content, str):
            # Simple string format
            prompt = content
            # Extract remix_target_id from prompt if not already provided
            if not remix_target_id:
                remix_target_id = _extract_remix_id(prompt)
        elif isinstance(content, list):
            # Array format (OpenAI multimodal)
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        prompt = item.get("text", "")
                        # Extract remix_target_id from prompt if not already provided
                        if not remix_target_id:
                            remix_target_id = _extract_remix_id(prompt)
                    elif item.get("type") == "image_url":
                        # Extract base64 image from data URI
                        image_url = item.get("image_url", {})
                        url = image_url.get("url", "")
                        if url.startswith("data:image"):
                            # Extract base64 data from data URI
                            if "base64," in url:
                                image_data = url.split("base64,", 1)[1]
                            else:
                                image_data = url
                    elif item.get("type") == "video_url":
                        # Extract video from video_url
                        video_url = item.get("video_url", {})
                        url = video_url.get("url", "")
                        if url.startswith("data:video") or url.startswith("data:application"):
                            # Extract base64 data from data URI
                            if "base64," in url:
                                video_data = url.split("base64,", 1)[1]
                            else:
                                video_data = url
                        else:
                            # It's a URL, pass it as-is (will be downloaded in generation_handler)
                            video_data = url
        else:
            raise HTTPException(status_code=400, detail="Invalid content format")

        # Validate model
        if request.model not in MODEL_CONFIG:
            raise HTTPException(status_code=400, detail=f"Invalid model: {request.model}")

        # Check if this is a video model
        model_config = MODEL_CONFIG[request.model]
        is_video_model = model_config["type"] == "video"

        # For video models with video parameter, we need streaming
        if is_video_model and (video_data or remix_target_id):
            if not request.stream:
                # Non-streaming mode: only check availability
                result = None
                async for chunk in generation_handler.handle_generation(
                    model=request.model,
                    prompt=prompt,
                    image=image_data,
                    video=video_data,
                    remix_target_id=remix_target_id,
                    stream=False
                ):
                    result = chunk

                if result:
                    import json
                    return JSONResponse(content=json.loads(result))
                else:
                    return JSONResponse(
                        status_code=500,
                        content={
                            "error": {
                                "message": "Availability check failed",
                                "type": "server_error",
                                "param": None,
                                "code": None
                            }
                        }
                    )

        # Handle streaming
        if request.stream:
            async def generate():
                import json as json_module  # Import inside function to avoid scope issues
                try:
                    async for chunk in generation_handler.handle_generation(
                        model=request.model,
                        prompt=prompt,
                        image=image_data,
                        video=video_data,
                        remix_target_id=remix_target_id,
                        stream=True
                    ):
                        yield chunk
                except Exception as e:
                    # Return OpenAI-compatible error format
                    error_response = {
                        "error": {
                            "message": str(e),
                            "type": "server_error",
                            "param": None,
                            "code": None
                        }
                    }
                    error_chunk = f'data: {json_module.dumps(error_response)}\n\n'
                    yield error_chunk
                    yield 'data: [DONE]\n\n'

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        else:
            # Non-streaming response (availability check only)
            result = None
            async for chunk in generation_handler.handle_generation(
                model=request.model,
                prompt=prompt,
                image=image_data,
                video=video_data,
                remix_target_id=remix_target_id,
                stream=False
            ):
                result = chunk

            if result:
                import json
                return JSONResponse(content=json.loads(result))
            else:
                # Return OpenAI-compatible error format
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": {
                            "message": "Availability check failed",
                            "type": "server_error",
                            "param": None,
                            "code": None
                        }
                    }
                )

    except Exception as e:
        # Return OpenAI-compatible error format
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": str(e),
                    "type": "server_error",
                    "param": None,
                    "code": None
                }
            }
        )
