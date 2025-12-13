"""API routes - OpenAI compatible endpoints"""
from fastapi import APIRouter, Depends, HTTPException, Path, UploadFile, File, Form, Header
from fastapi.responses import StreamingResponse, JSONResponse
from datetime import datetime
from typing import List, Optional
import json
import re
from ..core.auth import verify_api_key_header
from ..core.config import config
from ..core.models import ChatCompletionRequest, Task as DbTask
from ..core.database import Database
from ..services.generation_handler import GenerationHandler, MODEL_CONFIG

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

    model, prompt, remix_target_id = _normalize_video_task_request(body)

    try:
        task_id = await generation_handler.create_video_task(
            model=model,
            prompt=prompt,
            remix_target_id=remix_target_id,
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
@router.post("/v1/video/sora-video")
@router.post("/client/v1/video/sora")
@router.post("/client/v1/video/sora-video")
@router.post("/client/video/sora")
@router.post("/client/video/sora-video")
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

@router.post("/v1/video/sora-upload-character")
@router.post("/client/v1/video/sora-upload-character")
@router.post("/client/video/sora-upload-character")
async def upload_character(
    body: dict,
    api_key: str = Depends(verify_api_key_header),
):
    """Trigger character creation from a video URL (non-stream)."""
    if generation_handler is None:
        raise HTTPException(status_code=500, detail="Generation handler not initialized")

    video_url = (body.get("url") or body.get("video") or "").strip()
    if not video_url:
        raise HTTPException(status_code=400, detail="url is required")

    try:
        task_id = await generation_handler.create_character_task(video_url, is_pid=False)
        return {
            "id": task_id,
            "taskId": task_id,
            "status": "running",
            "progress": 0,
        }
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
