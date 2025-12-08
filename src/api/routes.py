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

@router.post("/v1/video/tasks")
async def create_video_task(
    body: dict,
    api_key: str = Depends(verify_api_key_header),
):
    """
    Create a video generation task (non-streaming).

    Expected payload:
    {
      "model": "sora-video-landscape-10s",
      "prompt": "...",
      "durationSeconds": 10,
      "orientation": "landscape"
    }
    """
    if generation_handler is None:
        raise HTTPException(status_code=500, detail="Generation handler not initialized")

    model = body.get("model")
    prompt = body.get("prompt") or ""
    if not model or not isinstance(model, str):
        raise HTTPException(status_code=400, detail="model is required")
    if model not in MODEL_CONFIG:
        raise HTTPException(status_code=400, detail=f"Invalid model: {model}")
    if MODEL_CONFIG[model]["type"] != "video":
        raise HTTPException(status_code=400, detail="Only video models are supported by this endpoint")
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    # 当前 durationSeconds/orientation 主要由 model 决定；保持参数兼容但暂不细化控制
    try:
        task_id = await generation_handler.create_video_task(model=model, prompt=prompt)
        return {
            "id": task_id,
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
        "status": status,
        "progress": progress,
        "video_url": video_url,
        "thumbnail_url": thumbnail_url,
        "content": content,
        "error": task.error_message,
        "upstream": upstream,
        "upstream_raw": upstream_raw,
    }

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
