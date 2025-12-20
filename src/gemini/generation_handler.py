"""Generation handler for Gemini/Flow (VideoFX/Veo).

Produces OpenAI-compatible streaming chunks (`data: ...\\n\\n`) similar to Sora handler.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from typing import Optional, AsyncGenerator, List, Dict, Any

from ..core.config import config
from ..core.logger import debug_logger
from ..services.file_cache import FileCache
from ..services.concurrency_manager import ConcurrencyManager
from .flow_client import FlowClient
from .token_manager import GeminiTokenManager
from .load_balancer import GeminiLoadBalancer
from .database import GeminiDatabase
from .models import GeminiTask, GeminiRequestLog


# Model configuration (from flow2api)
MODEL_CONFIG: Dict[str, Dict[str, Any]] = {
    "gemini-2.5-flash-image-landscape": {"type": "image", "model_name": "GEM_PIX", "aspect_ratio": "IMAGE_ASPECT_RATIO_LANDSCAPE"},
    "gemini-2.5-flash-image-portrait": {"type": "image", "model_name": "GEM_PIX", "aspect_ratio": "IMAGE_ASPECT_RATIO_PORTRAIT"},
    "gemini-3.0-pro-image-landscape": {"type": "image", "model_name": "GEM_PIX_2", "aspect_ratio": "IMAGE_ASPECT_RATIO_LANDSCAPE"},
    "gemini-3.0-pro-image-portrait": {"type": "image", "model_name": "GEM_PIX_2", "aspect_ratio": "IMAGE_ASPECT_RATIO_PORTRAIT"},
    "imagen-4.0-generate-preview-landscape": {"type": "image", "model_name": "IMAGEN_3_5", "aspect_ratio": "IMAGE_ASPECT_RATIO_LANDSCAPE"},
    "imagen-4.0-generate-preview-portrait": {"type": "image", "model_name": "IMAGEN_3_5", "aspect_ratio": "IMAGE_ASPECT_RATIO_PORTRAIT"},

    # T2V
    "veo_3_1_t2v_fast_portrait": {"type": "video", "video_type": "t2v", "model_key": "veo_3_1_t2v_fast_portrait", "aspect_ratio": "VIDEO_ASPECT_RATIO_PORTRAIT", "supports_images": False},
    "veo_3_1_t2v_fast_landscape": {"type": "video", "video_type": "t2v", "model_key": "veo_3_1_t2v_fast", "aspect_ratio": "VIDEO_ASPECT_RATIO_LANDSCAPE", "supports_images": False},
    "veo_2_1_fast_d_15_t2v_portrait": {"type": "video", "video_type": "t2v", "model_key": "veo_2_1_fast_d_15_t2v", "aspect_ratio": "VIDEO_ASPECT_RATIO_PORTRAIT", "supports_images": False},
    "veo_2_1_fast_d_15_t2v_landscape": {"type": "video", "video_type": "t2v", "model_key": "veo_2_1_fast_d_15_t2v", "aspect_ratio": "VIDEO_ASPECT_RATIO_LANDSCAPE", "supports_images": False},
    "veo_2_0_t2v_portrait": {"type": "video", "video_type": "t2v", "model_key": "veo_2_0_t2v", "aspect_ratio": "VIDEO_ASPECT_RATIO_PORTRAIT", "supports_images": False},
    "veo_2_0_t2v_landscape": {"type": "video", "video_type": "t2v", "model_key": "veo_2_0_t2v", "aspect_ratio": "VIDEO_ASPECT_RATIO_LANDSCAPE", "supports_images": False},

    # I2V (start/end)
    "veo_3_1_i2v_s_fast_fl_portrait": {"type": "video", "video_type": "i2v", "model_key": "veo_3_1_i2v_s_fast_fl", "aspect_ratio": "VIDEO_ASPECT_RATIO_PORTRAIT", "supports_images": True, "min_images": 1, "max_images": 2},
    "veo_3_1_i2v_s_fast_fl_landscape": {"type": "video", "video_type": "i2v", "model_key": "veo_3_1_i2v_s_fast_fl", "aspect_ratio": "VIDEO_ASPECT_RATIO_LANDSCAPE", "supports_images": True, "min_images": 1, "max_images": 2},
    "veo_2_1_fast_d_15_i2v_portrait": {"type": "video", "video_type": "i2v", "model_key": "veo_2_1_fast_d_15_i2v", "aspect_ratio": "VIDEO_ASPECT_RATIO_PORTRAIT", "supports_images": True, "min_images": 1, "max_images": 2},
    "veo_2_1_fast_d_15_i2v_landscape": {"type": "video", "video_type": "i2v", "model_key": "veo_2_1_fast_d_15_i2v", "aspect_ratio": "VIDEO_ASPECT_RATIO_LANDSCAPE", "supports_images": True, "min_images": 1, "max_images": 2},
    "veo_2_0_i2v_portrait": {"type": "video", "video_type": "i2v", "model_key": "veo_2_0_i2v", "aspect_ratio": "VIDEO_ASPECT_RATIO_PORTRAIT", "supports_images": True, "min_images": 1, "max_images": 2},
    "veo_2_0_i2v_landscape": {"type": "video", "video_type": "i2v", "model_key": "veo_2_0_i2v", "aspect_ratio": "VIDEO_ASPECT_RATIO_LANDSCAPE", "supports_images": True, "min_images": 1, "max_images": 2},

    # R2V (reference images)
    "veo_3_0_r2v_fast_portrait": {"type": "video", "video_type": "r2v", "model_key": "veo_3_0_r2v_fast", "aspect_ratio": "VIDEO_ASPECT_RATIO_PORTRAIT", "supports_images": True, "min_images": 0, "max_images": None},
    "veo_3_0_r2v_fast_landscape": {"type": "video", "video_type": "r2v", "model_key": "veo_3_0_r2v_fast", "aspect_ratio": "VIDEO_ASPECT_RATIO_LANDSCAPE", "supports_images": True, "min_images": 0, "max_images": None},
}


class GeminiGenerationHandler:
    def __init__(
        self,
        flow_client: FlowClient,
        token_manager: GeminiTokenManager,
        load_balancer: GeminiLoadBalancer,
        db: GeminiDatabase,
        concurrency_manager: Optional[ConcurrencyManager] = None,
        proxy_manager=None,
    ):
        self.flow_client = flow_client
        self.token_manager = token_manager
        self.load_balancer = load_balancer
        self.db = db
        self.concurrency_manager = concurrency_manager
        self.file_cache = FileCache(cache_dir="tmp", default_timeout=config.cache_timeout, proxy_manager=proxy_manager)

    async def check_token_availability(self, is_image: bool, is_video: bool) -> bool:
        token_obj = await self.load_balancer.select_token(for_image_generation=is_image, for_video_generation=is_video)
        return token_obj is not None

    def _get_base_url(self) -> str:
        base = (config.cache_base_url or "").strip()
        if base:
            return base.rstrip("/")
        host = config.server_host
        if host == "0.0.0.0":
            host = "127.0.0.1"
        return f"http://{host}:{config.server_port}"

    def _create_stream_chunk(self, content: str, finish_reason: Optional[str] = None) -> str:
        payload = {
            "id": f"chatcmpl-{int(time.time() * 1000)}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "gemini",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": content},
                    "finish_reason": finish_reason,
                }
            ],
        }
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def _create_completion_response(self, content: str, media_type: str = "text") -> str:
        text = content
        if media_type == "image":
            text = f"![Generated Image]({content})"
        elif media_type == "video":
            text = f"<video src='{content}' controls style='max-width:100%'></video>"

        payload = {
            "id": f"chatcmpl-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "gemini",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        }
        return json.dumps(payload, ensure_ascii=False)

    def _create_error_response(self, message: str) -> str:
        payload = {"error": {"message": message, "type": "server_error", "param": None, "code": None}}
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    async def handle_generation(
        self,
        model: str,
        prompt: str,
        images: Optional[List[bytes]] = None,
        stream: bool = False,
    ) -> AsyncGenerator[str, None]:
        start_time = time.time()
        token = None
        status_code = 200
        error_message: Optional[str] = None

        if model not in MODEL_CONFIG:
            yield self._create_error_response(f"不支持的模型: {model}")
            yield "data: [DONE]\n\n"
            return

        model_config = MODEL_CONFIG[model]
        generation_type = model_config["type"]

        try:
            # pick token
            token = await self.load_balancer.select_token(
                for_image_generation=(generation_type == "image"),
                for_video_generation=(generation_type == "video"),
            )
            if not token:
                yield self._create_error_response("没有可用的Gemini Token")
                yield "data: [DONE]\n\n"
                return

            if token.id is None:
                yield self._create_error_response("Token ID invalid")
                yield "data: [DONE]\n\n"
                return

            # Ensure AT valid
            await self.token_manager.is_at_valid(token.id)
            token = await self.token_manager.get_token(token.id)
            if not token or not token.at:
                yield self._create_error_response("AT 获取失败")
                yield "data: [DONE]\n\n"
                return

            # update usage
            await self.db.update_token_usage(token.id)

            if generation_type == "image":
                async for chunk in self._handle_image(token, model_config, prompt, images, stream):
                    yield chunk
            else:
                async for chunk in self._handle_video(token, model_config, prompt, images, stream):
                    yield chunk

        except Exception as e:
            status_code = 500
            error_message = str(e)
            debug_logger.log_error(error_message=str(e), status_code=0, response_text="")
            yield self._create_error_response(str(e))
            yield "data: [DONE]\n\n"
            return
        finally:
            duration = time.time() - start_time
            try:
                await self.db.add_request_log(
                    GeminiRequestLog(
                        token_id=(token.id if token else None),
                        operation=f"gemini:{generation_type}:{model}",
                        request_body=json.dumps({"model": model, "prompt": prompt[:200]}, ensure_ascii=False),
                        response_body=(error_message or ""),
                        status_code=status_code,
                        duration=duration,
                    )
                )
            except Exception:
                pass

    async def _handle_image(
        self,
        token,
        model_config: dict,
        prompt: str,
        images: Optional[List[bytes]],
        stream: bool,
    ) -> AsyncGenerator[str, None]:
        if self.concurrency_manager:
            if not await self.concurrency_manager.acquire_image(token.id):
                yield self._create_error_response("图片并发限制已达上限")
                yield "data: [DONE]\n\n"
                return

        try:
            project_id = token.current_project_id
            if not project_id:
                raise Exception("Token 缺少 project_id")

            image_inputs = []
            if images:
                if stream:
                    yield self._create_stream_chunk(f"上传 {len(images)} 张参考图片...\n")
                for idx, img in enumerate(images):
                    media_id = await self.flow_client.upload_image(token.at, img, model_config["aspect_ratio"])
                    image_inputs.append({"name": media_id, "imageInputType": "IMAGE_INPUT_TYPE_REFERENCE"})
                    if stream:
                        yield self._create_stream_chunk(f"已上传第 {idx + 1}/{len(images)} 张图片\n")

            if stream:
                yield self._create_stream_chunk("正在生成图片...\n")

            result = await self.flow_client.generate_image(
                at=token.at,
                project_id=project_id,
                prompt=prompt,
                model_name=model_config["model_name"],
                aspect_ratio=model_config["aspect_ratio"],
                image_inputs=image_inputs,
            )

            media = result.get("media", [])
            if not media:
                yield self._create_error_response("生成结果为空")
                yield "data: [DONE]\n\n"
                return

            image_url = media[0]["image"]["generatedImage"]["fifeUrl"]
            local_url = image_url
            if config.cache_enabled:
                try:
                    if stream:
                        yield self._create_stream_chunk("缓存图片中...\n")
                    cached = await self.file_cache.download_and_cache(image_url, "image")
                    local_url = f"{self._get_base_url()}/tmp/{cached}"
                except Exception as e:
                    debug_logger.log_error(f"Failed to cache image: {str(e)}", status_code=0, response_text="")

            await self.db.increment_token_stats(token.id, image=True, success=True)

            if stream:
                yield self._create_stream_chunk(f"![Generated Image]({local_url})", finish_reason="stop")
                yield "data: [DONE]\n\n"
            else:
                yield self._create_completion_response(local_url, media_type="image")

        except Exception as e:
            await self.db.increment_token_stats(token.id, image=True, error=True)
            await self._maybe_disable_token_on_errors(token.id)
            yield self._create_error_response(str(e))
            yield "data: [DONE]\n\n"
        finally:
            if self.concurrency_manager:
                await self.concurrency_manager.release_image(token.id)

    async def _handle_video(
        self,
        token,
        model_config: dict,
        prompt: str,
        images: Optional[List[bytes]],
        stream: bool,
    ) -> AsyncGenerator[str, None]:
        if self.concurrency_manager:
            if not await self.concurrency_manager.acquire_video(token.id):
                yield self._create_error_response("视频并发限制已达上限")
                yield "data: [DONE]\n\n"
                return

        try:
            project_id = token.current_project_id
            if not project_id:
                raise Exception("Token 缺少 project_id")

            video_type = model_config.get("video_type")
            image_count = len(images) if images else 0

            if video_type == "t2v" and image_count > 0:
                images = None
                image_count = 0

            if video_type == "i2v":
                min_images = model_config.get("min_images", 1)
                max_images = model_config.get("max_images", 2)
                if image_count < min_images or image_count > max_images:
                    raise Exception(f"首尾帧模型需要 {min_images}-{max_images} 张图片,当前提供了 {image_count} 张")

            start_media_id = None
            end_media_id = None
            reference_images = []

            if video_type == "i2v" and images:
                if image_count == 1:
                    if stream:
                        yield self._create_stream_chunk("上传首帧图片...\n")
                    start_media_id = await self.flow_client.upload_image(token.at, images[0], model_config["aspect_ratio"])
                else:
                    if stream:
                        yield self._create_stream_chunk("上传首帧和尾帧图片...\n")
                    start_media_id = await self.flow_client.upload_image(token.at, images[0], model_config["aspect_ratio"])
                    end_media_id = await self.flow_client.upload_image(token.at, images[1], model_config["aspect_ratio"])

            elif video_type == "r2v" and images:
                if stream:
                    yield self._create_stream_chunk(f"上传 {image_count} 张参考图片...\n")
                for img in images:
                    media_id = await self.flow_client.upload_image(token.at, img, model_config["aspect_ratio"])
                    reference_images.append({"imageUsageType": "IMAGE_USAGE_TYPE_ASSET", "mediaId": media_id})

            if stream:
                yield self._create_stream_chunk("提交视频生成任务...\n")

            if video_type == "i2v" and start_media_id:
                if end_media_id:
                    result = await self.flow_client.generate_video_start_end(
                        at=token.at,
                        project_id=project_id,
                        prompt=prompt,
                        model_key=model_config["model_key"],
                        aspect_ratio=model_config["aspect_ratio"],
                        start_media_id=start_media_id,
                        end_media_id=end_media_id,
                        user_paygate_tier=token.user_paygate_tier or "PAYGATE_TIER_ONE",
                    )
                else:
                    result = await self.flow_client.generate_video_start_image(
                        at=token.at,
                        project_id=project_id,
                        prompt=prompt,
                        model_key=model_config["model_key"],
                        aspect_ratio=model_config["aspect_ratio"],
                        start_media_id=start_media_id,
                        user_paygate_tier=token.user_paygate_tier or "PAYGATE_TIER_ONE",
                    )
            elif video_type == "r2v" and reference_images:
                result = await self.flow_client.generate_video_reference_images(
                    at=token.at,
                    project_id=project_id,
                    prompt=prompt,
                    model_key=model_config["model_key"],
                    aspect_ratio=model_config["aspect_ratio"],
                    reference_images=reference_images,
                    user_paygate_tier=token.user_paygate_tier or "PAYGATE_TIER_ONE",
                )
            else:
                result = await self.flow_client.generate_video_text(
                    at=token.at,
                    project_id=project_id,
                    prompt=prompt,
                    model_key=model_config["model_key"],
                    aspect_ratio=model_config["aspect_ratio"],
                    user_paygate_tier=token.user_paygate_tier or "PAYGATE_TIER_ONE",
                )

            operations = result.get("operations", [])
            if not operations:
                raise Exception("生成任务创建失败")

            operation = operations[0]
            task_id = operation["operation"]["name"]
            scene_id = operation.get("sceneId")
            await self.db.create_task(
                GeminiTask(task_id=task_id, token_id=token.id, model=model_config["model_key"], prompt=prompt, status="processing", scene_id=scene_id)
            )

            if stream:
                yield self._create_stream_chunk("视频生成中...\n")

            async for chunk in self._poll_video_result(token, operations, stream):
                yield chunk

        except Exception as e:
            await self.db.increment_token_stats(token.id, video=True, error=True)
            await self._maybe_disable_token_on_errors(token.id)
            yield self._create_error_response(str(e))
            yield "data: [DONE]\n\n"
        finally:
            if self.concurrency_manager:
                await self.concurrency_manager.release_video(token.id)

    async def _maybe_disable_token_on_errors(self, token_id: int):
        """Auto-disable token when consecutive errors exceed threshold."""
        try:
            threshold = await self.db.get_error_ban_threshold()
            consecutive = await self.db.get_consecutive_error_count(token_id)
            if threshold > 0 and consecutive >= threshold:
                await self.db.update_token(
                    token_id,
                    is_active=0,
                    ban_reason="error_threshold",
                    banned_at=datetime.utcnow(),
                )
        except Exception:
            return None

    async def _poll_video_result(self, token, operations: List[Dict[str, Any]], stream: bool) -> AsyncGenerator[str, None]:
        max_attempts = config.flow_max_poll_attempts
        poll_interval = config.flow_poll_interval

        for attempt in range(max_attempts):
            await asyncio.sleep(poll_interval)
            try:
                result = await self.flow_client.check_video_status(token.at, operations)
                checked_operations = result.get("operations", [])
                if not checked_operations:
                    continue

                op = checked_operations[0]
                status = op.get("status")

                progress_update_interval = max(1, int(20 / max(poll_interval, 0.1)))
                if stream and attempt % progress_update_interval == 0:
                    progress = min(int((attempt / max_attempts) * 100), 95)
                    yield self._create_stream_chunk(f"生成进度: {progress}%\n")

                if status == "MEDIA_GENERATION_STATUS_SUCCESSFUL":
                    metadata = op.get("operation", {}).get("metadata", {}) or {}
                    video_info = metadata.get("video", {}) or {}
                    video_url = video_info.get("fifeUrl")
                    if not video_url:
                        raise Exception("视频URL为空")

                    local_url = video_url
                    if config.cache_enabled:
                        try:
                            if stream:
                                yield self._create_stream_chunk("正在缓存视频文件...\n")
                            cached_filename = await self.file_cache.download_and_cache(video_url, "video")
                            local_url = f"{self._get_base_url()}/tmp/{cached_filename}"
                        except Exception as e:
                            debug_logger.log_error(f"Failed to cache video: {str(e)}", status_code=0, response_text="")
                            local_url = video_url

                    task_id = op.get("operation", {}).get("name")
                    if task_id:
                        await self.db.update_task(task_id, status="completed", progress=100, result_urls=[local_url], completed_at=time.time())

                    await self.db.increment_token_stats(token.id, video=True, success=True)

                    if stream:
                        yield self._create_stream_chunk(
                            f"<video src='{local_url}' controls style='max-width:100%'></video>",
                            finish_reason="stop",
                        )
                        yield "data: [DONE]\n\n"
                    else:
                        yield self._create_completion_response(local_url, media_type="video")
                    return

                if status and str(status).startswith("MEDIA_GENERATION_STATUS_ERROR"):
                    raise Exception(f"视频生成失败: {status}")

            except Exception as e:
                debug_logger.log_error(f"Poll error: {str(e)}", status_code=0, response_text="")
                continue

        yield self._create_error_response(f"视频生成超时 (已轮询{max_attempts}次)")
        yield "data: [DONE]\n\n"
