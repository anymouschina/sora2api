"""Flow API Client for VideoFX (Veo)"""
import time
import uuid
import random
import base64
import traceback
from typing import Dict, Any, Optional, List
from curl_cffi.requests import AsyncSession
from ..core.logger import debug_logger
from ..core.config import config


class FlowClient:
    """VideoFX API客户端"""

    def __init__(self, proxy_manager):
        self.proxy_manager = proxy_manager
        # Normalize base URL to reduce config foot-guns (some deployments accidentally include '/trpc')
        labs_base_url = (config.flow_labs_base_url or "").rstrip("/")
        if labs_base_url.endswith("/trpc"):
            labs_base_url = labs_base_url[: -len("/trpc")]
        self.labs_base_url = labs_base_url  # https://labs.google/fx/api
        self.api_base_url = config.flow_api_base_url    # https://aisandbox-pa.googleapis.com/v1
        self.timeout = config.flow_timeout

    @staticmethod
    def normalize_session_token(st_token: str) -> str:
        """Normalize user-provided ST input into the next-auth session token value.

        Supported inputs:
        - Raw token value (recommended)
        - Full cookie string containing '__Secure-next-auth.session-token=...'
        - A single cookie pair '__Secure-next-auth.session-token=...'
        - Optional 'Cookie:' prefix
        """
        if not st_token:
            raise ValueError("缺少 Session Token (ST)")

        raw = st_token.strip()
        if raw.lower().startswith("cookie:"):
            raw = raw.split(":", 1)[1].strip()

        if ";" not in raw and "=" not in raw:
            return raw

        cookies: Dict[str, str] = {}
        for part in raw.split(";"):
            part = part.strip()
            if not part or "=" not in part:
                continue
            key, value = part.split("=", 1)
            cookies[key.strip()] = value.strip().strip('"')

        for key in ("__Secure-next-auth.session-token", "next-auth.session-token"):
            if key in cookies and cookies[key]:
                return cookies[key]

        if "__Secure-next-auth.session-token" not in raw and "next-auth.session-token" not in raw:
            raise ValueError(
                "ST 格式不正确：请从 labs.google 获取 Cookie `__Secure-next-auth.session-token` 的值（不要粘贴整段 Google 账号 Cookie）"
            )

        raise ValueError("ST 格式不正确：未解析到有效的 `__Secure-next-auth.session-token`")

    async def _make_request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict] = None,
        json_data: Optional[Dict] = None,
        use_st: bool = False,
        st_token: Optional[str] = None,
        use_at: bool = False,
        at_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """统一HTTP请求处理

        Args:
            method: HTTP方法 (GET/POST)
            url: 完整URL
            headers: 请求头
            json_data: JSON请求体
            use_st: 是否使用ST认证 (Cookie方式)
            st_token: Session Token
            use_at: 是否使用AT认证 (Bearer方式)
            at_token: Access Token
        """
        proxy_url = await self.proxy_manager.get_proxy_url()

        if headers is None:
            headers = {}

        # ST认证 - 使用Cookie
        if use_st and st_token:
            normalized_st = self.normalize_session_token(st_token)
            headers["Cookie"] = f"__Secure-next-auth.session-token={normalized_st}"

        # AT认证 - 使用Bearer
        if use_at and at_token:
            headers["authorization"] = f"Bearer {at_token}"

        # 通用请求头
        headers.update({
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })

        # Log request
        if config.debug_enabled:
            debug_logger.log_request(
                method=method,
                url=url,
                headers=headers,
                body=json_data,
                proxy=proxy_url
            )

        start_time = time.time()

        try:
            async with AsyncSession() as session:
                if method.upper() == "GET":
                    response = await session.get(
                        url,
                        headers=headers,
                        proxy=proxy_url,
                        timeout=self.timeout,
                        impersonate="chrome110",
                        verify=False
                    )
                else:
                    response = await session.post(
                        url,
                        headers=headers,
                        json=json_data,
                        proxy=proxy_url,
                        timeout=self.timeout,
                        impersonate="chrome110",
                        verify=False
                    )

            duration = time.time() - start_time

            # Log response
            if config.debug_enabled:
                debug_logger.log_response(
                    url=url,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    body=response.text
                )

            # Check status
            if response.status_code >= 400:
                debug_logger.log_error(
                    error_message=f"[gemini][flow_client] HTTP {response.status_code} {method} {url}",
                    status_code=response.status_code,
                    response_text=response.text,
                )
                print(f"[gemini][flow_client] HTTP {response.status_code} method={method} url={url} proxy={proxy_url}")
                raise Exception(f"HTTP {response.status_code}: {response.text}")

            return response.json()

        except Exception as e:
            duration = time.time() - start_time
            debug_logger.log_error(
                error_message=f"[gemini][flow_client] {method} {url} proxy={proxy_url} err={str(e)}",
                status_code=0,
                response_text=traceback.format_exc(),
            )
            print(f"[gemini][flow_client] request failed method={method} url={url} proxy={proxy_url} err={e}")
            raise

    # ========== ST->AT 转换 ==========

    async def st_to_at(self, st: str) -> Dict[str, Any]:
        """ST转AT,并获取用户信息"""
        normalized_st = self.normalize_session_token(st)

        def _unwrap_trpc(result: Dict[str, Any]) -> Dict[str, Any]:
            inner = result.get("result", {}).get("data", {}).get("json")
            return inner if isinstance(inner, dict) else result

        # labs.google 的 session 接口在不同版本可能是：
        # - REST: GET /auth/session -> {access_token, expires, user}
        # - tRPC query: GET /trpc/auth.session -> {result:{data:{json:{accessToken,...}}}}
        #
        # 注意：用 POST 可能触发 "No \"mutation\"-procedure on path \"auth.session\""。
        attempts = [
            ("GET", f"{self.labs_base_url}/auth/session"),
            ("GET", f"{self.labs_base_url}/trpc/auth.session"),
        ]

        last_error: Exception = None  # type: ignore[assignment]
        for method, url in attempts:
            try:
                raw = await self._make_request(
                    method=method,
                    url=url,
                    use_st=True,
                    st_token=normalized_st,
                )
                data = _unwrap_trpc(raw)

                access_token = data.get("access_token") or data.get("accessToken")
                expires = data.get("expires")
                user = data.get("user", {}) or {}

                if access_token:
                    return {"access_token": access_token, "expires": expires, "user": user}

                raise Exception("ST转AT失败: 未获取到access_token/accessToken")
            except Exception as e:
                last_error = e

        raise last_error

    # ========== Credits查询 ==========

    async def get_credits(self, at: str) -> Dict[str, Any]:
        """查询VideoFX credits余额"""
        # Prefer the stable REST endpoint (same as Flow2API):
        # GET https://aisandbox-pa.googleapis.com/v1/credits
        try:
            url = f"{self.api_base_url}/credits"
            return await self._make_request(
                method="GET",
                url=url,
                use_at=True,
                at_token=at,
            )
        except Exception:
            # Backward compatibility for older tRPC endpoint
            url = f"{self.labs_base_url}/trpc/videoFx.credits"
            result = await self._make_request(
                method="POST",
                url=url,
                json_data={"json": None},
                use_at=True,
                at_token=at,
            )
            data = result.get("result", {}).get("data", {}).get("json", {})
            return {"credits": data.get("credits", 0), "userPaygateTier": data.get("userPaygateTier")}

    # ========== Project管理 ==========

    async def create_project(self, st: str, project_name: str) -> str:
        """创建新项目,返回project_id"""
        normalized_st = self.normalize_session_token(st)

        # Preferred endpoint (same as Flow2API):
        # POST /trpc/project.createProject with {projectTitle, toolName}
        attempts = [
            (
                f"{self.labs_base_url}/trpc/project.createProject",
                {"json": {"projectTitle": project_name, "toolName": "PINHOLE"}},
                "flow2api",
            ),
            (
                f"{self.labs_base_url}/trpc/projects.createProject",
                {"json": {"projectName": project_name, "toolName": "PINHOLE"}},
                "legacy",
            ),
        ]

        last_error: Exception = None  # type: ignore[assignment]
        for url, json_data, mode in attempts:
            try:
                result = await self._make_request(
                    method="POST",
                    url=url,
                    json_data=json_data,
                    use_st=True,
                    st_token=normalized_st,
                )

                if mode == "flow2api":
                    project_id = (
                        result.get("result", {})
                        .get("data", {})
                        .get("json", {})
                        .get("result", {})
                        .get("projectId")
                    )
                else:
                    data = result.get("result", {}).get("data", {}).get("json", {})
                    project_id = data.get("projectId")

                if not project_id:
                    raise Exception("创建项目失败: 未获取到projectId")
                return project_id
            except Exception as e:
                last_error = e

        raise last_error

    # ========== 图片上传 ==========

    async def upload_image(self, at: str, image_bytes: bytes, aspect_ratio: str) -> str:
        """上传图片,返回media_id"""
        url = f"{self.api_base_url}/media:upload"

        # base64编码
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        json_data = {
            "mimeType": "image/png",
            "data": image_base64,
            "aspectRatio": aspect_ratio
        }

        result = await self._make_request(
            method="POST",
            url=url,
            json_data=json_data,
            use_at=True,
            at_token=at
        )

        # 返回name作为mediaId
        media_id = result.get("name")
        if not media_id:
            raise Exception("图片上传失败: 未获取到mediaId")

        return media_id

    # ========== 图片生成 (使用AT) - 同步返回 ==========

    async def generate_image(
        self,
        at: str,
        project_id: str,
        prompt: str,
        model_name: str,
        aspect_ratio: str,
        image_inputs: Optional[List[Dict]] = None
    ) -> dict:
        """生成图片,同步返回结果

        Args:
            at: Access Token
            project_id: 项目ID
            prompt: 提示词
            model_name: GEM_PIX / GEM_PIX_2 / IMAGEN_3_5
            aspect_ratio: IMAGE_ASPECT_RATIO_LANDSCAPE / PORTRAIT
            image_inputs: 图片输入列表 (支持多图)

        Returns:
            {
                "media": [{
                    "image": {
                        "generatedImage": {
                            "fifeUrl": "..."
                        }
                    }
                }]
            }
        """
        url = f"{self.api_base_url}/image:batchGenerateImage"

        # 获取 reCAPTCHA token
        recaptcha_token = await self._get_recaptcha_token(project_id) or ""
        session_id = self._generate_session_id()

        request_data = {
            "clientContext": {
                "projectId": project_id,
                "tool": "PINHOLE"
            },
            "seed": random.randint(1, 99999),
            "imageModelName": model_name,
            "imageAspectRatio": aspect_ratio,
            "prompt": prompt,
            "imageInputs": image_inputs or []
        }

        json_data = {
            "clientContext": {
                "recaptchaToken": recaptcha_token,
                "sessionId": session_id
            },
            "requests": [request_data]
        }

        result = await self._make_request(
            method="POST",
            url=url,
            json_data=json_data,
            use_at=True,
            at_token=at
        )

        return result

    # ========== 视频生成 (使用AT) - 异步返回 ==========

    async def generate_video_text(
        self,
        at: str,
        project_id: str,
        prompt: str,
        model_key: str,
        aspect_ratio: str,
        user_paygate_tier: str = "PAYGATE_TIER_ONE"
    ) -> dict:
        """文生视频,返回task_id"""
        url = f"{self.api_base_url}/video:batchAsyncGenerateVideoText"

        recaptcha_token = await self._get_recaptcha_token(project_id) or ""
        session_id = self._generate_session_id()
        scene_id = str(uuid.uuid4())

        json_data = {
            "clientContext": {
                "recaptchaToken": recaptcha_token,
                "sessionId": session_id,
                "projectId": project_id,
                "tool": "PINHOLE",
                "userPaygateTier": user_paygate_tier
            },
            "requests": [{
                "aspectRatio": aspect_ratio,
                "seed": random.randint(1, 99999),
                "textInput": {
                    "prompt": prompt
                },
                "videoModelKey": model_key,
                "metadata": {
                    "sceneId": scene_id
                }
            }]
        }

        result = await self._make_request(
            method="POST",
            url=url,
            json_data=json_data,
            use_at=True,
            at_token=at
        )

        return result

    async def generate_video_reference_images(
        self,
        at: str,
        project_id: str,
        prompt: str,
        model_key: str,
        aspect_ratio: str,
        reference_images: List[Dict],
        user_paygate_tier: str = "PAYGATE_TIER_ONE"
    ) -> dict:
        """图生视频,返回task_id"""
        url = f"{self.api_base_url}/video:batchAsyncGenerateVideoReferenceImages"

        recaptcha_token = await self._get_recaptcha_token(project_id) or ""
        session_id = self._generate_session_id()
        scene_id = str(uuid.uuid4())

        json_data = {
            "clientContext": {
                "recaptchaToken": recaptcha_token,
                "sessionId": session_id,
                "projectId": project_id,
                "tool": "PINHOLE",
                "userPaygateTier": user_paygate_tier
            },
            "requests": [{
                "aspectRatio": aspect_ratio,
                "seed": random.randint(1, 99999),
                "textInput": {
                    "prompt": prompt
                },
                "videoModelKey": model_key,
                "referenceImages": reference_images,
                "metadata": {
                    "sceneId": scene_id
                }
            }]
        }

        result = await self._make_request(
            method="POST",
            url=url,
            json_data=json_data,
            use_at=True,
            at_token=at
        )

        return result

    async def generate_video_start_end(
        self,
        at: str,
        project_id: str,
        prompt: str,
        model_key: str,
        aspect_ratio: str,
        start_media_id: str,
        end_media_id: str,
        user_paygate_tier: str = "PAYGATE_TIER_ONE"
    ) -> dict:
        """收尾帧生成视频,返回task_id"""
        url = f"{self.api_base_url}/video:batchAsyncGenerateVideoStartAndEndImage"

        recaptcha_token = await self._get_recaptcha_token(project_id) or ""
        session_id = self._generate_session_id()
        scene_id = str(uuid.uuid4())

        json_data = {
            "clientContext": {
                "recaptchaToken": recaptcha_token,
                "sessionId": session_id,
                "projectId": project_id,
                "tool": "PINHOLE",
                "userPaygateTier": user_paygate_tier
            },
            "requests": [{
                "aspectRatio": aspect_ratio,
                "seed": random.randint(1, 99999),
                "textInput": {
                    "prompt": prompt
                },
                "videoModelKey": model_key,
                "startImage": {
                    "mediaId": start_media_id
                },
                "endImage": {
                    "mediaId": end_media_id
                },
                "metadata": {
                    "sceneId": scene_id
                }
            }]
        }

        result = await self._make_request(
            method="POST",
            url=url,
            json_data=json_data,
            use_at=True,
            at_token=at
        )

        return result

    async def generate_video_start_image(
        self,
        at: str,
        project_id: str,
        prompt: str,
        model_key: str,
        aspect_ratio: str,
        start_media_id: str,
        user_paygate_tier: str = "PAYGATE_TIER_ONE"
    ) -> dict:
        """仅首帧生成视频,返回task_id"""
        url = f"{self.api_base_url}/video:batchAsyncGenerateVideoStartAndEndImage"

        recaptcha_token = await self._get_recaptcha_token(project_id) or ""
        session_id = self._generate_session_id()
        scene_id = str(uuid.uuid4())

        json_data = {
            "clientContext": {
                "recaptchaToken": recaptcha_token,
                "sessionId": session_id,
                "projectId": project_id,
                "tool": "PINHOLE",
                "userPaygateTier": user_paygate_tier
            },
            "requests": [{
                "aspectRatio": aspect_ratio,
                "seed": random.randint(1, 99999),
                "textInput": {
                    "prompt": prompt
                },
                "videoModelKey": model_key,
                "startImage": {
                    "mediaId": start_media_id
                },
                "metadata": {
                    "sceneId": scene_id
                }
            }]
        }

        result = await self._make_request(
            method="POST",
            url=url,
            json_data=json_data,
            use_at=True,
            at_token=at
        )

        return result

    # ========== 任务轮询 (使用AT) ==========

    async def check_video_status(self, at: str, operations: List[Dict]) -> dict:
        """查询视频生成状态"""
        url = f"{self.api_base_url}/video:batchCheckAsyncVideoGenerationStatus"

        json_data = {
            "operations": operations
        }

        result = await self._make_request(
            method="POST",
            url=url,
            json_data=json_data,
            use_at=True,
            at_token=at
        )

        return result

    # ========== 媒体删除 (使用ST) ==========

    async def delete_media(self, st: str, media_names: List[str]):
        """删除媒体"""
        url = f"{self.labs_base_url}/trpc/media.deleteMedia"
        json_data = {
            "json": {
                "names": media_names
            }
        }

        await self._make_request(
            method="POST",
            url=url,
            json_data=json_data,
            use_st=True,
            st_token=st
        )

    # ========== 辅助方法 ==========

    def _generate_session_id(self) -> str:
        """生成sessionId: ;timestamp"""
        return f";{int(time.time() * 1000)}"

    def _generate_scene_id(self) -> str:
        """生成sceneId: UUID"""
        return str(uuid.uuid4())

    async def _get_recaptcha_token(self, project_id: str) -> Optional[str]:
        """获取reCAPTCHA token - 支持两种方式"""
        captcha_method = config.captcha_method

        # 恒定浏览器打码
        if captcha_method == "personal":
            try:
                from .browser_captcha_personal import BrowserCaptchaService
                service = await BrowserCaptchaService.get_instance(self.proxy_manager)
                return await service.get_token(project_id)
            except Exception as e:
                debug_logger.log_error(f"[reCAPTCHA Browser] error: {str(e)}")
                return None
        # 无头浏览器打码
        elif captcha_method == "browser":
            try:
                from .browser_captcha import BrowserCaptchaService
                service = await BrowserCaptchaService.get_instance(self.proxy_manager)
                return await service.get_token(project_id)
            except Exception as e:
                debug_logger.log_error(f"[reCAPTCHA Browser] error: {str(e)}")
                return None
        else:
            # YesCaptcha打码
            client_key = config.yescaptcha_api_key
            if not client_key:
                debug_logger.log_info("[reCAPTCHA] API key not configured, skipping")
                return None

            website_key = "6LdsFiUsAAAAAIjVDZcuLhaHiDn5nnHVXVRQGeMV"
            website_url = f"https://labs.google/fx/tools/flow/project/{project_id}"
            base_url = config.yescaptcha_base_url
            page_action = "FLOW_GENERATION"

            try:
                async with AsyncSession() as session:
                    create_url = f"{base_url}/createTask"
                    create_data = {
                        "clientKey": client_key,
                        "task": {
                            "websiteURL": website_url,
                            "websiteKey": website_key,
                            "type": "RecaptchaV3TaskProxylessM1",
                            "pageAction": page_action
                        }
                    }

                    result = await session.post(create_url, json=create_data, impersonate="chrome110")
                    result_json = result.json()
                    task_id = result_json.get('taskId')

                    debug_logger.log_info(f"[reCAPTCHA] created task_id: {task_id}")

                    if not task_id:
                        return None

                    get_url = f"{base_url}/getTaskResult"
                    for i in range(40):
                        get_data = {
                            "clientKey": client_key,
                            "taskId": task_id
                        }
                        result = await session.post(get_url, json=get_data, impersonate="chrome110")
                        result_json = result.json()

                        debug_logger.log_info(f"[reCAPTCHA] polling #{i+1}: {result_json}")

                        solution = result_json.get('solution', {})
                        response = solution.get('gRecaptchaResponse')

                        if response:
                            return response

                        time.sleep(3)

                    return None

            except Exception as e:
                debug_logger.log_error(f"[reCAPTCHA] error: {str(e)}")
                return None
