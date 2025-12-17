"""Gemini API客户端"""
import json
import requests
from typing import Dict, Any, Tuple, Optional

from .base_client import BaseApiClient, logger
from ..size_utils import pixel_size_to_gemini_aspect


class GeminiClient(BaseApiClient):
    """Google Gemini API客户端"""

    format_name = "gemini"

    def _make_request(
        self,
        prompt: str,
        model_config: Dict[str, Any],
        size: str,
        strength: float = None,
        input_image_base64: str = None
    ) -> Tuple[bool, str]:
        """发送Gemini格式的HTTP请求生成图片"""
        try:
            # API配置
            api_key = model_config.get("api_key", "").replace("Bearer ", "")
            model_name = model_config.get("model", "gemini-2.5-flash-image-preview")
            base_url = model_config.get("base_url", "https://generativelanguage.googleapis.com").rstrip('/')

            # 构建API端点
            url = f"{base_url}/v1beta/models/{model_name}:generateContent"

            # 请求头
            headers = {
                "x-goog-api-key": api_key,
                "Content-Type": "application/json"
            }

            # 获取模型特定的配置参数
            custom_prompt_add = model_config.get("custom_prompt_add", "")
            full_prompt = prompt + custom_prompt_add

            # 构建请求内容
            parts = [{"text": full_prompt}]

            # 如果有输入图片，添加到请求中
            if input_image_base64:
                logger.info(f"{self.log_prefix} (Gemini) 使用图生图模式")

                try:
                    clean_base64 = self._get_clean_base64(input_image_base64)
                    mime_type = self._detect_mime_type(input_image_base64)

                    parts.append({
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": clean_base64
                        }
                    })

                except Exception as e:
                    logger.error(f"{self.log_prefix} (Gemini) 图片处理失败: {e}")
                    return False, f"图片处理失败: {str(e)}"
            else:
                logger.info(f"{self.log_prefix} (Gemini) 使用文生图模式")

            # 构建请求体
            request_data = {
                "contents": [{
                    "role": "user",
                    "parts": parts
                }],
                "safetySettings": model_config.get("safety_settings") or [],
                "generationConfig": {
                    "responseModalities": ["TEXT", "IMAGE"]
                }
            }

            # 添加 Gemini 图片尺寸配置
            image_config = self._build_gemini_image_config(model_name, model_config, size)
            if image_config:
                request_data["generationConfig"]["imageConfig"] = image_config
                logger.info(f"{self.log_prefix} (Gemini) 图片配置: {image_config}")

            logger.info(f"{self.log_prefix} (Gemini) 发起图片请求: {model_name}")

            # 获取代理配置
            proxy_config = self._get_proxy_config()

            # 构建请求参数
            request_kwargs = {
                "url": url,
                "headers": headers,
                "json": request_data,
                "timeout": proxy_config.get('timeout', 120) if proxy_config else 120
            }

            if proxy_config:
                request_kwargs["proxies"] = {
                    "http": proxy_config["http"],
                    "https": proxy_config["https"]
                }

            # 发送请求
            response = requests.post(**request_kwargs)

            if response.status_code != 200:
                error_msg = response.text
                logger.error(f"{self.log_prefix} (Gemini) API请求失败: HTTP {response.status_code} - {error_msg}")
                return False, f"API请求失败: {error_msg[:100]}"

            # 解析响应
            try:
                response_json = response.json()

                if "candidates" in response_json and response_json["candidates"]:
                    candidate = response_json["candidates"][0]

                    if "content" in candidate and "parts" in candidate["content"]:
                        for part in candidate["content"]["parts"]:
                            if "inlineData" in part and "data" in part["inlineData"]:
                                image_base64 = part["inlineData"]["data"]
                                logger.info(f"{self.log_prefix} (Gemini) 图片生成成功")
                                return True, image_base64
                            elif "inline_data" in part and "data" in part["inline_data"]:
                                image_base64 = part["inline_data"]["data"]
                                logger.info(f"{self.log_prefix} (Gemini) 图片生成成功")
                                return True, image_base64

                if "error" in response_json:
                    error_info = response_json["error"]
                    error_message = error_info.get("message", "未知错误")
                    logger.error(f"{self.log_prefix} (Gemini) API返回错误: {error_message}")
                    return False, f"API错误: {error_message}"

                logger.warning(f"{self.log_prefix} (Gemini) 未找到图片数据")
                return False, "未收到图片数据，可能模型不支持图片生成或请求格式不正确"

            except json.JSONDecodeError as e:
                logger.error(f"{self.log_prefix} (Gemini) JSON解析失败: {e}")
                return False, f"响应解析失败: {str(e)}"

        except requests.RequestException as e:
            logger.error(f"{self.log_prefix} (Gemini) 网络请求异常: {e}")
            return False, f"网络请求失败: {str(e)}"

        except Exception as e:
            logger.error(f"{self.log_prefix} (Gemini) 请求异常: {e!r}", exc_info=True)
            return False, f"请求失败: {str(e)}"

    def _build_gemini_image_config(self, model_name: str, model_config: Dict[str, Any], llm_size: str = None) -> Optional[Dict[str, str]]:
        """构建 Gemini 图片配置"""
        fixed_size_enabled = model_config.get("fixed_size_enabled", False)
        default_size = model_config.get("default_size", "").strip()
        llm_original_size = model_config.get("_llm_original_size", "").strip() or None

        image_config = {}
        final_aspect_ratio = None
        final_image_size = None

        if not fixed_size_enabled:
            # 使用LLM提供的尺寸
            if llm_original_size:
                final_aspect_ratio = pixel_size_to_gemini_aspect(llm_original_size, self.log_prefix)
                if not final_aspect_ratio:
                    final_aspect_ratio = "1:1"
            else:
                final_aspect_ratio = "1:1"
        else:
            # 使用固定尺寸配置
            if not default_size:
                return None

            if default_size.startswith("-"):
                # 仅分辨率格式：-2K
                resolution = default_size[1:].strip().upper()
                if llm_original_size:
                    final_aspect_ratio = pixel_size_to_gemini_aspect(llm_original_size, self.log_prefix)
                    if final_aspect_ratio:
                        final_image_size = resolution
                    else:
                        return None
                else:
                    final_aspect_ratio = "1:1"
                    final_image_size = resolution
            elif "-" in default_size:
                # 宽高比-分辨率格式：16:9-2K
                parts = default_size.split("-", 1)
                final_aspect_ratio = parts[0].strip()
                final_image_size = parts[1].strip().upper()
            elif ":" in default_size:
                # 纯宽高比格式：16:9
                final_aspect_ratio = default_size
            elif "x" in default_size.lower():
                # 像素格式：1024x1024
                final_aspect_ratio = pixel_size_to_gemini_aspect(default_size, self.log_prefix)
                if not final_aspect_ratio:
                    final_aspect_ratio = "1:1"
            else:
                final_aspect_ratio = "1:1"

        if final_aspect_ratio:
            image_config["aspectRatio"] = final_aspect_ratio

        if final_image_size:
            if "gemini-3" in model_name.lower():
                if final_image_size in ["1K", "2K", "4K"]:
                    image_config["imageSize"] = final_image_size

        return image_config if image_config else None
