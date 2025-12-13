"""Zai API 客户端（OpenAI 兼容，走 chat/completions）
支持文生图 / 图生图，透传 Image Aspect Ratio、Image Resolution、seed。
"""
import json
import re
import urllib.request
import traceback
from typing import Dict, Any, Tuple, Optional

from .base_client import BaseApiClient, logger
from ..size_utils import pixel_size_to_gemini_aspect


class ZaiClient(BaseApiClient):
    """Zai 平台（Gemini 转发）的 OpenAI 兼容客户端"""

    format_name = "zai"

    def _make_request(
        self,
        prompt: str,
        model_config: Dict[str, Any],
        size: str,
        strength: float = None,
        input_image_base64: str = None
    ) -> Tuple[bool, str]:
        """发送 Zai chat/completions 请求"""
        base_url = model_config.get("base_url", "https://zai.is/api").rstrip('/')
        api_key = model_config.get("api_key", "")
        model = model_config.get("model", "")

        endpoint = f"{base_url}/chat/completions"

        # 组装 prompt
        custom_prompt_add = model_config.get("custom_prompt_add", "")
        full_prompt = prompt + custom_prompt_add

        # 构造 messages
        contents = [{"type": "text", "text": full_prompt}]
        if input_image_base64:
            image_data_uri = self._prepare_image_data_uri(input_image_base64)
            contents.append({
                "type": "image_url",
                "image_url": {"url": image_data_uri}
            })

        messages = [{
            "role": "user",
            "content": contents
        }]

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "n": 1,
        }

        # 处理图像配置（宽高比 / 分辨率）
        image_config = self._build_image_config(model_config)
        if image_config.get("image_aspect_ratio"):
            payload["image_aspect_ratio"] = image_config["image_aspect_ratio"]
        if image_config.get("image_resolution"):
            payload["image_resolution"] = image_config["image_resolution"]

        # 种子可选
        seed = model_config.get("seed")
        if seed is not None and seed != -1:
            payload["seed"] = seed

        # 代理配置
        proxy_config = self._get_proxy_config()

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"{api_key}"
        }

        logger.info(f"{self.log_prefix} (Zai) 发起请求: {model}, Prompt: {full_prompt[:50]}... To: {endpoint}")

        req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")

        try:
            if proxy_config:
                proxy_handler = urllib.request.ProxyHandler({
                    'http': proxy_config['http'],
                    'https': proxy_config['https']
                })
                opener = urllib.request.build_opener(proxy_handler)
                urllib.request.install_opener(opener)
                timeout = proxy_config.get('timeout', 600)
            else:
                timeout = 600

            with urllib.request.urlopen(req, timeout=timeout) as response:
                response_status = response.status
                body_bytes = response.read()
                body_str = body_bytes.decode("utf-8")
                preview = body_str[:200]
                logger.info(f"{self.log_prefix} (Zai) 响应: {response_status}. Preview: {preview}...")

                if 200 <= response_status < 300:
                    try:
                        resp_json = json.loads(body_str)
                    except json.JSONDecodeError:
                        logger.error(f"{self.log_prefix} (Zai) 响应 JSON 解析失败")
                        return False, "响应解析失败"

                    # 兼容 OpenAI images/generations 风格
                    if isinstance(resp_json.get("data"), list) and resp_json["data"]:
                        first = resp_json["data"][0]
                        if isinstance(first, dict):
                            if "b64_json" in first:
                                return True, first["b64_json"]
                            if "url" in first:
                                return True, first["url"]

                    # 兼容 chat/completions 风格
                    choices = resp_json.get("choices")
                    if isinstance(choices, list) and choices:
                        choice = choices[0]
                        message = choice.get("message", {})
                        content = message.get("content")
                        extracted = self._extract_image_from_content(content)
                        if extracted:
                            # 直接返回提取到的URL/base64，由下游处理
                            return True, extracted

                    logger.error(f"{self.log_prefix} (Zai) 响应中未找到图像数据")
                    return False, "未找到图像数据"
                else:
                    logger.error(f"{self.log_prefix} (Zai) API 请求失败. 状态 {response_status}. 正文: {body_str[:300]}...")
                    return False, f"API 请求失败(状态码 {response_status})"

        except Exception as e:
            logger.error(f"{self.log_prefix} (Zai) 请求异常: {e!r}", exc_info=True)
            traceback.print_exc()
            return False, f"HTTP 请求异常: {str(e)[:100]}"

    def _build_image_config(self, model_config: Dict[str, Any]) -> Dict[str, Optional[str]]:
        """将 size/default_size 转换为 image_aspect_ratio / image_resolution"""
        fixed_size_enabled = model_config.get("fixed_size_enabled", False)
        default_size = model_config.get("default_size", "").strip()
        llm_original_size = model_config.get("_llm_original_size", "").strip() or None

        aspect_ratio = None
        resolution = None

        if not fixed_size_enabled:
            if llm_original_size:
                aspect_ratio = pixel_size_to_gemini_aspect(llm_original_size, self.log_prefix) or "1:1"
            else:
                aspect_ratio = "1:1"
        else:
            if default_size.startswith("-"):
                resolution = default_size[1:].strip().upper()
                if llm_original_size:
                    aspect_ratio = pixel_size_to_gemini_aspect(llm_original_size, self.log_prefix) or "1:1"
                else:
                    aspect_ratio = "1:1"
            elif "-" in default_size and ":" in default_size:
                parts = default_size.split("-", 1)
                aspect_ratio = parts[0].strip()
                resolution = parts[1].strip().upper()
            elif ":" in default_size:
                aspect_ratio = default_size
            elif "x" in default_size.lower():
                aspect_ratio = pixel_size_to_gemini_aspect(default_size, self.log_prefix) or "1:1"
            else:
                aspect_ratio = "1:1"

        result = {}
        if aspect_ratio:
            result["image_aspect_ratio"] = aspect_ratio
        if resolution:
            result["image_resolution"] = resolution
        return result

    def _extract_image_from_content(self, content: Any) -> Optional[str]:
        """从 chat/completions 的 content 字段提取图片 URL 或 base64"""
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "image_url":
                    url = item.get("image_url", {}).get("url")
                    if url:
                        return url
                if item.get("type") == "text":
                    text = item.get("text", "") or ""
                    candidate = self._extract_from_text(text)
                    if candidate:
                        return candidate
        elif isinstance(content, str):
            return self._extract_from_text(content)
        return None


    def _extract_from_text(self, text: str) -> Optional[str]:
        """尝试从文本中提取 base64 或 URL"""
        if not text:
            return None

        stripped = text.strip()
        if self._looks_like_base64(stripped):
            return stripped

        url_match = re.search(r"https?://\S+", stripped)
        if url_match:
            return url_match.group(0).rstrip('",\'')

        return None

    def _looks_like_base64(self, data: str) -> bool:
        """粗略判断字符串是否像图片 base64"""
        if not data:
            return False

        if data.startswith("data:image"):
            return True

        prefixes = ("/9j/", "iVBORw", "UklGR", "R0lGOD")
        return any(data.startswith(p) for p in prefixes)
