"""OpenAI格式API客户端

支持：OpenAI官方、硅基流动、NewAPI、火山方舟等兼容OpenAI格式的服务
"""
import json
import urllib.request
import traceback
from typing import Dict, Any, Tuple

from .base_client import BaseApiClient, logger


class OpenAIClient(BaseApiClient):
    """OpenAI格式API客户端"""

    format_name = "openai"

    def _make_request(
        self,
        prompt: str,
        model_config: Dict[str, Any],
        size: str,
        strength: float = None,
        input_image_base64: str = None
    ) -> Tuple[bool, str]:
        """发送OpenAI格式的HTTP请求生成图片"""
        base_url = model_config.get("base_url", "")
        generate_api_key = model_config.get("api_key", "")
        model = model_config.get("model", "")

        # 直接拼接路径
        endpoint = f"{base_url.rstrip('/')}/images/generations"

        # 获取模型特定的配置参数
        custom_prompt_add = model_config.get("custom_prompt_add", "")
        negative_prompt_add = model_config.get("negative_prompt_add", "")
        seed = model_config.get("seed", -1)
        guidance_scale = model_config.get("guidance_scale", 7.5)
        watermark = model_config.get("watermark", True)
        num_inference_steps = model_config.get("num_inference_steps", 20)
        prompt_add = prompt + custom_prompt_add
        negative_prompt = negative_prompt_add

        # 构建基本请求参数
        payload_dict = {
            "model": model,
            "prompt": prompt_add,
            "size": size,
            "n": 1,
        }

        # 添加可选参数
        if negative_prompt:
            payload_dict["negative_prompt"] = negative_prompt
        if seed and seed != -1:
            payload_dict["seed"] = seed

        # 如果有输入图片，添加图生图参数
        if input_image_base64:
            image_data_uri = self._prepare_image_data_uri(input_image_base64)
            payload_dict["image"] = image_data_uri
            if strength is not None:
                payload_dict["strength"] = strength

        # 根据不同API添加特定参数
        if "ark.cn-beijing.volces.com" in base_url:  # 豆包火山方舟
            payload_dict["watermark"] = watermark
        else:  # 默认魔搭等其他
            payload_dict["guidance_scale"] = guidance_scale
            payload_dict["num_inference_steps"] = num_inference_steps

        # 平台兼容性处理
        is_siliconflow = "siliconflow" in base_url.lower() or "api.siliconflow.cn" in base_url.lower()
        is_openai_official = "api.openai.com" in base_url.lower()
        is_grok = "api.x.ai" in base_url.lower()

        if is_siliconflow:
            # 硅基流动：使用 image_size 代替 size，batch_size 代替 n
            if "size" in payload_dict:
                payload_dict["image_size"] = payload_dict.pop("size")
            if "n" in payload_dict:
                payload_dict["batch_size"] = payload_dict.pop("n")

            # 根据模型选择正确的参数
            model_lower = model.lower()
            if "qwen" in model_lower:
                # Qwen-Image 系列使用 cfg 而非 guidance_scale
                if "guidance_scale" in payload_dict:
                    payload_dict["cfg"] = payload_dict.pop("guidance_scale")
                # Qwen-Image-Edit 不支持 image_size
                if "image-edit" in model_lower and "image_size" in payload_dict:
                    del payload_dict["image_size"]
            else:
                # Kolors 等其他模型使用 guidance_scale
                pass

            logger.debug(f"{self.log_prefix} (OpenAI) 检测到硅基流动平台，使用 image_size/batch_size 参数")

        elif is_openai_official:
            # OpenAI官方：只保留标准参数
            standard_params = ["model", "prompt", "size", "n", "quality", "style", "response_format"]
            if input_image_base64:
                standard_params.extend(["image", "strength"])
            payload_dict = {k: v for k, v in payload_dict.items() if k in standard_params}
            logger.debug(f"{self.log_prefix} (OpenAI) 检测到OpenAI官方平台，仅使用标准参数")

        elif is_grok:
            # Grok：只保留 model, prompt, n, response_format
            supported = ["model", "prompt", "n", "response_format"]
            payload_dict = {k: v for k, v in payload_dict.items() if k in supported}
            logger.debug(f"{self.log_prefix} (OpenAI) 检测到Grok平台，仅保留支持的参数")

        data = json.dumps(payload_dict).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"{generate_api_key}",
        }

        # 详细调试信息
        verbose_debug = self.action.get_config("components.enable_verbose_debug", False)
        if verbose_debug:
            # 记录完整的请求payload（隐藏敏感信息）
            safe_payload = payload_dict.copy()
            # 不记录图片base64数据，因为太长
            if "image" in safe_payload:
                safe_payload["image"] = "[BASE64_DATA...]"
            # 创建安全的请求头副本，隐藏Authorization值
            safe_headers = headers.copy()
            if "Authorization" in safe_headers:
                auth_value = safe_headers["Authorization"]
                # 如果包含Bearer，保留Bearer前缀，隐藏其余部分
                if auth_value.startswith("Bearer "):
                    safe_headers["Authorization"] = "Bearer ***"
                else:
                    safe_headers["Authorization"] = "***"
            logger.info(f"{self.log_prefix} (OpenAI) 详细调试 - 请求端点: {endpoint}")
            logger.info(f"{self.log_prefix} (OpenAI) 详细调试 - 请求头: {safe_headers}")
            logger.info(f"{self.log_prefix} (OpenAI) 详细调试 - 请求体: {json.dumps(safe_payload, ensure_ascii=False, indent=2)}")

        logger.info(f"{self.log_prefix} (OpenAI) 发起图片请求: {model}, Prompt: {prompt_add[:30]}... To: {endpoint}")

        # 获取代理配置
        proxy_config = self._get_proxy_config()

        req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")

        try:
            # 如果启用了代理，设置代理处理器
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
                response_body_bytes = response.read()
                response_body_str = response_body_bytes.decode("utf-8")
                logger.info(f"{self.log_prefix} (OpenAI) 响应: {response_status}. Preview: {response_body_str[:150]}...")

                # 详细调试信息
                if verbose_debug:
                    logger.info(f"{self.log_prefix} (OpenAI) 详细调试 - 完整响应体: {response_body_str}")

                if 200 <= response_status < 300:
                    response_data = json.loads(response_body_str)
                    b64_data = None
                    image_url = None

                    # 优先检查Base64数据
                    if (
                        isinstance(response_data.get("data"), list)
                        and response_data["data"]
                        and isinstance(response_data["data"][0], dict)
                        and "b64_json" in response_data["data"][0]
                    ):
                        b64_data = response_data["data"][0]["b64_json"]
                        logger.info(f"{self.log_prefix} (OpenAI) 获取到Base64图片数据，长度: {len(b64_data)}")
                        return True, b64_data
                    elif (
                        isinstance(response_data.get("data"), list)
                        and response_data["data"]
                        and isinstance(response_data["data"][0], dict)
                    ):
                        image_url = response_data["data"][0].get("url")
                    elif (  # 魔搭社区返回的 json
                        isinstance(response_data.get("images"), list)
                        and response_data["images"]
                        and isinstance(response_data["images"][0], dict)
                    ):
                        image_url = response_data["images"][0].get("url")
                    elif response_data.get("url"):
                        image_url = response_data.get("url")

                    if image_url:
                        logger.info(f"{self.log_prefix} (OpenAI) 图片生成成功，URL: {image_url[:70]}...")
                        return True, image_url
                    else:
                        logger.error(f"{self.log_prefix} (OpenAI) API成功但无图片URL. 响应预览: {response_body_str[:300]}...")
                        return False, "图片生成API响应成功但未找到图片URL"
                else:
                    logger.error(f"{self.log_prefix} (OpenAI) API请求失败. 状态: {response.status}. 正文: {response_body_str[:300]}...")
                    return False, f"图片API请求失败(状态码 {response.status})"
        except Exception as e:
            logger.error(f"{self.log_prefix} (OpenAI) 图片生成时意外错误: {e!r}", exc_info=True)
            traceback.print_exc()
            return False, f"图片生成HTTP请求时发生意外错误: {str(e)[:100]}"
