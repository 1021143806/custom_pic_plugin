import asyncio
import json
import urllib.request
import traceback
import time
import requests
from typing import Dict, Any, Tuple, Optional

from src.common.logger import get_logger

logger = get_logger("pic_action")

class ApiClient:
    """统一的API客户端，处理不同格式的图片生成API"""

    def __init__(self, action_instance):
        self.action = action_instance
        self.log_prefix = action_instance.log_prefix

    def _get_proxy_config(self):
        """获取代理配置"""
        try:
            proxy_enabled = self.action.get_config("proxy.enabled", False)
            if not proxy_enabled:
                return None

            proxy_url = self.action.get_config("proxy.url", "http://127.0.0.1:7890")
            timeout = self.action.get_config("proxy.timeout", 60)

            proxy_config = {
                "http": proxy_url,
                "https": proxy_url,
                "timeout": timeout
            }

            logger.info(f"{self.log_prefix} 代理已启用: {proxy_url}")
            return proxy_config
        except Exception as e:
            logger.warning(f"{self.log_prefix} 获取代理配置失败: {e}, 将不使用代理")
            return None

    async def generate_image(self, prompt: str, model_config: Dict[str, Any], size: str,
                           strength: float = None, input_image_base64: str = None, max_retries: int = 2) -> Tuple[bool, str]:
        """根据API格式调用不同的请求方法，支持重试"""
        api_format = model_config.get("format", "openai")

        # 实现重试逻辑
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"{self.log_prefix} API调用重试第 {attempt} 次")
                    await asyncio.sleep(1.0 * attempt)  # 渐进式等待时间

                logger.debug(f"{self.log_prefix} 开始API调用（尝试 {attempt + 1}/{max_retries + 1}）")

                if api_format == "doubao":
                    success, result = await asyncio.to_thread(
                        self._make_doubao_request,
                        prompt=prompt,
                        model_config=model_config,
                        size=size,
                        input_image_base64=input_image_base64
                    )
                elif api_format == "modelscope":
                    success, result = await asyncio.to_thread(
                        self._make_modelscope_request,
                        prompt=prompt,
                        model_config=model_config,
                        input_image_base64=input_image_base64
                    )
                elif api_format == "gemini":
                    success, result = await asyncio.to_thread(
                        self._make_gemini_request,
                        prompt=prompt,
                        model_config=model_config,
                        input_image_base64=input_image_base64
                    )
                else:  # 默认为openai格式
                    success, result = await asyncio.to_thread(
                        self._make_openai_image_request,
                        prompt=prompt,
                        model_config=model_config,
                        size=size,
                        strength=strength,
                        input_image_base64=input_image_base64
                    )

                # 如果成功，直接返回
                if success:
                    if attempt > 0:
                        logger.info(f"{self.log_prefix} API调用重试第 {attempt} 次成功")
                    return True, result

                # 如果失败但还有重试次数
                if attempt < max_retries:
                    logger.warning(f"{self.log_prefix} 第 {attempt + 1} 次API调用失败: {result}，将重试（剩余 {max_retries - attempt} 次）")
                    continue
                else:
                    logger.error(f"{self.log_prefix} 重试 {max_retries} 次后API调用仍失败: {result}")
                    return False, result

            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"{self.log_prefix} 第 {attempt + 1} 次API调用异常: {e}，将重试（剩余 {max_retries - attempt} 次）")
                    continue
                else:
                    logger.error(f"{self.log_prefix} 重试后API调用仍异常: {e!r}", exc_info=True)
                    return False, f"API调用异常: {str(e)[:100]}"

        return False, "API调用失败"

    def _make_doubao_request(self, prompt: str, model_config: Dict[str, Any], size: str, input_image_base64: str = None) -> Tuple[bool, str]:
        """发送豆包格式的HTTP请求生成图片"""
        try:
            # 尝试导入豆包SDK
            try:
                from volcenginesdkarkruntime import Ark
            except ImportError:
                logger.error(f"{self.log_prefix} (Doubao) 缺少volcenginesdkarkruntime库，请安装: pip install 'volcengine-python-sdk[ark]'")
                return False, "缺少豆包SDK，请安装volcengine-python-sdk[ark]"

            # 获取代理配置
            proxy_config = self._get_proxy_config()

            # 初始化客户端
            api_key = model_config.get("api_key", "").replace("Bearer ", "")
            client_kwargs = {
                "base_url": model_config.get("base_url"),
                "api_key": api_key,
            }

            # 如果启用了代理，配置代理
            if proxy_config:
                # 豆包SDK使用httpx，需要传递proxies参数
                proxy_url = proxy_config["http"]  # 使用统一的代理地址
                client_kwargs["proxies"] = {
                    "http://": proxy_url,
                    "https://": proxy_url
                }
                client_kwargs["timeout"] = proxy_config["timeout"]

            client = Ark(**client_kwargs)

            # 获取模型特定的配置参数
            custom_prompt_add = model_config.get("custom_prompt_add", "")
            prompt_add = prompt + custom_prompt_add

            # 构建请求参数
            request_params = {
                "model": model_config.get("model"),
                "prompt": prompt_add,
                "size": size,
                "response_format": "url",
                "watermark": model_config.get("watermark", True)
            }

            # 如果有输入图片，需要特殊处理
            if input_image_base64:
                # 尝试data URI格式
                if not input_image_base64.startswith('data:image'):
                    # 检测图片格式
                    if input_image_base64.startswith('/9j/'):
                        image_data_uri = f"data:image/jpeg;base64,{input_image_base64}"
                    elif input_image_base64.startswith('iVBORw'):
                        image_data_uri = f"data:image/png;base64,{input_image_base64}"
                    else:
                        image_data_uri = f"data:image/jpeg;base64,{input_image_base64}"
                else:
                    image_data_uri = input_image_base64

                request_params["image"] = image_data_uri
                logger.info(f"{self.log_prefix} (Doubao) 使用图生图模式，图片格式: {image_data_uri[:50]}...")

            logger.info(f"{self.log_prefix} (Doubao) 发起图片请求: {model_config.get('model')}, Size: {size}")

            response = client.images.generate(**request_params)

            if response.data and len(response.data) > 0:
                image_url = response.data[0].url
                logger.info(f"{self.log_prefix} (Doubao) 图片生成成功: {image_url[:70]}...")
                return True, image_url
            else:
                logger.error(f"{self.log_prefix} (Doubao) 响应中没有图片数据")
                return False, "豆包API响应成功但未返回图片"

        except Exception as e:
            logger.error(f"{self.log_prefix} (Doubao) 请求异常: {e!r}", exc_info=True)
            return False, f"豆包API请求失败: {str(e)[:100]}"

    def _make_openai_image_request(self, prompt: str, model_config: Dict[str, Any], size: str, strength: float = None, input_image_base64: str = None) -> Tuple[bool, str]:
        """发送OpenAI格式的HTTP请求生成图片"""
        base_url = model_config.get("base_url", "")
        generate_api_key = model_config.get("api_key", "")
        model = model_config.get("model", "")

        # 直接拼接路径，base_url应该包含完整的API版本路径
        endpoint = f"{base_url.rstrip('/')}/images/generations"

        # 获取模型特定的配置参数
        custom_prompt_add = model_config.get("custom_prompt_add", "")
        negative_prompt_add = model_config.get("negative_prompt_add", "")
        seed = model_config.get("seed", 42)
        guidance_scale = model_config.get("guidance_scale", 2.5)
        watermark = model_config.get("watermark", True)
        num_inference_steps = model_config.get("num_inference_steps", 20)
        prompt_add = prompt + custom_prompt_add
        negative_prompt = negative_prompt_add

        # 构建基本请求参数
        payload_dict = {
            "model": model,
            "prompt": prompt_add,
            "negative_prompt": negative_prompt,
            "size": size,
            "seed": seed,
            "api-key": generate_api_key
        }

        # 如果有输入图片，添加图生图参数
        if input_image_base64:
            if not input_image_base64.startswith('data:image'):
                # 检测图片格式
                if input_image_base64.startswith('/9j/'):
                    image_data_uri = f"data:image/jpeg;base64,{input_image_base64}"
                elif input_image_base64.startswith('iVBORw'):
                    image_data_uri = f"data:image/png;base64,{input_image_base64}"
                else:
                    image_data_uri = f"data:image/jpeg;base64,{input_image_base64}"
            else:
                image_data_uri = input_image_base64

            payload_dict["image"] = image_data_uri
            if strength is not None:
                payload_dict["strength"] = strength

        # 根据不同API添加特定参数
        if base_url == "https://ark.cn-beijing.volces.com/api/v3": #豆包火山方舟
            payload_dict["watermark"] = watermark
        else: #默认魔搭等其他
            payload_dict["guidance_scale"] = guidance_scale
            payload_dict["num_inference_steps"] = num_inference_steps

        data = json.dumps(payload_dict).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"{generate_api_key}",
        }

        logger.info(f"{self.log_prefix} (OpenAI) 发起图片请求: {model}, Prompt: {prompt_add[:30]}... To: {endpoint}")
        logger.debug(f"{self.log_prefix} (OpenAI) Request Headers: {{...Authorization: {generate_api_key[:10]}...}}")
        logger.debug(f"{self.log_prefix} (OpenAI) Request Body (api-key omitted): {json.dumps({k: v for k, v in payload_dict.items() if k != 'api-key'})}")

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

    def _make_modelscope_request(self, prompt: str, model_config: Dict[str, Any], size: str = None, strength: float = None, input_image_base64: str = None) -> Tuple[bool, str]:
        """发送魔搭格式的HTTP请求生成图片"""
        try:
            # API配置
            api_key = model_config.get("api_key", "").replace("Bearer ", "")
            model_name = model_config.get("model", "MusePublic/489_ckpt_FLUX_1")
            base_url = model_config.get("base_url", "https://api-inference.modelscope.cn").rstrip('/')

            # 验证API密钥
            if not api_key or api_key in ["xxxxxxxxxxxxxx", "YOUR_API_KEY_HERE"]:
                logger.error(f"{self.log_prefix} (魔搭) API密钥未配置或无效")
                return False, "魔搭API密钥未配置，请在配置文件中设置正确的API密钥"
        
            # 请求头
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-ModelScope-Async-Mode": "true"
            }

            logger.info(f"{self.log_prefix} (魔搭) 使用模型: {model_name}, API地址: {base_url}")

            # 添加额外的提示词前缀
            custom_prompt_add = model_config.get("custom_prompt_add", "")
            full_prompt = prompt + custom_prompt_add

            # 获取其他配置参数
            guidance = model_config.get("guidance_scale", 3.5)
            steps = model_config.get("num_inference_steps", 30)
            negative_prompt = model_config.get("negative_prompt_add", "")
            seed = model_config.get("seed", 42)

            # 根据是否有输入图片，构建不同的请求参数
            if input_image_base64:
                # 处理图片格式
                if not input_image_base64.startswith('data:image'):
                    # 检测图片格式
                    if input_image_base64.startswith('/9j/'):
                        image_data_uri = f"data:image/jpeg;base64,{input_image_base64}"
                    elif input_image_base64.startswith('iVBORw'):
                        image_data_uri = f"data:image/png;base64,{input_image_base64}"
                    else:
                        image_data_uri = f"data:image/jpeg;base64,{input_image_base64}"
                else:
                    image_data_uri = input_image_base64

                # 图生图请求数据
                request_data = {
                    "model": model_name,
                    "prompt": full_prompt,
                    "image_url": image_data_uri
                }
                logger.info(f"{self.log_prefix} (魔搭) 使用图生图模式，图片格式: {image_data_uri[:50]}...")
            else:
                # 文生图模式：可以使用完整参数
                request_data = {
                    "model": model_name,
                    "prompt": full_prompt
                }

                # 添加文生图的可选参数
                if negative_prompt:
                    request_data["negative_prompt"] = negative_prompt
                if size:
                    request_data["size"] = size
                request_data["seed"] = seed
                request_data["steps"] = steps
                request_data["guidance"] = guidance

                logger.info(f"{self.log_prefix} (魔搭) 使用文生图模式")

            logger.info(f"{self.log_prefix} (魔搭) 发起异步图片生成请求，模型: {model_name}")

            # 获取代理配置
            proxy_config = self._get_proxy_config()

            # 直接拼接路径，base_url应该包含完整的API版本路径
            endpoint = f"{base_url.rstrip('/')}/images/generations"

            # 构建requests的参数
            request_kwargs = {
                "url": endpoint,
                "headers": headers,
                "data": json.dumps(request_data, ensure_ascii=False).encode('utf-8'),
                "timeout": proxy_config.get('timeout', 30) if proxy_config else 30
            }

            # 如果启用了代理，添加代理配置
            if proxy_config:
                request_kwargs["proxies"] = {
                    "http": proxy_config["http"],
                    "https": proxy_config["https"]
                }

            # 发送异步请求
            response = requests.post(**request_kwargs)
        
            if response.status_code != 200:
                error_msg = response.text
                logger.error(f"{self.log_prefix} (魔搭) 请求失败: HTTP {response.status_code} - {error_msg}")
                return False, f"请求失败: {error_msg[:100]}"

            # 获取任务ID
            task_response = response.json()
            if "task_id" not in task_response:
                logger.error(f"{self.log_prefix} (魔搭) 未获取到任务ID: {task_response}")
                return False, "未获取到任务ID"
        
            task_id = task_response["task_id"]
            logger.info(f"{self.log_prefix} (魔搭) 获得任务ID: {task_id}，开始轮询结果")

            # 轮询任务结果
            check_headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-ModelScope-Task-Type": "image_generation"
            }

            max_attempts = 24  # 最多检查2分钟
            for attempt in range(max_attempts):
                try:
                    # 构建状态检查请求参数
                    status_url = f"{base_url}/tasks/{task_id}"
                    check_kwargs = {
                        "url": status_url,
                        "headers": check_headers,
                        "timeout": 10
                    }

                    # 如果启用了代理，添加代理配置
                    if proxy_config:
                        check_kwargs["proxies"] = {
                            "http": proxy_config["http"],
                            "https": proxy_config["https"]
                        }

                    check_response = requests.get(**check_kwargs)
                
                    if check_response.status_code != 200:
                        logger.warning(f"{self.log_prefix} (魔搭) 状态检查失败: HTTP {check_response.status_code}")
                        continue
                
                    result_data = check_response.json()
                    task_status = result_data.get("task_status", "UNKNOWN")
                
                    if task_status == "SUCCEED":
                        if "output_images" in result_data and result_data["output_images"]:
                            image_url = result_data["output_images"][0]
                        
                            # 下载图片并转换为base64
                            try:
                                # 构建图片下载请求参数
                                img_kwargs = {
                                    "url": image_url,
                                    "timeout": 30
                                }

                                # 如果启用了代理，添加代理配置
                                if proxy_config:
                                    img_kwargs["proxies"] = {
                                        "http": proxy_config["http"],
                                        "https": proxy_config["https"]
                                    }

                                img_response = requests.get(**img_kwargs)
                                if img_response.status_code == 200:
                                    import base64
                                    image_base64 = base64.b64encode(img_response.content).decode('utf-8')
                                    logger.info(f"{self.log_prefix} (魔搭) 图片生成成功")
                                    return True, image_base64
                                else:
                                   logger.error(f"{self.log_prefix} (魔搭) 图片下载失败: HTTP {img_response.status_code}")
                                   return False, "图片下载失败"
                            except Exception as e:
                                logger.error(f"{self.log_prefix} (魔搭) 图片下载异常: {e}")
                                return False, f"图片下载异常: {str(e)}"
                        else:
                            logger.error(f"{self.log_prefix} (魔搭) 未找到生成的图片")
                            return False, "未找到生成的图片"
            
                    elif task_status == "FAILED":
                        error_msg = result_data.get("error_message", "任务执行失败")
                        logger.error(f"{self.log_prefix} (魔搭) 任务失败: {error_msg}")
                        return False, f"任务执行失败: {error_msg}"
            
                    elif task_status in ["PENDING", "RUNNING"]:
                        logger.info(f"{self.log_prefix} (魔搭) 任务状态: {task_status}，等待中...")
                        time.sleep(5)
                        continue
            
                    else:
                        logger.warning(f"{self.log_prefix} (魔搭) 未知任务状态: {task_status}")
                        time.sleep(5)
                        continue
                
                except Exception as e:
                    logger.warning(f"{self.log_prefix} (魔搭) 状态检查异常: {e}")
                    time.sleep(5)
                    continue
    
            logger.error(f"{self.log_prefix} (魔搭) 任务超时，未能在规定时间内完成")
            return False, "任务执行超时"
    
        except Exception as e:
            logger.error(f"{self.log_prefix} (魔搭) 请求异常: {e!r}", exc_info=True)
            return False, f"请求失败: {str(e)}"

    def _make_gemini_request(self, prompt: str, model_config: Dict[str, Any], input_image_base64: str = None) -> Tuple[bool, str]:
        """发送Gemini格式的HTTP请求生成图片"""
        try:
            # API配置
            api_key = model_config.get("api_key", "").replace("Bearer ", "")
            model_name = model_config.get("model", "gemini-2.5-flash-image-preview")  # 使用最新模型
            base_url = model_config.get("base_url", "https://generativelanguage.googleapis.com").rstrip('/')
        
            # 构建API端点
            url = f"{base_url}/v1beta/models/{model_name}:generateContent"
        
            # 请求头
            headers = {
                "x-goog-api-key": api_key,
                "Content-Type": "application/json"
            }

            # 构建请求内容
            parts = [{"text": prompt}]

            # 如果有输入图片，添加到请求中
            if input_image_base64:
                logger.info(f"{self.log_prefix} (Gemini) 使用图生图模式")

                try:
                    # 移除data URI前缀（如果存在）
                    clean_base64 = input_image_base64
                    if ',' in input_image_base64:
                        clean_base64 = input_image_base64.split(',')[1]

                    # 检测MIME类型
                    if clean_base64.startswith('/9j/'):
                        mime_type = "image/jpeg"
                    elif clean_base64.startswith('iVBORw'):
                        mime_type = "image/png"
                    elif clean_base64.startswith('UklGR'):
                        mime_type = "image/webp"
                    else:
                        mime_type = "image/jpeg"  # 默认

                    # 添加图片数据到请求
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
                    "responseModalities": ["TEXT", "IMAGE"]  # 关键配置
                }
            }

            # 添加 Gemini 图片尺寸配置
            image_config = self._build_gemini_image_config(model_name, model_config)
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

            # 如果启用了代理，添加代理配置
            if proxy_config:
                request_kwargs["proxies"] = {
                    "http": proxy_config["http"],
                    "https": proxy_config["https"]
                }

            # 发送请求
            response = requests.post(**request_kwargs)

            # 检查响应状态
            if response.status_code != 200:
                error_msg = response.text
                logger.error(f"{self.log_prefix} (Gemini) API请求失败: HTTP {response.status_code} - {error_msg}")
                return False, f"API请求失败: {error_msg[:100]}"

            # 解析响应
            try:
                response_json = response.json()
            
                # 查找生成的图片数据
                if "candidates" in response_json and response_json["candidates"]:
                    candidate = response_json["candidates"][0]
                
                    if "content" in candidate and "parts" in candidate["content"]:
                        for part in candidate["content"]["parts"]:
                            # 检查是否有inline_data（图片数据）
                            if "inlineData" in part and "data" in part["inlineData"]:
                                image_base64 = part["inlineData"]["data"]
                                logger.info(f"{self.log_prefix} (Gemini) 图片生成成功")
                                return True, image_base64
                            elif "inline_data" in part and "data" in part["inline_data"]:
                                image_base64 = part["inline_data"]["data"]
                                logger.info(f"{self.log_prefix} (Gemini) 图片生成成功")
                                return True, image_base64

                # 检查是否有错误信息
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

    def _build_gemini_image_config(self, model_name: str, model_config: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """构建 Gemini 图片配置

        支持的 default_size 格式：
        - "16:9"      → { "aspectRatio": "16:9" }
        - "16:9-2K"   → { "aspectRatio": "16:9", "imageSize": "2K" }
        - "1:1-4K"    → { "aspectRatio": "1:1", "imageSize": "4K" }

        Args:
            model_name: 模型名称
            model_config: 模型配置

        Returns:
            imageConfig 字典，如果不需要配置则返回 None
        """
        size = model_config.get("default_size", "").strip()

        if not size:
            return None

        image_config = {}

        # 检查是否包含 imageSize（用 - 分隔）
        if "-" in size:
            # 格式：16:9-2K 或 1:1-4K
            parts = size.split("-", 1)
            aspect_ratio = parts[0].strip()
            image_size = parts[1].strip().upper()  # 1K, 2K, 4K

            image_config["aspectRatio"] = aspect_ratio

            # 仅 Gemini 3 Pro 支持 imageSize
            if "gemini-3" in model_name.lower():
                if image_size in ["1K", "2K", "4K"]:
                    image_config["imageSize"] = image_size
                else:
                    logger.warning(f"{self.log_prefix} (Gemini) 无效的 imageSize: {image_size}，仅支持 1K/2K/4K")
            else:
                logger.warning(f"{self.log_prefix} (Gemini) imageSize 仅支持 Gemini 3 Pro，当前模型: {model_name}")
        else:
            # 格式：16:9（纯宽高比）
            # 验证是否是有效的宽高比格式（包含冒号）
            if ":" in size:
                image_config["aspectRatio"] = size
            elif "x" in size.lower():
                # 检测到传统格式（如 1024x1024），给出警告
                logger.warning(
                    f"{self.log_prefix} (Gemini) 检测到传统尺寸格式 '{size}'，Gemini 需要宽高比格式（如 16:9）。"
                    f"将使用默认值 1:1"
                )
                image_config["aspectRatio"] = "1:1"
            else:
                # 无法识别的格式
                logger.warning(f"{self.log_prefix} (Gemini) 无法识别的尺寸格式 '{size}'，使用默认值 1:1")
                image_config["aspectRatio"] = "1:1"

        return image_config if image_config else None