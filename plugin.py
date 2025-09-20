import asyncio
import json
import urllib.request
import base64
import traceback
import toml
import os
from typing import List, Tuple, Type, Optional, Dict, Any
from threading import Lock

# 导入新插件系统
from src.plugin_system.base.base_plugin import BasePlugin
from src.plugin_system.base.base_action import BaseAction
from src.plugin_system.base.component_types import ComponentInfo, ActionActivationType, ChatMode
from src.plugin_system.base.config_types import ConfigField

# 导入新插件系统
from src.plugin_system import BasePlugin, register_plugin, ComponentInfo, ActionActivationType
from src.plugin_system.base.config_types import ConfigField

# 导入依赖的系统组件
from src.common.logger import get_logger

# 导入回复生成器API导入
from src.plugin_system import generator_api

logger = get_logger("pic_action")

# ===== Action组件 =====

class Custom_Pic_Action(BaseAction):
    """生成一张图片并发送"""

    # 激活设置
    focus_activation_type = ActionActivationType.LLM_JUDGE  # Focus模式使用LLM判定，精确理解需求
    normal_activation_type = ActionActivationType.KEYWORD  # Normal模式使用关键词激活，快速响应
    mode_enable = ChatMode.ALL
    parallel_action = True

    # 动作基本信息
    action_name = "draw_picture"
    action_description = (
        "可以根据特定的描述，生成并发送一张图片，如果没提供描述，就根据聊天内容生成。支持指定模型编号（如model1、model2等），你可以立刻画好，不用等待"
    )

    # 关键词设置（用于Normal模式）
    activation_keywords = ["画", "绘制", "生成图片", "画图", "draw", "paint", "图片生成"]

    # LLM判定提示词（用于Focus模式）
    llm_judge_prompt = """
判定是否需要使用图片生成动作的条件：
1. 用户明确@你的名字并要求画图、生成图片或创作图像
2. 用户描述了想要看到的画面或场景
3. 对话中提到需要视觉化展示某些概念
4. 用户想要创意图片或艺术作品
5. 你想要通过画图来制作表情包来表达自己的一些情绪及文字不容易表达的画面时

适合使用的情况：
- "画一张..."、"画个..."、"生成图片"
- "我想看看...的样子"
- "能画出...吗"
- "创作一幅..."
- "用模型1画..."、"model2生成..."

绝对不要使用的情况：
1. 纯文字聊天和问答
2. 只是提到"图片"、"画"等词但不是要求生成
3. 谈论已存在的图片或照片
4. 技术讨论中提到绘图概念但无生成需求
5. 用户明确表示不需要图片时
6. 刚刚成功生成过图片，避免频繁请求
"""

    keyword_case_sensitive = False

    # 动作参数定义
    action_parameters = {
        "description": "图片描述，输入你想要生成并发送的图片的描述，将描述翻译为英文单词组合，并用','分隔，描述中不要出现中文，必填",
        "model_id": "要使用的模型ID（如model1、model2、model3等，默认使用default_model配置的模型）",
        "size": "图片尺寸，如512x512、1024x1024等（可选，不指定则使用模型默认尺寸）",
    }

    # 动作使用场景
    action_require = [
        "当有人要求你生成并发送一张图片时使用，不要频率太高",
        "重点：不要连续发，如果你在前10句内已经发送过[图片]或者[表情包]或记录出现过类似描述的[图片]，就不要选择此动作",
        "支持指定模型：用户可以通过'用模型1画'、'model2生成'等方式指定特定模型"
    ]
    associated_types = ["text", "image"]
    
    # 简单的请求缓存，避免短时间内重复请求
    _request_cache = {}
    _cache_max_size = 10

    async def execute(self) -> Tuple[bool, Optional[str]]:
        """执行图片生成动作"""
        logger.info(f"{self.log_prefix} 执行绘图模型图片生成动作")

        # 获取参数
        description = self.action_data.get("description", "").strip()
        model_id = self.action_data.get("model_id", "").strip()
        size = self.action_data.get("size", "").strip()

        # 参数验证
        if not description:
            logger.warning(f"{self.log_prefix} 图片描述为空，无法生成图片。")
            await self.send_text("你需要告诉我想要画什么样的图片哦~ 比如说'画一只可爱的小猫'")
            return False, "图片描述为空"

        # 清理和验证描述
        if len(description) > 1000:  # 限制描述长度
            description = description[:1000]
            logger.info(f"{self.log_prefix} 图片描述过长，已截断")

        # 获取模型配置
        model_config = self._get_model_config(model_id)
        if not model_config:
            error_msg = f"指定的模型 '{model_id}' 不存在或配置无效，请检查配置文件。"
            await self.send_text(error_msg)
            logger.error(f"{self.log_prefix} 模型配置获取失败: {model_id}")
            return False, "模型配置无效"

        # 配置验证
        http_base_url = model_config.get("base_url")
        http_api_key = model_config.get("api_key")
        if not (http_base_url and http_api_key):
            error_msg = "抱歉，图片生成功能所需的HTTP配置（如API地址或密钥）不完整，无法提供服务。"
            await self.send_text(error_msg)
            logger.error(f"{self.log_prefix} HTTP调用配置缺失: base_url 或 api_key.")
            return False, "HTTP配置不完整"

        # API密钥验证
        if "YOUR_API_KEY_HERE" in http_api_key or "xxxxxxxxxxxxxx" in http_api_key:
            error_msg = "图片生成功能尚未配置，请设置正确的API密钥。"
            await self.send_text(error_msg)
            logger.error(f"{self.log_prefix} API密钥未配置")
            return False, "API密钥未配置"

        # 获取模型配置参数
        model_name = model_config.get("model", "default-model")
        api_format = model_config.get("format", "openai")
        enable_default_size = model_config.get("fixed_size_enabled","false")#获取是否启用自定义图片大小，如果启用，将图片大小指定为固定值
        if enable_default_size:
            size = None 
            logger.info(f"{self.log_prefix} 使用自定义固定大小")
        image_size = size or model_config.get("default_size", "1024x1024")

        # 验证图片尺寸格式
        if not self._validate_image_size(image_size):
            logger.warning(f"{self.log_prefix} 无效的图片尺寸: {image_size}，使用模型默认值")
            image_size = model_config.get("default_size", "1024x1024")

        # 检查缓存
        cache_key = self._get_cache_key(description, model_name, image_size)
        if cache_key in self._request_cache:
            cached_result = self._request_cache[cache_key]
            logger.info(f"{self.log_prefix} 使用缓存的图片结果")
            await self.send_text("我之前画过类似的图片，用之前的结果~")

            # 直接发送缓存的结果
            send_success = await self.send_image(cached_result)
            if send_success:
                return True, "图片已发送(缓存)"
            else:
                # 缓存失败，清除这个缓存项并继续正常流程
                del self._request_cache[cache_key]

        # 获取其他模型配置参数
        seed_val = model_config.get("seed", 42)
        guidance_scale_val = model_config.get("guidance_scale", 2.5)
        watermark_val = model_config.get("watermark", True)
        enable_debug = self.get_config("components.enable_debug_info", False)

        if enable_debug:
            await self.send_text(
                f"收到！正在为您使用 {model_id or '默认'} 模型生成关于 '{description}' 的图片，请稍候...（模型: {model_name}, 尺寸: {image_size}）"
            )

        try:
            if api_format == "siliconflow":
                success, result = await asyncio.to_thread(
                    self._make_siliconflow_request,
                    prompt=description,
                    model_config=model_config,
                    size=image_size,
                    guidance_scale=guidance_scale_val,
                )
            elif api_format == "doubao":
                success, result = await asyncio.to_thread(
                    self._make_doubao_request,
                    prompt=description,
                    model_config=model_config,
                    size=image_size,
                    watermark=watermark_val,
                )
            elif api_format == "gemini":
                success, result = await asyncio.to_thread(
                    self._make_gemini_image_request,
                    prompt=description,
                    model_config=model_config,
                )
            else:  # 默认为openai格式
                success, result = await asyncio.to_thread(
                    self._make_openai_image_request,
                    prompt=description,
                    model_config=model_config,
                    size=image_size,
                    seed=seed_val,
                    guidance_scale=guidance_scale_val,
                    watermark=watermark_val,
                )
        except Exception as e:
            logger.error(f"{self.log_prefix} (HTTP) 异步请求执行失败: {e!r}", exc_info=True)
            traceback.print_exc()
            success = False
            result = f"图片生成服务遇到意外问题: {str(e)[:100]}"

        if success:
            # 如果返回的是Base64数据（以"iVBORw"等开头），直接使用
            if result.startswith(("iVBORw", "/9j/", "UklGR", "R0lGOD")):  # 常见图片格式的Base64前缀
                send_success = await self.send_image(result)
                if send_success:
                    await self.send_text("图片表情已发送！")
                    return True, "图片表情已发送(Base64)"
                else:
                    await self.send_text("图片已处理为Base64，但作为表情发送失败了")
                    return False, "图片表情发送失败 (Base64)"
            else:  # 否则认为是URL
                image_url = result
                logger.info(f"{self.log_prefix} 图片URL获取成功: {image_url[:70]}... 下载并编码.")

                try:
                    encode_success, encode_result = await asyncio.to_thread(self._download_and_encode_base64, image_url)
                except Exception as e:
                    logger.error(f"{self.log_prefix} (B64) 异步下载/编码失败: {e!r}", exc_info=True)
                    traceback.print_exc()
                    encode_success = False
                    encode_result = f"图片下载或编码时发生内部错误: {str(e)[:100]}"

                if encode_success:
                    base64_image_string = encode_result
                    send_success = await self.send_image(base64_image_string)
                    if send_success:
                        # 缓存成功的结果
                        self._request_cache[cache_key] = base64_image_string
                        self._cleanup_cache()
                        return True, "图片已成功生成并发送"
                    else:
                        await self.send_text("图片已处理为Base64，但发送失败了。")
                        return False, "图片发送失败 (Base64)"
                else:
                    await self.send_text(f"获取到图片URL，但在处理图片时失败了：{encode_result}")
                    return False, f"图片处理失败(Base64): {encode_result}"
        else:
            error_message = result
            await self.send_text(f"哎呀，生成图片时遇到问题：{error_message}")
            return False, f"图片生成失败: {error_message}"

    def _get_model_config(self, model_id: str = None) -> Dict[str, Any]:
        """获取指定模型的配置，支持热重载"""
        # 如果没有指定模型ID，使用默认模型
        if not model_id:
            model_id = self.get_config("generation.default_model", "model1")
        
        # 构建模型配置的路径
        model_config_path = f"models.{model_id}"
        model_config = self.get_config(model_config_path)
        
        if not model_config:
            logger.warning(f"{self.log_prefix} 模型 {model_id} 配置不存在，尝试使用默认模型")
            # 尝试获取默认模型
            default_model_id = self.get_config("generation.default_model", "model1")
            if default_model_id != model_id:
                model_config = self.get_config(f"models.{default_model_id}")
        
        return model_config or {}

    def _make_siliconflow_request(self, prompt: str, model_config: Dict[str, Any], size: str, guidance_scale: float) -> Tuple[bool, str]:
        """发送SiliconFlow格式的HTTP请求生成图片"""
        base_url = model_config.get("base_url", "")
        api_key = model_config.get("api_key", "")
        model = model_config.get("model", "")

        endpoint = f"{base_url.rstrip('/')}/images/generations"

        # 获取模型特定的配置参数
        custom_prompt_add = model_config.get("custom_prompt_add", "")
        prompt_add = prompt + custom_prompt_add

        payload = {
            "model": model,
            "prompt": prompt_add,
            "image_size": size,  # SiliconFlow使用image_size而不是size
            "batch_size": 1,
            "num_inference_steps": 20,
            "guidance_scale": guidance_scale
        }

        headers = {
            "Authorization": api_key if api_key.startswith("Bearer ") else f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")

        logger.info(f"{self.log_prefix} (SiliconFlow) 发起图片请求: {model}, Prompt: {prompt_add[:30]}... To: {endpoint}")

        try:
            with urllib.request.urlopen(req, timeout=600) as response:
                response_body = response.read().decode("utf-8")
                logger.info(f"{self.log_prefix} (SiliconFlow) 响应: {response.status}")

                if 200 <= response.status < 300:
                    response_data = json.loads(response_body)
                    
                    # 解析SiliconFlow响应格式
                    if "images" in response_data and response_data["images"]:
                        image_data = response_data["images"][0]
                        if "url" in image_data:
                            return True, image_data["url"]
                        elif "b64_json" in image_data:
                            return True, image_data["b64_json"]
                    
                    logger.error(f"{self.log_prefix} (SiliconFlow) 响应中未找到图片数据: {response_body[:300]}")
                    return False, "API响应成功但未找到图片数据"
                else:
                    logger.error(f"{self.log_prefix} (SiliconFlow) API错误: {response.status} - {response_body[:300]}")
                    return False, f"API请求失败(状态码 {response.status})"

        except Exception as e:
            logger.error(f"{self.log_prefix} (SiliconFlow) 请求异常: {e!r}", exc_info=True)
            return False, f"请求过程中发生错误: {str(e)[:100]}"

    def _make_doubao_request(self, prompt: str, model_config: Dict[str, Any], size: str, watermark: bool) -> Tuple[bool, str]:
        """发送豆包格式的HTTP请求生成图片"""
        try:
            # 尝试导入豆包SDK
            try:
                from volcenginesdkarkruntime import Ark
            except ImportError:
                logger.error(f"{self.log_prefix} (Doubao) 缺少volcenginesdkarkruntime库，请安装: pip install 'volcengine-python-sdk[ark]'")
                return False, "缺少豆包SDK，请安装volcengine-python-sdk[ark]"

            # 初始化客户端
            api_key = model_config.get("api_key", "").replace("Bearer ", "")
            client = Ark(
                base_url=model_config.get("base_url"),
                api_key=api_key,
            )

            # 获取模型特定的配置参数
            custom_prompt_add = model_config.get("custom_prompt_add", "")
            prompt_add = prompt + custom_prompt_add
            doubao_size = size  # 直接使用如 "1024x1024" 格式

            logger.info(f"{self.log_prefix} (Doubao) 发起图片请求: {model_config.get('model')}, Size: {doubao_size}")

            response = client.images.generate(
                model=model_config.get("model"),
                prompt=prompt_add,
                size=doubao_size,  # 使用标准格式
                response_format="url",
                watermark=watermark
            )

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

    def _download_and_encode_base64(self, image_url: str) -> Tuple[bool, str]:
        """下载图片并将其编码为Base64字符串"""
        logger.info(f"{self.log_prefix} (B64) 下载并编码图片: {image_url[:70]}...")
        try:
            with urllib.request.urlopen(image_url, timeout=600) as response:
                if response.status == 200:
                    image_bytes = response.read()
                    base64_encoded_image = base64.b64encode(image_bytes).decode("utf-8")
                    logger.info(f"{self.log_prefix} (B64) 图片下载编码完成. Base64长度: {len(base64_encoded_image)}")
                    return True, base64_encoded_image
                else:
                    error_msg = f"下载图片失败 (状态: {response.status})"
                    logger.error(f"{self.log_prefix} (B64) {error_msg} URL: {image_url}")
                    return False, error_msg
        except Exception as e: 
            logger.error(f"{self.log_prefix} (B64) 下载或编码时错误: {e!r}", exc_info=True)
            traceback.print_exc()
            return False, f"下载或编码图片时发生错误: {str(e)[:100]}"        
        
    @classmethod
    def _get_cache_key(cls, description: str, model: str, size: str) -> str:
        """生成缓存键"""
        return f"{description[:100]}|{model}|{size}"

    @classmethod
    def _cleanup_cache(cls):
        """清理缓存，保持大小在限制内"""
        if len(cls._request_cache) > cls._cache_max_size:
            keys_to_remove = list(cls._request_cache.keys())[: -cls._cache_max_size // 2]
            for key in keys_to_remove:
                del cls._request_cache[key]

    def _validate_image_size(self, image_size: str) -> bool:
        """验证图片尺寸格式"""
        try:
            width, height = map(int, image_size.split("x"))
            return 100 <= width <= 10000 and 100 <= height <= 10000
        except (ValueError, TypeError):
            return False

    def _make_openai_image_request(
        self, prompt: str, model_config: Dict[str, Any], size: str, seed: int | None, guidance_scale: float, watermark: bool
    ) -> Tuple[bool, str]:
        """发送OpenAI格式的HTTP请求生成图片"""
        base_url = model_config.get("base_url", "")
        generate_api_key = model_config.get("api_key", "")
        model = model_config.get("model", "")

        endpoint = f"{base_url.rstrip('/')}/images/generations"

        # 获取模型特定的配置参数
        custom_prompt_add = model_config.get("custom_prompt_add", "")
        negative_prompt_add = model_config.get("negative_prompt_add", "")

        prompt_add = prompt + custom_prompt_add
        negative_prompt = negative_prompt_add
        if base_url == "https://ark.cn-beijing.volces.com/api/v3": #豆包火山方舟
            payload_dict = {
                "model": model,
                "prompt": prompt_add,
                "negative_prompt": negative_prompt,
                "size": size,
                #"guidance_scale": guidance_scale,
                "seed": seed,
                "api-key": generate_api_key,
                "watermark": watermark
            }
        else :#默认魔搭等其他
            payload_dict = {
                "model": model,
                "prompt": prompt_add,  # 使用附加的正面提示词
                "negative_prompt": negative_prompt,
                "size": size,  # 固定size
                "guidance_scale": guidance_scale,#豆包会报错
                "seed": seed,  # seed is now always an int from process()
                "api-key": generate_api_key
                #"watermark": watermark#其他请求不需要发送水印
            }

        data = json.dumps(payload_dict).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"{generate_api_key}",
        }

        logger.info(f"{self.log_prefix} (OpenAI) 发起图片请求: {model}, Prompt: {prompt_add[:30]}... To: {endpoint}")
        logger.debug(f"{self.log_prefix} (OpenAI) Request Headers: {{...Authorization: {generate_api_key[:10]}...}}")
        logger.debug(f"{self.log_prefix} (OpenAI) Request Body (api-key omitted): {json.dumps({k: v for k, v in payload_dict.items() if k != 'api-key'})}")

        req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=600) as response:
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

    def _make_gemini_image_request(self, prompt: str, model_config: Dict[str, Any]) -> Tuple[bool, str]:
        """发送Gemini格式的HTTP请求生成图片"""
        base_url = model_config.get("base_url", "")
        generate_api_key = model_config.get("api_key", "")
        model = model_config.get("model", "")

        endpoint = base_url

        # 获取模型特定的配置参数
        custom_prompt_add = model_config.get("custom_prompt_add", "")
        prompt_add = prompt + custom_prompt_add

        # 构建Gemini格式的请求体
        payload_dict = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt_add
                        }
                    ]
                }
            ],
            "generationConfig": {
                "responseModalities": [
                    "TEXT",
                    "IMAGE"
                ]
            }
        }

        data = json.dumps(payload_dict).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"{generate_api_key}",
        }

        logger.info(f"{self.log_prefix} (Gemini) 发起图片请求: {model}, Prompt: {prompt_add[:30]}... To: {endpoint}")

        req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=600) as response:
                response_status = response.status
                response_body_bytes = response.read()
                response_body_str = response_body_bytes.decode("utf-8")

                logger.info(f"{self.log_prefix} (Gemini) 响应: {response_status}. Preview: {response_body_str[:150]}...")

                if 200 <= response_status < 300:
                    response_data = json.loads(response_body_str)
                    
                    # 解析Gemini格式的响应
                    if "candidates" in response_data and response_data["candidates"]:
                        candidate = response_data["candidates"][0]
                        if "content" in candidate and "parts" in candidate["content"]:
                            for part in candidate["content"]["parts"]:
                                if "inlineData" in part:
                                    inline_data = part["inlineData"]
                                    if "data" in inline_data:
                                        b64_data = inline_data["data"]
                                        logger.info(f"{self.log_prefix} (Gemini) 获取到Base64图片数据，长度: {len(b64_data)}")
                                        return True, b64_data
                    
                    logger.error(f"{self.log_prefix} (Gemini) API成功但无图片数据. 响应预览: {response_body_str[:300]}...")
                    return False, "图片生成API响应成功但未找到图片数据"
                else:
                    logger.error(f"{self.log_prefix} (Gemini) API请求失败. 状态: {response.status}. 正文: {response_body_str[:300]}...")
                    return False, f"图片API请求失败(状态码 {response.status})"
        except Exception as e:
            logger.error(f"{self.log_prefix} (Gemini) 图片生成时意外错误: {e!r}", exc_info=True)
            traceback.print_exc()
            return False, f"图片生成HTTP请求时发生意外错误: {str(e)[:100]}"


# ===== 插件注册 =====
@register_plugin
class CustomPicPlugin(BasePlugin):
    """支持多模型配置的图片生成插件"""
    
    # 插件基本信息
    plugin_name = "custom_pic_plugin"  # 内部标识符
    plugin_version = "3.1.0"
    plugin_author = "Ptrel"
    enable_plugin = True
    dependencies: List[str] = []  # 插件依赖列表
    python_dependencies: List[str] = []  # Python包依赖列表
    config_file_name = "config.toml"

    # 步骤1: 定义配置节的描述
    config_section_descriptions = {
        "plugin": "插件启用配置",
        "generation": "图片生成默认配置",
        "models": "多模型配置，每个模型都有独立的参数设置",
        "cache": "结果缓存配置",
        "components": "组件启用配置",
        "logging": "日志配置"
    }

    # 步骤2: 使用ConfigField定义详细的配置Schema
    config_schema = {
        "plugin": {
            "name": ConfigField(type=str, default="custom_pic_plugin", description="自定义多模型绘图插件", required=True),
            "config_version": ConfigField(type=str, default="3.1.6", description="插件版本号"),
            "enabled": ConfigField(type=bool, default=False, description="是否启用插件")
        },
        "generation": {
            "default_model": ConfigField(
                type=str,
                default="model1",
                description="用户指定模型,可以在配置文件中添加更多模型配置，示例中依次为，硅基流动、魔搭社区（每天免费次数）、豆包、GPT生图、谷歌Gemini",
                choices=["model1", "model2", "model3", "model4","model5"]
            ),
        },
                "cache": {
            "enabled": ConfigField(type=bool, default=True, description="是否启用请求缓存"),
            "max_size": ConfigField(type=int, default=10, description="最大缓存数量"),
        },
        "components": {
            "enable_image_generation": ConfigField(type=bool, default=True, description="是否启用图片生成Action"),
            "enable_debug_info": ConfigField(type=bool, default=False, description="是否启用调试信息显示,启用后会将生图提示词发送到群内")
        },
        "logging": {
            "level": ConfigField(
                type=str,
                default="INFO",
                description="日志记录级别",
                choices=["DEBUG", "INFO", "WARNING", "ERROR"]
            ),
            "prefix": ConfigField(type=str, default="[custom_pic_Plugin]", description="日志记录前缀")
        },
        "models":{},
        "models.model1": {
            "name": ConfigField(type=str, default="SiliconFlow模型", description="模型显示名称"),
            "base_url": ConfigField(
                type=str,
                default="https://api.siliconflow.cn/v1",
                description="SiliconFlow API的基础URL",
                required=True
            ),
            "api_key": ConfigField(
                type=str,
                default="Bearer sk-xxxxxxxxxxxxxxxxxxxxxx",
                description="SiliconFlow API密钥，需要Bearer前缀",
                required=True
            ),
            "format": ConfigField(
                type=str,
                default="siliconflow",
                description="API请求格式，使用siliconflow专用格式",
                choices=["siliconflow", "openai", "gemini", "doubao"]
            ),
            "model": ConfigField(
                type=str,
                default="Kwai-Kolors/Kolors",
                description="SiliconFlow平台的具体模型名称"
            ),
            "fixed_size_enable": ConfigField(
                type=bool,
                default= False,
                description="是否启用固定图片大小，启用后只会发配置文件中定义的大小，否则会由麦麦自己选择。（gpt-image-1 生图模型不支持 512 大小图片，需要固定 1024x1024）"
            ),
            "default_size": ConfigField(
                type=str,
                default="1024x1024",
                description="默认图片尺寸",
                choices=["512x512", "1024x1024", "1024x1280", "1280x1024", "1024x1536", "1536x1024"]
            ),
            "seed": ConfigField(type=int, default=42, description="随机种子"),
            "guidance_scale": ConfigField(type=float, default=7.5, description="SiliconFlow推荐的指导强度"),
            "watermark": ConfigField(type=bool, default=True, description="是否添加水印"),
            "custom_prompt_add": ConfigField(
                type=str,
                default=", high quality, detailed, masterpiece",
                description="SiliconFlow附加提示词，支持英文描述"
            ),
            "negative_prompt_add": ConfigField(
                type=str,
                default="low quality, blurry, distorted, ugly",
                description="SiliconFlow负面提示词"
            ),
        },
        "models.model2": {
            "name": ConfigField(type=str, default="魔搭潦草模型", description="模型显示名称"),
            "base_url": ConfigField(
                type=str,
                default="https://api-inference.modelscope.cn/v1",
                description="魔搭社区API的基础url",
                required=True
            ),
            "api_key": ConfigField(
                type=str,
                default="Bearer xxxxxxxxxxxxxxxxxxxxxx",
                description="魔搭社区API密钥，需要添加'Bearer '前缀",
                required=True
            ),
            "format": ConfigField(
                type=str,
                default="openai",
                description="API请求格式",
                choices=["openai", "gemini", "siliconflow", "doubao"]
            ),
            "model": ConfigField(
                type=str,
                default="cancel13/liaocao",
                description="具体的模型名称"
            ),
            "fixed_size_enable": ConfigField(
                type=bool,
                default= False,
                description="是否启用固定图片大小，启用后只会发配置文件中定义的大小，否则会由麦麦自己选择。（gpt-image-1 生图模型不支持 512 大小图片，需要固定 1024x1024）"
            ),
            "default_size": ConfigField(
                type=str,
                default="1024x1024",
                description="默认图片尺寸",
                choices=["512x512", "1024x1024", "1024x1280", "1280x1024", "1024x1536", "1536x1024"]
            ),
            "seed": ConfigField(type=int, default=42, description="随机种子"),
            "guidance_scale": ConfigField(type=float, default=2.5, description="模型指导强度"),
            "watermark": ConfigField(type=bool, default=True, description="是否添加水印"),
            "custom_prompt_add": ConfigField(
                type=str,
                default=",Nordic picture book art style, minimalist flat design, soft rounded lines, high saturation color blocks collision, dominant forest green and warm orange palette, low contrast lighting, hand-drawn pencil texture, healing fairy-tale atmosphere, geometric natural forms, ample white space composition, warm and clean aesthetic,liaocao\"#北欧绘本艺术风格，简约扁平设计，柔和圆润线条，高饱和度色块碰撞，森林绿与暖橙主色调，低对比度光影，手绘铅笔质感，治愈系童话氛围，几何化自然形态，留白构图，温暖干净画面",
                description="正面附加提示词（因为为附加，开头需要添加一个英文逗号','，该参数不参与 LLM 模型转换，属于直接发送的参数，使用英文，使用词语和逗号的形式，不使用描述的原因为：为了确保提示词能够精准生效，防止 lora 关键词被替换。豆包可以直接使用中文句子作为提示词）。"
            ),
            "negative_prompt_add": ConfigField(
                type=str,
                default="Pornography,nudity,lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
                description="负面附加提示词，保持默认或使用豆包时可留空，留空时保持两个英文双引号，否则会报错。"
            ),
        },
        "models.model3": {
            "name": ConfigField(type=str, default="豆包图像模型", description="模型显示名称"),
            "base_url": ConfigField(
                type=str,
                default="https://ark.cn-beijing.volces.com/api/v3",
                description="豆包API的基础url",
                required=True
            ),
            "api_key": ConfigField(
                type=str,
                default="Bearer xxxxxxxxxxxxxxxxxxxxxx",
                description="豆包API密钥，需要Bearer前缀",
                required=True
            ),
            "format": ConfigField(
                type=str,
                default="doubao",
                description="API请求格式，使用豆包专用格式,openai格式也兼容豆包",
                choices=["doubao", "openai", "gemini", "siliconflow"]
            ),
            "model": ConfigField(
                type=str,
                default="doubao-seedream-4-0-250828",
                description="豆包具体的模型名称"
            ),
            "fixed_size_enable": ConfigField(
                type=bool,
                default= False,
                description="是否启用固定图片大小，启用后只会发配置文件中定义的大小，否则会由麦麦自己选择。（gpt-image-1 生图模型不支持 512 大小图片，需要固定 1024x1024）"
            ),
            "default_size": ConfigField(
                type=str,
                default="1024x1024",
                description="默认图片尺寸",
                choices=["512x512", "1024x1024", "1024x1280", "1280x1024", "1024x1536", "1536x1024"]
            ),
            "seed": ConfigField(type=int, default=42, description="随机种子"),
            "guidance_scale": ConfigField(type=float, default=2.5, description="模型指导强度"),
            "watermark": ConfigField(type=bool, default=True, description="是否添加水印"),
            "custom_prompt_add": ConfigField(
                type=str,
                default="",
                description="豆包支持中文提示词，可以直接使用中文描述"
            ),
            "negative_prompt_add": ConfigField(
                type=str,
                default="",
                description="豆包负面提示词，可留空"
            ),
        },
        "models.model4": {
            "name": ConfigField(type=str, default="GPT图像模型", description="模型显示名称"),
            "base_url": ConfigField(
                type=str,
                default="https://apihk.unifyllm.top/v1/",
                description="apihk 提供的 GPT API的基础url",
                required=True
            ),
            "api_key": ConfigField(
                type=str,
                default="Bearer sk-xxxxxxxxxxxxxxxxxxxxxx",
                description="API密钥，chatanywhere不需要Bearer前缀",
                required=True
            ),
            "format": ConfigField(
                type=str,
                default="openai",
                description="API请求格式",
                choices=["openai", "gemini", "siliconflow", "doubao"]
            ),
            "model": ConfigField(
                type=str,
                default="gpt-image-1",
                description="具体的模型名称"
            ),
            "fixed_size_enable": ConfigField(
                type=bool,
                default= True,
                description="是否启用固定图片大小，启用后只会发配置文件中定义的大小，否则会由麦麦自己选择。（gpt-image-1 生图模型不支持 512 大小图片，需要固定 1024x1024）"
            ),
            "default_size": ConfigField(
                type=str,
                default="1024x1024",
                description="默认图片尺寸，GPT不支持512x512",
                choices=["1024x1024", "1024x1280", "1280x1024", "1024x1536", "1536x1024"]
            ),
            "seed": ConfigField(type=int, default=42, description="随机种子"),
            "guidance_scale": ConfigField(type=float, default=2.5, description="模型指导强度"),
            "watermark": ConfigField(type=bool, default=False, description="是否添加水印"),
            "custom_prompt_add": ConfigField(
                type=str,
                default=",masterpiece, best quality, high res, Japanese animation style, illustration,soft cinematic lighting, warm lighting from the side, muted color palette, intricate details, dynamic composition, detailed background,delicate colors, graceful composition, strong emotional tension",
                description="GPT图像生成附加提示词"
            ),
            "negative_prompt_add": ConfigField(
                type=str,
                default="Pornography,nudity,lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
                description="GPT图像生成负面提示词"
            ),
        },
        "models.model5": {
            "name": ConfigField(type=str, default="gemini图像模型", description="模型显示名称"),
            "base_url": ConfigField(
                type=str,
                default="https://apihk.unifyllm.top/v1beta/models/gemini-2.5-flash-image-preview:generateContent",
                description="apihk 提供的 gemini API的基础url",
                required=True
            ),
            "api_key": ConfigField(
                type=str,
                default="Bearer sk-xxxxxxxxxxxxxxxxxxxxxx",
                description="API密钥，chatanywhere不需要Bearer前缀",
                required=True
            ),
            "format": ConfigField(
                type=str,
                default="gemini",
                description="API请求格式",
                choices=["openai", "gemini", "siliconflow", "doubao"]
            ),
            "model": ConfigField(
                type=str,
                default="gemini-2.5-flash-image-preview",
                description="具体的模型名称"
            ),
            "fixed_size_enable": ConfigField(
                type=bool,
                default= False,
                description="是否启用固定图片大小，启用后只会发配置文件中定义的大小，否则会由麦麦自己选择。（gpt-image-1 生图模型不支持 512 大小图片，需要固定 1024x1024）"
            ),
            "default_size": ConfigField(
                type=str,
                default="1024x1024",
                description="默认图片尺寸，GPT不支持512x512",
                choices=["1024x1024", "1024x1280", "1280x1024", "1024x1536", "1536x1024"]
            ),
            "seed": ConfigField(type=int, default=42, description="随机种子"),
            "guidance_scale": ConfigField(type=float, default=2.5, description="模型指导强度"),
            "watermark": ConfigField(type=bool, default=False, description="是否添加水印"),
            "custom_prompt_add": ConfigField(
                type=str,
                default=",masterpiece, best quality, high res, Japanese animation style, illustration,soft cinematic lighting, warm lighting from the side, muted color palette, intricate details, dynamic composition, detailed background,delicate colors, graceful composition, strong emotional tension",
                description="GPT图像生成附加提示词"
            ),
            "negative_prompt_add": ConfigField(
                type=str,
                default="Pornography,nudity,lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
                description="GPT图像生成负面提示词"
            ),
        }
    }

    def get_plugin_components(self) -> List[Tuple[ComponentInfo, Type]]:
        """返回插件包含的组件列表"""
        enable_image_generation = self.get_config("components.enable_image_generation", True)
        components = []

        if enable_image_generation:
            components.append((Custom_Pic_Action.get_action_info(), Custom_Pic_Action))

        return components
