import asyncio
import json
import urllib.request
import base64
import traceback
import toml
import os
import time
import requests
from typing import List, Tuple, Type, Optional, Dict, Any
from threading import Lock

# 导入新插件系统
from src.plugin_system.base.base_plugin import BasePlugin
from src.plugin_system.base.base_action import BaseAction
from src.plugin_system.base.component_types import ComponentInfo, ActionActivationType, ChatMode
from src.plugin_system import BasePlugin, register_plugin, ComponentInfo, ActionActivationType
from src.plugin_system.base.config_types import ConfigField

# 导入依赖的系统组件
from src.common.logger import get_logger

# 导入回复生成器API导入
from src.plugin_system import generator_api

logger = get_logger("pic_action")

# ===== 统一图片生成Action组件 =====
class Custom_Pic_Action(BaseAction):
    """统一的图片生成动作，智能检测文生图或图生图"""

    # 激活设置
    focus_activation_type = ActionActivationType.LLM_JUDGE  # Focus模式使用LLM判定，精确理解需求
    normal_activation_type = ActionActivationType.KEYWORD  # Normal模式使用关键词激活，快速响应
    mode_enable = ChatMode.ALL
    parallel_action = True

    # 动作基本信息
    action_name = "draw_picture"
    action_description = (
        "智能图片生成：根据描述生成图片（文生图）或基于现有图片进行修改（图生图）。"
        "自动检测用户是否提供了输入图片来决定使用文生图还是图生图模式。"
        "支持多种API格式：OpenAI、豆包、Gemini、硅基流动、魔搭社区等。"
    )

    # 关键词设置（用于Normal模式）
    activation_keywords = [
        # 文生图关键词
        "画", "绘制", "生成图片", "画图", "draw", "paint", "图片生成", "创作",
        # 图生图关键词
        "图生图", "修改图片", "基于这张图", "img2img", "重画", "改图", "图片修改",
        "改成", "换成", "变成", "转换成", "风格", "画风", "改风格", "换风格",
        "这张图", "这个图", "图片风格", "改画风", "重新画", "再画", "重做"
    ]

    # LLM判定提示词（用于Focus模式）
    llm_judge_prompt = """
判定是否需要使用图片生成动作的条件：

**文生图场景：**
1. 用户明确@你的名字并要求画图、生成图片或创作图像
2. 用户描述了想要看到的画面或场景
3. 对话中提到需要视觉化展示某些概念
4. 用户想要创意图片或艺术作品
5. 你想要通过画图来制作表情包表达情绪

**图生图场景：**
1. 用户发送了图片并@你的名字要求基于该图片进行修改或重新生成
2. 用户明确@你的名字要求并提到"图生图"、"修改图片"、"基于这张图"等关键词
3. 用户想要改变现有图片的风格、颜色、内容等
4. 用户要求在现有图片基础上添加或删除元素

**绝对不要使用的情况：**
1. 纯文字聊天和问答
2. 只是提到"图片"、"画"等词但不是要求生成
3. 谈论已存在的图片或照片（仅讨论不修改）
4. 技术讨论中提到绘图概念但无生成需求
5. 用户明确表示不需要图片时
6. 刚刚成功生成过图片，避免频繁请求
"""

    keyword_case_sensitive = False

    # 动作参数定义
    action_parameters = {
        "description": "图片描述，输入你想要生成或修改的图片的描述，将描述翻译为英文单词组合，并用','分隔，描述中不要出现中文，必填",
        "model_id": "要使用的模型ID（如model1、model2、model3等，默认使用default_model配置的模型）",
        "strength": "图生图强度，0.1-1.0之间，值越高变化越大（仅图生图时使用，可选，默认0.7）",
        "size": "图片尺寸，如512x512、1024x1024等（可选，不指定则使用模型默认尺寸）",
    }

    # 动作使用场景
    action_require = [
        "当用户要求生成或修改图片时使用，不要频率太高",
        "自动检测是否有输入图片来决定文生图或图生图模式",
        "重点：不要连续发，如果你在前10句内已经发送过[图片]或者[表情包]或记录出现过类似描述的[图片]，就不要选择此动作",
        "支持指定模型：用户可以通过'用模型1画'、'model2生成'等方式指定特定模型"
    ]
    associated_types = ["text", "image"]
    
    # 缓存系统
    _request_cache = {}  # 文生图缓存
    _img2img_cache = {}  # 图生图缓存
    _cache_max_size = 100  # 最大缓存数量
    _img2img_cache_max_size = 50  # 图生图缓存最大数量

    async def execute(self) -> Tuple[bool, Optional[str]]:
        """执行统一图片生成动作"""
        logger.info(f"{self.log_prefix} 执行统一图片生成动作")

        # 获取参数
        description = self.action_data.get("description", "").strip()
        model_id = self.action_data.get("model_id", "").strip()
        strength = self.action_data.get("strength", 0.7)
        size = self.action_data.get("size", "").strip()

        # 参数验证
        if not description:
            logger.warning(f"{self.log_prefix} 图片描述为空，无法生成图片。")
            await self.send_text("你需要告诉我想要画什么样的图片哦~ 比如说'画一只可爱的小猫'")
            return False, "图片描述为空"

        # 清理和验证描述
        if len(description) > 1000:
            description = description[:1000]
            logger.info(f"{self.log_prefix} 图片描述过长，已截断至1000字符")

        # 验证strength参数
        try:
            strength = float(strength)
            if not (0.1 <= strength <= 1.0):
                strength = 0.7
        except (ValueError, TypeError):
            strength = 0.7

        # **智能检测：判断是文生图还是图生图**
        input_image_base64 = await self._get_recent_image()
        is_img2img_mode = input_image_base64 is not None

        if is_img2img_mode:
            logger.info(f"{self.log_prefix} 检测到输入图片，使用图生图模式")
            return await self._execute_unified_generation(description, model_id, size, strength, input_image_base64)
        else:
            logger.info(f"{self.log_prefix} 未检测到输入图片，使用文生图模式")
            return await self._execute_unified_generation(description, model_id, size, None, None)

    # ===== 统一的图片生成方法 =====
    async def _execute_unified_generation(self, description: str, model_id: str, size: str, strength: float = None, input_image_base64: str = None) -> Tuple[bool, Optional[str]]:
        """统一的图片生成执行方法"""
        
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
        enable_default_size = model_config.get("fixed_size_enabled", False)
        
        if enable_default_size:
            size = None 
            logger.info(f"{self.log_prefix} 使用自定义固定大小")
        image_size = size or model_config.get("default_size", "1024x1024")

        # 验证图片尺寸格式
        if not self._validate_image_size(image_size):
            logger.warning(f"{self.log_prefix} 无效的图片尺寸: {image_size}，使用模型默认值")
            image_size = model_config.get("default_size", "1024x1024")

        # 检查缓存
        is_img2img = input_image_base64 is not None
        if is_img2img:
            cache_key = self._get_img2img_cache_key(description, model_name, image_size, strength)
            cache_dict = self._img2img_cache
        else:
            cache_key = self._get_cache_key(description, model_name, image_size)
            cache_dict = self._request_cache

        if self.get_config("cache.enabled", True) and cache_key in cache_dict:
            cached_result = cache_dict[cache_key]
            logger.info(f"{self.log_prefix} 使用缓存的图片结果")
            await self.send_text("我之前画过类似的图片，用之前的结果~")
            send_success = await self.send_image(cached_result)
            if send_success:
                return True, "图片已发送(缓存)"
            else:
                del cache_dict[cache_key]

        # 显示处理信息
        enable_debug = self.get_config("components.enable_debug_info", False)
        if enable_debug:
            mode_text = "图生图" if is_img2img else "文生图"
            await self.send_text(
                f"收到！正在为您使用 {model_id or '默认'} 模型进行{mode_text}，描述: '{description}'，请稍候...（模型: {model_name}, 尺寸: {image_size}）"
            )

        try:
            # 根据API格式调用不同的请求方法
            if api_format == "doubao":
                success, result = await asyncio.to_thread(
                    self._make_doubao_request,
                    prompt=description,
                    model_config=model_config,
                    size=image_size,
                    input_image_base64=input_image_base64
                )
            elif api_format == "modelscope":
                success, result = await asyncio.to_thread(
                    self._make_modelscope_request,
                    prompt=description,
                    model_config=model_config,
                    input_image_base64=input_image_base64
                )
            elif api_format == "gemini":
                success, result = await asyncio.to_thread(
                    self._make_gemini_request,
                    prompt=description,
                    model_config=model_config,
                    input_image_base64=input_image_base64
                )
            else:  # 默认为openai格式
                success, result = await asyncio.to_thread(
                    self._make_openai_image_request,
                    prompt=description,
                    model_config=model_config,
                    size=image_size,
                    strength=strength,
                    input_image_base64=input_image_base64
                )
        except Exception as e:
            logger.error(f"{self.log_prefix} 异步请求执行失败: {e!r}", exc_info=True)
            traceback.print_exc()
            success = False
            result = f"图片生成服务遇到意外问题: {str(e)[:100]}"

        if success:
            final_image_data = self._process_api_response(result)
            
            if final_image_data:
                if final_image_data.startswith(("iVBORw", "/9j/", "UklGR", "R0lGOD")):  # Base64
                    send_success = await self.send_image(final_image_data)
                    if send_success:
                        mode_text = "图生图" if is_img2img else "文生图"
                        await self.send_text(f"{mode_text}完成！")
                        # 缓存成功的结果
                        cache_dict[cache_key] = final_image_data
                        if is_img2img:
                            self._cleanup_img2img_cache()
                        else:
                            self._cleanup_cache()
                        return True, f"{mode_text}已成功生成并发送"
                    else:
                        await self.send_text("图片已处理完成，但发送失败了")
                        return False, "图片发送失败"
                else:  # URL
                    try:
                        encode_success, encode_result = await asyncio.to_thread(
                            self._download_and_encode_base64, final_image_data
                        )
                        if encode_success:
                            send_success = await self.send_image(encode_result)
                            if send_success:
                                mode_text = "图生图" if is_img2img else "文生图"
                                await self.send_text(f"{mode_text}完成！")
                                # 缓存成功结果
                                cache_dict[cache_key] = encode_result
                                if is_img2img:
                                    self._cleanup_img2img_cache()
                                else:
                                    self._cleanup_cache()
                                return True, f"{mode_text}已完成"
                        else:
                            await self.send_text(f"获取到图片URL，但在处理图片时失败了：{encode_result}")
                            return False, f"图片处理失败: {encode_result}"
                    except Exception as e:
                        logger.error(f"{self.log_prefix} 图片下载编码失败: {e!r}")
                        await self.send_text("图片生成完成但下载时出错")
                        return False, "图片下载失败"
            else:
                await self.send_text("图片生成API返回了无法处理的数据格式")
                return False, "API返回数据格式错误"
        else:
            mode_text = "图生图" if is_img2img else "文生图"
            await self.send_text(f"哎呀，{mode_text}时遇到问题：{result}")
            return False, f"{mode_text}失败: {result}"
  
    # ===== 获取引用的图片 =====
    async def _get_recent_image(self) -> Optional[str]:
        """获取最近的图片消息"""
        try:
            logger.debug(f"{self.log_prefix} 开始获取图片消息")

            # 检查当前Action消息是否包含图片
            if self.has_action_message and self.action_message:
                logger.debug(f"{self.log_prefix} 检查action_message是否包含图片")

                # 1. 检查是否是回复消息，并尝试获取被回复的图片
                if self._is_reply_message():
                    logger.info(f"{self.log_prefix} 检测到回复消息，尝试获取被回复的图片")
                    reply_image = await self._get_image_from_reply()
                    if reply_image:
                        logger.info(f"{self.log_prefix} 从回复消息获取图片成功")
                        return reply_image

                # 2. 检查action_message中的图片信息
                images_data = None
                if isinstance(self.action_message, dict):
                    if "images" in self.action_message and self.action_message["images"]:
                        images_data = self.action_message["images"][0]
                else:
                    # DatabaseMessages 对象
                    if hasattr(self.action_message, 'images') and getattr(self.action_message, 'images', None):
                        images_list = getattr(self.action_message, 'images')
                        if images_list:
                            images_data = images_list[0] if isinstance(images_list, list) else images_list

                if images_data:
                    logger.info(f"{self.log_prefix} 从action_message获取图片")
                    return self._process_image_data(images_data)

                # 3. 检查message_content中的图片
                message_content = None
                if isinstance(self.action_message, dict):
                    if "message_content" in self.action_message:
                        message_content = self.action_message["message_content"]
                else:
                    # DatabaseMessages 对象
                    if hasattr(self.action_message, 'message_content'):
                        message_content = getattr(self.action_message, 'message_content', None)

                if message_content:
                    if isinstance(message_content, str) and self._is_image_data(message_content):
                        logger.info(f"{self.log_prefix} 从message_content获取图片")
                        return self._process_image_data(message_content)

            # 尝试从chat_stream获取最近的图片消息
            if self.chat_stream:
                logger.debug(f"{self.log_prefix} 尝试从chat_stream获取历史图片消息")

                try:
                    # 获取最近的消息历史
                    if hasattr(self.chat_stream, 'get_recent_messages'):
                        recent_messages = self.chat_stream.get_recent_messages(10)
                        logger.debug(f"{self.log_prefix} 获取到 {len(recent_messages)} 条历史消息")

                        for msg in reversed(recent_messages):
                            image_data = await self._extract_image_from_message(msg)
                            if image_data:
                                logger.info(f"{self.log_prefix} 从历史消息获取图片")
                                return image_data

                    # 尝试从消息存储获取
                    if hasattr(self.chat_stream, 'message_storage'):
                        storage = self.chat_stream.message_storage
                        if hasattr(storage, 'get_recent_messages'):
                            recent_messages = storage.get_recent_messages(10)
                            logger.debug(f"{self.log_prefix} 从存储获取到 {len(recent_messages)} 条消息")

                            for msg in reversed(recent_messages):
                                image_data = await self._extract_image_from_message(msg)
                                if image_data:
                                    logger.info(f"{self.log_prefix} 从存储消息获取图片")
                                    return image_data

                except Exception as e:
                    logger.debug(f"{self.log_prefix} 从chat_stream获取历史消息失败: {e}")

            # 最后尝试：使用插件系统的消息API
            try:
                from src.plugin_system.apis import message_api
                # 使用正确的API获取最近消息
                recent_messages = message_api.get_recent_messages(self.chat_id, hours=1.0, limit=20, filter_mai=True)
                logger.debug(f"{self.log_prefix} 从message_api获取到 {len(recent_messages)} 条消息")

                for msg in reversed(recent_messages):
                    image_data = await self._extract_image_from_message(msg)
                    if image_data:
                        logger.info(f"{self.log_prefix} 从message_api获取图片")
                        return image_data

            except Exception as e:
                logger.debug(f"{self.log_prefix} 使用message_api获取消息失败: {e}")

            logger.warning(f"{self.log_prefix} 未找到可用的图片消息")
            return None

        except Exception as e:
            logger.error(f"{self.log_prefix} 获取图片失败: {e!r}", exc_info=True)
            return None

    def _is_reply_message(self) -> bool:
        """检测当前消息是否是回复消息"""
        try:
            if not self.action_message:
                return False

            # 检查多种可能的回复消息字段
            potential_fields = [
                'raw_message', 'processed_plain_text', 'display_message',
                'message_content', 'content', 'text'
            ]

            if isinstance(self.action_message, dict):
                # 字典类型的action_message
                for field in potential_fields:
                    if field in self.action_message:
                        text = str(self.action_message[field])
                        # 检查是否包含回复格式的文本
                        if text and ('[回复' in text or 'reply' in text.lower() or '回复' in text):
                            logger.debug(f"{self.log_prefix} 在字段 {field} 中检测到回复消息格式")
                            return True

                # 检查是否有reply相关的字段
                reply_fields = ['reply_to', 'reply_message', 'quoted_message', 'reply']
                for field in reply_fields:
                    if field in self.action_message and self.action_message[field]:
                        logger.debug(f"{self.log_prefix} 检测到回复字段: {field}")
                        return True
            else:
                # DatabaseMessages 对象
                for field in potential_fields:
                    if hasattr(self.action_message, field):
                        text = str(getattr(self.action_message, field, ''))
                        # 检查是否包含回复格式的文本
                        if text and ('[回复' in text or 'reply' in text.lower() or '回复' in text):
                            logger.debug(f"{self.log_prefix} 在属性 {field} 中检测到回复消息格式")
                            return True

                # 检查是否有reply相关的属性
                reply_fields = ['reply_to', 'reply_message', 'quoted_message', 'reply']
                for field in reply_fields:
                    if hasattr(self.action_message, field) and getattr(self.action_message, field, None):
                        logger.debug(f"{self.log_prefix} 检测到回复属性: {field}")
                        return True

            return False

        except Exception as e:
            logger.debug(f"{self.log_prefix} 检测回复消息失败: {e}")
            return False

    async def _get_image_from_reply(self) -> Optional[str]:
        """从回复消息中获取被回复的图片"""
        try:
            if not self.action_message:
                return None

            # 1. 处理reply_to字段 - 这是最重要的
            reply_to = None
            if isinstance(self.action_message, dict):
                if 'reply_to' in self.action_message and self.action_message['reply_to']:
                    reply_to = self.action_message['reply_to']
            else:
                # DatabaseMessages 对象
                if hasattr(self.action_message, 'reply_to') and getattr(self.action_message, 'reply_to', None):
                    reply_to = getattr(self.action_message, 'reply_to')

            if reply_to:
                logger.info(f"{self.log_prefix} 发现reply_to字段: {reply_to}")

                # 尝试通过消息ID直接查询被回复的消息
                reply_message = await self._get_message_by_id(reply_to)
                if reply_message:
                    logger.info(f"{self.log_prefix} 通过ID获取到被回复的消息")
                    # 检查是否是图片消息
                    is_picid = False
                    if isinstance(reply_message, dict):
                        is_picid = reply_message.get('is_picid', False)
                    else:
                        is_picid = getattr(reply_message, 'is_picid', False)

                    if is_picid:
                        image_data = await self._extract_image_from_message(reply_message)
                        if image_data:
                            logger.info(f"{self.log_prefix} 从reply_to消息获取图片成功")
                            return image_data

                # 如果直接查询失败，在历史消息中搜索
                try:
                    from src.plugin_system.apis import message_api
                    # 获取更多历史消息来查找被回复的消息
                    recent_messages = message_api.get_recent_messages(self.chat_id, hours=2.0, limit=50, filter_mai=True)
                    logger.debug(f"{self.log_prefix} 获取 {len(recent_messages)} 条消息查找reply_to: {reply_to}")

                    for msg in recent_messages:
                        # 检查消息ID匹配
                        msg_id = None
                        is_picid = False

                        if isinstance(msg, dict):
                            msg_id = msg.get('message_id') or msg.get('id')
                            is_picid = msg.get('is_picid', False)
                        else:
                            # DatabaseMessages 对象
                            msg_id = getattr(msg, 'message_id', None) or getattr(msg, 'id', None)
                            is_picid = getattr(msg, 'is_picid', False)

                        if str(msg_id) == str(reply_to):
                            logger.info(f"{self.log_prefix} 在历史消息中找到被回复的消息: {msg_id}")
                            # 检查这条消息是否包含图片
                            if is_picid:
                                image_data = await self._extract_image_from_message(msg)
                                if image_data:
                                    logger.info(f"{self.log_prefix} 从reply_to消息获取图片成功")
                                    return image_data

                except Exception as e:
                    logger.debug(f"{self.log_prefix} 通过reply_to查找消息失败: {e}")

            # 2. 尝试从回复相关字段直接获取
            reply_fields = ['reply_message', 'quoted_message', 'reply']
            if isinstance(self.action_message, dict):
                for field in reply_fields:
                    if field in self.action_message and self.action_message[field]:
                        reply_data = self.action_message[field]
                        image_data = await self._extract_image_from_message(reply_data)
                        if image_data:
                            logger.info(f"{self.log_prefix} 从{field}字段获取回复图片")
                            return image_data
            else:
                # DatabaseMessages 对象
                for field in reply_fields:
                    if hasattr(self.action_message, field) and getattr(self.action_message, field, None):
                        reply_data = getattr(self.action_message, field)
                        image_data = await self._extract_image_from_message(reply_data)
                        if image_data:
                            logger.info(f"{self.log_prefix} 从{field}属性获取回复图片")
                            return image_data

            # 3. 解析回复格式的文本消息，提取被回复消息的ID或信息
            text_fields = ['processed_plain_text', 'display_message', 'raw_message', 'message_content']
            if isinstance(self.action_message, dict):
                for field in text_fields:
                    if field in self.action_message:
                        text = str(self.action_message[field])
                        if '[回复' in text and '[图片]' in text:
                            logger.debug(f"{self.log_prefix} 在{field}中发现回复图片格式: {text[:100]}...")

                            # 尝试从文本中提取图片相关信息
                            image_data = await self._extract_base64_from_text(text)
                            if image_data:
                                logger.info(f"{self.log_prefix} 从回复文本中提取图片成功")
                                return image_data
            else:
                # DatabaseMessages 对象
                for field in text_fields:
                    if hasattr(self.action_message, field):
                        text = str(getattr(self.action_message, field, ''))
                        if '[回复' in text and '[图片]' in text:
                            logger.debug(f"{self.log_prefix} 在{field}属性中发现回复图片格式: {text[:100]}...")

                            # 尝试从文本中提取图片相关信息
                            image_data = await self._extract_base64_from_text(text)
                            if image_data:
                                logger.info(f"{self.log_prefix} 从回复文本中提取图片成功")
                                return image_data

            # 4. 作为备选方案，查找最近的图片消息（扩大搜索范围）
            try:
                from src.plugin_system.apis import message_api
                # 扩大搜索范围到100条消息，2小时内
                recent_messages = message_api.get_recent_messages(self.chat_id, hours=2.0, limit=100, filter_mai=True)
                logger.debug(f"{self.log_prefix} 扩大搜索范围，获取最近 {len(recent_messages)} 条消息查找图片")

                for msg in reversed(recent_messages):
                    # 跳过当前消息
                    current_msg_id = None
                    msg_id = None
                    is_picid = False

                    if hasattr(self.action_message, 'get'):
                        current_msg_id = self.action_message.get('message_id') or self.action_message.get('id')
                    else:
                        current_msg_id = getattr(self.action_message, 'message_id', None) or getattr(self.action_message, 'id', None)

                    if isinstance(msg, dict):
                        msg_id = msg.get('message_id') or msg.get('id')
                        is_picid = msg.get('is_picid', False)
                    else:
                        # DatabaseMessages 对象
                        msg_id = getattr(msg, 'message_id', None) or getattr(msg, 'id', None)
                        is_picid = getattr(msg, 'is_picid', False)

                    if str(msg_id) == str(current_msg_id):
                        continue

                    # 查找图片消息
                    if is_picid:
                        image_data = await self._extract_image_from_message(msg)
                        if image_data:
                            logger.info(f"{self.log_prefix} 从扩大范围的历史消息中找到图片")
                            return image_data

            except Exception as e:
                logger.debug(f"{self.log_prefix} 扩大范围查找图片消息失败: {e}")

            return None

        except Exception as e:
            logger.error(f"{self.log_prefix} 从回复消息获取图片失败: {e!r}")
            return None

    async def _get_message_by_id(self, message_id: str) -> Optional[dict]:
        """通过消息ID直接查询消息"""
        try:
            # 尝试使用数据库直接查询
            from src.common.database.database_model import Messages

            try:
                # 查询消息记录
                message_record = Messages.select().where(Messages.id == message_id).first()
                if message_record:
                    logger.info(f"{self.log_prefix} 通过数据库查询到消息: {message_id}")
                    # 将消息记录转换为字典格式
                    message_dict = {
                        'id': message_record.id,
                        'message_id': message_record.id,
                        'is_picid': getattr(message_record, 'is_picid', False),
                        'processed_plain_text': getattr(message_record, 'processed_plain_text', ''),
                        'display_message': getattr(message_record, 'display_message', ''),
                        'additional_config': getattr(message_record, 'additional_config', ''),
                        'raw_message': getattr(message_record, 'raw_message', ''),
                    }
                    return message_dict
            except Exception as e:
                logger.debug(f"{self.log_prefix} 数据库查询消息失败: {e}")

            # 如果数据库查询失败，尝试其他方式
            logger.debug(f"{self.log_prefix} 无法通过ID直接查询消息: {message_id}")
            return None

        except Exception as e:
            logger.debug(f"{self.log_prefix} 查询消息ID {message_id} 失败: {e}")
            return None

    async def _extract_image_from_message(self, message) -> Optional[str]:
        """从消息对象中提取图片数据"""
        try:
            if not message:
                return None

            # 检查消息是否包含图片标记
            if isinstance(message, dict):
                # 优先检查is_picid标记
                if message.get('is_picid', False):
                    # 查找图片相关的字段
                    potential_keys = [
                        'message_segment', 'raw_message', 'display_message',
                        'processed_plain_text', 'additional_config'
                    ]

                    for key in potential_keys:
                        if key in message and message[key]:
                            image_data = await self._extract_base64_from_text(str(message[key]))
                            if image_data:
                                logger.debug(f"{self.log_prefix} 从{key}字段提取到图片数据")
                                return image_data

                # 通用字段检查
                for key in ['images', 'image', 'content', 'message_content', 'data']:
                    if key in message and message[key]:
                        data = message[key]
                        if isinstance(data, list) and data:
                            data = data[0]
                        image_data = self._process_image_data(data)
                        if image_data:
                            return image_data

            # 如果是消息对象（DatabaseMessages）
            else:
                # 检查是否有图片标记
                if hasattr(message, 'is_picid') and getattr(message, 'is_picid', False):
                    # 尝试从消息段中获取图片
                    if hasattr(message, 'message_segment') and message.message_segment:
                        segment = message.message_segment
                        image_data = self._extract_image_from_segment(segment)
                        if image_data:
                            return image_data

                    # 从其他属性获取
                    for attr in ['raw_message', 'processed_plain_text', 'display_message', 'additional_config']:
                        if hasattr(message, attr):
                            text = getattr(message, attr)
                            if text:
                                image_data = await self._extract_base64_from_text(str(text))
                                if image_data:
                                    logger.debug(f"{self.log_prefix} 从{attr}属性提取到图片数据")
                                    return image_data

                # 尝试多种方式获取图片
                image_sources = [
                    getattr(message, 'images', None),
                    getattr(message, 'image', None),
                    getattr(message, 'content', None),
                    getattr(message, 'message_content', None),
                    getattr(message, 'data', None),
                ]

                for source in image_sources:
                    if source:
                        if isinstance(source, list) and source:
                            # 如果是列表，取第一个
                            image_data = self._process_image_data(source[0])
                            if image_data:
                                return image_data
                        else:
                            # 如果是单个数据
                            image_data = self._process_image_data(source)
                            if image_data:
                                return image_data

            return None

        except Exception as e:
            logger.debug(f"{self.log_prefix} 从消息提取图片失败: {e}")
            return None

    def _extract_image_from_segment(self, segment) -> Optional[str]:
        """从消息段中提取图片"""
        try:
            if not segment:
                return None

            # 如果是字典格式的段
            if isinstance(segment, dict):
                if segment.get('type') == 'image' and 'data' in segment:
                    return self._process_image_data(segment['data'])

            # 如果有data属性
            elif hasattr(segment, 'data'):
                segment_data = getattr(segment, 'data')
                if segment_data:
                    return self._process_image_data(segment_data)

            # 如果有type属性
            elif hasattr(segment, 'type'):
                segment_type = getattr(segment, 'type')
                if segment_type == 'image' and hasattr(segment, 'data'):
                    return self._process_image_data(getattr(segment, 'data'))

            return None

        except Exception as e:
            logger.debug(f"{self.log_prefix} 从消息段提取图片失败: {e}")
            return None

    async def _extract_base64_from_text(self, text: str) -> Optional[str]:
        """从文本中提取base64图片数据"""
        try:
            if not text:
                return None

            # 检查是否直接是base64图片数据
            if self._is_image_data(text):
                return self._process_image_data(text)

            # 检查是否包含picid格式 [picid:xxxxx]
            import re
            picid_pattern = r'\[picid:([a-f0-9\-]+)\]'
            picid_match = re.search(picid_pattern, text)
            if picid_match:
                picid = picid_match.group(1)
                logger.info(f"{self.log_prefix} 找到picid: {picid}")

                # 尝试通过picid获取图片数据
                image_data = await self._get_image_by_picid(picid)
                if image_data:
                    return image_data

            # 尝试从可能的JSON格式中提取
            import json
            try:
                data = json.loads(text)
                if isinstance(data, dict):
                    for key in ['data', 'base64', 'image', 'content']:
                        if key in data and data[key]:
                            result = self._process_image_data(data[key])
                            if result:
                                return result
                elif isinstance(data, list) and data:
                    for item in data:
                        result = await self._extract_base64_from_text(str(item))
                        if result:
                            return result
            except (json.JSONDecodeError, TypeError):
                pass

            return None

        except Exception as e:
            logger.debug(f"{self.log_prefix} 从文本提取base64失败: {e}")
            return None

    async def _get_image_by_picid(self, picid: str) -> Optional[str]:
        """通过picid获取图片的base64数据"""
        try:
            # 尝试从图片管理器获取图片
            from src.chat.utils.utils_image import get_image_manager

            image_manager = get_image_manager()
            if hasattr(image_manager, 'get_image_by_id'):
                image_data = await image_manager.get_image_by_id(picid)
                if image_data:
                    return image_data

            # 尝试从数据库直接获取
            from src.common.database.database_model import Images
            try:
                # 查找对应的图片记录
                image_record = Images.select().where(Images.id == picid).first()
                if image_record and image_record.path:
                    # 从路径读取图片并转换为base64
                    from src.chat.utils.utils_image import image_path_to_base64
                    base64_data = image_path_to_base64(image_record.path)
                    if base64_data:
                        logger.info(f"{self.log_prefix} 通过picid从数据库获取图片成功")
                        return base64_data
            except Exception as e:
                logger.debug(f"{self.log_prefix} 从数据库获取图片失败: {e}")

            # 如果上述方法都失败，尝试构造路径
            # MaiBot可能将图片存储在特定目录
            import os
            possible_paths = [
                f"/tmp/images/{picid}",
                f"/tmp/images/{picid}.jpg",
                f"/tmp/images/{picid}.png",
                f"data/images/{picid}",
                f"data/images/{picid}.jpg",
                f"data/images/{picid}.png",
                f"images/{picid}",
                f"images/{picid}.jpg",
                f"images/{picid}.png"
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    from src.chat.utils.utils_image import image_path_to_base64
                    base64_data = image_path_to_base64(path)
                    if base64_data:
                        logger.info(f"{self.log_prefix} 通过路径 {path} 获取图片成功")
                        return base64_data

            logger.warning(f"{self.log_prefix} 无法通过picid {picid} 获取图片数据")
            return None

        except Exception as e:
            logger.error(f"{self.log_prefix} 通过picid获取图片异常: {e!r}")
            return None

    def _process_image_data(self, data) -> Optional[str]:
        """处理图片数据，统一转换为base64格式"""
        try:
            if not data:
                return None

            if isinstance(data, str):
                if self._is_image_data(data):
                    if data.startswith('data:image'):
                        base64_data = data.split(',', 1)[1] if ',' in data else data
                        logger.debug(f"{self.log_prefix} 提取data URL中的base64数据，长度: {len(base64_data)}")
                        return base64_data
                    elif data.startswith(('iVBORw', '/9j/', 'UklGR', 'R0lGOD')):
                        logger.debug(f"{self.log_prefix} 获取到base64图片数据，长度: {len(data)}")
                        return data

            elif isinstance(data, dict):
                for key in ['data', 'base64', 'content', 'url']:
                    if key in data and data[key]:
                        result = self._process_image_data(data[key])
                        if result:
                            return result

            return None

        except Exception as e:
            logger.debug(f"{self.log_prefix} 处理图片数据失败: {e}")
            return None

    def _is_image_data(self, data: str) -> bool:
        """检查字符串是否包含图片数据"""
        if not isinstance(data, str):
            return False
        return data.startswith(('data:image', 'iVBORw', '/9j/', 'UklGR', 'R0lGOD'))

    # ===== API请求方法 =====
    def _make_doubao_request(self, prompt: str, model_config: Dict[str, Any], size: str, input_image_base64: str = None) -> Tuple[bool, str]:
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

        endpoint = f"{base_url.rstrip('/')}/images/generations"

        # 获取模型特定的配置参数
        custom_prompt_add = model_config.get("custom_prompt_add", "")
        negative_prompt_add = model_config.get("negative_prompt_add", "")
        seed = model_config.get("seed", 42)
        guidance_scale = model_config.get("guidance_scale", 2.5)
        watermark = model_config.get("watermark", True)

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

    def _make_modelscope_request(self, prompt: str, model_config: Dict[str, Any], size: str = None, strength: float = None, input_image_base64: str = None) -> Tuple[bool, str]:
        """发送魔搭格式的HTTP请求生成图片"""
        try:
            import requests
            import json
            import time
        
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
        
            # 构建请求数据
            request_data = {
                "model": model_name,
                "prompt": prompt
            }
        
            # 如果有输入图片，需要特殊处理
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

                request_data["image"] = image_data_uri
                logger.info(f"{self.log_prefix} (魔搭) 使用图生图模式，图片格式: {image_data_uri[:50]}...")
            else:
                logger.info(f"{self.log_prefix} (魔搭) 使用文生图模式")

            logger.info(f"{self.log_prefix} (魔搭) 发起异步图片生成请求，模型: {model_name}")

            # 发送异步请求
            response = requests.post(
                f"{base_url}/v1/images/generations",
                headers=headers,
                data=json.dumps(request_data, ensure_ascii=False).encode('utf-8'),
                timeout=30
            )
        
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
                "Content-Type": "application/json"
            }
        
            max_attempts = 24  # 最多检查2分钟
            for attempt in range(max_attempts):
                try:
                    check_response = requests.get(
                        f"{base_url}/v1/tasks/{task_id}",
                        headers=check_headers,
                        timeout=10
                    )
                
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
                                img_response = requests.get(image_url, timeout=30)
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
            import requests
            import json
        
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
        
            # 构建请求体 - 包含必需的 responseModalities
            request_data = {
                "contents": [{
                    "parts": parts
                }],
                "generationConfig": {
                    "responseModalities": ["TEXT", "IMAGE"]  # 关键配置
                }
            }
        
            logger.info(f"{self.log_prefix} (Gemini) 发起图片请求: {model_name}")
        
            # 发送请求
            response = requests.post(
                url=url,
                headers=headers,
                json=request_data,
                timeout=120
            )
        
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
                            elif "inline_data" in part and "data" in part["inline_data"]:  # 兼容两种命名
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

    # ===== 辅助方法 =====
    def _convert_base64_to_url_if_needed(self, base64_data: str) -> str:
        """如果需要，将base64数据转换为临时URL"""
        # 对于某些API，可能需要将base64转换为可访问的URL
        # 这里简化处理，直接返回base64数据
        # 在实际实现中，可能需要上传到临时存储并返回URL
        return base64_data

    def _process_api_response(self, result) -> Optional[str]:
        """统一处理API响应，提取图片数据"""
        try:
            # 如果result是字符串，直接返回
            if isinstance(result, str):
                return result
            
            # 如果result是字典，尝试提取图片数据
            if isinstance(result, dict):
                # 尝试多种可能的字段
                for key in ['url', 'image', 'b64_json', 'data']:
                    if key in result and result[key]:
                        return result[key]
                        
                # 检查嵌套结构
                if 'output' in result and isinstance(result['output'], dict):
                    output = result['output']
                    for key in ['image_url', 'images']:
                        if key in output:
                            data = output[key]
                            return data[0] if isinstance(data, list) and data else data
            
            return None
        except Exception as e:
            logger.error(f"{self.log_prefix} 处理API响应失败: {e!r}")
            return None    
   
    # ===== 获取模型配置 =====
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

    # ===== 下载图片并将其编码 =====
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
    
    # ===== 缓存管理 =====
    @classmethod
    def _get_cache_key(cls, description: str, model: str, size: str) -> str:
        """生成文生图缓存键"""
        return f"txt2img_{description[:100]}|{model}|{size}"

    @classmethod
    def _get_img2img_cache_key(cls, description: str, model: str, size: str, strength: float) -> str:
        """生成图生图缓存键"""
        return f"img2img_{description[:50]}|{model}|{size}|{strength}"

    @classmethod
    def _cleanup_cache(cls):
        """清理文生图缓存"""
        if len(cls._request_cache) > cls._cache_max_size:
            keys_to_remove = list(cls._request_cache.keys())[: -cls._cache_max_size // 2]
            for key in keys_to_remove:
                del cls._request_cache[key]

    @classmethod
    def _cleanup_img2img_cache(cls):
        """清理图生图缓存"""
        if len(cls._img2img_cache) > cls._img2img_cache_max_size:
            keys_to_remove = list(cls._img2img_cache.keys())[: -cls._img2img_cache_max_size // 2]
            for key in keys_to_remove:
                del cls._img2img_cache[key]

    def _validate_image_size(self, image_size: str) -> bool:
        """验证图片尺寸格式"""
        try:
            width, height = map(int, image_size.split("x"))
            return 100 <= width <= 10000 and 100 <= height <= 10000
        except (ValueError, TypeError):
            return False

# ===== 插件注册 =====
@register_plugin
class CustomPicPlugin(BasePlugin):
    """统一的多模型图片生成插件，支持文生图和图生图"""
    
    # 插件基本信息
    plugin_name = "custom_pic_plugin"  # 插件唯一标识符
    plugin_version = "3.1.2"  # 插件版本号
    plugin_author = "Ptrel"  # 插件作者
    enable_plugin = True  # 是否启用插件
    dependencies: List[str] = []  # 插件依赖列表
    python_dependencies: List[str] = []  # Python包依赖列表
    config_file_name = "config.toml"

    # 配置节描述
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
            "name": ConfigField(type=str, default="custom_pic_plugin", description="自定义多模型统一图片生成插件", required=True),
            "config_version": ConfigField(type=str, default="3.1.2", description="插件版本号"),
            "enabled": ConfigField(type=bool, default=False, description="是否启用插件")
        },
        "generation": {
            "default_model": ConfigField(
                type=str,
                default="model1",
                description="默认使用的模型ID。支持文生图和图生图自动切换,可以在配置文件中添加更多模型配置",
                choices=["model1"]
            ),
        },
        "cache": {
            "enabled": ConfigField(type=bool, default=True, description="是否启用请求缓存"),
            "max_size": ConfigField(type=int, default=10, description="最大缓存数量"),
        },
        "components": {
            "enable_unified_generation": ConfigField(type=bool, default=True, description="是否启用统一图片生成Action"),
            "enable_debug_info": ConfigField(type=bool, default=False, description="是否启用调试信息显示，开启后会在聊天中显示生图参数")
        },
        "logging": {
            "level": ConfigField(type=str, default="INFO", description="日志记录级别", choices=["DEBUG", "INFO", "WARNING", "ERROR"]),
            "prefix": ConfigField(type=str, default="[unified_pic_Plugin]", description="日志记录前缀")
        },
        "models": {},
        # 基础模型配置
        "models.model1": {
            "name": ConfigField(type=str, default="魔搭潦草模型", description="模型显示名称"),
            "base_url": ConfigField(
                type=str,
                default="https://api-inference.modelscope.cn/v1",
                description="API基础URL。其他服务商URL示例: 豆包=https://ark.cn-beijing.volces.com/api/v3, 配置新模型：复制models.model1整个配置块，重命名为models.你的名称，修改相应参数",
                required=True
            ),
            "api_key": ConfigField(
                type=str,
                default="Bearer xxxxxxxxxxxxxxxxxxxxxx",
                description="API密钥。不同服务的密钥格式: OpenAI格式(魔搭/硅基流动)需要'Bearer '前缀, 豆包格式不需要Bearer前缀, Gemini可在URL中包含或单独配置",
                required=True
            ),
            "format": ConfigField(
                type=str,
                default="openai",
                description="API请求格式。支持的格式: openai(通用格式，适用于魔搭、硅基流动、NewAPI等), doubao(豆包专用格式), gemini(Google Gemini专用格式)",
                choices=["openai", "gemini", "doubao"]
            ),
            "model": ConfigField(
                type=str, 
                default="cancel13/liaocao", 
                description="具体的模型名称。不同服务的模型名示例: 魔搭=cancel13/liaocao, 豆包=doubao-seedream-4-0-250828, 硅基流动=Qwen/Qwen-Image, Gemini=gemini-2.5-flash-image-preview"
            ),
            "fixed_size_enabled": ConfigField(
                type=bool, 
                default=False, 
                description="是否启用固定图片大小。启用后只会使用default_size设定的尺寸，否则会由麦麦自己选择。"
            ),
            "default_size": ConfigField(
                type=str,  
                default="1024x1024",  
                description="默认图片尺寸, 部分模型可能有特定的尺寸要求",
                choices=["512x512", "1024x1024", "1024x1280", "1280x1024", "1024x1536", "1536x1024"]
            ),
            "seed": ConfigField(type=int, default=42, description="随机种子"),
            "guidance_scale": ConfigField(type=float, default=2.5, description="模型指导强度。豆包推荐5.5，其他服务推荐2.5。数值越高越严格按照提示词生成"),
            "watermark": ConfigField(type=bool, default=True, description="是否添加水印。豆包默认支持，其他服务根据情况设置"),
            "custom_prompt_add": ConfigField(
                type=str,
                default=", Nordic picture book art style, minimalist flat design, liaocao",
                description="正面附加提示词，用于增强画风效果。"
            ),
            "negative_prompt_add": ConfigField(
                type=str,
                default="Pornography,nudity,lowres, bad anatomy, bad hands, text, error",
                description="负面附加提示词，保持默认或使用豆包时可留空，留空时保持两个英文双引号，否则会报错。"
            ),
            "support_img2img": ConfigField(type=bool, default=True, description="是否支持图生图功能。大多数现代模型都支持基于现有图片进行修改"),
            "num_inference_steps": ConfigField(type=int, default=20, description="推理步数，影响图片质量和生成速度。通常20-50之间"),
        }
    }

    def get_plugin_components(self) -> List[Tuple[ComponentInfo, Type]]:
        """返回插件包含的组件列表"""
        enable_unified_generation = self.get_config("components.enable_unified_generation", True)
        components = []

        if enable_unified_generation:
            components.append((Custom_Pic_Action.get_action_info(), Custom_Pic_Action))

        return components