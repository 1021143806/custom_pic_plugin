import asyncio
import traceback
import base64
import os
from typing import List, Tuple, Type, Optional, Dict, Any

from src.plugin_system.base.base_action import BaseAction
from src.plugin_system.base.component_types import ActionActivationType, ChatMode
from src.common.logger import get_logger

from .api_clients import get_client_class
from .image_utils import ImageProcessor
from .cache_manager import CacheManager
from .size_utils import validate_image_size, get_image_size
from .runtime_state import runtime_state
from .prompt_optimizer import optimize_prompt

logger = get_logger("pic_action")

class Custom_Pic_Action(BaseAction):
    """统一的图片生成动作，智能检测文生图或图生图"""

    # 激活设置
    activation_type = ActionActivationType.ALWAYS  # 默认激活类型
    focus_activation_type = ActionActivationType.ALWAYS  # Focus模式使用LLM判定，精确理解需求
    normal_activation_type = ActionActivationType.KEYWORD  # Normal模式使用关键词激活，快速响应
    mode_enable = ChatMode.ALL
    parallel_action = True

    # 动作基本信息
    action_name = "draw_picture"
    action_description = (
        "智能图片生成：根据描述生成图片（文生图）或基于现有图片进行修改（图生图）。"
        "自动检测用户是否提供了输入图片来决定使用文生图还是图生图模式。"
        "支持多种API格式：OpenAI、豆包、Gemini、硅基流动、魔搭社区、砂糖云(NovelAI)、ComfyUI、梦羽AI等。"
    )

    # 关键词设置（用于Normal模式）
    activation_keywords = [
        # 文生图关键词
        "画", "绘制", "生成图片", "画图", "draw", "paint", "图片生成", "创作",
        # 图生图关键词
        "图生图", "修改图片", "基于这张图", "img2img", "重画", "改图", "图片修改",
        "改成", "换成", "变成", "转换成", "风格", "画风", "改风格", "换风格",
        "这张图", "这个图", "图片风格", "改画风", "重新画", "再画", "重做",
        # 自拍关键词
        "自拍", "selfie", "拍照", "对镜自拍", "镜子自拍", "照镜子"
    ]

    # LLM判定提示词（用于Focus模式）
    ALWAYS_prompt = """
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

**自拍场景：**
1. 用户明确要求你进行自拍、拍照等
2. 用户提到"自拍"、"selfie"、"照镜子"、"对镜自拍"等关键词
3. 用户想要看到你的照片或形象

**绝对不要使用的情况：**
1. 纯文字聊天和问答
2. 只是提到"图片"、"画"等词但不是要求生成
3. 谈论已存在的图片或照片（仅讨论不修改）
4. 技术讨论中提到绘图概念但无生成需求
5. 用户明确表示不需要图片时
6. 刚刚成功生成过图片，避免频繁请求
"""

    keyword_case_sensitive = False

    # 动作参数定义（简化版，提示词优化由独立模块处理）
    action_parameters = {
        "description": "从用户消息中提取的图片描述文本（例如：用户说'画一只小猫'，则填写'一只小猫'）。必填参数。",
        "model_id": "要使用的模型ID（如model1、model2、model3等，默认使用default_model配置的模型）",
        "strength": "图生图强度，0.1-1.0之间，值越高变化越大（仅图生图时使用，可选，默认0.7）",
        "size": "图片尺寸，如512x512、1024x1024等（可选，不指定则使用模型默认尺寸）",
        "selfie_mode": "是否启用自拍模式（true/false，可选，默认false）。启用后会自动添加自拍场景和手部动作",
        "selfie_style": "自拍风格，可选值：standard（标准自拍，适用于户外或无镜子场景），mirror（对镜自拍，适用于有镜子的室内场景）。仅在selfie_mode=true时生效，可选，默认standard",
        "free_hand_action": "自由手部动作描述（英文）。如果指定此参数，将使用此动作而不是随机生成。仅在selfie_mode=true时生效，可选"
    }

    # 动作使用场景
    action_require = [
        "当用户要求生成或修改图片时使用，不要频率太高",
        "自动检测是否有输入图片来决定文生图或图生图模式",
        "重点：不要连续发，如果你在前10句内已经发送过[图片]或者[表情包]或记录出现过类似描述的[图片]，就不要选择此动作",
        "支持指定模型：用户可以通过'用模型1画'、'model2生成'等方式指定特定模型"
    ]
    associated_types = ["text", "image"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_processor = ImageProcessor(self)
        self.cache_manager = CacheManager(self)
        self._api_clients = {}  # 缓存不同格式的API客户端

    def _get_api_client(self, api_format: str):
        """获取指定格式的API客户端（带缓存）"""
        if api_format not in self._api_clients:
            client_class = get_client_class(api_format)
            self._api_clients[api_format] = client_class(self)
        return self._api_clients[api_format]

    async def execute(self) -> Tuple[bool, Optional[str]]:
        """执行统一图片生成动作"""
        logger.info(f"{self.log_prefix} 执行统一图片生成动作")

        # 检查是否是 /dr 命令消息，如果是则跳过（由 Command 组件处理）
        if self.action_message and self.action_message.processed_plain_text:
            message_text = self.action_message.processed_plain_text.strip()
            if message_text.startswith("/dr ") or message_text == "/dr":
                logger.info(f"{self.log_prefix} 检测到 /dr 命令，跳过 Action 处理（由 Command 组件处理）")
                return False, "跳过 /dr 命令"

        # 检查插件是否在当前聊天流启用
        global_enabled = self.get_config("plugin.enabled", True)
        if not runtime_state.is_plugin_enabled(self.chat_id, global_enabled):
            logger.info(f"{self.log_prefix} 插件在当前聊天流已禁用")
            return False, "插件已禁用"

        # 获取参数
        description = self.action_data.get("description", "").strip()
        model_id = self.action_data.get("model_id", "").strip()
        strength = self.action_data.get("strength", 0.7)
        size = self.action_data.get("size", "").strip()
        selfie_mode = self.action_data.get("selfie_mode", False)
        selfie_style = self.action_data.get("selfie_style", "standard").strip().lower()
        free_hand_action = self.action_data.get("free_hand_action", "").strip()

        # 如果没有指定模型，使用运行时状态的默认模型
        if not model_id:
            global_default = self.get_config("generation.default_model", "model1")
            model_id = runtime_state.get_action_default_model(self.chat_id, global_default)

        # 检查模型是否在当前聊天流启用
        if not runtime_state.is_model_enabled(self.chat_id, model_id):
            logger.warning(f"{self.log_prefix} 模型 {model_id} 在当前聊天流已禁用")
            await self.send_text(f"模型 {model_id} 当前不可用")
            return False, f"模型 {model_id} 已禁用"

        # 参数验证和后备提取
        if not description:
            # 尝试从action_message中提取描述
            extracted_description = self._extract_description_from_message()
            if extracted_description:
                description = extracted_description
                logger.info(f"{self.log_prefix} 从消息中提取到图片描述: {description}")
            else:
                logger.warning(f"{self.log_prefix} 图片描述为空，无法生成图片。")
                await self.send_text("你需要告诉我想要画什么样的图片哦~ 比如说'画一只可爱的小猫'")
                return False, "图片描述为空"

        # 清理和验证描述
        if len(description) > 1000:
            description = description[:1000]
            logger.info(f"{self.log_prefix} 图片描述过长，已截断至1000字符")

        # 提示词优化
        optimizer_enabled = self.get_config("prompt_optimizer.enabled", True)
        if optimizer_enabled:
            logger.info(f"{self.log_prefix} 开始优化提示词: {description[:50]}...")
            success, optimized_prompt = await optimize_prompt(description, self.log_prefix)
            if success:
                logger.info(f"{self.log_prefix} 提示词优化完成: {optimized_prompt[:80]}...")
                description = optimized_prompt
            else:
                logger.warning(f"{self.log_prefix} 提示词优化失败，使用原始描述: {description[:50]}...")

        # 验证strength参数
        try:
            strength = float(strength)
            if not (0.1 <= strength <= 1.0):
                strength = 0.7
        except (ValueError, TypeError):
            strength = 0.7

        # 处理自拍模式
        if selfie_mode:
            # 检查自拍功能是否启用
            selfie_enabled = self.get_config("selfie.enabled", True)
            if not selfie_enabled:
                await self.send_text("自拍功能暂未启用~")
                return False, "自拍功能未启用"

            logger.info(f"{self.log_prefix} 启用自拍模式，风格: {selfie_style}")
            description = self._process_selfie_prompt(description, selfie_style, free_hand_action, model_id)
            logger.info(f"{self.log_prefix} 自拍模式处理后的提示词: {description[:100]}...")

            # 检查是否配置了参考图片
            reference_image = self._get_selfie_reference_image()
            if reference_image:
                # 检查模型是否支持图生图
                model_config = self._get_model_config(model_id)
                if model_config and model_config.get("support_img2img", True):
                    logger.info(f"{self.log_prefix} 使用自拍参考图片进行图生图")
                    return await self._execute_unified_generation(description, model_id, size, strength or 0.6, reference_image)
                else:
                    logger.warning(f"{self.log_prefix} 模型 {model_id} 不支持图生图，自拍回退为文生图模式")
            # 无参考图或模型不支持，继续使用文生图

        # **智能检测：判断是文生图还是图生图**
        input_image_base64 = await self.image_processor.get_recent_image()
        is_img2img_mode = input_image_base64 is not None

        if is_img2img_mode:
            # 检查指定模型是否支持图生图
            model_config = self._get_model_config(model_id)
            if model_config and not model_config.get("support_img2img", True):
                logger.warning(f"{self.log_prefix} 模型 {model_id} 不支持图生图，转为文生图模式")
                await self.send_text(f"当前模型 {model_id} 不支持图生图功能，将为您生成新图片")
                return await self._execute_unified_generation(description, model_id, size, None, None)

            logger.info(f"{self.log_prefix} 检测到输入图片，使用图生图模式")
            return await self._execute_unified_generation(description, model_id, size, strength, input_image_base64)
        else:
            logger.info(f"{self.log_prefix} 未检测到输入图片，使用文生图模式")
            return await self._execute_unified_generation(description, model_id, size, None, None)

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

        # 使用统一的尺寸处理逻辑
        image_size, llm_original_size = get_image_size(model_config, size, self.log_prefix)

        # 验证图片尺寸格式
        if not self._validate_image_size(image_size):
            logger.warning(f"{self.log_prefix} 无效的图片尺寸: {image_size}，使用模型默认值")
            image_size = model_config.get("default_size", "1024x1024")

        # 检查缓存
        is_img2img = input_image_base64 is not None
        cached_result = self.cache_manager.get_cached_result(description, model_name, image_size, strength, is_img2img)

        if cached_result:
            logger.info(f"{self.log_prefix} 使用缓存的图片结果")
            enable_debug = self.get_config("components.enable_debug_info", False)
            if enable_debug:
                await self.send_text("我之前画过类似的图片，用之前的结果~")
            send_success = await self.send_image(cached_result)
            if send_success:
                return True, "图片已发送(缓存)"
            else:
                self.cache_manager.remove_cached_result(description, model_name, image_size, strength, is_img2img)

        # 显示处理信息
        enable_debug = self.get_config("components.enable_debug_info", False)
        if enable_debug:
            mode_text = "图生图" if is_img2img else "文生图"
            await self.send_text(
                f"收到！正在为您使用 {model_id or '默认'} 模型进行{mode_text}，描述: '{description}'，请稍候...（模型: {model_name}, 尺寸: {image_size}）"
            )

        try:
            # 对于 Gemini/Zai 格式，将原始 LLM 尺寸添加到 model_config 中
            if api_format in ("gemini", "zai") and llm_original_size:
                model_config = dict(model_config)  # 创建副本避免修改原配置
                model_config["_llm_original_size"] = llm_original_size

            # 获取重试次数配置
            max_retries = self.get_config("components.max_retries", 2)

            # 获取对应格式的API客户端并调用
            api_client = self._get_api_client(api_format)
            success, result = await api_client.generate_image(
                prompt=description,
                model_config=model_config,
                size=image_size,
                strength=strength,
                input_image_base64=input_image_base64,
                max_retries=max_retries
            )
        except Exception as e:
            logger.error(f"{self.log_prefix} 异步请求执行失败: {e!r}", exc_info=True)
            traceback.print_exc()
            success = False
            result = f"图片生成服务遇到意外问题: {str(e)[:100]}"

        if success:
            final_image_data = self.image_processor.process_api_response(result)

            if final_image_data:
                if final_image_data.startswith(("iVBORw", "/9j/", "UklGR", "R0lGOD")):  # Base64
                    send_success = await self.send_image(final_image_data)
                    if send_success:
                        mode_text = "图生图" if is_img2img else "文生图"
                        if enable_debug:
                            await self.send_text(f"{mode_text}完成！")
                        # 缓存成功的结果
                        self.cache_manager.cache_result(description, model_name, image_size, strength, is_img2img, final_image_data)
                        # 安排自动撤回（如果该模型启用）
                        await self._schedule_auto_recall_for_recent_message(model_config)
                        return True, f"{mode_text}已成功生成并发送"
                    else:
                        await self.send_text("图片已处理完成，但发送失败了")
                        return False, "图片发送失败"
                else:  # URL
                    try:
                        encode_success, encode_result = await asyncio.to_thread(
                            self.image_processor.download_and_encode_base64, final_image_data
                        )
                        if encode_success:
                            send_success = await self.send_image(encode_result)
                            if send_success:
                                mode_text = "图生图" if is_img2img else "文生图"
                                if enable_debug:
                                    await self.send_text(f"{mode_text}完成！")
                                # 缓存成功结果
                                self.cache_manager.cache_result(description, model_name, image_size, strength, is_img2img, encode_result)
                                # 安排自动撤回（如果该模型启用）
                                await self._schedule_auto_recall_for_recent_message(model_config)
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

    def _validate_image_size(self, size: str) -> bool:
        """验证图片尺寸格式是否正确（委托给size_utils）"""
        return validate_image_size(size)

    def _process_selfie_prompt(self, description: str, selfie_style: str, free_hand_action: str, model_id: str) -> str:
        """处理自拍模式的提示词生成

        Args:
            description: 用户提供的描述
            selfie_style: 自拍风格（standard/mirror）
            free_hand_action: LLM生成的手部动作（可选）
            model_id: 模型ID（保留参数，用于后续扩展）

        Returns:
            处理后的完整提示词
        """
        import random

        # 1. 添加强制主体设置
        forced_subject = "(1girl:1.4), (solo:1.3)"

        # 2. 从独立的selfie配置中获取Bot的默认形象特征（不再从模型配置中获取）
        bot_appearance = self.get_config("selfie.prompt_prefix", "").strip()

        # 3. 定义自拍风格特定的场景设置
        if selfie_style == "mirror":
            # 对镜自拍风格（适用于有镜子的室内场景）
            selfie_scene = "mirror selfie, holding phone, reflection in mirror, bathroom, bedroom mirror, indoor"
        else:
            # 标准自拍风格（适用于户外或无镜子场景，前置摄像头视角）
            selfie_scene = "selfie, front camera view, arm extended, looking at camera"

        # 4. 智能手部动作库（40+种动作）
        hand_actions = [
            # 经典手势
            "peace sign, v sign",
            "waving hand, friendly gesture",
            "thumbs up, positive gesture",
            "finger heart, cute pose",
            "ok sign, hand gesture",

            # 可爱动作
            "touching face gently, soft expression",
            "hand near chin, thinking pose",
            "covering mouth with hand, shy expression",
            "both hands on cheeks, surprised",
            "one hand in hair, casual pose",

            # 时尚姿态
            "hand on hip, confident pose",
            "adjusting hair, elegant gesture",
            "fixing collar, neat appearance",
            "checking nails, stylish pose",
            "hand behind head, relaxed",

            # 表情包系列
            "saluting, military pose",
            "finger gun, playful gesture",
            "crossed arms, cool pose",
            "hand shielding eyes, looking far",
            "hands clasped together, pleading",

            # 甜美系列
            "blowing kiss, romantic",
            "heart shape with hands",
            "hugging self, content",
            "cat paw gesture, playful",
            "bunny ears with fingers",

            # 自然动作
            "resting chin on hand, relaxed",
            "stretching arms, energetic",
            "fixing glasses, nerdy",
            "touching necklace, delicate",
            "adjusting earring, fashionable",

            # 情绪表达
            "fist pump, excited",
            "hands together praying, hopeful",
            "wiping forehead, relieved",
            "scratching head, confused",
            "finger on lips, secretive",

            # 特殊pose
            "making frame with fingers, photographer pose",
            "counting on fingers, cute",
            "pointing at viewer, engaging",
            "covering one eye, mysterious",
            "both hands up, surprised reaction"
        ]

        # 5. 选择手部动作
        if free_hand_action:
            # 优先使用LLM生成的手部动作
            logger.info(f"{self.log_prefix} 使用LLM生成的手部动作: {free_hand_action}")
            hand_action = free_hand_action
        else:
            # 兜底：随机选择一个手部动作
            hand_action = random.choice(hand_actions)
            logger.info(f"{self.log_prefix} 随机选择手部动作: {hand_action}")

        # 6. 组装完整提示词
        # 格式：强制主体 + Bot形象 + 手部动作 + 自拍场景 + 用户描述
        prompt_parts = [forced_subject]

        if bot_appearance:
            prompt_parts.append(bot_appearance)

        prompt_parts.extend([
            hand_action,
            selfie_scene,
            description
        ])

        # 7. 合并并去重
        final_prompt = ", ".join(prompt_parts)

        # 8. 简单的去重处理（避免重复关键词）
        # 将提示词拆分，去除重复的关键词组合
        keywords = [kw.strip() for kw in final_prompt.split(',')]
        seen = set()
        unique_keywords = []
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower not in seen and kw:
                seen.add(kw_lower)
                unique_keywords.append(kw)

        final_prompt = ", ".join(unique_keywords)

        logger.info(f"{self.log_prefix} 自拍模式最终提示词: {final_prompt[:200]}...")
        return final_prompt

    def _get_selfie_reference_image(self) -> Optional[str]:
        """获取自拍参考图片的base64编码

        Returns:
            图片的base64编码，如果不存在则返回None
        """
        image_path = self.get_config("selfie.reference_image_path", "").strip()
        if not image_path:
            return None

        try:
            # 处理相对路径（相对于插件目录）
            if not os.path.isabs(image_path):
                plugin_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                image_path = os.path.join(plugin_dir, image_path)

            if os.path.exists(image_path):
                with open(image_path, 'rb') as f:
                    image_data = f.read()
                image_base64 = base64.b64encode(image_data).decode('utf-8')
                logger.info(f"{self.log_prefix} 从文件加载自拍参考图片: {image_path}")
                return image_base64
            else:
                logger.warning(f"{self.log_prefix} 自拍参考图片文件不存在: {image_path}")
                return None
        except Exception as e:
            logger.error(f"{self.log_prefix} 加载自拍参考图片失败: {e}")
            return None

    async def _schedule_auto_recall_for_recent_message(self, model_config: Dict[str, Any] = None):
        """安排最近发送消息的自动撤回

        通过查询数据库获取最近发送的消息ID，然后安排撤回任务

        Args:
            model_config: 当前使用的模型配置，用于检查撤回延时设置
        """
        # 检查全局开关
        global_enabled = self.get_config("auto_recall.enabled", False)
        if not global_enabled:
            return

        # 检查模型的撤回延时，大于0才启用
        if not model_config:
            return

        delay_seconds = model_config.get("auto_recall_delay", 0)
        if delay_seconds <= 0:
            return

        # 获取模型ID用于检查运行时撤回状态
        model_id = None
        models_config = self.get_config("models", {})
        for mid, config in models_config.items():
            # 通过模型名称匹配，避免字典比较问题
            if config.get("model") == model_config.get("model"):
                model_id = mid
                break

        # 检查运行时撤回状态
        if model_id and not runtime_state.is_recall_enabled(self.chat_id, model_id, global_enabled):
            logger.info(f"{self.log_prefix} 模型 {model_id} 撤回已在当前聊天流禁用")
            return

        # 创建异步任务
        async def recall_task():
            try:
                # 等待足够时间让消息存储和 echo 回调完成（平台返回真实消息ID需要时间）
                await asyncio.sleep(4)

                # 查询最近发送的消息获取消息ID
                import time as time_module
                from src.plugin_system.apis import message_api
                from src.config.config import global_config

                current_time = time_module.time()
                # 查询最近10秒内本聊天中Bot发送的消息
                messages = message_api.get_messages_by_time_in_chat(
                    chat_id=self.chat_id,
                    start_time=current_time - 10,
                    end_time=current_time + 1,
                    limit=5,
                    limit_mode="latest"
                )

                # 找到Bot发送的图片消息
                bot_id = str(global_config.bot.qq_account)
                target_message_id = None

                for msg in messages:
                    if str(msg.user_info.user_id) == bot_id:
                        # 找到Bot发送的最新消息
                        mid = str(msg.message_id)
                        # 只使用纯数字的消息ID（QQ平台真实ID），跳过 send_api_xxx 格式的内部ID
                        if mid.isdigit():
                            target_message_id = mid
                            break
                        else:
                            logger.debug(f"{self.log_prefix} 跳过非平台消息ID: {mid}")

                if not target_message_id:
                    logger.warning(f"{self.log_prefix} 未找到有效的平台消息ID（需要纯数字格式）")
                    return

                logger.info(f"{self.log_prefix} 安排消息自动撤回，延时: {delay_seconds}秒，消息ID: {target_message_id}")

                # 等待指定时间后撤回
                await asyncio.sleep(delay_seconds)

                # 尝试多个撤回命令名（参考 recall_manager_plugin）
                DELETE_COMMAND_CANDIDATES = ["DELETE_MSG", "delete_msg", "RECALL_MSG", "recall_msg"]
                recall_success = False

                for cmd in DELETE_COMMAND_CANDIDATES:
                    try:
                        result = await self.send_command(
                            command_name=cmd,
                            args={"message_id": str(target_message_id)},
                            storage_message=False
                        )

                        # 检查返回结果
                        if isinstance(result, bool) and result:
                            recall_success = True
                            logger.info(f"{self.log_prefix} 消息自动撤回成功，命令: {cmd}，消息ID: {target_message_id}")
                            break
                        elif isinstance(result, dict):
                            status = str(result.get("status", "")).lower()
                            if status in ("ok", "success") or result.get("retcode") == 0 or result.get("code") == 0:
                                recall_success = True
                                logger.info(f"{self.log_prefix} 消息自动撤回成功，命令: {cmd}，消息ID: {target_message_id}")
                                break
                    except Exception as e:
                        logger.debug(f"{self.log_prefix} 撤回命令 {cmd} 失败: {e}")
                        continue

                if not recall_success:
                    logger.warning(f"{self.log_prefix} 消息自动撤回失败，消息ID: {target_message_id}，已尝试所有命令")

            except asyncio.CancelledError:
                logger.debug(f"{self.log_prefix} 自动撤回任务被取消")
            except Exception as e:
                logger.error(f"{self.log_prefix} 自动撤回失败: {e}")

        # 启动后台任务
        asyncio.create_task(recall_task())

    def _extract_description_from_message(self) -> str:
        """从用户消息中提取图片描述
        
        Returns:
            str: 提取的图片描述，如果无法提取则返回空字符串
        """
        if not self.action_message:
            return ""
            
        # 获取消息文本
        message_text = (self.action_message.processed_plain_text or
                       self.action_message.display_message or
                       self.action_message.raw_message or "").strip()
        
        if not message_text:
            return ""
            
        import re
        
        # 移除常见的画图相关前缀
        patterns_to_remove = [
            r'^画',           # "画"
            r'^绘制',         # "绘制"
            r'^生成图片',     # "生成图片"
            r'^画图',         # "画图"
            r'^帮我画',       # "帮我画"
            r'^请画',         # "请画"
            r'^能不能画',     # "能不能画"
            r'^可以画',       # "可以画"
            r'^画一个',       # "画一个"
            r'^画一只',       # "画一只"
            r'^画张',         # "画张"
            r'^画幅',         # "画幅"
            r'^图[：:]',      # "图："或"图:"
            r'^生成图片[：:]', # "生成图片："或"生成图片:"
            r'^[：:]',        # 单独的冒号
        ]
        
        cleaned_text = message_text
        for pattern in patterns_to_remove:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)
        
        # 移除常见的后缀
        suffix_patterns = [
            r'图片$',         # "图片"
            r'图$',           # "图"
            r'一下$',         # "一下"
            r'呗$',           # "呗"
            r'吧$',           # "吧"
        ]
        
        for pattern in suffix_patterns:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)
        
        # 清理空白字符
        cleaned_text = cleaned_text.strip()
        
        # 如果清理后为空，返回原消息（可能是简单的描述）
        if not cleaned_text:
            cleaned_text = message_text
            
        # 限制长度，避免过长的描述
        if len(cleaned_text) > 100:
            cleaned_text = cleaned_text[:100]
            
        return cleaned_text

