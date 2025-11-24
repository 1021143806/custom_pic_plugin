import asyncio
import traceback
from typing import List, Tuple, Type, Optional, Dict, Any

from src.plugin_system.base.base_action import BaseAction
from src.plugin_system.base.component_types import ActionActivationType, ChatMode
from src.common.logger import get_logger

from .api_clients import ApiClient
from .image_utils import ImageProcessor
from .cache_manager import CacheManager

logger = get_logger("pic_action")

class Custom_Pic_Action(BaseAction):
    """统一的图片生成动作，智能检测文生图或图生图"""

    # 激活设置
    activation_type = ActionActivationType.LLM_JUDGE  # 默认激活类型
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.api_client = ApiClient(self)
        self.image_processor = ImageProcessor(self)
        self.cache_manager = CacheManager(self)

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
            # 调用API客户端生成图片
            success, result = await self.api_client.generate_image(
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
        """验证图片尺寸格式是否正确"""
        if not size or not isinstance(size, str):
            return False

        try:
            # 支持格式: "1024x1024", "512x512", "1024*1024" 等
            if 'x' in size:
                width, height = size.split('x', 1)
            elif '*' in size:
                width, height = size.split('*', 1)
            else:
                return False

            # 检查是否为数字且在合理范围内
            w, h = int(width.strip()), int(height.strip())
            return 64 <= w <= 4096 and 64 <= h <= 4096

        except (ValueError, AttributeError):
            return False