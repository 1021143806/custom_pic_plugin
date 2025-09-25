from typing import List, Tuple, Type

from src.plugin_system.base.base_plugin import BasePlugin
from src.plugin_system.base.component_types import ComponentInfo
from src.plugin_system import register_plugin
from src.plugin_system.base.config_types import ConfigField

from .core.pic_action import Custom_Pic_Action
from .core.pic_command import PicGenerationCommand, PicConfigCommand, PicStyleCommand

@register_plugin
class CustomPicPlugin(BasePlugin):
    """统一的多模型图片生成插件，支持文生图和图生图"""

    # 插件基本信息
    plugin_name = "custom_pic_plugin"  # 插件唯一标识符
    plugin_version = "3.1.2"  # 插件版本号
    plugin_author = "Ptrel，Rabbit"  # 插件作者
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
            "name": ConfigField(type=str, default="custom_pic_plugin", description="智能多模型图片生成插件，支持文生图/图生图自动识别", required=True),
            "config_version": ConfigField(type=str, default="3.1.2", description="插件配置版本号"),
            "enabled": ConfigField(type=bool, default=False, description="是否启用插件，开启后可使用画图和风格转换功能")
        },
        "generation": {
            "default_model": ConfigField(
                type=str,
                default="model1",
                description="默认使用的模型ID，用于智能图片生成。支持文生图和图生图自动识别",
                choices=["model1"]
            ),
        },
        "cache": {
            "enabled": ConfigField(type=bool, default=True, description="是否启用结果缓存，相同参数的请求会复用之前的结果"),
            "max_size": ConfigField(type=int, default=10, description="最大缓存数量，超出后删除最旧的缓存"),
        },
        "components": {
            "enable_unified_generation": ConfigField(type=bool, default=True, description="是否启用智能图片生成Action，支持文生图和图生图自动识别"),
            "enable_pic_command": ConfigField(type=bool, default=True, description="是否启用风格化图生图Command功能，支持/pic <风格>命令"),
            "enable_pic_config": ConfigField(type=bool, default=True, description="是否启用模型配置管理命令，支持/pic list、/pic set等"),
            "enable_pic_style": ConfigField(type=bool, default=True, description="是否启用风格管理命令，支持/pic styles、/pic style等"),
            "pic_command_model": ConfigField(type=str, default="model1", description="Command组件使用的模型ID，可通过/pic set命令动态切换"),
            "enable_debug_info": ConfigField(type=bool, default=False, description="是否启用调试信息显示，关闭后仅显示图片结果和错误信息"),
            "admin_users": ConfigField(
                type=list,
                default=[],
                description="有权限使用配置管理命令的管理员用户列表，请填写字符串形式的用户ID"
            ),
            "max_retries": ConfigField(type=int, default=2, description="API调用失败时的重试次数，建议2-5次。设置为0表示不重试"),
            "image_retry_count": ConfigField(type=int, default=3, description="图片获取失败时的重试次数，建议2-5次。设置为0表示不重试")
        },
        "logging": {
            "level": ConfigField(type=str, default="INFO", description="日志记录级别，DEBUG显示详细信息", choices=["DEBUG", "INFO", "WARNING", "ERROR"]),
            "prefix": ConfigField(type=str, default="[unified_pic_Plugin]", description="日志前缀标识")
        },
        "styles": {
            "cartoon": ConfigField(
                type=str,
                default="cartoon style, anime style, colorful, vibrant colors, clean lines",
                description="卡通风格提示词。可添加更多风格，格式: 英文名 = \"英文提示词\""
            )
        },
        "style_aliases": {
            "cartoon": ConfigField(
                type=str,
                default="卡通",
                description="风格中文别名，格式: 英文名 = \"中文名\"。支持多别名，用逗号分隔"
            )
        },
        "models": {},
        # 基础模型配置
        "models.model1": {
            "name": ConfigField(type=str, default="魔搭潦草模型", description="模型显示名称，在模型列表中展示"),
            "base_url": ConfigField(
                type=str,
                default="https://api-inference.modelscope.cn/v1",
                description="API服务地址。示例: OpenAI=https://api.openai.com/v1, 硅基流动=https://api.siliconflow.cn/v1, 豆包=https://ark.cn-beijing.volces.com/api/v3, 魔搭=https://api-inference.modelscope.cn/v1, Gemini=https://generativelanguage.googleapis.com",
                required=True
            ),
            "api_key": ConfigField(
                type=str,
                default="Bearer xxxxxxxxxxxxxxxxxxxxxx",
                description="API密钥。OpenAI/modelscope格式需'Bearer '前缀，豆包/Gemini格式无需前缀",
                required=True
            ),
            "format": ConfigField(
                type=str,
                default="openai",
                description="API格式。openai=通用格式，doubao=豆包专用，gemini=Gemini专用，modelscope=魔搭专用",
                choices=["openai", "gemini", "doubao", "modelscope"]
            ),
            "model": ConfigField(
                type=str,
                default="cancel13/liaocao",
                description="模型名称"
            ),
            "fixed_size_enabled": ConfigField(
                type=bool,
                default=False,
                description="是否固定图片尺寸。开启后强制使用default_size，关闭则麦麦选择"
            ),
            "default_size": ConfigField(
                type=str,
                default="1024x1024",
                description="默认图片尺寸，建议使用1024x1024以获得最佳效果",
                choices=["512x512", "1024x1024", "1024x1280", "1280x1024", "1024x1536", "1536x1024"]
            ),
            "seed": ConfigField(type=int, default=42, description="随机种子，固定值可确保结果可复现"),
            "guidance_scale": ConfigField(type=float, default=2.5, description="指导强度。豆包推荐5.5，其他推荐2.5。越高越严格遵循提示词"),
            "watermark": ConfigField(type=bool, default=True, description="是否添加水印，豆包默认支持"),
            "custom_prompt_add": ConfigField(
                type=str,
                default=", Nordic picture book art style, minimalist flat design, liaocao",
                description="正面提示词增强，自动添加到用户描述后"
            ),
            "negative_prompt_add": ConfigField(
                type=str,
                default="Pornography,nudity,lowres, bad anatomy, bad hands, text, error",
                description="负面提示词，避免不良内容。豆包可留空但需保留引号"
            ),
            "support_img2img": ConfigField(type=bool, default=True, description="是否支持图生图。不支持时自动降级为文生图"),
            "num_inference_steps": ConfigField(type=int, default=20, description="推理步数，影响质量和速度。推荐20-50"),
        }
    }

    def get_plugin_components(self) -> List[Tuple[ComponentInfo, Type]]:
        """返回插件包含的组件列表"""
        enable_unified_generation = self.get_config("components.enable_unified_generation", True)
        enable_pic_command = self.get_config("components.enable_pic_command", True)
        enable_pic_config = self.get_config("components.enable_pic_config", True)
        enable_pic_style = self.get_config("components.enable_pic_style", True)
        components = []

        if enable_unified_generation:
            components.append((Custom_Pic_Action.get_action_info(), Custom_Pic_Action))

        # 优先注册更具体的配置管理命令，避免被通用风格命令拦截
        if enable_pic_config:
            components.append((PicConfigCommand.get_command_info(), PicConfigCommand))

        if enable_pic_style:
            components.append((PicStyleCommand.get_command_info(), PicStyleCommand))

        # 最后注册通用的风格命令，以免覆盖特定命令
        if enable_pic_command:
            components.append((PicGenerationCommand.get_command_info(), PicGenerationCommand))

        return components