from typing import List, Tuple, Type

from src.plugin_system.base.base_plugin import BasePlugin
from src.plugin_system.base.component_types import ComponentInfo
from src.plugin_system import register_plugin
from src.plugin_system.base.config_types import ConfigField

from .core.pic_action import Custom_Pic_Action

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