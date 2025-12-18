from typing import List, Tuple, Type, Dict, Any
import os

from src.plugin_system.base.base_plugin import BasePlugin
from src.plugin_system.base.component_types import ComponentInfo
from src.plugin_system import register_plugin
from src.plugin_system.base.config_types import (
    ConfigField,
    ConfigSection,
    ConfigLayout,
    ConfigTab,
)

from .core.pic_action import Custom_Pic_Action
from .core.pic_command import PicGenerationCommand, PicConfigCommand, PicStyleCommand
from .core.config_manager import EnhancedConfigManager


@register_plugin
class CustomPicPlugin(BasePlugin):
    """统一的多模型图片生成插件，支持文生图和图生图"""

    # 插件基本信息
    plugin_name = "custom_pic_plugin"
    plugin_version = "3.3.3"
    plugin_author = "Ptrel，Rabbit"
    enable_plugin = True
    dependencies: List[str] = []
    python_dependencies: List[str] = []
    config_file_name = "config.toml"

    # 配置节元数据
    config_section_descriptions = {
        "plugin": ConfigSection(
            title="插件启用配置",
            icon="info",
            order=1
        ),
        "generation": ConfigSection(
            title="图片生成默认配置",
            icon="image",
            order=2
        ),
        "components": ConfigSection(
            title="组件启用配置",
            icon="puzzle-piece",
            order=3
        ),
        "proxy": ConfigSection(
            title="代理设置",
            icon="globe",
            order=4
        ),
        "cache": ConfigSection(
            title="结果缓存配置",
            icon="database",
            order=5
        ),
        "selfie": ConfigSection(
            title="自拍模式配置",
            icon="camera",
            order=6
        ),
        "auto_recall": ConfigSection(
            title="自动撤回配置",
            icon="trash",
            order=7
        ),
        "prompt_optimizer": ConfigSection(
            title="提示词优化器",
            description="使用 MaiBot 主 LLM 将用户描述优化为专业绘画提示词",
            icon="wand-2",
            order=8
        ),
        "styles": ConfigSection(
            title="风格定义",
            description="预设风格的提示词。添加更多风格请直接编辑 config.toml，格式：风格英文名 = \"提示词\"",
            icon="palette",
            order=9
        ),
        "style_aliases": ConfigSection(
            title="风格别名",
            description="风格的中文别名映射。添加更多别名请直接编辑 config.toml",
            icon="tag",
            order=10
        ),
        "logging": ConfigSection(
            title="日志配置",
            icon="file-text",
            collapsed=True,
            order=11
        ),
        "models": ConfigSection(
            title="多模型配置",
            description="添加更多模型请直接编辑 config.toml，复制 [models.model1] 整节并改名为 model2、model3 等",
            icon="cpu",
            order=12
        ),
        "models.model1": ConfigSection(
            title="模型1配置",
            icon="box",
            order=13
        ),
    }

    # 自定义布局：标签页
    config_layout = ConfigLayout(
        type="tabs",
        tabs=[
            ConfigTab(
                id="basic",
                title="基础设置",
                sections=["plugin", "generation", "components"],
                icon="settings"
            ),
            ConfigTab(
                id="network",
                title="网络配置",
                sections=["proxy", "cache"],
                icon="wifi"
            ),
            ConfigTab(
                id="features",
                title="功能配置",
                sections=["selfie", "auto_recall", "prompt_optimizer"],
                icon="zap"
            ),
            ConfigTab(
                id="styles",
                title="风格管理",
                sections=["styles", "style_aliases"],
                icon="palette"
            ),
            ConfigTab(
                id="models",
                title="模型管理",
                sections=["models", "models.model1"],
                icon="cpu"
            ),
            ConfigTab(
                id="advanced",
                title="高级",
                sections=["logging"],
                icon="terminal",
                badge="Dev"
            ),
        ]
    )

    # 配置Schema
    config_schema = {
        "plugin": {
            "name": ConfigField(
                type=str,
                default="custom_pic_plugin",
                description="智能多模型图片生成插件，支持文生图/图生图自动识别",
                required=True,
                disabled=True,
                order=1
            ),
            "config_version": ConfigField(
                type=str,
                default="3.3.3",
                description="插件配置版本号",
                disabled=True,
                order=2
            ),
            "enabled": ConfigField(
                type=bool,
                default=False,
                description="是否启用插件，开启后可使用画图和风格转换功能",
                order=3
            )
        },
        "generation": {
            "default_model": ConfigField(
                type=str,
                default="model1",
                description="默认使用的模型ID，用于智能图片生成。支持文生图和图生图自动识别",
                placeholder="model1",
                order=1
            ),
        },
        "cache": {
            "enabled": ConfigField(
                type=bool,
                default=True,
                description="是否启用结果缓存，相同参数的请求会复用之前的结果",
                order=1
            ),
            "max_size": ConfigField(
                type=int,
                default=10,
                description="最大缓存数量，超出后删除最旧的缓存",
                min=1,
                max=100,
                depends_on="cache.enabled",
                depends_value=True,
                order=2
            ),
        },
        "components": {
            "enable_unified_generation": ConfigField(
                type=bool,
                default=True,
                description="是否启用智能图片生成Action，支持文生图和图生图自动识别",
                order=1
            ),
            "enable_pic_command": ConfigField(
                type=bool,
                default=True,
                description="是否启用风格化图生图Command功能，支持/dr <风格>命令",
                order=2
            ),
            "enable_pic_config": ConfigField(
                type=bool,
                default=True,
                description="是否启用模型配置管理命令，支持/dr list、/dr set等",
                order=3
            ),
            "enable_pic_style": ConfigField(
                type=bool,
                default=True,
                description="是否启用风格管理命令，支持/dr styles、/dr style等",
                order=4
            ),
            "pic_command_model": ConfigField(
                type=str,
                default="model1",
                description="Command组件使用的模型ID，可通过/dr set命令动态切换",
                placeholder="model1",
                order=5
            ),
            "enable_debug_info": ConfigField(
                type=bool,
                default=False,
                description="是否启用调试信息显示，关闭后仅显示图片结果和错误信息",
                order=6
            ),
            "enable_verbose_debug": ConfigField(
                type=bool,
                default=False,
                description="是否启用详细调试信息，启用后会发送完整的调试信息以及打印完整的 POST 报文",
                order=7
            ),
            "admin_users": ConfigField(
                type=list,
                default=[],
                description="有权限使用配置管理命令的管理员用户列表，请填写字符串形式的用户ID",
                placeholder="[\"用户ID1\", \"用户ID2\"]",
                order=8
            ),
            "max_retries": ConfigField(
                type=int,
                default=2,
                description="API调用失败时的重试次数，建议2-5次。设置为0表示不重试",
                min=0,
                max=10,
                order=9
            )
        },
        "logging": {
            "level": ConfigField(
                type=str,
                default="INFO",
                description="日志记录级别，DEBUG显示详细信息",
                choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                order=1
            ),
            "prefix": ConfigField(
                type=str,
                default="[unified_pic_Plugin]",
                description="日志前缀标识",
                placeholder="[插件名]",
                order=2
            )
        },
        "proxy": {
            "enabled": ConfigField(
                type=bool,
                default=False,
                description="是否启用代理。开启后所有API请求将通过代理服务器",
                order=1
            ),
            "url": ConfigField(
                type=str,
                default="http://127.0.0.1:7890",
                description="代理服务器地址，格式：http://host:port。支持HTTP/HTTPS/SOCKS5代理",
                placeholder="http://127.0.0.1:7890",
                depends_on="proxy.enabled",
                depends_value=True,
                order=2
            ),
            "timeout": ConfigField(
                type=int,
                default=60,
                description="代理连接超时时间（秒），建议30-120秒",
                min=10,
                max=300,
                depends_on="proxy.enabled",
                depends_value=True,
                order=3
            )
        },
        "styles": {
            "hint": ConfigField(
                type=str,
                default="添加更多风格：编辑 config.toml，在 [styles] 节下添加 风格英文名 = \"提示词\"",
                description="配置说明",
                disabled=True,
                order=0
            ),
            "cartoon": ConfigField(
                type=str,
                default="cartoon style, anime style, colorful, vibrant colors, clean lines",
                description="卡通风格提示词",
                input_type="textarea",
                rows=3,
                order=1
            )
        },
        "style_aliases": {
            "hint": ConfigField(
                type=str,
                default="添加更多别名：编辑 config.toml，在 [style_aliases] 节下添加 风格英文名 = \"中文别名\"",
                description="配置说明",
                disabled=True,
                order=0
            ),
            "cartoon": ConfigField(
                type=str,
                default="卡通",
                description="cartoon 风格的中文别名，支持多别名用逗号分隔",
                placeholder="卡通,动漫",
                order=1
            )
        },
        "selfie": {
            "enabled": ConfigField(
                type=bool,
                default=True,
                description="是否启用自拍模式功能",
                order=1
            ),
            "reference_image_path": ConfigField(
                type=str,
                default="",
                description="自拍参考图片路径（相对于插件目录或绝对路径）。配置后自动使用图生图模式，留空则使用纯文生图。若模型不支持图生图会自动回退",
                placeholder="images/reference.png",
                depends_on="selfie.enabled",
                depends_value=True,
                order=2
            ),
            "prompt_prefix": ConfigField(
                type=str,
                default="",
                description="自拍模式专用提示词前缀。用于添加Bot的默认形象特征（发色、瞳色、服装风格等）。例如：'blue hair, red eyes, school uniform, 1girl'",
                input_type="textarea",
                rows=2,
                placeholder="blue hair, red eyes, school uniform, 1girl",
                depends_on="selfie.enabled",
                depends_value=True,
                order=3
            )
        },
        "auto_recall": {
            "enabled": ConfigField(
                type=bool,
                default=False,
                description="是否启用自动撤回功能（总开关）。关闭后所有模型的撤回都不生效",
                order=1
            )
        },
        "prompt_optimizer": {
            "enabled": ConfigField(
                type=bool,
                default=True,
                description="是否启用提示词优化器。开启后会使用 MaiBot 主 LLM 将用户描述优化为专业英文提示词",
                order=1
            ),
            "hint": ConfigField(
                type=str,
                default="优化器会自动将中文描述翻译并优化为专业的英文绘画提示词，提升生成效果。关闭后将直接使用用户原始描述。",
                description="功能说明",
                disabled=True,
                order=2
            )
        },
        "models": {
            "hint": ConfigField(
                type=str,
                default="添加更多模型：编辑 config.toml，复制 [models.model1] 整节并改名为 model2、model3 等",
                description="配置说明",
                disabled=True,
                order=1
            )
        },
        # 基础模型配置模板
        "models.model1": {
            "name": ConfigField(
                type=str,
                default="魔搭潦草模型",
                description="模型显示名称，在模型列表中展示，版本更新后请手动从 old 目录恢复配置",
                order=1
            ),
            "base_url": ConfigField(
                type=str,
                default="https://api-inference.modelscope.cn/v1",
                description="API服务地址。示例: OpenAI=https://api.openai.com/v1, 硅基流动=https://api.siliconflow.cn/v1, 豆包=https://ark.cn-beijing.volces.com/api/v3, 魔搭=https://api-inference.modelscope.cn/v1, Gemini=https://generativelanguage.googleapis.com",
                required=True,
                placeholder="https://api.example.com/v1",
                order=2
            ),
            "api_key": ConfigField(
                type=str,
                default="Bearer xxxxxxxxxxxxxxxxxxxxxx",
                description="API密钥。OpenAI/modelscope格式需'Bearer '前缀，豆包/Gemini格式无需前缀",
                input_type="password",
                required=True,
                placeholder="Bearer sk-xxx 或 sk-xxx",
                order=3
            ),
            "format": ConfigField(
                type=str,
                default="openai",
                description="API格式。openai=通用格式，doubao=豆包，gemini=Gemini，modelscope=魔搭，shatangyun=砂糖云(NovelAI)，mengyuai=梦羽AI，zai=Zai(Gemini转发)",
                choices=["openai", "gemini", "doubao", "modelscope", "shatangyun", "mengyuai", "zai"],
                order=4
            ),
            "model": ConfigField(
                type=str,
                default="cancel13/liaocao",
                description="模型名称。梦羽AI格式填写模型索引数字（如0、1、2）",
                placeholder="model-name 或 0",
                order=5
            ),
            "fixed_size_enabled": ConfigField(
                type=bool,
                default=False,
                description="是否固定图片尺寸。开启后强制使用default_size，关闭则麦麦选择",
                order=6
            ),
            "default_size": ConfigField(
                type=str,
                default="1024x1024",
                description="默认图片尺寸。OpenAI/豆包/魔搭格式填写如 1024x1024。Gemini格式填写宽高比如 16:9 或 16:9-2K，具体参考官方文档",
                placeholder="1024x1024 或 16:9-2K",
                order=7
            ),
            "seed": ConfigField(
                type=int,
                default=42,
                description="随机种子，固定值可确保结果可复现",
                min=-1,
                max=2147483647,
                order=8
            ),
            "guidance_scale": ConfigField(
                type=float,
                default=2.5,
                description="指导强度。豆包推荐5.5，其他推荐2.5。越高越严格遵循提示词",
                min=0.0,
                max=20.0,
                step=0.5,
                order=9
            ),
            "num_inference_steps": ConfigField(
                type=int,
                default=20,
                description="推理步数，影响质量和速度。推荐20-50",
                min=1,
                max=150,
                order=10
            ),
            "watermark": ConfigField(
                type=bool,
                default=True,
                description="是否添加水印，豆包默认支持",
                order=11
            ),
            "custom_prompt_add": ConfigField(
                type=str,
                default=", Nordic picture book art style, minimalist flat design, liaocao",
                description="正面提示词增强，自动添加到用户描述后",
                input_type="textarea",
                rows=2,
                order=12
            ),
            "negative_prompt_add": ConfigField(
                type=str,
                default="Pornography,nudity,lowres, bad anatomy, bad hands, text, error",
                description="负面提示词，避免不良内容。豆包可留空但需保留引号",
                input_type="textarea",
                rows=2,
                order=13
            ),
            "artist": ConfigField(
                type=str,
                default="",
                description="艺术家风格标签（砂糖云专用）。留空则不添加",
                order=14
            ),
            "support_img2img": ConfigField(
                type=bool,
                default=True,
                description="该模型是否支持图生图功能，请根据API文档自行判断。设为false时会自动降级为文生图",
                order=15
            ),
            "auto_recall_delay": ConfigField(
                type=int,
                default=0,
                description="自动撤回延时（秒）。大于0时启用撤回，0表示不撤回。需先在「自动撤回配置」中开启总开关",
                min=0,
                max=120,
                order=16
            ),
        }
    }

    def __init__(self, plugin_dir: str):
        """初始化插件，集成增强配置管理器"""
        import toml
        # 在父类初始化前读取原始配置文件
        config_path = os.path.join(plugin_dir, self.config_file_name)
        original_config = None
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    original_config = toml.load(f)
                print(f"[CustomPicPlugin] 读取原始配置文件: {config_path}")
            except Exception as e:
                print(f"[CustomPicPlugin] 读取原始配置失败: {e}")
        
        # 先调用父类初始化，这会加载配置并可能触发 MaiBot 迁移
        super().__init__(plugin_dir)
        
        # 初始化增强配置管理器
        self.enhanced_config_manager = EnhancedConfigManager(plugin_dir, self.config_file_name)
        
        # 检查并更新配置（如果需要），传入原始配置
        self._enhance_config_management(original_config)
    
    def _enhance_config_management(self, original_config=None):
        """增强配置管理：备份、版本检查、智能合并
        
        Args:
            original_config: 从磁盘读取的原始配置（在父类初始化前读取），用于恢复用户自定义值
        """
        # 获取期望的配置版本
        expected_version = self._get_expected_config_version()
        
        # 将config_schema转换为EnhancedConfigManager需要的格式
        schema_for_manager = self._convert_schema_for_manager()
        
        # 生成默认配置结构
        default_config = self._generate_default_config_from_schema()
        
        # 确定要使用的旧配置：优先使用传入的原始配置，其次从备份文件加载
        old_config = original_config
        if old_config is None:
            old_dir = os.path.join(self.plugin_dir, "old")
            if os.path.exists(old_dir):
                import toml
                # 查找最新的备份文件（按时间戳排序），包括 auto_backup、new_backup 和 backup 文件
                backup_files = []
                for fname in os.listdir(old_dir):
                    if (fname.startswith(self.config_file_name + ".backup_") or
                        fname.startswith(self.config_file_name + ".new_backup_") or
                        fname.startswith(self.config_file_name + ".auto_backup_")) and fname.endswith(".toml"):
                        backup_files.append(fname)
                if backup_files:
                    # 按时间戳排序（文件名中包含 _YYYYMMDD_HHMMSS）
                    backup_files.sort(reverse=True)
                    latest_backup = os.path.join(old_dir, backup_files[0])
                    try:
                        with open(latest_backup, "r", encoding="utf-8") as f:
                            old_config = toml.load(f)
                        print(f"[CustomPicPlugin] 从备份文件加载原始配置: {backup_files[0]}")
                    except Exception as e:
                        print(f"[CustomPicPlugin] 加载备份文件失败: {e}")
        
        # 每次启动时创建备份（无论版本是否相同）
        # 加载当前配置文件以获取版本
        current_config = self.enhanced_config_manager.load_config()
        if current_config:
            current_version = self.enhanced_config_manager.get_config_version(current_config)
            print(f"[CustomPicPlugin] 当前配置版本 v{current_version}，创建启动备份")
            self.enhanced_config_manager.backup_config(current_version)
        else:
            print(f"[CustomPicPlugin] 配置文件不存在，跳过启动备份")
        
        # 使用增强配置管理器检查并更新配置
        # 传入旧配置（如果存在）以恢复用户自定义值
        updated_config = self.enhanced_config_manager.update_config_if_needed(
            expected_version=expected_version,
            default_config=default_config,
            schema=schema_for_manager,
            old_config=old_config
        )
        
        # 如果配置有更新，更新self.config
        if updated_config and updated_config != self.config:
            self.config = updated_config
            # 同时更新enable_plugin状态
            if "plugin" in self.config and "enabled" in self.config["plugin"]:
                self.enable_plugin = self.config["plugin"]["enabled"]
    
    def _get_expected_config_version(self) -> str:
        """获取期望的配置版本号"""
        if "plugin" in self.config_schema and isinstance(self.config_schema["plugin"], dict):
            config_version_field = self.config_schema["plugin"].get("config_version")
            if isinstance(config_version_field, ConfigField):
                return config_version_field.default
        return "1.0.0"
    
    def _convert_schema_for_manager(self) -> Dict[str, Any]:
        """将ConfigField格式的schema转换为EnhancedConfigManager需要的格式"""
        schema_for_manager = {}
        
        for section, fields in self.config_schema.items():
            if not isinstance(fields, dict):
                continue
                
            section_schema = {}
            for field_name, field in fields.items():
                if isinstance(field, ConfigField):
                    section_schema[field_name] = {
                        "description": field.description,
                        "default": field.default,
                        "required": field.required,
                        "choices": field.choices if field.choices else None,
                        "example": field.example
                    }
            
            schema_for_manager[section] = section_schema
        
        return schema_for_manager
    
    def _generate_default_config_from_schema(self) -> Dict[str, Any]:
        """从schema生成默认配置结构"""
        default_config = {}
        
        for section, fields in self.config_schema.items():
            if not isinstance(fields, dict):
                continue
                
            section_config = {}
            for field_name, field in fields.items():
                if isinstance(field, ConfigField):
                    section_config[field_name] = field.default
            
            default_config[section] = section_config
        
        return default_config

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
