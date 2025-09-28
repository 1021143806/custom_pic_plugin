import asyncio
from typing import Tuple, Optional, Dict, Any

from src.plugin_system.base.base_command import BaseCommand
from src.common.logger import get_logger

from .api_clients import ApiClient
from .image_utils import ImageProcessor

logger = get_logger("pic_command")

class PicGenerationCommand(BaseCommand):
    """图生图Command组件，支持通过命令进行图生图，可选择特定模型"""

    # 类级别的配置覆盖
    _config_overrides = {}

    # Command基本信息
    command_name = "pic_generation_command"
    command_description = "图生图命令，使用风格化提示词：/pic <风格>"
    command_pattern = r"(?:.*，说：\s*)?/pic\s+(?P<style>[\u4e00-\u9fff\w]+)$"

    def get_config(self, key: str, default=None):
        """覆盖get_config方法以支持动态配置"""
        # 检查是否有配置覆盖
        if key in self._config_overrides:
            return self._config_overrides[key]
        # 否则使用父类的get_config
        return super().get_config(key, default)

    async def execute(self) -> Tuple[bool, Optional[str], bool]:
        """执行图生图命令"""
        logger.info(f"{self.log_prefix} 执行图生图命令")

        # 获取匹配的参数
        style_name = self.matched_groups.get("style", "").strip()

        if not style_name:
            await self.send_text("请指定风格，格式：/pic <风格>\n可用：/pic styles 查看")
            return False, "缺少风格参数", True

        # 检查是否是配置管理保留词，避免冲突
        config_reserved_words = {"list", "models", "config", "set", "reset", "styles", "style", "help"}
        if style_name.lower() in config_reserved_words:
            await self.send_text(f"'{style_name}' 是保留词，请使用其他风格名称")
            return False, f"使用了保留词: {style_name}", True

        # 从配置中获取Command组件使用的模型
        model_id = self.get_config("components.pic_command_model", "model1")

        # 获取模型配置
        model_config = self._get_model_config(model_id)
        if not model_config:
            await self.send_text(f"模型 '{model_id}' 不存在")
            return False, "模型配置不存在", True

        # 获取风格化提示词（支持别名映射）
        actual_style_name = self._resolve_style_alias(style_name)
        style_prompt = self._get_style_prompt(actual_style_name)
        if not style_prompt:
            await self.send_text(f"风格 '{style_name}' 不存在")
            return False, f"风格 '{style_name}' 不存在", True

        # 使用风格提示词作为描述
        final_description = style_prompt

        # 检查是否启用调试信息
        enable_debug = self.get_config("components.enable_debug_info", False)
        if enable_debug:
            await self.send_text(f"使用风格：{style_name}")

        # 获取最近的图片作为输入图片
        image_processor = ImageProcessor(self)
        input_image_base64 = await image_processor.get_recent_image()

        if not input_image_base64:
            await self.send_text("请先发送图片")
            return False, "未找到输入图片", True

        # 检查模型是否支持图生图
        if not model_config.get("support_img2img", True):
            await self.send_text(f"模型 {model_id} 不支持图生图")
            return False, f"模型 {model_id} 不支持图生图", True

        # 显示开始信息
        if enable_debug:
            await self.send_text(f"正在使用 {model_id} 模型进行 {style_name} 风格转换...")

        try:
            # 获取重试次数配置
            max_retries = self.get_config("components.max_retries", 2)

            # 调用API客户端生成图片
            api_client = ApiClient(self)
            success, result = await api_client.generate_image(
                prompt=final_description,
                model_config=model_config,
                size=model_config.get("default_size", "1024x1024"),
                strength=0.7,  # 默认强度
                input_image_base64=input_image_base64,
                max_retries=max_retries
            )

            if success:
                # 处理结果
                if result.startswith(("iVBORw", "/9j/", "UklGR", "R0lGOD")):  # Base64
                    send_success = await self.send_image(result)
                    if send_success:
                        if enable_debug:
                            await self.send_text(f"{style_name} 风格转换完成！")
                        return True, "图生图命令执行成功", True
                    else:
                        await self.send_text("图片发送失败")
                        return False, "图片发送失败", True
                else:  # URL
                    try:
                        # 下载并转换为base64
                        encode_success, encode_result = await asyncio.to_thread(
                            self._download_and_encode_base64, result
                        )
                        if encode_success:
                            send_success = await self.send_image(encode_result)
                            if send_success:
                                if enable_debug:
                                    await self.send_text(f"{style_name} 风格转换完成！")
                                return True, "图生图命令执行成功", True
                            else:
                                await self.send_text("图片发送失败")
                                return False, "图片发送失败", True
                        else:
                            await self.send_text(f"图片处理失败：{encode_result}")
                            return False, f"图片处理失败: {encode_result}", True
                    except Exception as e:
                        logger.error(f"{self.log_prefix} 图片下载编码失败: {e!r}")
                        await self.send_text("图片下载失败")
                        return False, "图片下载失败", True
            else:
                await self.send_text(f"{style_name} 风格转换失败：{result}")
                return False, f"图生图失败: {result}", True

        except Exception as e:
            logger.error(f"{self.log_prefix} 命令执行异常: {e!r}", exc_info=True)
            await self.send_text(f"执行失败：{str(e)[:100]}")
            return False, f"命令执行异常: {str(e)}", True

    def _get_model_config(self, model_id: str) -> Optional[Dict[str, Any]]:
        """获取模型配置"""
        try:
            model_config = self.get_config(f"models.{model_id}")
            if model_config and isinstance(model_config, dict):
                return model_config
            else:
                logger.warning(f"{self.log_prefix} 模型 {model_id} 配置不存在或格式错误")
                return None
        except Exception as e:
            logger.error(f"{self.log_prefix} 获取模型配置失败: {e!r}")
            return None

    def _resolve_style_alias(self, style_name: str) -> str:
        """解析风格别名，返回实际的风格名"""
        try:
            # 首先直接检查是否为有效的风格名
            if self.get_config(f"styles.{style_name}"):
                return style_name

            # 不是直接风格名，检查是否为别名
            style_aliases_config = self.get_config("style_aliases", {})
            if isinstance(style_aliases_config, dict):
                for english_name, aliases_str in style_aliases_config.items():
                    if isinstance(aliases_str, str):
                        # 支持多个别名，用逗号分隔
                        aliases = [alias.strip() for alias in aliases_str.split(',')]
                        if style_name in aliases:
                            logger.info(f"{self.log_prefix} 风格别名 '{style_name}' 解析为 '{english_name}'")
                            return english_name

            # 既不是直接风格名也不是别名，返回原名
            return style_name
        except Exception as e:
            logger.error(f"{self.log_prefix} 解析风格别名失败: {e!r}")
            return style_name

    def _get_style_prompt(self, style_name: str) -> Optional[str]:
        """获取风格提示词"""
        try:
            style_prompt = self.get_config(f"styles.{style_name}")
            if style_prompt and isinstance(style_prompt, str):
                return style_prompt.strip()
            else:
                logger.warning(f"{self.log_prefix} 风格 {style_name} 配置不存在或格式错误")
                return None
        except Exception as e:
            logger.error(f"{self.log_prefix} 获取风格配置失败: {e!r}")
            return None


    def _download_and_encode_base64(self, image_url: str) -> Tuple[bool, str]:
        """下载图片并转换为base64编码"""
        try:
            import requests
            import base64

            # 获取代理配置
            proxy_enabled = self.get_config("proxy.enabled", False)
            request_kwargs = {
                "url": image_url,
                "timeout": 30
            }

            if proxy_enabled:
                proxy_url = self.get_config("proxy.url", "http://127.0.0.1:7890")
                request_kwargs["proxies"] = {
                    "http": proxy_url,
                    "https": proxy_url
                }
                logger.info(f"{self.log_prefix} 下载图片使用代理: {proxy_url}")

            response = requests.get(**request_kwargs)
            if response.status_code == 200:
                image_base64 = base64.b64encode(response.content).decode('utf-8')
                return True, image_base64
            else:
                return False, f"HTTP {response.status_code}"
        except Exception as e:
            return False, str(e)


class PicConfigCommand(BaseCommand):
    """图片生成配置管理命令"""

    # Command基本信息
    command_name = "pic_config_command"
    command_description = "图片生成配置管理：/pic <操作> [参数]"
    command_pattern = r"(?:.*，说：\s*)?/pic\s+(?P<action>list|models|config|set|reset)(?:\s+(?P<params>.*))?$"

    def get_config(self, key: str, default=None):
        """使用与PicGenerationCommand相同的配置覆盖"""
        # 检查PicGenerationCommand的配置覆盖
        if key in PicGenerationCommand._config_overrides:
            return PicGenerationCommand._config_overrides[key]
        # 否则使用父类的get_config
        return super().get_config(key, default)

    async def execute(self) -> Tuple[bool, Optional[str], bool]:
        """执行配置管理命令"""
        logger.info(f"{self.log_prefix} 执行图片配置管理命令")

        # 获取匹配的参数
        action = self.matched_groups.get("action", "").strip()
        params = self.matched_groups.get("params", "") or ""
        params = params.strip()

        # 检查用户权限
        has_permission = self._check_permission()

        # 对于需要管理员权限的操作进行权限检查
        if not has_permission and action not in ["list", "models"]:
            await self.send_text("你无权使用此命令", storage_message=False)
            return False, "没有权限", True

        if action == "list" or action == "models":
            return await self._list_models()
        elif action == "set":
            return await self._set_model(params)
        elif action == "config":
            return await self._show_current_config()
        elif action == "reset":
            return await self._reset_config()
        else:
            await self.send_text(
                "配置管理命令使用方法：\n"
                "/pic list - 列出所有可用模型\n"
                "/pic config - 显示当前配置\n"
                "/pic set <模型ID> - 设置图生图命令模型\n"
                "/pic reset - 重置为默认配置"
            )
            return False, "无效的操作参数", True

    async def _list_models(self) -> Tuple[bool, Optional[str], bool]:
        """列出所有可用的模型"""
        try:
            models_config = self.get_config("models", {})
            if not models_config:
                await self.send_text("未找到任何模型配置")
                return False, "无模型配置", True

            # 获取当前默认模型
            current_default = self.get_config("generation.default_model", "model1")
            current_command = self.get_config("components.pic_command_model", "model1")

            message_lines = ["📋 可用模型列表：\n"]

            for model_id, config in models_config.items():
                if isinstance(config, dict):
                    model_name = config.get("model", "未知")
                    support_img2img = config.get("support_img2img", True)

                    # 标记当前使用的模型
                    default_mark = " ✅[默认]" if model_id == current_default else ""
                    command_mark = " 🔧[命令]" if model_id == current_command else ""
                    img2img_mark = " 🖼️[文/图生图]" if support_img2img else " 📝[仅文生图]"

                    message_lines.append(
                        f"• {model_id}{default_mark}{command_mark}{img2img_mark}\n"
                        f"  模型: {model_name}\n"
                    )

            message = "\n".join(message_lines)
            await self.send_text(message)
            return True, "模型列表查询成功", True

        except Exception as e:
            logger.error(f"{self.log_prefix} 列出模型失败: {e!r}")
            await self.send_text(f"获取模型列表失败：{str(e)[:100]}")
            return False, f"列出模型失败: {str(e)}", True

    async def _set_model(self, model_id: str) -> Tuple[bool, Optional[str], bool]:
        """设置图生图命令使用的模型"""
        try:
            if not model_id:
                await self.send_text("请指定模型ID，格式：/pic set <模型ID>")
                return False, "缺少模型ID参数", True

            # 检查模型是否存在
            model_config = self.get_config(f"models.{model_id}")
            if not model_config:
                await self.send_text(f"模型 '{model_id}' 不存在，请使用 /pic list 查看可用模型")
                return False, f"模型 '{model_id}' 不存在", True

            # 获取当前配置
            current_command_model = self.get_config("components.pic_command_model", "model1")
            model_name = model_config.get("model", "未知") if isinstance(model_config, dict) else "未知"

            if current_command_model == model_id:
                await self.send_text(f"✅ 当前图生图命令已经在使用模型 '{model_id}' ({model_name})")
                return True, "模型已是当前使用的模型", True

            # 尝试动态修改配置
            try:
                # 通过插件实例修改配置
                success = await self._update_command_model_config(model_id)

                if success:
                    await self.send_text(f"✅ 已切换到模型: {model_id}")
                    return True, f"模型切换成功: {model_id}", True
                else:
                    await self.send_text(f"⚠️ 切换失败，请手动修改配置文件")
                    return False, "动态配置更新失败", True

            except Exception as e:
                logger.error(f"{self.log_prefix} 动态更新配置失败: {e!r}")
                await self.send_text(f"⚠️ 配置更新失败：{str(e)[:50]}")
                return False, f"配置更新异常: {str(e)}", True

        except Exception as e:
            logger.error(f"{self.log_prefix} 设置模型失败: {e!r}")
            await self.send_text(f"设置失败：{str(e)[:100]}")
            return False, f"设置模型失败: {str(e)}", True

    async def _update_command_model_config(self, model_id: str) -> bool:
        """动态更新命令模型配置"""
        try:
            # 使用类级别的配置覆盖机制（这会影响所有PicGenerationCommand实例）
            PicGenerationCommand._config_overrides["components.pic_command_model"] = model_id

            logger.info(f"{self.log_prefix} 已设置配置覆盖: components.pic_command_model = {model_id}")
            return True

        except Exception as e:
            logger.error(f"{self.log_prefix} 更新配置时异常: {e!r}")
            return False

    async def _reset_config(self) -> Tuple[bool, Optional[str], bool]:
        """重置配置为默认值"""
        try:
            # 清除所有配置覆盖
            PicGenerationCommand._config_overrides.clear()

            # 获取默认配置
            default_model = super().get_config("components.pic_command_model", "model1")

            await self.send_text(
                f"✅ 配置已重置为默认值！\n\n"
                f"🔄 图生图命令模型: {default_model}\n"
                f"💡 所有运行时配置覆盖已清除\n\n"
                f"使用 /pic config 查看当前配置"
            )

            logger.info(f"{self.log_prefix} 配置已重置，清除了所有覆盖")
            return True, "配置重置成功", True

        except Exception as e:
            logger.error(f"{self.log_prefix} 重置配置失败: {e!r}")
            await self.send_text(f"重置失败：{str(e)[:100]}")
            return False, f"重置配置失败: {str(e)}", True

    async def _show_current_config(self) -> Tuple[bool, Optional[str], bool]:
        """显示当前配置信息"""
        try:
            # 获取当前配置
            default_model = self.get_config("generation.default_model", "model1")
            command_model = self.get_config("components.pic_command_model", "model1")
            cache_enabled = self.get_config("cache.enabled", True)
            debug_enabled = self.get_config("components.enable_debug_info", False)

            # 检查是否有配置覆盖
            original_command_model = super().get_config("components.pic_command_model", "model1")
            has_override = command_model != original_command_model

            # 获取默认模型详细信息
            default_config = self.get_config(f"models.{default_model}", {})
            command_config = self.get_config(f"models.{command_model}", {})

            # 构建配置信息
            message_lines = [
                "⚙️ 当前图片生成配置：\n",
                f"🎯 默认模型: {default_model}",
                f"   • 名称: {default_config.get('model', '未知') if isinstance(default_config, dict) else '未知'}\n",

                f"🔧 图生图命令模型: {command_model}" + (" 🔥[运行时]" if has_override else ""),
                f"   • 名称: {command_config.get('model', '未知') if isinstance(command_config, dict) else '未知'}",
            ]

            if has_override:
                message_lines.extend([
                    f"   • 原始配置: {original_command_model}",
                    f"   ⚡ 当前使用运行时覆盖配置"
                ])

            # 管理员命令提示
            message_lines.extend([
                "\n📖 管理员命令：",
                "• /pic list - 查看所有模型",
                "• /pic set <模型ID> - 设置图生图模型",
                "• /pic reset - 重置为默认配置",
                "• /pic <风格> - 使用风格进行图生图"
            ])

            message = "\n".join(message_lines)
            await self.send_text(message)
            return True, "配置信息查询成功", True

        except Exception as e:
            logger.error(f"{self.log_prefix} 显示配置失败: {e!r}")
            await self.send_text(f"获取配置失败：{str(e)[:100]}")
            return False, f"显示配置失败: {str(e)}", True

    def _check_permission(self) -> bool:
        """检查用户权限"""
        try:
            admin_users = self.get_config("components.admin_users", [])
            user_id = str(self.message.message_info.user_info.user_id) if self.message and self.message.message_info and self.message.message_info.user_info else None
            return user_id in admin_users
        except Exception:
            return False


class PicStyleCommand(BaseCommand):
    """图片风格管理命令"""

    # Command基本信息
    command_name = "pic_style_command"
    command_description = "图片风格管理：/pic <操作> [参数]"
    command_pattern = r"(?:.*，说：\s*)?/pic\s+(?P<action>styles|style|help)(?:\s+(?P<params>.*))?$"

    async def execute(self) -> Tuple[bool, Optional[str], bool]:
        """执行风格管理命令"""
        logger.info(f"{self.log_prefix} 执行图片风格管理命令")

        # 获取匹配的参数
        action = self.matched_groups.get("action", "").strip()
        params = self.matched_groups.get("params", "") or ""
        params = params.strip()

        # 检查用户权限
        has_permission = self._check_permission()

        # style命令需要管理员权限
        if action == "style" and not has_permission:
            await self.send_text("你无权使用此命令", storage_message=False)
            return False, "没有权限", True

        if action == "styles":
            return await self._list_styles()
        elif action == "style":
            return await self._show_style(params)
        elif action == "help":
            return await self._show_help()
        else:
            await self.send_text(
                "风格管理命令使用方法：\n"
                "/pic styles - 列出所有可用风格\n"
                "/pic style <风格名> - 显示风格详情\n"
                "/pic help - 显示帮助信息"
            )
            return False, "无效的操作参数", True

    async def _list_styles(self) -> Tuple[bool, Optional[str], bool]:
        """列出所有可用的风格"""
        try:
            styles_config = self.get_config("styles", {})
            aliases_config = self.get_config("style_aliases", {})

            if not styles_config:
                await self.send_text("未找到任何风格配置")
                return False, "无风格配置", True

            message_lines = ["🎨 可用风格列表：\n"]

            for style_id, prompt in styles_config.items():
                if isinstance(prompt, str):
                    # 查找这个风格的别名
                    aliases = []
                    for alias_style, alias_names in aliases_config.items():
                        if alias_style == style_id and isinstance(alias_names, str):
                            aliases = [name.strip() for name in alias_names.split(',')]
                            break

                    alias_text = f" (别名: {', '.join(aliases)})" if aliases else ""

                    message_lines.append(f"• {style_id}{alias_text}")

            message_lines.append("\n💡 使用方法: /pic <风格名>")
            message = "\n".join(message_lines)
            await self.send_text(message)
            return True, "风格列表查询成功", True

        except Exception as e:
            logger.error(f"{self.log_prefix} 列出风格失败: {e!r}")
            await self.send_text(f"获取风格列表失败：{str(e)[:100]}")
            return False, f"列出风格失败: {str(e)}", True

    async def _show_style(self, style_name: str) -> Tuple[bool, Optional[str], bool]:
        """显示指定风格的详细信息"""
        try:
            if not style_name:
                await self.send_text("请指定风格名，格式：/pic style <风格名>")
                return False, "缺少风格名参数", True

            # 解析风格别名
            actual_style = self._resolve_style_alias(style_name)
            style_prompt = self.get_config(f"styles.{actual_style}")

            if not style_prompt:
                await self.send_text(f"风格 '{style_name}' 不存在，请使用 /pic styles 查看可用风格")
                return False, f"风格 '{style_name}' 不存在", True

            # 查找别名
            aliases_config = self.get_config("style_aliases", {})
            aliases = []
            for alias_style, alias_names in aliases_config.items():
                if alias_style == actual_style and isinstance(alias_names, str):
                    aliases = [name.strip() for name in alias_names.split(',')]
                    break

            message_lines = [
                f"🎨 风格详情：{actual_style}\n",
                f"📝 完整提示词：",
                f"{style_prompt}\n"
            ]

            if aliases:
                message_lines.append(f"🏷️ 别名: {', '.join(aliases)}\n")

            message_lines.extend([
                "💡 使用方法：",
                f"/pic {style_name}",
                "\n⚠️ 注意：需要先发送一张图片作为输入"
            ])

            message = "\n".join(message_lines)
            await self.send_text(message)
            return True, "风格详情查询成功", True

        except Exception as e:
            logger.error(f"{self.log_prefix} 显示风格详情失败: {e!r}")
            await self.send_text(f"获取风格详情失败：{str(e)[:100]}")
            return False, f"显示风格详情失败: {str(e)}", True

    async def _show_help(self) -> Tuple[bool, Optional[str], bool]:
        """显示帮助信息"""
        try:
            # 检查用户权限
            has_permission = self._check_permission()

            if has_permission:
                # 管理员帮助信息
                help_text = """
🎨 图片风格系统帮助

📋 基本命令：
• /pic <风格名> - 对最近的图片应用风格
• /pic styles - 列出所有可用风格
• /pic list - 查看所有模型

⚙️ 管理员命令：
• /pic config - 查看当前配置
• /pic set <模型ID> - 设置图生图模型
• /pic reset - 重置为默认配置

💡 使用流程：
1. 发送一张图片
2. 使用 /pic <风格名> 进行风格转换
3. 等待处理完成
                """
            else:
                # 普通用户帮助信息
                help_text = """
🎨 图片风格系统帮助

📋 可用命令：
• /pic <风格名> - 对最近的图片应用风格
• /pic styles - 列出所有可用风格
• /pic list - 查看所有模型

💡 使用流程：
1. 发送一张图片
2. 使用 /pic <风格名> 进行风格转换
3. 等待处理完成
                """

            await self.send_text(help_text.strip())
            return True, "帮助信息显示成功", True

        except Exception as e:
            logger.error(f"{self.log_prefix} 显示帮助失败: {e!r}")
            await self.send_text(f"显示帮助信息失败：{str(e)[:100]}")
            return False, f"显示帮助失败: {str(e)}", True

    def _check_permission(self) -> bool:
        """检查用户权限"""
        try:
            admin_users = self.get_config("components.admin_users", [])
            user_id = str(self.message.message_info.user_info.user_id) if self.message and self.message.message_info and self.message.message_info.user_info else None
            return user_id in admin_users
        except Exception:
            return False

    def _resolve_style_alias(self, style_name: str) -> str:
        """解析风格别名，返回实际的风格名"""
        try:
            # 首先直接检查是否为有效的风格名
            if self.get_config(f"styles.{style_name}"):
                return style_name

            # 不是直接风格名，检查是否为别名
            style_aliases_config = self.get_config("style_aliases", {})
            if isinstance(style_aliases_config, dict):
                for english_name, aliases_str in style_aliases_config.items():
                    if isinstance(aliases_str, str):
                        # 支持多个别名，用逗号分隔
                        aliases = [alias.strip() for alias in aliases_str.split(',')]
                        if style_name in aliases:
                            logger.info(f"{self.log_prefix} 风格别名 '{style_name}' 解析为 '{english_name}'")
                            return english_name

            # 既不是直接风格名也不是别名，返回原名
            return style_name
        except Exception as e:
            logger.error(f"{self.log_prefix} 解析风格别名失败: {e!r}")
            return style_name