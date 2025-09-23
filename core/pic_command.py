import asyncio
from typing import Tuple, Optional, Dict, Any

from src.plugin_system.base.base_command import BaseCommand
from src.common.logger import get_logger

from .api_clients import ApiClient
from .image_utils import ImageProcessor

logger = get_logger("pic_command")

class PicGenerationCommand(BaseCommand):
    """图生图Command组件，支持通过命令进行图生图，可选择特定模型"""

    # Command基本信息
    command_name = "pic_generation_command"
    command_description = "图生图命令，使用风格化提示词：/pic <风格>"
    command_pattern = r"(?:.*，说：\s*)?/pic\s+(?P<style>[\u4e00-\u9fff\w]+)"

    async def execute(self) -> Tuple[bool, Optional[str], bool]:
        """执行图生图命令"""
        logger.info(f"{self.log_prefix} 执行图生图命令")

        # 获取匹配的参数
        style_name = self.matched_groups.get("style", "").strip()

        if not style_name:
            await self.send_text("请指定风格，格式：/pic <风格>\n当前可用风格：cartoon, 卡通\n可在配置文件styles节和style_aliases节中添加更多风格")
            return False, "缺少风格参数", True

        # 从配置中获取Command组件使用的模型
        model_id = self.get_config("components.pic_command_model", "model1")

        # 获取模型配置
        model_config = self._get_model_config(model_id)
        if not model_config:
            await self.send_text(f"配置的模型 '{model_id}' 不存在，请检查配置文件")
            return False, "模型配置不存在", True

        # 获取风格化提示词（支持别名映射）
        actual_style_name = self._resolve_style_alias(style_name)
        style_prompt = self._get_style_prompt(actual_style_name)
        if not style_prompt:
            await self.send_text(f"风格 '{style_name}' 不存在\n当前可用风格：cartoon, 卡通\n可在配置文件styles节和style_aliases节中添加更多风格")
            return False, f"风格 '{style_name}' 不存在", True

        # 使用风格提示词作为描述
        final_description = style_prompt
        await self.send_text(f"使用风格：{style_name}")

        # 获取最近的图片作为输入图片
        image_processor = ImageProcessor(self)
        input_image_base64 = await image_processor.get_recent_image()

        if not input_image_base64:
            await self.send_text("未找到要处理的图片，请先发送一张图片")
            return False, "未找到输入图片", True

        # 显示开始信息
        await self.send_text(f"正在使用 {model_id} 模型进行 {style_name} 风格转换...")

        try:
            # 调用API客户端生成图片
            api_client = ApiClient(self)
            success, result = await api_client.generate_image(
                prompt=final_description,
                model_config=model_config,
                size=model_config.get("default_size", "1024x1024"),
                strength=0.7,  # 默认强度
                input_image_base64=input_image_base64
            )

            if success:
                # 处理结果
                if result.startswith(("iVBORw", "/9j/", "UklGR", "R0lGOD")):  # Base64
                    send_success = await self.send_image(result)
                    if send_success:
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
            await self.send_text(f"命令执行时发生错误：{str(e)[:100]}")
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

            response = requests.get(image_url, timeout=30)
            if response.status_code == 200:
                image_base64 = base64.b64encode(response.content).decode('utf-8')
                return True, image_base64
            else:
                return False, f"HTTP {response.status_code}"
        except Exception as e:
            return False, str(e)