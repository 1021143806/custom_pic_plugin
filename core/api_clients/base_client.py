"""API客户端基类"""
import asyncio
import base64
from typing import Dict, Any, Tuple, Optional
from src.common.logger import get_logger

logger = get_logger("pic_action")


class BaseApiClient:
    """API客户端基类"""

    # 子类需要设置的格式名称
    format_name: str = "base"

    def __init__(self, action_instance):
        self.action = action_instance
        self.log_prefix = action_instance.log_prefix

    def _get_proxy_config(self) -> Optional[Dict[str, Any]]:
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

    def _prepare_image_data_uri(self, image_base64: str) -> str:
        """准备图片的data URI格式

        Args:
            image_base64: base64编码的图片数据

        Returns:
            带有正确MIME类型前缀的data URI
        """
        if image_base64.startswith('data:image'):
            return image_base64

        # 检测图片格式
        if image_base64.startswith('/9j/'):
            return f"data:image/jpeg;base64,{image_base64}"
        elif image_base64.startswith('iVBORw'):
            return f"data:image/png;base64,{image_base64}"
        elif image_base64.startswith('UklGR'):
            return f"data:image/webp;base64,{image_base64}"
        elif image_base64.startswith('R0lGOD'):
            return f"data:image/gif;base64,{image_base64}"
        else:
            return f"data:image/jpeg;base64,{image_base64}"

    def _detect_mime_type(self, image_base64: str) -> str:
        """检测图片MIME类型

        Args:
            image_base64: base64编码的图片数据

        Returns:
            MIME类型字符串
        """
        # 移除data URI前缀（如果存在）
        clean_base64 = image_base64
        if ',' in image_base64:
            clean_base64 = image_base64.split(',')[1]

        if clean_base64.startswith('/9j/'):
            return "image/jpeg"
        elif clean_base64.startswith('iVBORw'):
            return "image/png"
        elif clean_base64.startswith('UklGR'):
            return "image/webp"
        elif clean_base64.startswith('R0lGOD'):
            return "image/gif"
        else:
            return "image/jpeg"  # 默认

    def _get_clean_base64(self, image_base64: str) -> str:
        """获取干净的base64数据（移除data URI前缀）

        Args:
            image_base64: 可能包含data URI前缀的base64数据

        Returns:
            纯base64数据
        """
        if ',' in image_base64:
            return image_base64.split(',')[1]
        return image_base64

    async def generate_image(
        self,
        prompt: str,
        model_config: Dict[str, Any],
        size: str,
        strength: float = None,
        input_image_base64: str = None,
        max_retries: int = 2
    ) -> Tuple[bool, str]:
        """生成图片的基础方法，带重试逻辑

        Args:
            prompt: 提示词
            model_config: 模型配置
            size: 图片尺寸
            strength: 图生图强度
            input_image_base64: 输入图片的base64编码
            max_retries: 最大重试次数

        Returns:
            (成功标志, 结果数据或错误信息)
        """
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"{self.log_prefix} API调用重试第 {attempt} 次")
                    await asyncio.sleep(1.0 * attempt)  # 渐进式等待

                logger.debug(f"{self.log_prefix} 开始API调用（尝试 {attempt + 1}/{max_retries + 1}）")

                # 调用具体实现
                success, result = await asyncio.to_thread(
                    self._make_request,
                    prompt=prompt,
                    model_config=model_config,
                    size=size,
                    strength=strength,
                    input_image_base64=input_image_base64
                )

                if success:
                    if attempt > 0:
                        logger.info(f"{self.log_prefix} API调用重试第 {attempt} 次成功")
                    return True, result

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

    def _make_request(
        self,
        prompt: str,
        model_config: Dict[str, Any],
        size: str,
        strength: float = None,
        input_image_base64: str = None
    ) -> Tuple[bool, str]:
        """具体的请求实现，子类必须实现此方法

        Args:
            prompt: 提示词
            model_config: 模型配置
            size: 图片尺寸
            strength: 图生图强度
            input_image_base64: 输入图片的base64编码

        Returns:
            (成功标志, 结果数据或错误信息)
        """
        raise NotImplementedError("子类必须实现 _make_request 方法")
