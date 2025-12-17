"""API客户端模块

支持多种图片生成API：
- OpenAI 格式 (OpenAI, 硅基流动, NewAPI等)
- Doubao 豆包格式
- Gemini 格式
- Modelscope 魔搭格式
- Shatangyun 砂糖云格式 (NovelAI)
- Mengyuai 梦羽AI格式
- Zai 格式 (Gemini转发)
"""

from .base_client import BaseApiClient
from .openai_client import OpenAIClient
from .doubao_client import DoubaoClient
from .gemini_client import GeminiClient
from .modelscope_client import ModelscopeClient
from .shatangyun_client import ShatangyunClient
from .mengyuai_client import MengyuaiClient
from .zai_client import ZaiClient

__all__ = [
    'BaseApiClient',
    'OpenAIClient',
    'DoubaoClient',
    'GeminiClient',
    'ModelscopeClient',
    'ShatangyunClient',
    'MengyuaiClient',
    'ZaiClient',
    'ApiClient',
    'get_client_class',
]


# API格式到客户端类的映射
CLIENT_MAPPING = {
    'openai': OpenAIClient,
    'doubao': DoubaoClient,
    'gemini': GeminiClient,
    'modelscope': ModelscopeClient,
    'shatangyun': ShatangyunClient,
    'mengyuai': MengyuaiClient,
    'zai': ZaiClient,
}


def get_client_class(api_format: str):
    """根据API格式获取对应的客户端类

    Args:
        api_format: API格式名称

    Returns:
        客户端类，如果不存在则返回OpenAIClient作为默认
    """
    return CLIENT_MAPPING.get(api_format.lower(), OpenAIClient)


class ApiClient:
    """统一的API客户端包装类

    根据模型配置中的format字段自动选择正确的客户端
    提供与BaseApiClient相同的接口
    """

    def __init__(self, action_instance):
        self.action = action_instance
        self._clients = {}  # 缓存客户端实例

    def _get_client(self, api_format: str):
        """获取指定格式的客户端实例（带缓存）"""
        if api_format not in self._clients:
            client_class = get_client_class(api_format)
            self._clients[api_format] = client_class(self.action)
        return self._clients[api_format]

    async def generate_image(
        self,
        prompt: str,
        model_config: dict,
        size: str,
        strength: float = None,
        input_image_base64: str = None,
        max_retries: int = 2
    ):
        """生成图片，自动选择正确的API客户端

        Args:
            prompt: 提示词
            model_config: 模型配置（必须包含format字段）
            size: 图片尺寸
            strength: 图生图强度
            input_image_base64: 输入图片的base64编码
            max_retries: 最大重试次数

        Returns:
            (成功标志, 结果数据或错误信息)
        """
        api_format = model_config.get("format", "openai")
        client = self._get_client(api_format)
        return await client.generate_image(
            prompt=prompt,
            model_config=model_config,
            size=size,
            strength=strength,
            input_image_base64=input_image_base64,
            max_retries=max_retries
        )
