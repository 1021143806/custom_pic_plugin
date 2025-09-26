import asyncio
import base64
import json
import urllib.request
import traceback
import re
import os
from functools import lru_cache
from typing import Optional, Tuple, Dict, Any, Union, List
from concurrent.futures import ThreadPoolExecutor

from src.common.logger import get_logger
from maim_message import Seg

logger = get_logger("pic_action")

class ImageProcessor:
    """图片处理工具类"""

    # 图片格式检测模式
    _image_format_patterns = {
        'jpeg': ['/9j/', '\xff\xd8\xff'],
        'png': ['iVBORw', '\x89PNG'],
        'webp': ['UklGR', 'RIFF'],
        'gif': ['R0lGOD', 'GIF8']
    }

    def __init__(self, action_instance):
        self.action = action_instance
        self.log_prefix = action_instance.log_prefix

        # 使用实例级别的失败缓存，避免跨实例状态共享问题
        self._failed_picids_cache = {}
        self._max_failed_cache_size = 500

    def _is_picid_failed(self, picid: str) -> bool:
        """检查picid是否在失败缓存中"""
        return picid in self._failed_picids_cache

    def _mark_picid_failed(self, picid: str):
        """将picid标记为失败，使用LRU缓存机制"""
        import time
        self._failed_picids_cache[picid] = time.time()

        # LRU清理机制
        if len(self._failed_picids_cache) > self._max_failed_cache_size:
            # 按时间排序，移除最旧的条目
            sorted_items = sorted(self._failed_picids_cache.items(), key=lambda x: x[1])
            items_to_remove = len(sorted_items) - self._max_failed_cache_size // 2
            for i in range(items_to_remove):
                del self._failed_picids_cache[sorted_items[i][0]]

    def _is_action_component(self) -> bool:
        """判断是否为Action组件"""
        return hasattr(self.action, 'has_action_message')

    def _is_command_component(self) -> bool:
        """判断是否为Command组件"""
        return hasattr(self.action, 'message')

    async def get_recent_image(self) -> Optional[str]:
        """获取最近的图片消息，支持多种组件类型"""
        try:
            logger.debug(f"{self.log_prefix} 开始获取图片消息")

            # 方法1：从当前消息的message_segment中检索（最优先）
            message_segments = None

            # 兼容Action和Command组件
            if hasattr(self.action, 'message') and hasattr(self.action.message, 'message_segment'):
                # Command组件
                message_segments = self.action.message.message_segment
            elif hasattr(self.action, 'action_message') and hasattr(self.action.action_message, 'message_segment'):
                # Action组件
                message_segments = self.action.action_message.message_segment

            if message_segments:
                # 使用emoji插件的检索功能
                emoji_base64_list = self.find_and_return_emoji_in_message(message_segments)
                if emoji_base64_list:
                    logger.info(f"{self.log_prefix} 在当前消息中找到 {len(emoji_base64_list)} 张图片")
                    return emoji_base64_list[0]  # 返回第一张图片

            # 方法2：从历史消息中查找（作为后备）
            try:
                from src.plugin_system.apis import message_api

                # 获取chat_id
                chat_id = self._get_chat_id()
                if chat_id:
                    # 获取最近的消息
                    recent_messages = message_api.get_recent_messages(chat_id, hours=1.0, limit=15, filter_mai=True)
                    logger.debug(f"{self.log_prefix} 从历史消息获取到 {len(recent_messages)} 条消息")

                    for msg in reversed(recent_messages):
                        # 检查消息是否包含图片标记
                        is_picid = False
                        if isinstance(msg, dict):
                            is_picid = msg.get('is_picid', False)
                        else:
                            is_picid = getattr(msg, 'is_picid', False)

                        if is_picid:
                            # 尝试从消息段中提取
                            if hasattr(msg, 'message_segment') and msg.message_segment:
                                emoji_base64_list = self.find_and_return_emoji_in_message(msg.message_segment)
                                if emoji_base64_list:
                                    logger.info(f"{self.log_prefix} 从历史消息中找到图片")
                                    return emoji_base64_list[0]

            except Exception as e:
                logger.debug(f"{self.log_prefix} 从历史消息获取图片失败: {e}")

            logger.warning(f"{self.log_prefix} 未找到可用的图片消息")
            return None

        except Exception as e:
            logger.error(f"{self.log_prefix} 获取图片失败: {e!r}", exc_info=True)
            return None

    def _get_action_message(self) -> Optional[Any]:
        """获取action_message对象，兼容Action和Command"""
        if hasattr(self.action, 'has_action_message') and self.action.has_action_message:
            # Action组件
            return self.action.action_message
        elif hasattr(self.action, 'message') and hasattr(self.action.message, 'message_recv'):
            # Command组件，使用message.message_recv作为action_message
            return self.action.message.message_recv
        return None

    def _get_chat_stream(self) -> Optional[Any]:
        """获取chat_stream对象，兼容Action和Command"""
        if hasattr(self.action, 'chat_stream') and self.action.chat_stream:
            # Action组件
            return self.action.chat_stream
        elif hasattr(self.action, 'message') and hasattr(self.action.message, 'chat_stream'):
            # Command组件
            return self.action.message.chat_stream
        return None

    def _get_chat_id(self) -> Optional[str]:
        """获取chat_id，兼容Action和Command"""
        if hasattr(self.action, 'chat_id'):
            # Action组件
            return self.action.chat_id

        chat_stream = self._get_chat_stream()
        if chat_stream and hasattr(chat_stream, 'stream_id'):
            return chat_stream.stream_id
        return None

    def _process_image_data(self, data) -> Optional[str]:
        """处理图片数据，统一转换为base64格式"""
        try:
            if not data:
                return None

            # 如果是字符串类型
            if isinstance(data, str):
                # 检查是否是有效的base64图片数据
                if self._is_image_data(data):
                    return data
                # 如果不是，可能需要其他处理
                return None

            # 如果是字典类型，尝试提取内部数据
            if isinstance(data, dict):
                for key in ['data', 'base64', 'content', 'image']:
                    if key in data and data[key]:
                        result = self._process_image_data(data[key])
                        if result:
                            return result

            # 如果是字节类型，转换为base64
            if isinstance(data, bytes):
                try:
                    return base64.b64encode(data).decode('utf-8')
                except Exception as e:
                    logger.debug(f"{self.log_prefix} 字节数据转base64失败: {e}")
                    return None

            return None

        except Exception as e:
            logger.debug(f"{self.log_prefix} 处理图片数据失败: {str(e)[:50]}")
            return None

    def _is_image_data(self, data: str) -> bool:
        """检查字符串是否是有效的base64图片数据"""
        try:
            if not isinstance(data, str) or len(data) < 100:
                return False

            # 检查是否包含base64图片前缀
            if any(prefix in data[:50] for prefix in ['data:image/', '/9j/', 'iVBOR', 'UklGR', 'R0lGO']):
                return True

            # 检查base64格式特征
            if len(data) % 4 == 0 and all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=' for c in data[:100]):
                # 尝试解码前几个字符看是否是图片格式
                try:
                    decoded_start = base64.b64decode(data[:100])
                    for format_name, patterns in self._image_format_patterns.items():
                        for pattern in patterns:
                            if isinstance(pattern, str) and decoded_start.startswith(pattern.encode()):
                                return True
                            elif isinstance(pattern, bytes) and decoded_start.startswith(pattern):
                                return True
                except Exception:
                    pass

            return False

        except Exception:
            return False

    def validate_image_size(self, image_size: str) -> bool:
        """验证图片尺寸格式"""
        try:
            width, height = map(int, image_size.split("x"))
            return 100 <= width <= 10000 and 100 <= height <= 10000
        except (ValueError, TypeError):
            return False

    def download_and_encode_base64(self, image_url: str) -> Tuple[bool, str]:
        """下载图片并将其编码为Base64字符串"""
        logger.info(f"{self.log_prefix} (B64) 下载并编码图片: {image_url[:50]}...")
        try:
            with urllib.request.urlopen(image_url, timeout=600) as response:
                if response.status == 200:
                    image_bytes = response.read()
                    base64_encoded_image = base64.b64encode(image_bytes).decode("utf-8")
                    logger.info(f"{self.log_prefix} (B64) 图片下载编码完成. Base64长度: {len(base64_encoded_image)}")
                    return True, base64_encoded_image
                else:
                    error_msg = f"下载图片失败 (状态: {response.status})"
                    logger.error(f"{self.log_prefix} (B64) {error_msg} URL: {image_url[:30]}...")
                    return False, error_msg
        except Exception as e:
            logger.error(f"{self.log_prefix} (B64) 下载或编码时错误: {e!r}", exc_info=True)
            traceback.print_exc()
            return False, f"下载或编码图片时发生错误: {str(e)[:50]}"

    def process_api_response(self, result) -> Optional[str]:
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
            logger.error(f"{self.log_prefix} 处理API响应失败: {str(e)[:50]}")
            return None

    async def _is_reply_message(self, action_message) -> bool:
        """检测Action组件的消息是否是回复消息"""
        try:
            if not action_message:
                return False

            # 检查结构化的回复字段
            reply_fields = ['reply_to', 'reply_message', 'quoted_message', 'reply']
            if isinstance(action_message, dict):
                for field in reply_fields:
                    if field in action_message and action_message[field]:
                        logger.debug(f"{self.log_prefix} 检测到回复字段: {field} = {action_message[field]}")
                        return True
            else:
                # DatabaseMessages 对象
                for field in reply_fields:
                    reply_value = getattr(action_message, field, None)
                    if reply_value:
                        logger.debug(f"{self.log_prefix} 检测到回复属性: {field} = {reply_value}")
                        return True

            # 检查文本内容中的回复格式
            text_fields = ['processed_plain_text', 'display_message', 'raw_message', 'message_content']

            if isinstance(action_message, dict):
                for field in text_fields:
                    if field in action_message:
                        text = str(action_message[field])
                        if text and '[回复' in text and ']' in text:
                            logger.debug(f"{self.log_prefix} 在字段 {field} 中检测到回复消息格式")
                            return True
            else:
                # DatabaseMessages 对象
                for field in text_fields:
                    text = str(getattr(action_message, field, ''))
                    if text and '[回复' in text and ']' in text:
                        logger.debug(f"{self.log_prefix} 在属性 {field} 中检测到回复消息格式")
                        return True

            return False

        except Exception as e:
            logger.debug(f"{self.log_prefix} 检测回复消息失败: {e}")
            return False

    async def _get_image_from_reply(self, action_message) -> Optional[str]:
        """从Action组件的回复消息中获取被回复的图片"""
        try:
            if not action_message:
                return None

            # 1. 处理reply_to字段
            reply_to = None
            if isinstance(action_message, dict):
                if 'reply_to' in action_message and action_message['reply_to']:
                    reply_to = action_message['reply_to']
            else:
                # DatabaseMessages 对象
                reply_to = getattr(action_message, 'reply_to', None)

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

                    chat_id = self._get_chat_id()
                    if chat_id:
                        # 获取更多历史消息来查找被回复的消息
                        recent_messages = message_api.get_recent_messages(chat_id, hours=2.0, limit=50, filter_mai=True)
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
            if isinstance(action_message, dict):
                for field in reply_fields:
                    if field in action_message and action_message[field]:
                        reply_data = action_message[field]
                        image_data = await self._extract_image_from_message(reply_data)
                        if image_data:
                            logger.info(f"{self.log_prefix} 从{field}字段获取回复图片")
                            return image_data
            else:
                # DatabaseMessages 对象
                for field in reply_fields:
                    reply_data = getattr(action_message, field, None)
                    if reply_data:
                        image_data = await self._extract_image_from_message(reply_data)
                        if image_data:
                            logger.info(f"{self.log_prefix} 从{field}属性获取回复图片")
                            return image_data

            # 3. 解析回复格式的文本消息，提取被回复消息的ID或信息
            text_fields = ['processed_plain_text', 'display_message', 'raw_message', 'message_content']
            if isinstance(action_message, dict):
                for field in text_fields:
                    if field in action_message:
                        text = str(action_message[field])
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
                    text = str(getattr(action_message, field, ''))
                    if '[回复' in text and '[图片]' in text:
                        logger.debug(f"{self.log_prefix} 在{field}属性中发现回复图片格式: {text[:100]}...")

                        # 尝试从文本中提取图片相关信息
                        image_data = await self._extract_base64_from_text(text)
                        if image_data:
                            logger.info(f"{self.log_prefix} 从回复文本中提取图片成功")
                            return image_data

            # 4. 作为备选方案，查找最近的图片消息（但要确保时间匹配）
            try:
                from src.plugin_system.apis import message_api

                chat_id = self._get_chat_id()
                if chat_id:
                    # 限制搜索范围到30条消息，30分钟内，确保时效性
                    recent_messages = message_api.get_recent_messages(chat_id, hours=0.5, limit=30, filter_mai=True)
                    logger.debug(f"{self.log_prefix} 限制搜索范围，获取最近 {len(recent_messages)} 条消息查找图片")

                    for msg in reversed(recent_messages):
                        # 跳过当前消息
                        current_msg_id = None
                        msg_id = None
                        is_picid = False

                        if isinstance(action_message, dict):
                            current_msg_id = action_message.get('message_id') or action_message.get('id')
                        else:
                            current_msg_id = getattr(action_message, 'message_id', None) or getattr(action_message, 'id', None)

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
                                logger.warning(f"{self.log_prefix} 使用备选方案：从最近历史消息中获取图片，可能不是被回复的原图")
                                return image_data

            except Exception as e:
                logger.debug(f"{self.log_prefix} 限制范围查找图片消息失败: {e}")

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

    def find_and_return_emoji_in_message(self, message_segments) -> List[str]:
        """从消息中查找并返回表情包/图片的base64数据列表 (来自emoji_manage插件)"""
        emoji_base64_list = []

        # 处理单个Seg对象的情况
        if isinstance(message_segments, Seg):
            if message_segments.type == "emoji":
                emoji_base64_list.append(message_segments.data)
            elif message_segments.type == "image":
                # 假设图片数据是base64编码的
                emoji_base64_list.append(message_segments.data)
            elif message_segments.type == "seglist":
                # 递归处理嵌套的Seg列表
                emoji_base64_list.extend(self.find_and_return_emoji_in_message(message_segments.data))
            return emoji_base64_list

        # 处理Seg列表的情况
        for seg in message_segments:
            if seg.type == "emoji":
                emoji_base64_list.append(seg.data)
            elif seg.type == "image":
                # 假设图片数据是base64编码的
                emoji_base64_list.append(seg.data)
            elif seg.type == "seglist":
                # 递归处理嵌套的Seg列表
                emoji_base64_list.extend(self.find_and_return_emoji_in_message(seg.data))
        return emoji_base64_list

    async def _extract_image_from_message(self, message) -> Optional[str]:
        """从消息中提取图片数据"""
        try:
            if not message:
                return None

            # 如果消息有message_segment，直接从中提取
            message_segment = None
            if isinstance(message, dict):
                message_segment = message.get('message_segment')
            else:
                message_segment = getattr(message, 'message_segment', None)

            if message_segment:
                emoji_base64_list = self.find_and_return_emoji_in_message(message_segment)
                if emoji_base64_list:
                    return emoji_base64_list[0]

            return None

        except Exception as e:
            logger.debug(f"{self.log_prefix} 从消息提取图片失败: {e}")
            return None

    async def _extract_base64_from_text(self, text: str) -> Optional[str]:
        """从文本中提取base64图片数据"""
        try:
            if not text:
                return None

            # 尝试匹配base64数据模式
            import re

            # 匹配data:image/格式的base64
            data_url_pattern = r'data:image/[^;]+;base64,([A-Za-z0-9+/=]+)'
            match = re.search(data_url_pattern, text)
            if match:
                return match.group(1)

            # 匹配纯base64数据（长度较长的情况）
            base64_pattern = r'([A-Za-z0-9+/]{100,}={0,2})'
            matches = re.findall(base64_pattern, text)
            for match in matches:
                if self._is_image_data(match):
                    return match

            return None

        except Exception as e:
            logger.debug(f"{self.log_prefix} 从文本提取base64失败: {e}")
            return None
