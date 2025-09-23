import asyncio
import base64
import json
import urllib.request
import traceback
import re
import os
from typing import Optional, Tuple

from src.common.logger import get_logger

logger = get_logger("pic_action")

class ImageProcessor:
    """图片处理工具类"""

    def __init__(self, action_instance):
        self.action = action_instance
        self.log_prefix = action_instance.log_prefix

    async def get_recent_image(self) -> Optional[str]:
        """获取最近的图片消息"""
        try:
            logger.debug(f"{self.log_prefix} 开始获取图片消息")

            # 检查当前Action消息是否包含图片
            if self.action.has_action_message and self.action.action_message:
                logger.debug(f"{self.log_prefix} 检查action_message是否包含图片")

                # 1. 检查是否是回复消息，并尝试获取被回复的图片
                if self._is_reply_message():
                    logger.info(f"{self.log_prefix} 检测到回复消息，尝试获取被回复的图片")
                    reply_image = await self._get_image_from_reply()
                    if reply_image:
                        logger.info(f"{self.log_prefix} 从回复消息获取图片成功")
                        return reply_image

                # 2. 检查action_message中的图片信息
                images_data = None
                if isinstance(self.action.action_message, dict):
                    if "images" in self.action.action_message and self.action.action_message["images"]:
                        images_data = self.action.action_message["images"][0]
                else:
                    # DatabaseMessages 对象
                    if hasattr(self.action.action_message, 'images') and getattr(self.action.action_message, 'images', None):
                        images_list = getattr(self.action.action_message, 'images')
                        if images_list:
                            images_data = images_list[0] if isinstance(images_list, list) else images_list

                if images_data:
                    logger.info(f"{self.log_prefix} 从action_message获取图片")
                    return self._process_image_data(images_data)

                # 3. 检查message_content中的图片
                message_content = None
                if isinstance(self.action.action_message, dict):
                    if "message_content" in self.action.action_message:
                        message_content = self.action.action_message["message_content"]
                else:
                    # DatabaseMessages 对象
                    if hasattr(self.action.action_message, 'message_content'):
                        message_content = getattr(self.action.action_message, 'message_content', None)

                if message_content:
                    if isinstance(message_content, str) and self._is_image_data(message_content):
                        logger.info(f"{self.log_prefix} 从message_content获取图片")
                        return self._process_image_data(message_content)

            # 尝试从chat_stream获取最近的图片消息
            if self.action.chat_stream:
                logger.debug(f"{self.log_prefix} 尝试从chat_stream获取历史图片消息")

                try:
                    # 获取最近的消息历史
                    if hasattr(self.action.chat_stream, 'get_recent_messages'):
                        recent_messages = self.action.chat_stream.get_recent_messages(10)
                        logger.debug(f"{self.log_prefix} 获取到 {len(recent_messages)} 条历史消息")

                        for msg in reversed(recent_messages):
                            image_data = await self._extract_image_from_message(msg)
                            if image_data:
                                logger.info(f"{self.log_prefix} 从历史消息获取图片")
                                return image_data

                    # 尝试从消息存储获取
                    if hasattr(self.action.chat_stream, 'message_storage'):
                        storage = self.action.chat_stream.message_storage
                        if hasattr(storage, 'get_recent_messages'):
                            recent_messages = storage.get_recent_messages(10)
                            logger.debug(f"{self.log_prefix} 从存储获取到 {len(recent_messages)} 条消息")

                            for msg in reversed(recent_messages):
                                image_data = await self._extract_image_from_message(msg)
                                if image_data:
                                    logger.info(f"{self.log_prefix} 从存储消息获取图片")
                                    return image_data

                except Exception as e:
                    logger.debug(f"{self.log_prefix} 从chat_stream获取历史消息失败: {e}")

            # 最后尝试：使用插件系统的消息API
            try:
                from src.plugin_system.apis import message_api
                # 使用正确的API获取最近消息
                recent_messages = message_api.get_recent_messages(self.action.chat_id, hours=1.0, limit=20, filter_mai=True)
                logger.debug(f"{self.log_prefix} 从message_api获取到 {len(recent_messages)} 条消息")

                for msg in reversed(recent_messages):
                    image_data = await self._extract_image_from_message(msg)
                    if image_data:
                        logger.info(f"{self.log_prefix} 从message_api获取图片")
                        return image_data

            except Exception as e:
                logger.debug(f"{self.log_prefix} 使用message_api获取消息失败: {e}")

            logger.warning(f"{self.log_prefix} 未找到可用的图片消息")
            return None

        except Exception as e:
            logger.error(f"{self.log_prefix} 获取图片失败: {e!r}", exc_info=True)
            return None

    def _is_reply_message(self) -> bool:
        """检测当前消息是否是回复消息"""
        try:
            if not self.action.action_message:
                return False

            # 检查多种可能的回复消息字段
            potential_fields = [
                'raw_message', 'processed_plain_text', 'display_message',
                'message_content', 'content', 'text'
            ]

            if isinstance(self.action.action_message, dict):
                # 字典类型的action_message
                for field in potential_fields:
                    if field in self.action.action_message:
                        text = str(self.action.action_message[field])
                        # 检查是否包含回复格式的文本
                        if text and ('[回复' in text or 'reply' in text.lower() or '回复' in text):
                            logger.debug(f"{self.log_prefix} 在字段 {field} 中检测到回复消息格式")
                            return True

                # 检查是否有reply相关的字段
                reply_fields = ['reply_to', 'reply_message', 'quoted_message', 'reply']
                for field in reply_fields:
                    if field in self.action.action_message and self.action.action_message[field]:
                        logger.debug(f"{self.log_prefix} 检测到回复字段: {field}")
                        return True
            else:
                # DatabaseMessages 对象
                for field in potential_fields:
                    if hasattr(self.action.action_message, field):
                        text = str(getattr(self.action.action_message, field, ''))
                        # 检查是否包含回复格式的文本
                        if text and ('[回复' in text or 'reply' in text.lower() or '回复' in text):
                            logger.debug(f"{self.log_prefix} 在属性 {field} 中检测到回复消息格式")
                            return True

                # 检查是否有reply相关的属性
                reply_fields = ['reply_to', 'reply_message', 'quoted_message', 'reply']
                for field in reply_fields:
                    if hasattr(self.action.action_message, field) and getattr(self.action.action_message, field, None):
                        logger.debug(f"{self.log_prefix} 检测到回复属性: {field}")
                        return True

            return False

        except Exception as e:
            logger.debug(f"{self.log_prefix} 检测回复消息失败: {e}")
            return False

    async def _get_image_from_reply(self) -> Optional[str]:
        """从回复消息中获取被回复的图片"""
        try:
            if not self.action.action_message:
                return None

            # 1. 处理reply_to字段 - 这是最重要的
            reply_to = None
            if isinstance(self.action.action_message, dict):
                if 'reply_to' in self.action.action_message and self.action.action_message['reply_to']:
                    reply_to = self.action.action_message['reply_to']
            else:
                # DatabaseMessages 对象
                if hasattr(self.action.action_message, 'reply_to') and getattr(self.action.action_message, 'reply_to', None):
                    reply_to = getattr(self.action.action_message, 'reply_to')

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
                    # 获取更多历史消息来查找被回复的消息
                    recent_messages = message_api.get_recent_messages(self.action.chat_id, hours=2.0, limit=50, filter_mai=True)
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
            if isinstance(self.action.action_message, dict):
                for field in reply_fields:
                    if field in self.action.action_message and self.action.action_message[field]:
                        reply_data = self.action.action_message[field]
                        image_data = await self._extract_image_from_message(reply_data)
                        if image_data:
                            logger.info(f"{self.log_prefix} 从{field}字段获取回复图片")
                            return image_data
            else:
                # DatabaseMessages 对象
                for field in reply_fields:
                    if hasattr(self.action.action_message, field) and getattr(self.action.action_message, field, None):
                        reply_data = getattr(self.action.action_message, field)
                        image_data = await self._extract_image_from_message(reply_data)
                        if image_data:
                            logger.info(f"{self.log_prefix} 从{field}属性获取回复图片")
                            return image_data

            # 3. 解析回复格式的文本消息，提取被回复消息的ID或信息
            text_fields = ['processed_plain_text', 'display_message', 'raw_message', 'message_content']
            if isinstance(self.action.action_message, dict):
                for field in text_fields:
                    if field in self.action.action_message:
                        text = str(self.action.action_message[field])
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
                    if hasattr(self.action.action_message, field):
                        text = str(getattr(self.action.action_message, field, ''))
                        if '[回复' in text and '[图片]' in text:
                            logger.debug(f"{self.log_prefix} 在{field}属性中发现回复图片格式: {text[:100]}...")

                            # 尝试从文本中提取图片相关信息
                            image_data = await self._extract_base64_from_text(text)
                            if image_data:
                                logger.info(f"{self.log_prefix} 从回复文本中提取图片成功")
                                return image_data

            # 4. 作为备选方案，查找最近的图片消息（扩大搜索范围）
            try:
                from src.plugin_system.apis import message_api
                # 扩大搜索范围到100条消息，2小时内
                recent_messages = message_api.get_recent_messages(self.action.chat_id, hours=2.0, limit=100, filter_mai=True)
                logger.debug(f"{self.log_prefix} 扩大搜索范围，获取最近 {len(recent_messages)} 条消息查找图片")

                for msg in reversed(recent_messages):
                    # 跳过当前消息
                    current_msg_id = None
                    msg_id = None
                    is_picid = False

                    if hasattr(self.action.action_message, 'get'):
                        current_msg_id = self.action.action_message.get('message_id') or self.action.action_message.get('id')
                    else:
                        current_msg_id = getattr(self.action.action_message, 'message_id', None) or getattr(self.action.action_message, 'id', None)

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
                            logger.info(f"{self.log_prefix} 从扩大范围的历史消息中找到图片")
                            return image_data

            except Exception as e:
                logger.debug(f"{self.log_prefix} 扩大范围查找图片消息失败: {e}")

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

    async def _extract_image_from_message(self, message) -> Optional[str]:
        """从消息对象中提取图片数据"""
        try:
            if not message:
                return None

            # 检查消息是否包含图片标记
            if isinstance(message, dict):
                # 优先检查is_picid标记
                if message.get('is_picid', False):
                    # 查找图片相关的字段
                    potential_keys = [
                        'message_segment', 'raw_message', 'display_message',
                        'processed_plain_text', 'additional_config'
                    ]

                    for key in potential_keys:
                        if key in message and message[key]:
                            image_data = await self._extract_base64_from_text(str(message[key]))
                            if image_data:
                                logger.debug(f"{self.log_prefix} 从{key}字段提取到图片数据")
                                return image_data

                # 通用字段检查
                for key in ['images', 'image', 'content', 'message_content', 'data']:
                    if key in message and message[key]:
                        data = message[key]
                        if isinstance(data, list) and data:
                            data = data[0]
                        image_data = self._process_image_data(data)
                        if image_data:
                            return image_data

            # 如果是消息对象（DatabaseMessages）
            else:
                # 检查是否有图片标记
                if hasattr(message, 'is_picid') and getattr(message, 'is_picid', False):
                    # 尝试从消息段中获取图片
                    if hasattr(message, 'message_segment') and message.message_segment:
                        segment = message.message_segment
                        image_data = self._extract_image_from_segment(segment)
                        if image_data:
                            return image_data

                    # 从其他属性获取
                    for attr in ['raw_message', 'processed_plain_text', 'display_message', 'additional_config']:
                        if hasattr(message, attr):
                            text = getattr(message, attr)
                            if text:
                                image_data = await self._extract_base64_from_text(str(text))
                                if image_data:
                                    logger.debug(f"{self.log_prefix} 从{attr}属性提取到图片数据")
                                    return image_data

                # 尝试多种方式获取图片
                image_sources = [
                    getattr(message, 'images', None),
                    getattr(message, 'image', None),
                    getattr(message, 'content', None),
                    getattr(message, 'message_content', None),
                    getattr(message, 'data', None),
                ]

                for source in image_sources:
                    if source:
                        if isinstance(source, list) and source:
                            # 如果是列表，取第一个
                            image_data = self._process_image_data(source[0])
                            if image_data:
                                return image_data
                        else:
                            # 如果是单个数据
                            image_data = self._process_image_data(source)
                            if image_data:
                                return image_data

            return None

        except Exception as e:
            logger.debug(f"{self.log_prefix} 从消息提取图片失败: {e}")
            return None

    def _extract_image_from_segment(self, segment) -> Optional[str]:
        """从消息段中提取图片"""
        try:
            if not segment:
                return None

            # 如果是字典格式的段
            if isinstance(segment, dict):
                if segment.get('type') == 'image' and 'data' in segment:
                    return self._process_image_data(segment['data'])

            # 如果有data属性
            elif hasattr(segment, 'data'):
                segment_data = getattr(segment, 'data')
                if segment_data:
                    return self._process_image_data(segment_data)

            # 如果有type属性
            elif hasattr(segment, 'type'):
                segment_type = getattr(segment, 'type')
                if segment_type == 'image' and hasattr(segment, 'data'):
                    return self._process_image_data(getattr(segment, 'data'))

            return None

        except Exception as e:
            logger.debug(f"{self.log_prefix} 从消息段提取图片失败: {e}")
            return None

    async def _extract_base64_from_text(self, text: str) -> Optional[str]:
        """从文本中提取base64图片数据"""
        try:
            if not text:
                return None

            # 检查是否直接是base64图片数据
            if self._is_image_data(text):
                return self._process_image_data(text)

            # 检查是否包含picid格式 [picid:xxxxx]
            picid_pattern = r'\[picid:([a-f0-9\-]+)\]'
            picid_match = re.search(picid_pattern, text)
            if picid_match:
                picid = picid_match.group(1)
                logger.info(f"{self.log_prefix} 找到picid: {picid}")

                # 尝试通过picid获取图片数据
                image_data = await self._get_image_by_picid(picid)
                if image_data:
                    return image_data

            # 尝试从可能的JSON格式中提取
            try:
                data = json.loads(text)
                if isinstance(data, dict):
                    for key in ['data', 'base64', 'image', 'content']:
                        if key in data and data[key]:
                            result = self._process_image_data(data[key])
                            if result:
                                return result
                elif isinstance(data, list) and data:
                    for item in data:
                        result = await self._extract_base64_from_text(str(item))
                        if result:
                            return result
            except (json.JSONDecodeError, TypeError):
                pass

            return None

        except Exception as e:
            logger.debug(f"{self.log_prefix} 从文本提取base64失败: {e}")
            return None

    async def _get_image_by_picid(self, picid: str) -> Optional[str]:
        """通过picid获取图片的base64数据"""
        try:
            # 尝试从图片管理器获取图片
            from src.chat.utils.utils_image import get_image_manager

            image_manager = get_image_manager()
            if hasattr(image_manager, 'get_image_by_id'):
                image_data = await image_manager.get_image_by_id(picid)
                if image_data:
                    return image_data

            # 尝试从数据库直接获取
            from src.common.database.database_model import Images
            try:
                # 查找对应的图片记录
                image_record = Images.select().where(Images.id == picid).first()
                if image_record and image_record.path:
                    # 从路径读取图片并转换为base64
                    from src.chat.utils.utils_image import image_path_to_base64
                    base64_data = image_path_to_base64(image_record.path)
                    if base64_data:
                        logger.info(f"{self.log_prefix} 通过picid从数据库获取图片成功")
                        return base64_data
            except Exception as e:
                logger.debug(f"{self.log_prefix} 从数据库获取图片失败: {e}")

            # 如果上述方法都失败，尝试构造路径
            # MaiBot可能将图片存储在特定目录
            possible_paths = [
                f"/tmp/images/{picid}",
                f"/tmp/images/{picid}.jpg",
                f"/tmp/images/{picid}.png",
                f"data/images/{picid}",
                f"data/images/{picid}.jpg",
                f"data/images/{picid}.png",
                f"images/{picid}",
                f"images/{picid}.jpg",
                f"images/{picid}.png"
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    from src.chat.utils.utils_image import image_path_to_base64
                    base64_data = image_path_to_base64(path)
                    if base64_data:
                        logger.info(f"{self.log_prefix} 通过路径 {path} 获取图片成功")
                        return base64_data

            logger.warning(f"{self.log_prefix} 无法通过picid {picid} 获取图片数据")
            return None

        except Exception as e:
            logger.error(f"{self.log_prefix} 通过picid获取图片异常: {e!r}")
            return None

    def _process_image_data(self, data) -> Optional[str]:
        """处理图片数据，统一转换为base64格式"""
        try:
            if not data:
                return None

            if isinstance(data, str):
                if self._is_image_data(data):
                    if data.startswith('data:image'):
                        base64_data = data.split(',', 1)[1] if ',' in data else data
                        logger.debug(f"{self.log_prefix} 提取data URL中的base64数据，长度: {len(base64_data)}")
                        return base64_data
                    elif data.startswith(('iVBORw', '/9j/', 'UklGR', 'R0lGOD')):
                        logger.debug(f"{self.log_prefix} 获取到base64图片数据，长度: {len(data)}")
                        return data

            elif isinstance(data, dict):
                for key in ['data', 'base64', 'content', 'url']:
                    if key in data and data[key]:
                        result = self._process_image_data(data[key])
                        if result:
                            return result

            return None

        except Exception as e:
            logger.debug(f"{self.log_prefix} 处理图片数据失败: {e}")
            return None

    def _is_image_data(self, data: str) -> bool:
        """检查字符串是否包含图片数据"""
        if not isinstance(data, str):
            return False
        return data.startswith(('data:image', 'iVBORw', '/9j/', 'UklGR', 'R0lGOD'))

    def validate_image_size(self, image_size: str) -> bool:
        """验证图片尺寸格式"""
        try:
            width, height = map(int, image_size.split("x"))
            return 100 <= width <= 10000 and 100 <= height <= 10000
        except (ValueError, TypeError):
            return False

    def download_and_encode_base64(self, image_url: str) -> Tuple[bool, str]:
        """下载图片并将其编码为Base64字符串"""
        logger.info(f"{self.log_prefix} (B64) 下载并编码图片: {image_url[:70]}...")
        try:
            with urllib.request.urlopen(image_url, timeout=600) as response:
                if response.status == 200:
                    image_bytes = response.read()
                    base64_encoded_image = base64.b64encode(image_bytes).decode("utf-8")
                    logger.info(f"{self.log_prefix} (B64) 图片下载编码完成. Base64长度: {len(base64_encoded_image)}")
                    return True, base64_encoded_image
                else:
                    error_msg = f"下载图片失败 (状态: {response.status})"
                    logger.error(f"{self.log_prefix} (B64) {error_msg} URL: {image_url}")
                    return False, error_msg
        except Exception as e: 
            logger.error(f"{self.log_prefix} (B64) 下载或编码时错误: {e!r}", exc_info=True)
            traceback.print_exc()
            return False, f"下载或编码图片时发生错误: {str(e)[:100]}"

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
            logger.error(f"{self.log_prefix} 处理API响应失败: {e!r}")
            return None