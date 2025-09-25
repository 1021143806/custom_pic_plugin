import asyncio
import base64
import json
import urllib.request
import traceback
import re
import os
from functools import lru_cache
from typing import Optional, Tuple, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor

from src.common.logger import get_logger

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
        """获取最近的图片消息"""
        try:
            logger.debug(f"{self.log_prefix} 开始获取图片消息")

            # Command组件：直接从message.message_segment获取
            if self._is_command_component():
                logger.debug(f"{self.log_prefix} Command组件：检查message_segment")
                return await self._get_image_from_command()

            # Action组件：从action_message获取
            if self._is_action_component():
                logger.debug(f"{self.log_prefix} Action组件：检查action_message")
                return await self._get_image_from_action()

            logger.warning(f"{self.log_prefix} 无法识别组件类型")
            return None

        except Exception as e:
            logger.error(f"{self.log_prefix} 获取图片失败: {e!r}", exc_info=True)
            return None

    async def _get_image_from_command(self) -> Optional[str]:
        """从Command组件获取图片"""
        try:
            # 首先检查当前消息的message_segment
            message_segments = self.action.message.message_segment
            if message_segments:
                image_data = await self._extract_image_from_segments(message_segments)
                if image_data:
                    logger.info(f"{self.log_prefix} 从Command组件的message_segment获取图片成功")
                    return image_data

            # 如果当前消息没有图片，搜索历史消息
            logger.info(f"{self.log_prefix} Command组件：当前消息无图片，搜索历史图片")
            return await self._get_image_from_history()

        except Exception as e:
            logger.error(f"{self.log_prefix} 从Command组件获取图片失败: {e!r}")
            return None

    async def _get_image_from_action(self) -> Optional[str]:
        """从Action组件获取图片"""
        try:
            # 检查是否有action_message
            if not self.action.has_action_message:
                logger.info(f"{self.log_prefix} Action组件：无action_message，跳过历史搜索")
                return None

            action_message = self.action.action_message

            # 首先检查是否是回复消息
            if await self._is_reply_message(action_message):
                logger.info(f"{self.log_prefix} Action组件：检测到回复消息，尝试获取被回复的图片")
                reply_image = await self._get_image_from_reply(action_message)
                if reply_image:
                    logger.info(f"{self.log_prefix} Action组件：从回复消息获取图片成功")
                    return reply_image
                else:
                    logger.info(f"{self.log_prefix} Action组件：回复消息中未找到图片，继续检查当前消息")

            # 检查action_message中的图片信息
            if isinstance(action_message, dict):
                # 字典格式
                if "images" in action_message and action_message["images"]:
                    images_data = action_message["images"][0]
                    processed_data = self._process_image_data(images_data)
                    if processed_data:
                        logger.info(f"{self.log_prefix} 从action_message获取图片")
                        return processed_data

                if "message_content" in action_message:
                    message_content = action_message["message_content"]
                    if isinstance(message_content, str) and self._is_image_data(message_content):
                        logger.info(f"{self.log_prefix} 从message_content获取图片")
                        return self._process_image_data(message_content)
            else:
                # DatabaseMessages对象
                images_list = getattr(action_message, 'images', None)
                if images_list:
                    images_data = images_list[0] if isinstance(images_list, list) else images_list
                    processed_data = self._process_image_data(images_data)
                    if processed_data:
                        logger.info(f"{self.log_prefix} 从action_message获取图片")
                        return processed_data

                message_content = getattr(action_message, 'message_content', None)
                if message_content and isinstance(message_content, str) and self._is_image_data(message_content):
                    logger.info(f"{self.log_prefix} 从message_content获取图片")
                    return self._process_image_data(message_content)

            # Action组件不搜索历史图片（用于文生图场景）
            logger.info(f"{self.log_prefix} Action组件：当前消息无图片，认为是文生图场景")
            return None

        except Exception as e:
            logger.error(f"{self.log_prefix} 从Action组件获取图片失败: {e!r}")
            return None

    async def _get_image_from_history(self) -> Optional[str]:
        """从历史消息中获取图片"""
        try:
            # 通过chat_stream获取
            chat_stream = self._get_chat_stream()
            if chat_stream:
                logger.debug(f"{self.log_prefix} 尝试从chat_stream获取历史图片消息")

                try:
                    # 获取最近的消息历史
                    if hasattr(chat_stream, 'get_recent_messages'):
                        recent_messages = chat_stream.get_recent_messages(10)
                        logger.debug(f"{self.log_prefix} 获取到 {len(recent_messages)} 条历史消息")

                        for msg in reversed(recent_messages):
                            image_data = await self._extract_image_from_message(msg)
                            if image_data:
                                logger.info(f"{self.log_prefix} 从历史消息获取图片")
                                return image_data

                    # 尝试从消息存储获取
                    message_storage = getattr(chat_stream, 'message_storage', None)
                    if message_storage and hasattr(message_storage, 'get_recent_messages'):
                        recent_messages = message_storage.get_recent_messages(10)
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

                chat_id = self._get_chat_id()
                if chat_id:
                    recent_messages = message_api.get_recent_messages(chat_id, hours=1.0, limit=20, filter_mai=True)
                    logger.debug(f"{self.log_prefix} 从message_api获取到 {len(recent_messages)} 条消息")

                    for msg in reversed(recent_messages):
                        image_data = await self._extract_image_from_message(msg)
                        if image_data:
                            logger.info(f"{self.log_prefix} 从message_api获取图片")
                            return image_data

            except Exception as e:
                logger.debug(f"{self.log_prefix} 使用message_api获取消息失败: {e}")

            return None

        except Exception as e:
            logger.error(f"{self.log_prefix} 从历史消息获取图片失败: {e!r}")
            return None

    def _get_chat_stream(self):
        """获取chat_stream对象"""
        if self._is_action_component():
            return self.action.chat_stream
        elif self._is_command_component():
            return self.action.message.chat_stream
        return None

    def _get_chat_id(self) -> Optional[str]:
        """获取chat_id"""
        if self._is_action_component():
            return self.action.chat_id
        elif self._is_command_component():
            chat_stream = self.action.message.chat_stream
            return chat_stream.stream_id if chat_stream else None
        return None

    async def _extract_image_from_segments(self, message_segments) -> Optional[str]:
        """从message_segment中提取图片数据"""
        try:
            if not message_segments:
                return None

            logger.debug(f"{self.log_prefix} 处理message_segment: {type(message_segments)}")

            # 导入Seg类型
            try:
                from maim_message import Seg
            except ImportError:
                logger.debug(f"{self.log_prefix} 无法导入Seg类，使用通用处理")
                Seg = None

            # 处理单个Seg对象的情况
            if Seg and isinstance(message_segments, Seg):
                if message_segments.type == "emoji":
                    return message_segments.data
                elif message_segments.type == "image":
                    return self._process_image_data(message_segments.data)
                elif message_segments.type == "seglist":
                    return await self._extract_image_from_segments(message_segments.data)

            # 处理Seg列表的情况
            elif hasattr(message_segments, '__iter__'):
                try:
                    for seg in message_segments:
                        if Seg and isinstance(seg, Seg):
                            if seg.type == "emoji":
                                return seg.data
                            elif seg.type == "image":
                                return self._process_image_data(seg.data)
                            elif seg.type == "seglist":
                                nested_result = await self._extract_image_from_segments(seg.data)
                                if nested_result:
                                    return nested_result
                        else:
                            # 处理非Seg对象的情况
                            seg_type = getattr(seg, 'type', None)
                            if seg_type in ["emoji", "image"]:
                                seg_data = getattr(seg, 'data', None)
                                if seg_data:
                                    processed_data = self._process_image_data(seg_data)
                                    if processed_data:
                                        return processed_data
                except TypeError:
                    # 如果不可迭代，尝试直接处理
                    seg_type = getattr(message_segments, 'type', None)
                    seg_data = getattr(message_segments, 'data', None)
                    if seg_type in ["emoji", "image"] and seg_data:
                        return self._process_image_data(seg_data)

            # 处理字典格式的message_segment
            if isinstance(message_segments, dict):
                if message_segments.get('type') in ['image', 'emoji'] and 'data' in message_segments:
                    return self._process_image_data(message_segments['data'])
                if 'segments' in message_segments:
                    return await self._extract_image_from_segments(message_segments['segments'])

            # 处理列表格式
            if isinstance(message_segments, list):
                for item in message_segments:
                    result = await self._extract_image_from_segments(item)
                    if result:
                        return result

            logger.debug(f"{self.log_prefix} message_segment中未找到图片数据")
            return None

        except Exception as e:
            logger.debug(f"{self.log_prefix} 从message_segment提取图片失败: {str(e)[:50]}")
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
                if getattr(message, 'is_picid', False):
                    # 尝试从消息段中获取图片
                    message_segment = getattr(message, 'message_segment', None)
                    if message_segment:
                        image_data = self._extract_image_from_segment(message_segment)
                        if image_data:
                            return image_data

                    # 从其他属性获取
                    for attr in ['raw_message', 'processed_plain_text', 'display_message', 'additional_config']:
                        text = getattr(message, attr, None)
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
                            image_data = self._process_image_data(source[0])
                            if image_data:
                                return image_data
                        else:
                            image_data = self._process_image_data(source)
                            if image_data:
                                return image_data

            return None

        except Exception as e:
            logger.debug(f"{self.log_prefix} 从消息提取图片失败: {str(e)[:50]}")
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
            logger.debug(f"{self.log_prefix} 从消息段提取图片失败: {str(e)[:50]}")
            return None

    async def _extract_base64_from_text(self, text: str) -> Optional[str]:
        """从文本中提取base64图片数据"""
        try:
            if not text:
                return None

            # 检查是否直接是base64图片数据
            if self._is_image_data(text):
                return self._process_image_data(text)

            # 增强的picid格式匹配，支持更多变体
            picid_patterns = [
                r'\[picid:([a-f0-9\-]+)\]',  # 标准格式
                r'\[pic:([a-f0-9\-]+)\]',   # 简化格式
                r'\[image:([a-f0-9\-]+)\]', # 其他变体
                r'\[img:([a-f0-9\-]+)\]',   # 简称变体
                r'picid:([a-f0-9\-]+)',      # 无括号版本
                r'pic_id[:：]([a-f0-9\-]+)', # 下划线版本
                r'image_id[:：]([a-f0-9\-]+)' # 其他变体
            ]

            for pattern in picid_patterns:
                picid_match = re.search(pattern, text, re.IGNORECASE)
                if picid_match:
                    picid = picid_match.group(1)
                    logger.info(f"{self.log_prefix} 找到picid: {picid[:8]}...")

                    image_data = await self._get_image_by_picid(picid)
                    if image_data:
                        return image_data
                    else:
                        logger.warning(f"{self.log_prefix} picid {picid[:8]}... 无法获取图片数据")
                        return None

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
            logger.debug(f"{self.log_prefix} 从文本提取base64失败: {str(e)[:50]}")
            return None

    async def _get_image_by_picid(self, picid: str) -> Optional[str]:
        """通过picid获取图片的base64数据"""
        try:
            # 检查picid格式有效性
            if not picid or len(picid) < 10:
                logger.warning(f"{self.log_prefix} picid格式无效: {picid}")
                return None

            # 检查是否已经尝试过且失败的picid
            if self._is_picid_failed(picid):
                logger.debug(f"{self.log_prefix} picid {picid[:8]}... 已在失败缓存中，跳过")
                return None

            # 优先尝试从图片管理器获取最新图片
            try:
                from src.chat.utils.utils_image import get_image_manager
                image_manager = get_image_manager()
                if hasattr(image_manager, 'get_image_by_id'):
                    image_data = await image_manager.get_image_by_id(picid)
                    if image_data:
                        logger.info(f"{self.log_prefix} 通过图片管理器获取picid {picid} 成功 (最新)")
                        return image_data
            except Exception as e:
                logger.debug(f"{self.log_prefix} 图片管理器获取失败: {e}")

            # 次优选择：从数据库获取当前有效路径
            try:
                from src.common.database.database_model import Images
                image_record = Images.select().where(Images.id == picid).first()
                if image_record and hasattr(image_record, 'path') and image_record.path:
                    if os.path.exists(image_record.path):
                        try:
                            from src.chat.utils.utils_image import image_path_to_base64
                            base64_data = image_path_to_base64(image_record.path)
                            if base64_data:
                                logger.info(f"{self.log_prefix} 通过picid从数据库获取图片成功: {image_record.path}")
                                return base64_data
                        except (FileNotFoundError, IOError) as e:
                            logger.debug(f"{self.log_prefix} 读取图片文件失败: {e}")
                    else:
                        logger.debug(f"{self.log_prefix} 数据库路径文件不存在: {image_record.path}")
                else:
                    logger.debug(f"{self.log_prefix} 数据库中未找到picid {picid} 或路径为空")
            except Exception as e:
                logger.debug(f"{self.log_prefix} 从数据库获取图片失败: {e}")

            # 最后尝试：文件系统路径搜索（可能是历史缓存）
            try:
                base64_data = await self._check_paths_concurrently(picid)
                if base64_data:
                    logger.warning(f"{self.log_prefix} 通过文件系统获取图片成功，但可能是历史缓存文件")
                    return base64_data
            except Exception as e:
                logger.debug(f"{self.log_prefix} 并发文件系统查找失败: {e}")

            logger.warning(f"{self.log_prefix} 无法通过picid {picid[:8]}... 获取图片数据")
            self._mark_picid_failed(picid)
            return None

        except Exception as e:
            logger.error(f"{self.log_prefix} 通过picid获取图片异常: {str(e)[:100]}")
            self._mark_picid_failed(picid)
            return None

    async def _check_paths_concurrently(self, picid: str) -> Optional[str]:
        """并发检查多个可能的文件路径"""
        possible_paths = [
            f"data/images/{picid}.jpg",
            f"data/images/{picid}.png",
            f"data/images/{picid}.jpeg",
            f"data/images/{picid}.webp",
            f"images/{picid}.jpg",
            f"images/{picid}.png",
            f"images/{picid}.jpeg",
            f"images/{picid}.webp",
            f"temp/images/{picid}.jpg",
            f"temp/images/{picid}.png"
        ]

        def check_single_path(path: str) -> Optional[str]:
            """检查单个路径并返回base64数据"""
            try:
                if os.path.exists(path):
                    from src.chat.utils.utils_image import image_path_to_base64
                    base64_data = image_path_to_base64(path)
                    if base64_data:
                        logger.info(f"{self.log_prefix} 通过路径 {path} 获取图片成功")
                        return base64_data
            except (FileNotFoundError, IOError) as e:
                logger.debug(f"{self.log_prefix} 读取路径 {path} 失败: {e}")
            except Exception as e:
                logger.debug(f"{self.log_prefix} 检查路径 {path} 异常: {e}")
            return None

        # 使用线程池并发检查所有路径
        with ThreadPoolExecutor(max_workers=4) as executor:
            loop = asyncio.get_event_loop()
            futures = [loop.run_in_executor(executor, check_single_path, path) for path in possible_paths]

            # 等待第一个成功的结果
            for future in asyncio.as_completed(futures):
                try:
                    result = await future
                    if result:
                        # 取消其他任务
                        for f in futures:
                            if not f.done():
                                f.cancel()
                        return result
                except Exception as e:
                    logger.debug(f"{self.log_prefix} 并发检查路径异常: {e}")
                    continue

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
            logger.debug(f"{self.log_prefix} 处理图片数据失败: {str(e)[:50]}")
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