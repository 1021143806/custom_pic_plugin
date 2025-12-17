from typing import Optional, Dict, Any
from threading import Lock

from src.common.logger import get_logger

logger = get_logger("pic_action")

class CacheManager:
    """缓存管理器"""

    # 类级别的缓存存储
    _request_cache = {}  # 文生图缓存
    _img2img_cache = {}  # 图生图缓存
    _cache_lock = Lock()  # 线程安全锁

    def __init__(self, action_instance):
        self.action = action_instance
        self.log_prefix = action_instance.log_prefix

    def _get_max_size(self) -> int:
        """获取最大缓存数量配置"""
        return self.action.get_config("cache.max_size", 10)

    def get_cached_result(self, description: str, model: str, size: str, strength: float = None, is_img2img: bool = False) -> Optional[str]:
        """获取缓存的结果"""
        if not self.action.get_config("cache.enabled", True):
            return None

        try:
            with self._cache_lock:
                if is_img2img:
                    cache_key = self._get_img2img_cache_key(description, model, size, strength)
                    cache_dict = self._img2img_cache
                else:
                    cache_key = self._get_cache_key(description, model, size)
                    cache_dict = self._request_cache

                if cache_key in cache_dict:
                    logger.debug(f"{self.log_prefix} 找到缓存结果: {cache_key}")
                    return cache_dict[cache_key]

                return None
        except Exception as e:
            logger.warning(f"{self.log_prefix} 获取缓存失败: {e}")
            return None

    def cache_result(self, description: str, model: str, size: str, strength: float = None, is_img2img: bool = False, result: str = None):
        """缓存结果"""
        if not self.action.get_config("cache.enabled", True) or not result:
            return

        try:
            with self._cache_lock:
                if is_img2img:
                    cache_key = self._get_img2img_cache_key(description, model, size, strength)
                    cache_dict = self._img2img_cache
                else:
                    cache_key = self._get_cache_key(description, model, size)
                    cache_dict = self._request_cache

                max_size = self._get_max_size()

                # 添加到缓存
                cache_dict[cache_key] = result
                logger.debug(f"{self.log_prefix} 缓存结果: {cache_key}")

                # 清理过期缓存
                if len(cache_dict) > max_size:
                    self._cleanup_cache_dict(cache_dict, max_size)

        except Exception as e:
            logger.warning(f"{self.log_prefix} 缓存结果失败: {e}")

    def remove_cached_result(self, description: str, model: str, size: str, strength: float = None, is_img2img: bool = False):
        """移除缓存的结果"""
        try:
            with self._cache_lock:
                if is_img2img:
                    cache_key = self._get_img2img_cache_key(description, model, size, strength)
                    cache_dict = self._img2img_cache
                else:
                    cache_key = self._get_cache_key(description, model, size)
                    cache_dict = self._request_cache

                if cache_key in cache_dict:
                    del cache_dict[cache_key]
                    logger.debug(f"{self.log_prefix} 移除失效缓存: {cache_key}")

        except Exception as e:
            logger.warning(f"{self.log_prefix} 移除缓存失败: {e}")

    def clear_cache(self, cache_type: str = "all"):
        """清空缓存"""
        try:
            with self._cache_lock:
                if cache_type == "all" or cache_type == "txt2img":
                    self._request_cache.clear()
                    logger.info(f"{self.log_prefix} 清空文生图缓存")

                if cache_type == "all" or cache_type == "img2img":
                    self._img2img_cache.clear()
                    logger.info(f"{self.log_prefix} 清空图生图缓存")

        except Exception as e:
            logger.warning(f"{self.log_prefix} 清空缓存失败: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        try:
            with self._cache_lock:
                max_size = self._get_max_size()
                return {
                    "txt2img_cache_size": len(self._request_cache),
                    "txt2img_cache_max": max_size,
                    "img2img_cache_size": len(self._img2img_cache),
                    "img2img_cache_max": max_size,
                    "cache_enabled": self.action.get_config("cache.enabled", True)
                }
        except Exception as e:
            logger.warning(f"{self.log_prefix} 获取缓存统计失败: {e}")
            return {}

    @classmethod
    def _get_cache_key(cls, description: str, model: str, size: str) -> str:
        """生成文生图缓存键"""
        return f"txt2img_{description[:100]}|{model}|{size}"

    @classmethod
    def _get_img2img_cache_key(cls, description: str, model: str, size: str, strength: float = None) -> str:
        """生成图生图缓存键"""
        strength_str = str(strength) if strength is not None else "default"
        return f"img2img_{description[:50]}|{model}|{size}|{strength_str}"

    @classmethod
    def _cleanup_cache_dict(cls, cache_dict: Dict, max_size: int):
        """清理缓存字典"""
        if len(cache_dict) > max_size:
            # 移除一半的最旧条目
            keys_to_remove = list(cache_dict.keys())[: -max_size // 2]
            for key in keys_to_remove:
                del cache_dict[key]

