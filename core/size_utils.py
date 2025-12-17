"""尺寸转换工具模块

提供统一的图片尺寸解析、验证和转换功能，供各API客户端复用
"""
from typing import Tuple, Optional, Dict
from src.common.logger import get_logger

logger = get_logger("pic_action")

# LLM 尺寸选择系统提示词
SIZE_SELECTOR_SYSTEM_PROMPT = """You are an image size selector. Based on the image description, choose the most appropriate size.

## Rules:
1. Output ONLY the size, no explanations
2. Choose based on content:
   - Portrait/character/person → vertical: 832x1216 or 1024x1536
   - Landscape/scenery/wide scene → horizontal: 1216x832 or 1536x1024
   - Avatar/icon/square content → square: 1024x1024
   - Group photo/multiple people → horizontal: 1216x832
3. Default to 1024x1024 if unclear

## Examples:
Input: 一个女孩站在海边
Output: 832x1216

Input: 壮丽的山脉风景
Output: 1216x832

Input: 可爱的猫咪头像
Output: 1024x1024

Input: draw a cat
Output: 1024x1024

Now select size for:"""


async def select_size_with_llm(description: str, log_prefix: str = "") -> Optional[str]:
    """使用 LLM 选择合适的图片尺寸

    Args:
        description: 图片描述
        log_prefix: 日志前缀

    Returns:
        选择的尺寸字符串，如 "1024x1024"，失败返回 None
    """
    if not description or not description.strip():
        return None

    try:
        from src.plugin_system.apis import llm_api

        # 获取可用模型
        models = llm_api.get_available_models()
        if not models:
            logger.warning(f"{log_prefix} 没有可用的 LLM 模型，无法选择尺寸")
            return None

        # 使用 replyer 模型（首要回复模型）
        if "replyer" not in models:
            logger.warning(f"{log_prefix} 没有找到 replyer 模型，无法选择尺寸")
            return None
        model_config = models["replyer"]

        # 构建 prompt
        full_prompt = f"{SIZE_SELECTOR_SYSTEM_PROMPT}\nInput: {description.strip()}\nOutput:"

        logger.info(f"{log_prefix} 使用 LLM 选择尺寸...")

        # 调用 LLM（不传递 temperature 和 max_tokens，使用模型默认值）
        success, response, reasoning, model_name = await llm_api.generate_with_model(
            prompt=full_prompt,
            model_config=model_config,
            request_type="plugin.size_select",
        )

        if success and response:
            # 清理响应
            size = response.strip().split()[0]  # 只取第一个词
            size = size.strip('"\'')  # 移除引号

            # 验证格式
            if validate_image_size(size):
                logger.info(f"{log_prefix} LLM 选择尺寸: {size} (模型: {model_name})")
                return size
            else:
                logger.warning(f"{log_prefix} LLM 返回无效尺寸: {size}")
                return None
        else:
            logger.warning(f"{log_prefix} LLM 尺寸选择失败")
            return None

    except Exception as e:
        logger.error(f"{log_prefix} LLM 尺寸选择异常: {e}")
        return None


def get_image_size(model_config: dict, llm_size: str = None, log_prefix: str = "") -> Tuple[str, Optional[str]]:
    """统一的图片尺寸获取逻辑

    根据 fixed_size_enabled 配置决定使用 LLM 选择的尺寸还是配置文件的固定尺寸。
    供 Action 和 Command 组件复用。

    Args:
        model_config: 模型配置字典
        llm_size: LLM 选择的尺寸（可选，Command 组件无此参数）
        log_prefix: 日志前缀

    Returns:
        (image_size, llm_original_size) 元组
        - image_size: 最终使用的尺寸
        - llm_original_size: 原始 LLM 尺寸（用于 Gemini 特殊处理）
    """
    fixed_size_enabled = model_config.get("fixed_size_enabled", False)
    llm_original_size = llm_size if llm_size else None

    if fixed_size_enabled:
        # 使用配置文件的固定尺寸，忽略 LLM 选择
        size = None
        if log_prefix:
            logger.info(f"{log_prefix} 使用固定尺寸配置")
    else:
        # 允许使用 LLM 选择的尺寸
        size = llm_size

    image_size = size or model_config.get("default_size", "1024x1024")
    return image_size, llm_original_size


async def get_image_size_async(
    model_config: dict,
    description: str = "",
    llm_size: str = None,
    log_prefix: str = ""
) -> Tuple[str, Optional[str]]:
    """异步版本的图片尺寸获取逻辑

    当 fixed_size_enabled=False 且没有预设 llm_size 时，
    会调用 LLM 选择合适的尺寸。

    特殊情况：当 fixed_size_enabled=True 且 default_size 以 "-" 开头时（如 "-2K"），
    仍会调用 LLM 选择宽高比，分辨率使用配置值。

    Args:
        model_config: 模型配置字典
        description: 图片描述（用于 LLM 选择尺寸）
        llm_size: 预设的 LLM 尺寸（Action 组件已有时使用）
        log_prefix: 日志前缀

    Returns:
        (image_size, llm_original_size) 元组
    """
    fixed_size_enabled = model_config.get("fixed_size_enabled", False)
    default_size = model_config.get("default_size", "1024x1024")

    if fixed_size_enabled:
        # 特殊处理："-2K" 格式需要 LLM 选择宽高比
        if default_size.startswith("-") and description:
            if log_prefix:
                logger.info(f"{log_prefix} 使用固定分辨率配置，LLM 选择宽高比")
            selected_size = await select_size_with_llm(description, log_prefix)
            # 返回配置的尺寸，但保留 LLM 选择的尺寸用于宽高比转换
            return default_size, selected_size

        # 使用配置文件的固定尺寸
        if log_prefix:
            logger.info(f"{log_prefix} 使用固定尺寸配置")
        return default_size, None

    # fixed_size_enabled = False，让 MaiBot 选择
    if llm_size:
        # 已有 LLM 尺寸（Action 组件）
        return llm_size, llm_size

    # 没有预设尺寸，调用 LLM 选择（Command 组件）
    if description:
        selected_size = await select_size_with_llm(description, log_prefix)
        if selected_size:
            return selected_size, selected_size

    # 降级：使用默认尺寸
    return default_size, None


def gcd(a: int, b: int) -> int:
    """计算最大公约数

    Args:
        a: 第一个数
        b: 第二个数

    Returns:
        最大公约数
    """
    while b:
        a, b = b, a % b
    return a


def parse_pixel_size(size: str, default_width: int = 1024, default_height: int = 1024) -> Tuple[int, int]:
    """解析像素尺寸字符串

    支持格式：
    - "1024x1024"
    - "1024*1024"
    - "1024X1024"

    Args:
        size: 尺寸字符串
        default_width: 解析失败时的默认宽度
        default_height: 解析失败时的默认高度

    Returns:
        (width, height) 元组
    """
    if not size or not isinstance(size, str):
        return default_width, default_height

    size_lower = size.lower().strip()

    # 尝试解析 "WxH" 或 "W*H" 格式
    for separator in ['x', '*']:
        if separator in size_lower:
            try:
                parts = size_lower.split(separator)
                if len(parts) == 2:
                    width = int(parts[0].strip())
                    height = int(parts[1].strip())
                    if width > 0 and height > 0:
                        return width, height
            except (ValueError, IndexError):
                pass

    return default_width, default_height


def validate_image_size(size: str) -> bool:
    """验证图片尺寸格式是否正确

    支持的格式：
    1. 像素格式：1024x1024、512x512（64-4096范围）
    2. 宽高比格式：16:9、1:1、4:3
    3. 宽高比+分辨率：16:9-2K、1:1-4K
    4. 仅分辨率：-2K、-4K

    Args:
        size: 尺寸字符串

    Returns:
        是否有效
    """
    if not size or not isinstance(size, str):
        return False

    size = size.strip()

    try:
        # 格式1：仅分辨率（-2K、-4K）
        if size.startswith('-'):
            resolution = size[1:].strip().upper()
            return resolution in ['1K', '2K', '4K']

        # 格式2：宽高比-分辨率（16:9-2K、1:1-4K）
        if '-' in size and ':' in size:
            parts = size.split('-', 1)
            aspect_part = parts[0].strip()
            resolution = parts[1].strip().upper()

            # 验证宽高比部分
            if ':' in aspect_part:
                aspect_parts = aspect_part.split(':', 1)
                try:
                    w = int(aspect_parts[0].strip())
                    h = int(aspect_parts[1].strip())
                    if w <= 0 or h <= 0:
                        return False
                except ValueError:
                    return False

                # 验证分辨率部分
                return resolution in ['1K', '2K', '4K']
            return False

        # 格式3：纯宽高比（16:9、1:1）
        if ':' in size and 'x' not in size.lower():
            parts = size.split(':', 1)
            try:
                w = int(parts[0].strip())
                h = int(parts[1].strip())
                return w > 0 and h > 0
            except ValueError:
                return False

        # 格式4：像素格式（1024x1024、512x512）
        if 'x' in size.lower() or '*' in size:
            width, height = parse_pixel_size(size, 0, 0)
            return 64 <= width <= 4096 and 64 <= height <= 4096

        return False

    except (ValueError, AttributeError):
        return False


def pixel_to_aspect_ratio(width: int, height: int) -> Tuple[int, int]:
    """将像素尺寸转换为最简宽高比

    Args:
        width: 宽度
        height: 高度

    Returns:
        (宽高比宽, 宽高比高) 元组，如 (16, 9)
    """
    if width <= 0 or height <= 0:
        return 1, 1

    divisor = gcd(width, height)
    return width // divisor, height // divisor


def pixel_to_orientation(width: int, height: int) -> str:
    """根据像素尺寸判断图片方向

    Args:
        width: 宽度
        height: 高度

    Returns:
        方向字符串：方图/竖图/横图
    """
    if width > height:
        return "横图"
    elif height > width:
        return "竖图"
    else:
        return "方图"


def find_closest_aspect_ratio(
    width: int,
    height: int,
    supported_ratios: Optional[Dict[Tuple[int, int], str]] = None
) -> str:
    """查找最接近的支持宽高比

    Args:
        width: 宽度
        height: 高度
        supported_ratios: 支持的宽高比映射，格式为 {(w, h): "w:h", ...}
                         如果为None，使用默认的常见宽高比

    Returns:
        最接近的宽高比字符串，如 "16:9"
    """
    if supported_ratios is None:
        supported_ratios = {
            (1, 1): "1:1",
            (16, 9): "16:9",
            (9, 16): "9:16",
            (4, 3): "4:3",
            (3, 4): "3:4",
            (3, 2): "3:2",
            (2, 3): "2:3",
            (4, 5): "4:5",
            (5, 4): "5:4",
            (21, 9): "21:9",
        }

    if width <= 0 or height <= 0:
        return "1:1"

    # 先检查是否精确匹配
    aspect_w, aspect_h = pixel_to_aspect_ratio(width, height)
    if (aspect_w, aspect_h) in supported_ratios:
        return supported_ratios[(aspect_w, aspect_h)]

    # 查找最接近的宽高比
    target_ratio = width / height
    closest_ratio = "1:1"
    min_diff = float('inf')

    for (w, h), ratio_str in supported_ratios.items():
        diff = abs(w / h - target_ratio)
        if diff < min_diff:
            min_diff = diff
            closest_ratio = ratio_str

    return closest_ratio


def pixel_size_to_gemini_aspect(
    pixel_size: str,
    log_prefix: str = ""
) -> Optional[str]:
    """将像素格式转换为Gemini支持的宽高比

    Args:
        pixel_size: 像素尺寸字符串，如 "1024x1024"
        log_prefix: 日志前缀

    Returns:
        Gemini支持的宽高比字符串，如 "16:9"，失败返回None
    """
    if not pixel_size or 'x' not in pixel_size.lower():
        return None

    width, height = parse_pixel_size(pixel_size, 0, 0)
    if width <= 0 or height <= 0:
        return None

    # Gemini 支持的宽高比
    gemini_supported_ratios = {
        (1, 1): "1:1",
        (16, 9): "16:9",
        (9, 16): "9:16",
        (4, 3): "4:3",
        (3, 4): "3:4",
        (3, 2): "3:2",
        (2, 3): "2:3",
        (4, 5): "4:5",
        (5, 4): "5:4",
        (21, 9): "21:9",
    }

    aspect_w, aspect_h = pixel_to_aspect_ratio(width, height)

    # 精确匹配
    if (aspect_w, aspect_h) in gemini_supported_ratios:
        return gemini_supported_ratios[(aspect_w, aspect_h)]

    # 查找最接近的
    closest = find_closest_aspect_ratio(width, height, gemini_supported_ratios)
    if log_prefix:
        logger.warning(f"{log_prefix} 宽高比 {aspect_w}:{aspect_h} 不在支持列表，使用最接近的: {closest}")

    return closest


def pixel_size_to_orientation(pixel_size: str) -> str:
    """将像素格式转换为方向（方图/竖图/横图）

    Args:
        pixel_size: 像素尺寸字符串，如 "1024x1024"

    Returns:
        方向字符串：方图/竖图/横图
    """
    width, height = parse_pixel_size(pixel_size, 1, 1)
    return pixel_to_orientation(width, height)


# 预定义的尺寸映射
ORIENTATION_SIZE_MAPPING = {
    # 像素格式映射
    "1024x1024": "方图",
    "512x512": "方图",
    "832x1216": "竖图",
    "1216x832": "横图",
    # 英文映射
    "square": "方图",
    "portrait": "竖图",
    "landscape": "横图",
}


def size_to_orientation(size: str, default: str = "竖图") -> str:
    """将尺寸字符串转换为方向

    支持多种输入格式：
    - 像素格式：1024x1024
    - 方向英文：square, portrait, landscape
    - 方向中文：方图, 竖图, 横图

    Args:
        size: 尺寸字符串
        default: 默认方向

    Returns:
        方向字符串：方图/竖图/横图
    """
    if not size:
        return default

    size = size.strip()

    # 检查是否在预定义映射中
    if size in ORIENTATION_SIZE_MAPPING:
        return ORIENTATION_SIZE_MAPPING[size]

    # 检查是否是中文方向
    if size in ["方图", "竖图", "横图"]:
        return size

    # 尝试解析像素格式
    if 'x' in size.lower() or '*' in size:
        return pixel_size_to_orientation(size)

    return default
