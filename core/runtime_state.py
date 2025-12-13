"""运行时状态管理器

按聊天流分别管理插件状态，支持：
- 插件开关
- 模型开关
- 撤回开关
- 默认模型设置
"""
from typing import Dict, Any, Optional, Set
from dataclasses import dataclass, field
from src.common.logger import get_logger

logger = get_logger("runtime_state")


@dataclass
class ChatStreamState:
    """单个聊天流的状态"""
    # 插件是否启用（None表示使用全局配置）
    plugin_enabled: Optional[bool] = None
    # 被禁用的模型ID集合
    disabled_models: Set[str] = field(default_factory=set)
    # 禁用撤回的模型ID集合
    recall_disabled_models: Set[str] = field(default_factory=set)
    # Action组件默认模型（None表示使用全局配置）
    action_default_model: Optional[str] = None
    # Command组件默认模型（None表示使用全局配置）
    command_default_model: Optional[str] = None


class RuntimeStateManager:
    """运行时状态管理器（单例）

    按聊天流ID分别管理状态，所有状态仅在内存中保持，重启后重置。
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._states: Dict[str, ChatStreamState] = {}
        return cls._instance

    def _get_state(self, chat_id: str) -> ChatStreamState:
        """获取或创建聊天流状态"""
        if chat_id not in self._states:
            self._states[chat_id] = ChatStreamState()
        return self._states[chat_id]

    # ==================== 插件开关 ====================

    def is_plugin_enabled(self, chat_id: str, global_enabled: bool) -> bool:
        """检查插件是否启用

        Args:
            chat_id: 聊天流ID
            global_enabled: 全局配置的启用状态

        Returns:
            插件是否启用（优先使用聊天流状态，否则使用全局配置）
        """
        state = self._get_state(chat_id)
        if state.plugin_enabled is not None:
            return state.plugin_enabled
        return global_enabled

    def set_plugin_enabled(self, chat_id: str, enabled: bool) -> None:
        """设置插件启用状态"""
        state = self._get_state(chat_id)
        state.plugin_enabled = enabled
        logger.info(f"[RuntimeState] 聊天流 {chat_id} 插件状态设置为: {enabled}")

    def reset_plugin_enabled(self, chat_id: str) -> None:
        """重置插件启用状态为全局配置"""
        state = self._get_state(chat_id)
        state.plugin_enabled = None
        logger.info(f"[RuntimeState] 聊天流 {chat_id} 插件状态已重置为全局配置")

    # ==================== 模型开关 ====================

    def is_model_enabled(self, chat_id: str, model_id: str) -> bool:
        """检查模型是否启用"""
        state = self._get_state(chat_id)
        return model_id not in state.disabled_models

    def set_model_enabled(self, chat_id: str, model_id: str, enabled: bool) -> None:
        """设置模型启用状态"""
        state = self._get_state(chat_id)
        if enabled:
            state.disabled_models.discard(model_id)
            logger.info(f"[RuntimeState] 聊天流 {chat_id} 模型 {model_id} 已启用")
        else:
            state.disabled_models.add(model_id)
            logger.info(f"[RuntimeState] 聊天流 {chat_id} 模型 {model_id} 已禁用")

    def get_disabled_models(self, chat_id: str) -> Set[str]:
        """获取被禁用的模型列表"""
        state = self._get_state(chat_id)
        return state.disabled_models.copy()

    # ==================== 撤回开关 ====================

    def is_recall_enabled(self, chat_id: str, model_id: str, global_enabled: bool) -> bool:
        """检查模型的撤回功能是否启用

        Args:
            chat_id: 聊天流ID
            model_id: 模型ID
            global_enabled: 全局撤回开关状态

        Returns:
            撤回是否启用
        """
        if not global_enabled:
            return False
        state = self._get_state(chat_id)
        return model_id not in state.recall_disabled_models

    def set_recall_enabled(self, chat_id: str, model_id: str, enabled: bool) -> None:
        """设置模型的撤回启用状态"""
        state = self._get_state(chat_id)
        if enabled:
            state.recall_disabled_models.discard(model_id)
            logger.info(f"[RuntimeState] 聊天流 {chat_id} 模型 {model_id} 撤回已启用")
        else:
            state.recall_disabled_models.add(model_id)
            logger.info(f"[RuntimeState] 聊天流 {chat_id} 模型 {model_id} 撤回已禁用")

    def get_recall_disabled_models(self, chat_id: str) -> Set[str]:
        """获取撤回被禁用的模型列表"""
        state = self._get_state(chat_id)
        return state.recall_disabled_models.copy()

    # ==================== 默认模型 ====================

    def get_action_default_model(self, chat_id: str, global_default: str) -> str:
        """获取Action组件的默认模型"""
        state = self._get_state(chat_id)
        if state.action_default_model is not None:
            return state.action_default_model
        return global_default

    def set_action_default_model(self, chat_id: str, model_id: str) -> None:
        """设置Action组件的默认模型"""
        state = self._get_state(chat_id)
        state.action_default_model = model_id
        logger.info(f"[RuntimeState] 聊天流 {chat_id} Action默认模型设置为: {model_id}")

    def reset_action_default_model(self, chat_id: str) -> None:
        """重置Action组件的默认模型为全局配置"""
        state = self._get_state(chat_id)
        state.action_default_model = None
        logger.info(f"[RuntimeState] 聊天流 {chat_id} Action默认模型已重置为全局配置")

    def get_command_default_model(self, chat_id: str, global_default: str) -> str:
        """获取Command组件的默认模型"""
        state = self._get_state(chat_id)
        if state.command_default_model is not None:
            return state.command_default_model
        return global_default

    def set_command_default_model(self, chat_id: str, model_id: str) -> None:
        """设置Command组件的默认模型"""
        state = self._get_state(chat_id)
        state.command_default_model = model_id
        logger.info(f"[RuntimeState] 聊天流 {chat_id} Command默认模型设置为: {model_id}")

    def reset_command_default_model(self, chat_id: str) -> None:
        """重置Command组件的默认模型为全局配置"""
        state = self._get_state(chat_id)
        state.command_default_model = None
        logger.info(f"[RuntimeState] 聊天流 {chat_id} Command默认模型已重置为全局配置")

    # ==================== 状态重置 ====================

    def reset_chat_state(self, chat_id: str) -> None:
        """重置指定聊天流的所有状态"""
        if chat_id in self._states:
            del self._states[chat_id]
            logger.info(f"[RuntimeState] 聊天流 {chat_id} 所有状态已重置")

    def get_chat_state_summary(self, chat_id: str) -> Dict[str, Any]:
        """获取聊天流状态摘要"""
        state = self._get_state(chat_id)
        return {
            "plugin_enabled": state.plugin_enabled,
            "disabled_models": list(state.disabled_models),
            "recall_disabled_models": list(state.recall_disabled_models),
            "action_default_model": state.action_default_model,
            "command_default_model": state.command_default_model,
        }


# 全局单例
runtime_state = RuntimeStateManager()
