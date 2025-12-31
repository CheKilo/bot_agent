# -*- coding: utf-8 -*-
"""
API 请求处理器

提供简化的请求处理函数，可用于自定义集成场景。

注意：主要的 API 端点定义在 routes.py 中，
此文件提供独立的处理函数供程序化调用。
"""

from typing import Dict, List, Optional

from api.service import chat_service, PERSONAS


def handle_chat(
    user_id: str,
    message: str,
    bot_id: str = "default_bot",
    persona: Optional[str] = None,
    enable_memory: Optional[bool] = None,
) -> Dict:
    """
    处理对话请求

    Args:
        user_id: 用户ID
        message: 用户消息
        bot_id: Bot ID
        persona: 人设名称（可选）
        enable_memory: 是否启用记忆（可选）

    Returns:
        Dict: {
            "success": bool,
            "answer": str,
            "iterations": int,
            "error": Optional[str]
        }
    """
    return chat_service.chat(
        user_id=user_id,
        bot_id=bot_id,
        message=message,
        persona=persona,
        enable_memory=enable_memory,
    )


def handle_get_history(
    user_id: str,
    bot_id: str = "default_bot",
) -> List[Dict]:
    """
    获取对话历史

    Args:
        user_id: 用户ID
        bot_id: Bot ID

    Returns:
        List[Dict]: 对话历史列表
    """
    return chat_service.get_history(bot_id=bot_id, user_id=user_id)


def handle_clear_history(
    user_id: str,
    bot_id: str = "default_bot",
) -> bool:
    """
    清空对话历史

    Args:
        user_id: 用户ID
        bot_id: Bot ID

    Returns:
        bool: 是否成功
    """
    return chat_service.clear_history(bot_id=bot_id, user_id=user_id)


def handle_set_persona(
    user_id: str,
    persona: str,
    bot_id: str = "default_bot",
) -> bool:
    """
    设置人设

    Args:
        user_id: 用户ID
        persona: 人设名称
        bot_id: Bot ID

    Returns:
        bool: 是否成功
    """
    return chat_service.set_persona(
        bot_id=bot_id,
        user_id=user_id,
        persona_name=persona,
    )


def get_available_personas() -> Dict[str, Dict]:
    """
    获取可用人设列表

    Returns:
        Dict[str, Dict]: 人设配置字典
    """
    return chat_service.get_personas()


def handle_delete_session(
    user_id: str,
    bot_id: str = "default_bot",
) -> bool:
    """
    删除会话

    Args:
        user_id: 用户ID
        bot_id: Bot ID

    Returns:
        bool: 是否成功
    """
    return chat_service.delete_session(bot_id=bot_id, user_id=user_id)
