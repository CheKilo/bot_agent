# -*- coding: utf-8 -*-
"""
Agent 核心模块

包含 Agent 的核心功能组件：
- LLM: 大语言模型调用封装
- Message: 对话消息数据类
- LLMResponse: LLM 响应结果

工具相关组件请从 agent.tools 导入：
- Tool: 工具基类
- ToolResult: 工具执行结果
- ToolKit: 工具集管理器
"""

from .llm import (
    # 核心类
    LLM,
    Message,
    LLMResponse,
    # 便捷函数
    quick_chat,
    quick_embed,
)

__all__ = [
    # 核心类
    "LLM",
    "Message",
    "LLMResponse",
    # 便捷函数
    "quick_chat",
    "quick_embed",
]
