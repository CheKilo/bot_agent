# -*- coding: utf-8 -*-
"""
System Agent 模块

提供系统调度 Agent，协调 Memory Agent 和 Character Agent。
通过 AgentRegistry 实现松耦合的 Agent 调度。
"""

from agent.agents.system.config import SystemConfig, LLMConfig, ConversationConfig
from agent.agents.system.system_agent import SystemAgent

__all__ = [
    # 配置
    "SystemConfig",
    "LLMConfig",
    "ConversationConfig",
    # Agent
    "SystemAgent",
]
