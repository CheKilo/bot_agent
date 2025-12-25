# -*- coding: utf-8 -*-
"""
Character 模块配置

所有配置集中管理，包括 LLM 配置。
"""

from dataclasses import dataclass, field
from typing import Optional


# ========== LLM 默认配置 ==========
DEFAULT_LLM_ADDRESS = "localhost:50051"
DEFAULT_LLM_TIMEOUT = 30.0

# 情绪分析使用较小模型（快速）
EMOTION_LLM_MODEL = "gpt-4o-mini"

# 回复生成使用较好模型（质量）
RESPONSE_LLM_MODEL = "gpt-5"


# ========== 配置类 ==========


@dataclass
class LLMConfig:
    """LLM 配置"""

    address: str = DEFAULT_LLM_ADDRESS
    model: str = RESPONSE_LLM_MODEL
    timeout: float = DEFAULT_LLM_TIMEOUT


@dataclass
class EmotionToolConfig:
    """情绪分析工具配置"""

    llm: LLMConfig = field(default_factory=lambda: LLMConfig(model=EMOTION_LLM_MODEL))


@dataclass
class ResponseToolConfig:
    """回复生成工具配置"""

    llm: LLMConfig = field(default_factory=LLMConfig)


@dataclass
class CharacterConfig:
    """Character Agent 配置"""

    # 情绪分析工具配置
    emotion_tool: EmotionToolConfig = field(default_factory=EmotionToolConfig)

    # 回复生成工具配置
    response_tool: ResponseToolConfig = field(default_factory=ResponseToolConfig)

    # Agent 自身的 LLM（用于 ReAct 循环）
    agent_llm: LLMConfig = field(default_factory=LLMConfig)
