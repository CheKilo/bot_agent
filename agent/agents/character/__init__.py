# -*- coding: utf-8 -*-
"""
Character Agent 模块

提供角色扮演 Agent，支持人设配置和情绪管理。
"""

from agent.agents.character.character_agent import CharacterAgent, CharacterResult
from agent.agents.character.persona import (
    Persona,
    DEFAULT_PERSONA,
    EXAMPLE_PERSONA_GIRL,
    EXAMPLE_PERSONA_MATURE,
)
from agent.agents.character.tools.emotion import (
    UpdateEmotion,
    default_emotion,
    format_emotion_for_prompt,
)

__all__ = [
    # Agent
    "CharacterAgent",
    "CharacterResult",
    # 人设
    "Persona",
    "DEFAULT_PERSONA",
    "EXAMPLE_PERSONA_GIRL",
    "EXAMPLE_PERSONA_MATURE",
    # 情绪
    "UpdateEmotion",
    "default_emotion",
    "format_emotion_for_prompt",
]
