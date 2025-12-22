# -*- coding: utf-8 -*-
"""
Character Agent - 角色扮演 Agent

基于 ReAct 架构的角色扮演 Agent。
根据人设、记忆和情绪状态，生成符合角色特征的回复。

特点：
- 情绪状态存储在内存 dict 中，通过工具调用更新
- 每轮对话将情绪状态注入 system prompt
- 支持外部传入记忆上下文（来自 MemoryAgent）
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent.agents.base import Agent, AgentEventType, AgentResult
from agent.core import LLM
from agent.tools import Tool

from agent.agents.character.persona import Persona, DEFAULT_PERSONA
from agent.agents.character.tools.emotion import (
    UpdateEmotion,
    default_emotion,
    format_emotion_for_prompt,
)

logger = logging.getLogger(__name__)


# ============================================================================
# System Prompt 模板
# ============================================================================

SYSTEM_PROMPT_TEMPLATE = """你是一个角色扮演AI，需要严格按照以下人设进行对话。

{persona}

## 当前情绪状态
{emotion}

## 行为约束
1. **始终保持人设一致性**：说话方式、用词习惯、性格特征必须与人设匹配
2. **情绪影响表达**：
   - 心情好时：语气更活泼、回复更积极
   - 心情低落时：语气较低沉、回复更简短
   - 好感度高时：更亲近、更愿意分享
   - 好感度低时：较疏远、回复更公事公办
3. **主动更新情绪**：根据对话内容，适时调用 update_emotion 工具更新情绪状态
4. **自然交流**：不要暴露自己是AI，以角色身份自然对话

## 记忆上下文
{memory_context}

请以角色身份回复用户。"""


# ============================================================================
# Character Agent
# ============================================================================


@dataclass
class CharacterResult(AgentResult):
    """Character Agent 运行结果（扩展自 AgentResult）"""

    emotion_state: Dict[str, float] = None  # 当前情绪状态（对话后）

    def __post_init__(self):
        if self.emotion_state is None:
            self.emotion_state = {}


class CharacterAgent(Agent):
    """
    角色扮演 Agent

    职责：
    1. 根据人设生成符合角色特征的回复
    2. 结合记忆上下文保持对话一致性
    3. 动态管理和响应情绪状态

    使用示例：
        ```python
        from agent.agents.character import CharacterAgent
        from agent.agents.character.persona import EXAMPLE_PERSONA_GIRL

        agent = CharacterAgent(
            bot_id="my_bot",
            persona=EXAMPLE_PERSONA_GIRL,
        )

        # 运行（可选传入记忆上下文）
        result = agent.run("你好呀！", memory_context="用户喜欢猫咪")
        print(result.answer)
        print(result.emotion_state)  # 查看更新后的情绪
        ```
    """

    name = "character_agent"
    max_iterations = 5
    message_window = 20

    def __init__(
        self,
        bot_id: str = "default_bot",
        persona: Optional[Persona] = None,
        emotion_state: Optional[Dict[str, float]] = None,
        llm_address: str = LLM.DEFAULT_ADDRESS,
        model: str = LLM.DEFAULT_MODEL,
    ):
        """
        初始化 Character Agent

        Args:
            bot_id: 机器人 ID
            persona: 人设配置，默认使用 DEFAULT_PERSONA
            emotion_state: 初始情绪状态，默认使用 default_emotion()
            llm_address: LLM 服务地址
            model: 模型名称
        """
        self._persona = persona or DEFAULT_PERSONA
        self._emotion = (
            emotion_state if emotion_state is not None else default_emotion()
        )
        self._memory_context = ""  # 记忆上下文（每次 run 时设置）

        super().__init__(
            llm_address=llm_address,
            model=model,
            bot_id=bot_id,
        )

    # ========== 属性 ==========

    @property
    def persona(self) -> Persona:
        """当前人设"""
        return self._persona

    @property
    def emotion(self) -> Dict[str, float]:
        """当前情绪状态（返回副本）"""
        return self._emotion.copy()

    @emotion.setter
    def emotion(self, value: Dict[str, float]):
        """设置情绪状态"""
        self._emotion = value

    # ========== 核心方法 ==========

    def get_system_prompt(self) -> str:
        """动态生成系统提示词（注入人设、情绪、记忆）"""
        return SYSTEM_PROMPT_TEMPLATE.format(
            persona=self._persona.to_prompt(),
            emotion=format_emotion_for_prompt(self._emotion),
            memory_context=self._memory_context or "[无相关记忆]",
        )

    def get_tools(self) -> List[Tool]:
        """返回工具列表（情绪更新工具）"""
        return [
            UpdateEmotion(self._emotion),  # 传入情绪 dict 的引用
        ]

    def run(
        self,
        user_input: str,
        memory_context: str = "",
    ) -> CharacterResult:
        """
        执行角色对话

        Args:
            user_input: 用户输入
            memory_context: 来自 MemoryAgent 的记忆上下文（可选）

        Returns:
            CharacterResult: 包含回复和更新后的情绪状态
        """
        # 设置记忆上下文
        self._memory_context = memory_context

        # 刷新 system prompt（注入最新情绪和记忆）
        self._refresh_system_prompt()

        # 执行 ReAct 循环
        base_result = super().run(user_input)

        # 封装结果（附带情绪状态）
        return CharacterResult(
            answer=base_result.answer,
            iterations=base_result.iterations,
            tool_calls=base_result.tool_calls,
            success=base_result.success,
            error=base_result.error,
            emotion_state=self._emotion.copy(),
        )

    def _refresh_system_prompt(self):
        """刷新系统提示词"""
        new_prompt = self.get_system_prompt()
        if self._messages and self._messages[0]["role"] == "system":
            self._messages[0]["content"] = new_prompt
        else:
            self._messages.insert(0, {"role": "system", "content": new_prompt})

    # ========== 事件回调 ==========

    def on_event(self, event_type: AgentEventType, data: Dict[str, Any]):
        """ReAct 事件回调"""
        sep = "─" * 50

        if event_type == AgentEventType.ACTION:
            tool_name = data.get("tool_name", "unknown")
            tool_args = data.get("tool_args", {})
            if tool_name == "update_emotion":
                reason = tool_args.get("reason", "")
                logger.info(f"\n{sep}\n[EMOTION] 更新情绪: {reason}")
            else:
                logger.info(f"\n{sep}\n[ACTION] {tool_name}")

        elif event_type == AgentEventType.OBSERVATION:
            result = data.get("result")
            if result and result.success:
                logger.debug(f"[OBSERVATION] {result.to_string()}")

        elif event_type == AgentEventType.FINISH:
            answer = data.get("answer", "")
            if len(answer) > 200:
                answer = answer[:200] + "..."
            logger.info(f"\n{sep}\n[REPLY]\n{answer}")
            logger.info(f"[EMOTION] {self._emotion}")

    # ========== 便捷方法 ==========

    def set_persona(self, persona: Persona) -> "CharacterAgent":
        """更换人设"""
        self._persona = persona
        self._refresh_system_prompt()
        return self

    def reset_emotion(self) -> "CharacterAgent":
        """重置情绪到默认值"""
        self._emotion = default_emotion()
        return self

    def __repr__(self) -> str:
        return (
            f"CharacterAgent(bot_id={self.bot_id!r}, "
            f"persona={self._persona.name!r}, "
            f"emotion={self._emotion})"
        )
