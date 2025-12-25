# -*- coding: utf-8 -*-
"""
Character Agent - 角色扮演 Agent

基于 ReAct 架构的角色扮演 Agent。

职责：
1. 协调 ReAct 循环
2. 生成符合人设的角色回复

上下文传递：
- 输入：通过 invoke() 的 metadata 获取
    - memory_context: 记忆上下文（来自 MemoryAgent）
    - conversation_history: 对话历史（来自 SystemAgent）
- 输出：通过 AgentResponse.metadata.emotion_state 返回情绪状态
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent.agents.base import Agent, AgentEventType, AgentResult
from agent.agents.protocol import AgentProtocol, AgentMessage, AgentResponse
from agent.core import LLM
from agent.tools import Tool

from agent.agents.character.config import CharacterConfig
from agent.agents.character.persona import Persona, DEFAULT_PERSONA
from agent.agents.character.tools import (
    AnalyzeEmotion,
    GenerateResponse,
    default_emotion,
    normalize_emotion,
)

logger = logging.getLogger(__name__)


# ============================================================================
# System Prompt
# ============================================================================

SYSTEM_PROMPT_TEMPLATE = """你是一个角色扮演 Agent，负责以下任务：

## 你的角色
{persona}

## 你的任务
1. **分析情绪**：调用 analyze_emotion 工具，获取角色当前的情绪状态
2. **生成回复**：调用 generate_response 工具，生成角色回复
3. **输出结果**：直接输出 Final Answer（内容就是工具返回的回复）

## 强制工作流程（必须严格遵守）
**第一步：分析情绪**
- 调用 analyze_emotion 工具
- 传入 user_input 和 conversation_history
- 观察返回的情绪数值

**第二步：生成回复**
- 调用 generate_response 工具
- 传入 user_input、emotion、persona、memory_context
- 观察返回的回复内容

**第三步：输出 Final Answer（必须！）**
- 完成上述两次工具调用后，必须立即输出 Final Answer
- Final Answer 的内容**就是 generate_response 工具返回的回复内容**
- 禁止继续调用工具，禁止重新生成回复
- 禁止修改或调整回复内容

## 记忆上下文
{memory_context}

## 对话历史
{conversation_context}

## 关键约束（必须遵守）
1. 只能按顺序调用：analyze_emotion → generate_response → Final Answer
2. 完成两次工具调用后，必须立即输出 Final Answer
3. Final Answer 的内容就是 generate_response 工具返回的回复，不做任何修改
4. 禁止在 Final Answer 之前调用任何其他工具
5. 禁止在输出 Final Answer 后继续调用工具

## 正确示例
用户输入："下周会不知道说啥，烦"

Thought: 我需要先分析角色当前的情绪状态
Action: analyze_emotion
Action Input: {{"user_input": "下周会不知道说啥，烦", "conversation_history": [...]}}

[等待 Observation，得到情绪结果]

Thought: 我已经获取到情绪状态，现在生成角色回复
Action: generate_response
Action Input: {{"user_input": "下周会不知道说啥，烦", "emotion": {{"mood": -0.5, ...}}, "persona": "小雪", "memory_context": "..."}}

[等待 Observation，得到回复内容]

Thought: 已完成情绪分析和回复生成
Final Answer: [这里填写 generate_response 工具返回的回复内容，不做任何修改]
"""

# ============================================================================
# 数据类
# ============================================================================


@dataclass
class CharacterResult(AgentResult):
    """Character Agent 运行结果"""

    emotion_state: Dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.answer


# ============================================================================
# Character Agent
# ============================================================================


class CharacterAgent(Agent, AgentProtocol):
    """
    角色扮演 Agent（无状态）

    上下文传递：
    - 输入：通过 invoke() 的 metadata 获取
        - memory_context: 记忆上下文
        - conversation_history: 对话历史
    - 输出：通过 AgentResponse.metadata.emotion_state 返回情绪状态

    使用示例：
        ```python
        agent = CharacterAgent(persona=EXAMPLE_PERSONA)

        response = agent.invoke(AgentMessage(
            content="你好呀！",
            metadata={
                "memory_context": "用户喜欢猫咪",
                "conversation_history": [...]
            }
        ))

        print(response.content)  # 角色回复
        print(response.metadata["emotion_state"])  # 情绪状态
        ```
    """

    name = "character_agent"
    max_iterations = 5

    # ==================== AgentProtocol 实现 ====================

    @property
    def agent_name(self) -> str:
        return "character_agent"

    @property
    def agent_description(self) -> str:
        return f"角色扮演 Agent（{self._persona.name}）"

    def invoke(self, message: AgentMessage) -> AgentResponse:
        """
        统一调用入口

        Args:
            message: 输入消息
                - content: 用户输入
                - metadata.memory_context: 记忆上下文
                - metadata.conversation_history: 对话历史

        Returns:
            AgentResponse:
                - content: 角色回复
                - metadata.emotion_state: 情绪状态
        """
        try:
            # 从 metadata 获取上下文，保存为实例变量
            self._memory_context = message.get("memory_context", "")
            self._conversation_history = message.get("conversation_history", [])

            result = self.run(message.content)

            return AgentResponse(
                content=result.answer,
                metadata={"emotion_state": result.emotion_state},
                success=result.success,
                error=result.error,
            )

        except Exception as e:
            logger.error(f"CharacterAgent invoke failed: {e}")
            return AgentResponse(content="", success=False, error=str(e))

    # ==================== 初始化 ====================

    def __init__(
        self,
        bot_id: str = "default_bot",
        persona: Optional[Persona] = None,
        config: Optional[CharacterConfig] = None,
    ):
        self._persona = persona or DEFAULT_PERSONA
        self._config = config or CharacterConfig()

        # 本次调用的上下文（通过 invoke 设置）
        self._memory_context: str = ""
        self._conversation_history: List[Dict] = []

        llm_cfg = self._config.agent_llm

        super().__init__(
            llm_address=llm_cfg.address,
            model=llm_cfg.model,
            bot_id=bot_id,
        )

    # ==================== 属性 ====================

    @property
    def persona(self) -> Persona:
        return self._persona

    @property
    def config(self) -> CharacterConfig:
        return self._config

    # ==================== 配置方法 ====================

    def set_persona(self, persona: Persona) -> "CharacterAgent":
        """设置人设"""
        self._persona = persona
        return self

    # ==================== Agent 钩子实现 ====================

    def get_system_prompt(self) -> str:
        """获取系统提示词"""
        # 使用实例变量获取上下文
        conversation_context = self._format_conversation_history(
            self._conversation_history
        )

        return SYSTEM_PROMPT_TEMPLATE.format(
            persona=self._persona.to_prompt(),
            persona_name=self._persona.name,
            memory_context=self._memory_context or "[无相关记忆]",
            conversation_context=conversation_context or "[无历史对话]",
        )

    def get_tools(self) -> List[Tool]:
        """返回工具列表"""
        return [
            AnalyzeEmotion(config=self._config.emotion_tool),
            GenerateResponse(config=self._config.response_tool),
        ]

    def _format_conversation_history(self, history: List[Dict]) -> str:
        """格式化对话历史"""
        if not history:
            return ""

        lines = []
        for msg in history[-10:]:
            # 兼容字符串格式（如果 LLM 传错了格式）
            if isinstance(msg, str):
                lines.append(f"  {msg}")
                continue
            # 正常的字典格式
            role = msg.get("role", "")
            content = msg.get("content", "")[:200]
            timestamp = msg.get("timestamp", "")

            time_str = f" ({timestamp})" if timestamp else ""
            if role == "user":
                lines.append(f"用户{time_str}: {content}")
            elif role == "assistant":
                lines.append(f"助手{time_str}: {content}")

        return "\n".join(lines)

    def run(self, user_input: str) -> CharacterResult:
        """执行 ReAct 循环"""
        base_result = super().run(user_input)

        # 从 trace 中提取情绪状态
        emotion_state = self._extract_emotion_from_trace(base_result.trace)

        return CharacterResult(
            answer=base_result.answer,
            iterations=base_result.iterations,
            trace=base_result.trace,
            success=base_result.success,
            error=base_result.error,
            emotion_state=emotion_state,
        )

    def _extract_emotion_from_trace(self, trace: List[Dict]) -> Dict[str, float]:
        """从 ReAct 轨迹中提取情绪状态"""
        for msg in trace:
            content = msg.get("content", "")
            # 查找 Observation 中 analyze_emotion 的结果
            if msg.get("role") == "user" and content.startswith("Observation:"):
                # 尝试从 observation 内容中解析情绪
                obs_content = content[len("Observation:") :].strip()
                try:
                    # ToolResult 的字符串格式可能包含 JSON
                    if "success" in obs_content and "data" in obs_content:
                        # 尝试解析 ToolResult 格式
                        import re

                        json_match = re.search(r"\{.*\}", obs_content, re.DOTALL)
                        if json_match:
                            data = json.loads(json_match.group())
                            if isinstance(data, dict) and any(
                                k in data for k in ["valence", "arousal", "mood"]
                            ):
                                return normalize_emotion(data)
                except (json.JSONDecodeError, TypeError):
                    pass
        return default_emotion()

    def on_event(self, event_type: AgentEventType, data: Dict[str, Any]):
        """事件回调"""
        sep = "─" * 40

        if event_type == AgentEventType.THOUGHT:
            logger.info(f"\n{sep}\n[Character THOUGHT]\n{data.get('thought', '')}")

        elif event_type == AgentEventType.ACTION:
            tool_name = data.get("tool_name", "")
            tool_args = data.get("tool_args", {})
            logger.info(f"\n{sep}\n[Character ACTION] {tool_name}")
            for k, v in tool_args.items():
                v_str = str(v)[:100] if v else ""
                logger.info(f"   {k}: {v_str}")

        elif event_type == AgentEventType.OBSERVATION:
            result = data.get("result")
            logger.info(
                f"\n{sep}\n[Character OBSERVATION]\n{str(result)[:300] if result else ''}"
            )

        elif event_type == AgentEventType.FINISH:
            answer = data.get("answer", "")
            logger.info(f"\n{sep}\n[Character FINISH]\n{answer[:200]}")

    def __repr__(self) -> str:
        return f"CharacterAgent(bot_id={self.bot_id!r}, persona={self._persona.name!r})"
