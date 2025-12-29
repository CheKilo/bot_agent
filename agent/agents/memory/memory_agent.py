# -*- coding: utf-8 -*-
"""
记忆 Agent

基于 ReAct 架构的记忆管理 Agent。
三级记忆架构：短期（窗口）→ 中期（摘要）→ 长期（向量）

设计说明：
- 无状态：每次 invoke() 都是独立的 ReAct 循环
- 上下文传递：通过 invoke() 的 metadata.conversation_history 获取对话历史
- 输出传递：通过 AgentResponse.metadata.memory_context 返回记忆结果
"""

import logging
from typing import Callable, Dict, List, Optional

from agent.agents.base import Agent, AgentEventType, AgentResult
from agent.agents.protocol import AgentProtocol, AgentMessage, AgentResponse
from agent.client import StorageClient
from agent.core import LLM
from agent.tools import Tool

from agent.agents.memory.config import MemoryConfig, RECENT_SUMMARY_COUNT
from agent.agents.memory.manager import MemoryManager
from agent.agents.memory.tools import (
    SearchMemory,
    StoreLongTermMemory,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是记忆检索和存储模块，职责是检索相关记忆并存储重要信息。

## 行为边界
仅执行：记忆检索和存储操作
不执行：直接回答问题、闲聊、提供建议

## 记忆架构
1. **短期记忆**：当前对话上下文（已在下方提供，无需检索）
2. **中期记忆**：历史对话摘要 → 使用 search_memory
3. **长期记忆**：用户偏好/事实/事件 → 使用 search_memory

## 强制工作流程（必须严格遵守）
**第一步：必须先检索记忆**
- 无论用户说什么，都必须先调用 search_memory 检索相关记忆
- 检索可以帮助判断信息是否已存在、是否需要更新

**第二步：根据检索结果决定是否存储**
- 只有在检索完成后，才能决定是否需要存储新信息
- 如果信息已存在，不要重复存储

**第三步：输出结果**
- 汇总检索到的相关记忆和存储结果

## 检索策略
1. **先观察短期记忆**：查看下方的"当前对话上下文"，了解近期聊了什么
2. **必须检索中长期**：调用 search_memory 检索相关的中期摘要和长期记忆

## 存储规则
**必须存储**：
- 用户明确要求记住的信息
- 用户表达的偏好、习惯（如"我喜欢..."、"我习惯..."）
- 重要事实（如用户的姓名、职业、关系等）
- 重要事件（如会议、约会、计划等）

**不存储**：
- 日常闲聊、简单问候（如"今天天气还行"）
- 已存在的重复信息（检索后发现已有相同内容）

**存储格式要求**：
- 内容简洁明了，只保留核心信息
- 示例："用户喜欢喝美式咖啡"、"用户名字是小明"、"用户在北京工作"

## 关键约束（必须遵守）
1. **禁止跳过检索**：必须先调用 search_memory，再决定是否存储
2. **禁止直接存储**：不能在没有检索的情况下直接调用 store_long_term_memory
3. 如果在 Thought 中决定要执行某个操作，必须紧跟着输出 Action 和 Action Input
4. 只有当所有必要的工具调用都完成后，才能输出 Final Answer

## 正确示例
用户输入："我叫小明"

Thought: 我需要先检索是否已有用户姓名的记忆
Action: search_memory
Action Input: {{"query": "用户姓名 名字", "time_range_days": 365, "limit": 5}}

[等待 Observation]

Thought: 检索结果显示没有用户姓名的记录，需要存储这个重要信息
Action: store_long_term_memory
Action Input: {{"content": "用户名字是小明", "memory_type": "fact", "importance": 9}}

[等待 Observation]

Thought: 已完成检索和存储
Final Answer: ...

{recent_summaries}

{conversation_context}"""

# 输出 Schema
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "related_memory": {
            "type": "object",
            "properties": {
                "short_term": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "短期记忆（从当前对话上下文中提取的相关内容）",
                },
                "mid_term": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "中期摘要列表",
                },
                "long_term": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "长期记忆列表",
                },
            },
            "required": ["short_term", "mid_term", "long_term"],
        },
        "storage_result": {
            "type": "object",
            "properties": {
                "stored": {"type": "boolean"},
                "content": {"type": "string"},
            },
            "required": ["stored", "content"],
        },
    },
    "required": ["related_memory", "storage_result"],
}


class MemoryAgent(Agent, AgentProtocol):
    """
    记忆 Agent（无状态）

    上下文传递：
    - 输入：通过 invoke() 的 metadata.conversation_history 获取对话历史
    - 输出：通过 AgentResponse.metadata.memory_context 返回记忆结果

    使用示例：
        ```python
        agent = MemoryAgent(...)

        # 通过 invoke 调用
        response = agent.invoke(AgentMessage(
            content="用户的问题",
            metadata={"conversation_history": [...]}
        ))

        # 获取记忆结果
        memory_context = response.metadata.get("memory_context")
        ```
    """

    name = "memory_agent"
    max_iterations = 10

    # ==================== AgentProtocol 实现 ====================

    @property
    def agent_name(self) -> str:
        return "memory_agent"

    @property
    def agent_description(self) -> str:
        return (
            "记忆检索和存储 Agent，负责检索相关记忆（中期摘要/长期记忆）、存储重要信息"
        )

    def invoke(self, message: AgentMessage) -> AgentResponse:
        """
        统一调用入口

        Args:
            message: 输入消息
                - content: 用户输入/查询
                - metadata.conversation_history: 对话历史

        Returns:
            AgentResponse:
                - content: 格式化的记忆结果
                - metadata.memory_context: 记忆上下文（供其他 Agent 使用）
        """
        try:
            logger.info(f"[MemoryAgent] invoke 开始，content={message.content[:50]}")

            # 从 metadata 获取对话历史，保存为实例变量
            self._conversation_history = message.get("conversation_history", [])
            logger.info(
                f"[MemoryAgent] conversation_history 长度: {len(self._conversation_history)}"
            )

            # 执行 ReAct 循环
            logger.info(f"[MemoryAgent] 开始执行 ReAct 循环...")
            result = self.run(message.content)
            logger.info(f"[MemoryAgent] ReAct 循环完成，success={result.success}")

            return AgentResponse(
                content=result.answer,
                metadata={"memory_context": result.answer},
                success=result.success,
                error=result.error,
            )

        except Exception as e:
            logger.error(f"MemoryAgent invoke failed: {e}", exc_info=True)
            return AgentResponse(
                content="",
                success=False,
                error=str(e),
            )

    # ==================== 初始化 ====================

    def __init__(
        self,
        bot_id: str,
        user_id: str,
        storage_client: StorageClient,
        embed_func: Callable[[str], List[float]],
        config: Optional[MemoryConfig] = None,
        llm_address: str = LLM.DEFAULT_ADDRESS,
        model: str = LLM.DEFAULT_MODEL,
        recent_summary_count: int = RECENT_SUMMARY_COUNT,
    ):
        self.user_id = user_id
        self._config = config or MemoryConfig()
        self._recent_summary_count = recent_summary_count

        # 本次调用的对话历史（通过 invoke 设置）
        self._conversation_history: List[Dict] = []

        self._manager = MemoryManager(
            bot_id=bot_id,
            user_id=user_id,
            storage_client=storage_client,
            embed_func=embed_func,
            config=self._config,
        )

        super().__init__(llm=LLM(address=llm_address, model=model), bot_id=bot_id)

    @property
    def manager(self) -> MemoryManager:
        """获取 MemoryManager"""
        return self._manager

    # ==================== 钩子方法实现 ====================

    def get_system_prompt(self) -> str:
        """获取系统提示词"""
        # 获取近期摘要
        summaries = self._manager.get_recent_summaries(self._recent_summary_count)
        summary_text = self._format_summaries(summaries) if summaries else ""

        # 使用实例变量获取对话历史
        conversation_context = self._format_conversation_context(
            self._conversation_history
        )

        return SYSTEM_PROMPT.format(
            recent_summaries=summary_text,
            conversation_context=conversation_context,
        )

    def get_tools(self) -> List[Tool]:
        """返回工具列表"""
        return [
            SearchMemory(self._manager),
            StoreLongTermMemory(self._manager),
        ]

    def _format_conversation_context(self, history: List[Dict]) -> str:
        """格式化对话上下文"""
        if not history:
            return "\n## 当前对话上下文（短期记忆）\n（无历史对话）"

        lines = ["\n## 当前对话上下文（短期记忆）"]
        for msg in history:
            # 兼容字符串格式（如果 LLM 传错了格式）
            if isinstance(msg, str):
                lines.append(f"  {msg}")
                continue
            # 正常的字典格式
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                lines.append(f"用户: {content}")
            elif role == "assistant":
                lines.append(f"助手: {content}")
        return "\n".join(lines)

    def get_response_schema(self) -> dict:
        return RESPONSE_SCHEMA

    def get_finalize_prompt(self, schema_str: str) -> str:
        """生成结构化输出的提示词"""
        return f"""请根据以上 ReAct 对话过程，提取工具返回的信息，按照以下 JSON Schema 输出结果。

## 核心要求（必须严格遵守）
1. **必须从工具返回的 Observation 中提取信息**，不要自己编造或从 Thought 中推断
2. **必须直接输出纯 JSON，不要输出其他任何内容**（包括解释性文字、Markdown 格式 ```json 标记等）
3. 如果没有调用某个工具或工具返回为空，对应字段必须为空数组 []
4. JSON 必须完全符合 Schema 定义，缺少任何字段都会导致解析失败

## 如何提取信息

**重要**：请查看对话历史中每个工具调用后的 Observation（观察结果），从中提取信息。

### short_term 字段
- 从对话上下文的"当前对话上下文（短期记忆）"中提取与当前问题相关的内容
- 如果无相关内容，必须为空数组 []

### mid_term 字段
- **必须从 search_memory 工具的 Observation 中提取**
- 查找 Observation 中的 mid_term 数组
- 如果没有调用该工具或 Observation 中没有 mid_term，必须为空数组 []
- 示例：如果 Observation 返回 {{"mid_term": ["用户昨天提到了会议"]}}，则填写 ["用户昨天提到了会议"]

### long_term 字段
- **必须从 search_memory 工具的 Observation 中提取**
- 查找 Observation 中的 long_term 数组
- 如果没有调用该工具或 Observation 中没有 long_term，必须为空数组 []
- 示例：如果 Observation 返回 {{"long_term": ["用户名字是小明"]}}，则填写 ["用户名字是小明"]

### storage_result 字段
- **stored**：检查是否调用了 store_long_term_memory 工具
  - 如果调用了且 Observation 返回 "记忆已保存"，则为 true
  - 如果没有调用或调用失败，则为 false
- **content**：从 store_long_term_memory 工具的 **Action Input** 中提取 content 字段值
  - 注意：不是从 Observation 中提取，而是从调用工具时的输入参数中提取
  - 未存储时为空字符串
- 示例：如果 Action Input 是 {{"content": "用户喜欢喝咖啡", ...}} 且存储成功，则 {{"stored": true, "content": "用户喜欢喝咖啡"}}

## 完整示例

示例 1（检索到记忆，未存储）：
```json
{{
  "related_memory": {{
    "short_term": ["用户下周有个会议"],
    "mid_term": [],
    "long_term": ["用户下周有个重要会议，昨天Lauv告诉他"]
  }},
  "storage_result": {{
    "stored": false,
    "content": ""
  }}
}}
```

示例 2（无记忆，存储了新信息）：
```json
{{
  "related_memory": {{
    "short_term": [],
    "mid_term": [],
    "long_term": []
  }},
  "storage_result": {{
    "stored": true,
    "content": "用户名字是小明"
  }}
}}
```

## JSON Schema
{schema_str}

请直接输出纯 JSON，不要输出其他任何内容（包括 ```json 标记）。"""

    def format_final_output(self, data: dict) -> str:
        """格式化输出"""
        lines = ["[相关记忆]"]
        mem = data.get("related_memory", {})
        has_memory = False

        short_term = mem.get("short_term", [])
        if short_term and isinstance(short_term, list):
            lines.append("- [短期记忆]：")
            for item in short_term:
                lines.append(f"  · {item}")
            has_memory = True

        mid_term = mem.get("mid_term", [])
        if mid_term and isinstance(mid_term, list):
            lines.append("- [中期记忆]：")
            for item in mid_term:
                lines.append(f"  · {item}")
            has_memory = True

        long_term = mem.get("long_term", [])
        if long_term and isinstance(long_term, list):
            lines.append("- [长期记忆]：")
            for item in long_term:
                lines.append(f"  · {item}")
            has_memory = True

        if not has_memory:
            lines.append("- [无相关记忆]")

        lines.append("")
        lines.append("[存储结果]")
        storage = data.get("storage_result", {})
        if storage.get("stored") and storage.get("content"):
            lines.append(f"- [已存储]：{storage['content']}")
        else:
            lines.append("- [无需存储]")

        return "\n".join(lines)

    # ==================== 业务方法 ====================

    def _format_summaries(self, summaries: List[dict]) -> str:
        """格式化摘要"""
        lines = ["===近期摘要==="]
        for i, s in enumerate(summaries, 1):
            kw, sm = s.get("keywords", ""), s.get("summary", "")
            lines.append(f"[{i}] {kw}: {sm}" if kw else f"[{i}] {sm}")
        lines.append("===摘要结束===")
        return "\n".join(lines)

    def on_event(self, event_type: AgentEventType, data: dict):
        """事件回调"""
        sep = "─" * 40
        if event_type == AgentEventType.THOUGHT:
            logger.info(f"\n{sep}\n[Memory THOUGHT]\n{data.get('thought', '')}")
        elif event_type == AgentEventType.ACTION:
            logger.info(f"\n{sep}\n[Memory ACTION] {data.get('tool_name')}")
            for k, v in data.get("tool_args", {}).items():
                logger.info(f"   {k}: {str(v)[:200]}")
        elif event_type == AgentEventType.OBSERVATION:
            result = data.get("result")
            logger.info(
                f"\n{sep}\n[Memory OBSERVATION]\n{str(result)[:500] if result else ''}"
            )
        elif event_type == AgentEventType.FINISH:
            logger.info(f"\n{sep}\n[Memory FINISH]\n{data.get('answer', '')[:800]}")

    def close(self):
        self._manager.close()
        super().close()

    def __repr__(self) -> str:
        return f"MemoryAgent(bot_id={self.bot_id!r}, user_id={self.user_id!r})"
