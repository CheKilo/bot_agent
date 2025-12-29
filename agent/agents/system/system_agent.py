# -*- coding: utf-8 -*-
"""
System Agent - 系统调度 Agent

基于 ReAct 架构的系统调度 Agent，协调 Memory Agent 和 Character Agent。

职责：
1. 维护对话上下文（_messages）
2. 通过 ReAct 架构调度子 Agent
3. 窗口满时触发摘要存储

设计说明：
- 使用 CallAgentTool 工具调用其他 Agent
- CallAgentTool 自动注入 conversation_history 到 metadata
- 通过生命周期钩子管理对话历史：
  - _on_user_input: 用户输入到达时，记录到 _messages
  - _on_final_answer: 最终答案生成后，记录回复并触发摘要
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from agent.agents.base import Agent, AgentEventType, AgentResult
from agent.agents.protocol import AgentRegistry
from agent.agents.system.config import SystemConfig
from agent.agents.system.summarizer import ConversationSummarizer
from agent.client import StorageClient
from agent.tools import Tool

logger = logging.getLogger(__name__)


# ============================================================================
# System Prompt
# ============================================================================

SYSTEM_PROMPT_TEMPLATE = """你是一个对话系统调度Agent，负责协调记忆检索和角色回复生成。

## 可用的 Agent
{agent_descriptions}

## 强制执行规则（必须严格遵守）
1. **禁止直接输出 Final Answer**：你必须先调用工具，不能跳过工具调用
2. **必须调用 memory_agent**：无论用户说什么，都要先检索记忆
3. **必须调用 character_agent**：只有 character_agent 才能生成角色回复
4. **正确的执行顺序**：memory_agent → character_agent → Final Answer

## 完整执行示例（必须严格按此格式）

用户输入："今天天气不错"

### 第一步：检索记忆
Thought: 我需要先调用 memory_agent 检索与用户问题相关的记忆
Action: call_agent
Action Input: {{"agent_name": "memory_agent", "input": "今天天气不错"}}

[等待 Observation 返回]

### 第二步：生成回复
Thought: 记忆检索完成，现在调用 character_agent 生成角色回复
Action: call_agent
Action Input: {{"agent_name": "character_agent", "input": "今天天气不错", "memory_context": "[从上一步获取的记忆内容]"}}

[等待 Observation 返回]

### 第三步：输出结果
Thought: 角色回复已生成，现在输出最终答案
Final Answer: [character_agent 返回的回复内容]

## 工具参数说明
call_agent 工具参数：
- agent_name: Agent 名称（必需），如 "memory_agent" 或 "character_agent"
- input: 输入内容（必需），通常就是用户的原始输入
- memory_context: 记忆上下文（可选），仅在调用 character_agent 时需要传递

**重要**：conversation_history 会自动注入，无需手动传递！

## 关键约束
- **绝对禁止**在没有调用 memory_agent 和 character_agent 的情况下直接输出 Final Answer
- **Thought 后必须紧跟 Action 和 Action Input**，不能只输出 Thought 就结束
- Final Answer 的内容必须来自 character_agent 的返回结果
- 调用 character_agent 时，必须将 memory_agent 返回的记忆内容通过 memory_context 参数传递

{conversation_context}"""


# ============================================================================
# System Agent
# ============================================================================


class SystemAgent(Agent):
    """
    系统调度 Agent（有状态）

    职责：
    1. 维护对话上下文（_messages）
    2. 协调 Memory Agent 和 Character Agent
    3. 窗口满时触发摘要存储（通过 _on_final_answer 钩子）

    使用示例：
        ```python
        registry = AgentRegistry()
        registry.register(memory_agent)
        registry.register(character_agent)

        agent = SystemAgent(
            bot_id="test",
            user_id="user123",
            registry=registry,
            storage_client=storage_client,
        )

        # 直接调用 run 方法
        result = agent.run("你好")
        print(result.answer)
        ```
    """

    name = "system_agent"

    def __init__(
        self,
        bot_id: str,
        user_id: str,
        registry: AgentRegistry,
        storage_client: StorageClient,
        config: Optional[SystemConfig] = None,
    ):
        self._registry = registry
        self._storage = storage_client
        self._user_id = user_id
        self._config = config or SystemConfig()

        # 摘要器（懒加载）
        self._summarizer: Optional[ConversationSummarizer] = None

        # 初始化 CallAgentTool 工具（传入 _messages 引用）
        # 注意：必须在 super().__init__() 之前初始化，因为父类会调用 get_tools()
        from agent.agents.system.tools.call_agent import CallAgentTool

        # 先创建一个空的 _messages 列表，供 CallAgentTool 引用
        # 父类不会覆盖已存在的 _messages
        self._messages: List[Dict[str, Any]] = []
        self._call_tool = CallAgentTool(self._registry, self._messages)

        llm_cfg = self._config.llm
        super().__init__(
            llm_address=llm_cfg.address,
            model=llm_cfg.model,
            bot_id=bot_id,
            message_window=self._config.conversation.message_window,
        )

    # ==================== 属性 ====================

    @property
    def user_id(self) -> str:
        return self._user_id

    @property
    def registry(self) -> AgentRegistry:
        return self._registry

    @property
    def config(self) -> SystemConfig:
        return self._config

    @property
    def max_iterations(self) -> int:
        return self._config.max_iterations

    @property
    def summarizer(self) -> ConversationSummarizer:
        """摘要器（懒加载）"""
        if self._summarizer is None:
            summary_cfg = self._config.summary_llm
            storage_cfg = self._config.storage
            self._summarizer = ConversationSummarizer(
                storage_client=self._storage,
                llm_address=summary_cfg.address,
                llm_model=summary_cfg.model,
                database=storage_cfg.mysql_database,
                table=storage_cfg.mysql_table,
                llm_timeout=summary_cfg.timeout,
            )
        return self._summarizer

    # ==================== Agent 钩子实现 ====================

    def get_system_prompt(self) -> str:
        """获取系统提示词"""
        descriptions = self._registry.get_descriptions()
        desc_lines = [f"- {name}: {desc}" for name, desc in descriptions.items()]
        agent_desc_text = "\n".join(desc_lines) if desc_lines else "（无可用 Agent）"

        # 构建对话上下文
        conversation_context = self._format_conversation_context()

        return SYSTEM_PROMPT_TEMPLATE.format(
            agent_descriptions=agent_desc_text,
            conversation_context=conversation_context,
        )

    def get_tools(self) -> List[Tool]:
        """返回工具列表"""
        return [self._call_tool]

    # ==================== 生命周期钩子实现 ====================

    def _on_user_input(self, user_input: str):
        """
        用户输入到达时的回调

        职责：将用户输入添加到 _messages，确保子 Agent 能获取历史对话
        """
        now = datetime.now().isoformat()
        self._messages.append({"role": "user", "content": user_input, "timestamp": now})

    def _on_final_answer(self, answer: str):
        """
        最终答案生成后的回调

        职责：
        1. 记录 assistant 回复到 _messages
        2. 裁剪消息（可能触发摘要）
        """
        now = datetime.now().isoformat()
        self._messages.append(
            {"role": "assistant", "content": answer, "timestamp": now}
        )

        # 裁剪消息（可能触发摘要）
        self._trim_messages()

    def _format_conversation_context(self) -> str:
        """格式化对话上下文"""
        if not self._messages:
            return ""

        lines = ["\n## 当前对话上下文"]
        for msg in self._messages[-10:]:
            role = msg.get("role", "")
            content = msg.get("content", "")[:200]
            if role == "user":
                lines.append(f"用户: {content}")
            elif role == "assistant":
                lines.append(f"助手: {content}")
        return "\n".join(lines)

    def _trim_messages(self):
        """裁剪消息，窗口满时触发摘要"""
        if not self._config.conversation.auto_summary:
            super()._trim_messages()
            return

        user_count = sum(1 for m in self._messages if m.get("role") == "user")

        if user_count < self._message_window:
            return

        logger.info(f"[SystemAgent] 对话窗口已满（{user_count}轮），触发摘要...")

        messages_to_summarize = self._messages.copy()

        try:
            success = self.summarizer.summarize_and_save(
                bot_id=self.bot_id,
                user_id=self._user_id,
                messages=messages_to_summarize,
            )

            if success:
                logger.info("[SystemAgent] 摘要保存成功，清空对话历史")
                self._messages.clear()  # 使用 clear() 保持引用不变
            else:
                logger.warning("[SystemAgent] 摘要保存失败")
                super()._trim_messages()
        except Exception as e:
            logger.error(f"[SystemAgent] 触发摘要失败: {e}")
            super()._trim_messages()

    # ==================== 事件回调 ====================

    def on_event(self, event_type: AgentEventType, data: Dict[str, Any]):
        """事件回调"""
        sep = "─" * 50

        if event_type == AgentEventType.THOUGHT:
            logger.info(f"\n{sep}\n[System THOUGHT]\n{data.get('thought', '')}")

        elif event_type == AgentEventType.ACTION:
            tool_name = data.get("tool_name", "")
            tool_args = data.get("tool_args", {})
            logger.info(f"\n{sep}\n[System ACTION] {tool_name}")
            for k, v in tool_args.items():
                v_str = str(v)[:200] if v else ""
                logger.info(f"   {k}: {v_str}")

        elif event_type == AgentEventType.OBSERVATION:
            result = data.get("result")
            logger.info(
                f"\n{sep}\n[System OBSERVATION]\n{str(result)[:500] if result else ''}"
            )

        elif event_type == AgentEventType.FINISH:
            answer = data.get("answer", "")
            logger.info(f"\n{sep}\n[System FINISH]\n{answer[:500]}\n{sep}")

    # ==================== 便捷方法 ====================

    def clear_history(self) -> "SystemAgent":
        """清空对话历史"""
        self._messages.clear()  # 使用 clear() 保持引用不变
        self._loop_messages = []
        logger.info("[SystemAgent] 对话历史已清空")
        return self

    def close(self):
        """关闭资源"""
        if self._summarizer:
            self._summarizer.close()
            self._summarizer = None
        super().close()

    def __repr__(self) -> str:
        agents = self._registry.list_agents()
        return f"SystemAgent(bot_id={self.bot_id!r}, agents={agents})"
