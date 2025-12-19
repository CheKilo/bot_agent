# -*- coding: utf-8 -*-
"""
记忆 Agent

基于 ReAct 架构的记忆管理 Agent。
通过工具调用实现记忆的检索和存储。

架构：
- 窗口满时自动触发摘要存储
- 每次对话自动携带最近N条摘要作为上下文
- Agent 按需调用工具检索更多记忆
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from agent.agents.base import Agent, AgentEventType, AgentResult
from agent.client import StorageClient
from agent.core import LLM
from agent.tools import Tool

from agent.agents.memory.config import MESSAGE_WINDOW_CAPACITY, RECENT_SUMMARY_COUNT
from agent.agents.memory.manager import MemoryManager
from agent.agents.memory.tools import (
    SearchMidTermMemory,
    SearchLongTermMemory,
    StoreLongTermMemory,
)

logger = logging.getLogger(__name__)

# System Prompt 模板
SYSTEM_PROMPT_TEMPLATE = """你是记忆检索模块，职责是检索相关记忆并按格式输出。

## 行为约束
禁止：回答问题、闲聊、提供建议
允许：检索记忆、格式化输出、判断是否需要存储

## 检索时机
需要检索：
- 回溯性表达（之前/上次/你还记得吗）
- 提到人名但上下文无信息
- 涉及用户偏好/习惯/个人信息但上下文无

无需检索，输出"[无相关记忆]"：
- 通用问题（天气/时间/知识问答）
- 当前窗口已有足够信息

## 存储时机
用户明确要求记住，或涉及：重要事件、偏好习惯、特殊日期

## 输出格式
无记忆：[无相关记忆]
有记忆：
[相关记忆]
1. 【类型】内容
存储成功：[已记录] 已保存：内容摘要
信息不足：[信息不足] 未找到关于XXX的记忆"""


class MemoryAgent(Agent):
    """
    记忆 Agent

    三级记忆架构：
    - 短期：消息窗口（Agent 基类管理）
    - 中期：MySQL 摘要（窗口满时触发）
    - 长期：Milvus 向量存储（工具调用）
    """

    name = "memory_agent"
    max_iterations = 10
    message_window = MESSAGE_WINDOW_CAPACITY

    def __init__(
        self,
        bot_id: str,
        user_id: str,
        storage_client: StorageClient,
        embed_func: Callable[[str], List[float]],
        llm_address: str = LLM.DEFAULT_ADDRESS,
        model: str = LLM.DEFAULT_MODEL,
        recent_summary_count: int = RECENT_SUMMARY_COUNT,
    ):
        self.user_id = user_id
        self._recent_summary_count = recent_summary_count
        llm = LLM(address=llm_address, model=model)

        self._manager = MemoryManager(
            bot_id=bot_id,
            user_id=user_id,
            storage_client=storage_client,
            embed_func=embed_func,
        )

        super().__init__(llm=llm, bot_id=bot_id)

    def get_system_prompt(self) -> str:
        """获取系统提示词（动态注入最近摘要）"""
        recent_summaries = self._manager.get_recent_summaries(
            self._recent_summary_count
        )

        if recent_summaries:
            summary_text = self._format_summaries(recent_summaries)
            return f"{SYSTEM_PROMPT_TEMPLATE}\n\n{summary_text}"

        return SYSTEM_PROMPT_TEMPLATE

    def _format_summaries(self, summaries: List[dict]) -> str:
        """格式化摘要为上下文"""
        lines = ["===近期摘要==="]
        for i, s in enumerate(summaries, 1):
            keywords = s.get("keywords", "")
            summary = s.get("summary", "")
            if keywords:
                lines.append(f"[{i}] {keywords}: {summary}")
            else:
                lines.append(f"[{i}] {summary}")
        lines.append("===摘要结束===")
        return "\n".join(lines)

    def get_tools(self) -> List[Tool]:
        return [
            SearchMidTermMemory(self._manager),
            SearchLongTermMemory(self._manager),
            StoreLongTermMemory(self._manager),
        ]

    def run(self, user_input: str) -> AgentResult:
        """执行 ReAct 循环"""
        self._refresh_system_prompt()
        return super().run(user_input)

    def _trim_messages(self):
        """
        窗口满时：保存所有对话为摘要后清空

        策略：对话组数达到窗口容量时
        1. 生成摘要存入中期记忆
        2. 清空消息窗口（只保留 system prompt）
        3. 刷新 system prompt
        """
        has_system = self._messages and self._messages[0]["role"] == "system"
        msg_start = 1 if has_system else 0

        # 统计对话组数（以 user 消息为起点）
        user_count = sum(
            1 for m in self._messages[msg_start:] if m.get("role") == "user"
        )

        if user_count < self._message_window:
            return

        # 保存所有对话到中期记忆
        all_dialog = self._messages[msg_start:]
        self._save_to_mid_term_memory(all_dialog)

        # 清空并刷新
        self._messages = [self._messages[0]] if has_system else []
        self._refresh_system_prompt()

        logger.info(f"Window full ({user_count} groups), saved and cleared")

    def _save_to_mid_term_memory(self, messages: List[Dict]):
        """将对话消息保存到中期记忆"""
        # 完整原始消息
        raw_messages = [
            {
                "role": m["role"],
                "content": m.get("content", ""),
                **{k: v for k, v in m.items() if k not in ("role", "content")},
            }
            for m in messages
        ]

        # 用于生成摘要的对话（只要 user/assistant 且有内容）
        dialog_for_summary = [
            {"role": m["role"], "content": m.get("content", "")}
            for m in messages
            if m["role"] in ("user", "assistant") and m.get("content")
        ]

        if len(dialog_for_summary) >= 2:
            if self._manager.save_summary(
                dialog_for_summary, raw_messages=raw_messages
            ):
                logger.info(f"Saved {len(raw_messages)} messages as mid-term memory")

    def _refresh_system_prompt(self):
        """刷新系统提示词"""
        new_prompt = self.get_system_prompt()
        if self._messages and self._messages[0]["role"] == "system":
            self._messages[0]["content"] = new_prompt
        else:
            self._messages.insert(0, {"role": "system", "content": new_prompt})

    def on_event(self, event_type: AgentEventType, data: Dict[str, Any]):
        """ReAct 事件回调"""
        sep = "─" * 50

        if event_type == AgentEventType.THOUGHT:
            logger.info(f"\n{sep}\n💭 [THOUGHT]\n{data.get('thought', '')}")

        elif event_type == AgentEventType.ACTION:
            tool_name = data.get("tool_name", "unknown")
            logger.info(f"\n{sep}\n🔧 [ACTION] {tool_name}")
            for k, v in data.get("tool_args", {}).items():
                v_str = str(v)[:200] + "..." if len(str(v)) > 200 else str(v)
                logger.info(f"   {k}: {v_str}")

        elif event_type == AgentEventType.OBSERVATION:
            result = data.get("result")
            result_str = result.to_string() if result else ""
            if len(result_str) > 500:
                result_str = result_str[:500] + "...[截断]"
            logger.info(f"\n{sep}\n👁️ [OBSERVATION]\n{result_str}")

        elif event_type == AgentEventType.FINISH:
            answer = data.get("answer", "")
            if len(answer) > 800:
                answer = answer[:800] + "..."
            logger.info(f"\n{sep}\n✅ [FINISH]\n{answer}\n{sep}")

    def close(self):
        """关闭资源"""
        self._manager.close()
        super().close()

    def __repr__(self) -> str:
        return f"MemoryAgent(bot_id={self.bot_id!r}, user_id={self.user_id!r})"
