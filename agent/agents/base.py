# -*- coding: utf-8 -*-
"""
ReAct Agent 基类

基于 ReAct (Reasoning + Acting) 架构的 Agent 基类。
子类只需实现 get_system_prompt() 和 get_tools() 即可。

ReAct 循环：
    1. Thought: LLM 分析问题，决定下一步
    2. Action: 执行工具调用
    3. Observation: 获取工具结果
    4. 循环直到得出最终答案

使用示例：
    ```python
    from agent.agents import Agent
    from agent.tools import Tool, ToolResult

    class MyTool(Tool):
        name = "my_tool"
        description = "我的工具"
        parameters = {"type": "object", "properties": {}}

        def execute(self) -> ToolResult:
            return ToolResult.ok("done")

    class MyAgent(Agent):
        name = "my_agent"

        def get_system_prompt(self) -> str:
            return "你是一个助手"

        def get_tools(self) -> list:
            return [MyTool()]

    # 使用
    agent = MyAgent()
    answer = agent.run("你好")
    ```
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

from agent.core import LLM, Message, LLMResponse
from agent.tools import Tool, ToolKit, ToolResult

logger = logging.getLogger(__name__)

# 默认消息窗口容量
DEFAULT_MESSAGE_WINDOW = 20


# ============================================================================
# Agent 运行结果
# ============================================================================


class AgentEventType(Enum):
    """Agent 事件类型"""

    THOUGHT = "thought"  # 思考
    ACTION = "action"  # 执行工具前
    OBSERVATION = "observation"  # 工具执行后
    FINISH = "finish"  # 完成


@dataclass
class AgentResult:
    """Agent 运行结果"""

    answer: str  # 最终回复
    iterations: int = 0  # 迭代次数
    tool_calls: List[Dict] = field(default_factory=list)  # 所有工具调用记录
    success: bool = True
    error: Optional[str] = None

    def __str__(self) -> str:
        return self.answer


# ============================================================================
# ReAct Agent 基类
# ============================================================================


class Agent(ABC):
    """
    ReAct Agent 基类

    子类必须实现：
        - get_system_prompt(): 返回系统提示词
    可选实现：
        - get_tools(): 返回工具列表
    """

    name: str = "agent"
    max_iterations: int = 10  # 最大 ReAct 迭代次数
    message_window: int = DEFAULT_MESSAGE_WINDOW  # 消息窗口容量

    def __init__(
        self,
        llm_address: str = LLM.DEFAULT_ADDRESS,
        model: str = LLM.DEFAULT_MODEL,
        llm: Optional[LLM] = None,
        bot_id: str = "default_bot",
        message_window: Optional[int] = None,
    ):
        """
        初始化 Agent

        Args:
            llm_address: LLM 服务地址
            model: 模型名称
            llm: 可选，直接传入 LLM 实例
            bot_id: 机器人 ID
        """
        self._llm = llm or LLM(address=llm_address, model=model)
        self._toolkit = ToolKit(self.get_tools())
        self._messages: List[Dict] = []
        self._bot_id: str = bot_id  # 机器人 ID
        self._message_window = message_window or self.message_window
        self._init_system_prompt()

    # ========== 子类实现 ==========

    @abstractmethod
    def get_system_prompt(self) -> str:
        """返回系统提示词"""
        pass

    def get_tools(self) -> List[Tool]:
        """返回工具列表（可选）"""
        return []

    # ========== 属性 ==========

    @property
    def llm(self) -> LLM:
        return self._llm

    @property
    def toolkit(self) -> ToolKit:
        return self._toolkit

    @property
    def messages(self) -> List[Dict]:
        return self._messages

    @property
    def bot_id(self) -> str:
        return self._bot_id

    # ========== 消息管理 ==========

    def _add_message(self, message: Dict):
        """
        添加消息并自动维护窗口大小

        保留 system 消息 + 最近 message_window 条对话消息
        """
        self._messages.append(message)
        self._trim_messages()

    def _trim_messages(self):
        """
        裁剪消息列表，保持在窗口容量内

        策略：超出窗口容量时，移除最前面的对话组
        - 以 user 消息作为对话组的起点
        - 一个对话组包含：user + 后续所有消息（assistant、tool_calls、tool 等），直到下一个 user
        - 每次只移除一组，保持窗口满

        子类可重写此方法实现特殊需求（如 MemoryAgent 需要保存摘要）
        """
        # 检查是否有 system 消息
        has_system = self._messages and self._messages[0]["role"] == "system"
        msg_start = 1 if has_system else 0

        # 找出所有 user 消息的位置（作为对话组的起点）
        user_positions = [
            i
            for i, m in enumerate(self._messages)
            if i >= msg_start and m.get("role") == "user"
        ]

        # 如果对话组数量未超出窗口容量，无需处理
        if len(user_positions) <= self._message_window:
            return

        # 超出窗口容量：移除最前面的一组对话
        # 第一组的起点是 user_positions[0]，终点是 user_positions[1] - 1（或到末尾）
        first_group_start = user_positions[0]
        first_group_end = (
            user_positions[1] if len(user_positions) > 1 else len(self._messages)
        )

        # 移除第一组对话
        del self._messages[first_group_start:first_group_end]

        logger.debug(
            f"Trimmed 1 dialog group ({first_group_end - first_group_start} messages), "
            f"window now has {len(user_positions) - 1} groups"
        )

    # ========== ReAct 核心方法 ==========

    def _react_loop(
        self, tools: Optional[List[Dict]], use_stream_final: bool = False
    ) -> Generator[Tuple[str, AgentResult], None, None]:
        """
        ReAct 核心循环（内部方法）

        Args:
            tools: 工具 schema 列表
            use_stream_final: 最终回答是否使用流式输出

        Yields:
            Tuple[str, AgentResult]: (流式文本块, None) 或 ("", 最终结果)
        """
        tool_call_records = []

        for iteration in range(self.max_iterations):
            # 无工具场景且需要流式输出
            if not tools and use_stream_final and iteration == 0:
                full_content = ""
                for chunk in self._llm.stream(self._messages):
                    full_content += chunk
                    yield (chunk, None)

                self._add_message({"role": "assistant", "content": full_content})
                self.on_event(AgentEventType.FINISH, {"answer": full_content})
                yield (
                    "",
                    AgentResult(
                        answer=full_content, iterations=1, tool_calls=tool_call_records
                    ),
                )
                return

            # LLM 决策
            response = self._llm.chat(self._messages, tools=tools)

            # 无工具调用 = 最终答案
            if not response.has_tool_calls:
                answer = response.content or ""
                self._add_message({"role": "assistant", "content": answer})
                self.on_event(AgentEventType.FINISH, {"answer": answer})
                yield (
                    "",
                    AgentResult(
                        answer=answer,
                        iterations=iteration + 1,
                        tool_calls=tool_call_records,
                    ),
                )
                return

            # 有工具调用
            self._add_message(
                {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": response.tool_calls,
                }
            )

            if response.content:
                self.on_event(AgentEventType.THOUGHT, {"thought": response.content})

            # 执行工具
            pending_calls = [
                (call_id, name, args)
                for _, call_id, name, args in response.iter_tool_calls()
            ]
            for _, name, args in pending_calls:
                self.on_event(
                    AgentEventType.ACTION, {"tool_name": name, "tool_args": args}
                )

            for call_id, name, result in self._toolkit.execute(pending_calls):
                args = next(a for cid, _, a in pending_calls if cid == call_id)
                self.on_event(
                    AgentEventType.OBSERVATION, {"tool_name": name, "result": result}
                )
                tool_call_records.append(
                    {"name": name, "args": args, "result": result.to_string()}
                )
                self._add_message(
                    {
                        "role": "tool",
                        "content": result.to_string(),
                        "tool_call_id": call_id,
                    }
                )

        # 超过最大迭代次数
        logger.warning(
            f"Agent {self.name} exceeded max iterations {self.max_iterations}"
        )
        yield (
            "",
            AgentResult(
                answer="",
                iterations=self.max_iterations,
                tool_calls=tool_call_records,
                success=False,
                error="Exceeded max iterations",
            ),
        )

    def run(self, user_input: str) -> AgentResult:
        """
        执行 ReAct 循环

        Args:
            user_input: 用户输入

        Returns:
            AgentResult: 包含最终回复和执行信息
        """
        self._add_message({"role": "user", "content": user_input})
        tools = self._toolkit.get_schemas() or None

        for _, result in self._react_loop(tools, use_stream_final=False):
            if result is not None:
                return result

        # 理论上不会到这里
        return AgentResult(answer="", success=False, error="Unknown error")

    def run_stream(self, user_input: str) -> Generator[str, None, AgentResult]:
        """
        流式执行 ReAct 循环

        Args:
            user_input: 用户输入

        Yields:
            str: 流式输出的文本块

        Returns:
            AgentResult: 最终结果
        """
        self._add_message({"role": "user", "content": user_input})
        tools = self._toolkit.get_schemas() or None

        for chunk, result in self._react_loop(tools, use_stream_final=True):
            if chunk:
                yield chunk
            if result is not None:
                return result

        return AgentResult(answer="", success=False, error="Unknown error")

    # ========== 对话管理 ==========

    def clear_history(self) -> "Agent":
        """清空对话历史（保留系统提示词）"""
        if self._messages and self._messages[0]["role"] == "system":
            self._messages = [self._messages[0]]
        else:
            self._messages = []
            self._init_system_prompt()
        return self

    # ========== 事件回调（子类可覆盖） ==========

    def on_event(self, event_type: AgentEventType, data: Dict[str, Any]):
        """
        统一事件回调（子类覆盖此方法以自定义行为）

        Args:
            event_type: 事件类型 (THOUGHT/ACTION/OBSERVATION/FINISH)
            data: 事件数据
        """
        logger.debug(f"[{self.name}] {event_type.value}: {data}")

    # ========== 内部方法 ==========

    def _init_system_prompt(self):
        """初始化系统提示词"""
        prompt = self.get_system_prompt()
        if prompt:
            self._messages = [{"role": "system", "content": prompt}]

    # ========== 上下文管理 ==========

    def close(self):
        """关闭资源"""
        if self._llm:
            self._llm.close()

    def __enter__(self) -> "Agent":
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, tools={self._toolkit.names})"
