# -*- coding: utf-8 -*-
"""
ReAct Agent 基类

基于经典 ReAct (Reasoning + Acting) 架构的 Agent 基类。
通过 system prompt 模板强制 LLM 输出 Thought/Action/Final Answer 格式。

设计理念：
- 单一职责：_messages 用于持久化历史，_loop_messages 用于单轮 ReAct 轨迹
- 上下文传递：通过 AgentProtocol 的 invoke() metadata 传递
- 钩子扩展：子类可重写钩子方法定制行为
"""

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Generator, List, Optional, Tuple

from agent.core import LLM
from agent.tools import Tool, ToolKit, ToolResult

logger = logging.getLogger(__name__)

DEFAULT_MESSAGE_WINDOW = 20

# ReAct 输出格式模板
REACT_FORMAT_TEMPLATE = """
## 输出格式（必须严格遵守）

每次回复必须包含以下三行（缺一不可）：
```
Thought: [你的思考]
Action: [工具名称，必须是 {tool_names} 之一]
Action Input: [JSON格式的参数]
```

或者，当所有工具调用完成后：
```
Thought: 已完成所有必要的工具调用
Final Answer: [最终答案]
```

## 格式示例

### 正确示例 ✓
```
Thought: 我需要先检索相关记忆
Action: call_agent
Action Input: {{"agent_name": "memory_agent", "input": "用户的问题"}}
```

### 错误示例 ✗（只有 Thought 没有 Action）
```
Thought: 我需要先检索相关记忆
```
这是错误的！Thought 后面必须紧跟 Action 和 Action Input。

## 强制规则
1. **Thought 后必须紧跟 Action**：不能只输出 Thought 就结束
2. **必须先调用工具**：在输出 Final Answer 之前，必须至少调用一次工具
3. **格式必须精确**：Action: 和 Action Input: 必须各占一行
4. Action 和 Final Answer 不能同时出现
"""


class AgentEventType(Enum):
    """Agent 事件类型"""

    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    FINISH = "finish"


@dataclass
class AgentResult:
    """Agent 运行结果"""

    answer: str
    iterations: int = 0
    trace: List[Dict] = field(default_factory=list)  # 完整的 ReAct 轨迹
    success: bool = True
    error: Optional[str] = None

    def __str__(self) -> str:
        return self.answer


class Agent(ABC):
    """
    ReAct Agent 基类

    消息架构：
    - _messages: 持久化对话历史（有状态 Agent 使用，如 SystemAgent）
    - _loop_messages: 单轮 ReAct 轨迹（每次 run 重置）

    上下文传递：
    - Agent 间通信通过 AgentProtocol.invoke() 的 metadata 传递
    - 子类在 invoke() 中从 metadata 提取所需字段，设置为实例变量
    - 子类在 get_system_prompt() 中使用这些实例变量构建提示词

    子类必须实现：get_system_prompt()
    可选实现：get_tools(), get_response_schema(), format_final_output()
    """

    name: str = "agent"
    max_iterations: int = 10

    def __init__(
        self,
        llm_address: str = LLM.DEFAULT_ADDRESS,
        model: str = LLM.DEFAULT_MODEL,
        llm: Optional[LLM] = None,
        bot_id: str = "default_bot",
        message_window: Optional[int] = None,
    ):
        self._llm = llm or LLM(address=llm_address, model=model)
        self._toolkit = ToolKit(self.get_tools())
        self._bot_id = bot_id
        self._message_window = message_window or DEFAULT_MESSAGE_WINDOW

        # 持久化对话历史（有状态 Agent 使用）
        self._messages: List[Dict] = []

        # 单轮 ReAct 轨迹（每次 run 重置，记录完整的 thought/action/observation）
        self._loop_messages: List[Dict] = []

    # ==================== 子类钩子方法 ====================

    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        返回业务相关的系统提示词（必须实现）

        子类可通过实例变量获取本次调用的上下文（在 invoke 中设置）
        子类可通过 self._messages 获取对话历史
        """
        pass

    def get_tools(self) -> List[Tool]:
        """返回工具列表"""
        return []

    def get_response_schema(self) -> Optional[Dict]:
        """返回最终输出的 JSON Schema（可选）"""
        return None

    def format_final_output(self, data: Dict) -> str:
        """将结构化数据格式化为输出字符串"""
        return json.dumps(data, ensure_ascii=False, indent=2)

    def get_finalize_prompt(self, schema_str: str) -> str:
        """返回生成结构化输出时的提示词（子类可覆盖）"""
        return f"""请根据以上对话内容，按照以下 JSON Schema 输出结果。

## JSON Schema
{schema_str}

请直接输出 JSON，不要输出其他内容。"""

    def on_event(self, event_type: AgentEventType, data: Dict[str, Any]):
        """事件回调（子类可覆盖）"""
        logger.debug(f"[{self.name}] {event_type.value}: {data}")

    # ==================== 属性 ====================

    @property
    def llm(self) -> LLM:
        return self._llm

    @property
    def toolkit(self) -> ToolKit:
        return self._toolkit

    @property
    def messages(self) -> List[Dict]:
        """持久化对话历史"""
        return self._messages

    @property
    def loop_messages(self) -> List[Dict]:
        """单轮 ReAct 轨迹"""
        return self._loop_messages

    @property
    def bot_id(self) -> str:
        return self._bot_id

    @property
    def message_window(self) -> int:
        return self._message_window

    # ==================== 公开方法 ====================

    def run(self, user_input: str) -> AgentResult:
        """执行 ReAct 循环"""
        self._init_loop(user_input)

        for _, result in self._react_loop():
            if result is not None:
                return result

        return AgentResult(answer="", success=False, error="Unknown error")

    def run_stream(self, user_input: str) -> Generator[str, None, AgentResult]:
        """流式执行 ReAct 循环"""
        self._init_loop(user_input)

        for chunk, result in self._react_loop(use_stream_final=True):
            if chunk:
                yield chunk
            if result is not None:
                return result

        return AgentResult(answer="", success=False, error="Unknown error")

    def add_message(self, role: str, content: str) -> "Agent":
        """添加一条消息到持久化历史"""
        self._messages.append({"role": role, "content": content})
        return self

    def clear_history(self) -> "Agent":
        """清空对话历史"""
        self._messages = []
        self._loop_messages = []
        return self

    def close(self):
        """关闭资源"""
        if self._llm:
            self._llm.close()

    def __enter__(self) -> "Agent":
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"

    # ==================== 内部方法 ====================

    def _build_system_prompt(self) -> str:
        """构建完整的系统提示词（业务提示词 + 工具描述 + ReAct格式）"""
        parts = [self.get_system_prompt()]

        if self._toolkit:
            tool_descs = self._toolkit.get_descriptions()
            tool_names = self._toolkit.get_names_str()
            if tool_descs:
                parts.append(f"\n## 可用工具\n{tool_descs}")
                parts.append(REACT_FORMAT_TEMPLATE.replace("{tool_names}", tool_names))
            else:
                parts.append(self._get_no_tool_format())
        else:
            parts.append(self._get_no_tool_format())

        return "\n".join(parts)

    def _get_no_tool_format(self) -> str:
        """无工具时的输出格式"""
        return """
请使用以下格式回答：

Thought: 分析当前情况
Final Answer: 最终答案
"""

    def _init_loop(self, user_input: str):
        """
        初始化单轮 ReAct 循环（子类可重写）

        重置 _loop_messages，开始新的 ReAct 轨迹
        """
        self._loop_messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": user_input},
        ]

    def _trim_messages(self):
        """裁剪持久化消息，保持窗口容量（子类可重写）"""
        user_count = sum(1 for m in self._messages if m.get("role") == "user")

        if user_count <= self._message_window:
            return

        # 移除最早的一组对话（user + assistant）
        if self._messages and self._messages[0].get("role") == "user":
            self._messages.pop(0)
            if self._messages and self._messages[0].get("role") == "assistant":
                self._messages.pop(0)

    def _react_loop(
        self, use_stream_final: bool = False
    ) -> Generator[Tuple[str, AgentResult], None, None]:
        """ReAct 核心循环"""

        # 跟踪是否已调用过工具
        has_called_tool = False

        for iteration in range(self.max_iterations):
            logger.info(
                f"[{self.name}] ReAct 迭代 {iteration + 1}/{self.max_iterations}"
            )
            response = self._llm.chat(self._loop_messages, tools=None)
            content = response.content or ""
            # 输出完整的 LLM 原始返回，方便调试
            logger.info(
                f"[{self.name}] LLM 原始输出:\n--- BEGIN ---\n{content}\n--- END ---"
            )
            parsed = self._parse_react_output(content)
            logger.info(
                f"[{self.name}] 解析结果: thought={bool(parsed.get('thought'))}, action={parsed.get('action')}, action_input={bool(parsed.get('action_input'))}, final_answer={bool(parsed.get('final_answer'))}"
            )
            if parsed.get("final_answer"):
                logger.info(
                    f"[{self.name}] 解析到的 final_answer: {parsed.get('final_answer')[:200]}"
                )

            if parsed.get("thought"):
                self.on_event(AgentEventType.THOUGHT, {"thought": parsed["thought"]})

            has_action = parsed.get("action") and parsed.get("action_input") is not None
            has_final = parsed.get("final_answer") is not None

            # 格式错误：Action 和 Final Answer 同时出现
            if has_action and has_final:
                self._loop_messages.append(
                    {
                        "role": "user",
                        "content": "格式错误：Action 和 Final Answer 不能同时出现。",
                    }
                )
                continue

            # 检查：如果有工具可用，但没有调用任何工具就直接输出 Final Answer
            if (
                has_final
                and not has_called_tool
                and self._toolkit
                and len(self._toolkit) > 0
            ):
                self._loop_messages.append({"role": "assistant", "content": content})
                self._loop_messages.append(
                    {
                        "role": "user",
                        "content": "错误：你必须先调用工具完成任务，然后才能输出 Final Answer。请按照规定的工作流程，先调用必要的工具。",
                    }
                )
                continue

            # 记录 assistant 输出到轨迹
            self._loop_messages.append({"role": "assistant", "content": content})

            # Final Answer
            if has_final:
                answer = self._finalize_output(parsed["final_answer"])
                self._on_final_answer(answer)
                self.on_event(AgentEventType.FINISH, {"answer": answer})

                # 返回结果，trace 为完整的 _loop_messages
                yield "", AgentResult(
                    answer=answer,
                    iterations=iteration + 1,
                    trace=self._loop_messages.copy(),
                )
                return

            # Action
            if has_action:
                action, action_input = parsed["action"], parsed["action_input"]
                # 详细日志：输出工具参数
                logger.info(f"[{self.name}] 准备执行工具: {action}")
                logger.info(f"[{self.name}] 工具参数类型: {type(action_input)}")
                logger.info(f"[{self.name}] 工具参数内容: {action_input}")

                self.on_event(
                    AgentEventType.ACTION,
                    {"tool_name": action, "tool_args": action_input},
                )

                tool = self._toolkit.get(action)
                result = (
                    tool.safe_execute(**action_input)
                    if tool
                    else ToolResult.fail(f"Unknown tool: {action}")
                )

                # 标记已调用过工具
                has_called_tool = True

                self.on_event(
                    AgentEventType.OBSERVATION,
                    {"tool_name": action, "result": result},
                )

                # 记录 Observation 到轨迹
                self._loop_messages.append(
                    {"role": "user", "content": f"Observation: {result}"}
                )
                continue

            # 无 Action 也无 Final Answer
            if iteration < self.max_iterations - 1:
                logger.warning(
                    f"[{self.name}] 本轮无 Action 也无 Final Answer，要求 LLM 继续"
                )
                # 给出更明确的格式纠正提示
                error_prompt = """格式错误！你只输出了 Thought，但没有输出 Action。

请严格按照以下格式输出（三行缺一不可）：

Thought: [你的思考]
Action: [工具名称]
Action Input: [JSON格式的参数]

示例：
Thought: 我需要先检索记忆
Action: call_agent
Action Input: {"agent_name": "memory_agent", "input": "用户的问题", "metadata": {}}

请现在输出正确格式的 Action："""
                self._loop_messages.append(
                    {
                        "role": "user",
                        "content": error_prompt,
                    }
                )

        # 超过最大迭代次数
        yield "", AgentResult(
            answer="",
            iterations=self.max_iterations,
            trace=self._loop_messages.copy(),
            success=False,
            error="Exceeded max iterations",
        )

    def _on_final_answer(self, answer: str):
        """最终答案生成后的回调（子类可重写）"""
        pass

    def _parse_react_output(self, content: str) -> Dict[str, Any]:
        """解析 ReAct 格式输出"""
        result = {
            "thought": None,
            "action": None,
            "action_input": None,
            "final_answer": None,
        }

        thought_match = re.search(
            r"Thought:\s*(.+?)(?=Action:|Final Answer:|$)",
            content,
            re.DOTALL | re.IGNORECASE,
        )
        if thought_match:
            result["thought"] = thought_match.group(1).strip()

        # 先尝试解析 Action（优先级高于 Final Answer）
        action_match = re.search(r"Action:\s*(\S+)", content, re.IGNORECASE)
        if action_match:
            result["action"] = action_match.group(1).strip()

            input_match = re.search(
                r"Action Input:\s*(.+?)(?=Observation:|Thought:|Final Answer:|$)",
                content,
                re.DOTALL | re.IGNORECASE,
            )
            if input_match:
                input_str = input_match.group(1).strip()
                logger.debug(f"[{self.name}] 原始 Action Input: {input_str[:200]}")
                try:
                    if input_str.startswith("```"):
                        input_str = re.sub(r"```\w*\n?", "", input_str).strip()
                    result["action_input"] = json.loads(input_str)
                    logger.debug(
                        f"[{self.name}] 解析后 Action Input: {result['action_input']}"
                    )
                except json.JSONDecodeError as e:
                    logger.warning(f"[{self.name}] Action Input JSON 解析失败: {e}")
                    logger.warning(f"[{self.name}] 原始字符串: {input_str[:200]}")
                    # 尝试修复常见问题：单引号转双引号
                    try:
                        fixed_str = input_str.replace("'", '"')
                        result["action_input"] = json.loads(fixed_str)
                        logger.info(f"[{self.name}] 使用单引号修复后解析成功")
                    except json.JSONDecodeError:
                        # 最后的回退：作为纯文本参数
                        result["action_input"] = {"input": input_str}
                        logger.warning(
                            f"[{self.name}] 回退为纯文本参数: input={input_str[:100]}"
                        )

            # 如果有 Action，不解析 Final Answer（它们不应该同时出现）
            return result

        # 只有在没有 Action 的情况下，才解析 Final Answer
        final_match = re.search(
            r"Final Answer:\s*(.+?)$", content, re.DOTALL | re.IGNORECASE
        )
        if final_match:
            final_answer = final_match.group(1).strip()
            # 验证 Final Answer 不是空的或者不是示例中的占位符
            if (
                final_answer
                and not final_answer.startswith("[")
                and len(final_answer) > 5
            ):
                result["final_answer"] = final_answer
            else:
                # Final Answer 内容无效，可能是格式示例中的内容
                logger.warning(
                    f"[{self.name}] Final Answer 内容无效或为占位符: {final_answer[:50]}"
                )

        return result

    def _finalize_output(self, final_answer: str) -> str:
        """生成最终输出"""
        schema = self.get_response_schema()

        if not schema:
            return final_answer

        schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
        final_prompt = self.get_finalize_prompt(schema_str)

        final_messages = self._loop_messages + [
            {"role": "user", "content": final_prompt}
        ]
        response = self._llm.chat(
            final_messages, response_format="json_object", tools=None
        )
        raw = response.content or "{}"

        logger.info(f"[{self.name}] LLM 返回的原始 JSON: {raw[:500]}")

        try:
            data = json.loads(raw)
            logger.info(f"[{self.name}] 解析后的数据: {data}")
            formatted = self.format_final_output(data)
            logger.info(f"[{self.name}] 格式化后的输出: {formatted[:500]}")
            return formatted
        except json.JSONDecodeError as e:
            logger.warning(f"[{self.name}] Failed to parse JSON: {e}")
            logger.warning(f"[{self.name}] Raw content: {raw}")
            return raw
