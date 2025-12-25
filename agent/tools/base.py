# -*- coding: utf-8 -*-
"""
工具基类模块

为 Agent 提供工具定义和执行的基础设施。

使用示例：
    ```python
    from agent.tools import Tool, ToolResult, ToolKit

    class GetWeatherTool(Tool):
        name = "get_weather"
        description = "获取天气信息"
        parameters = {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }

        def execute(self, city: str) -> ToolResult:
            return ToolResult.ok({"city": city, "temp": 25})

    # 使用
    toolkit = ToolKit([GetWeatherTool()])
    schemas = toolkit.get_schemas()  # 传给 LLM
    result = toolkit.execute("get_weather", city="北京")
    ```
"""

import json
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# ============================================================================
# 工具执行结果
# ============================================================================


@dataclass
class ToolResult:
    """工具执行结果"""

    success: bool
    data: Any = None
    error: Optional[str] = None

    @classmethod
    def ok(cls, data: Any = None) -> "ToolResult":
        return cls(success=True, data=data)

    @classmethod
    def fail(cls, error: str) -> "ToolResult":
        return cls(success=False, error=error)

    def __str__(self) -> str:
        if not self.success:
            return f"Error: {self.error}"
        if self.data is None:
            return "Success"
        if isinstance(self.data, str):
            return self.data
        try:
            return json.dumps(self.data, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            return str(self.data)


# ============================================================================
# 工具基类
# ============================================================================


class Tool(ABC):
    """工具基类，子类需定义 name/description/parameters 并实现 execute"""

    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = {}

    def __init__(self):
        assert self.name, f"{self.__class__.__name__} must define 'name'"
        assert self.description, f"{self.__class__.__name__} must define 'description'"

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """执行工具"""
        pass

    def safe_execute(self, **kwargs) -> ToolResult:
        """安全执行（带异常捕获）"""
        try:
            # 调试日志：打印接收到的参数
            logger.info(f"[{self.name}] 接收参数: {list(kwargs.keys())}")
            logger.debug(f"[{self.name}] 参数详情: {kwargs}")
            return self.execute(**kwargs)
        except TypeError as e:
            # 参数不匹配错误，给出更详细的提示
            logger.error(
                f"Tool {self.name} 参数错误: {e}, 接收到的参数: {list(kwargs.keys())}"
            )
            return ToolResult.fail(f"参数错误: {e}. 接收到: {list(kwargs.keys())}")
        except Exception as e:
            logger.exception(f"Tool {self.name} execution failed")
            return ToolResult.fail(f"Execution error: {e}")

    def to_schema(self) -> Dict[str, Any]:
        """转换为 OpenAI Function Calling 格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def __repr__(self) -> str:
        return f"Tool({self.name})"


# ============================================================================
# 工具集管理
# ============================================================================


class ToolKit:
    """工具集管理器"""

    def __init__(self, tools: Optional[List[Tool]] = None):
        self._tools: Dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> "ToolKit":
        """注册工具"""
        self._tools[tool.name] = tool
        return self

    def get(self, name: str) -> Optional[Tool]:
        """获取工具"""
        return self._tools.get(name)

    def get_schemas(self) -> List[Dict[str, Any]]:
        """获取所有工具 schema"""
        return [t.to_schema() for t in self._tools.values()]

    def execute(
        self,
        tool_calls: List[Tuple[str, str, Dict[str, Any]]],
        max_workers: int = 5,
    ) -> List[Tuple[str, str, ToolResult]]:
        """
        执行工具调用（自动并行）

        Args:
            tool_calls: 工具调用列表，每项为 (call_id, name, args)
            max_workers: 最大并行数

        Returns:
            结果列表，每项为 (call_id, name, result)，顺序与输入一致
        """
        if not tool_calls:
            return []

        def _execute_one(name: str, args: Dict) -> ToolResult:
            tool = self._tools.get(name)
            if not tool:
                return ToolResult.fail(f"Unknown tool: {name}")
            return tool.safe_execute(**args)

        # 统一使用线程池执行（单任务开销可忽略）
        results: Dict[str, Tuple[str, ToolResult]] = {}

        with ThreadPoolExecutor(
            max_workers=min(max_workers, len(tool_calls))
        ) as executor:
            future_to_call = {
                executor.submit(_execute_one, name, args): (call_id, name)
                for call_id, name, args in tool_calls
            }

            for future in as_completed(future_to_call):
                call_id, name = future_to_call[future]
                try:
                    result = future.result()
                except Exception as e:
                    logger.exception(f"Tool {name} parallel execution failed")
                    result = ToolResult.fail(f"Parallel execution error: {e}")
                results[call_id] = (name, result)

        # 按原始顺序返回结果
        return [
            (call_id, results[call_id][0], results[call_id][1])
            for call_id, _, _ in tool_calls
        ]

    @property
    def names(self) -> List[str]:
        return list(self._tools.keys())

    def get_names_str(self) -> str:
        """获取所有工具名称的逗号分隔字符串"""
        return ", ".join(self._tools.keys())

    def get_descriptions(self, format_style: str = "markdown") -> str:
        """
        生成工具描述文本（用于 ReAct prompt）

        Args:
            format_style: 格式风格，支持 "markdown" 或 "plain"

        Returns:
            格式化的工具描述字符串
        """
        if not self._tools:
            return ""

        lines = []
        for tool in self._tools.values():
            if format_style == "markdown":
                params_str = json.dumps(tool.parameters, ensure_ascii=False, indent=2)
                lines.append(
                    f"### {tool.name}\n{tool.description}\n参数: {params_str}\n"
                )
            else:
                # plain 格式，更紧凑
                params_str = json.dumps(tool.parameters, ensure_ascii=False)
                lines.append(f"{tool.name}: {tool.description} 参数: {params_str}")

        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __repr__(self) -> str:
        return f"ToolKit({self.names})"


# ============================================================================
# 函数装饰器
# ============================================================================


def function_tool(
    name: str,
    description: str,
    parameters: Dict[str, Any],
) -> Callable[[Callable], Tool]:
    """将普通函数转换为 Tool"""

    def decorator(func: Callable) -> Tool:
        def _execute(self, **kwargs) -> ToolResult:
            result = func(**kwargs)
            return result if isinstance(result, ToolResult) else ToolResult.ok(result)

        cls = type(
            f"FunctionTool_{name}",
            (Tool,),
            {
                "name": name,
                "description": description,
                "parameters": parameters,
                "execute": _execute,
            },
        )
        return cls()

    return decorator


# ============================================================================
# 工具调用数据类（整合辅助函数）
# ============================================================================


@dataclass
class ToolCall:
    """工具调用封装类，整合解析和格式化功能"""

    id: str = ""
    name: str = ""
    args: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, tool_call: Dict[str, Any]) -> "ToolCall":
        """从字典解析工具调用"""
        call_id = tool_call.get("id", "")
        name = tool_call.get("function", {}).get("name", "")
        try:
            args_str = tool_call.get("function", {}).get("arguments", "{}")
            args = json.loads(args_str) if args_str else {}
        except json.JSONDecodeError:
            args = {}
        return cls(id=call_id, name=name, args=args)

    def format_result_for_llm(
        self, result: Union[ToolResult, str, Any]
    ) -> Dict[str, Any]:
        """格式化工具结果为 LLM 消息"""
        if isinstance(result, ToolResult):
            content = str(result)
        elif isinstance(result, str):
            content = result
        else:
            content = (
                json.dumps(result, ensure_ascii=False)
                if isinstance(result, dict)
                else str(result)
            )
        return {"role": "tool", "content": content, "tool_call_id": self.id}
