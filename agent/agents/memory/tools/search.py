# -*- coding: utf-8 -*-
"""
记忆搜索工具

Tool 层只负责参数校验和调用 Manager，不参与任何检索逻辑
"""

from typing import TYPE_CHECKING, List, Dict

from agent.tools import Tool, ToolResult

if TYPE_CHECKING:
    from agent.agents.memory.manager import MemoryManager


class SearchMidTermMemory(Tool):
    """搜索中期记忆"""

    name = "search_mid_term_memory"
    description = """搜索历史对话摘要（中期记忆）。

用于查找超出当前上下文范围的历史对话。
返回匹配的对话摘要及关键词。

直接传递相关问题片段，工具会自动进行检索优化。"""

    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索内容，直接传递相关的问题片段即可（如'上周讨论的项目'）",
            },
            "time_range_days": {
                "type": "integer",
                "description": "时间范围（天），默认30天",
                "default": 30,
            },
            "limit": {
                "type": "integer",
                "description": "返回结果数量，默认5条",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    def __init__(self, manager: "MemoryManager"):
        super().__init__()
        self._manager = manager

    def execute(
        self, query: str, time_range_days: int = 30, limit: int = 5
    ) -> ToolResult:
        # 参数校验
        if not query or not query.strip():
            return ToolResult.fail("搜索内容不能为空，请提供搜索内容")

        try:
            results = self._manager.search_mid_term(
                query=query,
                time_range_days=time_range_days,
                limit=limit,
            )

            if not results:
                return ToolResult.ok({"message": "未找到相关的中期记忆", "results": []})

            # 格式化输出：统一的摘要格式
            formatted = []
            for r in results:
                item = {
                    "content": r.content,
                    "score": round(r.score, 3),
                }
                if r.keywords:
                    item["keywords"] = r.keywords

                formatted.append(item)

            return ToolResult.ok(
                {
                    "message": f"找到 {len(formatted)} 条相关记忆",
                    "results": formatted,
                }
            )

        except Exception as e:
            return ToolResult.fail(f"搜索失败: {e}")


class SearchLongTermMemory(Tool):
    """搜索长期记忆"""

    name = "search_long_term_memory"
    description = """搜索长期记忆（用户偏好、事实、事件、人物关系等）。

用于查找重要的持久化信息，支持按类型和重要性过滤。
直接传递相关问题片段，工具会自动进行向量检索。"""

    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索内容，直接传递相关的问题片段即可（如'小明是谁'、'用户喜欢什么'）",
            },
            "memory_type": {
                "type": "string",
                "enum": ["all", "preference", "fact", "event", "promoted"],
                "description": "记忆类型过滤：all=全部, preference=偏好, fact=事实, event=事件, promoted=提升的记忆",
                "default": "all",
            },
            "min_importance": {
                "type": "integer",
                "description": "最小重要性级别（1-10），用于过滤不重要的记忆",
                "default": 1,
            },
            "limit": {
                "type": "integer",
                "description": "返回结果数量，默认5条",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    def __init__(self, manager: "MemoryManager"):
        super().__init__()
        self._manager = manager

    def execute(
        self,
        query: str,
        memory_type: str = "all",
        min_importance: int = 1,
        limit: int = 5,
    ) -> ToolResult:
        # 参数校验
        if not query or not query.strip():
            return ToolResult.fail("搜索内容不能为空，请提供搜索内容")

        min_importance = max(1, min(10, min_importance))

        try:
            results = self._manager.search_long_term(
                query=query,
                memory_type=memory_type,
                limit=limit,
                min_importance=min_importance,
            )

            if not results:
                return ToolResult.ok({"message": "未找到相关的长期记忆", "results": []})

            return ToolResult.ok(
                {
                    "message": f"找到 {len(results)} 条相关记忆",
                    "results": results,
                }
            )

        except Exception as e:
            return ToolResult.fail(f"搜索失败: {e}")
