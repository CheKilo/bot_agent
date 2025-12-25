# -*- coding: utf-8 -*-
"""
记忆搜索工具

统一搜索工具：一次调用同时检索中期和长期记忆
"""

import logging
from typing import TYPE_CHECKING

from agent.tools import Tool, ToolResult

if TYPE_CHECKING:
    from agent.agents.memory.manager import MemoryManager

logger = logging.getLogger(__name__)


class SearchMemory(Tool):
    """
    统一记忆搜索工具

    一次调用同时检索：
    - 中期记忆：历史对话摘要（MySQL + BM25）
    - 长期记忆：用户偏好/事实/事件（Milvus 向量）

    工具会自动进行 query 改写和多关键词扩展，提高召回率。
    """

    name = "search_memory"
    description = """搜索所有记忆（中期 + 长期）。

- 中期记忆：历史对话摘要，适合查找"之前聊过什么"
- 长期记忆：用户偏好、事实、事件，适合查找"用户喜欢什么"

工具会自动优化查询并分别检索两种记忆，直接传入相关问题即可。"""

    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索内容，直接传递相关问题片段（如'昨天聊了什么'、'用户喜欢什么'）",
            },
            "time_range_days": {
                "type": "integer",
                "description": "中期记忆的时间范围（天），默认90天",
                "default": 90,
            },
            "limit": {
                "type": "integer",
                "description": "每种记忆返回的结果数量，默认5条",
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
        query: str = None,
        time_range_days: int = 90,
        limit: int = 5,
        input: str = None,  # 兼容参数，当 Action Input 解析失败时可能传入
        **kwargs,  # 忽略其他未知参数
    ) -> ToolResult:
        # 详细日志：记录所有接收到的参数
        logger.info(f"[{self.name}] execute 被调用")
        logger.info(f"[{self.name}] query={query!r}, input={input!r}")
        logger.info(f"[{self.name}] time_range_days={time_range_days}, limit={limit}")
        if kwargs:
            logger.info(f"[{self.name}] 额外参数 kwargs={kwargs}")

        # 兼容处理：如果没有 query 但有 input，使用 input 作为 query
        if not query and input:
            query = input
            logger.info(f"[{self.name}] 使用 input 参数作为 query: {query[:50]}")

        # 参数校验
        if not query or not query.strip():
            logger.warning(f"[{self.name}] query 为空，返回错误")
            return ToolResult.fail("搜索内容不能为空")

        try:
            # 调用统一检索接口
            results = self._manager.search_all(
                query=query,
                time_range_days=time_range_days,
                limit=limit,
            )

            mid_term = results.get("mid_term", [])
            long_term = results.get("long_term", [])

            # 格式化中期记忆结果
            formatted_mid = []
            for r in mid_term:
                item = {
                    "content": r.content,
                    "score": round(r.score, 3),
                }
                if r.keywords:
                    item["keywords"] = r.keywords
                formatted_mid.append(item)

            # 检查是否有结果
            has_mid = len(formatted_mid) > 0
            has_long = len(long_term) > 0

            if not has_mid and not has_long:
                return ToolResult.ok(
                    {
                        "message": "未找到相关记忆",
                        "mid_term": [],
                        "long_term": [],
                    }
                )

            # 构建返回消息
            parts = []
            if has_mid:
                parts.append(f"中期记忆 {len(formatted_mid)} 条")
            if has_long:
                parts.append(f"长期记忆 {len(long_term)} 条")
            message = f"找到 {' + '.join(parts)}"

            return ToolResult.ok(
                {
                    "message": message,
                    "mid_term": formatted_mid,
                    "long_term": long_term,
                }
            )

        except Exception as e:
            return ToolResult.fail(f"搜索失败: {e}")


# 保留旧类名作为别名，兼容已有代码
SearchMidTermMemory = SearchMemory
SearchLongTermMemory = SearchMemory
