# -*- coding: utf-8 -*-
"""
记忆存储工具

提供长期记忆的存储能力。
"""

from typing import TYPE_CHECKING

from agent.tools import Tool, ToolResult

if TYPE_CHECKING:
    from agent.agents.memory.manager import MemoryManager


class StoreLongTermMemory(Tool):
    """存储长期记忆"""

    name = "store_long_term_memory"
    description = "将重要信息存储为长期记忆。用于保存用户偏好、重要事实、关键事件、人物关系等需要长期记住的信息。"
    parameters = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "要记住的内容，应该是结构化的信息（如'小明是用户的同事，在技术部工作'）",
            },
            "memory_type": {
                "type": "string",
                "enum": ["preference", "fact", "event"],
                "description": "记忆类型：preference=用户偏好, fact=事实信息, event=事件记录",
            },
            "importance": {
                "type": "integer",
                "description": "重要性级别（1-10），默认5。10=非常重要，1=一般信息",
                "default": 5,
            },
        },
        "required": ["content", "memory_type"],
    }

    def __init__(self, manager: "MemoryManager"):
        super().__init__()
        self._manager = manager

    def execute(
        self, content: str, memory_type: str, importance: int = 5
    ) -> ToolResult:
        if not content or not content.strip():
            return ToolResult.fail("存储内容不能为空")

        try:
            memory_id = self._manager.store_long_term(content, memory_type, importance)
            if memory_id:
                return ToolResult.ok({"message": "记忆已保存", "id": memory_id})
            return ToolResult.fail("存储记忆失败")
        except Exception as e:
            return ToolResult.fail(f"存储失败: {e}")
