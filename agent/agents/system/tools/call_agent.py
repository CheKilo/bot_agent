# -*- coding: utf-8 -*-
"""
System Agent 专用的 Agent 调用工具

相比 protocol.CallAgent 的增强：
1. 持有 System Agent 的 _messages 引用
2. 自动注入 conversation_history（排除当前用户输入）
3. 简化 Action Input，只需提供 agent_name 和 input
"""

import logging
from typing import Any, Dict, List

from agent.agents.protocol import CallAgent
from agent.tools import ToolResult

logger = logging.getLogger(__name__)


class CallAgentTool(CallAgent):
    """
    System Agent 专用的 Agent 调用包装工具

    继承自 protocol.CallAgent，增加自动注入 conversation_history 的功能。

    使用方式：
        # System Agent 初始化时
        class SystemAgent(Agent):
            def __init__(self, ...):
                self._call_tool = CallAgentTool(registry, self._messages)

            def get_tools(self):
                return [self._call_tool]
    """

    name = "call_agent"
    description = """调用指定的 Agent 执行任务。

参数：
- agent_name: Agent 名称（必需）
- input: 输入内容（必需）
- memory_context: 记忆上下文（可选），仅在调用 character_agent 时需要

返回：Agent 的执行结果

说明：conversation_history 会自动注入，无需手动传递。"""

    parameters = {
        "type": "object",
        "properties": {
            "agent_name": {
                "type": "string",
                "description": "要调用的 Agent 名称",
            },
            "input": {
                "type": "string",
                "description": "输入内容（用户输入或上游输出）",
            },
            "memory_context": {
                "type": "string",
                "description": "记忆上下文（可选），仅在调用 character_agent 时需要",
            },
        },
        "required": ["agent_name", "input"],
    }

    def __init__(self, registry: "AgentRegistry", messages_ref: List[Dict[str, Any]]):
        """
        初始化

        Args:
            registry: Agent 注册中心
            messages_ref: System Agent 的 _messages 引用（用于自动注入 conversation_history）
        """
        super().__init__(registry)
        self._messages_ref = messages_ref

    def execute(
        self,
        agent_name: str,
        input: str,
        **kwargs,
    ) -> ToolResult:
        """
        执行 Agent 调用（自动注入 conversation_history）

        Args:
            agent_name: Agent 名称
            input: 输入内容
            **kwargs: 额外的 metadata 字段（如 memory_context），会合并到 metadata 中

        Returns:
            ToolResult
        """
        logger.info(f"[CallAgentTool] 调用 Agent: {agent_name}")
        logger.info(f"[CallAgentTool] input: {input[:100] if input else 'None'}")

        # 获取 Agent
        agent = self._registry.get(agent_name)
        if not agent:
            available = self._registry.list_agents()
            return ToolResult.fail(
                f"Unknown agent: {agent_name}. Available: {available}"
            )

        try:
            # 自动注入 conversation_history
            # 注意：当前用户输入已在 SystemAgent._init_loop 中添加到 _messages
            # 所以这里直接传递完整的历史（包含当前用户输入）
            conversation_history = list(self._messages_ref)

            # 构造 metadata（合并自动注入的和手动传入的）
            metadata = kwargs.copy()
            metadata["conversation_history"] = conversation_history

            logger.info(
                f"[CallAgentTool] 自动注入 conversation_history，"
                f"长度={len(conversation_history)}"
            )
            logger.info(f"[CallAgentTool] metadata 内容: {metadata}")

            # 构造消息
            from agent.agents.protocol import AgentMessage

            message = AgentMessage(
                content=input,
                metadata=metadata,
            )
            logger.info(f"[CallAgentTool] AgentMessage 构造成功")

            # 调用 Agent
            logger.info(f"[CallAgentTool] 开始调用 {agent_name}.invoke()")
            response = agent.invoke(message)
            logger.info(
                f"[CallAgentTool] {agent_name}.invoke() 返回，success={response.success}"
            )

            if not response.success:
                return ToolResult.fail(response.error or "Agent execution failed")

            # 返回结果（包含 content 和 metadata）
            return ToolResult.ok(
                {
                    "content": response.content,
                    "metadata": response.metadata,
                }
            )

        except Exception as e:
            logger.error(f"Agent {agent_name} invocation failed: {e}", exc_info=True)
            return ToolResult.fail(str(e))
