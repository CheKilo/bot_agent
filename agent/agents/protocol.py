# -*- coding: utf-8 -*-
"""
Agent Protocol - Agent 间通信协议

定义统一的 Agent 调用接口，实现多 Agent 松耦合通信。

设计说明：
- AgentMessage: Agent 间传递的消息（输入）
- AgentResponse: Agent 的响应（输出）
- AgentProtocol: 统一调用协议接口
- AgentRegistry: Agent 注册中心
- CallAgent: 通用 Agent 调用工具（由注册中心提供）

设计原则：
- 最小化：只定义 content（必需）+ metadata（可选扩展）
- 通用性：不预设任何业务字段，各 Agent 自行约定 metadata 结构
- 松耦合：调用方和被调用方通过 metadata 传递定制数据
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent.tools import Tool, ToolResult

logger = logging.getLogger(__name__)


@dataclass
class AgentMessage:
    """
    Agent 间通信的输入消息

    Attributes:
        content: 主要输入内容（必需）
        metadata: 扩展数据（可选），各 Agent 自行约定结构

    使用示例：
        # Memory Agent 调用
        AgentMessage(
            content="用户说了什么",
            metadata={"conversation_history": [...]}
        )

        # Character Agent 调用
        AgentMessage(
            content="用户说了什么",
            metadata={
                "conversation_history": [...],
                "memory_context": "相关记忆..."
            }
        )
    """

    content: str  # 主要输入内容
    metadata: Dict[str, Any] = field(default_factory=dict)  # 扩展数据

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def get(self, key: str, default: Any = None) -> Any:
        """便捷方法：从 metadata 获取值"""
        return self.metadata.get(key, default)


@dataclass
class AgentResponse:
    """
    Agent 响应

    Attributes:
        content: 主要输出内容
        metadata: 扩展数据（可选），各 Agent 自行约定结构
        success: 是否成功
        error: 错误信息（失败时）

    使用示例：
        # Memory Agent 返回
        AgentResponse(
            content="检索到的记忆...",
            metadata={"memory_context": "格式化的记忆"}
        )

        # Character Agent 返回
        AgentResponse(
            content="角色回复内容",
            metadata={"emotion_state": {"mood": 0.8}}
        )
    """

    content: str  # 主要输出内容
    metadata: Dict[str, Any] = field(default_factory=dict)  # 扩展数据
    success: bool = True
    error: Optional[str] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def get(self, key: str, default: Any = None) -> Any:
        """便捷方法：从 metadata 获取值"""
        return self.metadata.get(key, default)


class AgentProtocol(ABC):
    """
    Agent 统一调用协议

    所有需要被其他 Agent 调用的 Agent 都应实现此协议。
    通过 invoke() 方法提供统一的调用入口。
    """

    @property
    @abstractmethod
    def agent_name(self) -> str:
        """Agent 唯一标识名称"""
        pass

    @property
    def agent_description(self) -> str:
        """Agent 功能描述（供调用方参考）"""
        return f"{self.agent_name} agent"

    @abstractmethod
    def invoke(self, message: AgentMessage) -> AgentResponse:
        """
        统一调用入口

        Args:
            message: 输入消息
                - content: 主要输入内容
                - metadata: 扩展数据（各 Agent 自行约定）

        Returns:
            AgentResponse: 响应结果
        """
        pass


# ============================================================================
# CallAgent 工具（通用 Agent 调用工具）
# ============================================================================


class CallAgent(Tool):
    """
    通用 Agent 调用工具

    通过 AgentRegistry 按名称调用指定的 Agent，
    不直接持有 Agent 实例，实现松耦合。

    使用方式：
        registry = AgentRegistry()
        registry.register(memory_agent)
        registry.register(character_agent)

        # 获取工具
        call_tool = registry.get_call_tool()

        # 在 Agent 中使用
        def get_tools(self):
            return [self._registry.get_call_tool()]
    """

    name = "call_agent"
    description = """调用指定的 Agent 执行任务。

参数：
- agent_name: Agent 名称（必需）
- input: 输入内容（必需）
- metadata: 扩展数据（可选，如 memory_context, conversation_history 等）

返回：Agent 的执行结果"""

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
            "metadata": {
                "type": "object",
                "description": "扩展数据（可选，各 Agent 自行约定结构）",
            },
        },
        "required": ["agent_name", "input"],
    }

    def __init__(self, registry: "AgentRegistry"):
        """
        初始化

        Args:
            registry: Agent 注册中心
        """
        super().__init__()
        self._registry = registry

    def execute(
        self,
        agent_name: str,
        input: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """执行 Agent 调用"""
        logger.info(f"[CallAgent] 调用 Agent: {agent_name}")
        logger.info(f"[CallAgent] input: {input[:100] if input else 'None'}")
        logger.info(f"[CallAgent] metadata 类型: {type(metadata)}")
        logger.info(f"[CallAgent] metadata 内容: {metadata}")

        # 获取 Agent
        agent = self._registry.get(agent_name)
        if not agent:
            available = self._registry.list_agents()
            return ToolResult.fail(
                f"Unknown agent: {agent_name}. Available: {available}"
            )

        try:
            # 构造消息
            message = AgentMessage(
                content=input,
                metadata=metadata or {},
            )
            logger.info(f"[CallAgent] AgentMessage 构造成功")

            # 调用 Agent
            logger.info(f"[CallAgent] 开始调用 {agent_name}.invoke()")
            response = agent.invoke(message)
            logger.info(
                f"[CallAgent] {agent_name}.invoke() 返回，success={response.success}"
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


# ============================================================================
# Agent 注册中心
# ============================================================================


class AgentRegistry:
    """
    Agent 注册中心

    管理所有可调用的 Agent，提供：
    1. Agent 注册/获取
    2. CallAgent 工具（用于调用已注册的 Agent）

    使用示例：
        # 链路层初始化
        registry = AgentRegistry()
        registry.register(memory_agent)
        registry.register(character_agent)

        # System Agent 使用
        class SystemAgent(Agent):
            def __init__(self, registry: AgentRegistry):
                self._registry = registry

            def get_tools(self):
                return [self._registry.get_call_tool()]
    """

    def __init__(self):
        self._agents: Dict[str, AgentProtocol] = {}
        self._call_tool: Optional[CallAgent] = None

    def register(self, agent: AgentProtocol) -> "AgentRegistry":
        """注册 Agent"""
        self._agents[agent.agent_name] = agent
        return self

    def unregister(self, name: str) -> "AgentRegistry":
        """取消注册"""
        self._agents.pop(name, None)
        return self

    def get(self, name: str) -> Optional[AgentProtocol]:
        """按名称获取 Agent"""
        return self._agents.get(name)

    def list_agents(self) -> List[str]:
        """列出所有已注册的 Agent 名称"""
        return list(self._agents.keys())

    def get_descriptions(self) -> Dict[str, str]:
        """获取所有 Agent 的描述"""
        return {name: agent.agent_description for name, agent in self._agents.items()}

    def get_call_tool(self) -> CallAgent:
        """
        获取 CallAgent 工具

        返回一个绑定了当前注册中心的 CallAgent 工具实例。
        Agent 可以通过这个工具调用其他已注册的 Agent。
        """
        if self._call_tool is None:
            self._call_tool = CallAgent(self)
        return self._call_tool

    def __contains__(self, name: str) -> bool:
        return name in self._agents

    def __len__(self) -> int:
        return len(self._agents)
