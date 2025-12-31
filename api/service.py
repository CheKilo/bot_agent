# -*- coding: utf-8 -*-
"""
API 服务层

管理 ChatPipeline 实例，提供核心业务逻辑。
"""

import logging
from datetime import datetime
from typing import Callable, Dict, List, Optional

from agent.agents import AgentRegistry
from agent.agents.system import SystemAgent
from agent.agents.memory import MemoryAgent
from agent.agents.character import (
    CharacterAgent,
    Persona,
    DEFAULT_PERSONA,
    EXAMPLE_PERSONA_GIRL,
    EXAMPLE_PERSONA_MATURE,
)
from agent.client import StorageClient
from agent.core import LLM

from api.config import settings, APIConfig

logger = logging.getLogger(__name__)


# 可用人设配置
PERSONAS: Dict[str, Persona] = {
    "default": DEFAULT_PERSONA,
    "girl": EXAMPLE_PERSONA_GIRL,
    "mature": EXAMPLE_PERSONA_MATURE,
}


class ChatPipeline:
    """
    对话链路

    职责：
    1. 初始化并注册所有 Agent 到 AgentRegistry
    2. 通过 System Agent 的 run() 方法执行对话
    3. 管理资源生命周期
    """

    def __init__(
        self,
        bot_id: str,
        user_id: str,
        storage_client: StorageClient,
        embed_func: Callable[[str], List[float]],
        persona: Optional[Persona] = None,
        llm_address: str = "localhost:50051",
        model: str = "gpt-5",
        enable_memory: bool = True,
    ):
        self._bot_id = bot_id
        self._user_id = user_id
        self._enable_memory = enable_memory
        self._storage_client = storage_client
        self._persona = persona or EXAMPLE_PERSONA_GIRL
        self._created_at = datetime.now().isoformat()

        # 创建 Agent 注册中心
        self._registry = AgentRegistry()

        # 初始化 Memory Agent（如果启用）
        self._memory_agent: Optional[MemoryAgent] = None
        if enable_memory:
            self._memory_agent = MemoryAgent(
                bot_id=bot_id,
                user_id=user_id,
                storage_client=storage_client,
                embed_func=embed_func,
                llm_address=llm_address,
                model=model,
            )
            self._registry.register(self._memory_agent)

        # 初始化 Character Agent
        self._character_agent = CharacterAgent(
            bot_id=bot_id,
            persona=self._persona,
        )
        self._registry.register(self._character_agent)

        # 初始化 System Agent
        self._system_agent = SystemAgent(
            bot_id=bot_id,
            user_id=user_id,
            registry=self._registry,
            storage_client=storage_client,
        )

    @property
    def bot_id(self) -> str:
        return self._bot_id

    @property
    def user_id(self) -> str:
        return self._user_id

    @property
    def persona(self) -> Persona:
        return self._persona

    @property
    def enable_memory(self) -> bool:
        return self._enable_memory

    @property
    def created_at(self) -> str:
        return self._created_at

    @property
    def conversation_history(self) -> List[Dict]:
        """获取对话历史"""
        return self._system_agent.messages

    def chat(self, user_input: str) -> Dict:
        """
        执行一次完整对话

        Returns:
            Dict: 包含 answer, iterations, success, error
        """
        try:
            result = self._system_agent.run(user_input)

            return {
                "answer": result.answer,
                "iterations": result.iterations,
                "success": result.success,
                "error": result.error,
            }

        except Exception as e:
            logger.error(f"[Pipeline] 对话执行错误: {e}", exc_info=True)
            return {
                "answer": "抱歉，我遇到了一些问题，请稍后再试。",
                "iterations": 0,
                "success": False,
                "error": str(e),
            }

    def clear_history(self):
        """清空对话历史"""
        self._system_agent.clear_history()
        logger.info(f"[Pipeline] 对话历史已清空 (user={self._user_id})")

    def set_persona(self, persona: Persona):
        """更换人设"""
        self._persona = persona
        self._character_agent.set_persona(persona)

    def close(self):
        """关闭资源"""
        self._system_agent.close()
        if self._memory_agent:
            self._memory_agent.close()
        if self._character_agent:
            self._character_agent.close()


class ChatService:
    """
    对话服务

    管理多个 ChatPipeline 实例，提供会话管理功能。
    使用 (bot_id, user_id) 作为会话标识。
    """

    def __init__(self, config: Optional[APIConfig] = None):
        self._config = config or settings
        self._pipelines: Dict[str, ChatPipeline] = {}
        self._storage_client: Optional[StorageClient] = None
        self._llm: Optional[LLM] = None
        self._initialized = False

    def initialize(self):
        """初始化服务"""
        if self._initialized:
            return

        logger.info("[ChatService] 正在初始化...")

        # 初始化 Storage 客户端
        self._storage_client = StorageClient(self._config.grpc.address)
        self._storage_client.connect()
        logger.info(f"[ChatService] Storage 客户端已连接 ({self._config.grpc.address})")

        # 初始化 LLM（用于 embed 函数）
        self._llm = LLM(
            address=self._config.grpc.address,
            model=self._config.llm.model,
        )
        logger.info(f"[ChatService] LLM 客户端已初始化 ({self._config.llm.model})")

        self._initialized = True
        logger.info("[ChatService] 初始化完成")

    def shutdown(self):
        """关闭服务"""
        logger.info("[ChatService] 正在关闭...")

        # 关闭所有 Pipeline
        for key, pipeline in self._pipelines.items():
            try:
                pipeline.close()
                logger.info(f"[ChatService] Pipeline 已关闭: {key}")
            except Exception as e:
                logger.warning(f"[ChatService] 关闭 Pipeline 时出错: {e}")
        self._pipelines.clear()

        # 关闭 LLM
        if self._llm:
            try:
                self._llm.close()
                logger.info("[ChatService] LLM 客户端已关闭")
            except Exception as e:
                logger.warning(f"[ChatService] 关闭 LLM 时出错: {e}")

        # 关闭 Storage
        if self._storage_client:
            try:
                self._storage_client.close()
                logger.info("[ChatService] Storage 客户端已关闭")
            except Exception as e:
                logger.warning(f"[ChatService] 关闭 Storage 时出错: {e}")

        self._initialized = False
        logger.info("[ChatService] 关闭完成")

    def _get_session_key(self, bot_id: str, user_id: str) -> str:
        """生成会话 key"""
        return f"{bot_id}:{user_id}"

    def _embed_func(self, text: str) -> List[float]:
        """Embedding 函数"""
        result = self._llm.embed(text, model=self._config.llm.embedding_model)
        return result[0] if result else []

    def _get_or_create_pipeline(
        self,
        bot_id: str,
        user_id: str,
        persona_name: Optional[str] = None,
        enable_memory: Optional[bool] = None,
    ) -> ChatPipeline:
        """获取或创建 Pipeline"""
        key = self._get_session_key(bot_id, user_id)

        if key not in self._pipelines:
            # 确定人设
            persona_name = persona_name or self._config.default_persona
            persona = PERSONAS.get(persona_name, EXAMPLE_PERSONA_GIRL)

            # 确定是否启用记忆
            memory_enabled = (
                enable_memory
                if enable_memory is not None
                else self._config.enable_memory
            )

            # 创建新的 Pipeline
            pipeline = ChatPipeline(
                bot_id=bot_id,
                user_id=user_id,
                storage_client=self._storage_client,
                embed_func=self._embed_func,
                persona=persona,
                llm_address=self._config.grpc.address,
                model=self._config.llm.model,
                enable_memory=memory_enabled,
            )
            self._pipelines[key] = pipeline
            logger.info(
                f"[ChatService] 创建新会话: {key} (persona={persona.name}, memory={memory_enabled})"
            )

        return self._pipelines[key]

    def chat(
        self,
        user_id: str,
        bot_id: str,
        message: str,
        persona: Optional[str] = None,
        enable_memory: Optional[bool] = None,
    ) -> Dict:
        """
        执行对话

        Args:
            user_id: 用户ID
            bot_id: Bot ID
            message: 用户消息
            persona: 人设名称（可选，仅在创建新会话时生效）
            enable_memory: 是否启用记忆（可选，仅在创建新会话时生效）

        Returns:
            Dict: 包含 answer, iterations, success, error
        """
        if not self._initialized:
            self.initialize()

        pipeline = self._get_or_create_pipeline(
            bot_id=bot_id,
            user_id=user_id,
            persona_name=persona,
            enable_memory=enable_memory,
        )

        return pipeline.chat(message)

    def get_history(self, bot_id: str, user_id: str) -> List[Dict]:
        """获取对话历史"""
        key = self._get_session_key(bot_id, user_id)
        pipeline = self._pipelines.get(key)
        if pipeline:
            return pipeline.conversation_history
        return []

    def clear_history(self, bot_id: str, user_id: str) -> bool:
        """清空对话历史"""
        key = self._get_session_key(bot_id, user_id)
        pipeline = self._pipelines.get(key)
        if pipeline:
            pipeline.clear_history()
            return True
        return False

    def set_persona(self, bot_id: str, user_id: str, persona_name: str) -> bool:
        """设置人设"""
        if persona_name not in PERSONAS:
            return False

        key = self._get_session_key(bot_id, user_id)
        pipeline = self._pipelines.get(key)
        if pipeline:
            pipeline.set_persona(PERSONAS[persona_name])
            return True
        return False

    def get_session_info(self, bot_id: str, user_id: str) -> Optional[Dict]:
        """获取会话信息"""
        key = self._get_session_key(bot_id, user_id)
        pipeline = self._pipelines.get(key)
        if pipeline:
            return {
                "user_id": pipeline.user_id,
                "bot_id": pipeline.bot_id,
                "persona": pipeline.persona.name,
                "enable_memory": pipeline.enable_memory,
                "message_count": len(pipeline.conversation_history),
                "created_at": pipeline.created_at,
            }
        return None

    def list_sessions(self) -> List[Dict]:
        """列出所有活跃会话"""
        sessions = []
        for key, pipeline in self._pipelines.items():
            sessions.append(
                {
                    "user_id": pipeline.user_id,
                    "bot_id": pipeline.bot_id,
                    "persona": pipeline.persona.name,
                    "enable_memory": pipeline.enable_memory,
                    "message_count": len(pipeline.conversation_history),
                    "created_at": pipeline.created_at,
                }
            )
        return sessions

    def delete_session(self, bot_id: str, user_id: str) -> bool:
        """删除会话"""
        key = self._get_session_key(bot_id, user_id)
        if key in self._pipelines:
            self._pipelines[key].close()
            del self._pipelines[key]
            logger.info(f"[ChatService] 会话已删除: {key}")
            return True
        return False

    @staticmethod
    def get_personas() -> Dict[str, Dict]:
        """获取可用人设列表"""
        return {
            name: {
                "name": name,
                "display_name": persona.name,
                "description": getattr(persona, "description", None),
            }
            for name, persona in PERSONAS.items()
        }


# 全局服务实例
chat_service = ChatService()
