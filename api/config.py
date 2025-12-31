# -*- coding: utf-8 -*-
"""
API 配置模块

集中管理 API 服务的所有配置项。
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ServerConfig:
    """服务器配置"""

    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    workers: int = 1


@dataclass
class GRPCConfig:
    """gRPC 服务配置"""

    # LLM 和 Storage 共用同一个 gRPC 服务
    host: str = "localhost"
    port: int = 50051

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"


@dataclass
class LLMConfig:
    """LLM 配置"""

    model: str = "gpt-5"
    embedding_model: str = "text-embedding-ada-002"
    timeout: float = 60.0


@dataclass
class APIConfig:
    """API 全局配置"""

    # 服务器配置
    server: ServerConfig = field(default_factory=ServerConfig)

    # gRPC 配置
    grpc: GRPCConfig = field(default_factory=GRPCConfig)

    # LLM 配置
    llm: LLMConfig = field(default_factory=LLMConfig)

    # 默认人设
    default_persona: str = "girl"

    # 是否启用记忆功能
    enable_memory: bool = True

    @classmethod
    def from_env(cls) -> "APIConfig":
        """从环境变量加载配置"""
        config = cls()

        # 服务器配置
        config.server.host = os.getenv("API_HOST", config.server.host)
        config.server.port = int(os.getenv("API_PORT", config.server.port))
        config.server.debug = os.getenv("API_DEBUG", "false").lower() == "true"

        # gRPC 配置
        config.grpc.host = os.getenv("GRPC_HOST", config.grpc.host)
        config.grpc.port = int(os.getenv("GRPC_PORT", config.grpc.port))

        # LLM 配置
        config.llm.model = os.getenv("LLM_MODEL", config.llm.model)
        config.llm.embedding_model = os.getenv(
            "EMBEDDING_MODEL", config.llm.embedding_model
        )

        # 功能配置
        config.default_persona = os.getenv("DEFAULT_PERSONA", config.default_persona)
        config.enable_memory = os.getenv("ENABLE_MEMORY", "true").lower() == "true"

        return config


# 全局配置实例
settings = APIConfig.from_env()
