# -*- coding: utf-8 -*-
"""
gRPC 客户端模块

提供与 Go 后端服务通信的客户端实现。
"""

from .llm_client import (
    LLMClient,
    LLMClientError,
    LLMConnectionError,
    LLMRequestError,
    # 便捷函数
    get_default_client,
    chat,
    chat_stream,
    embed,
)

from .storage_client import (
    StorageClient,
    StorageRequestError,
)

__all__ = [
    # LLM 客户端类
    "LLMClient",
    # LLM 异常类
    "LLMClientError",
    "LLMConnectionError",
    "LLMRequestError",
    # LLM 便捷函数
    "get_default_client",
    "chat",
    "chat_stream",
    "embed",
    # Storage 客户端类
    "StorageClient",
    # Storage 异常类
    "StorageRequestError",
]
