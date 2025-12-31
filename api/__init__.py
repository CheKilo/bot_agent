# -*- coding: utf-8 -*-
"""
Bot Agent API 模块

提供基于 FastAPI 的对话服务接口。
"""

from api.config import settings, APIConfig
from api.models import (
    ChatRequest,
    ChatResponse,
    ClearHistoryRequest,
    SetPersonaRequest,
    HistoryResponse,
    PersonaListResponse,
    SessionListResponse,
    GenericResponse,
    HealthResponse,
)
from api.service import ChatService, ChatPipeline, chat_service, PERSONAS
from api.routes import router
from api.main import app

__all__ = [
    # 配置
    "settings",
    "APIConfig",
    # 数据模型
    "ChatRequest",
    "ChatResponse",
    "ClearHistoryRequest",
    "SetPersonaRequest",
    "HistoryResponse",
    "PersonaListResponse",
    "SessionListResponse",
    "GenericResponse",
    "HealthResponse",
    # 服务
    "ChatService",
    "ChatPipeline",
    "chat_service",
    "PERSONAS",
    # FastAPI
    "router",
    "app",
]
