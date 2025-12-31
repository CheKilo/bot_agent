# -*- coding: utf-8 -*-
"""
API 数据模型

定义请求和响应的 Pydantic 模型。
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ============================================================================
# 请求模型
# ============================================================================


class ChatRequest(BaseModel):
    """对话请求"""

    user_id: str = Field(..., description="用户ID", min_length=1)
    bot_id: str = Field(default="default_bot", description="Bot ID")
    message: str = Field(..., description="用户消息", min_length=1)
    persona: Optional[str] = Field(default=None, description="人设名称（可选）")
    enable_memory: Optional[bool] = Field(default=None, description="是否启用记忆")

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user123",
                "bot_id": "my_bot",
                "message": "你好呀！",
                "persona": "girl",
                "enable_memory": True,
            }
        }


class ClearHistoryRequest(BaseModel):
    """清空对话历史请求"""

    user_id: str = Field(..., description="用户ID")
    bot_id: str = Field(default="default_bot", description="Bot ID")


class SetPersonaRequest(BaseModel):
    """设置人设请求"""

    user_id: str = Field(..., description="用户ID")
    bot_id: str = Field(default="default_bot", description="Bot ID")
    persona: str = Field(..., description="人设名称")


# ============================================================================
# 响应模型
# ============================================================================


class ChatResponse(BaseModel):
    """对话响应"""

    success: bool = Field(..., description="是否成功")
    answer: str = Field(default="", description="回复内容")
    iterations: int = Field(default=0, description="迭代次数")
    error: Optional[str] = Field(default=None, description="错误信息")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "answer": "你好呀～今天过得怎么样？",
                "iterations": 3,
                "error": None,
                "timestamp": "2024-01-01T12:00:00",
            }
        }


class HistoryMessage(BaseModel):
    """历史消息"""

    role: str = Field(..., description="角色 (user/assistant)")
    content: str = Field(..., description="消息内容")
    timestamp: Optional[str] = Field(default=None, description="时间戳")


class HistoryResponse(BaseModel):
    """对话历史响应"""

    success: bool = Field(default=True)
    user_id: str
    bot_id: str
    messages: List[HistoryMessage] = Field(default_factory=list)
    total_messages: int = Field(default=0)
    user_turns: int = Field(default=0, description="用户轮次")


class PersonaInfo(BaseModel):
    """人设信息"""

    name: str = Field(..., description="人设标识")
    display_name: str = Field(..., description="人设显示名称")
    description: Optional[str] = Field(default=None, description="人设描述")


class PersonaListResponse(BaseModel):
    """人设列表响应"""

    success: bool = Field(default=True)
    personas: List[PersonaInfo] = Field(default_factory=list)
    default_persona: str = Field(default="girl")


class SessionInfo(BaseModel):
    """会话信息"""

    user_id: str
    bot_id: str
    persona: str
    enable_memory: bool
    message_count: int
    created_at: str


class SessionListResponse(BaseModel):
    """会话列表响应"""

    success: bool = Field(default=True)
    sessions: List[SessionInfo] = Field(default_factory=list)
    total: int = Field(default=0)


class GenericResponse(BaseModel):
    """通用响应"""

    success: bool = Field(default=True)
    message: str = Field(default="")
    error: Optional[str] = Field(default=None)


class HealthResponse(BaseModel):
    """健康检查响应"""

    status: str = Field(default="ok")
    version: str = Field(default="1.0.0")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
