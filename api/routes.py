# -*- coding: utf-8 -*-
"""
API 路由定义

定义所有 API 端点。
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from api.models import (
    ChatRequest,
    ChatResponse,
    ClearHistoryRequest,
    SetPersonaRequest,
    HistoryResponse,
    HistoryMessage,
    PersonaListResponse,
    PersonaInfo,
    SessionListResponse,
    SessionInfo,
    GenericResponse,
    HealthResponse,
)
from api.service import chat_service

logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter()


# ============================================================================
# 健康检查
# ============================================================================


@router.get("/health", response_model=HealthResponse, tags=["系统"])
async def health_check():
    """
    健康检查

    返回服务状态信息。
    """
    return HealthResponse(status="ok", version="1.0.0")


# ============================================================================
# 对话接口
# ============================================================================


@router.post("/chat", response_model=ChatResponse, tags=["对话"])
async def chat(request: ChatRequest):
    """
    发送对话消息

    根据 user_id 和 bot_id 管理会话，支持多轮对话。

    - **user_id**: 用户唯一标识（必需）
    - **bot_id**: Bot 标识（默认 default_bot）
    - **message**: 用户消息内容（必需）
    - **persona**: 人设名称（可选，仅在创建新会话时生效）
    - **enable_memory**: 是否启用记忆功能（可选，仅在创建新会话时生效）
    """
    try:
        result = chat_service.chat(
            user_id=request.user_id,
            bot_id=request.bot_id,
            message=request.message,
            persona=request.persona,
            enable_memory=request.enable_memory,
        )

        return ChatResponse(
            success=result.get("success", False),
            answer=result.get("answer", ""),
            iterations=result.get("iterations", 0),
            error=result.get("error"),
        )

    except Exception as e:
        logger.error(f"[API] /chat 错误: {e}", exc_info=True)
        return ChatResponse(
            success=False,
            answer="",
            error=str(e),
        )


# ============================================================================
# 对话历史
# ============================================================================


@router.get("/history", response_model=HistoryResponse, tags=["对话"])
async def get_history(
    user_id: str = Query(..., description="用户ID"),
    bot_id: str = Query(default="default_bot", description="Bot ID"),
):
    """
    获取对话历史

    返回指定用户和 Bot 的对话历史记录。
    """
    try:
        history = chat_service.get_history(bot_id=bot_id, user_id=user_id)

        messages = [
            HistoryMessage(
                role=msg.get("role", ""),
                content=msg.get("content", ""),
                timestamp=msg.get("timestamp"),
            )
            for msg in history
        ]

        user_turns = sum(1 for msg in history if msg.get("role") == "user")

        return HistoryResponse(
            success=True,
            user_id=user_id,
            bot_id=bot_id,
            messages=messages,
            total_messages=len(messages),
            user_turns=user_turns,
        )

    except Exception as e:
        logger.error(f"[API] /history 错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/history/clear", response_model=GenericResponse, tags=["对话"])
async def clear_history(request: ClearHistoryRequest):
    """
    清空对话历史

    清空指定用户和 Bot 的对话历史记录。
    """
    try:
        success = chat_service.clear_history(
            bot_id=request.bot_id,
            user_id=request.user_id,
        )

        if success:
            return GenericResponse(
                success=True,
                message=f"对话历史已清空 (user={request.user_id})",
            )
        else:
            return GenericResponse(
                success=False,
                message="未找到对应的会话",
            )

    except Exception as e:
        logger.error(f"[API] /history/clear 错误: {e}", exc_info=True)
        return GenericResponse(success=False, error=str(e))


# ============================================================================
# 人设管理
# ============================================================================


@router.get("/personas", response_model=PersonaListResponse, tags=["人设"])
async def list_personas():
    """
    获取可用人设列表

    返回所有可用的人设配置。
    """
    try:
        personas_dict = chat_service.get_personas()

        personas = [
            PersonaInfo(
                name=info["name"],
                display_name=info["display_name"],
                description=info.get("description"),
            )
            for info in personas_dict.values()
        ]

        return PersonaListResponse(
            success=True,
            personas=personas,
            default_persona="girl",
        )

    except Exception as e:
        logger.error(f"[API] /personas 错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/persona/set", response_model=GenericResponse, tags=["人设"])
async def set_persona(request: SetPersonaRequest):
    """
    设置会话人设

    更改指定会话的人设配置。
    """
    try:
        success = chat_service.set_persona(
            bot_id=request.bot_id,
            user_id=request.user_id,
            persona_name=request.persona,
        )

        if success:
            return GenericResponse(
                success=True,
                message=f"人设已更新为: {request.persona}",
            )
        else:
            # 获取可用人设
            available = list(chat_service.get_personas().keys())
            return GenericResponse(
                success=False,
                message=f"无效的人设或会话不存在。可用人设: {available}",
            )

    except Exception as e:
        logger.error(f"[API] /persona/set 错误: {e}", exc_info=True)
        return GenericResponse(success=False, error=str(e))


# ============================================================================
# 会话管理
# ============================================================================


@router.get("/sessions", response_model=SessionListResponse, tags=["会话"])
async def list_sessions():
    """
    列出所有活跃会话

    返回当前所有活跃的会话信息。
    """
    try:
        sessions = chat_service.list_sessions()

        session_infos = [
            SessionInfo(
                user_id=s["user_id"],
                bot_id=s["bot_id"],
                persona=s["persona"],
                enable_memory=s["enable_memory"],
                message_count=s["message_count"],
                created_at=s["created_at"],
            )
            for s in sessions
        ]

        return SessionListResponse(
            success=True,
            sessions=session_infos,
            total=len(session_infos),
        )

    except Exception as e:
        logger.error(f"[API] /sessions 错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session", response_model=SessionInfo, tags=["会话"])
async def get_session(
    user_id: str = Query(..., description="用户ID"),
    bot_id: str = Query(default="default_bot", description="Bot ID"),
):
    """
    获取会话信息

    返回指定会话的详细信息。
    """
    try:
        session = chat_service.get_session_info(bot_id=bot_id, user_id=user_id)

        if session:
            return SessionInfo(**session)
        else:
            raise HTTPException(status_code=404, detail="会话不存在")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] /session 错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/session", response_model=GenericResponse, tags=["会话"])
async def delete_session(
    user_id: str = Query(..., description="用户ID"),
    bot_id: str = Query(default="default_bot", description="Bot ID"),
):
    """
    删除会话

    删除指定的会话，释放相关资源。
    """
    try:
        success = chat_service.delete_session(bot_id=bot_id, user_id=user_id)

        if success:
            return GenericResponse(
                success=True,
                message=f"会话已删除 (user={user_id}, bot={bot_id})",
            )
        else:
            return GenericResponse(
                success=False,
                message="会话不存在",
            )

    except Exception as e:
        logger.error(f"[API] /session DELETE 错误: {e}", exc_info=True)
        return GenericResponse(success=False, error=str(e))
