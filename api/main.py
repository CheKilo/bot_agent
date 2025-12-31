# -*- coding: utf-8 -*-
"""
Bot Agent API 服务

基于 FastAPI 的对话服务接口。

启动方式：
    # 开发模式
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

    # 生产模式
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

    # 或使用模块方式启动
    python -m api.main

API 文档：
    - Swagger UI: http://localhost:8000/docs
    - ReDoc: http://localhost:8000/redoc

环境变量：
    - API_HOST: 服务主机地址 (默认: 0.0.0.0)
    - API_PORT: 服务端口 (默认: 8000)
    - API_DEBUG: 调试模式 (默认: false)
    - GRPC_HOST: gRPC 服务主机 (默认: localhost)
    - GRPC_PORT: gRPC 服务端口 (默认: 50051)
    - LLM_MODEL: LLM 模型名称 (默认: gpt-5)
    - EMBEDDING_MODEL: Embedding 模型名称 (默认: text-embedding-ada-002)
    - DEFAULT_PERSONA: 默认人设 (默认: girl)
    - ENABLE_MEMORY: 是否启用记忆功能 (默认: true)
"""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import settings
from api.routes import router
from api.service import chat_service

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ============================================================================
# 生命周期管理
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    logger.info("=" * 60)
    logger.info("Bot Agent API 服务正在启动...")
    logger.info("=" * 60)

    try:
        chat_service.initialize()
        logger.info(f"服务地址: http://{settings.server.host}:{settings.server.port}")
        logger.info(f"gRPC 地址: {settings.grpc.address}")
        logger.info(
            f"API 文档: http://{settings.server.host}:{settings.server.port}/docs"
        )
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"服务初始化失败: {e}", exc_info=True)
        raise

    yield

    # 关闭时清理
    logger.info("=" * 60)
    logger.info("Bot Agent API 服务正在关闭...")
    chat_service.shutdown()
    logger.info("服务已关闭")
    logger.info("=" * 60)


# ============================================================================
# 创建应用
# ============================================================================


app = FastAPI(
    title="Bot Agent API",
    description="""
## Bot Agent 对话服务 API

基于 Multi-Agent 架构的智能对话系统，支持：

- 🤖 **多轮对话**：维护对话上下文，支持连续对话
- 🧠 **记忆管理**：三级记忆架构（短期/中期/长期）
- 🎭 **角色扮演**：支持多种人设配置
- 📊 **会话管理**：查看、清空、删除会话

### 快速开始

1. 发送对话请求到 `/chat` 端点
2. 使用 `user_id` 标识用户，系统自动维护会话
3. 可选配置 `persona` 选择人设，`enable_memory` 开关记忆功能

### 架构说明

- **System Agent**: 系统调度，协调各子 Agent
- **Memory Agent**: 记忆检索和存储
- **Character Agent**: 角色回复生成
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ============================================================================
# 中间件
# ============================================================================

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应配置具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# 注册路由
# ============================================================================

app.include_router(router, prefix="/api/v1")


# 根路径重定向到文档
@app.get("/", include_in_schema=False)
async def root():
    """根路径"""
    return {
        "service": "Bot Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


# ============================================================================
# 主入口
# ============================================================================


def main():
    """主入口函数"""
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=settings.server.debug,
        workers=settings.server.workers if not settings.server.debug else 1,
    )


if __name__ == "__main__":
    main()
