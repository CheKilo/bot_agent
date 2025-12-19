# -*- coding: utf-8 -*-
"""
gRPC 协议定义模块

包含 LLM Proxy 和 Storage 服务的协议定义。
"""

# ============================================================================
# LLM Proxy 协议
# ============================================================================

from .llm_pb2 import (
    # 消息内容类型
    ContentPart,
    ImageUrl,
    ContentList,
    ChatMessage,
    # Chat Completion 请求/响应
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    # 流式响应
    ChatCompletionChunk,
    StreamChoice,
    ChatMessageDelta,
    # 通用
    Usage,
    # Function Calling / Tools
    Tool,
    FunctionDefinition,
    ToolCall,
    FunctionCall,
    # Embedding
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingData,
)

from .llm_pb2_grpc import (
    LLMProxyServiceStub,
    LLMProxyServiceServicer,
    add_LLMProxyServiceServicer_to_server,
)

# ============================================================================
# Storage 协议
# ============================================================================

from .storage_pb2 import (
    # 通用类型
    TypedValue,
    NullValue,
    # MySQL 操作
    ExecuteRequest,
    ExecuteResponse,
    Operation,
    InsertOperation,
    InsertRow,
    UpdateOperation,
    DeleteOperation,
    SelectOperation,
    WhereClause,
    OrderBy,
    Pagination,
    # MySQL 结果
    OperationResult,
    InsertResult,
    UpdateResult,
    DeleteResult,
    SelectResult,
    ResultRow,
    # Milvus 向量操作
    ExecuteVectorRequest,
    ExecuteVectorResponse,
    VectorOperation,
    VectorInsertOperation,
    VectorUpsertOperation,
    VectorSearchOperation,
    VectorDeleteOperation,
    VectorData,
    # Milvus 结果
    VectorOperationResult,
    VectorInsertResult,
    VectorUpsertResult,
    VectorSearchResult,
    VectorDeleteResult,
    VectorMatch,
)

from .storage_pb2_grpc import (
    StorageServiceStub,
    StorageServiceServicer,
    add_StorageServiceServicer_to_server,
)

__all__ = [
    # ========== LLM Proxy ==========
    # 消息内容类型
    "ContentPart",
    "ImageUrl",
    "ContentList",
    "ChatMessage",
    # Chat Completion 请求/响应
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "Choice",
    # 流式响应
    "ChatCompletionChunk",
    "StreamChoice",
    "ChatMessageDelta",
    # 通用
    "Usage",
    # Function Calling / Tools
    "Tool",
    "FunctionDefinition",
    "ToolCall",
    "FunctionCall",
    # Embedding
    "EmbeddingRequest",
    "EmbeddingResponse",
    "EmbeddingData",
    # gRPC 服务
    "LLMProxyServiceStub",
    "LLMProxyServiceServicer",
    "add_LLMProxyServiceServicer_to_server",
    # ========== Storage ==========
    # 通用类型
    "TypedValue",
    "NullValue",
    # MySQL 操作
    "ExecuteRequest",
    "ExecuteResponse",
    "Operation",
    "InsertOperation",
    "InsertRow",
    "UpdateOperation",
    "DeleteOperation",
    "SelectOperation",
    "WhereClause",
    "OrderBy",
    "Pagination",
    # MySQL 结果
    "OperationResult",
    "InsertResult",
    "UpdateResult",
    "DeleteResult",
    "SelectResult",
    "ResultRow",
    # Milvus 向量操作
    "ExecuteVectorRequest",
    "ExecuteVectorResponse",
    "VectorOperation",
    "VectorInsertOperation",
    "VectorUpsertOperation",
    "VectorSearchOperation",
    "VectorDeleteOperation",
    "VectorData",
    # Milvus 结果
    "VectorOperationResult",
    "VectorInsertResult",
    "VectorUpsertResult",
    "VectorSearchResult",
    "VectorDeleteResult",
    "VectorMatch",
    # gRPC 服务
    "StorageServiceStub",
    "StorageServiceServicer",
    "add_StorageServiceServicer_to_server",
]
