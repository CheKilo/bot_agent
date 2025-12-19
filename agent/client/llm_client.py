# -*- coding: utf-8 -*-
"""
LLM Proxy gRPC 客户端

用于调用 Go 端提供的 LLM Proxy 服务，支持：
- 非流式对话 (ChatCompletion)
- 流式对话 (ChatCompletionStream)
- 获取 Embedding 向量 (GetEmbedding)
"""

import json
import logging
from typing import Dict, Generator, List, Optional, Union

import grpc

from agent.pb import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    ChatMessage,
    ContentPart,
    ContentList,
    ImageUrl,
    Tool,
    FunctionDefinition,
    ToolCall,
    FunctionCall,
    EmbeddingRequest,
    EmbeddingResponse,
    LLMProxyServiceStub,
    Usage,
)

logger = logging.getLogger(__name__)


class LLMClientError(Exception):
    """LLM 客户端异常基类"""

    pass


class LLMConnectionError(LLMClientError):
    """连接异常"""

    pass


class LLMRequestError(LLMClientError):
    """请求异常"""

    pass


class LLMClient:
    """
    LLM Proxy gRPC 客户端

    用于与 Go 端的 LLM Proxy 服务进行通信。

    使用示例:
        ```python
        # 创建客户端
        client = LLMClient("localhost:50051")

        # 非流式对话
        response = client.chat_completion(
            deployment_id="gpt-5",
            messages=[
                {"role": "system", "content": "你是一个助手"},
                {"role": "user", "content": "你好"}
            ]
        )
        print(response.choices[0].message.content)

        # 流式对话
        for chunk in client.chat_completion_stream(
            deployment_id="gpt-5",
            messages=[{"role": "user", "content": "讲个故事"}]
        ):
            if chunk.choices and chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)

        # 获取 Embedding
        embedding_response = client.get_embedding(
            deployment_id="text-embedding-ada-002",
            input_texts=["Hello, world!"]
        )
        print(embedding_response.data[0].embedding)
        ```
    """

    # 默认 API 版本
    DEFAULT_API_VERSION = "2024-05-01-preview"

    # 默认超时时间（秒）
    DEFAULT_TIMEOUT = 60.0

    # 流式请求默认超时时间（秒）
    STREAM_TIMEOUT = 300.0

    def __init__(
        self,
        address: str,
        timeout: float = DEFAULT_TIMEOUT,
        use_ssl: bool = False,
        ssl_credentials: Optional[grpc.ChannelCredentials] = None,
    ):
        """
        初始化 LLM 客户端

        Args:
            address: gRPC 服务地址，格式为 "host:port"
            timeout: 默认请求超时时间（秒）
            use_ssl: 是否使用 SSL/TLS 连接
            ssl_credentials: SSL 凭证（use_ssl=True 时可选）
        """
        self._address = address
        self._timeout = timeout
        self._use_ssl = use_ssl
        self._ssl_credentials = ssl_credentials
        self._channel: Optional[grpc.Channel] = None
        self._stub: Optional[LLMProxyServiceStub] = None

    def _get_channel(self) -> grpc.Channel:
        """获取或创建 gRPC channel"""
        if self._channel is None:
            try:
                if self._use_ssl:
                    credentials = (
                        self._ssl_credentials or grpc.ssl_channel_credentials()
                    )
                    self._channel = grpc.secure_channel(self._address, credentials)
                else:
                    self._channel = grpc.insecure_channel(self._address)
            except Exception as e:
                raise LLMConnectionError(f"Failed to create gRPC channel: {e}") from e
        return self._channel

    def _get_stub(self) -> LLMProxyServiceStub:
        """获取或创建 gRPC stub"""
        if self._stub is None:
            self._stub = LLMProxyServiceStub(self._get_channel())
        return self._stub

    def close(self):
        """关闭 gRPC 连接"""
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @staticmethod
    def _build_chat_message(msg: Union[Dict, ChatMessage]) -> ChatMessage:
        """
        构建 ChatMessage 对象

        Args:
            msg: 消息字典或 ChatMessage 对象
                字典格式：
                    - role: str - 角色 ("system" / "user" / "assistant" / "tool")
                    - content: str | List[Dict] - 消息内容
                    - name: str (可选) - 发送者名称
                    - tool_calls: List[Dict] (可选) - 工具调用
                    - tool_call_id: str (可选) - 工具调用 ID

        Returns:
            ChatMessage 对象
        """
        if isinstance(msg, ChatMessage):
            return msg

        role = msg.get("role", "")
        content = msg.get("content")
        name = msg.get("name", "")
        tool_call_id = msg.get("tool_call_id", "")

        chat_msg = ChatMessage(role=role, name=name, tool_call_id=tool_call_id)

        # 处理内容
        if content is not None:
            if isinstance(content, str):
                # 简单文本内容
                chat_msg.content = content
            elif isinstance(content, list):
                # 多模态内容
                content_parts = []
                for part in content:
                    content_part = ContentPart(type=part.get("type", "text"))
                    if part.get("type") == "text":
                        content_part.text = part.get("text", "")
                    elif part.get("type") == "image_url":
                        image_data = part.get("image_url", {})
                        content_part.image_url.CopyFrom(
                            ImageUrl(
                                url=image_data.get("url", ""),
                                detail=image_data.get("detail", "auto"),
                            )
                        )
                    content_parts.append(content_part)
                chat_msg.content_parts.CopyFrom(ContentList(parts=content_parts))

        # 处理工具调用
        tool_calls = msg.get("tool_calls", [])
        for tc in tool_calls:
            tool_call = ToolCall(
                id=tc.get("id", ""),
                type=tc.get("type", "function"),
            )
            func = tc.get("function", {})
            tool_call.function.CopyFrom(
                FunctionCall(
                    name=func.get("name", ""), arguments=func.get("arguments", "")
                )
            )
            chat_msg.tool_calls.append(tool_call)

        return chat_msg

    @staticmethod
    def _build_tool(tool: Union[Dict, Tool]) -> Tool:
        """
        构建 Tool 对象

        Args:
            tool: 工具字典或 Tool 对象
                字典格式：
                    - type: str - 工具类型（目前只支持 "function"）
                    - function: Dict - 函数定义
                        - name: str - 函数名称
                        - description: str - 函数描述
                        - parameters: Dict | str - 参数 JSON Schema

        Returns:
            Tool 对象
        """
        if isinstance(tool, Tool):
            return tool

        func_def = tool.get("function", {})
        parameters = func_def.get("parameters", {})
        if isinstance(parameters, dict):
            parameters = json.dumps(parameters, ensure_ascii=False)

        return Tool(
            type=tool.get("type", "function"),
            function=FunctionDefinition(
                name=func_def.get("name", ""),
                description=func_def.get("description", ""),
                parameters=parameters,
            ),
        )

    def _build_chat_completion_request(
        self,
        deployment_id: str,
        messages: List[Union[Dict, ChatMessage]],
        api_version: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop: Optional[List[str]] = None,
        user: Optional[str] = None,
        n: Optional[int] = None,
        seed: Optional[int] = None,
        response_format: Optional[str] = None,
        tools: Optional[List[Union[Dict, Tool]]] = None,
        tool_choice: Optional[str] = None,
    ) -> ChatCompletionRequest:
        """构建 ChatCompletionRequest 对象"""
        request = ChatCompletionRequest(
            deployment_id=deployment_id,
            api_version=api_version or self.DEFAULT_API_VERSION,
        )

        # 添加消息
        for msg in messages:
            request.messages.append(self._build_chat_message(msg))

        # 设置可选参数
        if temperature is not None:
            request.temperature = temperature
        if max_tokens is not None:
            request.max_tokens = max_tokens
        if top_p is not None:
            request.top_p = top_p
        if frequency_penalty is not None:
            request.frequency_penalty = frequency_penalty
        if presence_penalty is not None:
            request.presence_penalty = presence_penalty
        if stop:
            request.stop.extend(stop)
        if user:
            request.user = user
        if n is not None:
            request.n = n
        if seed is not None:
            request.seed = seed
        if response_format:
            request.response_format = response_format
        if tools:
            for tool in tools:
                request.tools.append(self._build_tool(tool))
        if tool_choice:
            request.tool_choice = tool_choice

        return request

    def chat_completion(
        self,
        deployment_id: str,
        messages: List[Union[Dict, ChatMessage]],
        api_version: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop: Optional[List[str]] = None,
        user: Optional[str] = None,
        n: Optional[int] = None,
        seed: Optional[int] = None,
        response_format: Optional[str] = None,
        tools: Optional[List[Union[Dict, Tool]]] = None,
        tool_choice: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> ChatCompletionResponse:
        """
        非流式对话

        Args:
            deployment_id: 模型部署 ID，如 "gpt-5" / "gpt-5-mini" / "gpt-5-nano"
            messages: 对话历史，每条消息包含 role 和 content
            api_version: API 版本，默认 "2024-05-01-preview"
            temperature: 温度参数 (0-2)，默认 1
            max_tokens: 最大生成 token 数
            top_p: nucleus sampling，默认 1
            frequency_penalty: 频率惩罚 (-2 到 2)，默认 0
            presence_penalty: 存在惩罚 (-2 到 2)，默认 0
            stop: 停止词列表
            user: 用户标识
            n: 返回几个候选回复，默认 1
            seed: 随机种子（用于可复现结果）
            response_format: 响应格式，"text" 或 "json_object"
            tools: 可用工具列表
            tool_choice: 工具选择策略，"none" / "auto" / "required"
            timeout: 请求超时时间（秒）

        Returns:
            ChatCompletionResponse 对象

        Raises:
            LLMRequestError: 请求失败
        """
        request = self._build_chat_completion_request(
            deployment_id=deployment_id,
            messages=messages,
            api_version=api_version,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            user=user,
            n=n,
            seed=seed,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
        )

        try:
            stub = self._get_stub()
            response = stub.ChatCompletion(request, timeout=timeout or self._timeout)
            return response
        except grpc.RpcError as e:
            logger.error(f"ChatCompletion request failed: {e.code()}: {e.details()}")
            raise LLMRequestError(
                f"ChatCompletion request failed: {e.details()}"
            ) from e
        except Exception as e:
            logger.error(f"ChatCompletion request error: {e}")
            raise LLMRequestError(f"ChatCompletion request error: {e}") from e

    def chat_completion_stream(
        self,
        deployment_id: str,
        messages: List[Union[Dict, ChatMessage]],
        api_version: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop: Optional[List[str]] = None,
        user: Optional[str] = None,
        n: Optional[int] = None,
        seed: Optional[int] = None,
        response_format: Optional[str] = None,
        tools: Optional[List[Union[Dict, Tool]]] = None,
        tool_choice: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Generator[ChatCompletionChunk, None, None]:
        """
        流式对话

        Args:
            deployment_id: 模型部署 ID
            messages: 对话历史
            api_version: API 版本
            temperature: 温度参数 (0-2)
            max_tokens: 最大生成 token 数
            top_p: nucleus sampling
            frequency_penalty: 频率惩罚 (-2 到 2)
            presence_penalty: 存在惩罚 (-2 到 2)
            stop: 停止词列表
            user: 用户标识
            n: 返回几个候选回复
            seed: 随机种子
            response_format: 响应格式
            tools: 可用工具列表
            tool_choice: 工具选择策略
            timeout: 请求超时时间（秒）

        Yields:
            ChatCompletionChunk 对象（流式响应）

        Raises:
            LLMRequestError: 请求失败
        """
        request = self._build_chat_completion_request(
            deployment_id=deployment_id,
            messages=messages,
            api_version=api_version,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            user=user,
            n=n,
            seed=seed,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
        )

        try:
            stub = self._get_stub()
            response_stream = stub.ChatCompletionStream(
                request, timeout=timeout or self.STREAM_TIMEOUT
            )
            for chunk in response_stream:
                yield chunk
        except grpc.RpcError as e:
            logger.error(
                f"ChatCompletionStream request failed: {e.code()}: {e.details()}"
            )
            raise LLMRequestError(
                f"ChatCompletionStream request failed: {e.details()}"
            ) from e
        except Exception as e:
            logger.error(f"ChatCompletionStream request error: {e}")
            raise LLMRequestError(f"ChatCompletionStream request error: {e}") from e

    def get_embedding(
        self,
        deployment_id: str,
        input_texts: List[str],
        api_version: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> EmbeddingResponse:
        """
        获取 Embedding 向量

        Args:
            deployment_id: Embedding 模型部署 ID
            input_texts: 待向量化的文本列表
            api_version: API 版本，默认 "2024-05-01-preview"
            timeout: 请求超时时间（秒）

        Returns:
            EmbeddingResponse 对象，包含向量数据

        Raises:
            LLMRequestError: 请求失败
        """
        request = EmbeddingRequest(
            deployment_id=deployment_id,
            api_version=api_version or self.DEFAULT_API_VERSION,
        )
        request.input.extend(input_texts)

        try:
            stub = self._get_stub()
            response = stub.GetEmbedding(request, timeout=timeout or self._timeout)
            return response
        except grpc.RpcError as e:
            logger.error(f"GetEmbedding request failed: {e.code()}: {e.details()}")
            raise LLMRequestError(f"GetEmbedding request failed: {e.details()}") from e
        except Exception as e:
            logger.error(f"GetEmbedding request error: {e}")
            raise LLMRequestError(f"GetEmbedding request error: {e}") from e


# 便捷函数（用于简单场景）
_default_client: Optional[LLMClient] = None


def get_default_client(address: str = "localhost:50051") -> LLMClient:
    """
    获取默认客户端单例

    Args:
        address: gRPC 服务地址

    Returns:
        LLMClient 实例
    """
    global _default_client
    if _default_client is None:
        _default_client = LLMClient(address)
    return _default_client


def chat(
    messages: List[Dict],
    deployment_id: str = "gpt-5",
    address: str = "localhost:50051",
    **kwargs,
) -> str:
    """
    便捷的对话函数

    Args:
        messages: 对话历史
        deployment_id: 模型部署 ID
        address: gRPC 服务地址
        **kwargs: 其他参数传递给 chat_completion

    Returns:
        助手回复内容
    """
    client = get_default_client(address)
    response = client.chat_completion(
        deployment_id=deployment_id, messages=messages, **kwargs
    )
    if response.choices:
        return response.choices[0].message.content
    return ""


def chat_stream(
    messages: List[Dict],
    deployment_id: str = "gpt-5",
    address: str = "localhost:50051",
    **kwargs,
) -> Generator[str, None, None]:
    """
    便捷的流式对话函数

    Args:
        messages: 对话历史
        deployment_id: 模型部署 ID
        address: gRPC 服务地址
        **kwargs: 其他参数传递给 chat_completion_stream

    Yields:
        增量文本内容
    """
    client = get_default_client(address)
    for chunk in client.chat_completion_stream(
        deployment_id=deployment_id, messages=messages, **kwargs
    ):
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def embed(
    texts: List[str],
    deployment_id: str = "text-embedding-ada-002",
    address: str = "localhost:50051",
) -> List[List[float]]:
    """
    便捷的 Embedding 函数

    Args:
        texts: 待向量化的文本列表
        deployment_id: Embedding 模型部署 ID
        address: gRPC 服务地址

    Returns:
        向量列表
    """
    client = get_default_client(address)
    response = client.get_embedding(
        deployment_id=deployment_id,
        input_texts=texts,
    )
    return [list(data.embedding) for data in response.data]
