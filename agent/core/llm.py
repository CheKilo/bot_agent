# -*- coding: utf-8 -*-
"""
Agent 核心 LLM 模块

封装 LLMClient，为 Agent 提供简单易用的 LLM 调用接口。
注：对话历史管理、工具调用等逻辑由各 Agent 模块自行负责。

使用示例：
    ```python
    from agent.core import LLM, Message

    # 简单对话
    response = llm.chat("你好")

    # 流式对话
    for chunk in llm.stream("写一首诗"):
        print(chunk, end="", flush=True)
    ```
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Union

from agent.client import LLMClient, LLMClientError
from agent.tools import ToolCall

logger = logging.getLogger(__name__)


# ============================================================================
# 数据类
# ============================================================================


@dataclass
class Message:
    """对话消息"""

    role: str
    content: Union[str, List[Dict], None] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None

    def to_dict(self) -> Dict:
        """转换为字典"""
        d = {"role": self.role}
        if self.content is not None:
            d["content"] = self.content
        if self.name:
            d["name"] = self.name
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: Union[str, List[Dict]]) -> "Message":
        return cls(role="user", content=content)

    @classmethod
    def assistant(
        cls, content: Optional[str] = None, tool_calls: Optional[List[Dict]] = None
    ) -> "Message":
        return cls(role="assistant", content=content, tool_calls=tool_calls)

    @classmethod
    def tool(cls, content: str, tool_call_id: str) -> "Message":
        return cls(role="tool", content=content, tool_call_id=tool_call_id)


@dataclass
class LLMResponse:
    """LLM 响应结果"""

    content: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    raw_response: Any = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def __str__(self) -> str:
        return self.content or ""

    def get_tool_call(self, index: int = 0) -> ToolCall:
        """获取工具调用对象"""
        if not self.tool_calls or index >= len(self.tool_calls):
            return ToolCall()
        return ToolCall.from_dict(self.tool_calls[index])

    def iter_tool_calls(self):
        """迭代所有工具调用 -> (index, call_id, name, args)"""
        for i, tc in enumerate(self.tool_calls or []):
            tool_call = ToolCall.from_dict(tc)
            yield i, tool_call.id, tool_call.name, tool_call.args


# ============================================================================
# LLM 核心类
# ============================================================================


class LLM:
    """Agent 核心 LLM 调用类"""

    DEFAULT_MODEL = "gpt-5"
    DEFAULT_ADDRESS = "localhost:50051"
    DEFAULT_EMBEDDING_MODEL = "text-embedding-ada-002"

    def __init__(
        self,
        address: str = DEFAULT_ADDRESS,
        model: str = DEFAULT_MODEL,
        timeout: float = 60.0,
    ):
        self._address = address
        self._model = model
        self._timeout = timeout
        self._client: Optional[LLMClient] = None

    @property
    def client(self) -> LLMClient:
        if self._client is None:
            self._client = LLMClient(address=self._address, timeout=self._timeout)
        return self._client

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, value: str):
        self._model = value

    def chat(
        self,
        messages: Union[str, Message, List[Union[Dict, Message]]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        response_format: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> LLMResponse:
        """发送对话请求（非流式）"""
        msg_list = self._to_msg_list(messages)

        response = self.client.chat_completion(
            deployment_id=model or self._model,
            messages=msg_list,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            timeout=timeout or self._timeout,
            **kwargs,
        )
        return self._parse_response(response)

    def stream(
        self,
        messages: Union[str, Message, List[Union[Dict, Message]]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Generator[str, None, None]:
        """发送流式对话请求，逐块 yield 文本"""
        msg_list = self._to_msg_list(messages)

        response_stream = self.client.chat_completion_stream(
            deployment_id=model or self._model,
            messages=msg_list,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            timeout=timeout,
            **kwargs,
        )

        for chunk in response_stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def embed(
        self,
        texts: Union[str, List[str]],
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> List[List[float]]:
        """获取文本 Embedding 向量"""
        if isinstance(texts, str):
            texts = [texts]

        response = self.client.get_embedding(
            deployment_id=model or self.DEFAULT_EMBEDDING_MODEL,
            input_texts=texts,
            timeout=timeout or self._timeout,
        )
        return [list(data.embedding) for data in response.data]

    def _to_msg_list(
        self, messages: Union[str, Message, List[Union[Dict, Message]]]
    ) -> List[Dict]:
        """标准化消息格式"""
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        if isinstance(messages, Message):
            return [messages.to_dict()]
        return [m.to_dict() if isinstance(m, Message) else m for m in messages]

    def _parse_response(self, response) -> LLMResponse:
        """解析 gRPC 响应"""
        if not response.choices:
            return LLMResponse(raw_response=response)

        choice = response.choices[0]
        msg = choice.message

        tool_calls = None
        if msg.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]

        usage = None
        if response.usage and response.usage.total_tokens > 0:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return LLMResponse(
            content=msg.content or None,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            usage=usage,
            raw_response=response,
        )

    def close(self):
        if self._client:
            self._client.close()
            self._client = None

    def __enter__(self) -> "LLM":
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self) -> str:
        return f"LLM(address={self._address!r}, model={self._model!r})"


# ============================================================================
# 便捷函数
# ============================================================================


def quick_chat(
    messages: Union[str, List[Dict]],
    model: str = LLM.DEFAULT_MODEL,
    address: str = LLM.DEFAULT_ADDRESS,
    **kwargs,
) -> str:
    """快速对话"""
    with LLM(address=address, model=model) as llm:
        return str(llm.chat(messages, **kwargs))


def quick_embed(
    texts: Union[str, List[str]],
    model: str = LLM.DEFAULT_EMBEDDING_MODEL,
    address: str = LLM.DEFAULT_ADDRESS,
) -> List[List[float]]:
    """快速获取 Embedding"""
    with LLM(address=address) as llm:
        return llm.embed(texts, model=model)
