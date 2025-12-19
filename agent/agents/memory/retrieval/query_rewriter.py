# -*- coding: utf-8 -*-
"""
Query 改写器

职责：
1. 中期记忆改写：将口语化 query 改写 + 时间具化
2. 长期记忆改写：精简改写，保持语义纯净
"""

import logging
from datetime import datetime
from typing import Optional

from agent.core import LLM
from agent.agents.memory.config import QueryRewriterConfig

logger = logging.getLogger(__name__)


class QueryRewriter:
    """
    Query 改写器

    提供两种改写模式：
    - rewrite_for_mid_term: 改写 + 时间具化（用于 BM25 文本匹配）
    - rewrite_for_long_term: 精简改写（用于向量语义匹配）
    """

    def __init__(self, config: Optional[QueryRewriterConfig] = None):
        self.config = config or QueryRewriterConfig()
        self._llm: Optional[LLM] = None

    @property
    def llm(self) -> LLM:
        """懒加载 LLM"""
        if self._llm is None:
            self._llm = LLM(
                address=self.config.llm.address,
                model=self.config.llm.model,
                timeout=self.config.llm.timeout,
            )
        return self._llm

    def _call_llm(self, system: str, prompt: str, max_tokens: int = 200) -> str:
        """统一的 LLM 调用"""
        try:
            response = self.llm.chat(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=max_tokens,
            )
            return (response.content or "").strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return ""

    def rewrite_for_mid_term(self, query: str) -> str:
        """
        中期记忆改写：去口语化 + 时间具化
        """
        if not query or not query.strip():
            return query

        query = query.strip()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        prompt = f"""当前时间：{now}
用户查询：{query}

任务：改写为适合记忆检索的形式，模糊时间转具体日期。
示例：昨天聊了什么 → 2025-12-17的对话内容

直接返回改写后的文本。"""

        result = self._call_llm("查询改写助手", prompt)
        return result if result else query

    def rewrite_for_long_term(self, query: str) -> str:
        """
        长期记忆改写：精简核心语义
        """
        if not query or not query.strip():
            return query

        query = query.strip()

        prompt = f"""查询：{query}

提取核心语义，去除口语化表达和时间词。
示例：我之前好像说过喜欢吃什么 → 用户喜欢的食物

直接返回精简后的文本。"""

        result = self._call_llm("查询精简助手", prompt, max_tokens=100)
        return result if result else query

    def normalize_for_storage(self, content: str) -> str:
        """
        存储规范化：与 rewrite_for_long_term 保持一致的语义空间
        """
        if not content or not content.strip():
            return content

        content = content.strip()

        prompt = f"""记忆内容：{content}

规范化为第三人称描述，提取核心信息。
示例：我喜欢吃川菜 → 用户喜欢吃川菜

直接返回规范化后的文本。"""

        result = self._call_llm("记忆规范化助手", prompt)
        return result if result else content

    def close(self):
        """关闭资源"""
        if self._llm:
            self._llm.close()
            self._llm = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
