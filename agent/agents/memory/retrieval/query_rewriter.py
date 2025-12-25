# -*- coding: utf-8 -*-
"""
Query 改写器

职责：
1. 统一改写：同时生成中期和长期记忆的检索 query
2. 多关键词扩展：生成同义词、相关词提高召回率
3. 时间具化：将模糊时间转为具体日期
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from agent.core import LLM
from agent.agents.memory.config import QueryRewriterConfig

logger = logging.getLogger(__name__)


@dataclass
class RewriteResult:
    """改写结果"""

    # 中期记忆检索用
    mid_term_query: str  # 时间具化后的 query
    mid_term_keywords: List[str]  # 扩展关键词列表（用于 BM25 多路召回）

    # 长期记忆检索用
    long_term_query: str  # 精简后的语义 query
    long_term_keywords: List[str]  # 核心关键词（用于向量召回后的精排）


class QueryRewriter:
    """
    Query 改写器

    核心改进：
    - 统一改写接口，一次 LLM 调用生成所有改写结果
    - 多关键词扩展，提高召回率
    - 同义词/相关词生成
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

    def rewrite_unified(self, query: str) -> RewriteResult:
        """
        统一改写入口

        一次 LLM 调用同时生成：
        - 中期记忆：时间具化 query + 扩展关键词
        - 长期记忆：精简 query + 核心关键词
        """
        if not query or not query.strip():
            return RewriteResult(
                mid_term_query=query,
                mid_term_keywords=[],
                long_term_query=query,
                long_term_keywords=[],
            )

        query = query.strip()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        prompt = f"""当前时间：{now}
用户查询：{query}

任务：为记忆检索生成优化的查询词。

要求：
1. mid_term_query：将模糊时间转为具体日期（如"昨天"→"2025-12-23"）
2. mid_term_keywords：提取/扩展关键词，包括同义词、相关词（3-8个）
3. long_term_query：提取核心语义，去除时间词和口语化表达
4. long_term_keywords：提取核心实体/概念词（2-5个）

示例：
输入："昨天和小明聊了啥关于旅游的"
输出：
{{
  "mid_term_query": "2025-12-23 小明 旅游",
  "mid_term_keywords": ["小明", "旅游", "出行", "度假", "聊天", "讨论"],
  "long_term_query": "小明 旅游相关话题",
  "long_term_keywords": ["小明", "旅游", "出行"]
}}

输入："我之前好像说过喜欢吃什么"
输出：
{{
  "mid_term_query": "喜欢吃 食物 偏好",
  "mid_term_keywords": ["喜欢", "食物", "口味", "偏好", "美食"],
  "long_term_query": "用户喜欢的食物",
  "long_term_keywords": ["食物偏好", "口味", "喜欢吃"]
}}

直接返回 JSON，不要其他内容。"""

        try:
            response = self.llm.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300,
            )
            result = json.loads(response.content or "{}")
            return RewriteResult(
                mid_term_query=result.get("mid_term_query", query),
                mid_term_keywords=result.get("mid_term_keywords", []),
                long_term_query=result.get("long_term_query", query),
                long_term_keywords=result.get("long_term_keywords", []),
            )
        except Exception as e:
            logger.warning(f"Unified rewrite failed: {e}, using original query")
            # 降级：简单分词作为关键词
            keywords = [w for w in query.split() if len(w) > 1]
            return RewriteResult(
                mid_term_query=query,
                mid_term_keywords=keywords,
                long_term_query=query,
                long_term_keywords=keywords,
            )

    def normalize_for_storage(self, content: str) -> str:
        """
        存储规范化：与检索时的语义空间对齐
        """
        if not content or not content.strip():
            return content

        content = content.strip()

        prompt = f"""记忆内容：{content}

规范化为第三人称描述，提取核心信息。
示例：我喜欢吃川菜 → 用户喜欢吃川菜

直接返回规范化后的文本。"""

        try:
            response = self.llm.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200,
            )
            return (response.content or "").strip() or content
        except Exception as e:
            logger.error(f"Normalize failed: {e}")
            return content

    def close(self):
        """关闭资源"""
        if self._llm:
            self._llm.close()
            self._llm = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
