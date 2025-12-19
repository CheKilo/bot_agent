# -*- coding: utf-8 -*-
"""
排序器

排序因子：
- 中期记忆：BM25(0.6) + 时间衰减(0.3) + 访问热度(0.1)
- 长期记忆：向量分数(0.5) + 重要性(0.25) + 上下文匹配(0.15) + 时间(0.1)
"""

import logging
import time

logger = logging.getLogger(__name__)
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent.agents.memory.retrieval.bm25 import BM25
from agent.agents.memory.config import RankerConfig


@dataclass
class RankItem:
    """可排序项"""

    id: Any
    source: str  # mid_term | long_term
    content: str  # 展示内容（摘要）
    raw_content: str = ""  # 完整原始内容（用于BM25匹配）

    # 分数
    bm25_score: float = 0.0
    vector_score: float = 0.0
    final_score: float = 0.0

    # 元数据
    created_at: int = 0
    access_count: int = 0
    importance: int = 5
    keywords: str = ""
    metadata: Dict = field(default_factory=dict)


class Ranker:
    """排序器"""

    def __init__(self, config: Optional[RankerConfig] = None):
        self.config = config or RankerConfig()
        self._bm25 = BM25()

    # ========== 中期记忆排序 ==========

    def rank(
        self,
        query: str,
        items: List[RankItem],
        limit: int = 10,
    ) -> List[RankItem]:
        """
        中期记忆排序

        排序因子：BM25(0.6) + 时间(0.3) + 热度(0.1)
        BM25 基于 raw_content（完整原始内容）计算
        """
        if not items:
            return []

        # BM25 打分（基于完整原始内容）
        self._calc_bm25_scores(query, items)

        # 打印所有 BM25 分数供调试
        for item in items:
            logger.info(
                f"[DEBUG] BM25 score for item {item.id}: {item.bm25_score:.4f}, "
                f"threshold: {self.config.min_score_threshold}"
            )

        # 过滤低分 + 去重
        # 如果只有少量结果（<=3），降低阈值要求以保留更多结果
        threshold = self.config.min_score_threshold
        if len(items) <= 3:
            threshold = min(threshold, 0.01)  # 对少量结果放宽阈值
            logger.info(
                f"[DEBUG] Lowered threshold to {threshold} for {len(items)} items"
            )

        items = [x for x in items if x.bm25_score >= threshold]
        items = self._dedupe(items, key=lambda x: x.bm25_score)
        if not items:
            logger.info("[DEBUG] All items filtered out by BM25 threshold")
            return []

        # 精排
        now = time.time()
        max_access = max((i.access_count for i in items), default=1) or 1

        for item in items:
            # 时间衰减：30天周期
            time_score = self._time_decay(item.created_at, now, 30 * 86400)
            access_score = (
                item.access_count / max_access if item.access_count > 0 else 0
            )

            item.final_score = (
                item.bm25_score * 0.6 + time_score * 0.3 + access_score * 0.1
            )

        items.sort(key=lambda x: x.final_score, reverse=True)
        return items[:limit]

    # ========== 长期记忆排序 ==========

    def rank_long_term(
        self,
        query: str,
        items: List[RankItem],
        limit: int = 10,
    ) -> List[RankItem]:
        """
        长期记忆排序

        排序因子：向量(0.5) + 重要性(0.25) + 上下文匹配(0.15) + 时间(0.1)
        上下文匹配基于 raw_content（完整原始内容）计算
        """
        if not items:
            return []

        now = time.time()
        query_terms = set(query.lower().split()) if query else set()

        # 归一化向量分数
        # Milvus 返回的分数可能是距离（越小越好）或相似度（越大越好）
        # 这里假设是相似度分数（IP 内积），需要归一化到 [0, 1]
        max_vector_score = (
            max((item.vector_score for item in items), default=1.0) or 1.0
        )

        for item in items:
            # 向量分数归一化到 [0, 1]
            vector_score = (
                item.vector_score / max_vector_score if max_vector_score > 0 else 0.0
            )

            # 重要性分数 [0, 1]
            importance_score = (item.importance - 1) / 9.0

            # 上下文匹配分数（基于完整原始内容）
            context_score = self._context_match(query_terms, item)

            # 时间分数：365天周期
            time_score = self._time_decay(item.created_at, now, 365 * 86400)

            item.final_score = (
                vector_score * 0.5
                + importance_score * 0.25
                + context_score * 0.15
                + time_score * 0.1
            )

        items.sort(key=lambda x: x.final_score, reverse=True)
        return items[:limit]

    # ========== 内部方法 ==========

    def _calc_bm25_scores(self, query: str, items: List[RankItem]):
        """
        计算 BM25 分数并归一化

        使用 raw_content（完整原始内容）而非摘要，因为：
        - BM25 是词袋模型，完整内容词汇更丰富
        - 摘要会丢失很多细节词汇
        """
        docs = [
            {
                "id": i,
                "summary": item.raw_content or item.content,  # 优先用完整内容
                "keywords": item.keywords,
            }
            for i, item in enumerate(items)
        ]
        self._bm25.fit(docs)
        scores = self._bm25.get_doc_score_map(query)

        # 归一化
        max_score = max(scores.values(), default=1.0) or 1.0
        for i, item in enumerate(items):
            item.bm25_score = scores.get(i, 0.0) / max_score

    def _time_decay(self, created_at: int, now: float, period: int) -> float:
        """时间衰减：线性衰减到 0.1，返回值限制在 [0.1, 1.0]"""
        if created_at <= 0:
            return 0.5
        age = now - created_at
        # 限制返回值在 [0.1, 1.0] 范围内
        # 防止 created_at 为未来时间或异常值导致分数异常
        return max(0.1, min(1.0, 1.0 - age / period))

    def _context_match(self, query_terms: set, item: RankItem) -> float:
        """
        上下文匹配分数

        使用 raw_content（完整原始内容）匹配
        """
        if not query_terms:
            return 0.0

        # 优先使用完整原始内容
        match_text = item.raw_content or item.content
        if not match_text:
            return 0.0

        match_text_lower = match_text.lower()
        hit_count = sum(1 for term in query_terms if term in match_text_lower)
        return hit_count / len(query_terms)

    def _dedupe(self, items: List[RankItem], key) -> List[RankItem]:
        """去重：相同内容保留分数最高的"""
        items = sorted(items, key=key, reverse=True)
        seen = set()
        result = []
        for item in items:
            content_key = item.content[:100] if item.content else ""
            if content_key and content_key not in seen:
                seen.add(content_key)
                result.append(item)
        return result[: self.config.coarse_rank_limit]
