# -*- coding: utf-8 -*-
"""
BM25 检索

基于 rank_bm25 库实现
"""

import logging
from typing import Dict, List, Any

import jieba
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# 中文停用词（高频无意义词，会导致 BM25 负分）
STOPWORDS = {
    "的",
    "了",
    "是",
    "在",
    "我",
    "有",
    "和",
    "就",
    "不",
    "人",
    "都",
    "一",
    "一个",
    "上",
    "也",
    "很",
    "到",
    "说",
    "要",
    "去",
    "你",
    "会",
    "着",
    "没有",
    "看",
    "好",
    "自己",
    "这",
    "那",
    "什么",
    "吗",
    "呢",
    "吧",
    "啊",
    "哦",
    "嗯",
    "呀",
    "，",
    "。",
    "！",
    "？",
    "、",
    "；",
    "：",
    """, """,
    "'",
    "'",
    "[",
    "]",
    "（",
    "）",
    "(",
    ")",
    " ",
    "\n",
    "\t",
}


def tokenize(text: str) -> List[str]:
    """
    中英文混合分词

    策略：
    1. 使用 jieba.cut_for_search（搜索引擎模式）：会对长词进行细粒度切分
       例如 "喝咖啡" -> ["喝", "咖啡", "喝咖啡"]
    2. 过滤停用词：避免高频词导致 BM25 负分
    3. 对中文词额外添加单字拆分：进一步提高召回率
    """
    if not text:
        return []

    text = text.lower()

    # 使用搜索引擎模式分词（会对长词进行细粒度切分）
    tokens = list(jieba.cut_for_search(text))

    # 对长度大于2的中文词，额外添加单字拆分以提高召回率
    extra_chars = []
    for token in tokens:
        if len(token) > 2 and any("\u4e00" <= c <= "\u9fff" for c in token):
            extra_chars.extend(list(token))

    # 合并、过滤停用词、去重（保持顺序）
    seen = set()
    result = []
    for t in tokens + extra_chars:
        t = t.strip()
        if t and t not in seen and t not in STOPWORDS:
            seen.add(t)
            result.append(t)

    return result


class BM25:
    """
    BM25 封装

    用法：
        bm25 = BM25()
        bm25.fit(docs, text_field="summary")
        scores = bm25.get_doc_score_map(query)
    """

    def __init__(self):
        self._bm25: BM25Okapi = None
        self._doc_ids: List[Any] = []
        self._corpus: List[List[str]] = []

    def fit(self, documents: List[Dict[str, Any]], text_field: str = "summary"):
        """
        构建索引

        Args:
            documents: 文档列表，每个文档是 dict，必须包含 id 字段
            text_field: 用于索引的文本字段
        """
        self._doc_ids = []
        self._corpus = []

        for doc in documents:
            doc_id = doc.get("id", 0)
            # 合并多个字段用于检索
            text = f"{doc.get(text_field, '')} {doc.get('keywords', '')}"
            tokens = tokenize(text)

            self._doc_ids.append(doc_id)
            self._corpus.append(tokens)

        if self._corpus:
            self._bm25 = BM25Okapi(self._corpus)

    def get_doc_score_map(self, query: str) -> Dict[Any, float]:
        """
        返回所有文档的分数 map

        Returns:
            {doc_id: score}
        """
        if not self._doc_ids or not self._corpus:
            return {}

        query_tokens = tokenize(query)
        if not query_tokens:
            return {}

        # 调试日志
        logger.info(f"[DEBUG] BM25 query tokens: {query_tokens}")
        if self._corpus:
            logger.info(
                f"[DEBUG] BM25 first doc tokens (sample): {self._corpus[0][:20]}..."
            )

        # 当文档数量很少时（<=3），BM25 的 IDF 计算会导致负分
        # 因为 IDF = log((N - df + 0.5) / (df + 0.5))，当 N=1, df=1 时，IDF 为负
        # 此时使用简单的词匹配计分
        if len(self._corpus) <= 3:
            result = {}
            query_set = set(query_tokens)
            for idx, doc_tokens in enumerate(self._corpus):
                doc_set = set(doc_tokens)
                # 计算查询词和文档词的交集比例
                if query_set:
                    hit_count = len(query_set & doc_set)
                    score = hit_count / len(query_set)  # [0, 1]
                else:
                    score = 0.0
                result[self._doc_ids[idx]] = score
            logger.info(f"[DEBUG] Simple match scores (few docs): {result}")
            return result

        # 文档数量足够时使用 BM25
        scores = self._bm25.get_scores(query_tokens)

        result = {}
        for idx, score in enumerate(scores):
            result[self._doc_ids[idx]] = float(score)

        logger.info(f"[DEBUG] BM25 scores: {result}")
        return result

    def search(self, query: str, top_k: int = 10) -> List[tuple]:
        """
        搜索并返回 Top-K 结果

        Returns:
            [(doc_id, score), ...] 按分数降序排列
        """
        score_map = self.get_doc_score_map(query)
        if not score_map:
            return []

        sorted_items = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:top_k]
