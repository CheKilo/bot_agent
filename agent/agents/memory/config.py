# -*- coding: utf-8 -*-
"""
记忆模块配置

所有配置集中管理
"""

from dataclasses import dataclass, field
from typing import Optional

# ========== 存储配置 ==========
MYSQL_DATABASE = "bot_memory"
MYSQL_MID_TERM_TABLE = "mid_term_memory"
MILVUS_COLLECTION = "memory_vectors"


def get_milvus_partition(bot_id: str) -> str:
    """获取 Milvus partition 名称"""
    safe_id = "".join(c if c.isalnum() else "_" for c in bot_id)
    return f"bot_{safe_id}"


# ========== 记忆参数 ==========
MESSAGE_WINDOW_CAPACITY = 20  # 短期记忆消息窗口容量
RECENT_SUMMARY_COUNT = 3  # 每次对话默认携带的最近摘要数量
PROMOTION_THRESHOLD = 3
DEFAULT_TIME_RANGE_DAYS = 30
DEFAULT_SEARCH_LIMIT = 5
DEFAULT_MIN_SCORE = 0.3

# ========== LLM 配置 ==========
DEFAULT_LLM_ADDRESS = "localhost:50051"
DEFAULT_LLM_TIMEOUT = 10.0
QUERY_LLM_MODEL = "gpt-4o-mini"

# ========== 排序器配置 ==========
COARSE_RANK_LIMIT = 100
MIN_SCORE_THRESHOLD = 0.1


# ========== 配置类 ==========


@dataclass
class LLMConfig:
    """LLM 配置"""

    address: str = DEFAULT_LLM_ADDRESS
    model: str = QUERY_LLM_MODEL
    timeout: float = DEFAULT_LLM_TIMEOUT


@dataclass
class QueryRewriterConfig:
    """Query 改写器配置"""

    llm: LLMConfig = field(default_factory=LLMConfig)


@dataclass
class RankerConfig:
    """排序器配置"""

    coarse_rank_limit: int = COARSE_RANK_LIMIT
    min_score_threshold: float = MIN_SCORE_THRESHOLD


@dataclass
class MemoryConfig:
    """记忆管理器配置"""

    # Query 改写器
    query_rewriter: QueryRewriterConfig = field(default_factory=QueryRewriterConfig)

    # 排序器
    ranker: RankerConfig = field(default_factory=RankerConfig)

    # 最近摘要数量（每次对话默认携带）
    recent_summary_count: int = RECENT_SUMMARY_COUNT

    # 检索
    min_score: float = DEFAULT_MIN_SCORE
