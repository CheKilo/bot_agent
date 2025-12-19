# -*- coding: utf-8 -*-
"""
记忆管理器

统一管理中期记忆和长期记忆的存储与检索

架构:
- QueryRewriter: Query 改写器（中期/长期两种模式）
- Ranker: 粗排 + 精排
- MemoryManager: 统一入口

retrieval/ 下只保留独立工具类：
- query_rewriter.py (QueryRewriter)
- ranker.py (Ranker, RankItem)
- bm25.py (BM25)

检索流程:
- 中期记忆：改写(+时间具化) → 召回 → BM25粗排 → 精排 → Top-K
- 长期记忆：精简改写 → 向量召回 → 精排 → Top-K

新架构（无缓冲区）:
- 窗口满时由 Agent 调用 save_summary() 触发摘要存储
- 每次对话自动携带 get_recent_summaries() 返回的最近摘要
- search_mid_term() 只检索 MySQL 中的摘要
"""
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from agent.client import StorageClient
from agent.core import LLM
from agent.agents.memory.config import (
    # 存储
    MYSQL_DATABASE,
    MYSQL_MID_TERM_TABLE,
    MILVUS_COLLECTION,
    get_milvus_partition,
    # 常量
    PROMOTION_THRESHOLD,
    DEFAULT_MIN_SCORE,
    # 配置
    MemoryConfig,
)
from agent.agents.memory.retrieval import (
    QueryRewriter,
    Ranker,
    RankItem,
)

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """搜索结果"""

    id: Any
    source: str  # database | long_term
    content: str
    score: float

    keywords: str = ""
    created_at: int = 0
    importance: int = 5


class MemoryManager:
    """
    记忆管理器

    统一管理：
    - 中期记忆：MySQL 摘要存储
    - 长期记忆：Milvus 向量存储

    新架构特点：
    - 无缓冲区：窗口满时由 Agent 调用 save_summary() 直接存储
    - 每次对话自动携带最近 N 条摘要作为上下文
    - search_mid_term() 只检索 MySQL，返回结构统一
    """

    def __init__(
        self,
        bot_id: str,
        user_id: str,
        storage_client: StorageClient,
        embed_func: Callable[[str], List[float]],
        config: Optional[MemoryConfig] = None,
    ):
        self.bot_id = bot_id
        self.user_id = user_id
        self.storage = storage_client
        self.embed_func = embed_func
        self.config = config or MemoryConfig()

        # 访问计数（用于记忆提升）
        self._access_counter: Dict[int, int] = {}

        # Milvus partition
        self._partition = get_milvus_partition(bot_id)

        # 子模块（懒加载）
        self._summary_llm: Optional[LLM] = None
        self._query_rewriter: Optional[QueryRewriter] = None
        self._ranker: Optional[Ranker] = None

    # ========== 子模块懒加载 ==========

    @property
    def summary_llm(self) -> LLM:
        """摘要生成 LLM"""
        if self._summary_llm is None:
            cfg = self.config.summary_llm
            self._summary_llm = LLM(
                address=cfg.address,
                model=cfg.model,
                timeout=cfg.timeout,
            )
        return self._summary_llm

    @property
    def query_rewriter(self) -> QueryRewriter:
        """Query 改写器（独立 LLM）"""
        if self._query_rewriter is None:
            self._query_rewriter = QueryRewriter(self.config.query_rewriter)
        return self._query_rewriter

    @property
    def ranker(self) -> Ranker:
        """排序器"""
        if self._ranker is None:
            self._ranker = Ranker(self.config.ranker)
        return self._ranker

    # ========== 摘要存储（新接口） ==========

    def save_summary(
        self, messages: List[Dict[str, str]], raw_messages: Optional[List[Dict]] = None
    ) -> bool:
        """
        保存消息摘要到中期记忆

        Args:
            messages: 用于生成摘要的消息列表，格式为 {"role": "user/assistant", "content": "..."}
            raw_messages: 完整的原始消息列表（包含 tool 消息等），如果不传则使用 messages

        Returns:
            是否保存成功
        """
        if not messages:
            return False

        try:
            # 构建对话文本
            conversation = "\n".join(
                [
                    f"[{m.get('role', 'unknown')}]: {m.get('content', '')}"
                    for m in messages
                ]
            )

            # 生成摘要和关键词
            summary, keywords = self._generate_summary(conversation)
            if not summary:
                logger.warning("Failed to generate summary")
                return False

            # 保存到 MySQL（raw_messages 保存完整对话，如果没有则用 messages）
            raw_to_save = raw_messages if raw_messages is not None else messages
            self._save_to_mysql(messages, summary, keywords, raw_to_save)
            logger.info(
                f"Mid-term memory saved: {len(raw_to_save)} messages (summary from {len(messages)} dialog turns)"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to save summary: {e}")
            return False

    def get_recent_summaries(self, count: Optional[int] = None) -> List[Dict]:
        """
        获取最近的摘要

        Args:
            count: 返回数量，默认使用配置值

        Returns:
            摘要列表，每条包含 id, summary, keywords, created_at
        """
        if count is None:
            count = self.config.recent_summary_count

        try:
            rows = self.storage.select(
                database=MYSQL_DATABASE,
                table=MYSQL_MID_TERM_TABLE,
                conditions={"bot_id": self.bot_id, "user_id": self.user_id},
                order_by="created_at",
                descending=True,
                limit=count,
            )

            summaries = []
            for row in rows:
                created_at = row.get("created_at", 0)
                if hasattr(created_at, "timestamp"):
                    created_at = int(created_at.timestamp())

                summaries.append(
                    {
                        "id": row.get("id", 0),
                        "summary": row.get("summary", ""),
                        "keywords": row.get("keywords", ""),
                        "created_at": created_at,
                    }
                )

            return summaries

        except Exception as e:
            logger.error(f"Failed to get recent summaries: {e}")
            return []

    # ========== 中期记忆检索 ==========

    def search_mid_term(
        self,
        query: str,
        time_range_days: int = 30,
        limit: int = 5,
    ) -> List[SearchResult]:
        """
        搜索中期记忆

        流程：改写(+时间具化) → 召回 → BM25粗排 → 精排

        注意：只检索 MySQL 中的摘要，不再包含缓冲区
        """
        if not query or not query.strip():
            return []

        # 1. Query 改写（中期模式：改写 + 时间具化）
        rewritten_query = self.query_rewriter.rewrite_for_mid_term(query)
        logger.debug(f"Mid-term rewritten query: {rewritten_query}")

        # 2. 从 MySQL 召回
        rank_items = self._recall_mysql(time_range_days=time_range_days)
        logger.debug(f"MySQL recall: {len(rank_items)} items")
        if not rank_items:
            return []

        # 3. 粗排 + 精排
        ranked = self.ranker.rank(
            query=rewritten_query,
            items=rank_items,
            limit=limit,
        )

        logger.debug(f"After ranking: {len(ranked)} items")

        # 更新访问计数
        db_ids = [
            item.id
            for item in ranked
            if item.source == "mid_term" and isinstance(item.id, int)
        ]
        if db_ids:
            self._update_access_counts(db_ids)

        return self._to_search_results(ranked)

    def _recall_mysql(self, time_range_days: int) -> List[RankItem]:
        """从 MySQL 召回"""
        # 注意：Go 端的 buildWhereClause 中 raw_clause 和 conditions 是互斥的
        # 当 raw_clause 不为空时，conditions 会被完全忽略
        # 所以必须把所有条件都放到 raw_clause 中
        # 注意：Go MySQL 驱动使用 ? 作为占位符，不是 %s
        raw_clause = (
            f"bot_id = ? AND user_id = ? AND "
            f"created_at > NOW() - INTERVAL {int(time_range_days)} DAY"
        )
        raw_params = [self.bot_id, self.user_id]

        try:
            rows = self.storage.select(
                database=MYSQL_DATABASE,
                table=MYSQL_MID_TERM_TABLE,
                raw_clause=raw_clause,
                raw_params=raw_params,
                order_by="created_at",
                descending=True,
                limit=100,
            )
            logger.debug(f"MySQL recall: {len(rows)} rows")
        except Exception as e:
            logger.error(f"MySQL search failed: {e}")
            return []

        items = []
        for row in rows:
            created_at = row.get("created_at", 0)
            if hasattr(created_at, "timestamp"):
                created_at = int(created_at.timestamp())

            # 解析 raw_messages JSON，提取纯文本内容用于 BM25 匹配
            raw_messages_json = row.get("raw_messages", "")
            raw_content = ""
            if raw_messages_json:
                try:
                    messages = json.loads(raw_messages_json)
                    # 将消息内容拼接成纯文本
                    raw_content = " ".join(
                        m.get("content", "") for m in messages if m.get("content")
                    )
                except (json.JSONDecodeError, TypeError):
                    raw_content = raw_messages_json  # 解析失败则使用原始字符串

            items.append(
                RankItem(
                    id=row.get("id", 0),
                    source="mid_term",
                    content=row.get("summary", ""),
                    raw_content=raw_content,  # 纯文本内容用于BM25
                    keywords=row.get("keywords", ""),
                    created_at=created_at,
                    access_count=row.get("access_count", 0),
                )
            )

        return items

    def _to_search_results(self, items: List[RankItem]) -> List[SearchResult]:
        """转换为 SearchResult"""
        return [
            SearchResult(
                id=item.id,
                source="database",
                content=item.content,
                score=round(item.final_score, 3),
                keywords=item.keywords,
                created_at=item.created_at,
            )
            for item in items
        ]

    def _update_access_counts(self, record_ids: List[int]) -> None:
        """更新访问计数（使用 raw_set 支持 SQL 表达式）"""
        if not record_ids:
            return

        # 更新内存计数器（用于记忆提升判断）
        for rid in record_ids:
            self._access_counter[rid] = self._access_counter.get(rid, 0) + 1

        # 批量更新数据库中的访问计数
        # 使用 raw_set 支持 SQL 表达式 access_count = access_count + 1
        for rid in record_ids:
            try:
                self.storage.update(
                    database=MYSQL_DATABASE,
                    table=MYSQL_MID_TERM_TABLE,
                    raw_set="access_count = access_count + 1",
                    conditions={"id": rid},
                )
            except Exception as e:
                logger.debug(f"Failed to update access_count for {rid}: {e}")

    # ========== 摘要生成 ==========

    def _generate_summary(self, conversation: str) -> tuple:
        """生成摘要和关键词"""
        prompt = f"""对话内容：
{conversation}

提取摘要(200字内)和关键词，JSON格式返回：
{{"summary": "摘要", "keywords": "关键词1,关键词2"}}"""

        response = self.summary_llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        try:
            result = json.loads(response.content)
            return result.get("summary", ""), result.get("keywords", "")
        except json.JSONDecodeError:
            return response.content[:500] if response.content else "", ""

    def _save_to_mysql(
        self,
        messages: List[Dict[str, str]],
        summary: str,
        keywords: str,
        raw_messages: Optional[List[Dict]] = None,
    ):
        """保存到 MySQL

        Args:
            messages: 用于生成摘要的消息（用于计数）
            summary: 生成的摘要
            keywords: 提取的关键词
            raw_messages: 完整的原始消息列表，如果不传则使用 messages
        """
        from datetime import datetime

        # 获取时间戳
        now = time.time()
        start_time = now
        end_time = now

        # raw_messages 保存完整对话，如果没有则用 messages
        raw_to_save = raw_messages if raw_messages is not None else messages
        raw = json.dumps(raw_to_save, ensure_ascii=False)

        self.storage.insert(
            database=MYSQL_DATABASE,
            table=MYSQL_MID_TERM_TABLE,
            rows=[
                {
                    "bot_id": self.bot_id,
                    "user_id": self.user_id,
                    "summary": summary,
                    "keywords": keywords,
                    "raw_messages": raw,
                    "message_count": len(messages),
                    "start_time": datetime.fromtimestamp(start_time),
                    "end_time": datetime.fromtimestamp(end_time),
                    "access_count": 0,
                }
            ],
        )
        logger.debug(f"MySQL insert: {MYSQL_MID_TERM_TABLE}")

    # ========== 长期记忆存储 ==========

    def store_long_term(
        self,
        content: str,
        memory_type: str,
        importance: int = 5,
        tags: Optional[List[str]] = None,
    ) -> Optional[str]:
        """存储长期记忆"""
        content = content.strip()
        if not content:
            return None

        # 规范化内容，使其与检索时的语义空间对齐
        normalized_content = self.query_rewriter.normalize_for_storage(content)

        # 对规范化后的内容进行向量化
        vector = self.embed_func(normalized_content)
        if not vector:
            logger.error("Failed to vectorize content")
            return None

        memory_id = f"mem_{uuid.uuid4().hex[:16]}"
        now = int(time.time())

        try:
            # 构造自定义metadata字典
            custom_metadata = {
                "importance": max(1, min(10, importance)),
                "tags": tags or [],
                "source": "agent",
                "normalized_content": normalized_content,
            }
            # 转为JSON字符串
            custom_metadata_str = json.dumps(custom_metadata, ensure_ascii=False)

            inserted = self.storage.vector_insert(
                collection=MILVUS_COLLECTION,
                partition=self._partition,
                vectors=[
                    {
                        "id": memory_id,
                        "vector": vector,  # 向量基于规范化内容生成
                        "metadata": {
                            "bot_id": self.bot_id,
                            "user_id": self.user_id,
                            "memory_type": memory_type,
                            "created_at": now,
                            "content": content,  # 保留原始内容用于展示
                            "metadata": custom_metadata_str,  # 自定义拓展字段放在metadata的metadata键下
                        },
                    }
                ],
            )
            logger.debug(f"Vector insert: {inserted} rows, partition={self._partition}")
            return memory_id if inserted > 0 else None
        except Exception as e:
            logger.error(f"Failed to store long-term memory: {e}")
            return None

    # ========== 长期记忆检索 ==========

    def search_long_term(
        self,
        query: str,
        memory_type: str = "all",
        limit: int = 5,
        min_score: float = 0.1,  # 从 0.3 降低到 0.1
        min_importance: int = 1,
    ) -> List[Dict]:
        """
        搜索长期记忆

        流程：精简改写 → 向量召回 → 精排
        """
        if not query or not query.strip():
            return []

        # 1. 精简改写（长期模式：保持语义纯净）
        rewritten_query = self.query_rewriter.rewrite_for_long_term(query)

        # 2. 向量化
        vector = self.embed_func(rewritten_query)
        if not vector:
            logger.warning("Failed to embed query")
            return []

        # 3. 向量召回
        try:
            raw_results = self._vector_recall(
                query_vector=vector,
                top_k=limit * 3,
                memory_type=memory_type,
            )
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

        logger.debug(f"Vector recall: {len(raw_results)} raw results")

        if not raw_results:
            return []

        # 4. 转换为 RankItem
        rank_items = self._to_long_term_rank_items(raw_results)

        # 5. 精排（传入 query 用于上下文匹配增强）
        ranked = self.ranker.rank_long_term(
            rewritten_query, rank_items, limit=limit * 2
        )

        logger.debug(f"After ranking: {len(ranked)} items")

        # 5.1 去重（相同内容保留分数最高的）
        ranked = self._dedupe_long_term(ranked)
        logger.debug(f"After dedup: {len(ranked)} items")

        # 6. 过滤并返回
        results = []
        for item in ranked:
            if item.final_score < min_score or item.importance < min_importance:
                continue

            results.append(
                {
                    "id": item.id,
                    "content": item.content,
                    "memory_type": item.metadata.get("memory_type", ""),
                    "created_at": item.created_at,
                    "importance": item.importance,
                    "tags": item.metadata.get("tags", []),
                    "score": round(item.final_score, 3),
                }
            )

            if len(results) >= limit:
                break

        logger.debug(f"Long-term search: {len(results)} final results")
        return results

    def _vector_recall(
        self,
        query_vector: List[float],
        top_k: int,
        memory_type: str,
    ) -> List[Dict]:
        """向量召回"""
        logger.debug(
            f"Vector search: partition={self._partition}, top_k={top_k}, type={memory_type}"
        )

        # 不指定 output_fields，让 Milvus 返回所有字段
        # id 和 score 是搜索结果的内置字段，不需要在 output_fields 中指定
        results = self.storage.vector_search(
            collection=MILVUS_COLLECTION,
            partition=self._partition,
            query_vector=query_vector,
            top_k=top_k * 3,  # 获取更多结果用于后续过滤
        )

        logger.debug(f"Vector search returned {len(results)} results")

        # 在 Python 层面过滤
        filtered = []
        for r in results:
            metadata = r.get("metadata", {})
            r_user_id = metadata.get("user_id", "")
            r_memory_type = metadata.get("memory_type", "")

            # 检查 user_id 是否匹配
            if r_user_id != self.user_id:
                continue

            # 检查 memory_type 是否匹配（如果指定了的话）
            if memory_type != "all" and r_memory_type != memory_type:
                continue

            filtered.append(r)

            if len(filtered) >= top_k:
                break

        logger.debug(f"After user_id filter: {len(filtered)} results")
        return filtered

    def _to_long_term_rank_items(self, results: List[Dict]) -> List[RankItem]:
        """
        转换向量检索结果为 RankItem

        新的存储结构：
        - vectors[i] = {"id": str, "vector": List[float], "metadata": {
            "bot_id": str, "user_id": str, "memory_type": str,
            "created_at": int, "content": str, "metadata": str(JSON)
        }
        }
        """
        items = []
        for r in results:
            # r 是从向量搜索返回的结果，包含 metadata 字段
            metadata_dict = r.get("metadata", {})
            content = metadata_dict.get("content", "")
            created_at = metadata_dict.get("created_at", 0)
            memory_type = metadata_dict.get("memory_type", "")

            # 解析嵌套的 metadata JSON 字符串
            nested_metadata_str = metadata_dict.get("metadata", "{}")
            try:
                nested_metadata = (
                    json.loads(nested_metadata_str) if nested_metadata_str else {}
                )
            except json.JSONDecodeError:
                nested_metadata = {}

            source = nested_metadata.get("source", "agent")

            # 根据来源确定 raw_content
            if source == "mid_term":
                # 提升的记忆：从嵌套 metadata 获取完整原始对话
                raw_content = nested_metadata.get("raw_messages", content)
            else:
                # 原生记忆：content 就是完整内容
                raw_content = content

            items.append(
                RankItem(
                    id=r.get("id", ""),
                    source="long_term",
                    content=content,
                    raw_content=raw_content,
                    vector_score=r.get("score", 0),
                    created_at=created_at,
                    importance=nested_metadata.get("importance", 5),
                    metadata={
                        "memory_type": memory_type,
                        "tags": nested_metadata.get("tags", []),
                    },
                )
            )
        return items

    def _dedupe_long_term(self, items: List[RankItem]) -> List[RankItem]:
        """
        长期记忆去重：相同内容保留分数最高的

        基于 content 的前100字符去重
        """
        if not items:
            return []

        # 按分数降序排列
        items = sorted(items, key=lambda x: x.final_score, reverse=True)

        seen = set()
        result = []
        for item in items:
            content_key = item.content[:100] if item.content else ""
            if content_key and content_key not in seen:
                seen.add(content_key)
                result.append(item)

        return result

    # ========== 记忆提升 ==========

    def promote_high_frequency(self, threshold: int = PROMOTION_THRESHOLD):
        """提升高频中期记忆"""
        high_freq_ids = [
            rid for rid, cnt in self._access_counter.items() if cnt >= threshold
        ]

        for record_id in high_freq_ids:
            try:
                rows = self.storage.select(
                    database=MYSQL_DATABASE,
                    table=MYSQL_MID_TERM_TABLE,
                    conditions={"id": record_id},
                    limit=1,
                )
                if rows:
                    row = rows[0]
                    summary = row.get("summary", "")
                    raw_messages = row.get("raw_messages", "")  # 完整原始对话
                    if summary:
                        self._promote_to_long_term(record_id, summary, raw_messages)
            except Exception as e:
                logger.error(f"Failed to promote memory {record_id}: {e}")

        self._access_counter.clear()

    def _promote_to_long_term(
        self, mid_term_id: int, summary: str, raw_messages: str = ""
    ):
        """
        提升到长期记忆

        存储策略：
        - 向量化：基于摘要改写（摘要是语义浓缩，更适合向量检索）
        - content：存储摘要（用于展示和精排）
        - metadata：存储原始对话内容（便于回溯细节）
        """
        # 规范化摘要，使其与检索时的语义空间对齐
        normalized_content = self.query_rewriter.normalize_for_storage(summary)

        # 对规范化后的摘要进行向量化
        vector = self.embed_func(normalized_content)
        if not vector:
            return

        memory_id = f"promoted_{mid_term_id}_{uuid.uuid4().hex[:8]}"
        now = int(time.time())

        try:
            # 构造自定义metadata字典
            custom_metadata = {
                "source": "mid_term",
                "source_id": mid_term_id,
                "normalized_content": normalized_content,
                "raw_messages": raw_messages,
            }
            # 转为JSON字符串
            custom_metadata_str = json.dumps(custom_metadata, ensure_ascii=False)

            inserted = self.storage.vector_insert(
                collection=MILVUS_COLLECTION,
                partition=self._partition,
                vectors=[
                    {
                        "id": memory_id,
                        "vector": vector,  # 向量基于规范化摘要生成
                        "metadata": {
                            "bot_id": self.bot_id,
                            "user_id": self.user_id,
                            "memory_type": "promoted",
                            "created_at": now,
                            "content": summary,  # 摘要用于展示和精排
                            "metadata": custom_metadata_str,  # 自定义拓展字段放在metadata的metadata键下
                        },
                    }
                ],
            )

            # 提升成功后删除中期记忆记录
            self._delete_mid_term_record(mid_term_id)
            logger.info(
                f"Memory {mid_term_id} promoted to long-term and deleted from mid-term"
            )
        except Exception as e:
            logger.error(f"Failed to promote memory: {e}")

    def _delete_mid_term_record(self, record_id: int):
        """删除中期记忆记录"""
        try:
            self.storage.delete(
                database=MYSQL_DATABASE,
                table=MYSQL_MID_TERM_TABLE,
                conditions={"id": record_id},
            )
            logger.info(f"Deleted mid-term memory record {record_id}")
        except Exception as e:
            logger.error(f"Failed to delete mid-term record {record_id}: {e}")

    # ========== 资源管理 ==========

    def close(self):
        """关闭资源"""
        # 关闭前执行记忆提升
        self.promote_high_frequency()

        if self._summary_llm:
            self._summary_llm.close()
            self._summary_llm = None
        if self._query_rewriter:
            self._query_rewriter.close()
            self._query_rewriter = None
