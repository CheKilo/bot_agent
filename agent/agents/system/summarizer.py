# -*- coding: utf-8 -*-
"""
Conversation Summarizer - 对话摘要生成与存储

负责：
1. 生成对话摘要和关键词
2. 存储摘要到 MySQL
"""

import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from agent.client import StorageClient
from agent.core import LLM

logger = logging.getLogger(__name__)


class ConversationSummarizer:
    """
    对话摘要生成与存储

    使用示例：
        ```python
        summarizer = ConversationSummarizer(
            storage_client=storage_client,
            llm_address="http://xxx",
            llm_model="gpt-4",
            database="bot_db",
            table="conversation_summary",
        )

        # 生成并存储摘要
        success = summarizer.summarize_and_save(
            bot_id="test_bot",
            user_id="user123",
            messages=[...],
        )
        ```
    """

    def __init__(
        self,
        storage_client: StorageClient,
        llm_address: str,
        llm_model: str,
        database: str,
        table: str,
        llm_timeout: int = 30,
    ):
        """
        初始化

        Args:
            storage_client: 存储客户端
            llm_address: 摘要 LLM 地址
            llm_model: 摘要 LLM 模型
            database: MySQL 数据库名
            table: MySQL 表名
            llm_timeout: LLM 超时时间
        """
        self._storage = storage_client
        self._database = database
        self._table = table

        self._llm_address = llm_address
        self._llm_model = llm_model
        self._llm_timeout = llm_timeout
        self._llm: Optional[LLM] = None

    @property
    def llm(self) -> LLM:
        """LLM 实例（懒加载）"""
        if self._llm is None:
            self._llm = LLM(
                address=self._llm_address,
                model=self._llm_model,
                timeout=self._llm_timeout,
            )
        return self._llm

    def summarize_and_save(
        self,
        bot_id: str,
        user_id: str,
        messages: List[Dict],
    ) -> bool:
        """
        生成摘要并保存到 MySQL

        Args:
            bot_id: Bot ID
            user_id: 用户 ID
            messages: 对话消息列表

        Returns:
            是否成功
        """
        if not messages:
            return False

        try:
            summary, keywords = self._generate_summary(messages)
            if not summary:
                logger.warning("[Summarizer] 摘要生成失败")
                return False

            return self._save_to_mysql(
                bot_id=bot_id,
                user_id=user_id,
                summary=summary,
                keywords=keywords,
                raw_messages=messages,
            )
        except Exception as e:
            logger.error(f"[Summarizer] 摘要生成或存储失败: {e}")
            return False

    def _generate_summary(self, messages: List[Dict]) -> Tuple[str, str]:
        """
        生成摘要和关键词

        Args:
            messages: 对话消息列表

        Returns:
            (摘要, 关键词)
        """
        conversation = "\n".join(
            [f"[{m.get('role', 'unknown')}]: {m.get('content', '')}" for m in messages]
        )

        prompt = f"""对话内容：
{conversation}

提取摘要(200字内)和关键词，JSON格式返回：
{{"summary": "摘要", "keywords": "关键词1,关键词2"}}"""

        try:
            response = self.llm.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            result = json.loads(response.content)
            return result.get("summary", ""), result.get("keywords", "")
        except json.JSONDecodeError:
            return response.content[:500] if response.content else "", ""
        except Exception as e:
            logger.error(f"[Summarizer] 生成摘要失败: {e}")
            return "", ""

    def _save_to_mysql(
        self,
        bot_id: str,
        user_id: str,
        summary: str,
        keywords: str,
        raw_messages: List[Dict],
    ) -> bool:
        """
        保存摘要到 MySQL

        Args:
            bot_id: Bot ID
            user_id: 用户 ID
            summary: 摘要内容
            keywords: 关键词
            raw_messages: 原始消息列表

        Returns:
            是否成功
        """
        if not summary:
            return False

        try:
            now = time.time()
            raw = json.dumps(raw_messages, ensure_ascii=False)

            self._storage.insert(
                database=self._database,
                table=self._table,
                rows=[
                    {
                        "bot_id": bot_id,
                        "user_id": user_id,
                        "summary": summary,
                        "keywords": keywords,
                        "raw_messages": raw,
                        "message_count": len(raw_messages),
                        "start_time": datetime.fromtimestamp(now),
                        "end_time": datetime.fromtimestamp(now),
                        "access_count": 0,
                    }
                ],
            )
            logger.info(f"[Summarizer] MySQL insert: {len(raw_messages)} messages")
            return True
        except Exception as e:
            logger.error(f"[Summarizer] Failed to save summary: {e}")
            return False

    def close(self):
        """关闭资源"""
        if self._llm:
            self._llm.close()
            self._llm = None
