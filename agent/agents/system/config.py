# -*- coding: utf-8 -*-
"""
System Agent 模块配置

所有配置集中管理，包括 LLM 配置和对话管理配置。
"""

from dataclasses import dataclass, field


# ========== LLM 默认配置 ==========
DEFAULT_LLM_ADDRESS = "localhost:50051"
DEFAULT_LLM_TIMEOUT = 30.0
DEFAULT_LLM_MODEL = "gpt-5"


# ========== 对话管理配置 ==========
DEFAULT_MESSAGE_WINDOW = 20  # 对话窗口大小（轮数）
DEFAULT_MAX_ITERATIONS = 10  # ReAct 最大迭代次数


# ========== 存储配置 ==========
MYSQL_DATABASE = "bot_memory"
MYSQL_MID_TERM_TABLE = "mid_term_memory"


# ========== 配置类 ==========


@dataclass
class LLMConfig:
    """LLM 配置"""

    address: str = DEFAULT_LLM_ADDRESS
    model: str = DEFAULT_LLM_MODEL
    timeout: float = DEFAULT_LLM_TIMEOUT


@dataclass
class StorageConfig:
    """存储配置"""

    mysql_database: str = MYSQL_DATABASE
    mysql_table: str = MYSQL_MID_TERM_TABLE


@dataclass
class ConversationConfig:
    """对话管理配置"""

    # 对话窗口大小（按轮数计，一问一答为一轮）
    message_window: int = DEFAULT_MESSAGE_WINDOW

    # 是否自动触发摘要（窗口满时）
    auto_summary: bool = True


@dataclass
class SystemConfig:
    """System Agent 配置"""

    # Agent 自身的 LLM（用于 ReAct 循环）
    llm: LLMConfig = field(default_factory=LLMConfig)

    # 摘要生成 LLM（可使用更小的模型）
    summary_llm: LLMConfig = field(
        default_factory=lambda: LLMConfig(model="gpt-4o-mini")
    )

    # 对话管理配置
    conversation: ConversationConfig = field(default_factory=ConversationConfig)

    # 存储配置
    storage: StorageConfig = field(default_factory=StorageConfig)

    # ReAct 最大迭代次数
    max_iterations: int = DEFAULT_MAX_ITERATIONS
