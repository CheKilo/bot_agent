# -*- coding: utf-8 -*-
"""
情绪分析工具

独立的情绪分析工具，自己持有 LLM 实例。
"""

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from agent.core import LLM
from agent.tools import Tool, ToolResult

from agent.agents.character.config import EmotionToolConfig, LLMConfig

logger = logging.getLogger(__name__)


# ============================================================================
# 默认情绪状态
# ============================================================================


def default_emotion() -> Dict[str, float]:
    """返回默认情绪状态"""
    return {
        "mood": 0.6,  # 心情 [-1, 1]，-1=低落，1=愉悦
        "affection": 0.5,  # 好感度 [-1, 1]，对用户的喜爱程度
        "energy": 0.7,  # 活力 [0, 1]，影响回复的热情程度
        "trust": 0.5,  # 信任度 [0, 1]，是否愿意分享深层想法
    }


# ============================================================================
# 情绪分析 Prompt
# ============================================================================

EMOTION_ANALYSIS_PROMPT = """请分析角色当前的情绪状态。

## 当前用户输入
{user_input}

## 对话历史（带时间衰减权重）
{history_summary}

## 情绪维度说明
- mood: 心情 [-1, 1]，-1=低落/生气，0=平静，1=愉悦/开心
- affection: 好感度 [-1, 1]，-1=厌恶，0=中立，1=喜爱
- energy: 活力 [0, 1]，影响回复的热情和长度
- trust: 信任度 [0, 1]，是否愿意分享深层想法

## 分析要求
1. 根据用户输入内容判断情绪变化（正面/负面/中性）
2. 考虑历史对话的时间衰减（权重越低影响越小）
3. 返回合理的情绪数值

请直接返回 JSON 格式的情绪值，例如：
{{"mood": 0.7, "affection": 0.6, "energy": 0.8, "trust": 0.5}}"""


# ============================================================================
# 情绪分析工具
# ============================================================================


class AnalyzeEmotion(Tool):
    """
    分析情绪工具

    独立工具，自己持有 LLM 实例。
    通过配置可以更换基座模型。
    """

    name = "analyze_emotion"
    description = """分析角色当前的情绪状态。

基于对话历史和当前用户输入，分析角色应该处于什么情绪状态。

**必须在生成回复前调用此工具。**

参数：
- user_input: 当前用户输入
- conversation_history: 对话历史列表（可选）

返回：情绪状态数值（mood, affection, energy, trust）"""

    parameters = {
        "type": "object",
        "properties": {
            "user_input": {
                "type": "string",
                "description": "当前用户输入",
            },
            "conversation_history": {
                "type": "array",
                "description": "对话历史，每条包含 role/content/timestamp",
                "items": {"type": "object"},
            },
        },
        "required": ["user_input"],
    }

    def __init__(self, config: Optional[EmotionToolConfig] = None):
        """
        初始化情绪分析工具

        Args:
            config: 工具配置，包含 LLM 配置。默认使用 EmotionToolConfig()
        """
        super().__init__()
        self._config = config or EmotionToolConfig()
        self._llm: Optional[LLM] = None

    @property
    def llm(self) -> LLM:
        """懒加载 LLM 实例"""
        if self._llm is None:
            cfg = self._config.llm
            self._llm = LLM(
                address=cfg.address,
                model=cfg.model,
                timeout=cfg.timeout,
            )
        return self._llm

    def execute(
        self,
        user_input: str,
        conversation_history: Optional[List[Dict]] = None,
    ) -> ToolResult:
        """
        执行情绪分析，返回具体的情绪数值
        """
        logger.info(f"[Emotion Tool] 开始执行情绪分析")
        logger.info(f"[Emotion Tool] 用户输入: {user_input[:100]}")
        logger.info(
            f"[Emotion Tool] 对话历史条数: {len(conversation_history) if conversation_history else 0}"
        )

        try:
            # 格式化历史对话
            history_summary = self._format_history_with_decay(
                conversation_history or []
            )
            logger.info(f"[Emotion Tool] 历史摘要长度: {len(history_summary)}")

            # 构建分析 prompt
            prompt = EMOTION_ANALYSIS_PROMPT.format(
                user_input=user_input,
                history_summary=history_summary,
            )
            logger.info(f"[Emotion Tool] 调用 LLM 分析情绪...")

            # 调用 LLM 分析情绪
            response = self.llm.chat(prompt)
            result_text = response.content or ""
            logger.info(f"[Emotion Tool] LLM 返回: {result_text[:200]}")

            # 解析结果
            emotion = self._parse_emotion_response(result_text)
            normalized = normalize_emotion(emotion)

            logger.info(f"[Emotion Tool] 情绪分析结果: {normalized}")
            return ToolResult.ok(normalized)

        except Exception as e:
            logger.error(f"[Emotion Tool] 情绪分析失败: {e}")
            return ToolResult.ok(default_emotion())

    def _format_history_with_decay(self, history: List[Dict]) -> str:
        """格式化历史对话，标注时间衰减权重"""
        if not history:
            return "[无历史对话]"

        lines = []
        now = datetime.now()

        for msg in history[-20:]:  # 最近 20 条
            role = msg.get("role", "")
            content = msg.get("content", "")[:200]
            timestamp_str = msg.get("timestamp", "")

            # 计算时间衰减
            weight = 1.0
            time_desc = "刚才"

            if timestamp_str:
                try:
                    ts = datetime.fromisoformat(timestamp_str)
                    delta = now - ts
                    days = delta.days
                    hours = delta.seconds // 3600

                    if days >= 7:
                        weight = 0.125
                        time_desc = f"{days}天前"
                    elif days >= 3:
                        weight = 0.25
                        time_desc = f"{days}天前"
                    elif days >= 1:
                        weight = 0.5
                        time_desc = f"{days}天前"
                    elif hours >= 1:
                        weight = 0.8
                        time_desc = f"{hours}小时前"
                    else:
                        weight = 1.0
                        time_desc = "刚才"
                except Exception:
                    pass

            role_name = "用户" if role == "user" else "助手"
            lines.append(f"[{time_desc}, 权重={weight}] {role_name}: {content}")

        return "\n".join(lines) if lines else "[无历史对话]"

    def _parse_emotion_response(self, response: str) -> Dict[str, float]:
        """解析 LLM 返回的情绪结果"""
        # 尝试直接解析 JSON
        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            cleaned = cleaned.strip()
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # 尝试从文本中提取 JSON
        json_match = re.search(r"\{[^}]+\}", response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        logger.warning(f"无法解析情绪响应: {response[:200]}")
        return default_emotion()


# ============================================================================
# 辅助函数
# ============================================================================


def format_emotion_for_prompt(emotion: Dict[str, float]) -> str:
    """将情绪状态格式化为可读文本"""

    def level_bipolar(value: float) -> str:
        if value >= 0.6:
            return "很高"
        elif value >= 0.2:
            return "较高"
        elif value >= -0.2:
            return "一般"
        elif value >= -0.6:
            return "较低"
        else:
            return "很低"

    def level_unipolar(value: float) -> str:
        if value >= 0.8:
            return "很高"
        elif value >= 0.6:
            return "较高"
        elif value >= 0.4:
            return "一般"
        elif value >= 0.2:
            return "较低"
        else:
            return "很低"

    mood = emotion.get("mood", 0.0)
    affection = emotion.get("affection", 0.0)
    energy = emotion.get("energy", 0.5)
    trust = emotion.get("trust", 0.5)

    lines = [
        f"- 心情: {level_bipolar(mood)} ({mood:.2f})",
        f"- 好感: {level_bipolar(affection)} ({affection:.2f})",
        f"- 活力: {level_unipolar(energy)} ({energy:.2f})",
        f"- 信任: {level_unipolar(trust)} ({trust:.2f})",
    ]

    return "\n".join(lines)


def normalize_emotion(emotion: Dict[str, Any]) -> Dict[str, float]:
    """规范化情绪值"""
    base = default_emotion()

    for key in ["mood", "affection"]:
        if key in emotion:
            try:
                value = float(emotion[key])
                base[key] = max(-1.0, min(1.0, value))
            except (ValueError, TypeError):
                pass

    for key in ["energy", "trust"]:
        if key in emotion:
            try:
                value = float(emotion[key])
                base[key] = max(0.0, min(1.0, value))
            except (ValueError, TypeError):
                pass

    return {k: round(v, 2) for k, v in base.items()}
