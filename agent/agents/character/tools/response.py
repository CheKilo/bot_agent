# -*- coding: utf-8 -*-
"""
回复生成工具

独立的回复生成工具，自己持有 LLM 实例。
"""

import logging
from typing import Dict, List, Optional

from agent.core import LLM
from agent.tools import Tool, ToolResult

from agent.agents.character.config import ResponseToolConfig, LLMConfig

logger = logging.getLogger(__name__)


# ============================================================================
# 回复生成 Prompt
# ============================================================================

RESPONSE_GENERATION_PROMPT = """请基于以下信息生成角色回复。

## 用户输入
{user_input}

## 角色人设
{persona}

## 当前情绪状态
{emotion_desc}

## 相关记忆
{memory_context}

## 回复要求（重要）
1. 回复要符合角色人设和当前情绪状态
2. 回复要自然、简洁，像真人聊天一样
3. 根据情绪状态调整语气：
   - 心情好时：语气轻快、热情
   - 心情差时：语气平淡、简短
   - 好感高时：更亲近、愿意分享
   - 好感低时：保持距离、回复简短
4. **直接输出角色的回复内容，不要输出任何说明、解释或格式标记**
5. **不要使用引号、冒号等特殊字符包裹回复**
6. **不要包含"小助手:"、"小雪:"等前缀**

请直接输出角色的回复（纯文本，无其他内容）："""


# ============================================================================
# 回复生成工具
# ============================================================================


class GenerateResponse(Tool):
    """
    生成角色回复工具

    独立工具，自己持有 LLM 实例。
    通过配置可以更换基座模型。
    """

    name = "generate_response"
    description = """基于情绪、记忆和人设生成角色回复。

**必须在 analyze_emotion 之后调用此工具。**

参数：
- user_input: 用户输入
- emotion: 情绪状态（来自 analyze_emotion 的结果）
- persona: 角色人设描述
- memory_context: 记忆上下文（可选）

返回：角色的回复内容"""

    parameters = {
        "type": "object",
        "properties": {
            "user_input": {
                "type": "string",
                "description": "用户输入",
            },
            "emotion": {
                "type": "object",
                "description": "情绪状态（来自 analyze_emotion 的结果）",
                "properties": {
                    "mood": {"type": "number", "description": "心情 [-1, 1]"},
                    "affection": {"type": "number", "description": "好感度 [-1, 1]"},
                    "energy": {"type": "number", "description": "活力 [0, 1]"},
                    "trust": {"type": "number", "description": "信任度 [0, 1]"},
                },
            },
            "persona": {
                "type": "string",
                "description": "角色人设描述",
            },
            "memory_context": {
                "type": "string",
                "description": "记忆上下文",
            },
        },
        "required": ["user_input", "emotion", "persona"],
    }

    def __init__(self, config: Optional[ResponseToolConfig] = None):
        """
        初始化回复生成工具

        Args:
            config: 工具配置，包含 LLM 配置。默认使用 ResponseToolConfig()
        """
        super().__init__()
        self._config = config or ResponseToolConfig()
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
        emotion: Dict[str, float],
        persona: str,
        memory_context: str = "",
    ) -> ToolResult:
        """
        执行回复生成，返回具体的回复内容
        """
        logger.info(f"[Response Tool] 开始执行回复生成")
        logger.info(f"[Response Tool] 用户输入: {user_input[:100]}")
        logger.info(f"[Response Tool] 情绪状态: {emotion}")
        logger.info(
            f"[Response Tool] 人设: {persona[:50] if persona else '[未设置]'}..."
        )
        logger.info(f"[Response Tool] 记忆上下文长度: {len(memory_context)}")

        try:
            # 格式化情绪描述
            emotion_desc = self._format_emotion(emotion)
            logger.info(f"[Response Tool] 情绪描述:\n{emotion_desc}")

            # 构建生成 prompt
            prompt = RESPONSE_GENERATION_PROMPT.format(
                user_input=user_input,
                persona=persona or "[未设置人设]",
                emotion_desc=emotion_desc,
                memory_context=memory_context or "[无相关记忆]",
            )
            logger.info(f"[Response Tool] 调用 LLM 生成回复...")

            # 调用 LLM 生成回复
            response = self.llm.chat(prompt)
            result_text = response.content or ""
            logger.info(f"[Response Tool] LLM 返回: {result_text[:200]}")

            # 清理回复
            cleaned = self._clean_response(result_text)

            logger.info(f"[Response Tool] 生成回复: {cleaned[:100]}")
            return ToolResult.ok(cleaned)

        except Exception as e:
            logger.error(f"[Response Tool] 回复生成失败: {e}")
            return ToolResult.fail(f"回复生成失败: {str(e)}")

    def _format_emotion(self, emotion: Dict[str, float]) -> str:
        """格式化情绪为描述性文本"""
        mood = emotion.get("mood", 0.0)
        affection = emotion.get("affection", 0.0)
        energy = emotion.get("energy", 0.5)
        trust = emotion.get("trust", 0.5)

        def describe_bipolar(value: float, low: str, mid: str, high: str) -> str:
            if value >= 0.5:
                return high
            elif value >= -0.5:
                return mid
            else:
                return low

        def describe_unipolar(value: float, low: str, mid: str, high: str) -> str:
            if value >= 0.7:
                return high
            elif value >= 0.3:
                return mid
            else:
                return low

        mood_desc = describe_bipolar(mood, "心情低落", "心情平静", "心情愉悦")
        affection_desc = describe_bipolar(
            affection, "对用户有些疏远", "对用户态度中立", "对用户很有好感"
        )
        energy_desc = describe_unipolar(energy, "精力不足", "精力一般", "精力充沛")
        trust_desc = describe_unipolar(trust, "比较戒备", "信任度一般", "非常信任")

        return f"""- {mood_desc} (mood={mood:.2f})
- {affection_desc} (affection={affection:.2f})
- {energy_desc} (energy={energy:.2f})
- {trust_desc} (trust={trust:.2f})"""

    def _clean_response(self, response: str) -> str:
        """清理 LLM 返回的回复"""
        cleaned = response.strip()

        # 移除可能的引号包裹
        if (cleaned.startswith('"') and cleaned.endswith('"')) or (
            cleaned.startswith("'") and cleaned.endswith("'")
        ):
            cleaned = cleaned[1:-1]

        # 移除可能的角色名前缀（如 "小助手: "）
        if ":" in cleaned[:20]:
            parts = cleaned.split(":", 1)
            if len(parts[0]) < 10:
                cleaned = parts[1].strip()

        return cleaned
