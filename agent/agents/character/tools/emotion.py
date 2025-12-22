# -*- coding: utf-8 -*-
"""
情绪管理工具

提供情绪状态更新工具，Agent 可通过工具调用动态调整情绪。
情绪状态存储在内存 dict 中，每轮对话通过 system prompt 传递给 LLM。
"""

from typing import Dict, Optional

from agent.tools import Tool, ToolResult


# ============================================================================
# 默认情绪状态
# ============================================================================


def default_emotion() -> Dict[str, float]:
    """返回默认情绪状态"""
    return {
        "mood": 0.6,  # 心情 [0, 1]，0=低落，1=愉悦
        "affection": 0.5,  # 好感度 [0, 1]，对用户的喜爱程度
        "energy": 0.7,  # 活力 [0, 1]，影响回复的热情程度
        "trust": 0.5,  # 信任度 [0, 1]，是否愿意分享深层想法
    }


# ============================================================================
# 情绪更新工具
# ============================================================================


class UpdateEmotion(Tool):
    """
    更新情绪状态工具

    Agent 根据对话内容主动调用此工具更新情绪。
    工具持有情绪 dict 的引用，调用时直接修改。
    """

    name = "update_emotion"
    description = """根据对话内容更新当前情绪状态。
当对话中发生以下情况时应调用此工具：
- 用户表达了关心、赞美、喜爱 → 提升 mood 和 affection
- 用户表达了批评、不满、冷淡 → 降低 mood 和 affection
- 进行了深入或有趣的交流 → 提升 energy 和 trust
- 用户分享了秘密或个人信息 → 提升 trust
- 对话氛围变得无聊或尴尬 → 降低 energy

每次变化幅度建议在 -0.2 到 +0.2 之间，情绪变化应该渐进自然。"""

    parameters = {
        "type": "object",
        "properties": {
            "mood_delta": {
                "type": "number",
                "description": "心情变化量，范围 [-0.3, 0.3]，正值=开心，负值=低落",
            },
            "affection_delta": {
                "type": "number",
                "description": "好感度变化量，范围 [-0.3, 0.3]，正值=更喜欢，负值=疏远",
            },
            "energy_delta": {
                "type": "number",
                "description": "活力变化量，范围 [-0.3, 0.3]，正值=兴奋，负值=疲惫",
            },
            "trust_delta": {
                "type": "number",
                "description": "信任度变化量，范围 [-0.3, 0.3]，正值=更信任，负值=戒备",
            },
            "reason": {"type": "string", "description": "情绪变化的原因（简短说明）"},
        },
        "required": ["reason"],
    }

    def __init__(self, emotion_ref: Dict[str, float]):
        """
        初始化工具

        Args:
            emotion_ref: 情绪状态 dict 的引用，工具会直接修改此 dict
        """
        super().__init__()
        self._emotion = emotion_ref

    def execute(
        self,
        reason: str,
        mood_delta: float = 0.0,
        affection_delta: float = 0.0,
        energy_delta: float = 0.0,
        trust_delta: float = 0.0,
    ) -> ToolResult:
        """执行情绪更新"""
        changes = []

        # 更新各情绪值，确保在 [0, 1] 范围内
        deltas = {
            "mood": mood_delta,
            "affection": affection_delta,
            "energy": energy_delta,
            "trust": trust_delta,
        }

        for key, delta in deltas.items():
            if delta != 0:
                # 限制变化幅度
                delta = max(-0.3, min(0.3, delta))
                old_value = self._emotion.get(key, 0.5)
                new_value = max(0.0, min(1.0, old_value + delta))
                self._emotion[key] = round(new_value, 2)

                # 记录变化
                direction = "↑" if delta > 0 else "↓"
                changes.append(f"{key}: {old_value:.2f} {direction} {new_value:.2f}")

        if changes:
            return ToolResult.ok(f"情绪已更新 ({reason}): {', '.join(changes)}")
        else:
            return ToolResult.ok(f"情绪保持不变 ({reason})")


# ============================================================================
# 情绪格式化（供 prompt 使用）
# ============================================================================


def format_emotion_for_prompt(emotion: Dict[str, float]) -> str:
    """
    将情绪状态格式化为可读文本（用于 system prompt）

    Args:
        emotion: 情绪状态 dict

    Returns:
        格式化的情绪描述
    """

    def level(value: float) -> str:
        """将数值转换为描述性词语"""
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

    mood = emotion.get("mood", 0.5)
    affection = emotion.get("affection", 0.5)
    energy = emotion.get("energy", 0.5)
    trust = emotion.get("trust", 0.5)

    lines = [
        f"- 心情: {level(mood)} ({mood:.2f}) — 影响说话语气和态度",
        f"- 好感: {level(affection)} ({affection:.2f}) — 影响对用户的亲近程度",
        f"- 活力: {level(energy)} ({energy:.2f}) — 影响回复的热情和字数",
        f"- 信任: {level(trust)} ({trust:.2f}) — 影响是否愿意分享心里话",
    ]

    return "\n".join(lines)
