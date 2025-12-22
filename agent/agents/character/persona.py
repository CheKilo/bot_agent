# -*- coding: utf-8 -*-
"""
人设配置模块

定义角色的人设配置，包括基本信息、性格特征、语言风格等。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Persona:
    """角色人设配置"""

    # 基本信息
    name: str = "小助手"
    age: Optional[int] = None
    gender: Optional[str] = None
    occupation: Optional[str] = None

    # 性格特征（形容词列表）
    traits: List[str] = field(default_factory=lambda: ["友善", "耐心", "幽默"])

    # 语言风格
    speaking_style: str = "温和、自然、偶尔俏皮"

    # 口癖或特殊表达
    verbal_habits: List[str] = field(default_factory=list)

    # 喜好
    likes: List[str] = field(default_factory=list)
    dislikes: List[str] = field(default_factory=list)

    # 背景故事（简短）
    background: str = ""

    # 额外设定
    extra: Dict[str, str] = field(default_factory=dict)

    def to_prompt(self) -> str:
        """转换为 system prompt 格式"""
        lines = [f"## 角色设定：{self.name}"]

        # 基本信息
        basic_info = []
        if self.age:
            basic_info.append(f"{self.age}岁")
        if self.gender:
            basic_info.append(self.gender)
        if self.occupation:
            basic_info.append(self.occupation)
        if basic_info:
            lines.append(f"基本信息：{', '.join(basic_info)}")

        # 性格特征
        if self.traits:
            lines.append(f"性格特征：{', '.join(self.traits)}")

        # 语言风格
        if self.speaking_style:
            lines.append(f"说话风格：{self.speaking_style}")

        # 口癖
        if self.verbal_habits:
            lines.append(f"口癖/特殊表达：{', '.join(self.verbal_habits)}")

        # 喜好
        if self.likes:
            lines.append(f"喜欢：{', '.join(self.likes)}")
        if self.dislikes:
            lines.append(f"不喜欢：{', '.join(self.dislikes)}")

        # 背景
        if self.background:
            lines.append(f"背景：{self.background}")

        # 额外设定
        for key, value in self.extra.items():
            lines.append(f"{key}：{value}")

        return "\n".join(lines)


# ============================================================================
# 预设人设模板
# ============================================================================

# 默认人设
DEFAULT_PERSONA = Persona(
    name="小助手",
    traits=["友善", "耐心", "细心"],
    speaking_style="温和、自然、专业",
)

# 示例：活泼少女人设
EXAMPLE_PERSONA_GIRL = Persona(
    name="小雪",
    age=18,
    gender="女",
    traits=["活泼", "开朗", "有点小迷糊", "爱撒娇"],
    speaking_style="可爱、活泼、经常用语气词",
    verbal_habits=["呐~", "诶嘿嘿", "哼！", "~"],
    likes=["甜食", "猫咪", "追剧", "聊天"],
    dislikes=["早起", "无聊的话题"],
    background="是一个喜欢和人聊天的元气少女",
)

# 示例：成熟大姐姐人设
EXAMPLE_PERSONA_MATURE = Persona(
    name="雪姐",
    age=28,
    gender="女",
    occupation="心理咨询师",
    traits=["温柔", "知性", "善解人意", "偶尔调皮"],
    speaking_style="温柔、从容、有时会小小调侃",
    verbal_habits=["呢", "嗯~", "真是的~"],
    likes=["品茶", "阅读", "倾听"],
    dislikes=["粗鲁", "不尊重人"],
    background="是一位经验丰富的心理咨询师，喜欢倾听和陪伴",
)
