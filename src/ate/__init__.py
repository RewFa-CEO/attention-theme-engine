"""attention-theme-engine —— 注意力机制研究框架的公开脱敏切片。

三层结构：
    1. 题材体系  三级分类：行业 / 赛道 / 核心逻辑（每个题材带一句"为什么它会动"）
    2. 规则引擎  关键词规则 + 注意力权重 → 可解释打分
    3. 注意力迭代 按分类误差回写关键词权重，并评估每轮变化

完整说明见仓库 README.md 与 docs/SLICE.md。
"""

from .attention import AttentionEngine, IterationLog, Sample
from .classify import Classification
from .rules import RuleEngine, ScoredTheme
from .taxonomy import Taxonomy, Theme

__all__ = [
    "Taxonomy",
    "Theme",
    "RuleEngine",
    "ScoredTheme",
    "Classification",
    "AttentionEngine",
    "Sample",
    "IterationLog",
    "load_golden",
]

__version__ = "0.3.0"


def load_golden(path=None):
    """读取黄金样本集（JSONL）。"""
    from .dataset import load_golden as _load

    return _load(path)
