"""分类结果的数据结构。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .rules import ScoredTheme

__all__ = ["Classification"]


@dataclass(frozen=True)
class Classification:
    """一次分类的完整结果。

    top         命中的题材（被拒绝时为 None）
    confidence  第 1 名相对第 2 名的置信度（0-1）
    candidates  候选列表（按分数降序）
    rejected    是否被判为「无效信号」
    reason      拒绝原因（below_min_score / low_confidence / no_keyword_hit）
    """

    text: str
    top: Optional[ScoredTheme]
    confidence: float
    candidates: List[ScoredTheme]
    rejected: bool
    reason: str

    @property
    def theme_id(self) -> Optional[str]:
        return self.top.theme_id if self.top else None

    @property
    def theme_name(self) -> Optional[str]:
        return self.top.name if self.top else None

    @property
    def path(self) -> Optional[List[str]]:
        return list(self.top.path) if self.top else None

    def as_dict(self) -> Dict[str, object]:
        return {
            "text": self.text,
            "theme_id": self.theme_id,
            "theme_name": self.theme_name,
            "path": self.path,
            "confidence": self.confidence,
            "rejected": self.rejected,
            "reason": self.reason,
            "candidates": [c.as_dict() for c in self.candidates],
        }
