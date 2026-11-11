"""规则引擎：文本 → 题材打分。

设计原则：**可解释**。每一次打分都能回答三个问题：
  1. 命中了哪些关键词？（hits）
  2. 每个关键词贡献了多少分？（weight × specificity）
  3. 为什么要拒绝分类？（低于阈值 / 置信度不足）

没有黑箱，没有模型权重 —— 全部是关键词级可读规则 + 一组可调的注意力权重。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .taxonomy import Taxonomy, Theme

__all__ = ["ScoredTheme", "RuleEngine"]

# 多关键词命中时的确认加成：命中 1 个 ×1.00，2 个 ×1.15，3 个 ×1.30 …
CONFIRM_STEP = 0.15
# 同一赛道出现 ≥2 个题材命中时的赛道聚拢加成
TRACK_BOOST = 1.25
# 文本长度超过该值后开始衰减（长文本容易堆关键词，导致弱信号被放大）
LENGTH_SOFT_LIMIT = 80
LENGTH_DECAY = 0.01

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """去掉全部空白。

    中英文混排里空格是排版产物而非语义（"AI 眼镜" 与 "AI眼镜" 是同一个词），
    不归一化就会漏命中；中文本身不依赖空格分隔，去掉它不引入歧义。
    """
    return _WS.sub("", text)


@dataclass(frozen=True)
class ScoredTheme:
    """一个题材的打分结果。"""

    theme_id: str
    name: str
    score: float
    hits: Tuple[str, ...]
    path: Tuple[str, str, str]
    logic: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "theme_id": self.theme_id,
            "name": self.name,
            "score": round(self.score, 4),
            "hits": list(self.hits),
            "path": list(self.path),
        }


@dataclass
class RuleEngine:
    """关键词规则引擎。

    weights: 关键词 → 注意力权重（默认全 1.0，由 AttentionEngine 迭代调优）
    """

    taxonomy: Taxonomy
    weights: Dict[str, float] = field(default_factory=dict)
    min_score: float = 1.0
    min_confidence: float = 0.45

    def __post_init__(self) -> None:
        if not self.weights:
            self.weights = {kw: 1.0 for kw in self.taxonomy.all_keywords()}

    # ── 打分 ────────────────────────────────────────────

    def hits_of(self, text: str, theme: Theme) -> Tuple[str, ...]:
        """该主题在文本里命中的关键词（按贡献从大到小）。"""
        ntext = normalize(text)
        scored = []
        for kw in theme.keywords:
            if kw and normalize(kw) in ntext:
                scored.append((self.weight(kw) * self.taxonomy.specificity(kw), kw))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return tuple(kw for _, kw in scored)

    def weight(self, keyword: str) -> float:
        return float(self.weights.get(keyword, 1.0))

    def score_text(self, text: str) -> List[ScoredTheme]:
        """给全部题材打分，降序返回。"""
        raw: Dict[str, float] = {}
        hits: Dict[str, List[str]] = {}
        track_hits: Dict[str, set] = {}

        for theme in self.taxonomy:
            kw_hits = self.hits_of(text, theme)
            if not kw_hits:
                continue
            total = sum(
                self.weight(kw) * self.taxonomy.specificity(kw) for kw in kw_hits
            )
            raw[theme.id] = total
            hits[theme.id] = list(kw_hits)
            track_hits.setdefault(theme.track_id, set()).add(theme.id)

        # 赛道聚拢：同一赛道里 ≥2 个题材都命中 → 整条赛道加成
        boosted_tracks = {tid for tid, ids in track_hits.items() if len(ids) >= 2}

        out: List[ScoredTheme] = []
        for theme in self.taxonomy:
            if theme.id not in raw:
                continue
            base = raw[theme.id]
            n = len(hits[theme.id])
            score = base * (1.0 + CONFIRM_STEP * (n - 1))
            if theme.track_id in boosted_tracks:
                score *= TRACK_BOOST
            if len(text) > LENGTH_SOFT_LIMIT:
                score /= 1.0 + LENGTH_DECAY * (len(text) - LENGTH_SOFT_LIMIT)
            out.append(
                ScoredTheme(
                    theme_id=theme.id,
                    name=theme.name,
                    score=round(score, 6),
                    hits=tuple(hits[theme.id]),
                    path=theme.path,
                    logic=theme.logic,
                )
            )

        out.sort(key=lambda s: (-s.score, s.name))
        return out

    # ── 分类 ────────────────────────────────────────────

    def classify(self, text: str, top_k: int = 1) -> "Classification":
        from .classify import Classification  # 避免循环导入

        ranked = self.score_text(text)
        if not ranked:
            return Classification(
                text=text,
                top=None,
                confidence=0.0,
                candidates=[],
                rejected=True,
                reason="no_keyword_hit",
            )

        top = ranked[0]
        second = ranked[1].score if len(ranked) > 1 else 0.0
        # 置信度只看前两名：第 1 名相对第 2 名的优势
        confidence = top.score / (top.score + second) if (top.score + second) > 0 else 0.0

        rejected = False
        reason = ""
        if top.score < self.min_score:
            rejected = True
            reason = "below_min_score"
        elif confidence < self.min_confidence:
            rejected = True
            reason = "low_confidence"

        return Classification(
            text=text,
            top=None if rejected else top,
            confidence=round(confidence, 4),
            candidates=ranked[:top_k],
            rejected=rejected,
            reason=reason,
        )

    def explain(self, text: str, top_k: int = 3) -> List[Dict[str, object]]:
        """人类可读的归因说明（给面试/复盘看的那种）。"""
        out = []
        for s in self.score_text(text)[:top_k]:
            contributions = []
            for kw in s.hits:
                contributions.append(
                    {
                        "keyword": kw,
                        "weight": round(self.weight(kw), 4),
                        "specificity": round(self.taxonomy.specificity(kw), 4),
                        "contribution": round(
                            self.weight(kw) * self.taxonomy.specificity(kw), 4
                        ),
                    }
                )
            out.append(
                {
                    "theme": s.as_dict(),
                    "contributions": contributions,
                }
            )
        return out
