"""注意力权重迭代。

思想：分类错了不该只记一笔账，而要把**注意力挪到正确的词上**。

一轮迭代做三件事：
  1. 命中错误主题的关键词 → 降权（它把人带偏了）
  2. 命中正确主题的关键词 → 升权（它才是该被注意的）
  3. 分类正确的样本 → 把权重往基准拉回一点（防止越训越飘）

约束：
  - 权重始终被夹在 [clip_lo, clip_hi] 内，不会某天突然跑飞
  - 每一轮的准确率都记录下来，可以看出「还学不学得动」
  - 正确主题在文本里一个关键词都没命中的样本属于**覆盖问题**而不是权重问题 ——
    这类样本单独计数，不会假装能被权重救回来
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from .rules import RuleEngine
from .taxonomy import Taxonomy

__all__ = ["Sample", "IterationLog", "AttentionEngine"]


@dataclass(frozen=True)
class Sample:
    """一条标注样本。label 为 None 表示「无效信号」（应当被拒绝）。"""

    text: str
    label: Optional[str] = None
    split: str = "train"


@dataclass
class IterationLog:
    iteration: int
    accuracy: float
    rejected_rate: float
    corrected: int
    unlearnable: int
    changed_keywords: int


@dataclass
class AttentionEngine:
    """在规则引擎之上迭代调优关键词权重。"""

    taxonomy: Taxonomy
    learning_rate: float = 0.15
    clip_lo: float = 0.2
    clip_hi: float = 3.0
    reinforce: float = 0.25
    base_weights: Dict[str, float] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)
    log: List[IterationLog] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.base_weights:
            self.base_weights = {kw: 1.0 for kw in self.taxonomy.all_keywords()}
        if not self.weights:
            self.weights = dict(self.base_weights)

    # ── 内部 ────────────────────────────────────────────

    def _engine(self) -> RuleEngine:
        return RuleEngine(taxonomy=self.taxonomy, weights=self.weights)

    def _clip(self, w: float) -> float:
        return max(self.clip_lo, min(self.clip_hi, w))

    # ── 评估 ────────────────────────────────────────────

    def evaluate(self, samples: Sequence[Sample]) -> Dict[str, object]:
        """在给定样本上跑一遍，统计准确率 / 拒绝率 / 覆盖缺口。"""
        engine = self._engine()
        labelled = [s for s in samples if s.label]
        negatives = [s for s in samples if not s.label]

        correct = 0
        rejected_labelled = 0
        unlearnable = 0
        mistakes: List[Tuple[Sample, Optional[str]]] = []

        for s in labelled:
            r = engine.classify(s.text)
            if r.rejected:
                rejected_labelled += 1
                mistakes.append((s, None))
                # 正确主题一个关键词都没命中 → 覆盖问题，权重救不了
                theme = self.taxonomy.by_id(s.label)
                if not engine.hits_of(s.text, theme):
                    unlearnable += 1
                continue
            if r.theme_id == s.label:
                correct += 1
            else:
                mistakes.append((s, r.theme_id))
                # 正确主题的关键词一个都没命中 → 覆盖问题，权重救不了
                theme = self.taxonomy.by_id(s.label)
                if not engine.hits_of(s.text, theme):
                    unlearnable += 1

        rejected_negatives = 0
        for s in negatives:
            if engine.classify(s.text).rejected:
                rejected_negatives += 1

        n_lab = len(labelled)
        n_neg = len(negatives)
        return {
            "samples": len(samples),
            "labelled": n_lab,
            "negatives": n_neg,
            "correct": correct,
            "accuracy": round(correct / n_lab, 4) if n_lab else 0.0,
            "rejected_labelled": rejected_labelled,
            "rejected_negatives": rejected_negatives,
            "reject_rate": round(rejected_negatives / n_neg, 4) if n_neg else 0.0,
            "unlearnable": unlearnable,
            "mistakes": mistakes,
        }

    # ── 训练 ────────────────────────────────────────────

    def fit(self, samples: Sequence[Sample], iterations: int = 5) -> List[IterationLog]:
        """跑若干轮权重迭代，每轮记录准确率变化。"""
        train = [s for s in samples if s.split == "train"]
        if not train:
            train = list(samples)

        for it in range(1, iterations + 1):
            engine = self._engine()
            report = self.evaluate(train)
            mistakes = report["mistakes"]
            assert isinstance(mistakes, list)

            changed: set = set()
            corrected = 0

            for sample, predicted in mistakes:
                assert sample.label is not None
                true_theme = self.taxonomy.by_id(sample.label)
                true_hits = engine.hits_of(sample.text, true_theme)

                # 1) 命中错误主题的关键词 → 降权
                if predicted and self.taxonomy.has(predicted):
                    wrong = self.taxonomy.by_id(predicted)
                    wrong_hits = engine.hits_of(sample.text, wrong)
                    share = self.learning_rate / max(len(wrong_hits), 1)
                    for kw in wrong_hits:
                        before = self.weights.get(kw, 1.0)
                        after = self._clip(before - share)
                        if after != before:
                            self.weights[kw] = after
                            changed.add(kw)

                # 2) 命中正确主题的关键词 → 升权
                if true_hits:
                    step = self.learning_rate / max(len(true_hits), 1)
                    for kw in true_hits:
                        before = self.weights.get(kw, 1.0)
                        after = self._clip(before + step * len(true_hits))
                        if after != before:
                            self.weights[kw] = after
                            changed.add(kw)

            # 3) 正确样本：往基准拉回一点，防止漂移
            for sample in train:
                if not sample.label:
                    continue
                r = engine.classify(sample.text)
                if r.rejected or r.theme_id != sample.label:
                    continue
                corrected += 1
                theme = self.taxonomy.by_id(sample.label)
                for kw in engine.hits_of(sample.text, theme):
                    base = self.base_weights.get(kw, 1.0)
                    before = self.weights.get(kw, 1.0)
                    after = self._clip(before + self.reinforce * self.learning_rate * (base - before))
                    if after != before:
                        self.weights[kw] = after
                        changed.add(kw)

            self.log.append(
                IterationLog(
                    iteration=it,
                    accuracy=float(report["accuracy"]),
                    rejected_rate=float(report["reject_rate"]),
                    corrected=corrected,
                    unlearnable=int(report["unlearnable"]),
                    changed_keywords=len(changed),
                )
            )
        return self.log

    # ── 输出 ────────────────────────────────────────────

    def top_moved(self, n: int = 10) -> List[Tuple[str, float]]:
        """权重变化最大的关键词（升权在前）。"""
        delta = {
            kw: round(w - self.base_weights.get(kw, 1.0), 4)
            for kw, w in self.weights.items()
        }
        return sorted(delta.items(), key=lambda kv: -abs(kv[1]))[:n]

    def export_weights(self) -> Dict[str, float]:
        return {k: round(v, 4) for k, v in self.weights.items()}
