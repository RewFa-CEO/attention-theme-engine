"""注意力权重迭代：能不能真的把准确率抬上去，以及权重往哪挪。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ate import AttentionEngine, RuleEngine, Taxonomy  # noqa: E402
from ate.dataset import load_golden  # noqa: E402


@pytest.fixture(scope="module")
def tax() -> Taxonomy:
    return Taxonomy.load()


@pytest.fixture(scope="module")
def samples():
    return load_golden()


def test_迭代把训练集准确率抬上去(tax: Taxonomy, samples) -> None:
    train = [s for s in samples if s.split == "train"]
    att = AttentionEngine(taxonomy=tax)
    before = att.evaluate(train)["accuracy"]
    log = att.fit(train, iterations=6)
    after = att.evaluate(train)["accuracy"]
    assert after > before, f"迭代后准确率应上升：{before} -> {after}"
    assert len(log) == 6
    assert log[-1].accuracy >= log[0].accuracy


def test_权重往正确方向挪(tax: Taxonomy, samples) -> None:
    train = [s for s in samples if s.split == "train"]
    att = AttentionEngine(taxonomy=tax)
    att.fit(train, iterations=6)
    # 固态电池在训练集里被「储能」压过，迭代后应当升权
    assert att.weights["固态电池"] > att.base_weights["固态电池"]
    # 被压过的那一方应当降权
    assert att.weights["储能"] < att.base_weights["储能"]


def test_权重始终被夹在区间内(tax: Taxonomy, samples) -> None:
    train = [s for s in samples if s.split == "train"]
    att = AttentionEngine(taxonomy=tax, learning_rate=0.9)
    att.fit(train, iterations=10)
    for kw, w in att.weights.items():
        assert att.clip_lo <= w <= att.clip_hi, kw


def test_迭代后泛化到测试集(tax: Taxonomy, samples) -> None:
    train = [s for s in samples if s.split == "train"]
    test = [s for s in samples if s.split == "test" and s.label]
    base = RuleEngine(taxonomy=tax)
    att = AttentionEngine(taxonomy=tax)
    att.fit(train, iterations=6)
    tuned = RuleEngine(taxonomy=tax, weights=att.export_weights())

    def acc(engine):
        hit = sum(1 for s in test if engine.classify(s.text).theme_id == s.label)
        return hit / len(test)

    assert acc(tuned) >= acc(base), "迭代后的权重不应在测试集上变差"


def test_负样本被全部拒绝(tax: Taxonomy, samples) -> None:
    negatives = [s for s in samples if not s.label]
    engine = RuleEngine(taxonomy=tax)
    rejected = sum(1 for s in negatives if engine.classify(s.text).rejected)
    assert rejected == len(negatives), f"负样本漏判 {len(negatives) - rejected} 条"


def test_覆盖缺口被单独计数(tax: Taxonomy) -> None:
    # 正确主题一个关键词都没命中的样本 —— 权重救不了，只能靠补概念
    from ate import Sample

    att = AttentionEngine(taxonomy=tax)
    report = att.evaluate([Sample(text="完全不相干的句子", label="baijiu")])
    assert report["unlearnable"] >= 1
