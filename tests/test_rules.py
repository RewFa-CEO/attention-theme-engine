"""规则引擎的打分、归因与拒绝机制。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ate import RuleEngine, Taxonomy  # noqa: E402


@pytest.fixture(scope="module")
def tax() -> Taxonomy:
    return Taxonomy.load()


@pytest.fixture()
def engine(tax: Taxonomy) -> RuleEngine:
    return RuleEngine(taxonomy=tax)


def test_命中关键词可枚举(engine: RuleEngine, tax: Taxonomy) -> None:
    text = "智算中心建设提速，多地算力租赁签约落地"
    ranked = engine.score_text(text)
    assert ranked, "应当有命中"
    top = ranked[0]
    assert top.theme_id == "compute-leasing"
    assert "智算中心" in top.hits
    assert "算力租赁" in top.hits


def test_打分降序排列(engine: RuleEngine) -> None:
    ranked = engine.score_text("液冷方案在数据中心加速渗透，智算中心同步建设")
    scores = [r.score for r in ranked]
    assert scores == sorted(scores, reverse=True)


def test_共享关键词区分度被摊薄(tax: Taxonomy) -> None:
    # 「配额」同时属于稀土与氟化工
    assert tax.keyword_df("配额") >= 2
    assert tax.specificity("配额") < 1.0


def test_归因给出每个关键词的贡献(engine: RuleEngine) -> None:
    out = engine.explain("算力租赁签约落地，智算中心上架率提升", top_k=1)
    assert len(out) == 1
    contribs = out[0]["contributions"]
    assert contribs
    for c in contribs:
        assert c["keyword"] and c["contribution"] > 0
        assert c["specificity"] == pytest.approx(1.0 / engine.taxonomy.keyword_df(c["keyword"]))
    # 贡献之和 = 基础分（未含确认加成）
    assert sum(c["contribution"] for c in contribs) <= out[0]["theme"]["score"]


def test_无效信号被拒绝(engine: RuleEngine) -> None:
    r = engine.classify("今天天气晴朗，适合出门散步")
    assert r.rejected is True
    assert r.reason == "no_keyword_hit"
    assert r.top is None


def test_弱信号因置信度不足被拒绝(tax: Taxonomy) -> None:
    # 只有泛词、且两个题材同分 —— 置信度不足则拒绝
    strict = RuleEngine(taxonomy=tax, min_confidence=0.99)
    r = strict.classify("配额收紧")
    assert r.rejected is True
    assert r.reason in ("low_confidence", "below_min_score")


def test_分类结果带完整三级路径(engine: RuleEngine) -> None:
    r = engine.classify("白酒批价企稳，库存周期接近底部")
    assert not r.rejected
    assert r.theme_id == "baijiu"
    assert r.path is not None and len(r.path) == 3
    assert r.confidence > 0.5


def test_同一输入结果稳定(engine: RuleEngine) -> None:
    text = "储能招标量高增，构网型储能渗透提升"
    a = engine.classify(text)
    b = engine.classify(text)
    assert a.theme_id == b.theme_id
    assert a.confidence == b.confidence
