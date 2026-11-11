"""题材体系本身的体检：层级完整性、关键词、区分度。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ate import Taxonomy  # noqa: E402


@pytest.fixture(scope="module")
def tax() -> Taxonomy:
    return Taxonomy.load()


def test_三级层级完整(tax: Taxonomy) -> None:
    assert len(tax) > 0
    for t in tax:
        assert t.industry_id and t.industry_name
        assert t.track_id and t.track_name
        assert t.id and t.name
        assert len(t.path) == 3


def test_题材_id唯一(tax: Taxonomy) -> None:
    ids = [t.id for t in tax]
    assert len(ids) == len(set(ids))


def test_每个题材至少两个关键词且带核心逻辑(tax: Taxonomy) -> None:
    for t in tax:
        assert len(t.keywords) >= 2, f"{t.id} 关键词太少"
        assert len(set(t.keywords)) == len(t.keywords), f"{t.id} 关键词重复"
        assert t.logic.strip(), f"{t.id} 缺核心逻辑"


def test_区分度反映共享程度(tax: Taxonomy) -> None:
    # 专属关键词区分度 = 1
    exclusive = [kw for kw in tax.all_keywords() if tax.keyword_df(kw) == 1]
    assert exclusive, "应当存在只属于一个题材的关键词"
    assert tax.specificity(exclusive[0]) == 1.0

    # 共享关键词区分度 < 1，且共享越多越低
    shared = [kw for kw in tax.all_keywords() if tax.keyword_df(kw) > 1]
    assert shared, "应当存在跨题材共享的关键词"
    for kw in shared:
        assert tax.specificity(kw) == pytest.approx(1.0 / tax.keyword_df(kw))


def test_统计口径(tax: Taxonomy) -> None:
    s = tax.stats()
    assert s["themes"] == len(tax)
    assert s["industries"] == len({t.industry_id for t in tax})
    assert s["tracks"] == len(tax.by_track())
    assert s["keywords"] == len(tax.all_keywords())
