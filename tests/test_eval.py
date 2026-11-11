"""端到端：黄金样本集 + 评测脚本能跑通，指标口径正确。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ate import AttentionEngine, RuleEngine, Taxonomy  # noqa: E402
from ate.dataset import load_golden  # noqa: E402


@pytest.fixture(scope="module")
def tax() -> Taxonomy:
    return Taxonomy.load()


@pytest.fixture(scope="module")
def samples():
    return load_golden()


def test_样本集结构正确(samples) -> None:
    assert samples
    labelled = [s for s in samples if s.label]
    negatives = [s for s in samples if not s.label]
    assert len(labelled) >= 50
    assert len(negatives) >= 20
    for s in labelled:
        assert s.split in ("train", "test")


def test_所有标签都在题材体系里(tax: Taxonomy, samples) -> None:
    for s in samples:
        if s.label:
            assert tax.has(s.label), f"未知标签：{s.label}"


def test_正样本都能命中至少一个关键词(tax: Taxonomy, samples) -> None:
    engine = RuleEngine(taxonomy=tax)
    misses = []
    for s in samples:
        if not s.label:
            continue
        theme = tax.by_id(s.label)
        if not engine.hits_of(s.text, theme):
            misses.append((s.text, s.label))
    assert not misses, f"这些正样本一个关键词都没命中（覆盖缺口）：{misses}"


def test_评测脚本可运行() -> None:
    out = subprocess.run(
        [sys.executable, str(ROOT / "eval" / "evaluate.py"), "--train", "3"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert out.returncode == 0, out.stderr
    assert "评测结果" in out.stdout
    assert "Top-1=" in out.stdout


def test_评测脚本_json输出可解析() -> None:
    out = subprocess.run(
        [sys.executable, str(ROOT / "eval" / "evaluate.py"), "--json"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert out.returncode == 0, out.stderr
    rows = json.loads(out.stdout[out.stdout.index("[") :])
    assert rows and "top1" in rows[0]


def test_归因模式输出合法JSON() -> None:
    out = subprocess.run(
        [
            sys.executable,
            str(ROOT / "eval" / "evaluate.py"),
            "--explain",
            "算力租赁签约落地，智算中心上架率提升",
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert out.returncode == 0, out.stderr
    data = json.loads(out.stdout)
    assert data and "contributions" in data[0]


def test_迭代与评测闭环(tax: Taxonomy, samples) -> None:
    train = [s for s in samples if s.split == "train"]
    att = AttentionEngine(taxonomy=tax)
    att.fit(train, iterations=6)
    report = att.evaluate(samples)
    assert report["accuracy"] > 0.9
    assert report["reject_rate"] == 1.0
