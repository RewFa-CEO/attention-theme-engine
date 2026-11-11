#!/usr/bin/env python3
"""评测脚本：Top-1 准确率 / Top-3 命中率 / 无效信号拒绝率。

用法：
    python3 eval/evaluate.py                 # 基线（未迭代权重）
    python3 eval/evaluate.py --train 5       # 先迭代 5 轮再评测
    python3 eval/evaluate.py --explain "..."  # 看某条文本的打分归因
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ate import AttentionEngine, RuleEngine, Taxonomy  # noqa: E402
from ate.dataset import load_golden  # noqa: E402


def top_k_accuracy(engine: RuleEngine, samples, k: int) -> float:
    """Top-k 命中率：正确答案出现在前 k 名里就算命中。"""
    labelled = [s for s in samples if s.label]
    if not labelled:
        return 0.0
    hit = 0
    for s in labelled:
        ranked = engine.score_text(s.text)[:k]
        if any(c.theme_id == s.label for c in ranked):
            hit += 1
    return hit / len(labelled)


def report(engine: RuleEngine, samples, tag: str) -> dict:
    labelled = [s for s in samples if s.label]
    negatives = [s for s in samples if not s.label]
    correct = 0
    rejected_labelled = 0
    for s in labelled:
        r = engine.classify(s.text)
        if r.rejected:
            rejected_labelled += 1
        elif r.theme_id == s.label:
            correct += 1
    rejected_negatives = sum(1 for s in negatives if engine.classify(s.text).rejected)
    out = {
        "tag": tag,
        "samples": len(samples),
        "labelled": len(labelled),
        "negatives": len(negatives),
        "top1": round(correct / len(labelled), 4) if labelled else 0.0,
        "top3": round(top_k_accuracy(engine, samples, 3), 4),
        "rejected_labelled": rejected_labelled,
        "rejected_negatives": rejected_negatives,
        "reject_rate": round(rejected_negatives / len(negatives), 4) if negatives else 0.0,
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=0, help="迭代轮数（0 = 只跑基线）")
    ap.add_argument("--taxonomy", default=None)
    ap.add_argument("--golden", default=None)
    ap.add_argument("--explain", default=None, help="打印某条文本的归因")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args()

    taxonomy = Taxonomy.load(args.taxonomy)
    samples = load_golden(args.golden)
    train = [s for s in samples if s.split == "train"]
    test = [s for s in samples if s.split == "test"]

    if args.explain:
        engine = RuleEngine(taxonomy=taxonomy)
        print(json.dumps(engine.explain(args.explain), ensure_ascii=False, indent=2))
        return 0

    baseline = RuleEngine(taxonomy=taxonomy)
    rows = [report(baseline, test, "baseline/test")]
    quiet = args.json

    if args.train:
        att = AttentionEngine(taxonomy=taxonomy)
        log = att.fit(train, iterations=args.train)
        tuned = RuleEngine(taxonomy=taxonomy, weights=att.export_weights())
        rows.append(report(tuned, test, f"tuned-{args.train}/test"))
        rows.append(report(tuned, samples, f"tuned-{args.train}/all"))
        if not quiet:
            print("── 迭代过程 ──")
        for it in (log if not quiet else []):
            print(
                f"  iter {it.iteration}: acc={it.accuracy:.3f} "
                f"reject={it.rejected_rate:.3f} 修正={it.corrected} "
                f"覆盖缺口={it.unlearnable} 变动关键词={it.changed_keywords}"
            )
        if not quiet:
            print("  权重变动最大的关键词：")
            for kw, d in att.top_moved(8):
                print(f"    {kw:12} {d:+.3f}")

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0

    print("── 评测结果 ──")
    for r in rows:
        print(
            f"  [{r['tag']}] Top-1={r['top1']:.3f}  Top-3={r['top3']:.3f}  "
            f"拒绝率={r['reject_rate']:.3f}  "
            f"(拒绝负样本 {r['rejected_negatives']}/{r['negatives']}，"
            f"误拒正样本 {r['rejected_labelled']}/{r['labelled']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
