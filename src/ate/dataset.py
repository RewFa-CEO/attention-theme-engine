"""黄金样本集读写。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .attention import Sample

__all__ = ["load_golden", "DEFAULT_GOLDEN_PATH"]

DEFAULT_GOLDEN_PATH = Path(__file__).resolve().parents[2] / "data" / "golden.jsonl"


def load_golden(path: Path | str | None = None) -> List[Sample]:
    """读取 JSONL 黄金样本。label 为 null 的行是负样本（应当被拒绝）。"""
    p = Path(path) if path else DEFAULT_GOLDEN_PATH
    samples: List[Sample] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        samples.append(
            Sample(
                text=row["text"],
                label=row.get("label"),
                split=row.get("split", "train"),
            )
        )
    return samples
