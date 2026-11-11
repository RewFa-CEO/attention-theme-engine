"""三级题材分类体系（行业 / 赛道 / 核心逻辑）的数据结构与加载。

这是「注意力机制」研究框架的公开切片：生产系统的完整概念集与原始数据源
不在公开范围内（见 docs/SLICE.md）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

__all__ = ["Theme", "Taxonomy", "DEFAULT_TAXONOMY_PATH"]

DEFAULT_TAXONOMY_PATH = Path(__file__).resolve().parents[2] / "data" / "taxonomy.json"


@dataclass(frozen=True)
class Theme:
    """一个题材（三级分类的最细一级）。

    id      题材唯一标识
    name    题材名
    keywords 命中即计分的关键词
    logic   核心逻辑（一句话说清「为什么这个题材会动」）
    track_id / track_name  上级赛道
    industry_id / industry_name  上级行业
    """

    id: str
    name: str
    keywords: Tuple[str, ...]
    logic: str
    track_id: str
    track_name: str
    industry_id: str
    industry_name: str

    @property
    def path(self) -> Tuple[str, str, str]:
        """完整三级路径：行业 / 赛道 / 核心逻辑。"""
        return (self.industry_name, self.track_name, self.name)


class Taxonomy:
    """题材体系。只读，一次加载多处复用。"""

    def __init__(self, themes: Iterable[Theme], version: str = "0") -> None:
        self._themes: Tuple[Theme, ...] = tuple(themes)
        self._by_id: Dict[str, Theme] = {t.id: t for t in self._themes}
        self.version = version

        # 关键词 -> 出现它的题材数（用于区分度：越专属越强）
        self._keyword_df: Dict[str, int] = {}
        for t in self._themes:
            for kw in t.keywords:
                self._keyword_df[kw] = self._keyword_df.get(kw, 0) + 1

    # ── 构造 ────────────────────────────────────────────

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Taxonomy":
        p = Path(path) if path else DEFAULT_TAXONOMY_PATH
        raw = json.loads(p.read_text(encoding="utf-8"))
        themes: List[Theme] = []
        for industry in raw.get("industries", []):
            for track in industry.get("tracks", []):
                for theme in track.get("themes", []):
                    themes.append(
                        Theme(
                            id=theme["id"],
                            name=theme["name"],
                            keywords=tuple(theme.get("keywords", [])),
                            logic=theme.get("logic", ""),
                            track_id=track["id"],
                            track_name=track["name"],
                            industry_id=industry["id"],
                            industry_name=industry["name"],
                        )
                    )
        return cls(themes, version=str(raw.get("version", "0")))

    # ── 查询 ────────────────────────────────────────────

    @property
    def themes(self) -> Tuple[Theme, ...]:
        return self._themes

    def by_id(self, theme_id: str) -> Theme:
        return self._by_id[theme_id]

    def has(self, theme_id: str) -> bool:
        return theme_id in self._by_id

    def keyword_df(self, keyword: str) -> int:
        """该关键词出现在几个题材里（1 = 专属，越大越泛）。"""
        return self._keyword_df.get(keyword, 0)

    def specificity(self, keyword: str) -> float:
        """区分度：专属关键词记满，共享关键词按出现次数摊薄。"""
        df = self.keyword_df(keyword)
        return 1.0 / df if df else 1.0

    def all_keywords(self) -> Tuple[str, ...]:
        seen: Dict[str, None] = {}
        for t in self._themes:
            for kw in t.keywords:
                seen.setdefault(kw, None)
        return tuple(seen)

    def by_track(self) -> Dict[str, List[Theme]]:
        out: Dict[str, List[Theme]] = {}
        for t in self._themes:
            out.setdefault(t.track_id, []).append(t)
        return out

    def stats(self) -> Dict[str, int]:
        tracks = self.by_track()
        return {
            "industries": len({t.industry_id for t in self._themes}),
            "tracks": len(tracks),
            "themes": len(self._themes),
            "keywords": len(self.all_keywords()),
        }

    def __len__(self) -> int:
        return len(self._themes)

    def __iter__(self):
        return iter(self._themes)
