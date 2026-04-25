# -*- coding: utf-8 -*-
# MIND_2000: MIND_DATASET_SUBDIR=MIND_2000
# Adressa_2000: MIND_DATASET_SUBDIR=Adressa_2000
"""dataset/<subdir> 아래 뉴스·train·test TSV 경로 (MIND_*, Adressa_* 등)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence


def _is_impression_header_line(line: str) -> bool:
    parts = line.strip().split("\t")
    return bool(parts) and parts[0].strip().lower() == "user"


def merge_impression_tsv_paths(paths: Sequence[Path], out_path: Path) -> None:
    """헤더 행(user로 시작)은 건너뛰고 데이터 행만 순서대로 이어붙인다."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        for p in paths:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if _is_impression_header_line(line):
                        continue
                    out.write(line if line.endswith("\n") else line + "\n")


def collect_test_tsv_merge_paths(
    dataset_dir: Path,
    primary: Path,
    *,
    merge_final: bool = True,
    extra_paths: Optional[Sequence[Path]] = None,
) -> list[Path]:
    """
    기본(비-final) test TSV + (옵션) dataset_dir/*test*final*.tsv + extra_paths.
    존재하는 파일만, resolve 기준 중복 제거.
    """
    paths: list[Path] = [primary]
    seen: set = {primary.resolve()}
    if merge_final:
        for p in sorted(dataset_dir.glob("*test*final*.tsv")):
            if p.is_file():
                rp = p.resolve()
                if rp not in seen:
                    seen.add(rp)
                    paths.append(p)
    for e in extra_paths or []:
        if e.is_file():
            rp = e.resolve()
            if rp not in seen:
                seen.add(rp)
                paths.append(e)
    return [p for p in paths if p.is_file()]


def resolve_news_tsv(dataset_dir: Path) -> Path:
    for name in ("MIND_news.tsv", "Adressa_news.tsv"):
        p = dataset_dir / name
        if p.is_file():
            return p
    cand = sorted(dataset_dir.glob("*news.tsv"))
    if len(cand) == 1:
        return cand[0]
    if not cand:
        raise FileNotFoundError(f"No *news.tsv in {dataset_dir}")
    raise RuntimeError(f"Multiple *news.tsv in {dataset_dir}; pass --news_tsv")


def resolve_train_tsv(dataset_dir: Path) -> Path:
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    cand = sorted(dataset_dir.glob("*_train_*.tsv"))
    if len(cand) == 1:
        return cand[0]
    if len(cand) == 0:
        raise FileNotFoundError(f"No *_train_*.tsv in {dataset_dir}")
    raise RuntimeError(f"Multiple train TSV in {dataset_dir}; pass --train_tsv")


def impression_tsv_header_skiprows(tsv: Path) -> int:
    """MIND/Adressa train TSV 첫 줄은 컬럼명(user, ...). 테스트 TSV는 보통 헤더 없음."""
    with open(tsv, "r", encoding="utf-8") as f:
        first = f.readline().split("\t")[0].strip().lower()
    if first == "user":
        return 1
    return 0


def news_tsv_skiprows(news_tsv: Path) -> int:
    """MIND 뉴스 TSV는 헤더 없음. Adressa_news 등은 첫 행이 컬럼명인 경우가 있음."""
    with open(news_tsv, "r", encoding="utf-8") as f:
        first = f.readline().split("\t")[0].strip().lower()
    if first in ("news_id", "clicked_news", "id"):
        return 1
    return 0


def resolve_test_tsv(dataset_dir: Path) -> Path:
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    cand = sorted(dataset_dir.glob("*_test_*.tsv"))
    cand = [p for p in cand if "_final" not in p.name.lower()]
    if len(cand) == 1:
        return cand[0]
    if len(cand) == 0:
        raise FileNotFoundError(f"No non-_final *_test_*.tsv in {dataset_dir}")
    raise RuntimeError(f"Multiple test TSV in {dataset_dir}; pass --test_tsv")
