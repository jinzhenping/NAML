#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MIND_2000 / Adressa_2000 파일명과 MM-Rec 준비 디렉터리."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

_MMREC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _MMREC_DIR.parent

DATASET_FILE_PRESETS: Dict[str, Tuple[str, str, str, str]] = {
    "MIND_2000": (
        "MIND_news.tsv",
        "MIND_train_(2000).tsv",
        "MIND_test_(2000).tsv",
        "MIND_test_2000_final.tsv",
    ),
    "Adressa_2000": (
        "Adressa_news.tsv",
        "Adressa_train_(2000).tsv",
        "Adressa_test_(2000).tsv",
        "Adressa_test_2000_final.tsv",
    ),
}

DEFAULT_THUMBNAIL_DIR = PROJECT_ROOT / "dataset" / "MIND_thumbnail"


def resolve_project_path(p: str) -> Path:
    p = (p or "").strip()
    if not p:
        return Path()
    path = Path(p)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def dataset_raw_dir(mind_dataset_subdir: str) -> Path:
    return PROJECT_ROOT / "dataset" / mind_dataset_subdir


def prepared_dir(mind_dataset_subdir: str) -> Path:
    return _MMREC_DIR / "data" / mind_dataset_subdir


def saved_dir(mind_dataset_subdir: str) -> Path:
    return _MMREC_DIR / "saved_models" / mind_dataset_subdir


def default_config_file() -> Path:
    return _MMREC_DIR / "config" / "vilbert_8layer.json"


def discover_tsv_names(subdir: str) -> Tuple[str, str, str, Optional[str]]:
    if subdir in DATASET_FILE_PRESETS:
        n, tr, te, tf = DATASET_FILE_PRESETS[subdir]
        return n, tr, te, tf
    base = dataset_raw_dir(subdir)
    news = "MIND_news.tsv"
    for fixed in ("MIND_news.tsv", "Adressa_news.tsv"):
        if (base / fixed).is_file():
            news = fixed
            break
    trains = sorted(base.glob("MIND_train_*.tsv")) or sorted(base.glob("*_train_*.tsv"))
    tests = [
        p
        for p in sorted(base.glob("MIND_test_*.tsv")) + sorted(base.glob("*_test_*.tsv"))
        if "_final" not in p.name.lower()
    ]
    finals = [
        p
        for p in sorted(base.glob("*test*final*.tsv"))
    ]
    train = trains[0].name if trains else "MIND_train_(2000).tsv"
    test = tests[0].name if tests else "MIND_test_(2000).tsv"
    final = finals[0].name if finals else None
    return news, train, test, final


def thumbnail_path(thumbnail_dir: Path, news_id: str) -> Optional[Path]:
    for sfx in (".jpg", ".jpeg", ".png", ".webp"):
        p = thumbnail_dir / f"{news_id}{sfx}"
        if p.is_file():
            return p
    return None
