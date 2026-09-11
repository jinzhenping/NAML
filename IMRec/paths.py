#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

_IMREC = Path(__file__).resolve().parent
PROJECT_ROOT = _IMREC.parent

# 썸네일 경로/해석은 MM_Rec 과 동일 소스 사용
_MMREC = PROJECT_ROOT / "MM_Rec"
if str(_MMREC) not in sys.path:
    sys.path.insert(0, str(_MMREC))
from dataset_paths import (  # noqa: E402
    DEFAULT_THUMBNAIL_DIR,
    resolve_project_path,
    thumbnail_path,
)

DATASET_FILE_PRESETS = {
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


def dataset_raw_dir(subdir: str) -> Path:
    return PROJECT_ROOT / "dataset" / subdir


def prepared_dir(subdir: str) -> Path:
    return _IMREC / "data" / subdir


def saved_dir(subdir: str) -> Path:
    return _IMREC / "saved_models" / subdir


def cards_dir(subdir: str) -> Path:
    return prepared_dir(subdir) / "cards"


def features_path(subdir: str) -> Path:
    return prepared_dir(subdir) / "impression_features.npz"
