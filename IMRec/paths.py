#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

_IMREC = Path(__file__).resolve().parent
PROJECT_ROOT = _IMREC.parent


def _load_mmrec_dataset_paths() -> Any:
    """MM_Rec을 sys.path에 넣지 않고 dataset_paths만 로드 (metrics 이름 충돌 방지)."""
    path = PROJECT_ROOT / "MM_Rec" / "dataset_paths.py"
    spec = importlib.util.spec_from_file_location("mm_rec_dataset_paths", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mm = _load_mmrec_dataset_paths()
DEFAULT_THUMBNAIL_DIR = _mm.DEFAULT_THUMBNAIL_DIR
resolve_project_path = _mm.resolve_project_path
thumbnail_path = _mm.thumbnail_path

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
