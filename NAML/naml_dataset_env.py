# -*- coding: utf-8 -*-
# MIND_2000: --mind-dataset-subdir MIND_2000
# Adressa_2000: --mind-dataset-subdir Adressa_2000
"""
naml_common import 이전에만 사용: dataset/<subdir>용 MIND_* 환경변수 설정.

naml_common._resolve_mind_filenames() 는 MIND_DATASET_SUBDIR 및
(선택) MIND_NEWS_FILENAME / MIND_TRAIN_FILENAME / MIND_TEST_FILENAME 환경변수를 읽는다.

--max-history-clicks N 이 있으면 NAML_MAX_HISTORY_CLICKS 환경변수를 설정한다.
naml_common 이 import 될 때 sync_max_history_clicks_from_env() 가 이를 반영한다.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

# naml_common.MIND_DATASET_PRESETS 단일 정의 (이 모듈만 수정하면 프리셋 일괄 반영)
DATASET_FILE_PRESETS: Dict[str, Tuple[str, str, str]] = {
    "MIND_2000": ("MIND_news.tsv", "MIND_train_(2000).tsv", "MIND_test_(2000).tsv"),
    "Adressa_2000": ("Adressa_news.tsv", "Adressa_train_(2000).tsv", "Adressa_test_(2000).tsv"),
}

# k-means / eval 기본 경로 (프로젝트 루트 기준 상대). --mind-dataset-subdir 가 Adressa 이면 Adressa CSV·가중치.
_DEFAULT_KMEANS_K = 3


def default_user_kmeans_csv(subdir: str, k: int = _DEFAULT_KMEANS_K) -> str:
    """
    user_kmeans CSV 기본값. subdir 이름에 adressa 가 들어가면 Adressa_2000용 파일명.
    """
    s = (subdir or "").strip() or "MIND_2000"
    if "adressa" in s.lower():
        return f"NAML/user_kmeans_k{k}_Adressa_2000.csv"
    return f"NAML/user_kmeans_k{k}_MIND_2000.csv"


def default_naml_eval_weights(subdir: str) -> str:
    """
    eval_cluster_batch 등에서 쓰는 사전학습 가중치 기본 경로 (프로젝트 루트 기준 상대).
    Adressa: saved_models/Adressa_2000/NAML_adressa_2000_actual.h5
    그 외: saved_models/NAML_mind_2000.h5
    """
    s = (subdir or "").strip() or "MIND_2000"
    if "adressa" in s.lower():
        return "saved_models/Adressa_2000/NAML_adressa_2000_actual.h5"
    return "saved_models/NAML_mind_2000.h5"


def _argv_value(argv: List[str], flag: str) -> Optional[str]:
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1].strip()
    return None


def news_tsv_skiprows(news_path: str) -> int:
    """MIND 뉴스 TSV는 헤더 없음. Adressa_news 등은 첫 행이 컬럼명인 경우가 있음."""
    with open(news_path, "r", encoding="utf-8") as f:
        first = f.readline().split("\t")[0].strip().lower()
    if first in ("news_id", "clicked_news", "id"):
        return 1
    return 0


def apply_dataset_env_from_argv(argv: Optional[List[str]] = None) -> str:
    """
    sys.argv 에서 --mind-dataset-subdir 만 읽어 환경변수를 맞춘다.
    이미 설정된 MIND_NEWS/TRAIN/TEST 는 setdefault 로 덮어쓰지 않는다.

    Returns: 적용된 dataset 하위 폴더명 (예: MIND_2000).
    """
    argv = argv if argv is not None else __import__("sys").argv
    argv_list = list(argv)
    cli_sub = _argv_value(argv_list, "--mind-dataset-subdir")
    sub = cli_sub or os.environ.get("MIND_DATASET_SUBDIR", "MIND_2000")
    os.environ["MIND_DATASET_SUBDIR"] = sub
    if sub in DATASET_FILE_PRESETS:
        n, tr, te = DATASET_FILE_PRESETS[sub]
        os.environ.setdefault("MIND_NEWS_FILENAME", n)
        os.environ.setdefault("MIND_TRAIN_FILENAME", tr)
        os.environ.setdefault("MIND_TEST_FILENAME", te)
    cli_hist = _argv_value(argv_list, "--max-history-clicks")
    if cli_hist is not None:
        os.environ["NAML_MAX_HISTORY_CLICKS"] = cli_hist.strip()
    return sub
