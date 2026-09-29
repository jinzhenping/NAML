#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""IMRec impression feature npz → NAML news-index 행렬 (S2 CLIP cache와 동일 역할)."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_CLIP = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_CLIP) not in sys.path:
    sys.path.insert(0, str(_CLIP))

from clip_embeddings import resolve_project_path

IM_LOCAL_DIM = 512
IM_GLOBAL_DIM = 2048
IM_N_COVER = 9
IM_MAX_TITLE_WORDS = 30


def default_im_feature_path(mind_dataset_subdir: str) -> str:
    return str(_ROOT / "IMRec" / "data" / mind_dataset_subdir / "impression_features.npz")


@dataclass
class ImNewsFeatures:
    """news_words와 같은 행 인덱스. 0행 = padding."""

    word_vis: np.ndarray  # [N, L, 512]
    word_vis_mask: np.ndarray  # [N, L]
    cover_regions: np.ndarray  # [N, 9, 512]
    category_vis: np.ndarray  # [N, 512]
    global_feat: np.ndarray  # [N, 2048]

    @property
    def n_rows(self) -> int:
        return int(self.word_vis.shape[0])

    @property
    def title_len(self) -> int:
        return int(self.word_vis.shape[1])


def build_im_news_features(
    news_index: Dict[str, int],
    n_rows: int,
    cache_path: str,
    title_len: Optional[int] = None,
) -> Tuple[ImNewsFeatures, int]:
    """
    npz news_ids 순서와 무관하게 news_index 행에 scatter.
    Returns (features, n_nonzero_global)
    """
    path = resolve_project_path(cache_path)
    if not path or not __import__("os").path.isfile(path):
        raise FileNotFoundError(
            f"IM feature cache 없음: {path}\n"
            "conda activate clip_cu128\n"
            "python IMRec/train_eval.py --stage prepare --mind-dataset-subdir MIND_2000\n"
            "python IMRec/train_eval.py --stage extract --mind-dataset-subdir MIND_2000"
        )
    L = int(title_len or IM_MAX_TITLE_WORDS)
    z = np.load(path, allow_pickle=False)
    feat_ids = [str(x) for x in z["news_ids"].tolist()]
    src_w = np.asarray(z["word_vis"], dtype=np.float32)
    src_wm = np.asarray(z["word_vis_mask"], dtype=np.float32)
    src_c = np.asarray(z["cover_regions"], dtype=np.float32)
    src_cat = np.asarray(z["category_vis"], dtype=np.float32)
    src_g = np.asarray(z["global_feat"], dtype=np.float32)
    id_to_row = {nid: i for i, nid in enumerate(feat_ids)}

    word_vis = np.zeros((int(n_rows), L, IM_LOCAL_DIM), dtype=np.float32)
    word_vis_mask = np.zeros((int(n_rows), L), dtype=np.float32)
    cover = np.zeros((int(n_rows), IM_N_COVER, IM_LOCAL_DIM), dtype=np.float32)
    category_vis = np.zeros((int(n_rows), IM_LOCAL_DIM), dtype=np.float32)
    global_feat = np.zeros((int(n_rows), IM_GLOBAL_DIM), dtype=np.float32)
    n_hit = 0
    for nid, idx in news_index.items():
        i = int(idx)
        if i <= 0 or i >= n_rows:
            continue
        row = id_to_row.get(str(nid))
        if row is None:
            continue
        wl = min(L, src_w.shape[1])
        word_vis[i, :wl] = src_w[row, :wl]
        word_vis_mask[i, :wl] = src_wm[row, :wl]
        cover[i] = src_c[row]
        category_vis[i] = src_cat[row]
        global_feat[i] = src_g[row]
        if np.any(src_g[row]):
            n_hit += 1
    return (
        ImNewsFeatures(
            word_vis=word_vis,
            word_vis_mask=word_vis_mask,
            cover_regions=cover,
            category_vis=category_vis,
            global_feat=global_feat,
        ),
        n_hit,
    )
