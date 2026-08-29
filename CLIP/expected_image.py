#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Δ = mean(CLIP_image(thumbnail)) − mean(CLIP_text(actual body))
기대이미지 = L2(CLIP_text(expected body) + Δ)
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from clip_embeddings import _save_pair_cache, load_clip_npz, load_pair_embed_dict
from route_embeddings import norm_pair_key


def news_embed_dict(path: str) -> Dict[str, np.ndarray]:
    emb, ids = load_clip_npz(path)
    out: Dict[str, np.ndarray] = {}
    for nid, vec in zip(ids, emb):
        out[str(nid)] = np.asarray(vec, dtype=np.float32).reshape(-1)
    return out


def l2_normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(x))
    if n < eps:
        return np.zeros_like(x)
    return x / n


def compute_text_image_delta(
    text_by_news: Dict[str, np.ndarray],
    image_by_news: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    μ_text, μ_img, Δ=μ_img−μ_text.
    실제본문 CLIP과 썸네일 CLIP이 둘 다 nonzero인 뉴스만 평균에 넣는다.
    """
    ids: List[str] = []
    texts: List[np.ndarray] = []
    images: List[np.ndarray] = []
    for nid, tvec in text_by_news.items():
        ivec = image_by_news.get(str(nid))
        if ivec is None:
            continue
        t = np.asarray(tvec, dtype=np.float32).reshape(-1)
        im = np.asarray(ivec, dtype=np.float32).reshape(-1)
        if t.shape[0] != im.shape[0]:
            raise ValueError(f"dim mismatch news={nid}: text={t.shape[0]} image={im.shape[0]}")
        if not np.any(t) or not np.any(im):
            continue
        ids.append(str(nid))
        texts.append(t)
        images.append(im)
    if not ids:
        raise ValueError("Δ를 계산할 교집합 뉴스가 없습니다 (둘 다 nonzero인 학습 뉴스가 없음).")
    text_mat = np.stack(texts, axis=0)
    img_mat = np.stack(images, axis=0)
    mu_text = text_mat.mean(axis=0).astype(np.float32)
    mu_img = img_mat.mean(axis=0).astype(np.float32)
    delta = (mu_img - mu_text).astype(np.float32)
    return mu_text, mu_img, delta, ids


def apply_delta_to_text_pairs(
    text_pairs: Dict[Tuple[str, str], np.ndarray],
    delta: np.ndarray,
    *,
    l2: bool = True,
) -> Dict[Tuple[str, str], np.ndarray]:
    delta = np.asarray(delta, dtype=np.float32).reshape(-1)
    out: Dict[Tuple[str, str], np.ndarray] = {}
    for key, vec in text_pairs.items():
        t = np.asarray(vec, dtype=np.float32).reshape(-1)
        nk = norm_pair_key(key[0], key[1])
        if t.shape[0] != delta.shape[0]:
            raise ValueError(f"pair dim {t.shape[0]} != delta dim {delta.shape[0]} key={nk}")
        if not np.any(t):
            out[nk] = np.zeros_like(delta)
            continue
        x = t + delta
        out[nk] = l2_normalize(x) if l2 else x.astype(np.float32)
    return out


def save_delta(
    path: str,
    mu_text: np.ndarray,
    mu_img: np.ndarray,
    delta: np.ndarray,
    news_ids: Sequence[str],
    extra: Optional[dict] = None,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    ids = [str(x) for x in news_ids]
    max_len = max((len(x) for x in ids), default=8)
    np.savez_compressed(
        path,
        mu_text=np.asarray(mu_text, dtype=np.float32),
        mu_img=np.asarray(mu_img, dtype=np.float32),
        delta=np.asarray(delta, dtype=np.float32),
        news_ids=np.asarray(ids, dtype=f"U{max(max_len, 8)}"),
    )
    meta = {
        "n_news_in_mean": len(ids),
        "dim": int(np.asarray(delta).reshape(-1).shape[0]),
        "mu_text_norm": float(np.linalg.norm(mu_text)),
        "mu_img_norm": float(np.linalg.norm(mu_img)),
        "delta_norm": float(np.linalg.norm(delta)),
        "news_ids_sample": ids[:50],
        "out_path": os.path.abspath(path),
    }
    if extra:
        meta.update(extra)
    meta_path = os.path.splitext(path)[0] + ".json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_delta(path: str) -> np.ndarray:
    data = np.load(path, allow_pickle=False)
    return np.asarray(data["delta"], dtype=np.float32).reshape(-1)


def save_pair_embed_dict(path: str, pair_dict: Dict[Tuple[str, str], np.ndarray]) -> None:
    keys = sorted(pair_dict.keys(), key=lambda x: (x[0], x[1]))
    if not keys:
        raise ValueError(f"저장할 pair가 없습니다: {path}")
    dim = int(np.asarray(pair_dict[keys[0]]).reshape(-1).shape[0])
    emb = np.zeros((len(keys), dim), dtype=np.float32)
    users = []
    news = []
    for i, k in enumerate(keys):
        users.append(str(k[0]))
        news.append(str(k[1]))
        emb[i] = np.asarray(pair_dict[k], dtype=np.float32).reshape(-1)
    _save_pair_cache(path, emb, users, news)


def load_pair_dict_normed(path: str) -> Dict[Tuple[str, str], np.ndarray]:
    raw = load_pair_embed_dict(path)
    return {norm_pair_key(u, n): np.asarray(v, dtype=np.float32).reshape(-1) for (u, n), v in raw.items()}


def pair_embed_dim(pair_dict: Dict[Tuple[str, str], np.ndarray]) -> int:
    if not pair_dict:
        raise ValueError("pair dict가 비어 있습니다.")
    return int(np.asarray(next(iter(pair_dict.values()))).reshape(-1).shape[0])


def build_test_candidate_image(
    user_ids: Sequence[str],
    news_ids: Sequence[str],
    pair_dict: Dict[Tuple[str, str], np.ndarray],
    dim: int,
) -> Tuple[np.ndarray, Dict[str, int]]:
    n = len(user_ids)
    mat = np.zeros((n, int(dim)), dtype=np.float32)
    n_hit = 0
    n_nonzero = 0
    for i, (uid, nid) in enumerate(zip(user_ids, news_ids)):
        vec = pair_dict.get(norm_pair_key(uid, nid))
        if vec is None:
            continue
        row = np.asarray(vec, dtype=np.float32).reshape(-1)
        if row.shape[0] != int(dim):
            raise ValueError(f"pair embed dim {row.shape[0]} != {dim}")
        mat[i] = row
        n_hit += 1
        if np.any(row):
            n_nonzero += 1
    return mat, {"n_rows": n, "n_cache_hit": n_hit, "n_nonzero": n_nonzero}


def build_train_candidate_image(
    user_ids: Sequence[str],
    news_id_rows: Sequence[Sequence[str]],
    pair_dict: Dict[Tuple[str, str], np.ndarray],
    dim: int,
    n_cand: int,
) -> Tuple[np.ndarray, Dict[str, int]]:
    n = len(user_ids)
    k = int(n_cand)
    mat = np.zeros((n, k, int(dim)), dtype=np.float32)
    n_hit = 0
    n_nonzero = 0
    for i, uid in enumerate(user_ids):
        row = news_id_rows[i] if i < len(news_id_rows) else []
        for p in range(k):
            nid = row[p] if p < len(row) else ""
            if nid is None or str(nid).strip() == "":
                continue
            vec = pair_dict.get(norm_pair_key(uid, nid))
            if vec is None:
                continue
            v = np.asarray(vec, dtype=np.float32).reshape(-1)
            if v.shape[0] != int(dim):
                raise ValueError(f"pair embed dim {v.shape[0]} != {dim}")
            mat[i, p] = v
            n_hit += 1
            if np.any(v):
                n_nonzero += 1
    return mat, {
        "n_sessions": n,
        "n_slots": n * k,
        "n_cache_hit": n_hit,
        "n_nonzero": n_nonzero,
    }
