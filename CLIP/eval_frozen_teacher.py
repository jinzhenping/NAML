#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Frozen full-text S2 teacher 위에서 후보 5번째 뷰만 교체해 평가.

히스토리 이미지 슬롯은 항상 B0 (썸네일 CLIP).
후보만 B0 / B1 / B2 / B3 / B4.

  conda activate tf28gpu
  python CLIP/eval_frozen_teacher.py --branches b0,b1,b2,b3,b4 --mind-dataset-subdir MIND_2000
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_CLIP_DIR = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "NAML") not in sys.path:
    sys.path.insert(0, str(_ROOT / "NAML"))
if str(_CLIP_DIR) not in sys.path:
    sys.path.insert(0, str(_CLIP_DIR))

from naml_dataset_env import apply_dataset_env_from_argv

apply_dataset_env_from_argv()

import tensorflow as tf
from tensorflow.keras import backend as K

from clip_embeddings import (
    DEFAULT_THUMBNAIL_DIR,
    build_news_image_matrix,
    count_missing_thumbnails,
    default_b1_cache_path,
    default_b2_cache_path,
    default_b3_cache_path,
    default_b4_cache_path,
    default_cache_path,
    load_news_ids_from_tsv,
    load_pair_embed_dict,
    print_missing_thumbnail_report,
    resolve_project_path,
)
from naml_common import (
    MIND_NEWS_FILENAME,
    SEED,
    get_embedding,
    mind_data_path,
    preprocess_news_file,
    preprocess_user_file,
    sync_max_history_clicks_from_env,
)
import naml_common
from naml_image_model import build_naml_models_with_image
from route_embeddings import default_expected_body_dir, load_expected_bodies_from_dir, norm_pair_key
from train_s1_s2 import evaluate_metrics, load_hparams


def _parse_branches(raw: str) -> List[str]:
    allowed = {"b0", "b1", "b2", "b3", "b4"}
    out: List[str] = []
    for part in (raw or "").split(","):
        key = part.strip().lower()
        if not key:
            continue
        if key not in allowed:
            raise ValueError(f"unknown branch {part!r}. use b0,b1,b2,b3,b4")
        if key not in out:
            out.append(key)
    if not out:
        raise ValueError("--branches 가 비어 있습니다. 예: b0,b1,b2,b3,b4")
    return out


def _set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _max_history_from_log(tune_log: str) -> Optional[int]:
    path = resolve_project_path(tune_log)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        v = data.get("max_history_clicks")
        return int(v) if v is not None else None
    except Exception:
        return None


def _pair_row_matrix(
    user_ids: Sequence[str],
    news_ids: Sequence[str],
    pair_dict: Dict[Tuple[str, str], np.ndarray],
    dim: int,
) -> Tuple[np.ndarray, Dict[str, int]]:
    n = len(user_ids)
    mat = np.zeros((n, int(dim)), dtype=np.float32)
    n_key = 0
    n_nonzero = 0
    for i, (uid, nid) in enumerate(zip(user_ids, news_ids)):
        vec = pair_dict.get(norm_pair_key(uid, nid))
        if vec is None:
            continue
        n_key += 1
        row = np.asarray(vec, dtype=np.float32).reshape(-1)
        if row.shape[0] != int(dim):
            raise ValueError(f"pair embed dim {row.shape[0]} != teacher clip_dim {dim}")
        mat[i] = row
        if np.any(row):
            n_nonzero += 1
    return mat, {"n_rows": n, "n_cache_hit": n_key, "n_nonzero": n_nonzero}


def _news_row_from_index(
    news_indices: np.ndarray,
    news_image: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, int]]:
    mat = news_image[news_indices]
    n = int(mat.shape[0])
    n_nonzero = int(np.any(mat, axis=1).sum()) if n else 0
    return mat, {"n_rows": n, "n_nonzero": n_nonzero}


def _expected_body_coverage(
    user_ids: Sequence[str],
    news_ids: Sequence[str],
    expected_bodies: Dict[Tuple[str, str], str],
) -> Dict[str, int]:
    n = len(user_ids)
    n_hit = 0
    for uid, nid in zip(user_ids, news_ids):
        if expected_bodies.get(norm_pair_key(uid, nid)):
            n_hit += 1
    return {"n_rows": n, "n_expected_body": n_hit}


def main() -> None:
    ap = argparse.ArgumentParser(description="Frozen S2 teacher 후보 이미지 슬롯 교체 평가")
    ap.add_argument("--mind-dataset-subdir", type=str, default="MIND_2000")
    ap.add_argument("--branches", type=str, default="b0,b1,b2,b3,b4")
    ap.add_argument(
        "--weights",
        type=str,
        default="CLIP/saved_models/MIND_2000/S2_naml_clip_tuned.h5",
    )
    ap.add_argument(
        "--tune-log",
        type=str,
        default="CLIP/saved_models/MIND_2000/naml_tune_s2_clip_log.json",
    )
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--max-history-clicks", type=int, default=None)
    ap.add_argument("--thumbnail-dir", type=str, default=DEFAULT_THUMBNAIL_DIR)
    ap.add_argument("--clip-cache", type=str, default=None)
    ap.add_argument("--b1-cache", type=str, default=None)
    ap.add_argument("--b2-cache", type=str, default=None)
    ap.add_argument("--b3-cache", type=str, default=None)
    ap.add_argument("--b4-cache", type=str, default=None)
    ap.add_argument("--expected-body-dir", type=str, default=None)
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="결과 JSON. 기본 CLIP/saved_models/<subdir>/frozen_teacher_<branches>.json",
    )
    args = ap.parse_args()

    branches = _parse_branches(args.branches)
    argv = ["--mind-dataset-subdir", args.mind_dataset_subdir]
    hist = args.max_history_clicks
    if hist is None:
        hist = _max_history_from_log(args.tune_log)
    if hist is not None:
        argv += ["--max-history-clicks", str(hist)]
    apply_dataset_env_from_argv(argv)
    sync_max_history_clicks_from_env()

    _set_seed(int(args.seed))
    hp = load_hparams(args.tune_log)
    weights_path = resolve_project_path(args.weights)
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"teacher 가중치 없음: {weights_path}")

    print(
        f"[eval] frozen full-text S2  weights={weights_path}\n"
        f"[eval] branches={branches}  history_image=B0  candidate_image=swapped",
        flush=True,
    )
    word_dict, category, subcategory, news_words, news_body, news_v, news_sv, news_index = (
        preprocess_news_file(
            expected_bodies_train=None,
            expected_bodies_test=None,
            expected_bodies_vocab_extra=None,
        )
    )
    (
        _userid_dict,
        _all_train_pn,
        _all_label,
        _all_train_id,
        all_test_pn,
        all_test_label,
        all_test_id,
        _all_user_pos,
        all_test_user_pos,
        all_test_index,
        _c1,
        _c2,
        _tr_u,
        _tr_n,
        all_test_userid_str,
        all_test_newsid_str,
    ) = preprocess_user_file(
        news_index=news_index,
        expected_bodies_train=None,
        expected_bodies_test=None,
        word_dict=word_dict,
    )
    embedding_mat = get_embedding(word_dict)

    thumb_dir = resolve_project_path(args.thumbnail_dir)
    catalog_ids = [nid for nid, idx in news_index.items() if nid != "0" and int(idx) != 0]
    print_missing_thumbnail_report(catalog_ids, thumb_dir)

    b0_cache = (
        resolve_project_path(args.clip_cache)
        if args.clip_cache
        else default_cache_path(args.mind_dataset_subdir)
    )
    if not os.path.isfile(b0_cache):
        raise FileNotFoundError(f"B0 CLIP cache 없음: {b0_cache}")
    news_tsv = mind_data_path(MIND_NEWS_FILENAME)
    fallback_ids, _ = count_missing_thumbnails(load_news_ids_from_tsv(news_tsv), thumb_dir)
    news_image_b0, n_hit_b0 = build_news_image_matrix(
        news_index, len(news_words), b0_cache, news_ids_fallback=fallback_ids
    )
    clip_dim = int(news_image_b0.shape[1])
    print(
        f"[eval] B0 matrix {news_image_b0.shape} nonzero news={n_hit_b0}/{len(catalog_ids)}  "
        f"history={naml_common.MAX_HISTORY_CLICKS}  val_rows={len(all_test_id)} sessions={len(all_test_index)}",
        flush=True,
    )

    expected_dir = (
        resolve_project_path(args.expected_body_dir)
        if args.expected_body_dir
        else default_expected_body_dir(str(_ROOT), args.mind_dataset_subdir, split="test")
    )
    expected_bodies = {}
    if any(b in branches for b in ("b1", "b2", "b3")):
        expected_bodies = load_expected_bodies_from_dir(expected_dir)
        print(
            f"[eval] expected_body_dir={expected_dir} files={len(expected_bodies)}",
            flush=True,
        )

    caches = {
        "b1": resolve_project_path(args.b1_cache) if args.b1_cache else default_b1_cache_path(args.mind_dataset_subdir),
        "b2": resolve_project_path(args.b2_cache) if args.b2_cache else default_b2_cache_path(args.mind_dataset_subdir),
        "b3": resolve_project_path(args.b3_cache) if args.b3_cache else default_b3_cache_path(args.mind_dataset_subdir),
        "b4": resolve_project_path(args.b4_cache) if args.b4_cache else default_b4_cache_path(args.mind_dataset_subdir),
    }

    built = build_naml_models_with_image(
        word_dict,
        embedding_mat,
        category,
        subcategory,
        hp["learning_rate"],
        clip_dim=clip_dim,
        clear_session=True,
        dropout_rate=hp["dropout_rate"],
        cnn_filters=hp["cnn_filters"],
        cnn_kernel_size=hp["cnn_kernel_size"],
        attention_dense_dim=hp["attention_dense_dim"],
        category_emb_dim=hp["category_emb_dim"],
    )
    model = built["model"]
    model_test = built["model_test"]
    model.load_weights(weights_path)
    print(f"[eval] loaded frozen weights: {weights_path}", flush=True)

    results: Dict[str, Any] = {}
    descriptions = {
        "b0": "thumbnail CLIP image embed (news-level)",
        "b1": "expected-body CLIP text embed (user, news) Route T",
        "b2": "expected-body Kandinsky prior image embed (user, news) Route E",
        "b3": "personalized pixel CLIP image embed (user, news) Route P",
        "b4": "title-only generated pixel CLIP image embed (news-level)",
    }

    for br in branches:
        cand_image = None
        cov: Dict[str, Any] = {"history": "B0"}
        if br == "b0":
            cand_mat, cov_rows = _news_row_from_index(all_test_pn, news_image_b0)
            cand_image = cand_mat
            cov.update(cov_rows)
            cov["space"] = "clip_image"
        elif br in ("b1", "b2", "b3"):
            cache_path = caches[br]
            if not os.path.isfile(cache_path):
                raise FileNotFoundError(
                    f"{br.upper()} cache 없음: {cache_path}\n"
                    + (
                        "먼저 clip_cu128 에서 python CLIP/extract_b3_embeds.py 를 실행하세요."
                        if br == "b3"
                        else f"먼저 clip_cu128 에서 python CLIP/extract_route_embeds.py --routes {br} 를 실행하세요."
                    )
                )
            pair_dict = {
                norm_pair_key(u, n): v for (u, n), v in load_pair_embed_dict(cache_path).items()
            }
            sample_dim = clip_dim
            if pair_dict:
                sample_dim = int(next(iter(pair_dict.values())).shape[0])
            if sample_dim != clip_dim:
                raise ValueError(
                    f"{br.upper()} embed dim={sample_dim} 이 teacher CLIP image dim={clip_dim} 과 다릅니다. "
                    f"같은 Kandinsky 2.2 prior CLIP을 써야 합니다."
                )
            cand_mat, cov_rows = _pair_row_matrix(
                all_test_userid_str, all_test_newsid_str, pair_dict, clip_dim
            )
            cand_image = cand_mat
            cov.update(cov_rows)
            cov["n_pair_cache"] = len(pair_dict)
            cov.update(_expected_body_coverage(all_test_userid_str, all_test_newsid_str, expected_bodies))
            cov["space"] = "clip_text" if br == "b1" else "clip_image"
        elif br == "b4":
            cache_path = caches["b4"]
            if not os.path.isfile(cache_path):
                raise FileNotFoundError(
                    f"B4 cache 없음: {cache_path}\n"
                    f"먼저 clip_cu128 에서 python CLIP/extract_route_embeds.py --routes b4 를 실행하세요."
                )
            news_image_b4, n_hit_b4 = build_news_image_matrix(
                news_index, len(news_words), cache_path, news_ids_fallback=fallback_ids
            )
            if int(news_image_b4.shape[1]) != clip_dim:
                raise ValueError(
                    f"B4 embed dim={news_image_b4.shape[1]} != teacher CLIP dim={clip_dim}"
                )
            cand_mat, cov_rows = _news_row_from_index(all_test_pn, news_image_b4)
            cand_image = cand_mat
            cov.update(cov_rows)
            cov["n_nonzero_news"] = int(n_hit_b4)
            cov["space"] = "clip_image"
        else:
            continue

        print(
            f"\n=== {br.upper()} {descriptions[br]} ===\n"
            f"[eval {br}] coverage={cov}",
            flush=True,
        )
        metrics = evaluate_metrics(
            model_test,
            all_test_pn,
            all_test_label,
            all_test_id,
            all_test_user_pos,
            all_test_index,
            news_words,
            news_body,
            news_v,
            news_sv,
            int(args.batch_size),
            news_image=news_image_b0,
            title_only=False,
            cand_image=cand_image,
        )
        results[br] = {
            "description": descriptions[br],
            "metrics": metrics,
            "coverage": cov,
        }
        print(
            f"[{br}] MRR={metrics['MRR']:.6f}  NDCG@5={metrics['NDCG@5']:.6f}  "
            f"Hit@1={metrics['Hit@1']:.6f}",
            flush=True,
        )

    print("\n=== frozen teacher candidate swap ===", flush=True)
    header = f"{'branch':<6} {'MRR':>10} {'NDCG@5':>10} {'Hit@1':>10}"
    print(header, flush=True)
    for br in branches:
        m = results[br]["metrics"]
        print(
            f"{br.upper():<6} {m['MRR']:10.6f} {m['NDCG@5']:10.6f} {m['Hit@1']:10.6f}",
            flush=True,
        )

    out_path = (
        resolve_project_path(args.out)
        if args.out
        else str(
            _CLIP_DIR
            / "saved_models"
            / args.mind_dataset_subdir
            / f"frozen_teacher_{'_'.join(branches)}.json"
        )
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    payload = {
        "teacher_weights": os.path.abspath(weights_path),
        "tune_log": resolve_project_path(args.tune_log),
        "hparams": hp,
        "clip_dim": clip_dim,
        "max_history_clicks": int(naml_common.MAX_HISTORY_CLICKS),
        "seed": int(args.seed),
        "history_image": "B0",
        "text_views": ["title", "body", "category", "subcategory"],
        "caches": {
            "b0": os.path.abspath(b0_cache),
            "b1": os.path.abspath(caches["b1"]),
            "b2": os.path.abspath(caches["b2"]),
            "b3": os.path.abspath(caches["b3"]),
            "b4": os.path.abspath(caches["b4"]),
        },
        "expected_body_dir": expected_dir,
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[eval] saved {out_path}", flush=True)
    K.clear_session()


if __name__ == "__main__":
    main()
