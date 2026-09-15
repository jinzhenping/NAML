#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
title+cat/subcat 로 튜닝된 S1/S2를 test(또는 val)에서 평가.

  conda activate tf28gpu
  python CLIP/eval_title_cat.py --variant both --split test --mind-dataset-subdir MIND_2000
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
_CLIP_DIR = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "NAML") not in sys.path:
    sys.path.insert(0, str(_ROOT / "NAML"))
if str(_CLIP_DIR) not in sys.path:
    sys.path.insert(0, str(_CLIP_DIR))

from naml_dataset_env import apply_dataset_env_from_argv, default_held_out_test_filename

apply_dataset_env_from_argv()

from tensorflow.keras import backend as K

from clip_embeddings import (
    DEFAULT_THUMBNAIL_DIR,
    build_news_image_matrix,
    count_missing_thumbnails,
    default_cache_path,
    load_news_ids_from_tsv,
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
from naml_image_model import build_naml_models_title_cat, build_naml_models_title_cat_image
from train_s1_s2 import evaluate_metrics, load_hparams


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


def _set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    import numpy as np
    import tensorflow as tf

    np.random.seed(seed)
    tf.random.set_seed(seed)


def _eval_one(args, variant: str, data: dict, test_tsv: Optional[str]) -> dict:
    use_image = variant == "s2"
    out_dir = str(_CLIP_DIR / "saved_models" / args.mind_dataset_subdir)
    if variant == "s1":
        weights = args.s1_weights or os.path.join(out_dir, "S1_naml_title_cat_tuned.h5")
        tune_log = args.s1_tune_log or os.path.join(out_dir, "naml_tune_s1_title_cat_log.json")
    else:
        weights = args.s2_weights or os.path.join(out_dir, "S2_naml_clip_title_cat_tuned.h5")
        tune_log = args.s2_tune_log or os.path.join(out_dir, "naml_tune_s2_title_cat_log.json")
    weights_path = resolve_project_path(weights)
    tune_log_path = resolve_project_path(tune_log)
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"{variant} 가중치 없음: {weights_path}")
    hp = load_hparams(tune_log_path)
    arch_kw = dict(
        dropout_rate=hp["dropout_rate"],
        cnn_filters=hp["cnn_filters"],
        cnn_kernel_size=hp["cnn_kernel_size"],
        attention_dense_dim=hp["attention_dense_dim"],
        category_emb_dim=hp["category_emb_dim"],
    )
    news_image = data["news_image"] if use_image else None
    if use_image:
        if news_image is None:
            raise ValueError("S2 평가에는 CLIP 캐시가 필요합니다.")
        built = build_naml_models_title_cat_image(
            data["word_dict"],
            data["embedding_mat"],
            data["category"],
            data["subcategory"],
            hp["learning_rate"],
            clip_dim=int(news_image.shape[1]),
            clear_session=True,
            **arch_kw,
        )
    else:
        built = build_naml_models_title_cat(
            data["word_dict"],
            data["embedding_mat"],
            data["category"],
            data["subcategory"],
            hp["learning_rate"],
            clear_session=True,
            **arch_kw,
        )
    model = built["model"]
    model_test = built["model_test"]
    model.load_weights(weights_path)
    metrics = evaluate_metrics(
        model_test,
        data["all_test_pn"],
        data["all_test_label"],
        data["all_test_id"],
        data["all_test_user_pos"],
        data["all_test_index"],
        data["news_words"],
        data["news_body"],
        data["news_v"],
        data["news_sv"],
        int(args.batch_size),
        news_image=news_image,
        text_mode="title_cat",
    )
    print(
        f"[{variant} {args.split}] MRR={metrics['MRR']:.6f}  "
        f"NDCG@5={metrics['NDCG@5']:.6f}  Hit@1={metrics['Hit@1']:.6f}",
        flush=True,
    )
    K.clear_session()
    return {
        "variant": variant,
        "weights": os.path.abspath(weights_path),
        "tune_log": os.path.abspath(tune_log_path),
        "hparams": hp,
        "image_view": use_image,
        "metrics": metrics,
    }


def run_eval_title_cat(args) -> dict:
    default_log = str(
        _CLIP_DIR / "saved_models" / args.mind_dataset_subdir / "naml_tune_s1_title_cat_log.json"
    )
    argv = ["--mind-dataset-subdir", args.mind_dataset_subdir]
    hist = args.max_history_clicks
    if hist is None:
        hist = _max_history_from_log(getattr(args, "s1_tune_log", None) or default_log)
    if hist is not None:
        argv += ["--max-history-clicks", str(hist)]
    apply_dataset_env_from_argv(argv)
    sync_max_history_clicks_from_env()
    _set_seed(int(args.seed))

    test_tsv = None
    if args.split == "test":
        test_tsv = mind_data_path(default_held_out_test_filename(args.mind_dataset_subdir))
        if not os.path.isfile(test_tsv):
            raise FileNotFoundError(f"held-out test TSV 없음: {test_tsv}")

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
        _te_u,
        _te_n,
    ) = preprocess_user_file(
        news_index=news_index,
        test_file=test_tsv,
        expected_bodies_train=None,
        expected_bodies_test=None,
        word_dict=word_dict,
    )
    embedding_mat = get_embedding(word_dict)
    news_image = None
    need_s2 = args.variant in ("s2", "both")
    if need_s2:
        clip_cache = (
            resolve_project_path(args.clip_cache)
            if args.clip_cache
            else default_cache_path(args.mind_dataset_subdir)
        )
        if not os.path.isfile(clip_cache):
            raise FileNotFoundError(
                f"B0 CLIP cache 없음: {clip_cache}\n"
                "python CLIP/clip_embeddings.py --mind-dataset-subdir "
                f"{args.mind_dataset_subdir}"
            )
        news_tsv = mind_data_path(MIND_NEWS_FILENAME)
        thumb_dir = resolve_project_path(args.thumbnail_dir)
        fallback_ids, _ = count_missing_thumbnails(load_news_ids_from_tsv(news_tsv), thumb_dir)
        news_image, n_hit = build_news_image_matrix(
            news_index, len(news_words), clip_cache, news_ids_fallback=fallback_ids
        )
        print(f"[eval title_cat] CLIP nonzero={n_hit}  tsv={test_tsv or 'val'}", flush=True)

    data = dict(
        word_dict=word_dict,
        embedding_mat=embedding_mat,
        category=category,
        subcategory=subcategory,
        news_words=news_words,
        news_body=news_body,
        news_v=news_v,
        news_sv=news_sv,
        news_image=news_image,
        all_test_pn=all_test_pn,
        all_test_label=all_test_label,
        all_test_id=all_test_id,
        all_test_user_pos=all_test_user_pos,
        all_test_index=all_test_index,
    )
    variants = ["s1", "s2"] if args.variant == "both" else [args.variant]
    results = {}
    for v in variants:
        results[v] = _eval_one(args, v, data, test_tsv)

    out_path = (
        resolve_project_path(args.out)
        if args.out
        else str(
            _CLIP_DIR
            / "saved_models"
            / args.mind_dataset_subdir
            / f"title_cat_{args.split}_{'_'.join(variants)}.json"
        )
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    payload = {
        "text_mode": "title_cat",
        "text_views": ["title", "category", "subcategory"],
        "split": args.split,
        "test_tsv": os.path.abspath(test_tsv) if test_tsv else None,
        "max_history_clicks": int(naml_common.MAX_HISTORY_CLICKS),
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[eval] saved {out_path}", flush=True)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="title+cat/subcat S1/S2 test/val 평가")
    ap.add_argument("--variant", type=str, default="both", choices=["s1", "s2", "both"])
    ap.add_argument("--split", type=str, default="test", choices=["test", "val"])
    ap.add_argument("--mind-dataset-subdir", type=str, default="MIND_2000")
    ap.add_argument("--s1-weights", type=str, default=None)
    ap.add_argument("--s1-tune-log", type=str, default=None)
    ap.add_argument("--s2-weights", type=str, default=None)
    ap.add_argument("--s2-tune-log", type=str, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--max-history-clicks", type=int, default=None)
    ap.add_argument("--thumbnail-dir", type=str, default=DEFAULT_THUMBNAIL_DIR)
    ap.add_argument("--clip-cache", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    run_eval_title_cat(args)


if __name__ == "__main__":
    main()
