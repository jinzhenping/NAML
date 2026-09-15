#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
S3 (prior(actual_body) 이미지 뷰) test/val 평가.

  conda activate tf28gpu
  python CLIP/eval_s3.py --text-mode both --split test --mind-dataset-subdir MIND_2000
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
    build_news_image_matrix,
    default_actual_body_prior_cache_path,
    resolve_project_path,
)
from naml_common import (
    SEED,
    get_embedding,
    mind_data_path,
    preprocess_news_file,
    preprocess_user_file,
    sync_max_history_clicks_from_env,
)
import naml_common
from naml_image_model import build_naml_models_title_cat_image, build_naml_models_with_image
from train_s1_s2 import evaluate_metrics, load_hparams


def s3_artifact_names(text_mode: str) -> tuple:
    if text_mode == "title_cat":
        return "S3_naml_prior_title_cat_tuned.h5", "naml_tune_s3_prior_title_cat_log.json"
    return "S3_naml_prior_tuned.h5", "naml_tune_s3_prior_log.json"


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


def _eval_one(args, text_mode: str, data: dict) -> dict:
    out_dir = str(_CLIP_DIR / "saved_models" / args.mind_dataset_subdir)
    default_w, default_l = s3_artifact_names(text_mode)
    if text_mode == "title_cat":
        weights = getattr(args, "title_cat_weights", None) or os.path.join(out_dir, default_w)
        tune_log = getattr(args, "title_cat_tune_log", None) or os.path.join(out_dir, default_l)
    else:
        weights = getattr(args, "full_weights", None) or os.path.join(out_dir, default_w)
        tune_log = getattr(args, "full_tune_log", None) or os.path.join(out_dir, default_l)
    weights_path = resolve_project_path(weights)
    tune_log_path = resolve_project_path(tune_log)
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"S3 {text_mode} 가중치 없음: {weights_path}")
    hp = load_hparams(tune_log_path)
    arch_kw = dict(
        dropout_rate=hp["dropout_rate"],
        cnn_filters=hp["cnn_filters"],
        cnn_kernel_size=hp["cnn_kernel_size"],
        attention_dense_dim=hp["attention_dense_dim"],
        category_emb_dim=hp["category_emb_dim"],
    )
    news_image = data["news_image"]
    if news_image is None:
        raise ValueError("S3 평가에는 prior 이미지 행렬이 필요합니다.")
    if text_mode == "title_cat":
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
        built = build_naml_models_with_image(
            data["word_dict"],
            data["embedding_mat"],
            data["category"],
            data["subcategory"],
            hp["learning_rate"],
            clip_dim=int(news_image.shape[1]),
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
        text_mode=text_mode,
    )
    print(
        f"[s3 {text_mode} {args.split}] MRR={metrics['MRR']:.6f}  "
        f"NDCG@5={metrics['NDCG@5']:.6f}  Hit@1={metrics['Hit@1']:.6f}",
        flush=True,
    )
    K.clear_session()
    return {
        "variant": "s3",
        "text_mode": text_mode,
        "weights": os.path.abspath(weights_path),
        "tune_log": os.path.abspath(tune_log_path),
        "hparams": hp,
        "image_view": "prior_actual_body",
        "metrics": metrics,
    }


def run_eval_s3(args) -> dict:
    out_dir = str(_CLIP_DIR / "saved_models" / args.mind_dataset_subdir)
    argv = ["--mind-dataset-subdir", args.mind_dataset_subdir]
    hist = args.max_history_clicks
    if hist is None:
        log_candidates = []
        if getattr(args, "full_tune_log", None):
            log_candidates.append(args.full_tune_log)
        if getattr(args, "title_cat_tune_log", None):
            log_candidates.append(args.title_cat_tune_log)
        if args.text_mode in ("full", "both"):
            log_candidates.append(os.path.join(out_dir, s3_artifact_names("full")[1]))
        if args.text_mode in ("title_cat", "both"):
            log_candidates.append(os.path.join(out_dir, s3_artifact_names("title_cat")[1]))
        for p in log_candidates:
            hist = _max_history_from_log(p)
            if hist is not None:
                break
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
    prior_cache = (
        resolve_project_path(args.prior_cache)
        if getattr(args, "prior_cache", None)
        else default_actual_body_prior_cache_path(args.mind_dataset_subdir)
    )
    if not os.path.isfile(prior_cache):
        raise FileNotFoundError(
            f"actual-body prior cache 없음: {prior_cache}\n"
            "conda activate clip_cu128\n"
            "python CLIP/extract_actual_body_prior_embeds.py --scope catalog "
            f"--mind-dataset-subdir {args.mind_dataset_subdir}"
        )
    catalog_ids = [nid for nid, idx in news_index.items() if nid != "0" and int(idx) != 0]
    news_image, n_hit = build_news_image_matrix(news_index, len(news_words), prior_cache)
    print(
        f"[eval s3] prior cache={prior_cache} nonzero={n_hit}/{len(catalog_ids)}  "
        f"tsv={test_tsv or 'val'}",
        flush=True,
    )
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
    modes = ["full", "title_cat"] if args.text_mode == "both" else [args.text_mode]
    results = {}
    for mode in modes:
        results[mode] = _eval_one(args, mode, data)

    out_path = (
        resolve_project_path(args.out)
        if args.out
        else str(
            _CLIP_DIR
            / "saved_models"
            / args.mind_dataset_subdir
            / f"s3_prior_{args.split}_{'_'.join(modes)}.json"
        )
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    payload = {
        "variant": "s3",
        "image_view": "prior_actual_body",
        "prior_cache": os.path.abspath(prior_cache),
        "split": args.split,
        "test_tsv": os.path.abspath(test_tsv) if test_tsv else None,
        "max_history_clicks": int(naml_common.MAX_HISTORY_CLICKS),
        "n_nonzero_prior": int(n_hit),
        "n_catalog": int(len(catalog_ids)),
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[eval] saved {out_path}", flush=True)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="S3 prior-image NAML test/val 평가")
    ap.add_argument("--text-mode", type=str, default="both", choices=["full", "title_cat", "both"])
    ap.add_argument("--split", type=str, default="test", choices=["test", "val"])
    ap.add_argument("--mind-dataset-subdir", type=str, default="MIND_2000")
    ap.add_argument("--full-weights", type=str, default=None)
    ap.add_argument("--full-tune-log", type=str, default=None)
    ap.add_argument("--title-cat-weights", type=str, default=None)
    ap.add_argument("--title-cat-tune-log", type=str, default=None)
    ap.add_argument("--prior-cache", type=str, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--max-history-clicks", type=int, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    run_eval_s3(args)


if __name__ == "__main__":
    main()
