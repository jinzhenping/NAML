#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
S3' test/val 평가.

텍스트: title + category + subcategory (본문 없음)
히스토리 이미지: prior(actual_body)  (S3와 동일, 뉴스 단위)
후보 이미지: prior(title + preference_profile)  (유저×뉴스)

  conda activate tf28gpu
  python CLIP/eval_s3_prime.py --split test --mind-dataset-subdir MIND_2000
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
    default_s3p_cache_path,
    resolve_project_path,
)
from expected_image import build_test_candidate_image, load_pair_dict_normed, pair_embed_dim
from naml_common import (
    SEED,
    get_embedding,
    mind_data_path,
    preprocess_news_file,
    preprocess_user_file,
    sync_max_history_clicks_from_env,
)
import naml_common
from naml_image_model import build_naml_models_title_cat_image
from train_s1_s2 import evaluate_metrics, load_hparams

S3P_WEIGHTS_NAME = "S3p_naml_summary_title_tuned.h5"
S3P_LOG_NAME = "naml_tune_s3p_summary_title_log.json"


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


def run_eval_s3_prime(args) -> dict:
    out_dir = str(_CLIP_DIR / "saved_models" / args.mind_dataset_subdir)
    weights = getattr(args, "weights", None) or os.path.join(out_dir, S3P_WEIGHTS_NAME)
    tune_log = getattr(args, "tune_log", None) or os.path.join(out_dir, S3P_LOG_NAME)
    argv = ["--mind-dataset-subdir", args.mind_dataset_subdir]
    hist = args.max_history_clicks
    if hist is None:
        hist = _max_history_from_log(tune_log)
    if hist is not None:
        argv += ["--max-history-clicks", str(hist)]
    apply_dataset_env_from_argv(argv)
    sync_max_history_clicks_from_env()
    _set_seed(int(args.seed))

    test_tsv = None
    s3p_split = "val"
    if args.split == "test":
        test_tsv = mind_data_path(default_held_out_test_filename(args.mind_dataset_subdir))
        if not os.path.isfile(test_tsv):
            raise FileNotFoundError(f"held-out test TSV 없음: {test_tsv}")
        s3p_split = "test"

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
        test_file=test_tsv,
        expected_bodies_train=None,
        expected_bodies_test=None,
        word_dict=word_dict,
    )
    embedding_mat = get_embedding(word_dict)

    hist_cache = (
        resolve_project_path(args.prior_cache)
        if getattr(args, "prior_cache", None)
        else default_actual_body_prior_cache_path(args.mind_dataset_subdir)
    )
    cand_cache = (
        resolve_project_path(args.s3p_cache)
        if getattr(args, "s3p_cache", None)
        else default_s3p_cache_path(args.mind_dataset_subdir, s3p_split)
    )
    weights_path = resolve_project_path(weights)
    tune_log_path = resolve_project_path(tune_log)
    if not os.path.isfile(hist_cache):
        raise FileNotFoundError(
            f"actual-body prior cache 없음: {hist_cache}\n"
            "python CLIP/extract_actual_body_prior_embeds.py --scope catalog "
            f"--mind-dataset-subdir {args.mind_dataset_subdir}"
        )
    if not os.path.isfile(cand_cache):
        raise FileNotFoundError(
            f"S3' candidate prior cache 없음: {cand_cache}\n"
            "python CLIP/extract_s3_prime_prior_embeds.py "
            f"--split {s3p_split} --mind-dataset-subdir {args.mind_dataset_subdir}"
        )
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"S3' 가중치 없음: {weights_path}")

    news_image, n_hist = build_news_image_matrix(news_index, len(news_words), hist_cache)
    pair_dict = load_pair_dict_normed(cand_cache)
    clip_dim = int(news_image.shape[1])
    pair_dim = pair_embed_dim(pair_dict)
    if pair_dim != clip_dim:
        raise ValueError(f"S3' cand dim={pair_dim} != history prior dim={clip_dim}")
    cand_image, cov = build_test_candidate_image(
        all_test_userid_str, all_test_newsid_str, pair_dict, clip_dim
    )
    print(
        f"[eval s3p] hist prior={hist_cache} nonzero={n_hist}\n"
        f"[eval s3p] cand prior={cand_cache} {cov}  tsv={test_tsv or 'val'}",
        flush=True,
    )

    hp = load_hparams(tune_log_path)
    built = build_naml_models_title_cat_image(
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
        news_image=news_image,
        cand_image=cand_image,
        text_mode="title_cat",
    )
    print(
        f"[s3p {args.split}] MRR={metrics['MRR']:.6f}  "
        f"NDCG@5={metrics['NDCG@5']:.6f}  Hit@1={metrics['Hit@1']:.6f}",
        flush=True,
    )
    K.clear_session()

    out_path = (
        resolve_project_path(args.out)
        if args.out
        else str(_CLIP_DIR / "saved_models" / args.mind_dataset_subdir / f"s3p_prior_{args.split}.json")
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    payload = {
        "variant": "s3p",
        "text_mode": "title_cat",
        "text_views": ["title", "category", "subcategory"],
        "history_image": "prior_actual_body",
        "candidate_image": "prior_summary_title",
        "prior_cache": os.path.abspath(hist_cache),
        "s3p_cache": os.path.abspath(cand_cache),
        "weights": os.path.abspath(weights_path),
        "tune_log": os.path.abspath(tune_log_path),
        "hparams": hp,
        "split": args.split,
        "test_tsv": os.path.abspath(test_tsv) if test_tsv else None,
        "max_history_clicks": int(naml_common.MAX_HISTORY_CLICKS),
        "history_nonzero": int(n_hist),
        "candidate_coverage": cov,
        "metrics": metrics,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[eval] saved {out_path}", flush=True)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="S3' summary+title candidate prior 평가")
    ap.add_argument("--split", type=str, default="test", choices=["test", "val"])
    ap.add_argument("--mind-dataset-subdir", type=str, default="MIND_2000")
    ap.add_argument("--weights", type=str, default=None)
    ap.add_argument("--tune-log", type=str, default=None)
    ap.add_argument("--prior-cache", type=str, default=None)
    ap.add_argument("--s3p-cache", type=str, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--max-history-clicks", type=int, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    run_eval_s3_prime(args)


if __name__ == "__main__":
    main()
