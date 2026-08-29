#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
기대이미지 NAML 학습.

후보: title + cat + subcat + L2(CLIP_text(기대본문)+Δ)
히스토리: title + 실제본문 + cat + subcat (이미지 뷰 없음)

  python CLIP/build_expected_image_embeds.py --mind-dataset-subdir MIND_2000
  conda activate tf28gpu
  python CLIP/train_expected_image.py --mind-dataset-subdir MIND_2000
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    default_expected_image_test_path,
    default_expected_image_train_path,
    resolve_project_path,
)
from expected_image import (
    build_test_candidate_image,
    build_train_candidate_image,
    load_pair_dict_normed,
    pair_embed_dim,
)
from naml_common import (
    SEED,
    get_embedding,
    preprocess_news_file,
    preprocess_user_file,
    sync_max_history_clicks_from_env,
)
import naml_common
from naml_expected_image_model import build_naml_models_expected_image
from naml_tune_actual import hit_at_k, mrr_score, ndcg_score
from train_s1_s2 import _split_by_slot, load_hparams


def generate_batch_data_train(
    all_train_pn,
    all_label,
    all_user_pos,
    news_words,
    news_body,
    news_v,
    news_sv,
    cand_image,
    batch_size,
):
    n = len(all_label)
    inputid = np.arange(n)
    np.random.shuffle(inputid)
    batches = [
        inputid[range(batch_size * i, min(n, batch_size * (i + 1)))]
        for i in range((n + batch_size - 1) // batch_size)
        if batch_size * i < n
    ]
    while True:
        for idx in batches:
            cand_i = all_train_pn[idx]
            hist_i = all_user_pos[idx]
            parts = (
                _split_by_slot(news_words[cand_i])
                + _split_by_slot(news_words[hist_i])
                + _split_by_slot(news_body[hist_i])
                + _split_by_slot(news_v[cand_i])
                + _split_by_slot(news_v[hist_i])
                + _split_by_slot(news_sv[cand_i])
                + _split_by_slot(news_sv[hist_i])
                + _split_by_slot(cand_image[idx])
            )
            yield (parts, np.asarray(all_label[idx], dtype=np.float32))


def generate_batch_data_test(
    all_test_pn,
    all_test_label,
    all_test_user_pos,
    news_words,
    news_body,
    news_v,
    news_sv,
    cand_image,
    batch_size,
):
    n = len(all_test_label)
    inputid = np.arange(n)
    batches = [
        inputid[range(batch_size * i, min(n, batch_size * (i + 1)))]
        for i in range((n + batch_size - 1) // batch_size)
        if batch_size * i < n
    ]
    while True:
        for idx in batches:
            cand_i = all_test_pn[idx]
            hist_i = all_test_user_pos[idx]
            parts = (
                [news_words[cand_i]]
                + _split_by_slot(news_words[hist_i])
                + _split_by_slot(news_body[hist_i])
                + [news_v[cand_i]]
                + _split_by_slot(news_v[hist_i])
                + [news_sv[cand_i]]
                + _split_by_slot(news_sv[hist_i])
                + [cand_image[idx]]
            )
            yield (parts, np.asarray(all_test_label[idx], dtype=np.float32))


def evaluate_metrics(
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
    cand_image,
    batch_size,
):
    n = len(all_test_id)
    steps = (n + batch_size - 1) // batch_size
    gen = generate_batch_data_test(
        all_test_pn,
        all_test_label,
        all_test_user_pos,
        news_words,
        news_body,
        news_v,
        news_sv,
        cand_image,
        batch_size,
    )
    click_score = model_test.predict(gen, steps=steps, verbose=0)
    all_mrr, all_ndcg, all_hit1 = [], [], []
    for m in all_test_index:
        if np.sum(all_test_label[m[0] : m[1]]) == 0:
            continue
        if m[1] > len(click_score):
            continue
        session_scores = click_score[m[0] : m[1], 0]
        session_labels = all_test_label[m[0] : m[1]]
        all_mrr.append(mrr_score(session_labels, session_scores))
        all_ndcg.append(ndcg_score(session_labels, session_scores, k=5))
        all_hit1.append(hit_at_k(session_labels, session_scores, k=1))
    if not all_mrr:
        return {"MRR": 0.0, "NDCG@5": 0.0, "Hit@1": 0.0}
    return {
        "MRR": float(np.mean(all_mrr)),
        "NDCG@5": float(np.mean(all_ndcg)),
        "Hit@1": float(np.mean(all_hit1)),
    }


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


def main() -> None:
    ap = argparse.ArgumentParser(description="기대이미지 NAML 학습")
    ap.add_argument("--mind-dataset-subdir", type=str, default="MIND_2000")
    ap.add_argument(
        "--tune-log",
        type=str,
        default="CLIP/saved_models/MIND_2000/naml_tune_s2_clip_log.json",
    )
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--max-history-clicks", type=int, default=None)
    ap.add_argument("--train-expected-image-cache", type=str, default=None)
    ap.add_argument("--test-expected-image-cache", type=str, default=None)
    ap.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="가중치/로그 저장 폴더. 기본 CLIP/saved_models/<mind-dataset-subdir>",
    )
    args = ap.parse_args()

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

    train_cache = (
        resolve_project_path(args.train_expected_image_cache)
        if args.train_expected_image_cache
        else default_expected_image_train_path(args.mind_dataset_subdir)
    )
    test_cache = (
        resolve_project_path(args.test_expected_image_cache)
        if args.test_expected_image_cache
        else default_expected_image_test_path(args.mind_dataset_subdir)
    )
    for p, name in ((train_cache, "학습 기대이미지"), (test_cache, "테스트 기대이미지")):
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"{name} cache 없음: {p}\n"
                "먼저 python CLIP/build_expected_image_embeds.py 를 실행하세요."
            )

    print(
        "[train] 히스토리=title+실제본문+cat/subcat  후보=title+cat/subcat+기대이미지",
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
        all_train_pn,
        all_label,
        all_train_id,
        all_test_pn,
        all_test_label,
        all_test_id,
        all_user_pos,
        all_test_user_pos,
        all_test_index,
        _c1,
        _c2,
        all_train_userid_str,
        all_train_newsid_str,
        all_test_userid_str,
        all_test_newsid_str,
    ) = preprocess_user_file(
        news_index=news_index,
        expected_bodies_train=None,
        expected_bodies_test=None,
        word_dict=word_dict,
    )
    embedding_mat = get_embedding(word_dict)

    train_pairs = load_pair_dict_normed(train_cache)
    test_pairs = load_pair_dict_normed(test_cache)
    clip_dim = pair_embed_dim(train_pairs)
    test_dim = pair_embed_dim(test_pairs)
    if test_dim != clip_dim:
        raise ValueError(f"train/test 기대이미지 dim 불일치: {clip_dim} vs {test_dim}")

    train_cand_img, train_cov = build_train_candidate_image(
        all_train_userid_str,
        all_train_newsid_str,
        train_pairs,
        clip_dim,
        n_cand=int(all_train_pn.shape[1]),
    )
    test_cand_img, test_cov = build_test_candidate_image(
        all_test_userid_str,
        all_test_newsid_str,
        test_pairs,
        clip_dim,
    )
    print(
        f"[train] train={len(all_train_id)} val_rows={len(all_test_id)} "
        f"sessions={len(all_test_index)} history={naml_common.MAX_HISTORY_CLICKS} "
        f"clip_dim={clip_dim}\n"
        f"[train] expected-image train {train_cand_img.shape} {train_cov}\n"
        f"[train] expected-image test  {test_cand_img.shape} {test_cov}",
        flush=True,
    )
    if train_cov["n_nonzero"] == 0:
        raise ValueError("학습 기대이미지가 전부 0벡터입니다. Δ 캐시와 pair 키를 확인하세요.")

    built = build_naml_models_expected_image(
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

    if args.out_dir:
        out_dir = resolve_project_path(args.out_dir)
    else:
        out_dir = str(_CLIP_DIR / "saved_models" / args.mind_dataset_subdir)
    os.makedirs(out_dir, exist_ok=True)
    out_weights = os.path.join(out_dir, "naml_expected_image.h5")
    out_log = os.path.join(out_dir, "naml_expected_image_log.json")

    n_train = len(all_train_id)
    batch_size = int(args.batch_size)
    steps_per_epoch = (n_train + batch_size - 1) // batch_size
    epochs = int(args.epochs)
    best_mrr = -1.0
    best_weights = None
    best_metrics = None
    best_epoch = -1
    epoch_logs: List[Dict[str, Any]] = []

    print(
        f"\n=== expected-image train  epochs={epochs}  batch={batch_size}  "
        f"seed={args.seed} ===",
        flush=True,
    )
    for ep in range(1, epochs + 1):
        traingen = generate_batch_data_train(
            all_train_pn,
            all_label,
            all_user_pos,
            news_words,
            news_body,
            news_v,
            news_sv,
            train_cand_img,
            batch_size,
        )
        hist_fit = model.fit(traingen, epochs=1, steps_per_epoch=steps_per_epoch, verbose=1)
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
            test_cand_img,
            batch_size,
        )
        loss = None
        if hist_fit.history.get("loss"):
            loss = float(hist_fit.history["loss"][0])
        row = {"epoch": ep, "loss": loss, **metrics}
        epoch_logs.append(row)
        print(
            f"[expected-image ep {ep}/{epochs}] loss={loss}  "
            f"MRR={metrics['MRR']:.6f}  NDCG@5={metrics['NDCG@5']:.6f}  "
            f"Hit@1={metrics['Hit@1']:.6f}",
            flush=True,
        )
        if metrics["MRR"] > best_mrr:
            best_mrr = float(metrics["MRR"])
            best_metrics = dict(metrics)
            best_weights = model.get_weights()
            best_epoch = ep

    if best_weights is not None:
        model.set_weights(best_weights)
    if best_metrics is None:
        best_metrics = {"MRR": 0.0, "NDCG@5": 0.0, "Hit@1": 0.0}

    model.save_weights(out_weights)
    summary = {
        "variant": "expected_image",
        "hparams": hp,
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": int(args.seed),
        "max_history_clicks": int(naml_common.MAX_HISTORY_CLICKS),
        "clip_dim": int(clip_dim),
        "train_expected_image_cache": os.path.abspath(train_cache),
        "test_expected_image_cache": os.path.abspath(test_cache),
        "train_coverage": train_cov,
        "test_coverage": test_cov,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "epoch_logs": epoch_logs,
        "out_weights": os.path.abspath(out_weights),
        "candidate_views": ["title", "category", "subcategory", "expected_image"],
        "history_views": ["title", "actual_body", "category", "subcategory"],
    }
    with open(out_log, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(
        f"[expected-image] best epoch={best_epoch}  "
        f"MRR={best_metrics['MRR']:.6f}  NDCG@5={best_metrics['NDCG@5']:.6f}  "
        f"Hit@1={best_metrics['Hit@1']:.6f}\n"
        f"[expected-image] weights → {out_weights}\n"
        f"[expected-image] log → {out_log}",
        flush=True,
    )
    K.clear_session()


if __name__ == "__main__":
    main()
