#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NAML-IM: full-text NAML + IMRec visual impression (5번째 뷰).

프로토콜은 S2(full-text + 이미지 뷰)와 동일:
  - val(MIND_dev) MRR 최고 에폭 저장
  - held-out test(MIND_test_(2000).tsv) 자동 평가

  conda activate clip_cu128
  python IMRec/train_eval.py --stage prepare --mind-dataset-subdir MIND_2000
  python IMRec/train_eval.py --stage extract --mind-dataset-subdir MIND_2000

  conda activate tf28gpu
  python CLIP/train_naml_im.py --mind-dataset-subdir MIND_2000 \\
      --tune-log saved_models/MIND_2000/naml_tune_actual_log.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

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

import tensorflow as tf
from tensorflow.keras import backend as K

from clip_embeddings import resolve_project_path
from im_features import build_im_news_features, default_im_feature_path
from naml_common import MAX_HISTORY_CLICKS, SEED, get_embedding, mind_data_path, preprocess_news_file, preprocess_user_file
from naml_im_batch import generate_batch_data_test_im, generate_batch_data_train_im
from naml_im_model import build_naml_models_im
from naml_tune_actual import hit_at_k, mrr_score, ndcg_score
from train_s1_s2 import load_hparams


def _set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def evaluate_metrics_im(
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
    batch_size,
    im,
):
    n = len(all_test_id)
    steps = (n + batch_size - 1) // batch_size
    gen = generate_batch_data_test_im(
        all_test_pn,
        all_test_label,
        all_test_user_pos,
        news_words,
        news_body,
        news_v,
        news_sv,
        batch_size,
        im,
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


def train_naml_im(args) -> Dict[str, Any]:
    _set_seed(int(args.seed))
    hp = load_hparams(args.tune_log)
    arch_kw = dict(
        dropout_rate=hp["dropout_rate"],
        cnn_filters=hp["cnn_filters"],
        cnn_kernel_size=hp["cnn_kernel_size"],
        attention_dense_dim=hp["attention_dense_dim"],
        category_emb_dim=hp["category_emb_dim"],
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
        _tr_u,
        _tr_n,
        _te_u,
        _te_n,
    ) = preprocess_user_file(
        news_index=news_index,
        expected_bodies_train=None,
        expected_bodies_test=None,
        word_dict=word_dict,
    )
    embedding_mat = get_embedding(word_dict)

    im_cache = (
        resolve_project_path(args.im_cache)
        if args.im_cache
        else default_im_feature_path(args.mind_dataset_subdir)
    )
    im, n_hit = build_im_news_features(news_index, len(news_words), im_cache)
    print(
        f"[naml-im] train={len(all_train_id)} val_rows={len(all_test_id)} "
        f"IM nonzero={n_hit} cache={im_cache}",
        flush=True,
    )

    built = build_naml_models_im(
        word_dict,
        embedding_mat,
        category,
        subcategory,
        hp["learning_rate"],
        clear_session=True,
        **arch_kw,
    )
    model = built["model"]
    model_test = built["model_test"]

    out_dir = (
        resolve_project_path(args.out_dir)
        if args.out_dir
        else str(_CLIP_DIR / "saved_models" / args.mind_dataset_subdir)
    )
    os.makedirs(out_dir, exist_ok=True)
    out_weights = os.path.join(out_dir, "NAML_im_tuned.h5")
    out_log = os.path.join(out_dir, "naml_tune_naml_im_log.json")

    n_train = len(all_train_id)
    batch_size = int(args.batch_size)
    steps_per_epoch = (n_train + batch_size - 1) // batch_size
    epochs = int(args.epochs)
    best_mrr = -1.0
    best_metrics = None
    best_epoch = -1
    epoch_logs = []

    print(
        f"\n=== NAML-IM train  epochs={epochs}  batch={batch_size}  "
        f"text=title+body+cat/subcat  view=impression ===",
        flush=True,
    )
    for ep in range(1, epochs + 1):
        traingen = generate_batch_data_train_im(
            all_train_pn,
            all_label,
            all_user_pos,
            news_words,
            news_body,
            news_v,
            news_sv,
            batch_size,
            im,
        )
        hist = model.fit(traingen, epochs=1, steps_per_epoch=steps_per_epoch, verbose=1)
        metrics = evaluate_metrics_im(
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
            batch_size,
            im,
        )
        loss = float(hist.history["loss"][0]) if hist.history.get("loss") else None
        row = {"epoch": ep, "loss": loss, **metrics}
        epoch_logs.append(row)
        print(
            f"[naml-im ep {ep}/{epochs}] loss={loss}  "
            f"MRR={metrics['MRR']:.6f}  NDCG@5={metrics['NDCG@5']:.6f}  "
            f"Hit@1={metrics['Hit@1']:.6f}",
            flush=True,
        )
        if metrics["MRR"] > best_mrr:
            best_mrr = float(metrics["MRR"])
            best_metrics = dict(metrics)
            best_epoch = ep
            model.save_weights(out_weights)

    summary = {
        "variant": "naml_im",
        "text_mode": "full",
        "text_views": ["title", "body", "category", "subcategory"],
        "impression_view": True,
        "im_cache": os.path.abspath(im_cache),
        "hparams": hp,
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": int(args.seed),
        "max_history_clicks": int(MAX_HISTORY_CLICKS),
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "best_mrr": best_mrr,
        "epoch_logs": epoch_logs,
        "out_weights": os.path.abspath(out_weights),
    }
    with open(out_log, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[naml-im] weights → {out_weights}\n[naml-im] log → {out_log}", flush=True)

    if args.skip_final_test:
        K.clear_session()
        return summary

    test_tsv = mind_data_path(default_held_out_test_filename(args.mind_dataset_subdir))
    if not os.path.isfile(test_tsv):
        raise FileNotFoundError(f"held-out test TSV 없음: {test_tsv}")
    (
        _u,
        _tpn,
        _tl,
        _tid,
        test_pn,
        test_label,
        test_id,
        _up,
        test_user_pos,
        test_index,
        *_,
    ) = preprocess_user_file(
        news_index=news_index,
        test_file=test_tsv,
        expected_bodies_train=None,
        expected_bodies_test=None,
        word_dict=word_dict,
    )
    model.load_weights(out_weights)
    test_metrics = evaluate_metrics_im(
        model_test,
        test_pn,
        test_label,
        test_id,
        test_user_pos,
        test_index,
        news_words,
        news_body,
        news_v,
        news_sv,
        batch_size,
        im,
    )
    summary["test_tsv"] = os.path.abspath(test_tsv)
    summary["test_metrics"] = test_metrics
    test_json = os.path.join(out_dir, "naml_im_test.json")
    with open(test_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(
        f"[naml-im TEST] MRR={test_metrics['MRR']:.6f}  "
        f"NDCG@5={test_metrics['NDCG@5']:.6f}  Hit@1={test_metrics['Hit@1']:.6f}",
        flush=True,
    )
    K.clear_session()
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="NAML-IM 학습 (full-text + impression)")
    ap.add_argument("--mind-dataset-subdir", type=str, default="MIND_2000")
    ap.add_argument("--tune-log", type=str, default="saved_models/MIND_2000/naml_tune_actual_log.json")
    ap.add_argument("--epochs", type=int, default=30, help="S2 기본과 동일 30")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--max-history-clicks", type=int, default=None)
    ap.add_argument("--im-cache", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--skip-final-test", action="store_true")
    args = ap.parse_args()

    argv = ["--mind-dataset-subdir", args.mind_dataset_subdir]
    if args.max_history_clicks is not None:
        argv += ["--max-history-clicks", str(args.max_history_clicks)]
    apply_dataset_env_from_argv(argv)
    train_naml_im(args)


if __name__ == "__main__":
    main()
