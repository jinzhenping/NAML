#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
튜닝/학습된 기대이미지 NAML을 최종 test TSV로 평가.

기존 기대이미지 캐시(MIND_test_(2000).tsv / val)와 다른 pair라서
최종 test용 CLIP text → Δ 적용 캐시가 먼저 필요하다.

  conda activate clip_cu128
  python CLIP/extract_expected_body_text_embeds.py --split test_final --mind-dataset-subdir MIND_2000
  python CLIP/build_expected_image_embeds.py --apply-delta-only --mind-dataset-subdir MIND_2000

  conda activate tf28gpu
  python CLIP/eval_expected_image.py --mind-dataset-subdir MIND_2000 \
    --mind-test-tsv dataset/MIND_2000/MIND_test_2000_final.tsv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

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

from tensorflow.keras import backend as K

from clip_embeddings import (
    default_expected_image_test_final_path,
    default_test_final_tsv,
    resolve_project_path,
)
from expected_image import (
    build_test_candidate_image,
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
from train_expected_image import _max_history_from_log, _set_seed, evaluate_metrics
from train_s1_s2 import load_hparams


def main() -> None:
    ap = argparse.ArgumentParser(description="기대이미지 NAML 최종 test 평가")
    ap.add_argument("--mind-dataset-subdir", type=str, default="MIND_2000")
    ap.add_argument(
        "--mind-test-tsv",
        type=str,
        default=None,
        help="기본 dataset/<subdir>/MIND_test_2000_final.tsv",
    )
    ap.add_argument(
        "--weights",
        type=str,
        default="CLIP/saved_models/MIND_2000/naml_expected_image_tuned.h5",
    )
    ap.add_argument(
        "--tune-log",
        type=str,
        default="CLIP/saved_models/MIND_2000/naml_tune_expected_image_log.json",
    )
    ap.add_argument("--expected-image-cache", type=str, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--max-history-clicks", type=int, default=None)
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="결과 JSON. 기본 CLIP/saved_models/<subdir>/naml_expected_image_test_final.json",
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

    weights_path = resolve_project_path(args.weights)
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"가중치 없음: {weights_path}")

    test_tsv = (
        resolve_project_path(args.mind_test_tsv)
        if args.mind_test_tsv
        else default_test_final_tsv(args.mind_dataset_subdir)
    )
    if not os.path.isfile(test_tsv):
        raise FileNotFoundError(f"test TSV 없음: {test_tsv}")

    cache_path = (
        resolve_project_path(args.expected_image_cache)
        if args.expected_image_cache
        else default_expected_image_test_final_path(args.mind_dataset_subdir)
    )
    if not os.path.isfile(cache_path):
        raise FileNotFoundError(
            f"최종 test 기대이미지 cache 없음: {cache_path}\n"
            "conda activate clip_cu128\n"
            "python CLIP/extract_expected_body_text_embeds.py --split test_final "
            f"--mind-dataset-subdir {args.mind_dataset_subdir}\n"
            "python CLIP/build_expected_image_embeds.py --apply-delta-only "
            f"--mind-dataset-subdir {args.mind_dataset_subdir}"
        )

    print(
        f"[eval] weights={weights_path}\n"
        f"[eval] test_tsv={test_tsv}\n"
        f"[eval] expected-image cache={cache_path}",
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
        test_file=test_tsv,
        expected_bodies_train=None,
        expected_bodies_test=None,
        word_dict=word_dict,
    )
    embedding_mat = get_embedding(word_dict)

    pair_dict = load_pair_dict_normed(cache_path)
    clip_dim = pair_embed_dim(pair_dict)
    cand_img, cov = build_test_candidate_image(
        all_test_userid_str,
        all_test_newsid_str,
        pair_dict,
        clip_dim,
    )
    print(
        f"[eval] rows={len(all_test_id)} sessions={len(all_test_index)} "
        f"history={naml_common.MAX_HISTORY_CLICKS} clip_dim={clip_dim} coverage={cov}",
        flush=True,
    )

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
        cand_img,
        int(args.batch_size),
    )
    print(
        f"[eval] MRR={metrics['MRR']:.6f}  NDCG@5={metrics['NDCG@5']:.6f}  "
        f"Hit@1={metrics['Hit@1']:.6f}",
        flush=True,
    )

    out_path = (
        resolve_project_path(args.out)
        if args.out
        else str(
            _CLIP_DIR
            / "saved_models"
            / args.mind_dataset_subdir
            / "naml_expected_image_test_final.json"
        )
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    summary = {
        "variant": "expected_image",
        "split": "test_final",
        "weights": os.path.abspath(weights_path),
        "tune_log": os.path.abspath(resolve_project_path(args.tune_log)),
        "test_tsv": os.path.abspath(test_tsv),
        "expected_image_cache": os.path.abspath(cache_path),
        "hparams": hp,
        "max_history_clicks": int(naml_common.MAX_HISTORY_CLICKS),
        "n_rows": int(len(all_test_id)),
        "n_sessions": int(len(all_test_index)),
        "coverage": cov,
        "metrics": metrics,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[eval] log → {out_path}", flush=True)
    K.clear_session()


if __name__ == "__main__":
    main()
