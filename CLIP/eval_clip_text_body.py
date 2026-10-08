#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NAML: title + category + subcategory + CLIP_text(caption(actual_body)).

caption(·) = 본문 문자열을 CLIP text encoder에 넣는 것 (77토큰 truncation).
임베딩 캐시: CLIP/cache/<subdir>_clip_text_actual_body_train.npz

  conda activate tf28gpu
  python CLIP/eval_clip_text_body.py --split test --mind-dataset-subdir MIND_2000
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
    default_actual_body_text_cache_path,
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
from naml_image_model import build_naml_models_title_cat_image
from train_s1_s2 import evaluate_metrics, load_hparams

WEIGHTS_NAME = "S5_naml_clip_text_actual_title_cat_tuned.h5"
LOG_NAME = "naml_tune_s5_clip_text_actual_log.json"


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


def run_eval_clip_text_body(args) -> dict:
    out_dir = str(_CLIP_DIR / "saved_models" / args.mind_dataset_subdir)
    weights = getattr(args, "weights", None) or os.path.join(out_dir, WEIGHTS_NAME)
    tune_log = getattr(args, "tune_log", None) or os.path.join(out_dir, LOG_NAME)
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
        *_,
    ) = preprocess_user_file(
        news_index=news_index,
        test_file=test_tsv,
        expected_bodies_train=None,
        expected_bodies_test=None,
        word_dict=word_dict,
    )
    embedding_mat = get_embedding(word_dict)

    text_cache = (
        resolve_project_path(args.text_cache)
        if getattr(args, "text_cache", None)
        else default_actual_body_text_cache_path(args.mind_dataset_subdir)
    )
    weights_path = resolve_project_path(weights)
    tune_log_path = resolve_project_path(tune_log)
    if not os.path.isfile(text_cache):
        raise FileNotFoundError(
            f"CLIP_text(actual_body) cache 없음: {text_cache}\n"
            "conda activate clip_cu128\n"
            "python CLIP/extract_actual_body_text_embeds.py --scope catalog "
            f"--mind-dataset-subdir {args.mind_dataset_subdir}"
        )
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"가중치 없음: {weights_path}")

    news_embed, n_hit = build_news_image_matrix(news_index, len(news_words), text_cache)
    catalog_n = sum(1 for nid, idx in news_index.items() if nid != "0" and int(idx) != 0)
    print(
        f"[eval s5] CLIP_text(actual_body)={text_cache} nonzero={n_hit}/{catalog_n}  "
        f"tsv={test_tsv or 'val'}",
        flush=True,
    )

    hp = load_hparams(tune_log_path)
    built = build_naml_models_title_cat_image(
        word_dict,
        embedding_mat,
        category,
        subcategory,
        hp["learning_rate"],
        clip_dim=int(news_embed.shape[1]),
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
        news_image=news_embed,
        text_mode="title_cat",
    )
    print(
        f"[s5 {args.split}] MRR={metrics['MRR']:.6f}  "
        f"NDCG@5={metrics['NDCG@5']:.6f}  Hit@1={metrics['Hit@1']:.6f}",
        flush=True,
    )
    K.clear_session()

    out_path = (
        resolve_project_path(args.out)
        if args.out
        else str(
            _CLIP_DIR
            / "saved_models"
            / args.mind_dataset_subdir
            / f"s5_clip_text_actual_{args.split}.json"
        )
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    payload = {
        "variant": "s5",
        "text_mode": "title_cat",
        "text_views": ["title", "category", "subcategory"],
        "embed_view": "clip_text_caption_actual_body",
        "caption_note": "caption(x)=CLIP text input of body string (77-token trunc), not a separate captioner",
        "text_cache": os.path.abspath(text_cache),
        "weights": os.path.abspath(weights_path),
        "tune_log": os.path.abspath(tune_log_path),
        "hparams": hp,
        "split": args.split,
        "test_tsv": os.path.abspath(test_tsv) if test_tsv else None,
        "max_history_clicks": int(naml_common.MAX_HISTORY_CLICKS),
        "n_nonzero_embed": int(n_hit),
        "n_catalog": int(catalog_n),
        "metrics": metrics,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[eval] saved {out_path}", flush=True)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="S5 CLIP_text(actual_body) + title_cat NAML 평가")
    ap.add_argument("--split", type=str, default="test", choices=["test", "val"])
    ap.add_argument("--mind-dataset-subdir", type=str, default="MIND_2000")
    ap.add_argument("--weights", type=str, default=None)
    ap.add_argument("--tune-log", type=str, default=None)
    ap.add_argument("--text-cache", type=str, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--max-history-clicks", type=int, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    run_eval_clip_text_body(args)


if __name__ == "__main__":
    main()
