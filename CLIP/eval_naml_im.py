#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NAML-IM val / held-out test 평가 (S2 eval 프로토콜과 동일 split).

  conda activate tf28gpu
  python CLIP/eval_naml_im.py --split test --mind-dataset-subdir MIND_2000
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

from clip_embeddings import resolve_project_path
from im_features import build_im_news_features, default_im_feature_path
from naml_common import SEED, get_embedding, mind_data_path, preprocess_news_file, preprocess_user_file
import naml_common
from naml_im_model import build_naml_models_im
from train_naml_im import evaluate_metrics_im
from train_s1_s2 import load_hparams


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


def run_eval_naml_im(args) -> dict:
    out_dir = str(_CLIP_DIR / "saved_models" / args.mind_dataset_subdir)
    weights_path = resolve_project_path(
        getattr(args, "weights", None) or os.path.join(out_dir, "NAML_im_tuned.h5")
    )
    tune_log_path = resolve_project_path(
        getattr(args, "tune_log", None) or os.path.join(out_dir, "naml_tune_naml_im_log.json")
    )
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"NAML-IM 가중치 없음: {weights_path}")

    mh = getattr(args, "max_history_clicks", None)
    if mh is None:
        mh = _max_history_from_log(tune_log_path)
    argv = ["--mind-dataset-subdir", args.mind_dataset_subdir]
    if mh is not None:
        argv += ["--max-history-clicks", str(mh)]
    apply_dataset_env_from_argv(argv)
    naml_common.sync_max_history_clicks_from_env()

    _set_seed(int(getattr(args, "seed", SEED)))
    hp = load_hparams(tune_log_path)
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
    split = getattr(args, "split", "test")
    test_tsv = None
    if split == "test":
        test_tsv = mind_data_path(default_held_out_test_filename(args.mind_dataset_subdir))
        if not os.path.isfile(test_tsv):
            raise FileNotFoundError(f"held-out test TSV 없음: {test_tsv}")

    (
        _userid_dict,
        _all_train_pn,
        _all_label,
        _all_train_id,
        test_pn,
        test_label,
        test_id,
        _all_user_pos,
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
    embedding_mat = get_embedding(word_dict)
    im_cache = (
        resolve_project_path(args.im_cache)
        if getattr(args, "im_cache", None)
        else default_im_feature_path(args.mind_dataset_subdir)
    )
    im, n_hit = build_im_news_features(news_index, len(news_words), im_cache)
    print(f"[eval NAML-IM] split={split} rows={len(test_id)} IM hit={n_hit}", flush=True)

    built = build_naml_models_im(
        word_dict,
        embedding_mat,
        category,
        subcategory,
        hp["learning_rate"],
        clear_session=True,
        **arch_kw,
    )
    model_test = built["model_test"]
    model_test.load_weights(weights_path)
    metrics = evaluate_metrics_im(
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
        int(args.batch_size),
        im,
    )
    print(
        f"[NAML-IM {split}] MRR={metrics['MRR']:.6f}  "
        f"NDCG@5={metrics['NDCG@5']:.6f}  Hit@1={metrics['Hit@1']:.6f}",
        flush=True,
    )
    K.clear_session()

    out_path = getattr(args, "out", None)
    payload = {
        "variant": "naml_im",
        "split": split,
        "weights": os.path.abspath(weights_path),
        "tune_log": os.path.abspath(tune_log_path),
        "im_cache": os.path.abspath(im_cache),
        "test_tsv": os.path.abspath(test_tsv) if test_tsv else None,
        "max_history_clicks": int(naml_common.MAX_HISTORY_CLICKS),
        "n_im_nonzero": int(n_hit),
        "hparams": hp,
        "metrics": metrics,
    }
    if out_path:
        out_path = resolve_project_path(out_path)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[eval NAML-IM] saved {out_path}", flush=True)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="NAML-IM 평가")
    ap.add_argument("--split", type=str, default="test", choices=["val", "test"])
    ap.add_argument("--mind-dataset-subdir", type=str, default="MIND_2000")
    ap.add_argument("--weights", type=str, default=None)
    ap.add_argument("--tune-log", type=str, default=None)
    ap.add_argument("--im-cache", type=str, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--max-history-clicks", type=int, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    run_eval_naml_im(args)


if __name__ == "__main__":
    main()
