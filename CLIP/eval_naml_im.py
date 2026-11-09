#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NAML-IM val / held-out test 평가 (S2 eval 프로토콜과 동일 split).

  conda activate tf28gpu
  python CLIP/eval_naml_im.py --split test --mind-dataset-subdir MIND_2000

Cross-eval (MIND_small 학습 ckpt → MIND_2000 held-out test 유저):

  # MIND_2000 IM feature가 없으면:
  conda activate clip_cu128
  python IMRec/train_eval.py --stage prepare --mind-dataset-subdir MIND_2000
  python IMRec/train_eval.py --stage extract --mind-dataset-subdir MIND_2000

  conda activate tf28gpu
  python CLIP/eval_naml_im.py --split test \\
      --mind-dataset-subdir MIND_2000 \\
      --ckpt-from-subdir MIND_small
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from nltk.tokenize import word_tokenize

_ROOT = Path(__file__).resolve().parent.parent
_CLIP_DIR = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "NAML") not in sys.path:
    sys.path.insert(0, str(_ROOT / "NAML"))
if str(_CLIP_DIR) not in sys.path:
    sys.path.insert(0, str(_CLIP_DIR))

from naml_dataset_env import (
    DATASET_FILE_PRESETS,
    apply_dataset_env_from_argv,
    default_held_out_test_filename,
)

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
    import tensorflow as tf

    np.random.seed(seed)
    tf.random.set_seed(seed)


def _switch_dataset(subdir: str, max_history: Optional[int] = None) -> None:
    """os.environ + naml_common 모듈 전역을 같은 subdir로 맞춤."""
    argv = ["--mind-dataset-subdir", subdir]
    if max_history is not None:
        argv += ["--max-history-clicks", str(max_history)]
    apply_dataset_env_from_argv(argv)
    naml_common.MIND_DATASET_SUBDIR = subdir
    if subdir in DATASET_FILE_PRESETS:
        n, tr, te = DATASET_FILE_PRESETS[subdir]
    else:
        n, tr, te = naml_common._resolve_mind_filenames()
    naml_common.MIND_NEWS_FILENAME = n
    naml_common.MIND_TRAIN_FILENAME = tr
    naml_common.MIND_TEST_FILENAME = te
    naml_common.sync_max_history_clicks_from_env()


def _news_tsv_for(subdir: str) -> str:
    news_name = DATASET_FILE_PRESETS.get(subdir, ("MIND_news.tsv", "", ""))[0]
    return str(_ROOT / "dataset" / subdir / news_name)


def encode_news_with_fixed_dicts(
    news_tsv: str,
    word_dict: dict,
    category: dict,
    subcategory: dict,
    *,
    title_len: int = 30,
    body_len: int = 300,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """학습 때 고정된 dict로 평가용 news.tsv를 인코딩 (OOV/unknown cat → 0)."""
    with open(news_tsv, "r", encoding="utf-8") as f:
        lines = f.readlines()

    news_words = [[0] * title_len]
    news_body = [[0] * body_len]
    news_v = [[0]]
    news_sv = [[0]]
    news_index: Dict[str, int] = {"0": 0}

    for newsline in lines:
        parts = newsline.strip().split("\t")
        if len(parts) < 5:
            continue
        news_id = parts[0]
        if str(news_id).strip().lower() in ("news_id", "clicked_news", "id"):
            continue
        cat = parts[1] if parts[1] else "None"
        subcat = parts[2] if parts[2] else "None"
        title = parts[3] if len(parts) > 3 else ""
        body = parts[4] if len(parts) > 4 else ""

        title_ids = []
        for w in word_tokenize(title.lower()) if title else []:
            if w in word_dict:
                title_ids.append(word_dict[w][0])
        title_ids = title_ids[:title_len]
        title_ids = title_ids + [0] * (title_len - len(title_ids))

        body_ids = []
        for w in word_tokenize(body.lower()) if body else []:
            if w in word_dict:
                body_ids.append(word_dict[w][0])
        body_ids = body_ids[:body_len]
        body_ids = body_ids + [0] * (body_len - len(body_ids))

        news_index[news_id] = len(news_index)
        news_words.append(title_ids)
        news_body.append(body_ids)
        news_v.append([int(category.get(cat, 0))])
        news_sv.append([int(subcategory.get(subcat, 0))])

    return (
        np.asarray(news_words, dtype="int32"),
        np.asarray(news_body, dtype="int32"),
        np.asarray(news_v, dtype="int32"),
        np.asarray(news_sv, dtype="int32"),
        news_index,
    )


def run_eval_naml_im(args) -> dict:
    eval_sub = args.mind_dataset_subdir
    ckpt_sub = getattr(args, "ckpt_from_subdir", None)
    out_dir = str(_CLIP_DIR / "saved_models" / eval_sub)
    ckpt_dir = str(_CLIP_DIR / "saved_models" / ckpt_sub) if ckpt_sub else out_dir

    weights_path = resolve_project_path(
        getattr(args, "weights", None) or os.path.join(ckpt_dir, "NAML_im_tuned.h5")
    )
    tune_log_path = resolve_project_path(
        getattr(args, "tune_log", None) or os.path.join(ckpt_dir, "naml_tune_naml_im_log.json")
    )
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"NAML-IM 가중치 없음: {weights_path}")

    mh = getattr(args, "max_history_clicks", None)
    if mh is None:
        mh = _max_history_from_log(tune_log_path)

    _set_seed(int(getattr(args, "seed", SEED)))
    hp = load_hparams(tune_log_path)
    arch_kw = dict(
        dropout_rate=hp["dropout_rate"],
        cnn_filters=hp["cnn_filters"],
        cnn_kernel_size=hp["cnn_kernel_size"],
        attention_dense_dim=hp["attention_dense_dim"],
        category_emb_dim=hp["category_emb_dim"],
    )

    vocab_sub = ckpt_sub or eval_sub
    _switch_dataset(vocab_sub, mh)
    # 모델 크기 = 학습(ckpt) 데이터 vocab/category
    word_dict, category, subcategory, *_rest = preprocess_news_file(
        expected_bodies_train=None,
        expected_bodies_test=None,
        expected_bodies_vocab_extra=None,
    )

    _switch_dataset(eval_sub, mh)
    if ckpt_sub and ckpt_sub != eval_sub:
        news_tsv = _news_tsv_for(eval_sub)
        if not os.path.isfile(news_tsv):
            raise FileNotFoundError(f"평가용 news TSV 없음: {news_tsv}")
        news_words, news_body, news_v, news_sv, news_index = encode_news_with_fixed_dicts(
            news_tsv, word_dict, category, subcategory
        )
        print(
            f"[eval NAML-IM] cross-eval vocab_from={ckpt_sub} eval={eval_sub} "
            f"news={len(news_index) - 1} word_dict={len(word_dict)}",
            flush=True,
        )
    else:
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
        test_tsv = str(_ROOT / "dataset" / eval_sub / default_held_out_test_filename(eval_sub))
        if not os.path.isfile(test_tsv):
            # fallback to mind_data_path after _switch_dataset(eval_sub)
            test_tsv = mind_data_path(default_held_out_test_filename(eval_sub))
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
        else default_im_feature_path(eval_sub)
    )
    im, n_hit = build_im_news_features(news_index, len(news_words), im_cache)
    print(
        f"[eval NAML-IM] split={split} rows={len(test_id)} IM hit={n_hit} "
        f"weights={weights_path}",
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
    if out_path is None and ckpt_sub and ckpt_sub != eval_sub:
        out_path = os.path.join(out_dir, f"naml_im_test_from_{ckpt_sub}.json")
    payload = {
        "variant": "naml_im",
        "split": split,
        "eval_subdir": eval_sub,
        "ckpt_from_subdir": ckpt_sub,
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
    ap.add_argument(
        "--ckpt-from-subdir",
        type=str,
        default=None,
        help="학습에 쓴 데이터셋 (예: MIND_small). 지정 시 그 쪽 weights/tune-log/vocab을 쓰고 "
        "--mind-dataset-subdir 의 test/IM으로 평가",
    )
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
