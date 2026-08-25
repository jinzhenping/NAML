#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
S1 / S2 NAML 학습 (실제 본문).

S1: 기존 NAML (title + body + cat/subcat)
S2: S1 + Kandinsky 2.2 CLIP 이미지 임베딩을 5번째 뷰로 view-attention

  # 1) (S2) 썸네일 CLIP 
  conda activate clip_cu128
  python CLIP/clip_embeddings.py --mind-dataset-subdir MIND_2000

  # 2) S1, S2 학습 (val = MIND_test_(2000).tsv)
  conda activate tf28gpu
  python CLIP/train_s1_s2.py --variant both --mind-dataset-subdir MIND_2000 \
    --tune-log saved_models/MIND_2000/naml_tune_actual_log.json
  # 가중치 기본 저장: CLIP/saved_models/MIND_2000/
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
    DEFAULT_THUMBNAIL_DIR,
    build_news_image_matrix,
    count_missing_thumbnails,
    default_cache_path,
    extract_clip_embeddings,
    load_news_ids_from_tsv,
    print_missing_thumbnail_report,
    resolve_project_path,
)
from naml_common import (
    MAX_HISTORY_CLICKS,
    MIND_NEWS_FILENAME,
    SEED,
    get_embedding,
    mind_data_path,
    npratio,
    preprocess_news_file,
    preprocess_user_file,
)
from naml_image_model import build_naml_models_with_image
from naml_model_builder import build_naml_models
from naml_tune_actual import evaluate_session_metrics, hit_at_k, mrr_score, ndcg_score

NCAND = 1 + npratio

_DEFAULT_HPARAMS = {
    "learning_rate": 0.0005,
    "dropout_rate": 0.3,
    "cnn_filters": 400,
    "cnn_kernel_size": 3,
    "attention_dense_dim": 200,
    "category_emb_dim": 50,
}
_HP_KEYS = (
    "learning_rate",
    "dropout_rate",
    "cnn_filters",
    "cnn_kernel_size",
    "attention_dense_dim",
    "category_emb_dim",
)


def _split_by_slot(arr: np.ndarray) -> List[np.ndarray]:
    """(B, K, ...) -> K arrays of (B, ...)."""
    return [arr[:, k] for k in range(arr.shape[1])]


def generate_batch_data_train(
    all_train_pn,
    all_label,
    all_user_pos,
    news_words,
    news_body,
    news_v,
    news_sv,
    batch_size,
    news_image=None,
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
                + _split_by_slot(news_body[cand_i])
                + _split_by_slot(news_body[hist_i])
                + _split_by_slot(news_v[cand_i])
                + _split_by_slot(news_v[hist_i])
                + _split_by_slot(news_sv[cand_i])
                + _split_by_slot(news_sv[hist_i])
            )
            if news_image is not None:
                parts = (
                    parts
                    + _split_by_slot(news_image[cand_i])
                    + _split_by_slot(news_image[hist_i])
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
    batch_size,
    news_image=None,
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
                + [news_body[cand_i]]
                + _split_by_slot(news_body[hist_i])
                + [news_v[cand_i]]
                + _split_by_slot(news_v[hist_i])
                + [news_sv[cand_i]]
                + _split_by_slot(news_sv[hist_i])
            )
            if news_image is not None:
                parts = parts + [news_image[cand_i]] + _split_by_slot(news_image[hist_i])
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
    batch_size,
    news_image=None,
):
    if news_image is None:
        return evaluate_session_metrics(
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
        )
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
        batch_size,
        news_image=news_image,
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


def load_hparams(tune_log: Optional[str]) -> Dict[str, Any]:
    hp = dict(_DEFAULT_HPARAMS)
    if not tune_log:
        print(f"[train] tune-log 미지정 → 기본 hparams {hp}", flush=True)
        return hp
    path = resolve_project_path(tune_log)
    if not os.path.isfile(path):
        print(f"[train] 경고: tune-log 없음 {path} → 기본 hparams {hp}", flush=True)
        return hp
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    gb = data.get("global_best_hparams")
    if not isinstance(gb, dict):
        print(f"[train] 경고: global_best_hparams 없음 {path} → 기본 hparams {hp}", flush=True)
        return hp
    for k in _HP_KEYS:
        if k not in gb:
            continue
        hp[k] = float(gb[k]) if k in ("learning_rate", "dropout_rate") else int(gb[k])
    print(f"[train] hparams from {path}: {hp}", flush=True)
    return hp


def _set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train_one(
    variant: str,
    hp: Dict[str, Any],
    epochs: int,
    batch_size: int,
    seed: int,
    word_dict,
    embedding_mat,
    category,
    subcategory,
    news_words,
    news_body,
    news_v,
    news_sv,
    all_train_pn,
    all_label,
    all_train_id,
    all_user_pos,
    all_test_pn,
    all_test_label,
    all_test_id,
    all_test_user_pos,
    all_test_index,
    news_image: Optional[np.ndarray],
    out_weights: str,
    out_log: str,
) -> Dict[str, Any]:
    _set_seed(seed)
    use_image = variant == "s2"
    if use_image:
        if news_image is None:
            raise ValueError("S2 학습에는 CLIP 임베딩 행렬이 필요합니다.")
        built = build_naml_models_with_image(
            word_dict,
            embedding_mat,
            category,
            subcategory,
            hp["learning_rate"],
            clip_dim=int(news_image.shape[1]),
            clear_session=True,
            dropout_rate=hp["dropout_rate"],
            cnn_filters=hp["cnn_filters"],
            cnn_kernel_size=hp["cnn_kernel_size"],
            attention_dense_dim=hp["attention_dense_dim"],
            category_emb_dim=hp["category_emb_dim"],
        )
    else:
        built = build_naml_models(
            word_dict,
            embedding_mat,
            category,
            subcategory,
            hp["learning_rate"],
            clear_session=True,
            dropout_rate=hp["dropout_rate"],
            cnn_filters=hp["cnn_filters"],
            cnn_kernel_size=hp["cnn_kernel_size"],
            attention_dense_dim=hp["attention_dense_dim"],
            category_emb_dim=hp["category_emb_dim"],
        )
    model = built["model"]
    model_test = built["model_test"]
    img = news_image if use_image else None

    n_train = len(all_train_id)
    steps_per_epoch = (n_train + batch_size - 1) // batch_size
    best_mrr = -1.0
    best_weights = None
    best_metrics = None
    best_epoch = -1
    epoch_logs: List[Dict[str, Any]] = []

    print(
        f"\n=== {variant.upper()} train  epochs={epochs}  batch={batch_size}  "
        f"seed={seed}  image_view={use_image} ===",
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
            batch_size,
            news_image=img,
        )
        hist = model.fit(traingen, epochs=1, steps_per_epoch=steps_per_epoch, verbose=1)
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
            batch_size,
            news_image=img,
        )
        loss = None
        if hist.history.get("loss"):
            loss = float(hist.history["loss"][0])
        row = {"epoch": ep, "loss": loss, **metrics}
        epoch_logs.append(row)
        print(
            f"[{variant} ep {ep}/{epochs}] loss={loss}  "
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

    os.makedirs(os.path.dirname(os.path.abspath(out_weights)) or ".", exist_ok=True)
    model.save_weights(out_weights)
    summary = {
        "variant": variant,
        "hparams": hp,
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "max_history_clicks": int(MAX_HISTORY_CLICKS),
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "epoch_logs": epoch_logs,
        "out_weights": os.path.abspath(out_weights),
        "use_image": use_image,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_log)) or ".", exist_ok=True)
    with open(out_log, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(
        f"[{variant}] best epoch={best_epoch}  "
        f"MRR={best_metrics['MRR']:.6f}  NDCG@5={best_metrics['NDCG@5']:.6f}  "
        f"Hit@1={best_metrics['Hit@1']:.6f}\n"
        f"[{variant}] weights → {out_weights}\n"
        f"[{variant}] log → {out_log}",
        flush=True,
    )
    K.clear_session()
    return summary


def ensure_clip_cache(
    cache_path: str,
    thumbnail_dir: str,
    news_tsv: str,
    device: str,
    extract_batch_size: int,
) -> str:
    if os.path.isfile(cache_path):
        print(f"[train] CLIP cache 사용: {cache_path}", flush=True)
        return cache_path
    print(f"[train] CLIP cache 없음 → 추출 시작: {cache_path}", flush=True)
    if not os.path.isdir(thumbnail_dir):
        raise FileNotFoundError(
            f"썸네일 폴더 없음: {thumbnail_dir}  (서버의 dataset/MIND_thumbnail 경로를 확인하세요)"
        )
    news_ids = load_news_ids_from_tsv(news_tsv)
    extract_clip_embeddings(
        news_ids,
        thumbnail_dir,
        cache_path,
        device=device,
        batch_size=extract_batch_size,
    )
    return cache_path


def main() -> None:
    ap = argparse.ArgumentParser(description="S1(text NAML) / S2(NAML+CLIP image view) 학습")
    ap.add_argument("--variant", type=str, default="both", choices=["s1", "s2", "both"])
    ap.add_argument("--mind-dataset-subdir", type=str, default="MIND_2000")
    ap.add_argument(
        "--tune-log",
        type=str,
        default="saved_models/MIND_2000/naml_tune_actual_log.json",
        help="global_best_hparams 를 읽을 실제본문 튜닝 로그. 없으면 NAML 기본값",
    )
    ap.add_argument("--epochs", type=int, default=10, help="S1 에폭 수 (기본 10). S2는 --epochs-s2")
    ap.add_argument("--epochs-s2", type=int, default=20, help="S2 에폭 수 (기본 20)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--max-history-clicks", type=int, default=None)
    ap.add_argument("--thumbnail-dir", type=str, default=DEFAULT_THUMBNAIL_DIR)
    ap.add_argument("--clip-cache", type=str, default=None, help="CLIP npz 경로. 기본 CLIP/cache/<subdir>_clip_image_embeds.npz")
    ap.add_argument("--clip-device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--clip-batch-size", type=int, default=16)
    ap.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="가중치/로그 저장 폴더. 기본 CLIP/saved_models/<mind-dataset-subdir>",
    )
    args = ap.parse_args()

    argv = ["--mind-dataset-subdir", args.mind_dataset_subdir]
    if args.max_history_clicks is not None:
        argv += ["--max-history-clicks", str(args.max_history_clicks)]
    apply_dataset_env_from_argv(argv)

    _set_seed(int(args.seed))
    hp = load_hparams(args.tune_log)

    print("[train] 실제 본문으로 전처리 (기대본문 미사용)", flush=True)
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
    print(
        f"[train] train={len(all_train_id)}  val(test rows)={len(all_test_id)}  "
        f"sessions={len(all_test_index)}  news={len(news_index)}  "
        f"history={MAX_HISTORY_CLICKS}",
        flush=True,
    )

    thumb_dir = resolve_project_path(args.thumbnail_dir)
    catalog_ids = [nid for nid, idx in news_index.items() if nid != "0" and int(idx) != 0]
    print_missing_thumbnail_report(catalog_ids, thumb_dir)

    news_image = None
    clip_cache = None
    need_s2 = args.variant in ("s2", "both")
    if need_s2:
        clip_cache = (
            resolve_project_path(args.clip_cache)
            if args.clip_cache
            else default_cache_path(args.mind_dataset_subdir)
        )
        news_tsv = mind_data_path(MIND_NEWS_FILENAME)
        ensure_clip_cache(
            clip_cache,
            thumb_dir,
            news_tsv,
            args.clip_device,
            args.clip_batch_size,
        )
        fallback_ids, _ = count_missing_thumbnails(load_news_ids_from_tsv(news_tsv), thumb_dir)
        news_image, n_hit = build_news_image_matrix(
            news_index,
            len(news_words),
            clip_cache,
            news_ids_fallback=fallback_ids,
        )
        print(
            f"[train] CLIP matrix {news_image.shape}  nonzero news={n_hit}/{len(catalog_ids)}",
            flush=True,
        )

    if args.out_dir:
        out_dir = resolve_project_path(args.out_dir)
    else:
        out_dir = str(_CLIP_DIR / "saved_models" / args.mind_dataset_subdir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[train] out_dir={out_dir}", flush=True)
    results = {}
    shared = dict(
        hp=hp,
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        word_dict=word_dict,
        embedding_mat=embedding_mat,
        category=category,
        subcategory=subcategory,
        news_words=news_words,
        news_body=news_body,
        news_v=news_v,
        news_sv=news_sv,
        all_train_pn=all_train_pn,
        all_label=all_label,
        all_train_id=all_train_id,
        all_user_pos=all_user_pos,
        all_test_pn=all_test_pn,
        all_test_label=all_test_label,
        all_test_id=all_test_id,
        all_test_user_pos=all_test_user_pos,
        all_test_index=all_test_index,
        news_image=news_image,
    )
    if args.variant in ("s1", "both"):
        results["s1"] = train_one(
            "s1",
            **shared,
            epochs=int(args.epochs),
            out_weights=os.path.join(out_dir, "S1_naml_actual.h5"),
            out_log=os.path.join(out_dir, "S1_naml_actual_log.json"),
        )
    if args.variant in ("s2", "both"):
        results["s2"] = train_one(
            "s2",
            **shared,
            epochs=int(args.epochs_s2),
            out_weights=os.path.join(out_dir, "S2_naml_clip.h5"),
            out_log=os.path.join(out_dir, "S2_naml_clip_log.json"),
        )

    if "s1" in results and "s2" in results:
        s1m = results["s1"]["best_metrics"]
        s2m = results["s2"]["best_metrics"]
        print("\n=== S1 vs S2 (best val MRR epoch) ===", flush=True)
        for k in ("MRR", "NDCG@5", "Hit@1"):
            d = float(s2m[k]) - float(s1m[k])
            print(
                f"  {k}: S1={s1m[k]:.6f}  S2={s2m[k]:.6f}  Δ(S2-S1)={d:+.6f}",
                flush=True,
            )
        cmp_path = os.path.join(out_dir, "S1_S2_compare.json")
        with open(cmp_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "s1": results["s1"]["best_metrics"],
                    "s2": results["s2"]["best_metrics"],
                    "s1_best_epoch": results["s1"]["best_epoch"],
                    "s2_best_epoch": results["s2"]["best_epoch"],
                    "hparams": hp,
                    "clip_cache": clip_cache,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[train] compare → {cmp_path}", flush=True)


if __name__ == "__main__":
    main()
