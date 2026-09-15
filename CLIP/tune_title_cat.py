#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
title + category + subcategory NAML 튜닝 (본문 없음).

S1: 텍스트 3뷰, 썸네일 없음
S2: 같은 텍스트 3뷰 + CLIP 썸네일 4번째 뷰

val MRR 최고 조합을 저장한 뒤, 같은 프로세스가 held-out test를 평가한다.

  conda activate tf28gpu
  python CLIP/tune_title_cat.py --variant both --two-phase --trials 108 \
    --screening-epochs 3 --refine-top-k 10 --epochs-per-trial 10 \
    --mind-dataset-subdir MIND_2000

test 평가만 생략하려면 --skip-test-eval.
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
    preprocess_news_file,
    preprocess_user_file,
)
from naml_image_model import build_naml_models_title_cat, build_naml_models_title_cat_image
from naml_tune_actual import (
    HPARAM_CHOICES,
    _hparam_grid_size,
    _hp_key,
    _load_json_or_none,
    _load_previous_best_from_log,
    _load_seen_hparam_keys_from_log,
    plan_hparam_trials,
    sample_hparams,
)
from train_s1_s2 import ensure_clip_cache, evaluate_metrics, generate_batch_data_train


def run_trial(
    hp: dict,
    epochs: int,
    batch_size: int,
    use_image: bool,
    word_dict,
    embedding_mat,
    category,
    subcategory,
    news_words,
    news_body,
    news_v,
    news_sv,
    news_image,
    all_train_pn,
    all_label,
    all_train_id,
    all_user_pos,
    all_test_pn,
    all_test_label,
    all_test_id,
    all_test_user_pos,
    all_test_index,
    trial_seed: int,
):
    np.random.seed(trial_seed)
    random.seed(trial_seed)
    tf.random.set_seed(trial_seed)
    arch_kw = dict(
        dropout_rate=hp["dropout_rate"],
        cnn_filters=hp["cnn_filters"],
        cnn_kernel_size=hp["cnn_kernel_size"],
        attention_dense_dim=hp["attention_dense_dim"],
        category_emb_dim=hp["category_emb_dim"],
    )
    if use_image:
        if news_image is None:
            raise ValueError("S2 title_cat 튜닝에는 CLIP 행렬이 필요합니다.")
        built = build_naml_models_title_cat_image(
            word_dict,
            embedding_mat,
            category,
            subcategory,
            hp["learning_rate"],
            clip_dim=int(news_image.shape[1]),
            clear_session=True,
            **arch_kw,
        )
        img = news_image
    else:
        built = build_naml_models_title_cat(
            word_dict,
            embedding_mat,
            category,
            subcategory,
            hp["learning_rate"],
            clear_session=True,
            **arch_kw,
        )
        img = None
    model = built["model"]
    model_test = built["model_test"]
    n_train = len(all_train_id)
    steps_per_epoch = (n_train + batch_size - 1) // batch_size
    best_mrr = -1.0
    best_weights = None
    best_metrics = None
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
            text_mode="title_cat",
        )
        hist = model.fit(traingen, epochs=1, steps_per_epoch=steps_per_epoch, verbose=0)
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
            text_mode="title_cat",
        )
        loss = float(hist.history["loss"][0]) if hist.history.get("loss") else None
        print(
            f"    ep {ep}/{epochs} loss={loss}  "
            f"MRR={metrics['MRR']:.6f}  NDCG@5={metrics['NDCG@5']:.6f}  "
            f"Hit@1={metrics['Hit@1']:.6f}",
            flush=True,
        )
        if metrics["MRR"] > best_mrr:
            best_mrr = float(metrics["MRR"])
            best_metrics = dict(metrics)
            best_weights = model.get_weights()
    if best_weights is not None:
        model.set_weights(best_weights)
    if best_metrics is None:
        best_metrics = {"MRR": 0.0, "NDCG@5": 0.0, "Hit@1": 0.0}
    return best_mrr, best_metrics, model


def _tune_one(args, variant: str, data: dict) -> None:
    use_image = variant == "s2"
    out_dir = str(_CLIP_DIR / "saved_models" / args.mind_dataset_subdir)
    os.makedirs(out_dir, exist_ok=True)
    if variant == "s1":
        default_w, default_l = "S1_naml_title_cat_tuned.h5", "naml_tune_s1_title_cat_log.json"
    else:
        default_w, default_l = "S2_naml_clip_title_cat_tuned.h5", "naml_tune_s2_title_cat_log.json"
    out_weights = (
        resolve_project_path(args.out_weights)
        if args.out_weights and args.variant != "both"
        else os.path.join(out_dir, default_w)
    )
    out_log = (
        resolve_project_path(args.out_log)
        if args.out_log and args.variant != "both"
        else os.path.join(out_dir, default_l)
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_weights)) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_log)) or ".", exist_ok=True)
    print(
        f"[tune {variant.upper()} title_cat] text=title+cat/subcat  image={use_image}  "
        f"val=MIND_dev  weights={out_weights}",
        flush=True,
    )

    trial_kw = dict(data)
    trial_kw["use_image"] = use_image
    if not use_image:
        trial_kw["news_image"] = None

    rng = random.Random(args.seed)
    global_best_mrr = -1.0
    global_best_hp: Optional[Dict[str, Any]] = None
    log_trials: List[dict] = []
    grid_n = _hparam_grid_size()
    seen_hparam_keys: set = set()
    resume_log_path: Optional[str] = None
    if args.resume_log and args.variant != "both":
        resume_log_path = resolve_project_path(args.resume_log)
        if os.path.isfile(resume_log_path):
            seen_hparam_keys = _load_seen_hparam_keys_from_log(resume_log_path)
            prev_best_mrr, prev_best_hp = _load_previous_best_from_log(resume_log_path)
            if prev_best_mrr > global_best_mrr:
                global_best_mrr = prev_best_mrr
                global_best_hp = prev_best_hp

    if args.allow_duplicate_hparams:
        trial_hparams = []
        local_keys: set = set()
        max_attempts = max(args.trials * 300, 3000)
        attempts = 0
        while len(trial_hparams) < args.trials and attempts < max_attempts:
            attempts += 1
            hp = sample_hparams(rng)
            k = _hp_key(hp)
            if k in seen_hparam_keys or k in local_keys:
                continue
            trial_hparams.append(hp)
            local_keys.add(k)
    else:
        all_planned = plan_hparam_trials(rng, grid_n)
        trial_hparams = [hp for hp in all_planned if _hp_key(hp) not in seen_hparam_keys][: args.trials]
        if len(trial_hparams) < args.trials:
            print(
                f"경고: unseen 고유 조합이 부족하여 요청 {args.trials}개 중 {len(trial_hparams)}개만 실행합니다."
            )
    run_trials = len(trial_hparams)
    if run_trials == 0:
        print("실행할 새 조합이 없습니다.")
        return

    def _one_trial(trial_idx: int, total: int, hp: dict, epochs: int, trial_seed: int, phase: str) -> None:
        nonlocal global_best_mrr, global_best_hp
        print(f"\n--- [{variant} {phase}] {trial_idx + 1}/{total}  hparams={hp} ---", flush=True)
        best_mrr, best_metrics, model = run_trial(
            hp=hp, epochs=epochs, trial_seed=trial_seed, batch_size=args.batch_size, **trial_kw
        )
        print(
            f"  trial best MRR: {best_mrr:.6f}  | "
            f"MRR={best_metrics['MRR']:.6f} NDCG@5={best_metrics['NDCG@5']:.6f} "
            f"Hit@1={best_metrics['Hit@1']:.6f}",
            flush=True,
        )
        log_trials.append(
            {
                "phase": phase,
                "hparams": hp,
                "epochs_in_phase": epochs,
                "best_mrr_in_trial": best_mrr,
                "best_epoch": best_metrics,
            }
        )
        if best_mrr > global_best_mrr:
            global_best_mrr = best_mrr
            global_best_hp = dict(hp)
            model.save_weights(out_weights)
            print(f"  [전역 갱신] 저장 → {out_weights}  MRR={global_best_mrr:.6f}", flush=True)
        K.clear_session()

    if args.two_phase:
        k = min(args.refine_top_k, run_trials)
        print(
            f"\n[2-phase] 1차: trials={run_trials}, epochs={args.screening_epochs} → "
            f"상위 {k}개를 2차에서 epochs={args.epochs_per_trial}",
            flush=True,
        )
        screening_rows: list = []
        for t in range(run_trials):
            hp = trial_hparams[t]
            trial_seed = args.seed + t * 9973
            print(f"\n--- [{variant} screening] {t + 1}/{run_trials}  hparams={hp} ---", flush=True)
            best_mrr, best_metrics, model = run_trial(
                hp=hp,
                epochs=args.screening_epochs,
                trial_seed=trial_seed,
                batch_size=args.batch_size,
                **trial_kw,
            )
            print(
                f"  trial best MRR: {best_mrr:.6f}  | "
                f"MRR={best_metrics['MRR']:.6f} NDCG@5={best_metrics['NDCG@5']:.6f} "
                f"Hit@1={best_metrics['Hit@1']:.6f}",
                flush=True,
            )
            log_trials.append(
                {
                    "phase": "screening",
                    "hparams": hp,
                    "epochs_in_phase": args.screening_epochs,
                    "best_mrr_in_trial": best_mrr,
                    "best_epoch": best_metrics,
                }
            )
            if best_mrr > global_best_mrr:
                global_best_mrr = best_mrr
                global_best_hp = dict(hp)
                model.save_weights(out_weights)
                print(f"  [전역 갱신] 저장 → {out_weights}  MRR={global_best_mrr:.6f}", flush=True)
            screening_rows.append((best_mrr, dict(hp), best_metrics))
            K.clear_session()
        screening_rows.sort(key=lambda x: -x[0])
        top_hps: List[dict] = []
        seen_keys: set = set()
        for _mrr, hp, _lm in screening_rows:
            key = tuple(sorted(hp.items()))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            top_hps.append(hp)
            if len(top_hps) >= k:
                break
        print(f"\n[2-phase] 2차(refine): 상위 {len(top_hps)}개, 각 {args.epochs_per_trial} epochs\n", flush=True)
        for j, hp in enumerate(top_hps):
            trial_seed = args.seed + 884422 + j * 9973
            _one_trial(j, len(top_hps), hp, args.epochs_per_trial, trial_seed, "refine")
    else:
        for t in range(run_trials):
            hp = trial_hparams[t]
            trial_seed = args.seed + t * 9973
            _one_trial(t, run_trials, hp, args.epochs_per_trial, trial_seed, "single")

    summary = {
        "variant": variant,
        "text_mode": "title_cat",
        "text_views": ["title", "category", "subcategory"],
        "image_view": use_image,
        "global_best_mrr": global_best_mrr,
        "global_best_hparams": global_best_hp,
        "trials": log_trials,
        "max_history_clicks": int(MAX_HISTORY_CLICKS),
        "epochs_per_trial": args.epochs_per_trial,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "hparam_grid_size": grid_n,
        "hparam_choices": HPARAM_CHOICES,
        "two_phase": bool(args.two_phase),
        "screening_epochs": args.screening_epochs if args.two_phase else None,
        "refine_top_k": args.refine_top_k if args.two_phase else None,
        "out_weights": out_weights,
    }
    with open(out_log, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[{variant}] 완료. 전역 최고 val MRR={global_best_mrr:.6f}, 로그: {out_log}")
    if global_best_hp:
        print(f"[{variant}] 최적 hparams: {global_best_hp}")


def main() -> None:
    ap = argparse.ArgumentParser(description="title+cat/subcat S1/S2 하이퍼파라미터 탐색")
    ap.add_argument("--variant", type=str, default="both", choices=["s1", "s2", "both"])
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--epochs-per-trial", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--mind-dataset-subdir", type=str, default="MIND_2000")
    ap.add_argument("--max-history-clicks", type=int, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-weights", type=str, default=None)
    ap.add_argument("--out-log", type=str, default=None)
    ap.add_argument("--allow-duplicate-hparams", action="store_true")
    ap.add_argument("--two-phase", action="store_true")
    ap.add_argument("--screening-epochs", type=int, default=2)
    ap.add_argument("--refine-top-k", type=int, default=5)
    ap.add_argument("--resume-log", type=str, default=None)
    ap.add_argument("--thumbnail-dir", type=str, default=DEFAULT_THUMBNAIL_DIR)
    ap.add_argument("--clip-cache", type=str, default=None)
    ap.add_argument("--clip-device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--clip-batch-size", type=int, default=16)
    ap.add_argument(
        "--skip-test-eval",
        action="store_true",
        help="튜닝 후 held-out test 평가를 하지 않음 (기본은 자동 평가)",
    )
    args = ap.parse_args()

    argv = ["--mind-dataset-subdir", args.mind_dataset_subdir]
    if args.max_history_clicks is not None:
        argv += ["--max-history-clicks", str(args.max_history_clicks)]
    apply_dataset_env_from_argv(argv)

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    tf.random.set_seed(args.seed)

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
    news_image = None
    need_s2 = args.variant in ("s2", "both")
    if need_s2:
        thumb_dir = resolve_project_path(args.thumbnail_dir)
        catalog_ids = [nid for nid, idx in news_index.items() if nid != "0" and int(idx) != 0]
        print_missing_thumbnail_report(catalog_ids, thumb_dir)
        clip_cache = (
            resolve_project_path(args.clip_cache)
            if args.clip_cache
            else default_cache_path(args.mind_dataset_subdir)
        )
        news_tsv = mind_data_path(MIND_NEWS_FILENAME)
        ensure_clip_cache(clip_cache, thumb_dir, news_tsv, args.clip_device, args.clip_batch_size)
        fallback_ids, _ = count_missing_thumbnails(load_news_ids_from_tsv(news_tsv), thumb_dir)
        news_image, n_hit = build_news_image_matrix(
            news_index, len(news_words), clip_cache, news_ids_fallback=fallback_ids
        )
        print(f"[tune title_cat] CLIP nonzero={n_hit}/{len(catalog_ids)}", flush=True)

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
        all_train_pn=all_train_pn,
        all_label=all_label,
        all_train_id=all_train_id,
        all_user_pos=all_user_pos,
        all_test_pn=all_test_pn,
        all_test_label=all_test_label,
        all_test_id=all_test_id,
        all_test_user_pos=all_test_user_pos,
        all_test_index=all_test_index,
    )
    variants = ["s1", "s2"] if args.variant == "both" else [args.variant]
    for v in variants:
        _tune_one(args, v, data)

    if args.skip_test_eval:
        print("[tune title_cat] --skip-test-eval: held-out test 평가를 건너뜁니다.", flush=True)
        return
    from types import SimpleNamespace

    from eval_title_cat import run_eval_title_cat

    K.clear_session()
    print("\n[tune title_cat] val 튜닝 종료 → held-out test 평가", flush=True)
    run_eval_title_cat(
        SimpleNamespace(
            variant=args.variant,
            split="test",
            mind_dataset_subdir=args.mind_dataset_subdir,
            s1_weights=None,
            s1_tune_log=None,
            s2_weights=None,
            s2_tune_log=None,
            batch_size=args.batch_size,
            seed=args.seed,
            max_history_clicks=args.max_history_clicks,
            thumbnail_dir=args.thumbnail_dir,
            clip_cache=args.clip_cache,
            out=None,
        )
    )


if __name__ == "__main__":
    main()
