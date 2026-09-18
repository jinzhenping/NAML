#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
S3': title+cat/subcat NAML.

히스토리 이미지 = prior(actual_body)  (S3와 동일, 뉴스 단위)
후보 이미지     = prior(title + preference_profile)  (유저×뉴스)
텍스트          = title + category + subcategory (본문 없음)

val MRR 최고 조합을 저장한 뒤 held-out test를 평가한다.

  conda activate clip_cu128
  python CLIP/extract_actual_body_prior_embeds.py --scope catalog --mind-dataset-subdir MIND_2000
  python CLIP/extract_s3_prime_prior_embeds.py --split all --mind-dataset-subdir MIND_2000

  conda activate tf28gpu
  python CLIP/tune_s3_prime.py --two-phase --trials 108 \
    --screening-epochs 3 --refine-top-k 10 --epochs-per-trial 10 \
    --mind-dataset-subdir MIND_2000
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
    build_news_image_matrix,
    default_actual_body_prior_cache_path,
    default_s3p_cache_path,
    resolve_project_path,
)
from eval_s3_prime import S3P_LOG_NAME, S3P_WEIGHTS_NAME
from expected_image import (
    build_test_candidate_image,
    build_train_candidate_image,
    load_pair_dict_normed,
    pair_embed_dim,
)
from naml_common import (
    MAX_HISTORY_CLICKS,
    SEED,
    get_embedding,
    preprocess_news_file,
    preprocess_user_file,
)
from naml_image_model import build_naml_models_title_cat_image
from naml_tune_actual import (
    HPARAM_CHOICES,
    _hparam_grid_size,
    _hp_key,
    _load_previous_best_from_log,
    _load_seen_hparam_keys_from_log,
    plan_hparam_trials,
    sample_hparams,
)
from train_s1_s2 import evaluate_metrics, generate_batch_data_train


def run_trial(
    hp: dict,
    epochs: int,
    batch_size: int,
    word_dict,
    embedding_mat,
    category,
    subcategory,
    news_words,
    news_body,
    news_v,
    news_sv,
    news_image,
    cand_image_train,
    cand_image_val,
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
    if news_image is None or cand_image_train is None or cand_image_val is None:
        raise ValueError("S3' 튜닝에는 history prior 행렬과 candidate pair 행렬이 필요합니다.")
    np.random.seed(trial_seed)
    random.seed(trial_seed)
    tf.random.set_seed(trial_seed)
    clip_dim = int(news_image.shape[1])
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
            news_image=news_image,
            cand_image=cand_image_train,
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
            news_image=news_image,
            cand_image=cand_image_val,
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


def main() -> None:
    ap = argparse.ArgumentParser(description="S3' summary+title candidate prior NAML 튜닝")
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
    ap.add_argument("--prior-cache", type=str, default=None)
    ap.add_argument("--s3p-train-cache", type=str, default=None)
    ap.add_argument("--s3p-val-cache", type=str, default=None)
    ap.add_argument("--skip-test-eval", action="store_true")
    args = ap.parse_args()

    argv = ["--mind-dataset-subdir", args.mind_dataset_subdir]
    if args.max_history_clicks is not None:
        argv += ["--max-history-clicks", str(args.max_history_clicks)]
    apply_dataset_env_from_argv(argv)

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    tf.random.set_seed(args.seed)

    out_dir = str(_CLIP_DIR / "saved_models" / args.mind_dataset_subdir)
    os.makedirs(out_dir, exist_ok=True)
    out_weights = (
        resolve_project_path(args.out_weights)
        if args.out_weights
        else os.path.join(out_dir, S3P_WEIGHTS_NAME)
    )
    out_log = resolve_project_path(args.out_log) if args.out_log else os.path.join(out_dir, S3P_LOG_NAME)

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

    hist_cache = (
        resolve_project_path(args.prior_cache)
        if args.prior_cache
        else default_actual_body_prior_cache_path(args.mind_dataset_subdir)
    )
    train_pair_path = (
        resolve_project_path(args.s3p_train_cache)
        if args.s3p_train_cache
        else default_s3p_cache_path(args.mind_dataset_subdir, "train")
    )
    val_pair_path = (
        resolve_project_path(args.s3p_val_cache)
        if args.s3p_val_cache
        else default_s3p_cache_path(args.mind_dataset_subdir, "val")
    )
    for p, name in (
        (hist_cache, "actual-body prior"),
        (train_pair_path, "S3' train candidate prior"),
        (val_pair_path, "S3' val candidate prior"),
    ):
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"{name} cache 없음: {p}\n"
                "conda activate clip_cu128\n"
                "python CLIP/extract_actual_body_prior_embeds.py --scope catalog "
                f"--mind-dataset-subdir {args.mind_dataset_subdir}\n"
                "python CLIP/extract_s3_prime_prior_embeds.py --split all "
                f"--mind-dataset-subdir {args.mind_dataset_subdir}"
            )

    news_image, n_hist = build_news_image_matrix(news_index, len(news_words), hist_cache)
    train_pairs = load_pair_dict_normed(train_pair_path)
    val_pairs = load_pair_dict_normed(val_pair_path)
    clip_dim = int(news_image.shape[1])
    for name, d in (("train", train_pairs), ("val", val_pairs)):
        dim = pair_embed_dim(d)
        if dim != clip_dim:
            raise ValueError(f"S3' {name} cand dim={dim} != history prior dim={clip_dim}")
    cand_image_train, train_cov = build_train_candidate_image(
        all_train_userid_str,
        all_train_newsid_str,
        train_pairs,
        clip_dim,
        n_cand=int(all_train_pn.shape[1]),
    )
    cand_image_val, val_cov = build_test_candidate_image(
        all_test_userid_str, all_test_newsid_str, val_pairs, clip_dim
    )
    print(
        f"[tune s3p] text=title+cat/subcat  hist=prior(actual_body)  "
        f"cand=prior(title+summary)\n"
        f"[tune s3p] hist nonzero={n_hist}  train cand {cand_image_train.shape} {train_cov}\n"
        f"[tune s3p] val cand {cand_image_val.shape} {val_cov}  weights={out_weights}",
        flush=True,
    )
    if train_cov["n_nonzero"] == 0:
        raise ValueError("학습 후보 prior가 전부 0벡터입니다. preference/title 캐시를 확인하세요.")

    trial_kw = dict(
        word_dict=word_dict,
        embedding_mat=embedding_mat,
        category=category,
        subcategory=subcategory,
        news_words=news_words,
        news_body=news_body,
        news_v=news_v,
        news_sv=news_sv,
        news_image=news_image,
        cand_image_train=cand_image_train,
        cand_image_val=cand_image_val,
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

    rng = random.Random(args.seed)
    global_best_mrr = -1.0
    global_best_hp: Optional[Dict[str, Any]] = None
    log_trials: List[dict] = []
    grid_n = _hparam_grid_size()
    seen_hparam_keys: set = set()
    if args.resume_log:
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
        print(f"\n--- [s3p {phase}] {trial_idx + 1}/{total}  hparams={hp} ---", flush=True)
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
            print(f"\n--- [s3p screening] {t + 1}/{run_trials}  hparams={hp} ---", flush=True)
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
        "variant": "s3p",
        "text_mode": "title_cat",
        "text_views": ["title", "category", "subcategory"],
        "history_image": "prior_actual_body",
        "candidate_image": "prior_summary_title",
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
        "prior_cache": os.path.abspath(hist_cache),
        "s3p_train_cache": os.path.abspath(train_pair_path),
        "s3p_val_cache": os.path.abspath(val_pair_path),
        "train_cand_coverage": train_cov,
        "val_cand_coverage": val_cov,
    }
    with open(out_log, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[s3p] 완료. 전역 최고 val MRR={global_best_mrr:.6f}, 로그: {out_log}")
    if global_best_hp:
        print(f"[s3p] 최적 hparams: {global_best_hp}")

    if args.skip_test_eval:
        print("[tune s3p] --skip-test-eval: held-out test 평가를 건너뜁니다.", flush=True)
        return
    from types import SimpleNamespace

    from eval_s3_prime import run_eval_s3_prime

    K.clear_session()
    print("\n[tune s3p] val 튜닝 종료 → held-out test 평가", flush=True)
    run_eval_s3_prime(
        SimpleNamespace(
            split="test",
            mind_dataset_subdir=args.mind_dataset_subdir,
            weights=out_weights,
            tune_log=out_log,
            prior_cache=args.prior_cache,
            s3p_cache=None,
            batch_size=args.batch_size,
            seed=args.seed,
            max_history_clicks=args.max_history_clicks,
            out=None,
        )
    )


if __name__ == "__main__":
    main()
