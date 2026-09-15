#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
S3: 히스토리/후보 이미지 슬롯 = prior(actual_body) (뉴스 단위 CLIP image embed).

썸네일 CLIP(B0)이 아니라 Kandinsky prior가 실제 본문을 본 이미지 임베딩을 쓴다.
B2 기대본문 prior / Δ expected-image 가 아님.

text-mode:
  full      — title + body + cat + subcat + prior image
  title_cat — title + cat + subcat + prior image (본문 없음)

val MRR 최고 조합을 저장한 뒤, 같은 프로세스가 held-out test를 평가한다.

  conda activate tf28gpu
  python CLIP/tune_s3.py --text-mode both --two-phase --trials 108 \
    --screening-epochs 3 --refine-top-k 10 --epochs-per-trial 10 \
    --mind-dataset-subdir MIND_2000

prior 캐시가 없으면 추출하지 않고 에러. clip_cu128에서 먼저:

  python CLIP/extract_actual_body_prior_embeds.py --scope catalog \
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
    build_news_image_matrix,
    default_actual_body_prior_cache_path,
    resolve_project_path,
)
from naml_common import (
    MAX_HISTORY_CLICKS,
    SEED,
    get_embedding,
    preprocess_news_file,
    preprocess_user_file,
)
from naml_image_model import build_naml_models_title_cat_image, build_naml_models_with_image
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

from eval_s3 import s3_artifact_names


def run_trial(
    hp: dict,
    epochs: int,
    batch_size: int,
    text_mode: str,
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
    if news_image is None:
        raise ValueError("S3 튜닝에는 prior(actual_body) 이미지 행렬이 필요합니다.")
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
    clip_dim = int(news_image.shape[1])
    if text_mode == "title_cat":
        built = build_naml_models_title_cat_image(
            word_dict,
            embedding_mat,
            category,
            subcategory,
            hp["learning_rate"],
            clip_dim=clip_dim,
            clear_session=True,
            **arch_kw,
        )
    else:
        built = build_naml_models_with_image(
            word_dict,
            embedding_mat,
            category,
            subcategory,
            hp["learning_rate"],
            clip_dim=clip_dim,
            clear_session=True,
            **arch_kw,
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
            text_mode=text_mode,
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
            text_mode=text_mode,
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


def _tune_one(args, text_mode: str, data: dict) -> None:
    out_dir = str(_CLIP_DIR / "saved_models" / args.mind_dataset_subdir)
    os.makedirs(out_dir, exist_ok=True)
    default_w, default_l = s3_artifact_names(text_mode)
    out_weights = (
        resolve_project_path(args.out_weights)
        if args.out_weights and args.text_mode != "both"
        else os.path.join(out_dir, default_w)
    )
    out_log = (
        resolve_project_path(args.out_log)
        if args.out_log and args.text_mode != "both"
        else os.path.join(out_dir, default_l)
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_weights)) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_log)) or ".", exist_ok=True)
    text_desc = (
        "title+body+cat/subcat"
        if text_mode == "full"
        else "title+cat/subcat (no body)"
    )
    print(
        f"[tune S3 {text_mode}] text={text_desc}  image=prior(actual_body)  "
        f"val=MIND_dev  weights={out_weights}",
        flush=True,
    )

    trial_kw = dict(data)
    trial_kw.pop("prior_cache", None)
    trial_kw["text_mode"] = text_mode

    rng = random.Random(args.seed)
    global_best_mrr = -1.0
    global_best_hp: Optional[Dict[str, Any]] = None
    log_trials: List[dict] = []
    grid_n = _hparam_grid_size()
    seen_hparam_keys: set = set()
    if args.resume_log and args.text_mode != "both":
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
        print(f"\n--- [s3 {text_mode} {phase}] {trial_idx + 1}/{total}  hparams={hp} ---", flush=True)
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
            print(f"\n--- [s3 {text_mode} screening] {t + 1}/{run_trials}  hparams={hp} ---", flush=True)
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
        "variant": "s3",
        "text_mode": text_mode,
        "text_views": (
            ["title", "body", "category", "subcategory"]
            if text_mode == "full"
            else ["title", "category", "subcategory"]
        ),
        "image_view": "prior_actual_body",
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
        "prior_cache": data.get("prior_cache"),
    }
    with open(out_log, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[s3 {text_mode}] 완료. 전역 최고 val MRR={global_best_mrr:.6f}, 로그: {out_log}")
    if global_best_hp:
        print(f"[s3 {text_mode}] 최적 hparams: {global_best_hp}")


def main() -> None:
    ap = argparse.ArgumentParser(description="S3 prior(actual_body) 이미지 NAML 하이퍼파라미터 탐색")
    ap.add_argument("--text-mode", type=str, default="both", choices=["full", "title_cat", "both"])
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
    prior_cache = (
        resolve_project_path(args.prior_cache)
        if args.prior_cache
        else default_actual_body_prior_cache_path(args.mind_dataset_subdir)
    )
    if not os.path.isfile(prior_cache):
        raise FileNotFoundError(
            f"actual-body prior cache 없음: {prior_cache}\n"
            "conda activate clip_cu128\n"
            "python CLIP/extract_actual_body_prior_embeds.py --scope catalog "
            f"--mind-dataset-subdir {args.mind_dataset_subdir}"
        )
    catalog_ids = [nid for nid, idx in news_index.items() if nid != "0" and int(idx) != 0]
    news_image, n_hit = build_news_image_matrix(news_index, len(news_words), prior_cache)
    print(
        f"[tune s3] prior cache={prior_cache} nonzero={n_hit}/{len(catalog_ids)}",
        flush=True,
    )
    if n_hit < len(catalog_ids):
        print(
            f"[tune s3] 경고: catalog {len(catalog_ids) - n_hit}개가 0벡터입니다. "
            "val/test ID를 채우려면 extract_actual_body_prior_embeds.py --scope catalog",
            flush=True,
        )

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
        prior_cache=os.path.abspath(prior_cache),
    )
    modes = ["full", "title_cat"] if args.text_mode == "both" else [args.text_mode]
    for mode in modes:
        _tune_one(args, mode, data)

    if args.skip_test_eval:
        print("[tune s3] --skip-test-eval: held-out test 평가를 건너뜁니다.", flush=True)
        return
    from types import SimpleNamespace

    from eval_s3 import run_eval_s3

    K.clear_session()
    print("\n[tune s3] val 튜닝 종료 → held-out test 평가", flush=True)
    run_eval_s3(
        SimpleNamespace(
            text_mode=args.text_mode,
            split="test",
            mind_dataset_subdir=args.mind_dataset_subdir,
            full_weights=None,
            full_tune_log=None,
            title_cat_weights=None,
            title_cat_tune_log=None,
            prior_cache=args.prior_cache,
            batch_size=args.batch_size,
            seed=args.seed,
            max_history_clicks=args.max_history_clicks,
            out=None,
        )
    )


if __name__ == "__main__":
    main()
