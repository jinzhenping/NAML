#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
S2 (full-text NAML + CLIP image view) 하이퍼파라미터 탐색.

텍스트: title + body + category + subcategory (기존 NAML 4뷰)
이미지: CLIP 썸네일 5번째 뷰
그리드/two-phase/resume 는 NAML/naml_tune_actual.py 와 동일.

  conda activate tf28gpu
  python CLIP/tune_s2.py --two-phase --trials 108 --screening-epochs 3 \
    --refine-top-k 10 --epochs-per-trial 10 \
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
from naml_image_model import build_naml_models_with_image
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


def run_trial_s2(
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
            title_only=False,
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
            title_only=False,
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
    default_out_dir = str(_CLIP_DIR / "saved_models" / "MIND_2000")
    ap = argparse.ArgumentParser(description="S2 full-text+CLIP 하이퍼파라미터 탐색 (naml_tune_actual 과 동일 그리드)")
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--epochs-per-trial", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--mind-dataset-subdir", type=str, default="MIND_2000")
    ap.add_argument("--max-history-clicks", type=int, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--out-weights",
        type=str,
        default=None,
        help="전역 최고 val MRR 가중치. 기본 CLIP/saved_models/<subdir>/S2_naml_clip_tuned.h5",
    )
    ap.add_argument(
        "--out-log",
        type=str,
        default=None,
        help="튜닝 로그 JSON. 기본 CLIP/saved_models/<subdir>/naml_tune_s2_clip_log.json",
    )
    ap.add_argument("--allow-duplicate-hparams", action="store_true")
    ap.add_argument("--two-phase", action="store_true")
    ap.add_argument("--screening-epochs", type=int, default=2)
    ap.add_argument("--refine-top-k", type=int, default=5)
    ap.add_argument("--resume-log", type=str, default=None)
    ap.add_argument("--append-log", action="store_true")
    ap.add_argument("--repeat-per-combo", type=int, default=1)
    ap.add_argument("--fixed-filter-kernel-grid", action="store_true")
    ap.add_argument("--grid-cnn-filters", type=int, nargs="+", default=[256, 384, 512])
    ap.add_argument("--grid-cnn-kernel-sizes", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--fixed-learning-rate", type=float, default=0.001)
    ap.add_argument("--fixed-dropout-rate", type=float, default=0.25)
    ap.add_argument("--fixed-attention-dense-dim", type=int, default=160)
    ap.add_argument("--fixed-category-emb-dim", type=int, default=64)
    ap.add_argument("--thumbnail-dir", type=str, default=DEFAULT_THUMBNAIL_DIR)
    ap.add_argument("--clip-cache", type=str, default=None)
    ap.add_argument("--clip-device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--clip-batch-size", type=int, default=16)
    args = ap.parse_args()

    argv = ["--mind-dataset-subdir", args.mind_dataset_subdir]
    if args.max_history_clicks is not None:
        argv += ["--max-history-clicks", str(args.max_history_clicks)]
    apply_dataset_env_from_argv(argv)

    if args.two_phase and args.repeat_per_combo > 1:
        print("오류: --two-phase 와 --repeat-per-combo>1 은 함께 사용할 수 없습니다.", file=sys.stderr)
        sys.exit(2)

    out_dir = str(_CLIP_DIR / "saved_models" / args.mind_dataset_subdir)
    os.makedirs(out_dir, exist_ok=True)
    out_weights = resolve_project_path(args.out_weights) if args.out_weights else os.path.join(
        out_dir, "S2_naml_clip_tuned.h5"
    )
    out_log = resolve_project_path(args.out_log) if args.out_log else os.path.join(
        out_dir, "naml_tune_s2_clip_log.json"
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_weights)) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_log)) or ".", exist_ok=True)

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    tf.random.set_seed(args.seed)

    print(
        f"[tune S2] full-text (title+body+cat/subcat) + CLIP image view, "
        f"dataset={args.mind_dataset_subdir}",
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
    print(
        f"[tune S2] train={len(all_train_id)} val_rows={len(all_test_id)} "
        f"CLIP nonzero={n_hit}/{len(catalog_ids)}",
        flush=True,
    )
    print(f"[tune S2] log={out_log}", flush=True)
    print(f"[tune S2] weights(best MRR)={out_weights}", flush=True)

    trial_kw = dict(
        batch_size=args.batch_size,
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

    rng = random.Random(args.seed)
    global_best_mrr = -1.0
    global_best_hp: Optional[Dict[str, Any]] = None
    log_trials: List[dict] = []

    grid_n = _hparam_grid_size()
    seen_hparam_keys: set = set()
    resume_log_path: Optional[str] = None
    if args.resume_log:
        resume_log_path = resolve_project_path(args.resume_log)
        if os.path.isfile(resume_log_path):
            seen_hparam_keys = _load_seen_hparam_keys_from_log(resume_log_path)
            print(f"resume-log 로드: {resume_log_path} (이미 시도한 조합 {len(seen_hparam_keys)}개)")
            prev_best_mrr, prev_best_hp = _load_previous_best_from_log(resume_log_path)
            if prev_best_mrr > global_best_mrr:
                global_best_mrr = prev_best_mrr
                global_best_hp = prev_best_hp
        else:
            print(f"경고: --resume-log 파일이 없어 skip-seen 생략: {resume_log_path}")

    if args.fixed_filter_kernel_grid:
        filters = [int(x) for x in args.grid_cnn_filters]
        kernels = [int(x) for x in args.grid_cnn_kernel_sizes]
        fixed_base = {
            "learning_rate": float(args.fixed_learning_rate),
            "dropout_rate": float(args.fixed_dropout_rate),
            "attention_dense_dim": int(args.fixed_attention_dense_dim),
            "category_emb_dim": int(args.fixed_category_emb_dim),
        }
        all_grid = []
        for f in filters:
            for ksz in kernels:
                hp = dict(fixed_base)
                hp["cnn_filters"] = int(f)
                hp["cnn_kernel_size"] = int(ksz)
                all_grid.append(hp)
        rng.shuffle(all_grid)
        trial_hparams = [hp for hp in all_grid if _hp_key(hp) not in seen_hparam_keys][: args.trials]
        print(f"fixed filter-kernel grid: {len(all_grid)} 조합, 고정값={fixed_base}")
    elif args.allow_duplicate_hparams:
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

    repeat_per_combo = max(1, int(args.repeat_per_combo))
    if repeat_per_combo > 1:
        trial_hparams = [hp for hp in trial_hparams for _ in range(repeat_per_combo)]
    run_trials = len(trial_hparams)
    if run_trials == 0:
        print("실행할 새 조합이 없습니다.")
        return

    def _one_trial(trial_idx: int, total: int, hp: dict, epochs: int, trial_seed: int, phase: str) -> None:
        nonlocal global_best_mrr, global_best_hp
        print(f"\n--- [{phase}] {trial_idx + 1}/{total}  hparams={hp} ---", flush=True)
        best_mrr, best_metrics, model = run_trial_s2(
            hp=hp, epochs=epochs, trial_seed=trial_seed, **trial_kw
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
            print(f"\n--- [screening] {t + 1}/{run_trials}  hparams={hp} ---", flush=True)
            best_mrr, best_metrics, model = run_trial_s2(
                hp=hp, epochs=args.screening_epochs, trial_seed=trial_seed, **trial_kw
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
        "variant": "s2",
        "title_only": False,
        "text_views": ["title", "body", "category", "subcategory"],
        "image_view": True,
        "clip_cache": clip_cache,
        "global_best_mrr": global_best_mrr,
        "global_best_hparams": global_best_hp,
        "trials": log_trials,
        "max_history_clicks": int(MAX_HISTORY_CLICKS),
        "epochs_per_trial": args.epochs_per_trial,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "hparam_grid_size": grid_n,
        "hparam_choices": HPARAM_CHOICES,
        "allow_duplicate_hparams": bool(args.allow_duplicate_hparams),
        "resume_log": resume_log_path,
        "num_seen_hparams_loaded": len(seen_hparam_keys),
        "num_trials_requested": int(args.trials),
        "num_trials_executed": int(run_trials),
        "two_phase": bool(args.two_phase),
        "screening_epochs": args.screening_epochs if args.two_phase else None,
        "refine_top_k": args.refine_top_k if args.two_phase else None,
        "repeat_per_combo": int(args.repeat_per_combo),
        "out_weights": out_weights,
    }
    append_mode = bool(args.append_log or args.resume_log)
    if append_mode and os.path.isfile(out_log):
        old = _load_json_or_none(out_log)
        if old is not None:
            old_trials = old.get("trials", [])
            if not isinstance(old_trials, list):
                old_trials = []
            merged_trials = old_trials + summary["trials"]
            try:
                old_best = float(old.get("global_best_mrr", -1.0))
            except Exception:
                old_best = -1.0
            if old_best > summary["global_best_mrr"]:
                summary["global_best_mrr"] = old_best
                old_hp = old.get("global_best_hparams", None)
                if isinstance(old_hp, dict):
                    summary["global_best_hparams"] = old_hp
            summary["trials"] = merged_trials
            summary["append_log"] = True
    with open(out_log, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n완료. 전역 최고 MRR={global_best_mrr:.6f}, 로그: {out_log}")
    if global_best_hp:
        print(f"최적 hparams: {global_best_hp}")


if __name__ == "__main__":
    main()
