#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NAML-IM (full-text + impression 5th view) 하이퍼파라미터 탐색.

프로토콜: tune_s2.py 와 동일 그리드 / two-phase / resume.

  conda activate clip_cu128
  python IMRec/train_eval.py --stage prepare --mind-dataset-subdir MIND_2000
  python IMRec/train_eval.py --stage extract --mind-dataset-subdir MIND_2000

  conda activate tf28gpu
  python CLIP/tune_naml_im.py --two-phase --trials 108 --screening-epochs 3 \\
    --refine-top-k 10 --epochs-per-trial 10 --mind-dataset-subdir MIND_2000

val MRR 최고 가중치 저장 후 held-out test(MIND_test_(2000).tsv) 자동 평가 (S3 tune 과 동일).
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

from clip_embeddings import resolve_project_path
from im_features import build_im_news_features, default_im_feature_path
from naml_common import MAX_HISTORY_CLICKS, SEED, get_embedding, preprocess_news_file, preprocess_user_file
from naml_im_batch import generate_batch_data_test_im, generate_batch_data_train_im
from naml_im_model import build_naml_models_im
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
from train_naml_im import evaluate_metrics_im


def run_trial_naml_im(
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
    im,
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

    built = build_naml_models_im(
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
    n_train = len(all_train_id)
    steps_per_epoch = (n_train + batch_size - 1) // batch_size

    best_mrr = -1.0
    best_weights = None
    best_metrics = None
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
        hist = model.fit(traingen, epochs=1, steps_per_epoch=steps_per_epoch, verbose=0)
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
    ap = argparse.ArgumentParser(description="NAML-IM 하이퍼파라미터 탐색 (S2와 동일 그리드)")
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--epochs-per-trial", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--mind-dataset-subdir", type=str, default="MIND_2000")
    ap.add_argument("--max-history-clicks", type=int, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-weights", type=str, default=None)
    ap.add_argument("--out-log", type=str, default=None)
    ap.add_argument("--im-cache", type=str, default=None)
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

    if args.two_phase and args.repeat_per_combo > 1:
        print("오류: --two-phase 와 --repeat-per-combo>1 은 함께 사용할 수 없습니다.", file=sys.stderr)
        sys.exit(2)

    out_dir = str(_CLIP_DIR / "saved_models" / args.mind_dataset_subdir)
    os.makedirs(out_dir, exist_ok=True)
    out_weights = resolve_project_path(args.out_weights) if args.out_weights else os.path.join(
        out_dir, "NAML_im_tuned.h5"
    )
    out_log = resolve_project_path(args.out_log) if args.out_log else os.path.join(
        out_dir, "naml_tune_naml_im_log.json"
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_weights)) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_log)) or ".", exist_ok=True)

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    tf.random.set_seed(args.seed)

    print(
        f"[tune NAML-IM] full-text + impression view, dataset={args.mind_dataset_subdir}",
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

    im_cache = (
        resolve_project_path(args.im_cache)
        if args.im_cache
        else default_im_feature_path(args.mind_dataset_subdir)
    )
    im, n_hit = build_im_news_features(news_index, len(news_words), im_cache)
    print(
        f"[tune NAML-IM] train={len(all_train_id)} val_rows={len(all_test_id)} "
        f"IM nonzero={n_hit} cache={im_cache}",
        flush=True,
    )
    print(f"[tune NAML-IM] log={out_log}", flush=True)
    print(f"[tune NAML-IM] weights(best MRR)={out_weights}", flush=True)

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
        im=im,
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
        best_mrr, best_metrics, model = run_trial_naml_im(
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
            best_mrr, best_metrics, model = run_trial_naml_im(
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
        "variant": "naml_im",
        "title_only": False,
        "text_views": ["title", "body", "category", "subcategory"],
        "impression_view": True,
        "im_cache": os.path.abspath(im_cache),
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
    print(f"\n완료. 전역 최고 val MRR={global_best_mrr:.6f}, 로그: {out_log}")
    if global_best_hp:
        print(f"최적 hparams: {global_best_hp}")

    if args.skip_test_eval:
        print("[tune NAML-IM] --skip-test-eval: held-out test 평가를 건너뜁니다.", flush=True)
        return
    if global_best_hp is None or not os.path.isfile(out_weights):
        print("[tune NAML-IM] 유효한 best 가중치 없음 → test 평가 생략", flush=True)
        return

    from types import SimpleNamespace

    from eval_naml_im import run_eval_naml_im

    K.clear_session()
    print("\n[tune NAML-IM] val 튜닝 종료 → held-out test 평가", flush=True)
    test_payload = run_eval_naml_im(
        SimpleNamespace(
            split="test",
            mind_dataset_subdir=args.mind_dataset_subdir,
            weights=out_weights,
            tune_log=out_log,
            im_cache=args.im_cache,
            batch_size=args.batch_size,
            seed=args.seed,
            max_history_clicks=args.max_history_clicks,
            out=os.path.join(out_dir, "naml_im_tune_test.json"),
        )
    )
    summary["test_tsv"] = test_payload.get("test_tsv")
    summary["test_metrics"] = test_payload.get("metrics")
    with open(out_log, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    tm = test_payload.get("metrics") or {}
    print(
        f"[tune NAML-IM] TEST MRR={tm.get('MRR', 0.0):.6f}  "
        f"NDCG@5={tm.get('NDCG@5', 0.0):.6f}  Hit@1={tm.get('Hit@1', 0.0):.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
