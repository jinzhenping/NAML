#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
teacher model (actual-body) 학습을 “고정된 hparams 1조합”만으로 수행합니다.

입력:
  - expected-body 튜닝 로그(json)에서 global_best_hparams 추출
  - 그 hparams로 actual-body teacher를 학습 (best MRR epoch 가중치 저장)

예시:
  python NAML/train_teacher_actual_fixed_hparams.py \
    --tune-log saved_models/Adressa_2000/naml_tune_expected_log.json \
    --mind-dataset-subdir Adressa_2000 \
    --epochs 10 \
    --batch-size 64 \
    --seed 42 \
    --num-runs 3 \
    --out-weights saved_models/Adressa_2000/NAML_adressa_2000_teacher_from_expected.h5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import tensorflow as tf

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "NAML") not in sys.path:
    sys.path.insert(0, str(_ROOT / "NAML"))

from naml_dataset_env import apply_dataset_env_from_argv
# 중요: naml_common import 전에 dataset env를 먼저 적용해야
# MIND_DATASET_SUBDIR 기준 파일명이 올바르게 고정됩니다.
apply_dataset_env_from_argv()

from naml_common import preprocess_news_file, preprocess_user_file, get_embedding


def _resolve_project_path(p: str) -> str:
    p = p.strip()
    return os.path.normpath(p) if os.path.isabs(p) else os.path.normpath(str(_ROOT / p))


def _load_global_best_hparams(tune_log_path: str) -> Dict[str, Any]:
    path = _resolve_project_path(tune_log_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"--tune-log 파일 없음: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    gb = data.get("global_best_hparams")
    if not isinstance(gb, dict):
        raise ValueError(f"tune-log에서 global_best_hparams(dict)를 찾지 못했습니다: {path}")
    return gb


def main() -> None:
    ap = argparse.ArgumentParser(
        description="expected-body 튜닝 global_best_hparams로 actual-body teacher 학습(고정 1조합)"
    )
    ap.add_argument(
        "--tune-log",
        type=str,
        required=True,
        help="expected-body 튜닝 로그 json 경로 (global_best_hparams를 읽음)",
    )
    ap.add_argument("--mind-dataset-subdir", type=str, default="MIND_2000")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="동일 설정으로 실제본문 teacher 학습을 몇 번 반복할지 (기본 3)",
    )
    ap.add_argument(
        "--run-seed-step",
        type=int,
        default=1,
        help="반복 실행 시 run마다 seed 증가 폭 (run_seed = seed + i*step, 기본 1)",
    )
    ap.add_argument(
        "--out-weights",
        type=str,
        required=True,
        help="저장할 teacher 가중치 .h5 경로 (프로젝트 루트 기준 상대도 가능)",
    )
    ap.add_argument(
        "--out-log",
        type=str,
        default=None,
        help="선택: 학습 요약 json 경로",
    )
    args = ap.parse_args()

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # naml_common 경로 고정 (naml_tune_actual 내부도 동일 규칙)
    apply_dataset_env_from_argv(["--mind-dataset-subdir", args.mind_dataset_subdir])
    os.environ["MIND_DATASET_SUBDIR"] = args.mind_dataset_subdir

    hp = _load_global_best_hparams(args.tune_log)
    required_keys = [
        "learning_rate",
        "dropout_rate",
        "cnn_filters",
        "cnn_kernel_size",
        "attention_dense_dim",
        "category_emb_dim",
    ]
    for k in required_keys:
        if k not in hp:
            raise KeyError(f"global_best_hparams에 '{k}' 키가 없습니다.")

    # 중요: expected와 동일한 아키텍처로 teacher를 만들어야 KD 호환이 됩니다.
    print(f"[teacher] fixed hparams from expected tune-log: {hp}", flush=True)

    # ---- 데이터 전처리/모델 구성 ----
    word_dict, category, subcategory, news_words, news_body, news_v, news_sv, news_index = (
        preprocess_news_file(expected_bodies_train=None, expected_bodies_test=None, expected_bodies_vocab_extra=None)
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
        f"[teacher] train samples={len(all_train_id)}  test samples={len(all_test_id)}  news={len(news_index)}",
        flush=True,
    )
    print("[teacher] 학습과 평가 모두 실제본문(actual body)만 사용합니다.", flush=True)

    # ---- 기존 튜닝 스크립트의 run_trial 로직 재사용 ----
    import naml_tune_actual as nta

    if args.num_runs < 1:
        raise ValueError("--num-runs 는 1 이상이어야 합니다.")

    best_overall_mrr = float("-inf")
    best_overall_metrics: Optional[Dict[str, Any]] = None
    best_overall_model = None
    best_overall_run = 1
    run_summaries = []

    for run_idx in range(args.num_runs):
        run_no = run_idx + 1
        run_seed = int(args.seed) + run_idx * int(args.run_seed_step)
        print(f"\n=== teacher run {run_no}/{args.num_runs} (seed={run_seed}) ===", flush=True)
        np.random.seed(run_seed)
        tf.random.set_seed(run_seed)

        best_mrr, best_metrics, model = nta.run_trial(
            hp=hp,
            epochs=args.epochs,
            batch_size=args.batch_size,
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
            trial_seed=run_seed,
        )
        run_summaries.append(
            {
                "run": run_no,
                "seed": run_seed,
                "best_mrr": float(best_mrr),
                "best_metrics": dict(best_metrics),
            }
        )
        print(
            f"[teacher run {run_no}] MRR={float(best_mrr):.6f}  "
            f"NDCG@5={float(best_metrics['NDCG@5']):.6f}  Hit@1={float(best_metrics['Hit@1']):.6f}",
            flush=True,
        )
        if best_mrr > best_overall_mrr:
            best_overall_mrr = float(best_mrr)
            best_overall_metrics = dict(best_metrics)
            best_overall_model = model
            best_overall_run = run_no

    out_w = _resolve_project_path(args.out_weights)
    out_dir = os.path.dirname(out_w)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    if best_overall_model is None or best_overall_metrics is None:
        raise RuntimeError("저장할 teacher 모델이 없습니다.")

    # run_trial 내부에서 각 run의 best epoch 가중치로 set_weights까지 해둠
    best_overall_model.save_weights(out_w)

    print(
        f"\n[teacher] saved best weights -> {out_w}\n"
        f"  best_run={best_overall_run}/{args.num_runs}\n"
        f"  best_mrr={float(best_overall_mrr):.6f}\n"
        f"  best_metrics: MRR={float(best_overall_metrics['MRR']):.6f}  "
        f"NDCG@5={float(best_overall_metrics['NDCG@5']):.6f}  "
        f"Hit@1={float(best_overall_metrics['Hit@1']):.6f}",
        flush=True,
    )

    if args.out_log:
        out_log = _resolve_project_path(args.out_log)
        os.makedirs(os.path.dirname(out_log) or ".", exist_ok=True)
        with open(out_log, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "mind_dataset_subdir": args.mind_dataset_subdir,
                    "tune_log": args.tune_log,
                    "fixed_hparams": hp,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "seed": args.seed,
                    "num_runs": args.num_runs,
                    "run_seed_step": args.run_seed_step,
                    "best_run": best_overall_run,
                    "best_mrr": float(best_overall_mrr),
                    "best_metrics": best_overall_metrics,
                    "runs": run_summaries,
                    "out_weights": out_w,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )


if __name__ == "__main__":
    main()

