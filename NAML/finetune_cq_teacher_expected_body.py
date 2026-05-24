#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
이미 학습된 CQ 교사(actual-body 등) 가중치를 로드한 뒤, 기대본문으로 추가 학습(파인튜닝).

- 그래프: `build_naml_models_candidate_query_user` (naml_tune_actual_cq_teacher 와 동일)
- 아키텍처: `--tune-log` 의 global_best_hparams (CQ actual 튜닝 로그 권장)
- 학습·평가: 기대본문 (학습은 --expected-body-first-n-sentences, 평가는 전체 본문)

예시 (MIND):

  python NAML/finetune_cq_teacher_expected_body.py \
    --init-weights saved_models/MIND_2000/NAML_cq_teacher_mind_2000_actual.h5 \
    --tune-log saved_models/MIND_2000/naml_tune_actual_cq_teacher_log.json \
    --expected-train-dir user_preference/expected_body/MIND_2000/train_3cluster_11_13_8_rawtitle \
    --expected-test-dir user_preference/expected_body/MIND_2000/test_3cluster_11_13_8_rawtitle \
    --expected-body-first-n-sentences 3 \
    --learning-rate 1e-4 \
    --mind-dataset-subdir MIND_2000 \
    --epochs 10 \
    --batch-size 64 \
    --num-runs 3 \
    --out-weights saved_models/MIND_2000/NAML_cq_teacher_finetuned_expected.h5 \
    --out-log saved_models/MIND_2000/finetune_cq_teacher_expected_log.json

예시 (Adressa):

  python NAML/finetune_cq_teacher_expected_body.py \
    --init-weights saved_models/Adressa_2000/NAML_cq_teacher_adressa_2000_actual.h5 \
    --tune-log saved_models/Adressa_2000/naml_tune_actual_cq_teacher_log.json \
    --expected-train-dir user_preference/expected_body/Adressa_2000/train_3cluster_15_10_1 \
    --expected-test-dir user_preference/expected_body/Adressa_2000/test_3cluster_15_10_1 \
    --expected-body-first-n-sentences 0 \
    --learning-rate 5e-5 \
    --mind-dataset-subdir Adressa_2000 \
    --epochs 10 \
    --out-weights saved_models/Adressa_2000/NAML_cq_teacher_finetuned_expected.h5 \
    --out-log saved_models/Adressa_2000/finetune_cq_teacher_expected_log.json
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

apply_dataset_env_from_argv()

import naml_common as _naml_common
from naml_common import MAX_HISTORY_CLICKS, get_embedding, preprocess_news_file, preprocess_user_file
from naml_eval_test import _arch_from_tune_log
from naml_tune_actual_cq_teacher import run_trial_cq
from naml_tune_expected import _resolve_expected_body_dir, load_expected_bodies_from_dir

_REQUIRED_HP_KEYS = (
    "learning_rate",
    "dropout_rate",
    "cnn_filters",
    "cnn_kernel_size",
    "attention_dense_dim",
    "category_emb_dim",
)


def _resolve_project_path(p: str) -> str:
    p = p.strip()
    return os.path.normpath(p) if os.path.isabs(p) else os.path.normpath(str(_ROOT / p))


def _load_hparams_from_tune_log(tune_log_path: str) -> Dict[str, Any]:
    path = _resolve_project_path(tune_log_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"--tune-log 파일 없음: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    gb = data.get("global_best_hparams")
    if not isinstance(gb, dict):
        raise ValueError(f"tune-log에서 global_best_hparams(dict)를 찾지 못했습니다: {path}")
    return dict(gb)


def _build_hp(
    tune_log: str,
    learning_rate: Optional[float],
    learning_rate_scale: float,
    dropout_rate: Optional[float],
    cnn_filters: Optional[int],
    cnn_kernel_size: Optional[int],
    attention_dense_dim: Optional[int],
    category_emb_dim: Optional[int],
) -> Dict[str, Any]:
    hp = _load_hparams_from_tune_log(tune_log)
    for k in _REQUIRED_HP_KEYS:
        if k not in hp:
            raise KeyError(f"global_best_hparams에 '{k}' 키가 없습니다.")
    base_lr = float(hp["learning_rate"])
    if learning_rate is not None:
        hp["learning_rate"] = float(learning_rate)
    else:
        hp["learning_rate"] = base_lr * float(learning_rate_scale)
    if dropout_rate is not None:
        hp["dropout_rate"] = float(dropout_rate)
    if cnn_filters is not None:
        hp["cnn_filters"] = int(cnn_filters)
    if cnn_kernel_size is not None:
        hp["cnn_kernel_size"] = int(cnn_kernel_size)
    if attention_dense_dim is not None:
        hp["attention_dense_dim"] = int(attention_dense_dim)
    if category_emb_dim is not None:
        hp["category_emb_dim"] = int(category_emb_dim)
    return hp


def main() -> None:
    ap = argparse.ArgumentParser(
        description="CQ 교사 가중치 로드 후 기대본문으로 파인튜닝 (고정 hparam 1조합)"
    )
    ap.add_argument(
        "--init-weights",
        type=str,
        required=True,
        help="파인튜닝 시작 CQ 교사 .h5 (예: naml_tune_actual_cq_teacher actual-body 산출물)",
    )
    ap.add_argument(
        "--tune-log",
        type=str,
        required=True,
        help="CQ 튜닝 JSON (global_best_hparams). init-weights 학습 시와 동일 아키텍처여야 함",
    )
    ap.add_argument("--expected-train-dir", type=str, required=True)
    ap.add_argument("--expected-test-dir", type=str, required=True)
    ap.add_argument(
        "--expected-body-first-n-sentences",
        type=int,
        default=3,
        help="학습 시 기대본문 앞 N문장만 (0=전체). 평가는 항상 전체 본문",
    )
    ap.add_argument("--mind-dataset-subdir", type=str, default="MIND_2000")
    ap.add_argument(
        "--max-history-clicks",
        type=int,
        default=None,
        metavar="N",
        help="클릭 히스토리 최대 길이(기본 50)",
    )
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="매 run마다 init-weights에서 다시 로드 후 학습 (기본 3)",
    )
    ap.add_argument("--run-seed-step", type=int, default=1)
    ap.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Adam 학습률. 미지정 시 tune-log LR × --learning-rate-scale",
    )
    ap.add_argument(
        "--learning-rate-scale",
        type=float,
        default=0.1,
        help="미지정 --learning-rate 일 때 tune-log LR에 곱함 (기본 0.1)",
    )
    ap.add_argument("--dropout-rate", type=float, default=None)
    ap.add_argument("--cnn-filters", type=int, default=None)
    ap.add_argument("--cnn-kernel-size", type=int, default=None)
    ap.add_argument("--attention-dense-dim", type=int, default=None)
    ap.add_argument("--category-emb-dim", type=int, default=None)
    ap.add_argument(
        "--history-body-title-only",
        action="store_true",
        help="히스토리 본문 슬롯은 제목만. 후보는 기대본문",
    )
    ap.add_argument(
        "--out-weights",
        type=str,
        required=True,
        help="테스트 MRR 최고 run의 가중치 저장 경로",
    )
    ap.add_argument("--out-log", type=str, default=None, help="학습 요약 JSON (선택)")
    args = ap.parse_args()

    apply_dataset_env_from_argv(["--mind-dataset-subdir", args.mind_dataset_subdir])
    os.environ["MIND_DATASET_SUBDIR"] = args.mind_dataset_subdir

    init_w = _resolve_project_path(args.init_weights)
    if not os.path.isfile(init_w):
        print(f"오류: --init-weights 없음: {init_w}", file=sys.stderr)
        sys.exit(1)

    hp = _build_hp(
        args.tune_log,
        args.learning_rate,
        args.learning_rate_scale,
        args.dropout_rate,
        args.cnn_filters,
        args.cnn_kernel_size,
        args.attention_dense_dim,
        args.category_emb_dim,
    )

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    train_dir = _resolve_expected_body_dir(args.expected_train_dir)
    test_dir = _resolve_expected_body_dir(args.expected_test_dir)
    if not train_dir:
        print(f"오류: --expected-train-dir 없음: {args.expected_train_dir}", file=sys.stderr)
        sys.exit(1)
    if not test_dir:
        print(f"오류: --expected-test-dir 없음: {args.expected_test_dir}", file=sys.stderr)
        sys.exit(1)

    _naml_common.EXPECTED_BODY_FIRST_N_SENTENCES = max(0, int(args.expected_body_first_n_sentences))
    expected_bodies_train = load_expected_bodies_from_dir(train_dir)
    expected_bodies_test = load_expected_bodies_from_dir(test_dir)
    if not expected_bodies_train:
        print(f"오류: train 기대본문 0개: {train_dir}", file=sys.stderr)
        sys.exit(1)
    if not expected_bodies_test:
        print(f"오류: test 기대본문 0개: {test_dir}", file=sys.stderr)
        sys.exit(1)

    n_clip = _naml_common.EXPECTED_BODY_FIRST_N_SENTENCES
    train_clip_msg = f"학습 앞 {n_clip}문장" if n_clip > 0 else "학습 전체 문장"
    print(
        f"[CQ finetune] dataset={args.mind_dataset_subdir}  init={init_w}\n"
        f"  hparams={hp}\n"
        f"  expected train={len(expected_bodies_train)} ({train_dir})\n"
        f"  expected test={len(expected_bodies_test)} ({test_dir})\n"
        f"  {train_clip_msg}, 평가=전체",
        flush=True,
    )

    arch_from_log = _arch_from_tune_log(_resolve_project_path(args.tune_log))
    if arch_from_log:
        for k in ("dropout_rate", "cnn_filters", "cnn_kernel_size", "attention_dense_dim", "category_emb_dim"):
            if k in arch_from_log and k in hp and arch_from_log[k] != hp[k]:
                print(f"  경고: tune-log 아키텍처와 hp 불일치 {k}: log={arch_from_log[k]} hp={hp[k]}", flush=True)

    word_dict, category, subcategory, news_words, news_body, news_v, news_sv, news_index = preprocess_news_file(
        expected_bodies_train=expected_bodies_train,
        expected_bodies_test=expected_bodies_test,
        expected_bodies_vocab_extra=None,
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
        expected_bodies_train=expected_bodies_train,
        expected_bodies_test=expected_bodies_test,
        word_dict=word_dict,
    )

    embedding_mat = get_embedding(word_dict)
    print(
        f"  train 샘플={len(all_train_id)}  test 행={len(all_test_id)}  뉴스={len(news_index)}",
        flush=True,
    )

    if args.num_runs < 1:
        raise ValueError("--num-runs 는 1 이상이어야 합니다.")

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
        news_index=news_index,
        all_train_pn=all_train_pn,
        all_label=all_label,
        all_train_id=all_train_id,
        all_user_pos=all_user_pos,
        all_train_userid_str=all_train_userid_str,
        all_train_newsid_str=all_train_newsid_str,
        all_test_pn=all_test_pn,
        all_test_label=all_test_label,
        all_test_id=all_test_id,
        all_test_user_pos=all_test_user_pos,
        all_test_index=all_test_index,
        all_test_userid_str=all_test_userid_str,
        all_test_newsid_str=all_test_newsid_str,
        use_expected_body=True,
        expected_bodies_train=expected_bodies_train,
        expected_bodies_test=expected_bodies_test,
        history_body_title_only=bool(args.history_body_title_only),
        init_weights=init_w,
    )

    best_overall_mrr = float("-inf")
    best_overall_metrics: Optional[Dict[str, Any]] = None
    best_overall_model = None
    best_overall_run = 1
    run_summaries = []

    for run_idx in range(args.num_runs):
        run_no = run_idx + 1
        run_seed = int(args.seed) + run_idx * int(args.run_seed_step)
        print(f"\n=== CQ finetune run {run_no}/{args.num_runs} (seed={run_seed}) ===", flush=True)
        best_mrr, best_metrics, model = run_trial_cq(
            hp,
            args.epochs,
            run_seed,
            **trial_kw,
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
            f"[run {run_no}] MRR={float(best_mrr):.6f}  "
            f"NDCG@5={float(best_metrics['NDCG@5']):.6f}  "
            f"Hit@1={float(best_metrics['Hit@1']):.6f}",
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
        raise RuntimeError("저장할 모델이 없습니다.")

    best_overall_model.save_weights(out_w)
    print(
        f"\n[CQ finetune] saved -> {out_w}\n"
        f"  best_run={best_overall_run}/{args.num_runs}\n"
        f"  MRR={best_overall_metrics['MRR']:.6f}  "
        f"NDCG@5={best_overall_metrics['NDCG@5']:.6f}  "
        f"Hit@1={best_overall_metrics['Hit@1']:.6f}",
        flush=True,
    )

    if args.out_log:
        out_log = _resolve_project_path(args.out_log)
        os.makedirs(os.path.dirname(out_log) or ".", exist_ok=True)
        with open(out_log, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "task": "cq_teacher_finetune_expected_body",
                    "model_user_encoder": "candidate_query_cross_attention",
                    "mind_dataset_subdir": args.mind_dataset_subdir,
                    "max_history_clicks": int(MAX_HISTORY_CLICKS),
                    "init_weights": init_w,
                    "tune_log": _resolve_project_path(args.tune_log),
                    "hparams": hp,
                    "expected_train_dir": train_dir,
                    "expected_test_dir": test_dir,
                    "expected_body_first_n_sentences": int(args.expected_body_first_n_sentences),
                    "history_body_title_only": bool(args.history_body_title_only),
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "seed": args.seed,
                    "num_runs": args.num_runs,
                    "run_seed_step": args.run_seed_step,
                    "learning_rate_scale": float(args.learning_rate_scale),
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
        print(f"  log -> {out_log}", flush=True)


if __name__ == "__main__":
    main()
