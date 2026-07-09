#!/usr/bin/env python
# -*- coding: utf-8 -*-
# MIND_2000: --mind-dataset-subdir MIND_2000
# Adressa_2000: --mind-dataset-subdir Adressa_2000
"""
배치 인덱스 N = start .. end 에 대해 순차 실행:

  1) body_generation/generate_body_cluster_train_batches.py  (--batch-index N)
  2) NAML/eval_cluster_batch.py  (--batch-index N, --train-body-dir .../cluster{C}_batch{N}, --result-index N)
  3) coordinator_LLM/coordinator.py  (--n N)

Ablation (한 가지 구성 요소만 제거):
  full            — 기본 (클러스터 + 취향 + coordinator 정책)
  no_policy       — coordinator 정책 없이 기대본문 (취향 O)
  no_cluster      — 클러스터 없이 전체 train 유저/세션 (--full-train)
  no_preference   — preference_profile 없이 (히스토리 + 정책만)

출력 경로 (ablation != full):
  body_generation/output/<dataset>/ablation_<name>/...
  NAML/results/ablation_<name>/resultN.txt
  coordinator_LLM/ablations/<name>/output/N.txt  (시드: coordinator_LLM/output/0.txt 등에서 복사)

eval 은 resultN.txt 로 저장해 coordinator 가 resultN.txt 와 coordinator_LLM/.../output/N.txt 를 짝지어 읽도록 맞춤.
coordinator 는 응답을 (N+1).txt 로 저장 (기존 coordinator 동작).

프로젝트 루트에서:
  python scripts/run_cluster_batch_pipeline.py --start 0 --end 2
  python scripts/run_cluster_batch_pipeline.py --start 0 --end 5 --cluster-id 0 --mind-dataset-subdir Adressa_2000
  python scripts/run_cluster_batch_pipeline.py --start 0 --end 5 --ablation no_policy --cluster-id 0
  python scripts/run_cluster_batch_pipeline.py --start 0 --end 5 --ablation no_cluster
  python scripts/run_cluster_batch_pipeline.py --start 0 --end 5 --ablation no_preference --cluster-id 0
  python scripts/run_cluster_batch_pipeline.py --start 0 --end 11 --cluster-id 0 --mind-dataset-subdir MIND_2000 \
    --weights saved_models/MIND_2000/NAML_cq_teacher_mind_2000_actual.h5 \
    --tune-log saved_models/MIND_2000/naml_tune_actual_cq_teacher_log.json \
    --cq-user-encoder
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "NAML") not in sys.path:
    sys.path.insert(0, str(_ROOT / "NAML"))

from body_generation.ablation_config import (
    ABLATION_CHOICES,
    body_output_dir,
    coordinator_output_dir,
    naml_results_dir,
    normalize_ablation,
    seed_coordinator_policy,
    uses_full_train,
)
from naml_dataset_env import default_naml_eval_weights, default_user_kmeans_csv


def _run(cmd: List[str], env: Optional[dict] = None) -> None:
    print("\n" + "=" * 72)
    print(" ", " ".join(cmd))
    print("=" * 72 + "\n", flush=True)
    r = subprocess.run(cmd, cwd=str(_ROOT), env=env)
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def main() -> None:
    p = argparse.ArgumentParser(description="클러스터 배치 생성 → NAML 평가 → 조율기 LLM 을 N 범위로 순차 실행")
    p.add_argument("--start", type=int, required=True, help="배치 인덱스 시작 (포함)")
    p.add_argument("--end", type=int, required=True, help="배치 인덱스 끝 (포함)")
    p.add_argument(
        "--ablation",
        type=str,
        default="full",
        choices=ABLATION_CHOICES,
        help="full | no_policy | no_cluster | no_preference",
    )
    p.add_argument(
        "--cluster-id",
        type=int,
        default=0,
        help="클러스터 번호 (기본 0). --ablation no_cluster 이면 무시",
    )
    p.add_argument(
        "--full-train",
        action="store_true",
        help="(레거시) --ablation no_cluster 와 동일 효과",
    )
    p.add_argument(
        "--cluster-csv",
        type=str,
        default=None,
        help="기본: --mind-dataset-subdir 가 Adressa 이면 NAML/user_kmeans_k3_Adressa_2000.csv, 아니면 MIND CSV",
    )
    p.add_argument("--mind-dataset-subdir", type=str, default="MIND_2000")
    p.add_argument(
        "--cuda-visible-devices",
        type=str,
        default="1",
        help="eval_cluster_batch 시 CUDA_VISIBLE_DEVICES (기본 1)",
    )
    p.add_argument(
        "--no-cuda-env",
        action="store_true",
        help="CUDA_DEVICE_ORDER / CUDA_VISIBLE_DEVICES 를 설정하지 않음 (CPU 등)",
    )
    p.add_argument(
        "--skip-generate",
        action="store_true",
        help="1단계(기대본문 생성) 생략",
    )
    p.add_argument("--skip-eval", action="store_true", help="2단계(NAML 평가) 생략")
    p.add_argument("--skip-coordinator", action="store_true", help="3단계(조율기) 생략")
    p.add_argument(
        "--weights",
        type=str,
        default=None,
        help="eval_cluster_batch --weights (예: saved_models/MIND_2000/NAML_cq_teacher_mind_2000_actual.h5)",
    )
    p.add_argument(
        "--tune-log",
        type=str,
        default=None,
        help="eval_cluster_batch --tune-log (예: saved_models/MIND_2000/naml_tune_actual_cq_teacher_log.json)",
    )
    p.add_argument(
        "--cq-user-encoder",
        action="store_true",
        help="eval_cluster_batch --cq-user-encoder (CQ 교사/학생 가중치 평가 시 필수)",
    )
    p.add_argument(
        "--sessions-per-batch",
        type=int,
        default=300,
        metavar="S",
        help="배치당 트레이닝 세션 수 (generate·eval 공통, 기본 300)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="B",
        help="eval_cluster_batch NAML forward 미니배치 크기 (기본 64)",
    )
    args = p.parse_args()

    if args.end < args.start:
        print("오류: --end 는 --start 보다 작을 수 없습니다.", file=sys.stderr)
        sys.exit(2)
    if args.sessions_per_batch < 1:
        print("오류: --sessions-per-batch 는 1 이상이어야 합니다.", file=sys.stderr)
        sys.exit(2)
    if args.batch_size < 1:
        print("오류: --batch-size 는 1 이상이어야 합니다.", file=sys.stderr)
        sys.exit(2)

    ablation = normalize_ablation(args.ablation)
    if args.full_train:
        ablation = "no_cluster"

    py = sys.executable
    sub = args.mind_dataset_subdir
    if args.cluster_csv is None:
        args.cluster_csv = default_user_kmeans_csv(sub)
    if args.weights is None:
        args.weights = default_naml_eval_weights(sub)

    full_train = uses_full_train(ablation)
    coord_out = coordinator_output_dir(ablation)
    results_dir = naml_results_dir(ablation)
    seed_coordinator_policy(ablation, args.start)

    print(f"[pipeline] ablation={ablation}, dataset={sub}, full_train={full_train}", flush=True)

    for n in range(args.start, args.end + 1):
        train_body = body_output_dir(sub, ablation, n, cluster_id=args.cluster_id)

        if not args.skip_generate:
            gen_cmd = [
                py,
                str(_ROOT / "body_generation" / "generate_body_cluster_train_batches.py"),
                "--batch-index",
                str(n),
                "--mind-dataset-subdir",
                sub,
                "--ablation",
                ablation,
            ]
            if full_train:
                gen_cmd.append("--full-train")
            else:
                gen_cmd.extend(
                    [
                        "--cluster-csv",
                        args.cluster_csv,
                        "--cluster-id",
                        str(args.cluster_id),
                    ]
                )
            gen_cmd.extend(["--sessions-per-batch", str(args.sessions_per_batch)])
            _run(gen_cmd)

        if not args.skip_eval:
            eval_cmd = [
                py,
                str(_ROOT / "NAML" / "eval_cluster_batch.py"),
                "--batch-index",
                str(n),
                "--train-body-dir",
                train_body,
                "--result-index",
                str(n),
                "--weights",
                args.weights,
                "--mind-dataset-subdir",
                sub,
                "--results-dir",
                results_dir,
            ]
            if full_train:
                eval_cmd.append("--full-train")
            else:
                eval_cmd.extend(
                    [
                        "--cluster-csv",
                        args.cluster_csv,
                        "--cluster-id",
                        str(args.cluster_id),
                    ]
                )
            eval_cmd.extend(
                [
                    "--sessions-per-batch",
                    str(args.sessions_per_batch),
                    "--batch-size",
                    str(args.batch_size),
                ]
            )
            if args.tune_log:
                eval_cmd.extend(["--tune-log", args.tune_log])
            if args.cq_user_encoder:
                eval_cmd.append("--cq-user-encoder")
            env = os.environ.copy()
            if not args.no_cuda_env:
                env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
            _run(eval_cmd, env=env)

        if not args.skip_coordinator:
            if ablation == "no_policy":
                print(
                    f"[배치 {n}] coordinator 단계: no_policy ablation 에서도 정책 갱신은 실행 "
                    f"(다음 배치 생성에는 정책 미사용).",
                    flush=True,
                )
            _run(
                [
                    py,
                    str(_ROOT / "coordinator_LLM" / "coordinator.py"),
                    "--n",
                    str(n),
                    "--output_dir",
                    str(_ROOT / coord_out),
                    "--results_dir",
                    str(_ROOT / results_dir),
                ]
            )

        print(f"\n>>> 배치 {n} 완료 (ablation={ablation}, 다음: {n + 1})\n", flush=True)

    print("전체 파이프라인 종료.")


if __name__ == "__main__":
    main()
