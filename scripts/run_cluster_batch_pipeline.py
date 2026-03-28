#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
배치 인덱스 N = start .. end 에 대해 순차 실행:

  1) body_generation/generate_body_cluster_train_batches.py  (--batch-index N)
  2) NAML/eval_cluster_batch.py  (--batch-index N, --train-body-dir .../cluster{C}_batch{N}, --result-index N)
  3) coordinator_LLM/coordinator.py  (--n N)

eval 은 resultN.txt 로 저장해 coordinator 가 resultN.txt 와 coordinator_LLM/output/N.txt 를 짝지어 읽도록 맞춤.
coordinator 는 응답을 (N+1).txt 로 저장 (기존 coordinator 동작).

사전 조건: 시작 N에 대해 coordinator_LLM/output/{N}.txt 가 있어야 함 (예: N=0 이면 0.txt).

프로젝트 루트에서:
  python scripts/run_cluster_batch_pipeline.py --start 0 --end 2
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/run_cluster_batch_pipeline.py --start 0 --end 5 --cluster-id 0

  # 클러스터 없이 전체 트레이닝 세션 (출력 .../fulltrain_batch<N>/)
  python scripts/run_cluster_batch_pipeline.py --start 0 --end 5 --full-train --sessions-per-batch 500
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

_ROOT = Path(__file__).resolve().parent.parent


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
        "--cluster-id",
        type=int,
        default=0,
        help="클러스터 번호 (기본 0). --full-train 이면 사용하지 않음",
    )
    p.add_argument(
        "--full-train",
        action="store_true",
        help="클러스터 없이 전체 트레이닝 세션 (출력 fulltrain_batch<N>, generate/eval 에 --full-train 전달)",
    )
    p.add_argument("--cluster-csv", type=str, default="NAML/user_kmeans_k3_MIND_2000.csv")
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
        default="saved_models/NAML_mind_2000.h5",
        help="eval_cluster_batch --weights (프로젝트 루트 기준)",
    )
    p.add_argument(
        "--sessions-per-batch",
        type=int,
        default=300,
        metavar="S",
        help="배치 인덱스 하나당 포함할 트레이닝 세션 수 (generate·eval 공통, 기본 300)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
        metavar="B",
        help="eval_cluster_batch NAML forward 미니배치 크기 (기본 16)",
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

    py = sys.executable
    sub = args.mind_dataset_subdir
    cid = args.cluster_id

    for n in range(args.start, args.end + 1):
        if args.full_train:
            train_body = f"body_generation/output/{sub}/fulltrain_batch{n}"
        else:
            train_body = f"body_generation/output/{sub}/cluster{cid}_batch{n}"

        if not args.skip_generate:
            gen_cmd = [
                py,
                str(_ROOT / "body_generation" / "generate_body_cluster_train_batches.py"),
                "--batch-index",
                str(n),
                "--mind-dataset-subdir",
                sub,
            ]
            if args.full_train:
                gen_cmd.append("--full-train")
            else:
                gen_cmd.extend(
                    [
                        "--cluster-csv",
                        args.cluster_csv,
                        "--cluster-id",
                        str(cid),
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
            ]
            if args.full_train:
                eval_cmd.append("--full-train")
            else:
                eval_cmd.extend(
                    [
                        "--cluster-csv",
                        args.cluster_csv,
                        "--cluster-id",
                        str(cid),
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
            env = os.environ.copy()
            if not args.no_cuda_env:
                env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
            _run(eval_cmd, env=env)

        if not args.skip_coordinator:
            _run(
                [
                    py,
                    str(_ROOT / "coordinator_LLM" / "coordinator.py"),
                    "--n",
                    str(n),
                ]
            )

        print(f"\n>>> 배치 {n} 완료 (다음 루프: {n + 1})\n", flush=True)

    print("전체 파이프라인 종료.")


if __name__ == "__main__":
    main()
