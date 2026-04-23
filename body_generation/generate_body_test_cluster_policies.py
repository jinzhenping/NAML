#!/usr/bin/env python
# -*- coding: utf-8 -*-
# MIND_2000: --mind-dataset-subdir MIND_2000
# Adressa_2000: --mind-dataset-subdir Adressa_2000
"""
테스트 유저 클러스터 CSV(user_id,cluster)에 따라 클러스터마다 서로 다른 정책 JSON 파일로
기대본문을 생성하고, 결과는 하나의 출력 폴더에만 저장 (user_<id>/news_<id>.json 구조 유지).

정책 파일은 coordinator 출력과 동일 형식: updated_policy 또는 policy 키를 가진 JSON
(.txt 확장자여도 내용이 JSON이면 됨).

프로젝트 루트에서:
  python body_generation/generate_body_test_cluster_policies.py \\
    --cluster-csv NAML/user_kmeans_k3_MIND_2000_test.csv \\
    --policy-files policies/c0.json policies/c1.json policies/c2.json \\
    --output body_generation/output/MIND_2000/test_cluster_mixed_run1 \\
    --mind-dataset-subdir MIND_2000

--policy-files 순서: 클러스터 0번용, 1번용, 2번용, ... (CSV에 등장하는 cluster 값이
0 .. len(policy-files)-1 안에 있어야 함)
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "body_generation") not in sys.path:
    sys.path.insert(0, str(_ROOT / "body_generation"))

import generate_body as gb


def _norm_uid(u) -> str:
    try:
        return str(int(float(str(u).strip())))
    except (ValueError, TypeError):
        return str(u).strip()


def load_user_cluster_map(csv_path: Path) -> Dict[str, int]:
    m: Dict[str, int] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            uid = str(row.get("user_id", row.get("user", ""))).strip()
            cl = row.get("cluster", row.get("Cluster", ""))
            if not uid or cl == "":
                continue
            try:
                c = int(float(cl))
            except ValueError:
                continue
            m[_norm_uid(uid)] = c
    return m


def collect_pairs_by_cluster(
    train_df, user_cluster: Dict[str, int], news_dict: dict
) -> Dict[int, List[Tuple[int, str]]]:
    """테스트 TSV 행마다 (user, candidate_news_id) 수집, 클러스터별 중복 제거."""
    buckets: Dict[int, Set[Tuple[int, str]]] = defaultdict(set)
    for _, row in train_df.iterrows():
        uid_raw = row["user"]
        if uid_raw is None or (isinstance(uid_raw, float) and str(uid_raw) == "nan"):
            continue
        uid_norm = _norm_uid(uid_raw)
        if uid_norm not in user_cluster:
            continue
        cl = user_cluster[uid_norm]
        cand_str = str(row.get("candidate_news", "") or "")
        for cid in cand_str.split():
            ns = str(cid).strip()
            if not ns or ns not in news_dict:
                continue
            try:
                uid_int = int(uid_norm)
            except ValueError:
                continue
            buckets[cl].add((uid_int, ns))
    return {c: list(s) for c, s in buckets.items()}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="테스트셋 유저 클러스터별 서로 다른 정책 파일로 기대본문 생성 → 단일 출력 폴더"
    )
    ap.add_argument(
        "--cluster-csv",
        type=str,
        required=True,
        help="user_id,cluster 형식 CSV (예: NAML/user_kmeans_k3_MIND_2000_test.csv)",
    )
    ap.add_argument(
        "--policy-files",
        type=str,
        nargs="+",
        required=True,
        metavar="PATH",
        help="클러스터 0,1,2,... 순 정책 JSON 경로 (프로젝트 루트 기준 상대 가능)",
    )
    ap.add_argument(
        "--output",
        type=str,
        required=True,
        help="모든 생성 결과를 넣을 단일 폴더 (생성 시 없으면 만듦)",
    )
    ap.add_argument("--mind-dataset-subdir", type=str, default=None)
    ap.add_argument("--api-key", type=str, default=None)
    ap.add_argument("--model", type=str, default="gpt-4o-mini")
    ap.add_argument("--dry-run", action="store_true", help="쌍 집계만 하고 API 호출 없음")
    args = ap.parse_args()

    csv_path = _ROOT / args.cluster_csv
    if not csv_path.is_file():
        print(f"오류: cluster CSV 없음: {csv_path}")
        sys.exit(1)

    policy_paths: List[str] = []
    for p in args.policy_files:
        abs_p = str(_ROOT / p) if not os.path.isabs(p) else p
        if not os.path.isfile(abs_p):
            print(f"오류: 정책 파일 없음: {abs_p}")
            sys.exit(1)
        policy_paths.append(abs_p)

    user_cluster = load_user_cluster_map(csv_path)
    if not user_cluster:
        print("오류: CSV에서 유효한 (user, cluster) 행이 없습니다.")
        sys.exit(1)

    max_c = max(user_cluster.values())
    if max_c >= len(policy_paths):
        print(
            f"오류: CSV 최대 클러스터 id={max_c} 인데 --policy-files 가 {len(policy_paths)}개뿐입니다. "
            f"클러스터 id는 0 .. {len(policy_paths)-1} 이어야 합니다."
        )
        sys.exit(1)

    out_dir = os.path.normpath(str(_ROOT / args.output))
    os.makedirs(out_dir, exist_ok=True)

    gen = gb.BodyGenerator(
        api_key=args.api_key,
        model=args.model,
        use_test=True,
        mind_dataset_subdir=args.mind_dataset_subdir,
        coordinator_policy_path=policy_paths[0],
        coordinator_policy_n=None,
    )

    buckets = collect_pairs_by_cluster(gen.train_df, user_cluster, gen.news_dict)
    print(f"클러스터별 (user,후보) 쌍 수: {{{', '.join(f'{k}: {len(v)}' for k, v in sorted(buckets.items()))}}}")
    print(f"출력 폴더: {out_dir}")

    if args.dry_run:
        print("--dry-run 이므로 생성하지 않습니다.")
        return

    for cl in sorted(buckets.keys()):
        pairs = buckets[cl]
        if not pairs:
            continue
        pf = policy_paths[cl]
        print(f"\n>>> 클러스터 {cl}: 정책 {pf} — {len(pairs)}쌍 생성\n")
        gen.set_coordinator_policy_file(pf)
        gen.generate_bodies_for_pairs(pairs, output_dir=out_dir)

    print("\n전체 완료.")


if __name__ == "__main__":
    main()
