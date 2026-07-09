#!/usr/bin/env python
# -*- coding: utf-8 -*-
# MIND_2000: --mind-dataset-subdir MIND_2000
# Adressa_2000: --mind-dataset-subdir Adressa_2000
"""
트레이닝셋 유저 클러스터 CSV(user_id, cluster)에 따라 클러스터마다 서로 다른 정책 JSON으로
기대본문을 한 번에 생성하고, 결과는 하나의 출력 폴더에 저장 (user_<id>/news_<id>.json).

NAML preprocess_user_file 과 동일한 트레이닝 세션 순서·(user,후보) 쌍 규칙을 사용합니다.

프로젝트 루트에서:
  set PYTHONPATH=NAML
  python body_generation/generate_body_train_cluster_policies.py \
    --cluster-csv NAML/user_kmeans_k3_MIND_2000.csv \
    --policy-files coordinator_LLM/output/0.txt coordinator_LLM/output/1.txt coordinator_LLM/output/2.txt \
    --output body_generation/output/MIND_2000/train_cluster_all \
    --mind-dataset-subdir MIND_2000

--policy-files 순서: 클러스터 0, 1, 2, ... 용 (CSV의 cluster id가 0 .. len(policy-files)-1 이어야 함)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "NAML") not in sys.path:
    sys.path.insert(0, str(_ROOT / "NAML"))
if str(_ROOT / "body_generation") not in sys.path:
    sys.path.insert(0, str(_ROOT / "body_generation"))

import generate_body as gb

from naml_dataset_env import default_user_kmeans_csv


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


def collect_pairs_train_by_cluster(
    all_train_userid_str,
    all_train_newsid_str,
    n_train: int,
    user_cluster: Dict[str, int],
    news_dict: dict,
) -> Dict[int, List[Tuple[int, str]]]:
    """학습 세션별 (user, 후보 뉴스 id) 수집, 클러스터별로 중복 제거."""
    buckets: Dict[int, Set[Tuple[int, str]]] = defaultdict(set)
    for i in range(n_train):
        u_norm = _norm_uid(all_train_userid_str[i])
        if u_norm not in user_cluster:
            continue
        cl = user_cluster[u_norm]
        try:
            uid_int = int(float(u_norm))
        except ValueError:
            continue
        for nid in all_train_newsid_str[i]:
            ns = str(nid).strip() if nid is not None else ""
            if not ns or ns not in news_dict:
                continue
            buckets[cl].add((uid_int, ns))
    return {c: list(s) for c, s in buckets.items()}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="트레이닝셋 클러스터별 정책 파일로 기대본문 일괄 생성 → 단일 출력 폴더"
    )
    ap.add_argument(
        "--cluster-csv",
        type=str,
        default=None,
        help="user_id, cluster 형식 CSV. 기본: 데이터셋이 Adressa 이면 user_kmeans_k3_Adressa_2000.csv",
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
        help="모든 생성 결과를 넣을 단일 폴더",
    )
    ap.add_argument("--mind-dataset-subdir", type=str, default=None)
    ap.add_argument("--api-key", type=str, default=None)
    ap.add_argument("--model", type=str, default="gpt-4o-mini")
    ap.add_argument("--dry-run", action="store_true", help="쌍 집계만 하고 API 호출 없음")
    args = ap.parse_args()

    sub = gb._resolve_mind_dataset_subdir(args.mind_dataset_subdir)
    if args.cluster_csv is None:
        args.cluster_csv = default_user_kmeans_csv(sub)

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

    os.environ["MIND_DATASET_SUBDIR"] = sub

    from naml_common import preprocess_news_file, preprocess_user_file

    print(f"데이터셋: dataset/{sub}/")
    print(f"클러스터 CSV: {csv_path}")
    print(f"출력 폴더: {out_dir}")

    word_dict, category, subcategory, news_words, news_body, news_v, news_sv, news_index = (
        preprocess_news_file(expected_bodies_train=None, expected_bodies_test=None)
    )
    (
        _userid_dict,
        _all_train_pn,
        all_label,
        _all_train_id,
        _all_test_pn,
        _all_test_label,
        _all_test_id,
        _all_user_pos,
        _all_test_user_pos,
        _all_test_index,
        _cand_tr,
        _cand_te,
        all_train_userid_str,
        all_train_newsid_str,
        _all_test_userid_str,
        _all_test_newsid_str,
    ) = preprocess_user_file(
        news_index=news_index,
        expected_bodies_train=None,
        expected_bodies_test=None,
        word_dict=word_dict,
    )

    n_train = len(all_label)

    gen = gb.BodyGenerator(
        api_key=args.api_key,
        model=args.model,
        use_test=False,
        mind_dataset_subdir=args.mind_dataset_subdir,
        coordinator_policy_path=policy_paths[0],
        coordinator_policy_n=None,
    )

    buckets = collect_pairs_train_by_cluster(
        all_train_userid_str,
        all_train_newsid_str,
        n_train,
        user_cluster,
        gen.news_dict,
    )
    print(
        f"클러스터별 (user,후보) 쌍 수: "
        f"{{{', '.join(f'{k}: {len(v)}' for k, v in sorted(buckets.items()))}}}"
    )

    if args.dry_run:
        print("--dry-run 이므로 생성하지 않습니다.")
        return

    merged_results: List[dict] = []
    for cl in sorted(buckets.keys()):
        pairs = buckets[cl]
        if not pairs:
            continue
        pf = policy_paths[cl]
        print(f"\n>>> 클러스터 {cl}: 정책 {pf} — {len(pairs)}쌍 생성\n")
        gen.set_coordinator_policy_file(pf)
        part = gen.generate_bodies_for_pairs(pairs, output_dir=out_dir)
        merged_results.extend(part)

    if merged_results:
        merged_path = os.path.join(out_dir, "all_results_pairs.json")
        with open(merged_path, "w", encoding="utf-8") as f:
            json.dump(merged_results, f, ensure_ascii=False, indent=2)
        print(f"\n통합 all_results_pairs.json 저장: {merged_path} ({len(merged_results)}개 항목)")

    print("\n전체 완료.")


if __name__ == "__main__":
    main()
