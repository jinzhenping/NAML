#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
클러스터 CSV(NAML/user_kmeans_k*.csv) + NAML preprocess_user_file 기준으로
특정 클러스터에 속한 트레이닝 세션만 골라, 세션을 N개씩 배치로 나눈 뒤
각 배치마다 coordinator_LLM/output/{batch_index}.txt 정책으로 기대본문 생성.

프로젝트 루트에서:
  python body_generation/generate_body_cluster_train_batches.py \\
    --cluster-csv NAML/user_kmeans_k3_MIND_2000.csv --cluster-id 0 --batch-index 0

배치 0 → 0.txt, 배치 1 → 1.txt (기본). --policy-file 로 덮어쓸 수 있음.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

# 프로젝트 루트
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "NAML") not in sys.path:
    sys.path.insert(0, str(_ROOT / "NAML"))
if str(_ROOT / "body_generation") not in sys.path:
    sys.path.insert(0, str(_ROOT / "body_generation"))

import generate_body as gb


def _load_cluster_users(csv_path: Path, cluster_id: int) -> Set[str]:
    users: Set[str] = set()
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
            if c == cluster_id:
                users.add(uid)
    return users


def main() -> None:
    parser = argparse.ArgumentParser(
        description="클러스터별 트레이닝 세션 배치 → coordinator N.txt 정책으로 기대본문 생성"
    )
    parser.add_argument(
        "--cluster-csv",
        type=str,
        default="NAML/user_kmeans_k3_MIND_2000.csv",
        help="user_id,cluster CSV (프로젝트 루트 기준 상대 경로)",
    )
    parser.add_argument("--cluster-id", type=int, required=True, help="대상 클러스터 번호 (예: 0)")
    parser.add_argument(
        "--batch-index",
        type=int,
        default=None,
        metavar="B",
        help="몇 번째 배치 (0부터). policy 기본값과 동일. 생략 시 --batch-count-only 와 함께만 사용",
    )
    parser.add_argument(
        "--sessions-per-batch",
        type=int,
        default=300,
        metavar="N",
        help="배치당 트레이닝 세션 수 (기본 300)",
    )
    parser.add_argument(
        "--policy-file",
        type=int,
        default=None,
        metavar="N",
        help="coordinator_LLM/output/N.txt (기본: batch-index와 동일)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="body_generation/output",
        help="출력 루트 (실제 저장은 <루트>/<데이터셋>/cluster<C>_batch<B>/)",
    )
    parser.add_argument("--mind-dataset-subdir", type=str, default=None, help="dataset 하위 폴더 (예: MIND_2000)")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="API 호출 없이 세션 수·(user,후보) 쌍 수만 출력",
    )
    parser.add_argument(
        "--batch-count-only",
        action="store_true",
        help="해당 클러스터로 나눌 수 있는 배치 개수(및 세션 수)만 출력하고 종료 (API·batch-index 불필요)",
    )
    args = parser.parse_args()

    if not args.batch_count_only and args.batch_index is None:
        parser.error("--batch-index 는 필수입니다. (배치 개수만 보려면 --batch-count-only 사용)")

    policy_n = (
        (args.policy_file if args.policy_file is not None else args.batch_index)
        if not args.batch_count_only
        else None
    )

    csv_path = _ROOT / args.cluster_csv
    if not csv_path.is_file():
        print(f"오류: 클러스터 CSV 없음: {csv_path}")
        sys.exit(1)

    cluster_users = _load_cluster_users(csv_path, args.cluster_id)
    if not cluster_users:
        print(f"오류: 클러스터 {args.cluster_id}에 해당하는 user가 CSV에 없습니다.")
        sys.exit(1)

    sub = gb._resolve_mind_dataset_subdir(args.mind_dataset_subdir)
    os.environ["MIND_DATASET_SUBDIR"] = sub

    from naml_common import preprocess_news_file, preprocess_user_file

    print(f"데이터셋: dataset/{sub}/ (MIND_DATASET_SUBDIR)")
    print(f"클러스터 CSV: {csv_path}")
    print(f"클러스터 {args.cluster_id} 유저 수 (CSV): {len(cluster_users)}")
    if not args.batch_count_only:
        print(f"배치: index={args.batch_index}, 세션 {args.sessions_per_batch}개/배치, policy_file={policy_n}.txt")

    word_dict, category, subcategory, news_words, news_body, news_v, news_sv, news_index = (
        preprocess_news_file(expected_bodies_train=None, expected_bodies_test=None)
    )
    (
        _userid_dict,
        all_train_pn,
        all_label,
        all_train_id,
        _all_test_pn,
        _all_test_label,
        _all_test_id,
        all_user_pos,
        all_test_user_pos,
        all_test_index,
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
    session_indices: List[int] = []
    for i in range(n_train):
        u = str(all_train_userid_str[i]).strip()
        if u in cluster_users:
            session_indices.append(i)

    total_sess = len(session_indices)
    print(f"해당 클러스터 트레이닝 세션 수 (전처리 후): {total_sess}")

    spb = max(1, int(args.sessions_per_batch))
    n_batches = math.ceil(total_sess / spb) if total_sess > 0 else 0
    last_batch_sessions = total_sess - (n_batches - 1) * spb if n_batches > 0 else 0
    print(
        f"배치 개수 (세션 {spb}개/배치): {n_batches}개  "
        f"→ batch-index 는 0 ~ {n_batches - 1} (마지막 배치는 {last_batch_sessions}세션)"
        if n_batches
        else "배치 개수: 0 (세션 없음)"
    )

    if args.batch_count_only:
        print("--batch-count-only 로 종료합니다.")
        return

    start = args.batch_index * args.sessions_per_batch
    end = min(start + args.sessions_per_batch, total_sess)
    if start >= total_sess:
        print(f"오류: batch-index={args.batch_index} 에 해당하는 세션이 없습니다 (시작 인덱스 {start} >= {total_sess}).")
        sys.exit(1)

    batch_sessions = session_indices[start:end]
    print(f"이번 배치 세션 인덱스 범위: 전체 중 [{start}, {end}) → {len(batch_sessions)} 세션")

    pairs: List[Tuple[int, str]] = []
    seen: Set[Tuple[int, str]] = set()
    for sidx in batch_sessions:
        uid_raw = all_train_userid_str[sidx]
        try:
            uid = int(str(uid_raw).strip())
        except ValueError:
            print(f"경고: 유저 ID 변환 실패: {uid_raw}")
            continue
        for nid in all_train_newsid_str[sidx]:
            ns = str(nid).strip() if nid is not None else ""
            if not ns:
                continue
            key = (uid, ns)
            if key in seen:
                continue
            seen.add(key)
            pairs.append((uid, ns))

    print(f"(user, 후보) 유효 쌍 수 (뉴스 메타 존재·중복 제거 전 설계상 쌍): {len(pairs)}")

    base_out = os.path.join(os.path.normpath(args.output), sub)
    os.makedirs(base_out, exist_ok=True)
    run_dir = os.path.join(base_out, f"cluster{args.cluster_id}_batch{args.batch_index}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"저장 폴더: {run_dir}")

    if args.dry_run:
        print("(--dry-run) 생성 스킵.")
        return

    generator = gb.BodyGenerator(
        api_key=args.api_key,
        model=args.model,
        use_test=False,
        coordinator_policy_n=policy_n,
        mind_dataset_subdir=args.mind_dataset_subdir,
    )
    generator.generate_bodies_for_pairs(pairs, output_dir=run_dir)
    print("완료.")


if __name__ == "__main__":
    main()
