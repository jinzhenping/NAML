#!/usr/bin/env python
# -*- coding: utf-8 -*-
# MIND_2000: --mind-dataset-subdir MIND_2000
# Adressa_2000: --mind-dataset-subdir Adressa_2000
"""
클러스터 CSV(NAML/user_kmeans_k*.csv) + NAML preprocess_user_file 기준으로
특정 클러스터에 속한 트레이닝 세션만 골라, 세션을 N개씩 배치로 나눈 뒤
각 배치마다 coordinator_LLM/output/{batch_index}.txt 정책으로 기대본문 생성.

기본 프롬프트는 user_preference/generate_expected_body_train_cluster_policies.py 와 동일하게
user_preference/body_generation.yaml + user_preference/preference/<데이터셋>/train/user_<id>.json 의
preference_profile 을 geb.build_prompt 로 주입. 옛 방식은 --legacy-body-prompt.
Adressa_* 데이터셋이면 기대본문 노르웨이어(bokmål) 지시를 프롬프트 끝에 자동 접미(generate_body.py).

프로젝트 루트에서:
  python body_generation/generate_body_cluster_train_batches.py \\
    --cluster-csv NAML/user_kmeans_k3_MIND_2000.csv --cluster-id 0 --batch-index 0

배치 0 → 0.txt, 배치 1 → 1.txt (기본). --policy-file 로 N 덮어쓰기, --policy-path 로 JSON 파일 직접 지정(우선).
배치 없이 클러스터 전체: --all-sessions (출력은 cluster<C>_batch0, 정책 기본 0.txt).

클러스터 없이 전체 트레이닝: --full-train --batch-index N (출력 .../fulltrain_batch<N>/, CSV 불필요).
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# 프로젝트 루트
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "NAML") not in sys.path:
    sys.path.insert(0, str(_ROOT / "NAML"))
if str(_ROOT / "body_generation") not in sys.path:
    sys.path.insert(0, str(_ROOT / "body_generation"))

import generate_body as gb
from body_generation.ablation_config import (
    ABLATION_CHOICES,
    body_output_dir,
    coordinator_output_dir,
    normalize_ablation,
)
from naml_dataset_env import default_user_kmeans_csv


def _resolve_project_path(p: str) -> str:
    p = p.strip()
    return os.path.normpath(p) if os.path.isabs(p) else os.path.normpath(str(_ROOT / p))


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
        default=None,
        help="user_id,cluster CSV. --full-train 이면 사용 안 함. 기본: Adressa → user_kmeans_k3_Adressa_2000.csv",
    )
    parser.add_argument(
        "--cluster-id",
        type=int,
        default=None,
        metavar="C",
        help="대상 클러스터 번호 (예: 0). --full-train 이면 생략",
    )
    parser.add_argument(
        "--full-train",
        action="store_true",
        help="클러스터 CSV 없이 전체 트레이닝 세션을 배치로 나눔. 출력: .../fulltrain_batch<B>/",
    )
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
        help="coordinator_LLM/output/N.txt (기본: batch-index와 동일). --policy-path 가 있으면 무시",
    )
    parser.add_argument(
        "--policy-path",
        type=str,
        default=None,
        metavar="PATH",
        help="coordinator 형식 JSON 정책 파일 직접 지정 (프로젝트 루트 기준 상대 경로 가능). 지정 시 N.txt 및 --policy-file 보다 우선",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="body_generation/output",
        help="출력 루트 (--output-dir 미지정 시 <루트>/<데이터셋>/cluster<C>_batch<B>/)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="생성 JSON을 넣을 최종 폴더 (프로젝트 루트 기준 상대 경로 가능). 지정 시 --output 및 dataset/cluster*_batch* 하위 경로를 쓰지 않음",
    )
    parser.add_argument("--mind-dataset-subdir", type=str, default=None, help="dataset 하위 폴더 (예: MIND_2000)")
    parser.add_argument(
        "--legacy-body-prompt",
        action="store_true",
        help="옛 body_generation/prompt.yaml (news1~10만, 선호도 JSON 미사용). 기본은 body_generation.yaml + 선호도",
    )
    parser.add_argument(
        "--history-k",
        type=int,
        default=10,
        metavar="K",
        help="선호도 프롬프트에 넣을 최근 클릭 제목 개수 (기본 10; --legacy-body-prompt 시 무시)",
    )
    parser.add_argument(
        "--preference-base",
        type=str,
        default=None,
        help="train 선호도 JSON 디렉터리 (기본: user_preference/preference/<mind-dataset-subdir>/train)",
    )
    parser.add_argument(
        "--body-generation-yaml",
        type=str,
        default=None,
        help="geb 본문 프롬프트 템플릿 (기본: user_preference/body_generation.yaml)",
    )
    parser.add_argument(
        "--generation-settings",
        type=str,
        default=None,
        help="geb generation_settings.yaml (기본: user_preference/generation_settings.yaml)",
    )
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
    parser.add_argument(
        "--all-sessions",
        action="store_true",
        help="배치 분할 없이 해당 클러스터 트레이닝 세션 전체를 한 번에 처리 (내부적으로 batch-index=0, sessions-per-batch=세션 전체 수)",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default="full",
        choices=ABLATION_CHOICES,
        help="full | no_policy (취향 O, coordinator 정책 X) | no_cluster (전체 유저, --full-train) | no_preference (정책 O, 취향 LLM X)",
    )
    args = parser.parse_args()
    ablation = normalize_ablation(args.ablation)
    if ablation == "no_cluster":
        args.full_train = True

    sub = gb._resolve_mind_dataset_subdir(args.mind_dataset_subdir)
    if args.cluster_csv is None:
        args.cluster_csv = default_user_kmeans_csv(sub)

    if not args.full_train and args.cluster_id is None:
        parser.error("--cluster-id 는 필수입니다. (전체 트레이닝: --full-train 또는 --ablation no_cluster)")
    if args.full_train and args.cluster_id is not None and ablation != "no_cluster":
        print("경고: --full-train 이므로 --cluster-id 는 무시됩니다.", flush=True)
    if ablation != "full":
        print(f"[ablation] {ablation}", flush=True)

    if not args.batch_count_only and args.batch_index is None and not args.all_sessions:
        parser.error("--batch-index 는 필수입니다. (전체 한 번에: --all-sessions, 배치 개수만: --batch-count-only)")

    policy_path_resolved: Optional[str] = None
    if (args.policy_path or "").strip():
        policy_path_resolved = _resolve_project_path(args.policy_path)
        if not os.path.isfile(policy_path_resolved):
            print(f"오류: --policy-path 파일 없음: {policy_path_resolved}")
            sys.exit(1)

    csv_path = _ROOT / args.cluster_csv
    cluster_users: Optional[Set[str]] = None
    if not args.full_train:
        if not csv_path.is_file():
            print(f"오류: 클러스터 CSV 없음: {csv_path}")
            sys.exit(1)
        assert args.cluster_id is not None
        cluster_users = _load_cluster_users(csv_path, args.cluster_id)
        if not cluster_users:
            print(f"오류: 클러스터 {args.cluster_id}에 해당하는 user가 CSV에 없습니다.")
            sys.exit(1)

    os.environ["MIND_DATASET_SUBDIR"] = sub

    from naml_common import preprocess_news_file, preprocess_user_file

    print(f"데이터셋: dataset/{sub}/ (MIND_DATASET_SUBDIR)")
    if not args.legacy_body_prompt:
        print(
            "본문 프롬프트: 선호도 JSON + user_preference/body_generation.yaml (train_cluster_policies 와 동일 계열)",
            flush=True,
        )
    if "adressa" in sub.lower():
        print(
            "[prompt] Adressa: 기대본문 노르웨이어(bokmål) 지시가 각 API 프롬프트 끝에 자동 접미됩니다.",
            flush=True,
        )
    if args.full_train:
        print("모드: --full-train (전체 트레이닝 세션, 클러스터 미사용)")
    else:
        print(f"클러스터 CSV: {csv_path}")
        print(f"클러스터 {args.cluster_id} 유저 수 (CSV): {len(cluster_users)}")

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
    if args.full_train:
        session_indices = list(range(n_train))
    else:
        assert cluster_users is not None
        for i in range(n_train):
            u = str(all_train_userid_str[i]).strip()
            if u in cluster_users:
                session_indices.append(i)

    total_sess = len(session_indices)
    if args.full_train:
        print(f"전체 트레이닝 세션 수 (전처리 후): {total_sess}")
    else:
        print(f"해당 클러스터 트레이닝 세션 수 (전처리 후): {total_sess}")

    if args.all_sessions and not args.batch_count_only:
        args.sessions_per_batch = max(1, total_sess)
        args.batch_index = 0
        print(f"--all-sessions: 배치 분할 없이 전체 {total_sess}세션 → batch-index=0, sessions-per-batch={args.sessions_per_batch}")

    policy_n: Optional[int] = None
    if not args.batch_count_only:
        if policy_path_resolved:
            print(f"정책: --policy-path → {policy_path_resolved}")
        else:
            policy_n = args.policy_file if args.policy_file is not None else args.batch_index
            coord_dir = coordinator_output_dir(ablation)
            if ablation == "no_policy":
                print(
                    f"배치: index={args.batch_index}, 세션 {args.sessions_per_batch}개/배치, "
                    f"coordinator 정책: 사용 안 함 (ablation no_policy)"
                )
            else:
                print(
                    f"배치: index={args.batch_index}, 세션 {args.sessions_per_batch}개/배치, "
                    f"coordinator 정책: {coord_dir}/{policy_n}.txt"
                )

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

    if (args.output_dir or "").strip():
        out = (args.output_dir or "").strip()
        run_dir = os.path.normpath(out) if os.path.isabs(out) else os.path.normpath(str(_ROOT / out))
        os.makedirs(run_dir, exist_ok=True)
        print(f"저장 폴더 (--output-dir): {run_dir}")
    else:
        rel = body_output_dir(
            sub,
            ablation,
            int(args.batch_index),
            cluster_id=int(args.cluster_id or 0),
            output_root=os.path.normpath(args.output),
        )
        run_dir = os.path.normpath(str(_ROOT / rel))
        os.makedirs(run_dir, exist_ok=True)
        print(f"저장 폴더: {run_dir}")

    if args.dry_run:
        print("(--dry-run) 생성 스킵.")
        return

    gen_kw = dict(
        api_key=args.api_key,
        model=args.model,
        use_test=False,
        mind_dataset_subdir=args.mind_dataset_subdir,
        use_preference_prompt=not bool(args.legacy_body_prompt),
        preference_history_k=int(args.history_k),
        ablation=ablation,
        coordinator_output_dir=str(_ROOT / coordinator_output_dir(ablation)),
    )
    if args.preference_base:
        gen_kw["preference_base_dir"] = _resolve_project_path(args.preference_base)
    if args.body_generation_yaml:
        gen_kw["preference_body_prompt_path"] = _resolve_project_path(args.body_generation_yaml)
    if args.generation_settings:
        gen_kw["preference_generation_settings_path"] = _resolve_project_path(args.generation_settings)
    if policy_path_resolved:
        gen_kw["coordinator_policy_path"] = policy_path_resolved
        gen_kw["coordinator_policy_n"] = None
    else:
        gen_kw["coordinator_policy_n"] = policy_n
    generator = gb.BodyGenerator(**gen_kw)
    generator.generate_bodies_for_pairs(pairs, output_dir=run_dir)
    print("완료.")


if __name__ == "__main__":
    main()
