#!/usr/bin/env python
# -*- coding: utf-8 -*-
# MIND_2000: --mind-dataset-subdir MIND_2000
# Adressa_2000: --mind-dataset-subdir Adressa_2000
"""
트레이닝셋 유저 클러스터 CSV에 따라 클러스터마다 서로 다른 정책 파일로
`generate_expected_body_from_preference.py`와 동일한 방식으로 기대본문을 생성합니다.

실행 순서: (1) `--title-abstraction`일 때만 고유 후보뉴스마다 추상 제목 생성·캐시 → (2) 쌍별 기대본문.
  기본은 (1) 생략, 후보는 MIND 원본 제목만 {candidate_news}에 사용.

- 히스토리: train TSV의 clicked_news에서 최근 history_k개 제목
- 취향: user_preference/preference/<dataset>/train/user_<id>.json (기본)
- 후보 제목: 기본은 원본만; `--title-abstraction`이면 title-abstraction-yaml로 LLM 추상화 후 {candidate_news}
- Adressa_* (`--mind-dataset-subdir`에 adressa 포함): 기대본문 LLM 프롬프트 끝에 노르웨이어(bokmål) 본문 생성 지시 자동 추가
- 정책: --policy-files 를 클러스터 0,1,2,... 순으로 매핑
- 배치: --num-batches N --batch-index i 로 전체 (유저,후보) 쌍을 N등분한 i번째만 처리
  (출력 폴더는 배치마다 다르게 주는 것을 권장: .../train_batch0 등)

프로젝트 루트에서:

  python user_preference/generate_expected_body_train_cluster_policies.py \
    --cluster-csv NAML/user_kmeans_k3_MIND_2000.csv \
    --policy-files coordinator_LLM/output_cluster0/11.txt \
                 coordinator_LLM/output_cluster1/13.txt \
                 coordinator_LLM/output_cluster2/8.txt \
    --output user_preference/expected_body/MIND_2000/train_3cluster_11_13_8 \
    --mind-dataset-subdir MIND_2000 \
    --max-run-attempts 5

python user_preference/generate_expected_body_train_cluster_policies.py \
    --cluster-csv NAML/user_kmeans_k3_Adressa_2000.csv \
    --policy-files coordinator_LLM/Adressa_2000_output_cluster0/15.txt \
                 coordinator_LLM/Adressa_2000_output_cluster1/10.txt \
                 coordinator_LLM/Adressa_2000_output_cluster2/1.txt \
    --output user_preference/expected_body/Adressa_2000/train_3cluster_15_10_1 \
    --mind-dataset-subdir Adressa_2000

  후보 제목 LLM 추상화를 쓰려면:

  ... 동일 옵션 ... --title-abstraction
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
from openai import OpenAI

_GEB_PATH = _ROOT / "user_preference" / "generate_expected_body_from_preference.py"
_spec = importlib.util.spec_from_file_location("_geb_pref", _GEB_PATH)
_geb = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_geb)

DEFAULT_MODEL = _geb.DEFAULT_MODEL
PROJECT_ROOT = _geb.PROJECT_ROOT
build_prompt = _geb.build_prompt
extra_body_prompt_suffix_for_dataset = _geb.extra_body_prompt_suffix_for_dataset
clean_abstracted_title = _geb.clean_abstracted_title
get_recent_titles = _geb.get_recent_titles
load_abstract_cache = _geb.load_abstract_cache
load_news_map = _geb.load_news_map
load_policy = _geb.load_policy
parse_settings = _geb.parse_settings
resolve_news_tsv = _geb.resolve_news_tsv
resolve_train_tsv = _geb.resolve_train_tsv
impression_tsv_header_skiprows = _geb.impression_tsv_header_skiprows
save_abstract_cache = _geb.save_abstract_cache
safe_api_text = _geb.safe_api_text


def _thread_local_openai_factory(api_key: str):
    """스레드마다 별도 OpenAI 클라이언트 (공유 클라이언트 동시 호출 시 HTTP 본문 깨짐/400 방지)."""
    local = threading.local()

    def get_client() -> OpenAI:
        if getattr(local, "client", None) is None:
            local.client = OpenAI(api_key=api_key)
        return local.client  # type: ignore[return-value]

    return get_client


def _validate_chat_json_payload(
    model: str,
    messages: List[dict],
    temperature: float,
    max_tokens: int,
) -> None:
    """클라이언트가 보내는 JSON이 RFC에 맞는지 선검증 (NaN 등으로 서버 400 방지)."""
    payload = {
        "model": str(model),
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    json.dumps(payload, ensure_ascii=False, allow_nan=False)


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
    test_df, user_cluster: Dict[str, int], news_dict: dict
) -> Dict[int, List[Tuple[int, str]]]:
    buckets: Dict[int, Set[Tuple[int, str]]] = defaultdict(set)
    for _, row in test_df.iterrows():
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


def default_preference_dir_train(dataset_subdir: str) -> Path:
    return PROJECT_ROOT / "user_preference" / "preference" / dataset_subdir / "train"


def flatten_bucket_pairs(
    buckets: Dict[int, List[Tuple[int, str]]],
) -> List[Tuple[int, int, str]]:
    """(cluster, user_id, candidate_id) 정렬 리스트 (배치 샤딩용)."""
    items: List[Tuple[int, int, str]] = []
    for cl in sorted(buckets.keys()):
        for uid, cid in sorted(buckets[cl], key=lambda x: (x[0], x[1])):
            items.append((cl, uid, cid))
    items.sort()
    return items


def slice_batch_items(
    items: List[Tuple[int, int, str]], num_batches: int, batch_index: int
) -> List[Tuple[int, int, str]]:
    n = len(items)
    if num_batches < 1:
        raise ValueError("num_batches must be >= 1")
    if batch_index < 0 or batch_index >= num_batches:
        raise ValueError(f"batch_index must be in [0, {num_batches - 1}]")
    if n == 0:
        return []
    chunk = (n + num_batches - 1) // num_batches
    start = batch_index * chunk
    end = min(start + chunk, n)
    return items[start:end]


def pairs_by_cluster_from_items(
    items: List[Tuple[int, int, str]],
) -> Dict[int, List[Tuple[int, str]]]:
    out: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
    for cl, uid, cid in items:
        out[cl].append((uid, cid))
    return dict(out)


def abstract_cache_path_for_prompt(title_abstraction_prompt_path: Path) -> Path:
    prompt_name = title_abstraction_prompt_path.name.lower()
    if "title_abstraction" in prompt_name:
        cache_name = "abstracted_titles.json"
    elif "keyword_extraction" in prompt_name:
        cache_name = "keyword_titles.json"
    else:
        cache_name = "transformed_titles.json"
    return PROJECT_ROOT / "user_preference" / cache_name


def main() -> None:
    ap = argparse.ArgumentParser(
        description="트레이닝셋 클러스터별 정책 + preference로 기대본문 배치 생성 (선택: 제목 추상화)"
    )
    ap.add_argument(
        "--cluster-csv",
        type=str,
        required=True,
        help="user_id,cluster 형식 CSV",
    )
    ap.add_argument(
        "--policy-files",
        type=str,
        nargs="+",
        required=True,
        metavar="PATH",
        help="클러스터 0,1,2,... 순 정책 JSON 경로",
    )
    ap.add_argument("--output", type=str, required=True, help="결과 루트 폴더 (배치마다 다른 경로 권장)")
    ap.add_argument("--mind-dataset-subdir", type=str, default="MIND_2000")
    ap.add_argument("--train-tsv", type=str, default=None, help="학습 TSV 직접 지정")
    ap.add_argument(
        "--num-batches",
        type=int,
        default=1,
        help="전체 (유저,후보) 쌍을 몇 개의 배치로 나눌지 (기본 1=전체)",
    )
    ap.add_argument(
        "--batch-index",
        type=int,
        default=0,
        help="처리할 배치 인덱스 0 .. num-batches-1",
    )
    ap.add_argument(
        "--preference-base",
        type=str,
        default=None,
        help="train split preference 디렉토리 (기본: user_preference/preference/<dataset>/train)",
    )
    ap.add_argument("--history-k", type=int, default=10)
    ap.add_argument(
        "--body-generation-yaml",
        type=str,
        default=str(PROJECT_ROOT / "user_preference" / "body_generation.yaml"),
    )
    ap.add_argument(
        "--title-abstraction-yaml",
        type=str,
        default=str(PROJECT_ROOT / "user_preference" / "title_abstraction.yaml"),
    )
    ap.add_argument(
        "--generation-settings",
        type=str,
        default=str(PROJECT_ROOT / "user_preference" / "generation_settings.yaml"),
    )
    ap.add_argument(
        "--abstract-cache-path",
        type=str,
        default=None,
        help="제목 변환 캐시 JSON (--title-abstraction 시만 사용; 미지정 시 yaml 종류에 따라 자동)",
    )
    ap.add_argument(
        "--title-abstraction",
        action="store_true",
        help="후보 제목을 title-abstraction-yaml로 LLM 추상화·캐시 (기본: 원본 제목만 사용)",
    )
    ap.add_argument("--api-key", type=str, default=None)
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--title-abstraction-model", type=str, default=None)
    ap.add_argument("--concurrency", type=int, default=8, help="동시 API 요청 수(본문 생성 단계)")
    ap.add_argument(
        "--title-prefetch-concurrency",
        type=int,
        default=1,
        help="추상 제목 사전 생성 시 동시 요청 수 (기본 1=고유 뉴스당 1회만, 중복 호출 없음)",
    )
    ap.add_argument("--dry-run", action="store_true", help="쌍 집계만 하고 API 호출 없음")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="이미 존재하는 user_*/news_*.json 이 있어도 다시 생성",
    )
    ap.add_argument(
        "--max-run-attempts",
        type=int,
        default=1,
        help="실패 시 같은 옵션으로 전체 파이프라인을 최대 몇 번 실행할지 (기본 1=재시도 없음)",
    )
    ap.add_argument(
        "--retry-delay-sec",
        type=int,
        default=5,
        help="재시도 전 대기 초",
    )
    args = ap.parse_args()
    if args.num_batches < 1:
        print("오류: --num-batches 는 1 이상이어야 합니다.")
        sys.exit(1)
    if args.max_run_attempts < 1:
        print("오류: --max-run-attempts 는 1 이상이어야 합니다.")
        sys.exit(1)

    for attempt in range(1, args.max_run_attempts + 1):
        try:
            if attempt > 1:
                print(f"\n=== 재시도 {attempt}/{args.max_run_attempts} ===\n")
            run_pipeline(args)
            return
        except Exception as e:
            if attempt >= args.max_run_attempts:
                print(f"실패 ({attempt}/{args.max_run_attempts}): {e}")
                raise
            print(f"[재시도] ({attempt}/{args.max_run_attempts}) {e}")
            time.sleep(args.retry_delay_sec)


def run_pipeline(args: argparse.Namespace) -> None:
    csv_path = _ROOT / args.cluster_csv
    if not csv_path.is_file():
        print(f"오류: cluster CSV 없음: {csv_path}")
        sys.exit(1)

    policy_paths: List[Path] = []
    for p in args.policy_files:
        abs_p = Path(p) if os.path.isabs(p) else _ROOT / p
        if not abs_p.is_file():
            print(f"오류: 정책 파일 없음: {abs_p}")
            sys.exit(1)
        policy_paths.append(abs_p.resolve())

    user_cluster = load_user_cluster_map(csv_path)
    if not user_cluster:
        print("오류: CSV에서 유효한 (user, cluster) 행이 없습니다.")
        sys.exit(1)

    max_c = max(user_cluster.values())
    if max_c >= len(policy_paths):
        print(
            f"오류: CSV 최대 클러스터 id={max_c} 인데 --policy-files 가 {len(policy_paths)}개뿐입니다."
        )
        sys.exit(1)

    ds = args.mind_dataset_subdir
    dataset_dir = PROJECT_ROOT / "dataset" / ds
    news_tsv = resolve_news_tsv(dataset_dir)
    train_tsv = Path(args.train_tsv) if args.train_tsv else resolve_train_tsv(dataset_dir)
    pref_base = (
        Path(args.preference_base)
        if args.preference_base
        else default_preference_dir_train(ds)
    )

    body_yaml = Path(args.body_generation_yaml)
    title_yaml = Path(args.title_abstraction_yaml)
    settings_path = Path(args.generation_settings)

    if not args.title_abstraction:
        abstract_cache_path = None
    elif args.abstract_cache_path:
        abstract_cache_path = Path(args.abstract_cache_path)
    else:
        abstract_cache_path = abstract_cache_path_for_prompt(title_yaml)

    required_files = [news_tsv, train_tsv, body_yaml, settings_path]
    if args.title_abstraction:
        required_files.append(title_yaml)
    for p in required_files:
        if not p.is_file():
            print(f"오류: 파일 없음: {p}")
            sys.exit(1)

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key and not args.dry_run:
        print("오류: OPENAI_API_KEY 또는 --api-key 필요")
        sys.exit(1)

    news_map = load_news_map(news_tsv)

    train_df = pd.read_csv(
        train_tsv,
        sep="\t",
        skiprows=impression_tsv_header_skiprows(train_tsv),
        names=["user", "clicked_news", "candidate_news", "clicked"],
        dtype=str,
    )
    train_df = train_df.dropna(subset=["user", "clicked_news"])

    buckets_full = collect_pairs_by_cluster(train_df, user_cluster, news_map)
    flat_all = flatten_bucket_pairs(buckets_full)
    total_pairs = len(flat_all)
    try:
        flat_batch = slice_batch_items(flat_all, args.num_batches, args.batch_index)
    except ValueError as e:
        print(f"오류: {e}")
        sys.exit(1)
    buckets = pairs_by_cluster_from_items(flat_batch)

    print(
        f"클러스터별 (user,후보) 쌍 수 (필터 전): {{{', '.join(f'{k}: {len(v)}' for k, v in sorted(buckets_full.items()))}}}"
    )
    print(
        f"배치: --num-batches {args.num_batches} --batch-index {args.batch_index} "
        f"→ 이번 실행 쌍 수 {len(flat_batch)} / 전체 {total_pairs}"
    )
    print(
        f"클러스터별 (user,후보) 쌍 수 (이번 배치): {{{', '.join(f'{k}: {len(v)}' for k, v in sorted(buckets.items()))}}}"
    )

    out_root = Path(_ROOT / args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"출력: {out_root}")
    print(f"학습 TSV: {train_tsv}")
    print(f"preference 디렉토리: {pref_base}")
    body_prompt_extra = extra_body_prompt_suffix_for_dataset(ds)
    if body_prompt_extra:
        print(
            f"[prompt] dataset {ds}: 기대본문 생성 프롬프트에 노르웨이어(bokmål) 출력 지시 추가",
            flush=True,
        )
    if not args.title_abstraction:
        print("후보 제목: 원본만 사용 (기본)")
    else:
        print(f"제목 변환 캐시: {abstract_cache_path}")

    if args.dry_run:
        print("--dry-run 이므로 생성하지 않습니다.")
        return

    with open(body_yaml, "r", encoding="utf-8") as f:
        prompt_template = f.read()
    settings = parse_settings(settings_path)

    get_openai_client = _thread_local_openai_factory(api_key)
    print_lock = threading.Lock()

    abstract_cache: Dict[str, Dict[str, str]] = {}
    cache_lock = threading.Lock()

    if args.title_abstraction:
        assert abstract_cache_path is not None
        with open(title_yaml, "r", encoding="utf-8") as f:
            title_transform_template = f.read()

        title_model = args.title_abstraction_model or args.model
        abstract_cache = load_abstract_cache(abstract_cache_path)

        def get_abstract_title(news_id: str, original: str) -> str:
            """뉴스 ID당 추상 제목 1회만 LLM 호출(캐시 있으면 스킵). 캐시 갱신은 lock으로 보호."""
            original = safe_api_text(original)
            if not original:
                original = "[untitled]"
            with cache_lock:
                if news_id in abstract_cache and abstract_cache[news_id].get(
                    "abstracted_title"
                ):
                    return abstract_cache[news_id]["abstracted_title"]
            model3_prompt = safe_api_text(
                title_transform_template.replace("{title}", original)
            )
            _validate_chat_json_payload(
                str(title_model),
                [{"role": "user", "content": model3_prompt}],
                0.3,
                120,
            )
            resp = get_openai_client().chat.completions.create(
                model=str(title_model),
                messages=[{"role": "user", "content": model3_prompt}],
                temperature=0.3,
                max_tokens=120,
            )
            ab = clean_abstracted_title(resp.choices[0].message.content or "")
            if not ab:
                ab = original
            with cache_lock:
                if news_id in abstract_cache and abstract_cache[news_id].get(
                    "abstracted_title"
                ):
                    return abstract_cache[news_id]["abstracted_title"]
                abstract_cache[news_id] = {
                    "original_title": original,
                    "abstracted_title": ab,
                }
                save_abstract_cache(abstract_cache_path, abstract_cache)
                return ab

        # --- Phase 1: 고유 후보뉴스마다 추상 제목만 미리 생성 (쌍 단위 중복 호출 방지) ---
        unique_cids: Set[str] = set()
        for _cl, pairs in buckets.items():
            for _uid, cid in pairs:
                unique_cids.add(cid)
        unique_cids = {c for c in unique_cids if c in news_map}
        print(
            f"\n>>> [1/2] 추상 제목 사전 생성: 고유 후보 뉴스 {len(unique_cids)}개 (캐시 제외 시 LLM 호출)\n"
        )
        title_workers = max(1, args.title_prefetch_concurrency)

        def _prefetch_one(cid: str) -> None:
            get_abstract_title(cid, news_map[cid])

        if title_workers == 1:
            for i, cid in enumerate(sorted(unique_cids), 1):
                _prefetch_one(cid)
                if i % 200 == 0 or i == len(unique_cids):
                    print(f"  추상 제목 진행: {i}/{len(unique_cids)}")
        else:
            with ThreadPoolExecutor(max_workers=title_workers) as ex:
                futs = [ex.submit(_prefetch_one, cid) for cid in sorted(unique_cids)]
                for i, fut in enumerate(as_completed(futs), 1):
                    fut.result()
                    if i % 200 == 0 or i == len(futs):
                        print(f"  추상 제목 진행: {i}/{len(futs)}")
    else:
        print("\n>>> [1/2] 추상 제목 단계 생략 (기본: 원본 제목)\n")

    policies: Dict[int, Dict[str, str]] = {}
    for cl, pp in enumerate(policy_paths):
        policies[cl] = load_policy(pp)

    def run_one(
        cl: int,
        uid: int,
        cid: str,
    ) -> Tuple[str, Optional[dict]]:
        policy = policies[cl]
        pf = policy_paths[cl]
        uid_s = str(uid)
        user_dir = out_root / f"user_{uid_s}"
        out_path = user_dir / f"news_{cid}.json"
        if out_path.is_file() and not args.overwrite:
            with print_lock:
                print(f"skip exists: {out_path.relative_to(out_root)}")
            return ("skipped", None)

        pref_path = pref_base / f"user_{uid_s}.json"
        if not pref_path.is_file():
            with print_lock:
                print(f"skip no preference: user={uid_s} ({pref_path})")
            return ("no_pref", None)

        with open(pref_path, "r", encoding="utf-8") as f:
            pref_json = json.load(f)
        model1_out = safe_api_text(pref_json.get("preference_profile", ""))
        if not model1_out:
            with print_lock:
                print(f"skip empty preference_profile: {pref_path}")
            return ("empty_pref", None)

        if cid not in news_map:
            return ("bad_news", None)
        candidate_title = news_map[cid]

        hist = get_recent_titles(train_df, news_map, uid_s, args.history_k)
        if not hist:
            with print_lock:
                print(f"skip no history: user={uid_s}")
            return ("no_hist", None)

        if not args.title_abstraction:
            abstracted = safe_api_text(candidate_title)
        else:
            ent = abstract_cache.get(cid) or {}
            abstracted = safe_api_text(ent.get("abstracted_title") or "")
            if not abstracted:
                abstracted = get_abstract_title(cid, candidate_title)
            abstracted = safe_api_text(abstracted)
        prompt = build_prompt(
            template=prompt_template,
            model1_output=model1_out,
            history_titles=hist,
            candidate_news=abstracted,
            policy=policy,
            settings=settings,
        )
        if body_prompt_extra:
            prompt = prompt + body_prompt_extra
        msg_content = safe_api_text(prompt)
        _validate_chat_json_payload(
            str(args.model),
            [{"role": "user", "content": msg_content}],
            0.7,
            500,
        )
        try:
            resp = get_openai_client().chat.completions.create(
                model=str(args.model),
                messages=[{"role": "user", "content": msg_content}],
                temperature=0.7,
                max_tokens=500,
            )
            body = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            # 디버깅: 어떤 (cluster, user, candidate)가 깨지는지 저장
            err_dir = out_root / "_bad_request_logs"
            err_dir.mkdir(parents=True, exist_ok=True)
            err_path = err_dir / f"cl{cl}_u{uid_s}_cid{cid}.json"
            with open(err_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "error": str(e),
                        "cluster": cl,
                        "user_id": uid_s,
                        "candidate_news_id": cid,
                        "candidate_title": candidate_title,
                        "history_k": args.history_k,
                        "history_titles": hist,
                        "policy_path": str(pf),
                        "policy": policy,
                        "model": args.model,
                        "prompt_preview": (prompt or "")[:2000],
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            with print_lock:
                print(f"[BAD_REQUEST] saved: {err_path.relative_to(out_root)} / err={e}")
            return ("bad_request", None)
        user_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "split": "train",
            "num_batches": args.num_batches,
            "batch_index": args.batch_index,
            "user_id": uid_s,
            "cluster": cl,
            "candidate_news_id": cid,
            "candidate_title": candidate_title,
            "candidate_title_abstracted": None
            if not args.title_abstraction
            else abstracted,
            "no_title_abstraction": not bool(args.title_abstraction),
            "history_k": args.history_k,
            "history_count_used": len(hist),
            "history_titles": hist,
            "preference_path": str(pref_path),
            "policy_path": str(pf),
            "policy": policy,
            "model": args.model,
            "prompt": prompt,
            "mind_dataset_subdir": ds,
            "prompt_body_language_suffix": body_prompt_extra.strip() or None,
            "generated_body": body,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        with print_lock:
            print(f"saved {out_path.relative_to(out_root)}")
        return ("ok", result)

    all_results: List[dict] = []
    stats = defaultdict(int)
    max_workers = max(1, args.concurrency)

    print(f"\n>>> [2/2] 기대본문 생성 (클러스터별 정책)\n")

    for cl in sorted(buckets.keys()):
        pairs = buckets[cl]
        if not pairs:
            continue
        print(f"\n>>> 클러스터 {cl}: 정책 {policy_paths[cl]} — {len(pairs)}쌍\n")
        if max_workers == 1:
            for uid, cid in pairs:
                st, res = run_one(cl, uid, cid)
                stats[st] += 1
                if res:
                    all_results.append(res)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = [ex.submit(run_one, cl, uid, cid) for uid, cid in pairs]
                for fut in as_completed(futs):
                    st, res = fut.result()
                    stats[st] += 1
                    if res:
                        all_results.append(res)

    if args.num_batches > 1:
        summary_path = out_root / f"all_results_batch{args.batch_index}.json"
    else:
        summary_path = out_root / "all_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n통계: {dict(stats)}")
    print(f"요약 저장: {summary_path}")
    print("전체 완료.")


if __name__ == "__main__":
    main()
