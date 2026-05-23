#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
클러스터 없이 impression TSV의 **모든 유저×후보**에 대해
유저별 preference + **동일한 coordinator 스타일 정책**으로 기대본문을 생성합니다.

기본 정책: user_preference/fixed_policy_default.json
  tone=neutral, abstraction_level=mixed, speculation_count=1,
  length_bucket=medium, format=bullet

기본 프롬프트: body_generation.yaml
기본 preference:
  - train: user_preference/preference/<dataset>/train/
  - test:  user_preference/preference/<dataset>/test/

프로젝트 루트에서:

  python user_preference/generate_expected_body_all_users_single_policy.py \
    --split both \
    --output user_preference/expected_body/MIND_2000/single_policy_all \
    --mind-dataset-subdir MIND_2000

  python user_preference/generate_expected_body_all_users_single_policy.py \
    --split train \
    --output user_preference/expected_body/MIND_2000/single_policy_train \
    --mind-dataset-subdir MIND_2000 \
    --num-batches 4 --batch-index 0
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_UPREF = _ROOT / "user_preference"
if str(_UPREF) not in sys.path:
    sys.path.insert(0, str(_UPREF))

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
resolve_test_tsv = _geb.resolve_test_tsv
impression_tsv_header_skiprows = _geb.impression_tsv_header_skiprows
save_abstract_cache = _geb.save_abstract_cache
safe_api_text = _geb.safe_api_text

from dataset_tsv_utils import collect_test_tsv_merge_paths, merge_impression_tsv_paths

BODY_GENERATION_YAML = PROJECT_ROOT / "user_preference" / "body_generation.yaml"
FIXED_POLICY_JSON = PROJECT_ROOT / "user_preference" / "fixed_policy_default.json"

DEFAULT_POLICY: Dict[str, str] = {
    "tone": "neutral",
    "abstraction_level": "mixed",
    "speculation_count": "1",
    "length_bucket": "medium",
    "format": "bullet",
}

POLICY_KEYS = (
    "tone",
    "abstraction_level",
    "speculation_count",
    "length_bucket",
    "format",
)


def _thread_local_openai_factory(api_key: str):
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
    payload = {
        "model": str(model),
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    json.dumps(payload, ensure_ascii=False, allow_nan=False)


def normalize_policy_dict(raw: dict) -> Dict[str, str]:
    """build_prompt / generation_settings 키와 맞춘 문자열 정책."""
    out: Dict[str, str] = {}
    for key in POLICY_KEYS:
        if key not in raw:
            raise ValueError(f"policy missing key: {key}")
        val = raw[key]
        if key == "speculation_count":
            if isinstance(val, bool):
                raise ValueError("speculation_count must be int-like")
            if isinstance(val, (int, float)):
                out[key] = str(int(val))
            else:
                out[key] = str(val).strip()
        else:
            out[key] = str(val).strip()
    return out


def resolve_policy(args: argparse.Namespace) -> Tuple[Dict[str, str], str]:
    if args.policy_file:
        path = Path(args.policy_file)
        if not path.is_absolute():
            path = _ROOT / args.policy_file
        if not path.is_file():
            print(f"오류: 정책 파일 없음: {path}")
            sys.exit(1)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and (
            "updated_policy" in data or "current_policy" in data or "policy" in data
        ):
            policy = load_policy(path)
        else:
            policy = normalize_policy_dict(data)
        return policy, str(path.resolve())

    if FIXED_POLICY_JSON.is_file():
        with open(FIXED_POLICY_JSON, "r", encoding="utf-8") as f:
            policy = normalize_policy_dict(json.load(f))
        return policy, str(FIXED_POLICY_JSON.resolve())

    return deepcopy(DEFAULT_POLICY), "builtin"


def _norm_uid(u) -> str:
    try:
        return str(int(float(str(u).strip())))
    except (ValueError, TypeError):
        return str(u).strip()


def collect_pairs_all(impression_df, news_dict: dict) -> List[Tuple[int, str]]:
    pairs: Set[Tuple[int, str]] = set()
    for _, row in impression_df.iterrows():
        uid_raw = row["user"]
        if uid_raw is None or (isinstance(uid_raw, float) and str(uid_raw) == "nan"):
            continue
        uid_norm = _norm_uid(uid_raw)
        cand_str = str(row.get("candidate_news", "") or "")
        for cid in cand_str.split():
            ns = str(cid).strip()
            if not ns or ns not in news_dict:
                continue
            try:
                uid_int = int(uid_norm)
            except ValueError:
                continue
            pairs.add((uid_int, ns))
    return sorted(pairs, key=lambda x: (x[0], x[1]))


def slice_batch_pairs(
    pairs: List[Tuple[int, str]], num_batches: int, batch_index: int
) -> List[Tuple[int, str]]:
    n = len(pairs)
    if num_batches < 1:
        raise ValueError("num_batches must be >= 1")
    if batch_index < 0 or batch_index >= num_batches:
        raise ValueError(f"batch_index must be in [0, {num_batches - 1}]")
    if n == 0:
        return []
    chunk = (n + num_batches - 1) // num_batches
    start = batch_index * chunk
    end = min(start + chunk, n)
    return pairs[start:end]


def default_preference_dir(dataset_subdir: str, split: str) -> Path:
    return PROJECT_ROOT / "user_preference" / "preference" / dataset_subdir / split


def abstract_cache_path_for_prompt(title_abstraction_prompt_path: Path) -> Path:
    prompt_name = title_abstraction_prompt_path.name.lower()
    if "title_abstraction" in prompt_name:
        cache_name = "abstracted_titles.json"
    elif "keyword_extraction" in prompt_name:
        cache_name = "keyword_titles.json"
    else:
        cache_name = "transformed_titles.json"
    return PROJECT_ROOT / "user_preference" / cache_name


def resolve_impression_tsv(
    split: str,
    args: argparse.Namespace,
    dataset_dir: Path,
) -> Tuple[Path, str, Optional[Path]]:
    tmp: Optional[Path] = None
    if split == "train":
        if args.train_tsv:
            p = Path(args.train_tsv)
            tsv = p if p.is_absolute() else (_ROOT / args.train_tsv)
            return tsv, str(tsv), None
        tsv = resolve_train_tsv(dataset_dir)
        return tsv, str(tsv), None

    if args.test_tsv:
        p = Path(args.test_tsv)
        tsv = p if p.is_absolute() else (_ROOT / args.test_tsv)
        return tsv, str(tsv), None

    primary = resolve_test_tsv(dataset_dir)
    merge_final = not bool(args.use_test_no_merge_final)
    extra_paths: List[Path] = []
    for s in args.extra_test_tsv or []:
        e = Path(s)
        extra_paths.append(e if e.is_absolute() else (_ROOT / s))
    merged_list = collect_test_tsv_merge_paths(
        dataset_dir,
        primary,
        merge_final=merge_final,
        extra_paths=extra_paths,
    )
    if len(merged_list) > 1:
        fd, tmp_name = tempfile.mkstemp(prefix="merged_all_single_policy_test_", suffix=".tsv", text=True)
        os.close(fd)
        tmp = Path(tmp_name)
        merge_impression_tsv_paths(merged_list, tmp)
        log = "병합 " + str(len(merged_list)) + "개: " + " | ".join(str(x.resolve()) for x in merged_list)
        return tmp, log, tmp
    return primary, str(primary), None


def run_split(
    args: argparse.Namespace,
    split: str,
    out_root: Path,
    policy: Dict[str, str],
    policy_source: str,
) -> None:
    ds = args.mind_dataset_subdir
    dataset_dir = PROJECT_ROOT / "dataset" / ds
    news_tsv = resolve_news_tsv(dataset_dir)
    impression_tsv, impression_log, tmp_merged = resolve_impression_tsv(split, args, dataset_dir)

    pref_base = (
        Path(args.preference_base)
        if args.preference_base
        else default_preference_dir(ds, split)
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

    required = [news_tsv, impression_tsv, body_yaml, settings_path]
    if args.title_abstraction:
        required.append(title_yaml)
    for p in required:
        if not p.is_file():
            print(f"오류: 파일 없음: {p}")
            sys.exit(1)

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key and not args.dry_run:
        print("오류: OPENAI_API_KEY 또는 --api-key 필요")
        sys.exit(1)

    news_map = load_news_map(news_tsv)
    try:
        impression_df = pd.read_csv(
            impression_tsv,
            sep="\t",
            skiprows=impression_tsv_header_skiprows(impression_tsv),
            names=["user", "clicked_news", "candidate_news", "clicked"],
            dtype=str,
        )
    finally:
        if tmp_merged is not None and tmp_merged.is_file():
            try:
                tmp_merged.unlink()
            except OSError:
                pass
    impression_df = impression_df.dropna(subset=["user", "clicked_news"])

    pairs_full = collect_pairs_all(impression_df, news_map)
    total_pairs = len(pairs_full)
    try:
        pairs_batch = slice_batch_pairs(pairs_full, args.num_batches, args.batch_index)
    except ValueError as e:
        print(f"오류: {e}")
        sys.exit(1)

    print(f"\n=== split={split} (single policy, all users) ===")
    print(f"policy: {policy}")
    print(f"policy source: {policy_source}")
    print(f"전체 (user,후보) 쌍: {total_pairs}")
    print(
        f"배치: --num-batches {args.num_batches} --batch-index {args.batch_index} "
        f"→ 이번 실행 {len(pairs_batch)}쌍"
    )

    out_root.mkdir(parents=True, exist_ok=True)
    print(f"출력: {out_root}")
    print(f"impression TSV ({split}): {impression_log}")
    print(f"preference: {pref_base}")
    print(f"body prompt: {body_yaml}")
    body_prompt_extra = extra_body_prompt_suffix_for_dataset(ds)
    if body_prompt_extra:
        print("[prompt] dataset: Norwegian (bokmal) body suffix added", flush=True)

    if args.dry_run:
        print("--dry-run: API 호출 없음")
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
            original = safe_api_text(original)
            if not original:
                original = "[untitled]"
            with cache_lock:
                if news_id in abstract_cache and abstract_cache[news_id].get("abstracted_title"):
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
                if news_id in abstract_cache and abstract_cache[news_id].get("abstracted_title"):
                    return abstract_cache[news_id]["abstracted_title"]
                abstract_cache[news_id] = {
                    "original_title": original,
                    "abstracted_title": ab,
                }
                save_abstract_cache(abstract_cache_path, abstract_cache)
                return ab

        unique_cids = {cid for _, cid in pairs_batch if cid in news_map}
        print(f"\n>>> [1/2] 추상 제목: 고유 후보 {len(unique_cids)}개\n")
        title_workers = max(1, args.title_prefetch_concurrency)

        def _prefetch_one(cid: str) -> None:
            get_abstract_title(cid, news_map[cid])

        if title_workers == 1:
            for i, cid in enumerate(sorted(unique_cids), 1):
                _prefetch_one(cid)
                if i % 200 == 0 or i == len(unique_cids):
                    print(f"  추상 제목: {i}/{len(unique_cids)}")
        else:
            with ThreadPoolExecutor(max_workers=title_workers) as ex:
                futs = [ex.submit(_prefetch_one, cid) for cid in sorted(unique_cids)]
                for i, fut in enumerate(as_completed(futs), 1):
                    fut.result()
                    if i % 200 == 0 or i == len(futs):
                        print(f"  추상 제목: {i}/{len(futs)}")
    else:
        print("\n>>> [1/2] 추상 제목 생략 (원본 제목)\n")

    def run_one(uid: int, cid: str) -> Tuple[str, Optional[dict]]:
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
                print(f"skip no preference: user={uid_s}")
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

        hist = get_recent_titles(impression_df, news_map, uid_s, args.history_k)
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
            err_dir = out_root / "_bad_request_logs"
            err_dir.mkdir(parents=True, exist_ok=True)
            err_path = err_dir / f"u{uid_s}_cid{cid}.json"
            with open(err_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "error": str(e),
                        "split": split,
                        "user_id": uid_s,
                        "candidate_news_id": cid,
                        "model": args.model,
                        "policy": policy,
                        "prompt_preview": (prompt or "")[:2000],
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            with print_lock:
                print(f"[BAD_REQUEST] {err_path.name}: {e}")
            return ("bad_request", None)

        user_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "user_id": uid_s,
            "split": split,
            "mode": "single_policy",
            "cluster": None,
            "candidate_news_id": cid,
            "candidate_title": candidate_title,
            "candidate_title_abstracted": None if not args.title_abstraction else abstracted,
            "no_title_abstraction": not bool(args.title_abstraction),
            "history_k": args.history_k,
            "history_count_used": len(hist),
            "history_titles": hist,
            "preference_path": str(pref_path),
            "policy_path": policy_source,
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
    print(f"\n>>> [2/2] 기대본문 생성 ({len(pairs_batch)}쌍, workers={max_workers})\n")

    if max_workers == 1:
        for uid, cid in pairs_batch:
            st, res = run_one(uid, cid)
            stats[st] += 1
            if res:
                all_results.append(res)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(run_one, uid, cid) for uid, cid in pairs_batch]
            for fut in as_completed(futs):
                st, res = fut.result()
                stats[st] += 1
                if res:
                    all_results.append(res)

    summary_path = out_root / "all_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    meta_path = out_root / "policy_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {"policy_source": policy_source, "policy": policy},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"통계 ({split}): {dict(stats)}")
    print(f"요약: {summary_path}")
    print(f"정책 메타: {meta_path}")


def run_pipeline(args: argparse.Namespace) -> None:
    policy, policy_source = resolve_policy(args)
    out_base = Path(_ROOT / args.output)
    splits: List[str] = ["train", "test"] if args.split == "both" else [args.split]

    for sp in splits:
        out_root = out_base / sp if args.split == "both" else out_base
        run_split(args, sp, out_root, policy, policy_source)
    print("\n전체 완료.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="단일 정책 + 유저별 preference로 train/test 전체 유저 기대본문 생성"
    )
    ap.add_argument(
        "--split",
        type=str,
        choices=["train", "test", "both"],
        default="both",
    )
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--mind-dataset-subdir", type=str, default="MIND_2000")
    ap.add_argument(
        "--policy-file",
        type=str,
        default=None,
        help=f"정책 JSON (plain 또는 coordinator 형식). 기본: {FIXED_POLICY_JSON.name}",
    )
    ap.add_argument("--train-tsv", type=str, default=None)
    ap.add_argument("--test-tsv", type=str, default=None)
    ap.add_argument("--extra-test-tsv", type=str, nargs="*", default=None)
    ap.add_argument("--use-test-no-merge-final", action="store_true")
    ap.add_argument("--num-batches", type=int, default=1)
    ap.add_argument("--batch-index", type=int, default=0)
    ap.add_argument("--preference-base", type=str, default=None)
    ap.add_argument("--history-k", type=int, default=10)
    ap.add_argument("--body-generation-yaml", type=str, default=str(BODY_GENERATION_YAML))
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
    ap.add_argument("--abstract-cache-path", type=str, default=None)
    ap.add_argument("--title-abstraction", action="store_true")
    ap.add_argument("--api-key", type=str, default=None)
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--title-abstraction-model", type=str, default=None)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--title-prefetch-concurrency", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-run-attempts", type=int, default=1)
    ap.add_argument("--retry-delay-sec", type=int, default=5)
    args = ap.parse_args()

    if args.num_batches < 1:
        print("오류: --num-batches >= 1")
        sys.exit(1)
    if args.split == "both" and args.preference_base:
        print("오류: --split both 일 때 --preference-base 는 사용할 수 없습니다.")
        sys.exit(1)

    for attempt in range(1, args.max_run_attempts + 1):
        try:
            if attempt > 1:
                print(f"\n=== 재시도 {attempt}/{args.max_run_attempts} ===\n")
            run_pipeline(args)
            return
        except Exception as e:
            if attempt >= args.max_run_attempts:
                raise
            print(f"[재시도] ({attempt}/{args.max_run_attempts}) {e}")
            time.sleep(args.retry_delay_sec)


if __name__ == "__main__":
    main()
