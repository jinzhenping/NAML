# MIND_2000: --dataset_subdir MIND_2000
# Adressa_2000: --dataset_subdir Adressa_2000
"""
Generate expected body using:
- model2 prompt template
- model1 preference output (per user file; default path user_preference/preference/<dataset_subdir>/<train|test>/user_<id>.json)
- coordinator policy file (selectable)
- generation settings descriptions
- candidate title: raw from MIND_news.tsv by default; pass --use_title_abstraction to run model3 title transform first

python user_preference/generate_expected_body_from_preference.py --user_id 1291 --candidate_news_id N76665 \
--policy_file_path "coordinator_LLM/output_cluster0/11.txt"

python user_preference/generate_expected_body_from_preference.py \
  --user_id 138 --candidate_news_id N129416 \
  --policy_file_path "coordinator_LLM/output_cluster0/11.txt" \
  --history_k 5 --history_include_bodies

python user_preference/generate_expected_body_from_preference.py \
  --user_id 1291 --candidate_news_id N76665 \
  --policy_file_path "coordinator_LLM/output_cluster0/11.txt" \
  --use_title_abstraction \
  --title_abstraction_prompt_path "user_preference/keyword_extraction.yaml"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_UPREF = Path(__file__).resolve().parent
if str(_DEFAULT_UPREF) not in sys.path:
    sys.path.insert(0, str(_DEFAULT_UPREF))
from dataset_tsv_utils import (
    impression_tsv_header_skiprows,
    news_tsv_skiprows,
    resolve_news_tsv,
    resolve_test_tsv,
    resolve_train_tsv,
)

DEFAULT_DATASET_SUBDIR = "MIND_2000"
DEFAULT_MODEL = "gpt-4o-mini"


def default_preference_user_path(dataset_subdir: str, preference_split: str, user_id: str) -> Path:
    """Same layout as infer_user_preferences / generate_expected_body_*_cluster_policies: preference/<dataset>/<train|test>/user_*.json"""
    return (
        PROJECT_ROOT
        / "user_preference"
        / "preference"
        / dataset_subdir
        / preference_split
        / f"user_{user_id}.json"
    )


def safe_api_text(value: object) -> str:
    """Ensure OpenAI request `content` is a strict JSON-safe string (no NaN, no NUL)."""
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    s = str(value).strip()
    if s.lower() == "nan":
        return ""
    s = s.replace("\x00", "")
    # C0 controls (except TAB/LF/CR) break some gateways / strict JSON
    s = "".join(ch for ch in s if ord(ch) >= 32 or ch in "\t\n\r")
    # lone surrogates 등 비정상 코드포인트 정리
    s = s.encode("utf-8", errors="replace").decode("utf-8")
    return s


def load_news_records(news_tsv: Path) -> Dict[str, Dict[str, str]]:
    """news_id -> {title, body} (body may be empty)."""
    news_df = pd.read_csv(
        news_tsv,
        sep="\t",
        skiprows=news_tsv_skiprows(news_tsv),
        names=["news_id", "category", "subcategory", "title", "body"],
        dtype=str,
    )
    news_df["news_id"] = news_df["news_id"].map(safe_api_text)
    news_df["title"] = news_df["title"].map(safe_api_text)
    news_df["body"] = news_df["body"].map(lambda x: safe_api_text(x) if pd.notna(x) else "")
    news_df = news_df[(news_df["news_id"] != "") & (news_df["title"] != "")]
    out: Dict[str, Dict[str, str]] = {}
    for _, row in news_df.iterrows():
        out[str(row["news_id"])] = {
            "title": str(row["title"]),
            "body": str(row["body"]),
        }
    return out


def load_news_map(news_tsv: Path) -> Dict[str, str]:
    """Backward compatible: news_id -> title only."""
    return {k: v["title"] for k, v in load_news_records(news_tsv).items()}


def parse_settings(settings_path: Path) -> Dict[str, Dict[str, str]]:
    settings: Dict[str, Dict[str, str]] = {}
    current_category: Optional[str] = None
    current_key: Optional[str] = None
    with open(settings_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                current_category = line[1:-1].strip()
                settings[current_category] = {}
                current_key = None
                continue
            if line.startswith("{") and line.endswith("}"):
                current_key = line[1:-1].strip()
                if current_category is not None:
                    settings[current_category][current_key] = ""
                continue
            if current_category and current_key:
                prev = settings[current_category][current_key]
                settings[current_category][current_key] = (prev + " " + line).strip() if prev else line
    return settings


def policy_file_from_num(output_dir: Path, num: int) -> Path:
    return output_dir / f"{num}.txt"


def load_policy(policy_path: Path) -> Dict[str, str]:
    with open(policy_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    policy = data.get("updated_policy") or data.get("current_policy") or data.get("policy")
    if not isinstance(policy, dict):
        raise ValueError(f"Invalid policy format in {policy_path}")
    return policy


def to_history_text(titles: List[str]) -> str:
    return "\n".join(
        f"{i + 1}. {safe_api_text(title)}" for i, title in enumerate(titles)
    )


def get_recent_clicked_news_ids(train_df: pd.DataFrame, user_id: str, k: int) -> List[str]:
    user_df = train_df[train_df["user"].astype(str) == str(user_id)]
    if user_df.empty:
        return []
    clicked = str(user_df.iloc[0]["clicked_news"])
    ids = [x.strip() for x in clicked.split() if x.strip()]
    return ids[-k:] if len(ids) > k else ids


def get_recent_titles(train_df: pd.DataFrame, news_map: Dict[str, str], user_id: str, k: int) -> List[str]:
    ids = get_recent_clicked_news_ids(train_df, user_id, k)
    return [news_map[nid] for nid in ids if nid in news_map]


def format_history_block(
    news_ids: List[str],
    records: Dict[str, Dict[str, str]],
    *,
    include_bodies: bool,
    body_max_chars: int,
) -> str:
    """Numbered block for the prompt: titles only, or title + truncated actual body per item."""
    lines: List[str] = []
    for i, nid in enumerate(news_ids, 1):
        rec = records.get(nid)
        if not rec:
            continue
        title = safe_api_text(rec.get("title", ""))
        if include_bodies:
            body = safe_api_text(rec.get("body", ""))
            if body_max_chars > 0 and len(body) > body_max_chars:
                body = body[:body_max_chars].rstrip() + " …"
            lines.append(f"{i}. Title: {title}")
            lines.append(f"   Body: {body}")
        else:
            lines.append(f"{i}. {title}")
    return "\n".join(lines)


def get_description(settings: Dict[str, Dict[str, str]], category: str, value: str) -> str:
    return safe_api_text(settings.get(category, {}).get(value, ""))


def load_abstract_cache(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            # 기대 형태: {news_id: {"original_title": ..., "abstracted_title": ...}}
            return data
        return {}
    except Exception:
        return {}


def save_abstract_cache(path: Path, cache: Dict[str, Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def clean_abstracted_title(text: str) -> str:
    """Normalize model3 output to a plain one-line title-like string."""
    t = (text or "").strip()
    if not t:
        return t
    # Remove fenced code block wrappers if model outputs them.
    if t.startswith("```") and t.endswith("```"):
        lines = [ln for ln in t.splitlines() if not ln.strip().startswith("```")]
        t = "\n".join(lines).strip()
    # Remove surrounding quote pairs repeatedly.
    quote_pairs = [('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’")]
    changed = True
    while changed and len(t) >= 2:
        changed = False
        for ql, qr in quote_pairs:
            if t.startswith(ql) and t.endswith(qr):
                t = t[1:-1].strip()
                changed = True
    # Collapse multiline to one line.
    t = " ".join(t.split())
    return t


def build_prompt(
    template: str,
    model1_output: str,
    history_block: Optional[str] = None,
    history_titles: Optional[List[str]] = None,
    candidate_news: str = "",
    policy: Optional[Dict[str, str]] = None,
    settings: Optional[Dict[str, Dict[str, str]]] = None,
) -> str:
    if history_block is None:
        if history_titles is None:
            raise ValueError("build_prompt requires history_block or history_titles")
        history_block = to_history_text(history_titles)
    if policy is None:
        policy = {}
    if settings is None:
        settings = {}
    tone = safe_api_text(str(policy.get("tone", "neutral")))
    abstraction = safe_api_text(str(policy.get("abstraction_level", "mixed")))
    speculation = safe_api_text(str(policy.get("speculation_count", 1)))
    length_bucket = safe_api_text(str(policy.get("length_bucket", "medium")))
    fmt = safe_api_text(str(policy.get("format", "narrative")))

    prompt = template
    prompt = prompt.replace("{model1_output}", safe_api_text(model1_output))
    hb = safe_api_text(history_block)
    prompt = prompt.replace("{history_block}", hb)
    prompt = prompt.replace("{history_titles}", hb)
    prompt = prompt.replace("{candidate_news}", safe_api_text(candidate_news))

    prompt = prompt.replace("{Tone}", tone)
    prompt = prompt.replace("{Tone_description}", get_description(settings, "Tone", tone))
    prompt = prompt.replace("{Abstraction}", abstraction)
    prompt = prompt.replace(
        "{Abstraction_description}",
        get_description(settings, "Abstraction Level", abstraction),
    )
    prompt = prompt.replace("{SpeculationCount}", speculation)
    prompt = prompt.replace(
        "{SpeculationCount_description}",
        get_description(settings, "Speculation Count", speculation),
    )
    prompt = prompt.replace("{LengthBucket}", length_bucket)
    prompt = prompt.replace(
        "{LengthBucket_description}",
        get_description(settings, "Length Bucket", length_bucket),
    )
    prompt = prompt.replace("{Format}", fmt)
    prompt = prompt.replace("{Format_description}", get_description(settings, "Format", fmt))
    return safe_api_text(prompt)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate expected body from user preference + policy.")
    parser.add_argument("--user_id", type=str, required=True, help="target user id")
    parser.add_argument("--candidate_news_id", type=str, required=True, help="candidate news id")
    parser.add_argument("--dataset_subdir", type=str, default=DEFAULT_DATASET_SUBDIR)
    parser.add_argument("--history_k", type=int, default=10, help="recent clicked news count (use 5 with --history_include_bodies)")
    parser.add_argument(
        "--history_include_bodies",
        action="store_true",
        help="include each history item's actual body (from MIND_news.tsv) in the prompt, not only titles",
    )
    parser.add_argument(
        "--history_body_max_chars",
        type=int,
        default=500,
        help="per-history-article body truncation when --history_include_bodies (0 = no truncation)",
    )
    parser.add_argument(
        "--model2_prompt_path",
        type=str,
        default=str(PROJECT_ROOT / "user_preference" / "body_generation.yaml"),
    )
    parser.add_argument(
        "--use_title_abstraction",
        action="store_true",
        help="run model3 on the candidate title before body generation; default uses the raw title from MIND_news.tsv (preference-only pipeline)",
    )
    parser.add_argument(
        "--title_abstraction_prompt_path",
        type=str,
        default=str(PROJECT_ROOT / "user_preference" / "title_abstraction.yaml"),
        help="prompt for model3 when --use_title_abstraction (default: title_abstraction.yaml; keyword_extraction.yaml for keyword mode)",
    )
    parser.add_argument(
        "--settings_path",
        type=str,
        default=str(PROJECT_ROOT / "user_preference" / "generation_settings.yaml"),
    )
    parser.add_argument(
        "--preference_split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="with default layout: user_preference/preference/<dataset_subdir>/<split>/user_<id>.json",
    )
    parser.add_argument(
        "--preference_dir",
        type=str,
        default=None,
        help="folder containing user_<id>.json (overrides default layout; omit to use .../preference/<dataset_subdir>/<preference_split>/)",
    )
    parser.add_argument(
        "--preference_path",
        type=str,
        default=None,
        help="direct path to one preference json (overrides preference_dir and default layout)",
    )
    parser.add_argument(
        "--coordinator_output_dir",
        type=str,
        default=str(PROJECT_ROOT / "coordinator_LLM" / "output"),
    )
    parser.add_argument(
        "--policy_file_num",
        type=int,
        default=0,
        help="coordinator policy file number N -> output/N.txt",
    )
    parser.add_argument(
        "--policy_file_path",
        type=str,
        default=None,
        help="explicit policy file path (overrides --policy_file_num)",
    )
    parser.add_argument("--news_tsv", type=str, default=None)
    parser.add_argument("--train_tsv", type=str, default=None)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--title_abstraction_model",
        type=str,
        default=DEFAULT_MODEL,
        help="LLM model for title/keyword transformation (defaults to --model)",
    )
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "user_preference" / "expected_body"),
    )
    parser.add_argument(
        "--abstract_cache_path",
        type=str,
        default=None,
        help="JSON cache file to store/reuse transformed titles per news_id "
             "(if omitted, chosen automatically based on prompt path)",
    )
    args = parser.parse_args()

    dataset_dir = PROJECT_ROOT / "dataset" / args.dataset_subdir
    news_tsv = Path(args.news_tsv) if args.news_tsv else resolve_news_tsv(dataset_dir)
    train_tsv = Path(args.train_tsv) if args.train_tsv else resolve_train_tsv(dataset_dir)
    prompt_path = Path(args.model2_prompt_path)
    title_abstraction_prompt_path = Path(args.title_abstraction_prompt_path)
    settings_path = Path(args.settings_path)

    if args.preference_path:
        preference_path = Path(args.preference_path)
    elif args.preference_dir is not None:
        preference_path = Path(args.preference_dir) / f"user_{args.user_id}.json"
    else:
        preference_path = default_preference_user_path(
            args.dataset_subdir, args.preference_split, str(args.user_id)
        )

    if args.policy_file_path:
        policy_path = Path(args.policy_file_path)
    else:
        policy_path = policy_file_from_num(Path(args.coordinator_output_dir), args.policy_file_num)

    required_files = [news_tsv, train_tsv, prompt_path, settings_path, preference_path, policy_path]
    if args.use_title_abstraction:
        required_files.append(title_abstraction_prompt_path)
    for p in required_files:
        if not p.is_file():
            raise FileNotFoundError(f"Required file not found: {p}")

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key is required (use --api_key or OPENAI_API_KEY).")

    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()
    with open(preference_path, "r", encoding="utf-8") as f:
        pref_json = json.load(f)
    model1_output = safe_api_text(pref_json.get("preference_profile", ""))
    if not model1_output:
        raise ValueError(f"preference_profile is empty in {preference_path}")

    settings = parse_settings(settings_path)
    policy = load_policy(policy_path)
    news_records = load_news_records(news_tsv)
    news_map = {k: v["title"] for k, v in news_records.items()}

    if args.candidate_news_id not in news_records:
        raise ValueError(f"candidate_news_id not found in news TSV: {args.candidate_news_id}")
    candidate_title = safe_api_text(news_records[args.candidate_news_id]["title"]) or "[untitled]"

    train_df = pd.read_csv(
        train_tsv,
        sep="\t",
        names=["user", "clicked_news", "candidate_news", "clicked"],
        dtype=str,
    )
    train_df = train_df.dropna(subset=["user", "clicked_news"])
    history_ids = get_recent_clicked_news_ids(train_df, args.user_id, args.history_k)
    history_ids = [nid for nid in history_ids if nid in news_records]
    if not history_ids:
        raise ValueError(f"No valid history news ids for user {args.user_id}")
    history_block = format_history_block(
        history_ids,
        news_records,
        include_bodies=args.history_include_bodies,
        body_max_chars=max(0, int(args.history_body_max_chars)),
    )
    history_titles = [news_map[nid] for nid in history_ids if nid in news_map]

    # prepare OpenAI client
    client = OpenAI(api_key=api_key)

    # Optional step1: abstract candidate title via model3, with caching per news_id
    if not args.use_title_abstraction:
        abstracted_title = candidate_title
    else:
        # 자동 캐시 경로 결정:
        # - title_abstraction.yaml 사용 시: abstracted_titles.json
        # - keyword_extraction.yaml 사용 시: keyword_titles.json
        # - 그 외: transformed_titles.json
        if args.abstract_cache_path:
            abstract_cache_path = Path(args.abstract_cache_path)
        else:
            prompt_name = title_abstraction_prompt_path.name.lower()
            if "title_abstraction" in prompt_name:
                cache_name = "abstracted_titles.json"
            elif "keyword_extraction" in prompt_name:
                cache_name = "keyword_titles.json"
            else:
                cache_name = "transformed_titles.json"
            abstract_cache_path = PROJECT_ROOT / "user_preference" / cache_name
        abstract_cache = load_abstract_cache(abstract_cache_path)
        news_key = str(args.candidate_news_id)

        if news_key in abstract_cache and abstract_cache[news_key].get("abstracted_title"):
            abstracted_title = abstract_cache[news_key]["abstracted_title"]
        else:
            with open(title_abstraction_prompt_path, "r", encoding="utf-8") as f:
                model3_template = f.read()
            model3_prompt = safe_api_text(model3_template.replace("{title}", candidate_title))
            model3_response = client.chat.completions.create(
                model=args.title_abstraction_model or args.model,
                messages=[{"role": "user", "content": model3_prompt}],
                temperature=0.3,
                max_tokens=120,
            )
            abstracted_title = clean_abstracted_title(model3_response.choices[0].message.content or "")
            if not abstracted_title:
                abstracted_title = candidate_title
            abstract_cache[news_key] = {
                "original_title": candidate_title,
                "abstracted_title": abstracted_title,
            }
            save_abstract_cache(abstract_cache_path, abstract_cache)

    prompt = build_prompt(
        template=prompt_template,
        model1_output=model1_output,
        history_block=history_block,
        candidate_news=abstracted_title,
        policy=policy,
        settings=settings,
    )

    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": safe_api_text(prompt)}],
        temperature=0.7,
        max_tokens=500,
    )
    generated_body = (response.choices[0].message.content or "").strip()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"user_{args.user_id}_news_{args.candidate_news_id}.json"
    result = {
        "user_id": str(args.user_id),
        "candidate_news_id": str(args.candidate_news_id),
        # 원본 제목
        "candidate_title": candidate_title,
        # model3 추상화 사용 시 변환 제목, 미사용 시 원본과 동일
        "candidate_title_abstracted": abstracted_title,
        "use_title_abstraction": bool(args.use_title_abstraction),
        "history_k": args.history_k,
        "history_count_used": len(history_ids),
        "history_news_ids": history_ids,
        "history_titles": history_titles,
        "history_include_bodies": bool(args.history_include_bodies),
        "history_body_max_chars": args.history_body_max_chars if args.history_include_bodies else None,
        "preference_path": str(preference_path),
        "preference_split": args.preference_split,
        "dataset_subdir": args.dataset_subdir,
        "policy_path": str(policy_path),
        "policy": policy,
        "model": args.model,
        "prompt": prompt,
        "generated_body": generated_body,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

