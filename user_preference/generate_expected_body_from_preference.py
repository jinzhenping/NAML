"""
Generate expected body using:
- model2 prompt template
- model1 preference output (per user file)
- coordinator policy file (selectable)
- generation settings descriptions

python user_preference/generate_expected_body_from_preference.py --user_id 1291 --candidate_news_id N76665 \
--policy_file_path "coordinator_LLM/output_cluster0/11.txt"

python user_preference/generate_expected_body_from_preference.py \
  --user_id 1291 --candidate_news_id N76665 \
  --policy_file_path "coordinator_LLM/output_cluster0/11.txt" \
  --title_abstraction_prompt_path "user_preference/keyword_extraction.yaml"
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_SUBDIR = "MIND_2000"
DEFAULT_MODEL = "gpt-4o-mini"


def resolve_train_tsv(dataset_subdir: str) -> Path:
    base = PROJECT_ROOT / "dataset" / dataset_subdir
    candidates = sorted(base.glob("MIND_train_*.tsv"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 0:
        raise FileNotFoundError(f"No train TSV found in {base}")
    raise RuntimeError(f"Multiple train TSV files found in {base}; pass --train_tsv")


def resolve_test_tsv(dataset_subdir: str) -> Path:
    base = PROJECT_ROOT / "dataset" / dataset_subdir
    candidates = sorted(base.glob("MIND_test_*.tsv"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 0:
        raise FileNotFoundError(f"No test TSV found in {base}")
    raise RuntimeError(f"Multiple test TSV files found in {base}; pass --test_tsv")


def load_news_map(news_tsv: Path) -> Dict[str, str]:
    news_df = pd.read_csv(
        news_tsv,
        sep="\t",
        names=["news_id", "category", "subcategory", "title", "body"],
        dtype=str,
    )
    news_df = news_df.dropna(subset=["news_id", "title"])
    return dict(zip(news_df["news_id"], news_df["title"]))


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
    return "\n".join(f"{i + 1}. {title}" for i, title in enumerate(titles))


def get_recent_titles(train_df: pd.DataFrame, news_map: Dict[str, str], user_id: str, k: int) -> List[str]:
    user_df = train_df[train_df["user"].astype(str) == str(user_id)]
    if user_df.empty:
        return []
    clicked = str(user_df.iloc[0]["clicked_news"])
    ids = [x.strip() for x in clicked.split() if x.strip()]
    ids = ids[-k:] if len(ids) > k else ids
    return [news_map[nid] for nid in ids if nid in news_map]


def get_description(settings: Dict[str, Dict[str, str]], category: str, value: str) -> str:
    return settings.get(category, {}).get(value, "")


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
    history_titles: List[str],
    candidate_news: str,
    policy: Dict[str, str],
    settings: Dict[str, Dict[str, str]],
) -> str:
    tone = str(policy.get("tone", "neutral"))
    abstraction = str(policy.get("abstraction_level", "mixed"))
    speculation = str(policy.get("speculation_count", 1))
    length_bucket = str(policy.get("length_bucket", "medium"))
    fmt = str(policy.get("format", "narrative"))

    prompt = template
    prompt = prompt.replace("{model1_output}", model1_output)
    prompt = prompt.replace("{history_titles}", to_history_text(history_titles))
    prompt = prompt.replace("{candidate_news}", candidate_news)

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
    return prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate expected body from user preference + policy.")
    parser.add_argument("--user_id", type=str, required=True, help="target user id")
    parser.add_argument("--candidate_news_id", type=str, required=True, help="candidate news id")
    parser.add_argument("--dataset_subdir", type=str, default=DEFAULT_DATASET_SUBDIR)
    parser.add_argument("--history_k", type=int, default=10, help="recent history title count")
    parser.add_argument(
        "--model2_prompt_path",
        type=str,
        default=str(PROJECT_ROOT / "user_preference" / "body_generation.yaml"),
    )
    parser.add_argument(
        "--title_abstraction_prompt_path",
        type=str,
        default=str(PROJECT_ROOT / "user_preference" / "title_abstraction.yaml"),
        help="prompt to transform candidate title before passing as {candidate_news} "
             "(default: title_abstraction.yaml; use keyword_extraction.yaml explicitly for keyword mode)",
    )
    parser.add_argument(
        "--settings_path",
        type=str,
        default=str(PROJECT_ROOT / "user_preference" / "generation_settings.yaml"),
    )
    parser.add_argument(
        "--preference_dir",
        type=str,
        default=str(PROJECT_ROOT / "user_preference" / "preference"),
    )
    parser.add_argument(
        "--preference_path",
        type=str,
        default=None,
        help="optional direct preference json path (overrides preference_dir/user_{id}.json)",
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
    news_tsv = Path(args.news_tsv) if args.news_tsv else dataset_dir / "MIND_news.tsv"
    train_tsv = Path(args.train_tsv) if args.train_tsv else resolve_train_tsv(args.dataset_subdir)
    prompt_path = Path(args.model2_prompt_path)
    title_abstraction_prompt_path = Path(args.title_abstraction_prompt_path)
    settings_path = Path(args.settings_path)

    preference_path = (
        Path(args.preference_path)
        if args.preference_path
        else Path(args.preference_dir) / f"user_{args.user_id}.json"
    )

    if args.policy_file_path:
        policy_path = Path(args.policy_file_path)
    else:
        policy_path = policy_file_from_num(Path(args.coordinator_output_dir), args.policy_file_num)

    for p in [news_tsv, train_tsv, prompt_path, settings_path, preference_path, policy_path, title_abstraction_prompt_path]:
        if not p.is_file():
            raise FileNotFoundError(f"Required file not found: {p}")

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key is required (use --api_key or OPENAI_API_KEY).")

    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()
    with open(preference_path, "r", encoding="utf-8") as f:
        pref_json = json.load(f)
    model1_output = str(pref_json.get("preference_profile", "")).strip()
    if not model1_output:
        raise ValueError(f"preference_profile is empty in {preference_path}")

    settings = parse_settings(settings_path)
    policy = load_policy(policy_path)
    news_map = load_news_map(news_tsv)

    if args.candidate_news_id not in news_map:
        raise ValueError(f"candidate_news_id not found in news TSV: {args.candidate_news_id}")
    candidate_title = news_map[args.candidate_news_id]

    train_df = pd.read_csv(
        train_tsv,
        sep="\t",
        names=["user", "clicked_news", "candidate_news", "clicked"],
        dtype=str,
    )
    train_df = train_df.dropna(subset=["user", "clicked_news"])
    history_titles = get_recent_titles(train_df, news_map, args.user_id, args.history_k)
    if not history_titles:
        raise ValueError(f"No valid history titles for user {args.user_id}")

    # prepare OpenAI client
    client = OpenAI(api_key=api_key)

    # step1: abstract candidate title via model3, with caching per news_id
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
        model3_prompt = model3_template.replace("{title}", candidate_title)
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
        history_titles=history_titles,
        candidate_news=abstracted_title,
        policy=policy,
        settings=settings,
    )

    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": prompt}],
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
        # model3로 추상화된 제목
        "candidate_title_abstracted": abstracted_title,
        "history_k": args.history_k,
        "history_count_used": len(history_titles),
        "history_titles": history_titles,
        "preference_path": str(preference_path),
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

