# MIND_2000: --dataset_subdir MIND_2000
# Adressa_2000: --dataset_subdir Adressa_2000
"""
Infer per-user preference profiles from click history titles.

- Uses `user_preference/model1.yaml` prompt template.
- Uses only one row per user (dataset may contain repeated user rows).
- Uses the most recent N clicked news titles (default: 10).
- Saves one JSON file per user under `user_preference/preference`.

# 트레이닝셋 유저 취향
python user_preference/infer_user_preferences.py --dataset_subdir MIND_2000 --history_k 50

# 테스트셋 유저 취향
python user_preference/infer_user_preferences.py --dataset_subdir MIND_2000 --use_test --history_k 50
"""

from __future__ import annotations

import argparse
import json
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


def load_news_title_map(news_tsv: Path) -> Dict[str, str]:
    news_df = pd.read_csv(
        news_tsv,
        sep="\t",
        skiprows=news_tsv_skiprows(news_tsv),
        names=["news_id", "category", "subcategory", "title", "body"],
        dtype=str,
    )
    news_df = news_df.dropna(subset=["news_id", "title"])
    return dict(zip(news_df["news_id"], news_df["title"]))


def load_prompt_template(prompt_path: Path) -> str:
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def to_history_text(titles: List[str]) -> str:
    return "\n".join(f"{i + 1}. {title}" for i, title in enumerate(titles))


def build_prompt(prompt_template: str, history_titles: List[str]) -> str:
    return prompt_template.replace("{history_titles}", to_history_text(history_titles))


def parse_clicked_news_ids(clicked_news: str) -> List[str]:
    return [nid.strip() for nid in str(clicked_news).split() if nid.strip()]


def recent_titles_from_clicked(
    clicked_news: str, news_title_map: Dict[str, str], max_history: int
) -> List[str]:
    ids = parse_clicked_news_ids(clicked_news)
    recent_ids = ids[-max_history:] if len(ids) > max_history else ids
    titles = [news_title_map[nid] for nid in recent_ids if nid in news_title_map]
    return titles


def infer_profile(client: OpenAI, model: str, prompt: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300,
    )
    return (response.choices[0].message.content or "").strip()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Infer user preferences from recent click history titles."
    )
    parser.add_argument(
        "--dataset_subdir",
        type=str,
        default=DEFAULT_DATASET_SUBDIR,
        help="dataset subdir name (default: MIND_2000)",
    )
    parser.add_argument(
        "--train_tsv",
        type=str,
        default=None,
        help="explicit train tsv path (optional)",
    )
    parser.add_argument(
        "--test_tsv",
        type=str,
        default=None,
        help="explicit test tsv path (optional)",
    )
    parser.add_argument(
        "--use_test",
        action="store_true",
        help="use test TSV users/history instead of train TSV",
    )
    parser.add_argument(
        "--news_tsv",
        type=str,
        default=None,
        help="explicit news tsv path (optional)",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default=str(PROJECT_ROOT / "user_preference" / "preference_extraction.yaml"),
        help="preference-extraction prompt template path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="output directory for user preference files "
             "(default: train->user_preference/preference/<dataset>/train, "
             "test->user_preference/preference/<dataset>/test)",
    )
    parser.add_argument(
        "--history_k",
        type=int,
        default=10,
        help="number of most recent clicked news titles to use",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="OpenAI model",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="OpenAI API key (optional; uses OPENAI_API_KEY if omitted)",
    )
    parser.add_argument(
        "--max_users",
        type=int,
        default=None,
        help="process only first N users (debug)",
    )
    parser.add_argument(
        "--user_id",
        type=str,
        default=None,
        help="process only this user id",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing user output files",
    )
    args = parser.parse_args()

    dataset_dir = PROJECT_ROOT / "dataset" / args.dataset_subdir
    if args.use_test:
        data_tsv = Path(args.test_tsv) if args.test_tsv else resolve_test_tsv(dataset_dir)
        split_name = "test"
    else:
        data_tsv = Path(args.train_tsv) if args.train_tsv else resolve_train_tsv(dataset_dir)
        split_name = "train"
    news_tsv = Path(args.news_tsv) if args.news_tsv else resolve_news_tsv(dataset_dir)
    prompt_path = Path(args.prompt_path)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "user_preference" / "preference" / args.dataset_subdir / split_name

    if not news_tsv.is_file():
        raise FileNotFoundError(f"news TSV not found: {news_tsv}")
    if not data_tsv.is_file():
        raise FileNotFoundError(f"{split_name} TSV not found: {data_tsv}")
    if not prompt_path.is_file():
        raise FileNotFoundError(f"prompt file not found: {prompt_path}")

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key is required (use --api_key or OPENAI_API_KEY).")

    print("Loading data...")
    news_title_map = load_news_title_map(news_tsv)
    train_df = pd.read_csv(
        data_tsv,
        sep="\t",
        skiprows=impression_tsv_header_skiprows(data_tsv),
        names=["user", "clicked_news", "candidate_news", "clicked"],
        dtype=str,
    )
    train_df = train_df.dropna(subset=["user", "clicked_news"])
    # Dataset can repeat a user across rows; keep first row only per user.
    unique_users_df = train_df.drop_duplicates(subset=["user"], keep="first")
    if args.user_id is not None:
        unique_users_df = unique_users_df[unique_users_df["user"].astype(str) == str(args.user_id)]
        if unique_users_df.empty:
            raise ValueError(f"user_id {args.user_id} not found in {split_name} data.")
    if args.max_users is not None and args.max_users > 0:
        unique_users_df = unique_users_df.head(args.max_users)

    prompt_template = load_prompt_template(prompt_path)
    client = OpenAI(api_key=api_key)

    ensure_dir(output_dir)

    total = len(unique_users_df)
    success = 0
    skipped = 0
    failed = 0

    print(f"Start inference for {total} unique users ({split_name}, history_k={args.history_k})")
    for idx, row in enumerate(unique_users_df.itertuples(index=False), start=1):
        user_id = str(row.user).strip()
        clicked_news = str(row.clicked_news)

        out_path = output_dir / f"user_{user_id}.json"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            print(f"[{idx}/{total}] user={user_id} skipped (exists)")
            continue

        history_titles = recent_titles_from_clicked(
            clicked_news, news_title_map, max_history=args.history_k
        )
        if not history_titles:
            failed += 1
            print(f"[{idx}/{total}] user={user_id} failed (no valid history titles)")
            continue

        prompt = build_prompt(prompt_template, history_titles)
        try:
            profile_text = infer_profile(client, args.model, prompt)
            result = {
                "user_id": user_id,
                "history_k": args.history_k,
                "history_count_used": len(history_titles),
                "history_titles": history_titles,
                "prompt_path": str(prompt_path),
                "model": args.model,
                "preference_profile": profile_text,
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            success += 1
            print(f"[{idx}/{total}] user={user_id} saved -> {out_path.name}")
        except Exception as e:
            failed += 1
            print(f"[{idx}/{total}] user={user_id} failed ({e})")

    print(
        f"Done. success={success}, skipped={skipped}, failed={failed}, output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()

