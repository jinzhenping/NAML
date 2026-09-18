#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
S3' 후보 prior: Summarizer(preference_profile) + 뉴스 title → Kandinsky prior.

히스토리는 이 스크립트가 아니라 extract_actual_body_prior_embeds.py (실제 본문).
B2 기대본문 prior 캐시는 덮어쓰지 않는다.

preference:
  train → user_preference/preference/<subdir>/train
  val/test → user_preference/preference/<subdir>/test

  conda activate clip_cu128
  python CLIP/extract_s3_prime_prior_embeds.py --split train --mind-dataset-subdir MIND_2000
  python CLIP/extract_s3_prime_prior_embeds.py --split val --mind-dataset-subdir MIND_2000
  python CLIP/extract_s3_prime_prior_embeds.py --split test --mind-dataset-subdir MIND_2000
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_CLIP_DIR = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "NAML") not in sys.path:
    sys.path.insert(0, str(_ROOT / "NAML"))
if str(_CLIP_DIR) not in sys.path:
    sys.path.insert(0, str(_CLIP_DIR))

from naml_dataset_env import DATASET_FILE_PRESETS, apply_dataset_env_from_argv, default_held_out_test_filename

from clip_embeddings import default_s3p_cache_path, resolve_project_path
from route_embeddings import (
    build_s3p_pairs_and_texts,
    collect_candidate_pairs_from_tsv,
    default_preference_dir,
    extract_prior_image_embeddings,
    load_news_titles_from_news_tsv,
    load_preference_profiles,
)


def _split_paths(mind_dataset_subdir: str, split: str):
    names = DATASET_FILE_PRESETS.get(mind_dataset_subdir)
    news_name, train_name, val_name = names if names else (
        "MIND_news.tsv",
        "MIND_train_(2000).tsv",
        "MIND_dev_(2000).tsv",
    )
    base = _ROOT / "dataset" / mind_dataset_subdir
    news_tsv = str(base / news_name)
    if split == "train":
        tsv = str(base / train_name)
    elif split in ("val", "dev"):
        tsv = str(base / val_name)
    else:
        tsv = str(base / default_held_out_test_filename(mind_dataset_subdir))
    return news_tsv, tsv


def _run_one(args, split: str) -> None:
    news_tsv, tsv_path = _split_paths(args.mind_dataset_subdir, split)
    if args.news_tsv:
        news_tsv = resolve_project_path(args.news_tsv)
    if args.tsv and args.split != "all":
        tsv_path = resolve_project_path(args.tsv)
    pref_dir = (
        resolve_project_path(args.preference_dir)
        if args.preference_dir and args.split != "all"
        else default_preference_dir(str(_ROOT), args.mind_dataset_subdir, split)
    )
    out_path = (
        resolve_project_path(args.out)
        if args.out and args.split != "all"
        else default_s3p_cache_path(args.mind_dataset_subdir, "val" if split in ("val", "dev") else split)
    )
    if not os.path.isfile(news_tsv):
        raise FileNotFoundError(f"news tsv 없음: {news_tsv}")
    if not os.path.isfile(tsv_path):
        raise FileNotFoundError(f"interaction tsv 없음: {tsv_path}")
    if not os.path.isdir(pref_dir):
        raise FileNotFoundError(
            f"preference 폴더 없음: {pref_dir}\n"
            "python user_preference/infer_user_preferences.py --dataset_subdir "
            f"{args.mind_dataset_subdir}"
            + ("" if split == "train" else " --use_test")
        )
    titles = load_news_titles_from_news_tsv(news_tsv)
    profiles = load_preference_profiles(pref_dir)
    pairs = collect_candidate_pairs_from_tsv(tsv_path)
    items, n_hit, n_miss_pref, n_miss_title = build_s3p_pairs_and_texts(pairs, profiles, titles)
    print(
        f"[s3p prior] split={split}\n"
        f"[s3p prior] news_tsv={news_tsv} titles={len(titles)}\n"
        f"[s3p prior] preference={pref_dir} profiles={len(profiles)}\n"
        f"[s3p prior] tsv={tsv_path} candidate_pairs={len(pairs)} "
        f"hit={n_hit} missing_pref={n_miss_pref} missing_title={n_miss_title}\n"
        f"[s3p prior] text=title + preference_profile (title first, CLIP 77 tokens)\n"
        f"[s3p prior] out={out_path}",
        flush=True,
    )
    extract_prior_image_embeddings(
        items,
        out_path,
        device=args.device,
        batch_size=int(args.batch_size),
        num_inference_steps=int(args.num_inference_steps),
        guidance_scale=float(args.guidance_scale),
        seed=int(args.seed),
        resume=not bool(args.no_resume),
    )


def main() -> None:
    apply_dataset_env_from_argv()
    ap = argparse.ArgumentParser(description="S3' candidate prior: preference summary + news title")
    ap.add_argument("--mind-dataset-subdir", type=str, default="MIND_2000")
    ap.add_argument("--split", type=str, default="train", choices=["train", "val", "test", "all"])
    ap.add_argument("--news-tsv", type=str, default=None)
    ap.add_argument("--tsv", type=str, default=None)
    ap.add_argument("--preference-dir", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-inference-steps", type=int, default=25)
    ap.add_argument("--guidance-scale", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    apply_dataset_env_from_argv(["--mind-dataset-subdir", args.mind_dataset_subdir])
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for split in splits:
        _run_one(args, split)


if __name__ == "__main__":
    main()
